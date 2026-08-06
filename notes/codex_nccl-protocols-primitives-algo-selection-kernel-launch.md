# NCCL 协议、Primitives、算法选择与 Kernel 发射机制

> 分析对象：本仓库 `nccl/` 当前代码（NCCL v2.29.7 分支）。本文聚焦四个相互依赖的核心机制：协议（Protocol）、设备端 `Primitives`、host 侧算法/协议选择，以及从 plan 到 CUDA kernel 发射的完整路径。
>
> 本文以 collective，尤其是 AllReduce，为主线；P2P、CollNet、NVLS、Symmetric 和 CUDA Graph 只在它们影响这四项机制时展开。源码导航优先使用文件和函数/符号名，行号会随版本变化，不能替代源码本身。

---

## 0. 先建立总图：四项机制如何串起来

NCCL 的核心不是“选一个算法然后调用一个 kernel”，而是一条逐层收敛的流水：

```text
用户 collective API
  ↓
host task：用户意图、buffer、count、datatype、reduction op
  ↓ ncclPrepareTasks
算法/协议选择：algorithm + protocol + nMaxChannels + nWarps
  ↓ calcCollChunking / registration
通信 pattern + chunk/slice + direct flags + proxy op
  ↓ schedule*TasksToPlan
kernel plan：channelMask + work batch + dev work + kernelFn
  ↓ finishPlan / uploadWork
ncclDevKernelArgs + work storage + stream/proxy ordering
  ↓ ncclLaunchKernel
CUDA grid：一个 block 对应一个 active channel
  ↓ ncclKernelMain
work batch dispatch：specialized kernel 或 ncclDevFuncTable[funcId]
  ↓ RunWorkColl
algorithm body：runRing / runTree / CollNet / NVLS
  ↓ Primitives<T, RedOp, Fan, Direct, Proto, ...>
协议特定的等待、地址选择、reduce/copy、step 发布和回收
```

其中：

- **算法**决定数据在 rank/channel 之间采用 Ring、Tree、CollNet、NVLS 等拓扑模式；
- **协议**决定一个通信 step 的数据格式、粒度、flag/FIFO 语义和线程内存操作；
- **Primitives** 是算法与协议之间的 device-side 适配层；
- **kernel plan** 把 host 侧的选择编码为 GPU 能够直接消费的 `ncclDevWork*` 和 batch 元数据；
- **kernel launch** 负责将 plan 按 channel、stream、CUDA Graph、proxy 和 communicator clique 的约束发射出去。

一个重要边界是：算法选择发生在 host，算法执行发生在 device；proxy op 也在 host 侧生成并由 proxy 线程推进。GPU kernel 不负责 topology discovery、算法搜索或建立连接。

---

## 1. 源码中四条主线的入口

| 机制 | host 入口 | device 入口 | 主要源码 |
|---|---|---|---|
| 协议 | `ncclGetAlgoInfo` 的 protocol 结果、`calcCollChunking` | `ProtoSimple`、`ProtoLL`、`ProtoLL128` | `src/device/primitives.h`、`prims_simple.h`、`prims_ll.h`、`prims_ll128.h` |
| Primitives | host 只准备 `ncclDevWorkColl`、连接和 flags | `Primitives<T, RedOp, Fan, Direct, Proto, P2p>` | `src/device/primitives.h`、各协议特化 |
| 算法选择 | `ncclPrepareTasks` → `ncclGetAlgoInfo` → `topoGetAlgoInfo` | `RunWorkColl<..., Algo, Proto>` | `src/enqueue.cc`、`src/graph/tuning.cc`、`src/device/all_reduce.h` |
| Kernel 发射 | `doLaunches` → `ncclLaunchPrepare` → `ncclLaunchKernel` | `ncclKernelMain` → `RunWorkBatch` | `src/group.cc`、`src/enqueue.cc`、`src/device/common.h` |

---

## 2. 协议：不只是消息大小的标签

### 2.1 协议类型通过模板进入 device code

`src/device/primitives.h` 没有把协议简单地作为运行时整数传入 `Primitives`，而是定义了三类 compile-time protocol type：

```cpp
ProtoSimple<SlicePerChunk, StepPerSlice, Unroll,
            MultimemSrcs, MultimemDsts>
ProtoLL
ProtoLL128
```

原因有两个：

1. SIMPLE 除了 protocol id，还需要携带 `SlicePerChunk`、`StepPerSlice`、unroll 和 multimem source/destination 数量；
2. 三种协议都需要在 device 侧计算 step bytes、grain size、group width，并让算法代码以统一接口调用，而不把大量运行时分支放进热路径。

所有协议都提供类似的静态接口：

| 接口 | 作用 |
|---|---|
| `Id` | 对应 `NCCL_PROTO_*` 枚举值 |
| `calcBytePerStep()` | 当前协议一个 logical step 能承载的有效数据量 |
| `calcBytePerGrain()` | thread/data movement 的基本粒度 |
| `MaxGroupWidth` | 一个 subchannel 在 group 中占用的连续值宽度 |

算法模板因此可以写成：

```cpp
template<typename T, typename RedOp, typename Proto>
void runRing(..., ncclDevWorkColl* work) {
  Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims(...);
  prims.directRecvReduceDirectSend(...);
}
```

Ring 的算法顺序不需要知道 LL 的 flag 布局，也不需要知道 SIMPLE 的 `connFifo`；这些行为由 `Proto` 对应的 `Primitives` 特化实现。

### 2.2 三种协议的物理布局

| 协议 | 有效 payload | 就绪/清理标记 | 主要 device 实现 | 典型特征 |
|---|---|---|---|---|
| SIMPLE | 连续数据，step 大小为 `buffSizes[SIMPLE] / NCCL_STEPS` | 可选 `connFifo[slot].size`，`-1` 表示空槽；direct path 可绕过普通 FIFO payload | `prims_simple.h` | 大消息，copy/reduce 友好，支持 direct、registered、proxy、network registration |
| LL | `ncclLLFifoLine` 中的数据字段；每条 line 还带两个 flag | 接收端按 `NCCL_LL_FLAG(step+1)` 检查 line 的 flag；发送端按 head 等待 credit | `prims_ll.h` | 小消息低延迟；数据与 flag 紧凑交织 |
| LL128 | 128B line 中只有一部分是有效 data，其余位置用于 flag/布局 | flag thread 写入 step flag；接收端按 line 检查 flag | `prims_ll128.h` | 以 warp/register/128B load-store 为中心，受对齐、架构、GDR 条件影响 |

三种协议都使用 `NCCL_STEPS` 的环形 step 编号，但不能把 LL/LL128 误读为 SIMPLE `connFifo` 的同一实现：

- SIMPLE 的槽状态通常通过 `connFifo[step % NCCL_STEPS].size` 表达；
- LL 的 payload readiness 主要由每条 line 的 flag 表达；
- LL128 也使用 line flag，但有独立的 128B 数据布局和 flag thread；
- proxy 对三者的检查条件不同，LL 甚至在 NET proxy 的某些路径中不依赖普通 tail 检查；
- direct read/write、P2P registered buffer、NVLS 和 device-side network unpack 可能改变实际地址和完成通知方式。

### 2.3 LL/LL128 的 flag：step-based ready token

LL 和 LL128 中的 flag 都不是普通的布尔状态，而是随通信 step 递增的 generation token。它回答的是一个非常具体的问题：

> 当前 line 是否属于接收方期待的 step，并且发送方已经把这一 line 的数据写完？

接收方不能只判断 flag 是否非零，因为同一个 buffer slot 会按 `step % NCCL_STEPS` 循环复用；它必须将 flag 与当前 `recvStep + 1` 计算出的期望值比较。这样可以区分“本轮数据”“上一个循环遗留的数据”和“尚未写完的数据”。

| 协议 | line 布局 | flag 位置/类型 | 接收判断 |
|---|---|---|---|
| LL | 16B：8B data + 两个 4B flag | `data1/flag1/data2/flag2`，flag 为 32-bit | `flag1 == expected && flag2 == expected` |
| LL128 | 128B：15 个 8B data + 1 个 8B flag | 每个 128B line 的最后一个 `uint64_t`，flag 为 64-bit | 保留 flag word 等于 expected |

LL 的布局定义在 `ncclLLFifoLine` 中：

```cpp
struct {
  uint32_t data1;
  uint32_t flag1;
  uint32_t data2;
  uint32_t flag2;
};
```

`flag1` 和 `flag2` 不是两个独立的状态位，而是分别伴随两个 32-bit data half；接收端必须同时看到它们匹配，才接受这条 line。发送端的 `storeLL()` 一次写入 data 和两个 flag，接收端的 `readLL()` 则持续加载并检查两个 flag。

LL128 定义 `NCCL_LL128_LINEELEMS = 16`、`NCCL_LL128_DATAELEMS = 15`，因此每条 128B line 的第 16 个 64-bit word 专门用于 flag。发送端由 `flagThread` 将当前 `sendStep + 1` 写入该位置；接收端通过 warp 协作轮询 flag，匹配后才使用刚加载的 data 做 reduce/copy。

### 2.4 flag 的生产、消费和内存可见性

一条发送/接收路径可以抽象为：

```text
发送方：
  waitSend → 等待 head credit，确认 slot 可复用
  写入 payload
  写入当前 step 的 flag
  sendStep++

接收方：
  expectedFlag = recvStep + 1
  轮询 line flag
  flag 匹配 → 读取/归约 payload
  recvStep++
  postRecv → 更新 head，归还 slot credit
```

flag 的位置和写入顺序很关键：LL 的结构定义特意把 flag 放在 data 后面，避免网络传输中接收方先看到了 flag、但 data 还没有完整到达。LL128 的接收循环也会在 flag 不匹配时重新加载整条 line；flag 匹配后才把加载的数据用于后续计算。

这里有三类不同的同步信息：

| 同步信息 | 解决的问题 |
|---|---|
| `head` | consumer 已经消费到哪里，producer 能否复用发送 slot |
| `connFifo[slot].size` | 当前 descriptor slot 是否存在以及本次传输的 size/mode |
| LL/LL128 flag | 当前协议 line 是否属于期待的 step，并且 payload 已经 ready |

因此，flag 不替代发送窗口的 head credit 或可选的 `connFifo.size` descriptor；tail 是否参与则取决于协议/连接。LL device receive 直接等 line flag，并不执行 SIMPLE 式 tail wait；LL128 只在对应 `connFifo`/tail 路径存在时发布 send tail。LL/LL128 仍可能用 `connFifo.size` 提供 size 信息，再用 line flag 判断数据是否完整可读。特别是 NET proxy 路径中，LL 会扫描每条 line 的两个 flag；LL128 在非 GDR 的 system-memory 路径会扫描每条 128B line 的最后一个 flag word。GDR 路径可以依赖其数据可见性条件，不一定执行同样的 host-side flag 扫描。

### 2.5 flag 的 step 复用、清理与边界

协议 buffer 的物理地址通常通过 `step % NCCL_STEPS` 选择。每次复用 slot 时，新的 step flag 必须覆盖旧值，否则旧 flag 可能使接收端错误地跳过等待。

- LL 使用 `NCCL_LL_FLAG(step + 1)` 生成 32-bit flag；正常构建中它直接取 step 的 32-bit 值，测试 cleanup 配置下可以使用受限的循环值；
- LL 在 flag 即将循环的条件下，通过 `NCCL_LL_CLEAN_MASK` 触发 cleanup，将 slice 中剩余 line 的 flag 一并写入，避免 flag 循环造成数据误判；
- LL128 使用 64-bit step flag，flag 位于每条 line 的保留 word，数据有效容量因此是 `15/16` 个 64-bit word，而不是完整 128B；
- abort 检查会嵌入 LL/LL128 的轮询循环，避免 flag 永远不匹配时 kernel 无法响应 communicator abort。

flag 本身不表示传输了多少字节、不校验数据内容，也不表示整个 collective 或 kernel 已完成；它只标记协议 line 级别的“当前 generation 已写完”。

### 2.6 SIMPLE 的 `SlicePerChunk` 与 `StepPerSlice`

SIMPLE 是唯一将这些参数编码进 `ProtoSimple<...>` 的协议。其重要关系是：

```text
一个 chunk = SlicePerChunk 个 slice
一个 slice = StepPerSlice 个 step 的数据窗口
一个 step  = buffSizes[SIMPLE] / NCCL_STEPS 的 payload 容量
```

在 host `calcCollChunking()`/proxy op 中这些容量以 bytes 表示；进入 `prims_simple.h` 后，局部 `stepSize`、`sliceSize` 和 `nelem` 按 `T` 转成 elements。公式相同，但排查时不能混用单位。

在 `prims_simple.h::genericOp` / `process` 中：

- 完整 slice 的协议容量先取 `stepSize * StepPerSlice`；运行时目标值为“按 `nelem/(16*SlicePerChunk)` 均衡后向 16 elements 对齐”与“协议容量的 1/32”两者较大值，最后再裁剪到当前剩余 `nelem`，因此小消息的实际 `sliceSize` 可以小于完整协议容量；
- 每个 slice 都有 wait、线程内 reduce/copy、barrier、post；
- 当数据量不足以填满所有 slice 时，仍会处理空 slice，以保持参与线程和 step 推进的一致性；
- constructor 会将本地 step round 到 `SlicePerChunk * StepPerSlice` 的边界。

AllReduce Ring 的 SIMPLE 特化使用：

```cpp
ProtoSimple<ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS,
            ALLREDUCE_SLICESTEPS>
```

而 Tree、CollNet 和 NVLS 的 SIMPLE 特化通常使用 `ProtoSimple<1,1>` 或带有特定 unroll/multimem 参数的变体。也就是说，同一个 `NCCL_PROTO_SIMPLE` enum 值在不同算法中仍可能有不同的编译期 chunk/slice 行为。

### 2.7 LL 的执行模型

`prims_ll.h` 的关键数据结构包括：

- `recvStep[i]`、`sendStep[i]`：每个 fan peer 的当前 step；
- `recvBuff[i]`、`sendBuff[i]`：指向 `NCCL_PROTO_LL` buffer；
- `recvFlag(i)`、`sendFlag(i)`：由 step 计算出的 line flag；
- `sendConnHeadPtr` 和 `sendConnHeadCache`：发送窗口的 credit。

一次 LL 操作大致为：

```text
send：waitSend(nbytes)
      → 必要时写 send connFifo.size
      → 读取本地输入/接收 peer line
      → applyReduce / applyPostOp
      → storeLL(data, flag)
      → sendStep++

recv：按 recvStep 计算期望 flag
      → 轮询 line.flag1 和 line.flag2
      → 读取 data
      → reduce/copy 到本地 output
      → recvStep++
      → postRecv 写 recv head credit
```

LL 的 line flag 让接收端能判断某个 step 的数据是否已经写完；它不是依赖普通 payload size 的 SIMPLE descriptor。`waitSend` 仍需要 head credit，避免 sender 覆盖尚未被 consumer 使用的 slot。

### 2.8 LL128 的执行模型

`prims_ll128.h` 使用 warp/register-oriented 的 128B load/store：

- `flagThread` 负责 line flag 的特殊寄存器/内存位置；
- 对齐的 user buffer 直接使用 128-bit load/store；未对齐地址先经过 per-warp shared-memory staging；
- `recvReduceSendCopy` 先等待接收 flag，再把本地输入与 peer 数据归约，最后写出带 flag 的 line；
- 发送端完成写入后由 `postSend` 执行 threadfence，再推进 send tail（存在对应 connection FIFO/同步路径时）；
- 接收端通过 `postRecv` 更新 receive head。

LL128 的有效数据不是整个 128B line，`calcBytePerStep` 和 `calcBytePerGrain` 会扣除 flag/layout 的开销。host tuning 还会根据 compute capability、path 类型、是否启用 C2C/LL128 C2C、所有 rank 的 capability 是否一致来决定是否默认启用 LL128。

### 2.9 SIMPLE 的地址路径：FIFO、direct 和 registration

`prims_simple.h::waitPeer` 在等待 step 后，根据 flags 选择 destination/source pointer。优先级可以概括为：

1. NET registration 场景下，直接使用 user buffer 或 registered network buffer；
2. `ConnFifo` 的 `NCCL_MODE_OFFSET` 场景下，用 descriptor offset 定位 FIFO 内的数据；
3. `DirectWrite`：写对端提供的 direct buffer；
4. `DirectRead`：从对端提供的 direct buffer 读取；
5. 普通情况：使用 `connEltsFifo + (step % NCCL_STEPS) * connStepSize`。

IPC/NVLS registered direct path 还需要通过 `ptrExchange` 交换远端地址：provider 发布地址，acceptor 取走地址并清空 exchange slot。这个过程发生在 `Primitives` constructor 的 `setDataPtrs` 中，不是算法函数临时完成的。

### 2.10 协议与 `ncclProtoGrainSize`

host/device 共享的 `ncclProtoGrainSize(proto)` 为 work 的 continuous-byte-distribution 提供对齐单位：

- LL：16B；
- LL128：由 warp、line、有效 data 元素共同决定；
- SIMPLE：512B。

`ncclCollCbdPart()` 使用 `ncclProtoGrainSize(proto) / eltSize` 将 `countLo/countMid/countHi` 转成当前 channel 的 element count，并将 `chunkGrains*` 转成当前协议下的 `chunkCount`。因此，协议选择改变的不仅是 line layout，也改变了 collective 如何在 channel 上切块和对齐。

---

## 3. Primitives：算法与协议之间的 device-side 执行抽象

### 3.1 模板参数各自负责什么

```cpp
Primitives<T, RedOp, Fan, Direct, Proto, P2p, isNetOffload>
```

| 参数 | 作用 |
|---|---|
| `T` | device 端元素类型 |
| `RedOp` | reduction/pre-op/post-op 逻辑和 reduction 参数 |
| `Fan` | 静态最大收发扇出 + 当前运行时的实际 peer 数 |
| `Direct` | 该算法/connection 是否允许 direct read/write 变体 |
| `Proto` | 选择 SIMPLE、LL 或 LL128 特化，以及其编译期参数 |
| `P2p` | 区分 collective 与 P2P 的 registration/address 处理 |
| `isNetOffload` | 是否使用 network offload 的特殊数据路径 |

这种设计将“算法语义”与“数据搬运机制”拆开：算法只调用 `directRecvReduceDirectSend` 等操作，`Primitives` 负责把它们翻译为当前协议下的 wait、pointer setup、reduce/copy、fence 和 step update。

### 3.2 `FanSymmetric` 与 `FanAsymmetric`

`Fan` 同时承担编译期上限和运行时数量：

```cpp
FanAsymmetric<MaxRecv, MaxSend>
FanSymmetric<MaxArity>
```

- `FanAsymmetric` 分别存储 `nr` 和 `ns`，适合 Tree reduce 的“多收一发”和 broadcast 的“一收多发”；
- `FanSymmetric` 只存储一个 `n`，静态保证收发数量相同，减少 register/predicate 压力；
- `MaxRecv/MaxSend` 让模板展开可以去掉不可能的 peer 分支；
- 实际 peer 数由 topology 的 `-1` sentinel、`recvPeers[]`、`sendPeers[]` 计算。

Tree 中 `NCCL_MAX_TREE_ARITY` 是 connection slot 上限，不代表每次一定有该数量的真实 peer。`Fan` 的实际值由 constructor 扫描有效 peer 得到。

### 3.3 Constructor：给线程分配通信角色

以 SIMPLE 为例，constructor 根据 `nrecv`、`nsend` 将一个 primitives group 的线程划分为不同角色：

```text
低 tid                         高 tid
RoleWaitRecv | RoleWaitSend | RolePostSend | RolePostRecv | worker/data threads
```

角色的意义是：

| 角色 | 读取/写入 | 目的 |
|---|---|---|
| `RoleWaitRecv` | 读取 receive tail | 等待 peer/proxy 发布数据可读 |
| `RolePostRecv` | 写 receive head | 归还接收 slot credit |
| `RoleWaitSend` | 读取 send head | 等待发送 FIFO 有空槽 |
| `RolePostSend` | 写 send tail | 发布本端数据已经写完 |
| worker | 操作 source/destination | 执行 copy、reduce、pre/post-op |

constructor 还会：

1. 从 `ncclShmem.channel.peers[]` 取出 `ncclConnInfo`；
2. 保存各 peer 的 `conn->step`，作为本次 primitives 的起始 step；
3. 根据 `conn->connFifo` 设置 `ConnFifoEnabled`；
4. 根据 `NCCL_P2P_READ/WRITE`、`NCCL_DIRECT_NIC`、IPC/NET registration 设置 `DirectRead`、`DirectWrite`、`NetRegMode`；
5. 初始化 user input/output、reduction argument 和 direct pointer exchange；
6. 为 group 分配用于 wait/post/worker 的 barrier id。

LL/LL128 也会在 constructor 中加载 connection，但其角色更偏向 warp/flag：LL/LL128 的 receive side 使用 receive buffer 和 line flag，send side 等待 head credit 并推进自己的 step；LL128 额外维护 send tail pointer 和 flag thread。

### 3.4 `process` / `genericOp` 的统一执行骨架

SIMPLE 的 `genericOp`/`process` 是最能体现 Primitives 设计的地方。每个 slice 的共同骨架是：

```text
1. wait：RoleWaitRecv 等 tail，RoleWaitSend 等 head + NCCL_STEPS credit
2. pointer：根据 direct/registered/FIFO flags 设置 srcs/dsts
3. subBarrier：确保所有 worker 看到一致的指针
4. data op：reduceCopy / copy / direct write / multimem
5. barrier：等待 worker 完成数据写入
6. descriptor：`waitPeer()` 在发送端必要时先写 `connFifo[slot].size`
7. fence：数据操作完成后，发送路径需要 system fence
8. post：`postPeer()` 推进 tail 或 head，发布本 slice 已经可见
```

`SlicePerChunk` 个 slice 完成后，constructor 保存的 connection step 会在 destructor 中写回。这样下一个 `Primitives` 实例可以从正确的 step 继续，而算法本身不需要手动维护每个 connection 的 persistent state。

### 3.5 public operation 与底层模板参数

Classic 算法常用的 public operation 可按语义理解：

| 操作 | 典型含义 | 底层行为 |
|---|---|---|
| `send` | 本地输入发送 | local input → peer |
| `recv` | 接收并写 output | peer → local output |
| `directSend` | direct-aware 发送 | 允许将 destination 解释为对端 direct buffer |
| `directRecv` | direct-aware 接收 | 允许从对端 direct buffer 读取或直接写入 |
| `recvReduceCopy` | 接收、归约、写本地 output | peer + local input → output |
| `recvReduceSend` | 接收、归约、继续发送 | peer + local input → next peer |
| `recvCopySend` | 接收、原样复制、继续发送 | peer → local/next |
| `directRecvReduceDirectSend` | direct receive + reduce + direct send | Ring/Tree 常见流水操作 |
| `directRecvReduceCopyDirectSend` | 完成归约、写 output、继续发送 | 通常承担 post-op 的最终阶段 |
| `scatter/gather` | 按 peer/head 分散或汇聚 | CollNet/NVLS/PAT 使用 |
| `sendPeerNotify/recvPeerNotify` | 只推进连接通知 | registered/offload 等无需普通 copy 的路径 |

`prims_ll.h` 和 `prims_ll128.h` 并不复制所有 direct 语义，而是通过 `PrimitivesWithoutDirect` 将 direct API 降级为对应的普通 send/recv/reduce 操作；只有 SIMPLE 的 direct/registered path 通常需要显式处理对端地址。

### 3.6 head/tail：credit 与 data-ready 必须分开理解

对一条普通 send/recv connection，最重要的方向是：

```text
发送方 GPU                                      接收方 GPU / proxy
──────────                                      ────────────────
RoleWaitSend 读取 send->head  ←── credit ───  RolePostRecv 写 recv->head
    ↓
写 connFifo.size；读取 proxy 发布的 offset（若启用）
选择 FIFO/direct 地址并写 payload
system fence
RolePostSend 写 send->tail   ── data-ready →  RoleWaitRecv 读取 recv->tail
```

同一条物理共享内存映射在 host proxy 中可能被称为 `sendMem` 或 `recvMem`，但语义仍然是：

- `head`：consumer 已经释放到哪里，producer 可以复用哪些 slot；
- `tail`：producer 已经发布到哪里，consumer 可以读取哪些 slot。

在 SIMPLE 中，`waitPeer()` 可在 payload copy/reduce 之前写 `connFifo[slot].size`，供 proxy 获得本 slice 的 descriptor；只有 payload 写入、system fence 和 tail 发布形成完整顺序后，consumer 才能按协议消费数据。因此提前可见的 `size` 本身不是 data-ready 信号，也不能替代 head/tail 的窗口语义。LL/LL128 则更多依赖 line flag 判断 payload 是否可读。

### 3.7 destructor：step 保存、direct read 和 network registration 安全

`Primitives` destructor 不是普通 C++ 清理，而是 device-side 协议的一部分：

1. 将 receive/send 的当前 step 写回 `conn->step`；
2. 对 `NetRegMode + RoleWaitSend`，等待 proxy 已经使用完直接注册的发送 buffer，避免下一个 kernel 覆盖 NIC 仍在访问的内存；
3. 对 `NetDeviceUnpack` 保存 network device head；
4. 对 `DirectRead + P2p + RoleWaitSend`，等待 receiver 读取完 direct source buffer；
5. 用最终 barrier 保证所有线程完成 connection step 和 shared group state 的写回。

因此，“kernel 中的 send 操作返回”不一定意味着 user send buffer 立刻可以被后续 kernel 覆盖；是否需要继续等 proxy/NIC/remote reader，取决于 registration 和 direct mode。

### 3.8 abort 检查

所有长时间等待通常通过 `checkAbort` 周期性读取 `ncclShmem.comm.abortFlag`：

- 默认每 `NCCL_SPINS_BEFORE_CHECK_ABORT` 次 spin 检查一次；
- 发现 abort 后设置 `ncclShmem.aborted` 和本 primitives 的 abort cache；
- 后续数据操作可以将 work size 置零，避免在 abort 状态继续搬运；
- kernel 主循环最终因 `ncclShmem.aborted` 退出。

这也是为什么调试 hang 时不能只看 CUDA kernel 是否在 spin，还要检查 communicator abort、connection head/tail 和 proxy progress。

---

## 4. 算法与协议选择：从 cost table 到 device work

### 4.1 task 先分组，再选择，不是每个 API 独立选择

`ncclPrepareTasks()` 的 collective 主流程是：

```text
collSorter 按 trafficBytes 排序
  ↓
Symmetric task 分离（若 communicator 支持）
  ↓
按 (func, opDev.op, datatype) 分桶
  ↓
相邻、大小在 4 倍范围内的 task 组成临时 aggregation
  ↓
ncclGetAlgoInfo(aggregation)
  ↓
将选择结果写回 aggregation 覆盖的每个原 task
```

这里的 aggregation 主要用于 tuning 的 `numPipeOps` 和粗粒度 cost 估算，不是把多个用户操作改变成一个语义上的 collective。被聚合的 task 仍然分别生成 device work。

选择结果写回原 task 时，LL task 的 `trafficBytes` 会乘以 4；随后 `scheduleCollTasksToPlan()` 在连续字节分布时也会把 LL 的 `trafficPerByte` 乘以 4。这是 planner 对 LL line/flag 开销和有效 payload 比例的调度权重修正，不是改变用户 payload 大小。

在加入 cost model 前，若 `comm->symmetricSupport`，`ncclMakeSymmetricTaskList()` 会先按 Symmetric kernel 可用性、window/reg type 和 task 类型筛选；不适合 Symmetric 的任务回到 legacy queue。

### 4.2 算法/协议候选矩阵

当前 `NCCL_ALGO_*` 包括：

| 算法 | 主要路径 | AllReduce 当前情况 |
|---|---|---|
| Ring | 环形 reduce-scatter + allgather | 支持 SIMPLE/LL/LL128 |
| Tree | reduce up + broadcast down | 支持 SIMPLE/LL/LL128 |
| CollNet Direct | 网内交换机/插件归约 | SIMPLE；需要 CollNet/NVSwitch 等能力 |
| CollNet Chain | 网内链式 reduce/broadcast | SIMPLE；需要 CollNet 能力 |
| NVLS | NVLink/NVLS 节点内，必要时配 NET | SIMPLE |
| NVLS_TREE | NVLS 节点内 + Tree 跨节点 | SIMPLE |
| PAT | pattern-based reduce-scatter/allgather | 不用于 AllReduce |

候选不是由 enum 自动保证可用，而是要经过 collective、datatype、reduction op、拓扑、节点数、local rank 数、compute capability、plugin support 和注册状态过滤。

### 4.3 cost table 的建立

`ncclGetAlgoInfo()` 首先创建 `NCCL_NUM_ALGORITHMS × NCCL_NUM_PROTOCOLS` 的 cost table，初始值为 `NCCL_ALGO_PROTO_IGNORE`，然后调用 `updateCollCostTable()`：

1. 单 rank 时直接把 Ring/SIMPLE 设为零成本。
2. CollNet 需要 `collNetSupport == 1`，且受 local GPU 数和 direct arity 限制。
3. NVLS/NVLS_TREE 需要 `nvlsSupport`；多节点 NVLS 还需要 CollNet/网络条件。
4. 不支持的 collective/algorithm 组合被跳过，例如 AllReduce 不使用 PAT。
5. 对每个合法 `(algorithm, protocol)` 调用 `ncclTopoGetAlgoTime()`。

`ncclTopoGetAlgoTime()` 的基本模型是：

```text
estimated_time ≈ latency × latency_count + nBytes / (1000 × bandwidth)
```

但当前代码还有若干修正：

- Tree/AllReduce 和 NVLS_TREE/AllReduce 对不同消息规模应用 correction factor；
- 多节点 Ring/SIMPLE 在一定每-channel 数据量下增加 plateau latency；
- Ring 的 `latency_count` 使用 aggregation 的 `numPipeOps`；
- Tree 等算法按 `NCCL_MAX_DEV_WORK_BATCH_COLLS` 对 pipelined ops 分组；
- tuning 初始化时已经根据 protocol 的 payload 效率、算法并发度、topology graph、硬件链路和网络条件计算 bandwidth/latency。

这说明 NCCL 的算法选择不是一个固定的“消息小于 X 用 Tree，大于 Y 用 Ring”规则，而是 topology/tuning/cost 的动态比较。

### 4.4 tuner plugin 与环境变量

若 communicator 有 tuner plugin：

1. core 先准备 cost table 和合法候选；
2. 读取 buffer registration 是否有效；
3. 将 cost table、bytes、`numPipeOps`、registration 信息交给 `tuner->getCollInfo()`；
4. tuner 可以设置 algorithm/protocol 和 `nMaxChannels`；
5. core 的 `topoGetAlgoInfo()` 仍负责补全或确认 channel/thread 参数。

`NCCL_ALGO` 和 `NCCL_PROTO` 在 `tuning.cc` 中解析为 per-collective 的 enable matrix：

- 支持通用列表，例如 `NCCL_PROTO=LL,Simple`；
- 支持按 collective 限定或排除；
- 被禁用的候选会将相应 bandwidth 置零，最终不参与选择；
- 强制配置导致没有合法组合时，NCCL 报 `no algorithm available`，通常返回 invalid usage。

LL128 默认并非所有机器、链路和 collective 都启用。它会根据 `minCompCap/maxCompCap`、path type、是否 C2C、CUDA 版本、架构一致性等条件动态启用；因此“LL128 在 enum 中存在”不代表本次运行一定可选。

### 4.5 channel 数与线程数的联动

算法/协议确定后，`topoGetAlgoInfo()` 继续计算：

- `nMaxChannels`：该任务最多使用的 channel 数；
- `nWarps`：每个 work 需要的 warp 数。

初始值来自：

```text
nc = comm->nChannels
nt = comm->maxThreads[algorithm][protocol]
threshold = comm->threadThresholds[algorithm][protocol]
```

主要规则包括：

- Ring/Tree 按 `nBytes < nc × nt × threshold` 缩减 channel；
- CollNet Direct 根据 `nHeads` 和 channel switch 规则调整 channel；
- NVLS/NVLS_TREE 通常限制在 `nvlsChannels`，因为超过一定 channel 后收益有限；
- 非 NVLS/非 CollNet Direct 的算法在小消息下可能减少线程数；
- SIMPLE Ring 额外增加一个 warp 处理同步；
- SIMPLE Tree 额外增加多个 warp 适配 split thread model；
- Tree 和 PAT 最终使用 `NCCL_MAX_NTHREADS`；
- 所有算法至少保证 3 warps。

thread threshold 初始化于 `graph/tuning.cc`：LL/LL128/SIMPLE 有默认值，Ring LL 会乘以 rank 数，CollNet Simple 有单独的较大阈值；`NCCL_THREAD_THRESHOLDS` 可覆盖 Tree/Ring 两组阈值。它们是线程/channel 调整参数，不是算法选择的唯一阈值。

### 4.6 `devFuncId`：把选择结果映射到 device function

选择结果会调用：

```cpp
agg.devFuncId = ncclDevFuncId(
    agg.func, agg.opDev.op, agg.datatype,
    agg.algorithm, agg.protocol);
```

`ncclDevFuncId()` 的 row layout 必须与 `device/generate.py` 的 `enumerate_func_rows()` 保持一致。AllReduce 的 row 维度包括：

```text
collective
  × device reduction op
  × datatype
  × algorithm
  × protocol
```

生成器再通过 `equivalent_primary()` 和 `best_kernel()`：

- 将某些组合映射到可复用的 primary device function；
- 为常用组合生成 specialized `ncclDevKernel_*`，减少 device function call；
- 对没有专用 kernel 或被过滤的组合使用 generic kernel；
- 生成 `ncclDevFuncRowToId`、`ncclDevFuncTable`、`ncclDevKernelList` 和 `ncclDevKernelForFunc`。

因此 `devFuncId` 是 algorithm/protocol 选择到 kernel dispatch 的桥梁，但它不是 kernel symbol 本身：一个 specialized kernel 可以覆盖多个等价 func row，而 generic kernel 则在 device 端按 funcId 查表。

### 4.7 `calcCollChunking`：选择结果继续影响 pattern 和 proxy

`calcCollChunking()` 将 `(func, algorithm, protocol)` 转换为：

- collective pattern：Ring、RingTwice、TreeUp/Down、CollNet Direct、CollNet Chain、NVLS、NVLS_TREE、PAT；
- `stepSize`、`chunkSteps`、`sliceSteps`、`chunkSize`；
- `nstepsPerLoop`、`nchunksPerLoop`、`loopSize`、`nLoops`；
- proxy op 的 `nsteps`、`chunkSize`、`sliceSize`、`loopOffset`、`pattern`、`protocol`、`algorithm`；
- `outDirectFlags`，例如 CollNet Direct 使用 `NCCL_P2P_WRITE`。

重要的 protocol 相关换算是：

```text
stepSize = buffSizes[protocol] / NCCL_STEPS
chunkSize = stepSize × chunkSteps
LL      ：chunkSize 按 2 除，因为 line 中一部分是 flag
LL128   ：chunkSize 按有效 data elems / line elems 换算
最后按 ncclProtoGrainSize 对齐
```

Ring AllReduce 的 pattern 是 `ncclPatternRingTwice`，需要 `2*(nRanks-1)` steps；Tree/普通 pipeline 多为一个 step per loop；CollNet/NVLS 的 loop 通常按 heads 计算。proxy 看到的 step 数由这里生成，而不是由 device `Primitives` 临时推导。

### 4.8 registration 会改变选择后的执行方式

buffer registration 不一定直接改变算法 enum，但会改变 cost table、channel policy 或 device work 的可用能力：

- tuner 可以通过 `regBuff` 感知 send/recv buffer 是否注册；
- `NCCL_CTA_POLICY_EFFICIENCY` 在满足条件时会根据注册状态影响某些算法选择；
- `NCCL_NET_REG_BUFFER` 使 proxy 直接使用 registered network buffer；
- `NCCL_IPC_REG_BUFFER` / `NCCL_NVLS_REG_BUFFER` 使 `Primitives` 能做 direct pointer exchange 或 NVLS registered path；
- NVLS registered work 使用 `ncclDevWorkCollReg`，携带 `dnInputs/dnOutputs/upOutputs` 数组。

所以算法选择完成后，还必须继续看 registration/connect flags，才能判断 device kernel 的实际数据地址和 proxy 是否需要普通 staging buffer。

---

## 5. Kernel 发射：plan 如何变成 GPU 执行

### 5.1 group end 与 launch rounds

`src/group.cc::doLaunches()` 负责将多个 communicator 的 plan 协调为多轮 launch：

```text
对每个 intra-process clique：
  ncclLaunchPrepare(comm)
  while 还有未发射 plan：
    各 communicator 取一个 plan
    ncclLaunchKernelBefore_NoUncapturedCuda
    发射 CE/RMA/kernel
    ncclLaunchKernelAfter_NoCuda
  ncclLaunchFinish(comm)
```

当 `NCCL_LAUNCH_MODE=GROUP` 时，sibling communicator 在每轮通过 intra-process barrier 对齐；非 GROUP 模式使用各 planner 的 `unlaunchedPlansHead` 判断是否还有 round。多轮的原因通常是 plan budget、work FIFO 或不同任务类型不能放进一个 kernel，而不是 collective 算法自身只能运行一小段。

### 5.2 `ncclLaunchPrepare` 的 plan 优先级

`ncclLaunchPrepare()` 循环创建 `ncclKernelPlan`，直到 planner 中的任务清空。当前顺序是：

```text
RMA
  → CE collective
  → Symmetric collective
  → ordinary collective
  → broadcast
  → P2P
```

普通 plan 中必须先 drain collective，再 drain P2P。源码明确指出，这保证各 rank 在相同位置切断 plan，避免不同 rank 因 P2P 任务先被消费而得到不一致的 channel picker/batch 边界。

plan 的重要字段包括：

| 字段 | 作用 |
|---|---|
| `kernelFn` | 最终 CUDA kernel symbol |
| `kernelSpecialized` | 是否使用 specialized wrapper |
| `kernelArgs` / `kernelSymArgs` | Classic/Symmetric 各自的参数区 |
| `kernelArgsSize` | driver API launch 的参数 buffer 大小 |
| `channelMask` | 本 plan 使用的 channel bitset |
| `threadPerBlock` | block 线程数，通常至少覆盖 plan 中最大的 work 要求 |
| `workStorageType` | Args、Fifo 或 Persistent |
| `workQueue` / `workBytes` | host 端待上传的 device work payload |
| `nWorkBatches` | batch descriptor 数量 |
| `proxyOpQueue` | 与 plan 相关的 host proxy 操作 |
| `isSymColl/isCeColl/isRma` | 选择不同发射路径 |

### 5.3 schedule collective：构造 `ncclDevWorkColl`

`scheduleCollTasksToPlan()` 消费已经完成算法选择的 task：

1. 依据 CollNet/NVLS/普通 collective 的 channel 约束确定 `[channelLo, channelHi]`；
2. 对普通 Ring/Tree/PAT 使用 continuous-byte-distribution，生成 `countLo/countMid/countHi` 和 `chunkGrainsLo/Mid/Hi`；
3. 对 CollNet/NVLS 使用均匀 channel 分片和 `collnet.chunkCount`；
4. 为每个 channel 生成或复用 proxy op；
5. 调用 `ncclAddWorkBatchToPlan()` 记录 work 类型、funcId、offset 和 P2P metadata；
6. 更新 `channelMask`、`threadPerBlock`、`kernelFn`、`collOpCount` 和 plan work queue。

`ncclDevWorkColl` 是 GPU 实际消费的 collective 描述，包含：

- channel 范围、`nWarps`；
- send/recv buffer 与 remote address/offset；
- `redOpArg`、是否为 pointer；
- `regUsed`、`netRegUsed`、`direct`、`oneNode`；
- continuous-byte distribution 或 CollNet scheduling 数据；
- `profilerEnabled`。

host `ncclTaskColl` 的算法字段不会直接被 GPU 使用；它们在这里被编码为 device work 和 `devFuncId`。

### 5.4 `ncclAddWorkBatchToPlan`：控制 batch 是否可融合

一个 batch 可以包含多个 work，但不能任意合并。创建新 batch 的主要条件是：

- `workType` 不同；
- `funcId` 不同；
- P2P epoch 不同（启用 `P2P_EPOCH_ENABLE` 时）；
- P2P 已达到 `NCCL_MAX_DEV_WORK_P2P_PER_BATCH`；
- 同一 P2P round 重复使用同一 connection；
- P2P round group 不一致；
- work bytes 超过 `NCCL_MAX_DEV_WORK_BATCH_BYTES`；
- broadcast 数量超过当前架构允许的 batch 容量。

即便可以复用同一 batch，也可能因为 work offset 无法放进 64-bit `offsetBitset` 而创建 extension batch：

```text
batch.nextExtends = 1
  → device 端将下一个 batch 与当前 batch 融合执行
batch.nextJump
  → 指向同一 channel 的下一个 batch
```

`offsetBitset` 的每个 bit 表示一个 `workSize` 槽位；这样 work payload 可以在全局 storage 中非连续排列，而 GPU 仍能按 batch 取得连续的 shared-memory work storage。

### 5.5 `finishPlan`：kernel args 和 batchZero

普通 Classic plan 的参数布局为：

```text
ncclDevKernelArgs
  ├── comm = comm->devComm
  ├── channelMask
  ├── workStorageType
  ├── workMask / workBuf
  └── ncclDevWorkBatch batchZero[nWorkBatches]
```

work payload 的 storage 选择：

| 条件 | storage |
|---|---|
| args + batch + work 能放进 `comm->workArgsBytes` | Args，work 紧跟 batch 数组 |
| 普通非 persistent plan 容量不足 | Fifo，上传到 communicator work FIFO |
| CUDA Graph persistent plan | Persistent，分配独立 device buffer 并异步 memcpy |

`finishPlan()` 按 channel ascending、round-robin 方式把各 channel 的 batch 写入 `batchZero[]`，并保证：

```text
batchZero[blockIdx.x] = channel mask 中第 blockIdx.x 个 active channel 的第一 batch
```

因此 `blockIdx.x` 不是直接的 channel id，而是 active-channel 的紧凑索引。后续 `nextJump` 将同一 channel 的 batch 串起来。

proxy op 也在 `finishPlan()` 中从 per-channel queue merge-sort 到 `plan->proxyOpQueue`。opCount 的低位 tag 用来区分 collective 与 P2P，使 proxy 顺序与 plan/work 顺序保持一致。

### 5.6 `uploadWork`：只上传 device work，不负责创建 proxy op

`ncclLaunchKernelBefore_NoUncapturedCuda()` 调用 `uploadWork()`：

- Args：work 已经位于 kernel args 后面；
- Fifo：等待 `workFifoConsumed` 提供空间，写入 host/GDR FIFO，并设置 device FIFO 指针；
- Persistent：分配 device buffer，使用 device stream 异步 copy，再把 buffer 指针写入 args。

所有 work 以 16B 对齐单元复制，batch `offsetBase` 按 storage 类型修正；FIFO 写完更新 `comm->workFifoProduced`。`uploadWork()` 对 Symmetric、CE、RMA plan 直接跳过。

proxy op 的创建/启动是另一条路径：

```text
schedule*TasksToPlan
  → per-channel proxyOpQueue
  → finishPlan merge
  → ncclLaunchPrepare 可能排 host callback
  → hostStreamPlanTask
  → uploadProxyOps
  → ncclProxySaveOp / ncclProxyStart
```

不要把 `uploadWork` 和 `uploadProxyOps` 合并成一个动作；前者上传 GPU 的 work metadata，后者把 transport 控制信息提交给 proxy。

### 5.7 stream、host callback 与 proxy ordering

`ncclLaunchPrepare()` 会处理多个 user stream、device strong stream、context launch order 和 host stream：

1. 选择 `planner->streams` 链表头作为 launch stream，并让它等待同一 group 中其他 user stream 的 event；新 stream 采用头插，因此这里不是“第一个 API 使用的 stream”；
2. 让 launch stream 等待 shared device stream；
3. 在支持 implicit order 时，让 launch stream 等待 context 的 `launchOrder`；
4. 对含 proxy op 的 plan，在必要时于 host stream 排入 `cudaLaunchHostFunc`；
5. 让 launch stream 等待 host stream，使 host callback 先完成 proxy op upload/start。

若没有排 host callback，`ncclLaunchKernelAfter_NoCuda()` 直接调用 `hostStreamPlanTask()`。这使 proxy 启动可能发生在 kernel launch 前的 callback，也可能发生在 kernel launch 后的 host hook；两者都由 NCCL 的 plan/stream ordering 保护。

### 5.8 `doLaunches` 中每轮真正做什么

对于一个 plan，`group.cc::doLaunches()` 依次执行：

```text
ncclLaunchKernelBefore_NoUncapturedCuda
  → uploadWork

ncclLaunchKernel / ncclLaunchCeColl / ncclLaunchRma
  → 真正发射对应 device work

ncclLaunchKernelAfter_NoCuda
  → 必要时提交 proxy op、结束 profiler task event，并排入 host plan reclaimer
```

多 communicator clique 中，每个 communicator 每轮最多弹出一个未发射 plan；GROUP launch mode 使用 barrier 让各本地 comm 对齐 round。非 persistent plan 在 `hostStreamPlanTask()` 转移 proxy 描述符所有权后即可把 reclaimer 排入 `callbackQueue`；host-callback 路径的入队可能早于 CPU launch，但主线程只在 launch 后的 safe point 消费，且消费也不表示 GPU 或 proxy 已完成。最后 `ncclLaunchFinish()` 建立 stream completion dependency，并结束本 group 的 launch order。

### 5.9 `ncclLaunchKernel` 的 grid、block、shared memory

`ncclLaunchKernel()` 根据 plan 设置：

```cpp
int nChannels = countOneBits(plan->channelMask);
dim3 grid  = { (unsigned)nChannels, 1, 1 };
dim3 block = { (unsigned)plan->threadPerBlock, 1, 1 };
int smem = plan->isSymColl
         ? plan->kernelDynSmem
         : ncclShmemDynamicSize(comm->cudaArch);
```

关键点：

- grid 的 block 数是 active channel 数，不是 communicator 总 channel 数；
- Classic block size 至少满足 `NCCL_MIN_NTHREADS`，并覆盖 plan 内最大的 work `nWarps`；
- 单个 work 的 `nWarps` 通过 `RunWorkBatch` 的 `subtn` 控制，不代表 plan 一定为每个 work 发射独立 kernel；
- Classic 使用 NCCL shared-memory layout，Symmetric 使用自己的 dynamic shared-memory 参数；
- launch 参数通过 driver API 的 `CU_LAUNCH_PARAM_BUFFER_POINTER/SIZE` 传入，而不是把复杂 args 展开成普通 host 参数列表。

### 5.10 specialized、generic 与 `ncclKernelMain`

Classic kernel 由 `DEFINE_ncclDevKernel` 宏生成：

```cpp
__global__ void ncclDevKernel_<suffix>(ncclDevKernelArgs4K const args4K) {
  ncclKernelMain<specializedFnId,
    RunWorkBatch<coll, ty, redop<ty>, algo, proto>>(&args4K.args);
}
```

generic 入口是 `common.cu::ncclDevKernel_Generic`，它使用 `SpecializedFnId = -1`，让 `ncclKernelMain` 按 `ncclShmem.funcId` 调用 `ncclDevFuncTable[funcId]()`。

`ncclKernelMain` 的执行顺序：

1. 将 args 拷到 `ncclShmem.args`；
2. 根据 `channelMask` 将紧凑的 `blockIdx.x` 反映射为真实 `channelId`；
3. warp 0 加载 `ncclKernelComm`；
4. warp 1 加载 `ncclDevChannel`；
5. 其余线程加载当前 channel 的 work batch；
6. barrier 发布 shared state；
7. 执行 specialized `RunWorkBatch` 或 generic `ncclDevFuncTable`；
8. 有 `nextBatchIx` 时继续加载同一 channel 的下一 batch，否则退出。

`RunWorkBatch` 再对 work 做两层处理：

- 需要 pointer reduction argument 时先读取 `redOpArg`；
- 遍历 `nWorks`，调用 `RunWorkColl<Fn,T,RedOp,Algo,Proto>().run(tid, subtn, work)`；
- 相邻 work 的 `nWarps` 不同时插入同步；
- work 算法通过 `ncclCollCbdPart` 得到当前 channel 的 `gridOffset/channelCount/chunkCount`。

### 5.11 CUDA launch attributes

在 CUDA/driver 支持 `cuLaunchKernelEx` 时，`ncclLaunchKernel()` 还可能设置：

| 属性 | 条件/作用 |
|---|---|
| CGA cluster dimension | `sm90+` 使用 `config.cgaClusterSize`，若 grid 不能整除则退化为 1；用于 thread block cluster 的调度/协作 |
| cluster scheduling policy | 当前使用 spread preference |
| memory sync domain | CUDA 12+、sm90+ 可设置 NCCL 的 remote mem-sync domain |
| launch completion event | CUDA 12.3+ 的 implicit launch order 可能附加 completion event |
| programmatic stream serialization | Symmetric、sm90+、CUDA 12.3+ 条件下启用；这不是 cooperative kernel launch |
| NVLink utilization-centric scheduling | CUDA 13+/对应架构下按配置启用 |

不满足条件时使用普通 `cuLaunchKernel`。这些属性属于 launch scheduling/memory ordering，不改变 `Primitives` 的 protocol semantics。

### 5.12 Symmetric 的 kernel 发射旁路

Symmetric 仍然通过 `ncclLaunchKernel` 这个 host 包装发射，但它不是 Classic `ncclKernelMain` 的分支：

1. `ncclMakeSymmetricTaskList` 先筛选可用任务；
2. `ncclSymmetricTaskScheduler` 设置 `isSymColl`、Symmetric kernelFn、args 和 dynamic shared memory；
3. `finishPlan` 对 Symmetric plan 直接返回；
4. `uploadWork` 跳过；
5. generated `__global__` Symmetric wrapper 调用 `ncclSymkRun_*` device function；
6. plan 设置 `hasProxyOps=false`，不进入 Classic proxy op 生命周期。

Symmetric 使用 `ncclLsaBarrierSession<ncclCoopCta>` 或 `ncclLLA2ASession<ncclCoopCta>`，但 `ncclCoopCta` 只表示 CTA 级 `__syncthreads()` 协作，不等于 CUDA grid cooperative launch。

---

## 6. 一个 AllReduce/SIMPLE/Ring 的端到端示例

假设某次 AllReduce 最终选择 Ring + SIMPLE：

```text
1. ncclAllReduce 填充 host ncclInfo，taskAppend 进入 collSorter。
2. ncclPrepareTasks 按 (AllReduce, op, datatype) 分桶。
3. ncclGetAlgoInfo 构造候选 cost table，选 Ring/SIMPLE，得到 nMaxChannels/nWarps。
4. calcCollChunking 选择 ringTwice pattern，计算 chunkSteps/sliceSteps、nsteps、loopSize 和 proxy op。
5. ncclTasksRegAndEnqueue 生成 ncclDevWorkColl。
6. scheduleCollTasksToPlan 为 work 分配 channel 范围和 countLo/Mid/Hi。
7. ncclAddWorkBatchToPlan 写入 batch 的 funcId、offsetBase 和 offsetBitset。
8. finishPlan 生成 ncclDevKernelArgs、batchZero 和 channelMask。
9. uploadWork 将 ncclDevWorkColl 写入 Args/Fifo/Persistent storage。
10. ncclLaunchKernel 发射 grid=active channel count 的 Classic kernel。
11. ncclKernelMain 取得真实 channelId，加载 batch/work。
12. RunWorkColl<AllReduce, ..., Ring, Simple> 构造
    Primitives<..., FanSymmetric<1>, Direct=1, ProtoSimple<...>>。
13. runRing 调 directSend、directRecvReduceDirectSend、
    directRecvReduceCopyDirectSend、directRecvCopyDirectSend 和 directRecv。
14. 每个 slice 由 Primitives 完成 wait → pointer → reduce/copy → fence → post。
15. direct/FIFO/proxy 的具体路径由 connection flags、registration 和 transport 决定。
16. 所有 work/batch/channel 完成后 kernel 返回；proxy op 若存在则独立推进到完成。
```

这个例子中，Ring 是算法层选择，SIMPLE 是协议层选择，`ProtoSimple<...>` 是编译期 device 参数，`Primitives` 是执行抽象，`ncclDevWorkColl`/batch 是 host 到 device 的编码，`ncclLaunchKernel` 则是最终的 CUDA 发射。

---

## 7. 最重要的不变量与排查顺序

### 7.1 必须保持的不变量

1. **协议与 buffer layout 一致**：选 LL/LL128/SIMPLE 后，host chunking、proxy nsteps、device buffer 和 Primitives 必须使用同一协议语义。
2. **head/tail 方向不能混淆**：head 是 credit，tail 是 data-ready；send/recv 的 connection 视角必须区分。
3. **work 的 `devFuncId` 必须与 batch/kernel 兼容**：specialized kernel 和 generic table 必须能解释当前 batch。
4. **channelMask、grid.x、batchZero 起点必须一致**：`blockIdx.x` 通过 mask 映射真实 channel。
5. **P2P epoch/round 不能错误融合**：同一 connection 的冲突 round 进入同一个 batch 可能导致 hang。
6. **proxy op 与 kernel work 顺序必须可观察且可重放**：`finishPlan` 的 merge 和 `uploadProxyOps` 在 communicator shared resource 上分配的单调 `opCount` 不能破坏顺序。
7. **registration/direct flags 必须在 producer/consumer 两端匹配**：否则 direct pointer exchange、address selection 或 destructor wait 会失配。
8. **abort 需要贯穿等待点**：GPU spin、proxy progress 和 host completion 都必须能响应 communicator abort。

### 7.2 推荐的源码排查顺序

```text
1. src/graph/tuning.cc
   → 候选是否被 enable、cost table 是什么、最终为什么选该组合

2. src/enqueue.cc::ncclPrepareTasks/ncclGetAlgoInfo/calcCollChunking
   → task 得到的 algorithm/protocol/channel/warp/chunk/proxy 参数

3. src/enqueue.cc::scheduleCollTasksToPlan/ncclAddWorkBatchToPlan/finishPlan
   → device work、batch、channelMask 和 kernelFn 如何形成

4. src/enqueue.cc::uploadWork + src/group.cc::doLaunches
   → work 是否上传、plan 是否多轮、stream/proxy 是否有序

5. src/device/common.h
   → block 如何映射 channel、batch 如何 dispatch、work 如何进入 RunWorkColl

6. src/device/primitives.h + prims_*.h
   → 当前协议下实际等待什么、访问什么地址、什么时候发布 step
```

### 7.3 关键源码导航

| 主题 | 主要符号/文件 |
|---|---|
| protocol type | `src/device/primitives.h::ProtoSimple/ProtoLL/ProtoLL128` |
| SIMPLE execution | `src/device/prims_simple.h::waitPeer/postPeer/process/genericOp` |
| LL execution | `src/device/prims_ll.h::waitSend/readLL/storeLL/LLGenericOp` |
| LL128 execution | `src/device/prims_ll128.h::waitSend/postSend/recvReduceSendCopy/GenericOp` |
| connection role setup | `prims_simple.h::loadRecvConn/loadSendConn`、LL/LL128 对应 constructor |
| algorithm/protocol cost | `src/enqueue.cc::ncclGetAlgoInfo/topoGetAlgoInfo`、`src/graph/tuning.cc::ncclTopoGetAlgoTime` |
| chunking/pattern | `src/enqueue.cc::calcCollChunking` |
| function id generation | `src/include/device.h::ncclDevFuncId`、`src/device/generate.py` |
| plan scheduling | `src/enqueue.cc::ncclLaunchPrepare/scheduleCollTasksToPlan` |
| batch formation | `src/enqueue.cc::ncclAddWorkBatchToPlan` |
| args/storage | `src/enqueue.cc::finishPlan/uploadWork` |
| launch coordination | `src/group.cc::doLaunches` |
| CUDA launch | `src/enqueue.cc::ncclLaunchKernel` |
| device dispatch | `src/device/common.h::ncclKernelMain/RunWorkBatch` |
| AllReduce consumer | `src/device/all_reduce.h::runRing/runTreeSplit` 及各 `RunWorkColl` 特化 |

---

## 8. 总结

可以把这四项机制压缩成一句话：

> NCCL 先用 topology/tuner 在 host 选择 algorithm + protocol，再把选择结果编码为 channel、warp、chunk、proxy 和 `devFuncId`；kernel 发射后，算法通过 `Primitives` 使用协议特定的 buffer/flag/FIFO/direct 语义完成每个 step。

其中最重要的因果关系是：

```text
algorithm/protocol choice
  → chunk/pattern/connection mode
  → Primitives specialization
  → work/batch/kernel selection
  → actual wait/address/reduce/copy/post behavior
```

如果只看 `all_reduce.h`，只能看到算法步骤；如果只看 `prims_simple.h`，只能看到搬运细节；只有把 tuning、chunking、plan、kernel dispatch 和 Primitives 串起来，才能解释 NCCL 为什么在当前硬件、消息大小和 registration 条件下选择并执行某条通信路径。
