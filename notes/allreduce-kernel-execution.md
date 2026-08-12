# NCCL AllReduce 的算法实现与“kernel launch → 执行完毕”全流程

> 分析对象：本仓库 `nccl/`（NCCL v2.29.7 分支）当前代码。本文聚焦 `ncclFuncAllReduce` 的设备端实现，以及从 `ncclLaunchKernel` 发射 kernel 到本地 kernel 返回之间的执行、同步和 proxy 协作流程。
>
> 本文是 [`user-thread-enqueue-kernel-flow.md`](user-thread-enqueue-kernel-flow.md) 的下篇：上篇说明“用户线程 enqueue → plan → kernel launch”，本文从 `ncclLaunchKernel` 接续，说明 Classic/Symmetric 两套 kernel 如何执行 AllReduce，以及它们如何与 GPU memory、transport 和 proxy 协同。
>
> 文中的源码位置优先使用文件和函数/符号名；行号会随 NCCL 版本变化，不能替代源码本身。除非特别说明，本文描述的是当前工作区代码，而不是所有历史版本的统一行为。

---

## 0. 一句话概览

AllReduce 在当前代码中有两条设备端路径。两条路径可以共用 host 侧的 `ncclLaunchKernel` 包装，但 device kernel、work 参数和同步机制不同。

1. **Classic 路径**（`src/device/all_reduce.h`）：以 `Primitives` 为核心，按“算法 × 协议”特化出 `RunWorkColl<ncclFuncAllReduce, T, RedOp, Algo, Proto>`。Ring、Tree、CollNet Direct、CollNet Chain、NVLS 和 NVLS_TREE 都在这里实现。数据移动是否经过 proxy 取决于协议、transport、buffer registration 和 direct flags；不能简单等同为“节点内直连、节点间 proxy”。

2. **Symmetric 路径**（`src/device/symmetric/all_reduce.cuh`）：面向支持的 NVL/LSA 域内对称内存场景，由单独生成的 kernel 使用 `ncclSymPtr::peerPtr`、multimem 或 LL A2A session 完成 one-shot/少步骤的数据交换。它不使用 Classic 的 `connFifo`、`Primitives` 或 `ncclDevFuncTable`，当前调度器将其 plan 标记为 `hasProxyOps = false`；但它仍然使用 LSA barrier 或 LL A2A mailbox/slot 等同步结构。

设备端调用链可以概括为：

```text
host: ncclLaunchKernel(comm, plan)
        ├── Classic plan
        │     └── __global__ ncclDevKernel_<suffix>
        │             └── ncclKernelMain
        │                   └── RunWorkBatch → RunWorkColl<AllReduce,...>
        │                         └── runRing / runTreeSplit / CollNet / NVLS ...
        │                               └── Primitives::process
        └── Symmetric plan
              └── generated __global__ Symmetric wrapper
                    └── ncclSymkRun_* device function
                          └── symmetric allreduce implementation
```

Classic generic kernel 内部还存在第二层 dispatch：外层 `ncclDevKernel_Generic` 进入 `ncclKernelMain` 后，按 batch 的 `funcId` 调用 `ncclDevFuncTable[funcId]()`；这只是 Classic 的 generic path，不是 Symmetric 的入口。

---

## 1. 算法与协议的取值

算法和协议枚举定义在 `nccl/src/include/plugin/nccl_tuner.h`：

```c
// 算法
NCCL_ALGO_UNDEF          = -1
NCCL_ALGO_TREE           = 0
NCCL_ALGO_RING           = 1
NCCL_ALGO_COLLNET_DIRECT = 2
NCCL_ALGO_COLLNET_CHAIN  = 3
NCCL_ALGO_NVLS           = 4
NCCL_ALGO_NVLS_TREE      = 5
NCCL_ALGO_PAT            = 6    // PAT 不用于 AllReduce

// 协议
NCCL_PROTO_LL             = 0  // 低延迟，小消息
NCCL_PROTO_LL128          = 1  // cache-line/flag 格式，中小消息
NCCL_PROTO_SIMPLE         = 2  // 通用，大消息带宽
```

AllReduce 在 `all_reduce.h` 中实际提供的 `RunWorkColl` 特化如下：

| 算法 | SIMPLE | LL | LL128 |
|------|:------:|:--:|:-----:|
| RING | ✅ `runRing` | ✅ | ✅ |
| TREE | ✅ `runTreeSplit`，特定 CUDA 条件下为 `runTreeUpDown` | ✅ | ✅ |
| COLLNET_DIRECT | ✅ | — | — |
| COLLNET_CHAIN | ✅ | — | — |
| NVLS | ✅ | — | — |
| NVLS_TREE | ✅ | — | — |

因此，AllReduce 的 CollNet/NVLS 类算法当前只使用 SIMPLE；Ring/Tree 可以使用三种协议。这里的“协议”描述的是 `Primitives` 的数据格式和同步方法，不等于某一类 transport 一定使用 proxy。

---

## 2. Classic 路径：AllReduce 各算法的 kernel 实现

`all_reduce.h` 中所有 Classic 算法共享模板入口 `RunWorkColl<...>::run(tid, nthreads, work)`，其中 `work` 是 `ncclDevWorkColl`。每个实现主要完成两件事：

1. 从 channel topology 中取出连接，例如 Ring 的 `prev/next`、Tree 的 `up/down`、CollNet 的 `up/down/out` 或 NVLS 的 `up/down/out`。
2. 构造合适的 `Primitives<T, RedOp, Fan, Direct, Proto, ...>`，调用 `directSend`、`directRecv`、`directRecvReduceDirectSend` 等操作。

`Fan` 表示收发扇出，例如 `FanSymmetric<1>` 是一收一发，`FanAsymmetric<3,1>` 是最多三路接收、一路发送。Tree 中的 arity 是 connection slot 的上限：内部二叉树的两个真实 child 加一个 intra-node/local slot，不能直接理解为三个远端子节点。

### 2.1 RING（`runRing`）

Ring 对每个 chunk 执行 reduce-scatter 和 allgather，总体需要 `2(k-1)` 次邻居操作，`k = nRanks`。源码中的循环变量和概念步骤如下：

```text
step 0       : directSend
源码 j=2..k-1: directRecvReduceDirectSend，共 k-2 次中间归约转发
step k-1     : directRecvReduceCopyDirectSend(postOp=true)
               完成最终归约，把结果写入本 rank 并继续转发
源码 j=1..k-2: directRecvCopyDirectSend，共 k-2 次结果转发
final        : directRecv，把邻居负责的最终 chunk 收回 output
```

源码中两个循环使用的是 `j < nranks` 和 `j < nranks-1`，因此不要把源码 `j` 直接当作完整的 step 编号。chunk 位置由 `ring->index`、`nranks` 和 `j` 计算，保证每个 rank 在每一步处理正确的 chunk。`postOp` 只在 chunk 首次完成全量归约时执行，例如 Sum 的平均化或特定 datatype 的后处理。

Ring 使用 `FanSymmetric<1>` 和 `Direct=1`。是否真的采用 direct buffer，仍由连接 flags、registration 和 transport 共同决定。

### 2.2 TREE（`runTreeSplit` / `runTreeUpDown`）

Tree 的基本语义是 reduce 上行和 broadcast 下行，但当前实现需要区分 root 与非 root：

- **非 root**：`runTreeSplit` 将线程划分为两个组。reduce 组使用 `FanAsymmetric<NCCL_MAX_TREE_ARITY, 1>`，从 `tree->down[]` 收数据、归约后发往 `tree->up`；broadcast 组使用 `FanAsymmetric<1, NCCL_MAX_TREE_ARITY>`，从 `tree->up` 接收并发送给 `tree->down[]`。叶子节点分别退化为 `directSend` 或 `directRecv`，内部节点执行 reduce-copy-forward 或 receive-copy-forward。
- **root**：root 没有 `tree->up`，使用 `FanSymmetric<NCCL_MAX_TREE_ARITY_TOP>` 同时从 `down` 接收并向 `down` 发送，在一次 `directRecvReduceCopyDirectSend(postOp=true)` 中完成归约和广播。root 并不是简单地把线程再分成两个独立的 reduce/broadcast 组。

默认特化是 `runTreeSplit`。`runTreeUpDown` 只在 `CUDART_VERSION >= 11020 && CUDART_VERSION < 11040 && __CUDA_ARCH__ >= 800` 的编译条件下启用，也就是 CUDA 11.2/11.3 范围内的特定架构组合；其余情况使用 split 版本。

Tree reduce 侧即使当前线程组不执行 direct 操作，`Primitives` 构造仍可能使用 `Direct=1`，因为构造阶段需要与对端交换 direct pointer；这不是说该线程组一定会直接访问对端 buffer。

### 2.3 COLLNET_DIRECT（交换机 reduce）

CollNet Direct 使用网内交换机或网络插件提供的归约能力。kernel 根据 topology 将线程分为 gather、broadcast、scatter 和 reduce 组：

```text
gather   —— 从网络侧汇聚本 rank 需要的片段
bcast    —— 从网络接收结果并向本地 down 广播
scatter  —— 将本地数据按 head 分发给网络
reduce   —— 收 down 数据并归约后发往网络 out
```

`nHeads` 表示并行的 head 数，`loopSize` 由 channel 数、head 数和 chunk size 决定。`netRegUsed` 会改变跨 channel 的 buffer 排布；`sendPeerNotify/recvPeerNotify` 则用于某些只需通知、不需要普通 payload copy 的路径。是否经过 proxy 由 CollNet transport 和具体连接决定。

### 2.4 COLLNET_CHAIN

CollNet Chain 将网络连接组织成链：上半线程做 reduce-up，下半线程做 broadcast-down。reduce 侧使用 `tree->down[0] → tree->up`，broadcast 侧使用 `tree->up → tree->down[0]`。链首在 broadcast 入口执行 `postOp`，其余节点执行接收或接收后转发。它与 Tree 共享“reduce 后 broadcast”的抽象，但 topology 和 connection index 不同。

### 2.5 NVLS（NVLink SHARP / multimem）

NVLS 利用 NVLink 共享域的 NVLS/multimem 能力做节点内归约。实现分为 `work->oneNode` 和跨节点两类：

- **单节点**：scatter → NVLS head 执行 `directRecvDirectSend` → gather。
- **跨节点**：节点内使用 NVLS，跨节点使用 NET。reduce 侧将 head chunk 发往 `nvls->out`，broadcast 侧从网络接收后再通过 NVLS 广播。

线程组数量会随 `work->regUsed` 变化。NVLS 的节点内部分可以是 direct/multimem；跨节点的网络 leg 仍可能由 proxy 驱动。

### 2.6 NVLS_TREE

NVLS_TREE 将 NVLS 的节点内 reduce/broadcast 与 Tree 的跨节点层级结合：节点内使用 NVLS，跨节点通过 `treeUp/treeDown` 连接。源码中对应的 reduce/broadcast `Fan` 上限为 `<3,1>` 和 `<1,3>`，其 slot 语义仍包含 topology 中的本地/层级连接。

---

## 3. 三种协议在 kernel 层的区别

Ring/Tree 可以复用同一个算法骨架，但 `Proto` 参数会选择不同的 `Primitives` 实现：

| 协议 | 主要实现 | 数据格式 | 主要同步方式 | 典型用途 |
|------|----------|----------|--------------|----------|
| SIMPLE | `prims_simple.h` | 连续 payload；启用 GPU↔proxy FIFO 时使用 `connFifo[step].size` | `head`/`tail` step 计数，必要时配合 `connFifo` | 大消息、带宽优先 |
| LL | `prims_ll.h` | `ncclLLFifoLine`，每行有 data 和 flag | step 与每行 flag | 小消息、低延迟 |
| LL128 | `prims_ll128.h` | 128B cache-line 布局，line 尾部有 flag | step 与 cache-line flag，host 侧还受 GDR/flush 条件影响 | 中小消息、带宽优先 |

三者都有“按 step 推进”的流水窗口，但不能把它们都简化成 SIMPLE 的 `connFifo`。`connFifo` 可能为空；direct read/write、registered buffer、P2P/NVLS 等路径也可能绕开普通的 FIFO payload。

---

## 4. 从 kernel launch 到执行完毕的全流程

### 4.1 launch 配置：grid = plan 使用的 channel 数

group end 阶段的 `group.cc::doLaunches` 调用 `ncclLaunchPrepare`，随后对每个 plan 调用 `ncclLaunchKernel`。`ncclLaunchKernel` 使用：

- `grid.x = countOneBits(plan->channelMask)`；
- 一个 block 对应一个 active channel，`blockIdx.x` 是 channel mask 中第几个置位 bit，而不是 channel id 本身；
- `block.x = plan->threadPerBlock`；Classic plan 通常由所调度 work 的最大 `nWarps * WARP_SIZE` 决定，并至少满足 NCCL 的最小线程数；
- `plan->kernelFn` 决定是某个 Classic 特化 kernel、Classic generic kernel，还是 Symmetric generated kernel。

`RunWorkBatch` 处理某个 work 时再使用该 work 自己的 `nWarps * WARP_SIZE` 作为 `subtn`。因此，“plan 的 block 大小”和“单个 work 的参与线程数”是两个层次，不应混写成每个 channel 都有一个独立 kernel 特化。

Classic 特化与 generic dispatch 的关系是：

```text
plan->kernelFn = ncclDevKernel_<suffix> 或 ncclDevKernel_Generic
        ↓
ncclKernelMain
        ├── SpecializedFnId 命中：SpecializedRunWorkBatch().run()
        └── 未命中：ncclDevFuncTable[ncclShmem.funcId]()
```

一个 plan 只有一个外层 `kernelFn`；generic kernel 内部可以按每个 batch 的 `funcId` 选择 device function。Symmetric 不通过这张 Classic function table。

### 4.2 `ncclKernelMain`：加载 args、channel 和 work batch

`src/device/common.h::ncclKernelMain` 的主要步骤是：

1. 将 kernel args 放入 shared memory，避免频繁从参数区读取或发生不必要的 thread-local spill。
2. 根据 `channelMask` 的 popcount，将 `blockIdx.x` 反映射为实际 `channelId`。
3. 由不同 warp 分工加载 `ncclKernelComm`、当前 `ncclDevChannel` 和本 block 的 work batch。
4. `__syncthreads()` 发布 shared memory 内容。
5. 循环执行当前 batch；如果存在 `nextBatchIx`，同步后装载下一个 batch；没有下一个 batch 时退出。
6. 对被 profiler 激活的 work 执行对应的 START/STOP/FINI 记录。

`loadWorkBatchToShmem` 根据 batch 的 offset bitset 将 `ncclDevWorkColl` 拷贝到 `ncclShmem.workStorage`，并设置 `nWorks`、`funcId` 和 `workType`。一个 channel 的多个 batch 可以通过 next-batch 链接在同一个 kernel 中继续处理。

这一节只描述 Classic。Symmetric kernel 使用 `ncclSymkDevWorkArgs4K` 和生成的 `ncclSymkRun_*` 入口，不经过 `ncclKernelMain`、`RunWorkColl`、`Primitives` 或 `ncclDevFuncTable`。

### 4.3 `RunWorkBatch` → `RunWorkColl`：消费 work

`src/device/common.h::RunWorkBatch::run`：

1. 对带 pointer 参数的 reduction op，先读取每个 work 的 `redOpArg`。
2. 按顺序遍历当前 batch 的 `nWorks`。
3. 对每个 work 调用 `RunWorkColl<Fn, T, RedOp, Algo, Proto>().run(tid, subtn, work)`，其中 `subtn = work->nWarps * WARP_SIZE`。
4. 相邻 work 的 `nWarps` 不同时插入必要的 block barrier，防止不同线程子集之间继续重叠。

`RunWorkColl::run` 再通过 `ncclCollCbdPart` 将 collective 按 channel、grid 和 chunk 切分，得到当前 block 负责的 `gridOffset`、`channelCount` 和 `chunkCount`，最后进入 §2 中的算法步骤。

### 4.4 device ↔ consumer 的 step/FIFO 同步

Classic 的连接设备结构是 `ncclConnInfo`（见 `src/include/device.h`），主要字段包括：

- `buffs[proto]`：按协议划分的 payload buffer；
- `head` / `tail`：单调递增的 step 计数器；
- `connFifo`：可选的 `ncclConnFifo` 描述符数组，SIMPLE 的 GPU↔proxy 路径用 `size` 标记槽状态，并可能携带 offset。

这里最容易混淆的是：`head` 和 `tail` 不是一个统一的“发布指针”。在 `prims_simple.h::loadSendConn` 与 `loadRecvConn` 中，它们的角色按发送/接收方向分别绑定：

```text
发送方 GPU                                      接收方 GPU / proxy
──────────                                      ────────────────
RoleWaitSend 读 send conn->head                 等待 free-slot credit
    ↓
写 connFifo[slot].size；读取 proxy 发布的 offset（若启用）
选择 FIFO/direct 地址并写 payload
system fence
RolePostSend 写 send conn->tail                  发布 data-ready step
                                                ↓
                                      RoleWaitRecv 读 recv conn->tail
                                      读取/归约 payload
                                      RolePostRecv 写 recv conn->head
                                                ↑
发送方下一轮 RoleWaitSend 看到 credit
```

对 GPU↔proxy 的 NET/SIMPLE 连接，同一对共享内存映射会让 proxy 看到 GPU 发布的 tail，并把 head credit 写回 GPU；具体 host 侧字段名会出现在 `resources->recvMem`、`resources->sendMem` 或 GDR sync 地址中，但逻辑仍是“tail 表示数据可用、head 表示槽位可复用”。

`prims_simple.h::process` 将单个 slice 的操作组织为：

1. `RoleWaitRecv` 等待 receive tail，或 `RoleWaitSend` 等待 send head 提供空槽；
2. 对 send connector 写入当前 slice 的 `connFifo[slot].size`；若使用 `NCCL_MODE_OFFSET`，读取 proxy 已发布的 offset，再设置 step 对应的 source/destination pointer；
3. 执行 copy/reduce；此时单独看到 `size` 不代表 payload 已经 ready；
4. 对发送数据执行 system fence；
5. `RolePostSend` 或 `RolePostRecv` 推进对应的 tail/head，完成数据或 credit 的发布。

`connFifo` 只在连接启用 `ConnFifoEnabled` 时参与上述描述符步骤；direct read/write、registered buffer 或某些通知路径可能改变 payload 的实际地址，但不能改变 head/tail 的“credit/data-ready”方向。

### 4.5 transport：直连与 proxy 是条件分支

Classic 的数据通路由 connection flags、registration、协议和 transport 决定：

- **P2P direct**：满足 direct read/write 条件时，GPU 可以直接访问对端 GPU buffer；`Primitives` 的 `DirectRead/DirectWrite` flags 决定具体地址和访问方向。
- **P2P/SHM 的 memcpy/CE 路径**：即使是 intra-node，也可能使用 `p2pSendProxyProgress` 或 `shmSendProxyProgress`，由 proxy 线程提交 CUDA memcpy、查询 event，并在完成后更新接收侧 tail。
- **NET**：通常由 NET proxy 调用 `isend/irecv/test/iflush`，但 registered/GDR 和 device-side network path 会改变 buffer 和 flag 的细节。
- **CollNet/NVLS**：节点内可能使用 direct/NVLS，多节点的网络 leg 仍可能使用 proxy；CollNet 的通知路径和 NVLS 的 head/out 连接也不应统一简化成一种 FIFO。

因此，“intra-node 一定不经过 proxy、inter-node 一定由同一个 proxy FIFO 搬运”不是可靠的源码级结论。更准确的判断方式是沿 `transportResources`、connection flags 和 `proxyProgress` 注册点追踪具体连接。

### 4.6 proxy op 的创建、启动与 progress

proxy 有独立的 host/control-plane 生命周期，不是 `uploadWork` 的副作用，也不等于在 kernel 所在 CUDA stream 中运行。

#### 4.6.1 plan 到 proxy op

Classic plan 的 proxy op 来源于各 channel 的 `proxyOpQueue`：

```text
schedule*TasksToPlan
    ↓
每个 channel 的 proxyOpQueue
    ↓ finishPlan
合并为 plan->proxyOpQueue，并设置 plan->hasProxyOps
    ↓ ncclLaunchPrepare
根据 launch/capture 情况在 hostStream 上排入 host callback
    ↓ hostStreamPlanTask
uploadProxyOps → ncclProxySaveOp → ncclProxyStart
```

`finishPlan` 对 Symmetric plan 直接返回，因为 Symmetric scheduler 已经将 `hasProxyOps` 设为 false。`uploadWork` 位于 `ncclLaunchKernelBefore_NoUncapturedCuda`，只负责把 Classic 的 device work 写入 args/FIFO/persistent storage；对 Symmetric、CE 和 RMA plan 会直接跳过，因此不能把它写成 proxy op 的生成步骤。

在需要 host callback 的模式下，`ncclLaunchPrepare` 将 `hostStream` 的 callback 排在前面，并让 launch stream 等待 host stream；在不需要 callback 的路径下，`ncclLaunchKernelAfter_NoCuda` 直接调用 `hostStreamPlanTask`。这解释了为什么 proxy op 的启动可能位于 kernel 发射前的 host callback，也可能位于 kernel launch 后的 no-CUDA callback；两种情况都由 plan 和 stream ordering 保证正确性。

#### 4.6.2 proxy 主循环和 NET progress

`src/proxy.cc::progressOps` 遍历 active `ncclProxyArgs`，调用连接注册的 `transport->send.proxyProgress` 或 `transport->recv.proxyProgress`。NET 发送端 `net.cc::sendProxyProgress` 的高层状态是 `Ready → Progress → None`：

1. `Ready`：为 op 计算 step base，初始化 posted/transmitted/done 计数。
2. **post**：在资源允许时为 GPU/共享 buffer 开出可用窗口。shared 与 non-shared 资源的 head 更新时机不同，不能概括为每次 post 都直接写同一个 `sendHead`。
3. **transmit**：检查 GPU 已发布的 tail 和 payload readiness。SIMPLE 通常检查 `connFifo[slot].size` 与 tail；LL/LL128 还要检查各自的 flag，LL128 是否需要 host 侧检查还受 GDR 条件影响，然后调用 `ncclNet->isend`。
4. **done**：`ncclNet->test` 完成后先将 SIMPLE 槽的 `size` 重置为 `-1`，再按 shared/non-shared 资源规则推进 head 和 `done`。
5. 所有 sub 完成后将 op 状态置为 `None`。

NET 接收端 `net.cc::recvProxyProgress` 对称地完成：

```text
irecv → test → 必要的 GDR flush/iflush →
确认数据对 GPU 可见 → 更新 recv tail → 唤醒 GPU RoleWaitRecv
```

接收端必须在数据可见之后再推进 tail；否则 GPU 可能只看到“step 已到达”而读到尚未完成的 payload。P2P/SHM 的 proxy progress 使用 CUDA stream 和 event 完成类似的“GPU/CPU buffer copy → event completion → 更新接收 tail”流程，但它们不是 NET `isend/irecv`。

### 4.7 kernel 何时算“执行完毕”

需要区分四个完成边界：

- **单个 work**：该 work 的 `RunWorkColl::run` 完成所有算法步骤并返回。
- **单个 channel block**：`RunWorkBatch::run` 遍历完当前 batch，`ncclKernelMain` 没有下一个 batch，block 退出。
- **本地 kernel/stream**：所有 channel block 都返回后，CUDA kernel 在其 launch stream 上完成；用户 stream 的同步可以观察到这个本地完成。
- **proxy op**：proxy 的各 sub 完成后独立进入 `None`。它与 kernel 返回不是同一个事件，可能早于或晚于本地 kernel 的返回。

`common.h::profiler` 只有在 work 的 `profilerEnabled` 为真时才写入 `workStarted/workCompleted`。该标志由 NCCL profiler plugin 和 activation mask 决定，并不是每个 work 都无条件记录。DCCL 的 GPU timing 走 `trace_probe.cc` 中的 CUDA event/完成 callback 路径，不能与 `workCompleted` 等同。

因此：

```text
kernel return ≠ remote network bytes 已物理到达最终显存
```

NCCL 通过每条连接的 tail/head、payload readiness、direct/registered transport 和 proxy progress 保证接收方在使用数据前完成相应等待。一次 group 内的多个 collective 则由 `group.cc::doLaunches` 的多 round 机制串联，通常不是在一个 kernel 内跨 collective 做全局同步。

---

## 5. Symmetric 路径：NVL/LSA 域内的 one-shot AllReduce

Symmetric 不是 `ncclKernelMain` 中的一个 `if` 分支，而是 host scheduler 和 device generated kernel 的独立路径。

### 5.1 host 侧选择与 fallback

当 communicator 支持 Symmetric 时，`enqueue.cc` 会调用 `ncclMakeSymmetricTaskList`：

1. 用 `ncclSymkAvailable` 按 collective、op、datatype 和 count 判断任务是否有 Symmetric kernel 可能性。
2. 查询 send/recv buffer 的 window 和 registration type，将可兼容任务按 function/op/type/reg type 分组。
3. 用 `ncclSymkPickKernel` 选择 kernel id、channel 数和 warp 数。
4. 对 LL 类 Symmetric kernel 检查 buffer registration、单进程多 GPU、`SYM_NOWIN_ENABLE` 和 legacy cost model；不满足条件时将任务放回 legacy queue。

因此，Symmetric 不是“只要 `symmetricSupport` 就必然使用”。不可用、没有合适 kernel 或需要 fallback 时，任务仍然由 Classic 的 `ncclGetAlgoInfo` 和 legacy scheduler 处理。

Symmetric scheduler 创建 plan 时设置：

```text
plan->isSymColl  = true
plan->hasProxyOps = false
plan->kernelFn   = ncclSymkKernelList[...]
plan->kernelArgs = ncclSymkDevWorkArgs
```

随后仍由 `ncclLaunchKernel` 使用 `plan->kernelFn` 发射，但 `finishPlan` 不再构造 Classic work batch，`uploadWork` 也直接跳过。Symmetric kernel 使用自己的动态 shared memory 和 args layout。

### 5.2 device kernel variants

`src/device/symmetric/all_reduce.cuh` 当前包含几类 AllReduce kernel：

- `ncclSymkRun_AllReduce_RSxLD_AGxST`：ReduceScatter 逻辑使用对称 peer pointer LD，结果通过 ST 写回所有 peer。
- `ncclSymkRun_AllReduce_RSxLDMC_AGxSTMC`：使用 multimem load/store，由硬件 multicast 语义完成域内读取/归约/写回。
- `ncclSymkRun_AllReduce_AGxLL_R` 和 `_AGxLLMC_R`：使用 `ncclLLA2ASession` 做 LL 风格 all-to-all reduce；它们有自己的 slot/epoch/mailbox，不是简单的 peerPtr 一次读取。

核心 `allreduceDeep` 的 RSxLD 类路径大致是：

1. 按 warp 和 pack 划分数据，并通过 `flattenIx` 在 rank、block、warp 之间轮转负载。
2. 读取本 rank chunk，再分批读取 peer chunk，通过 `applyReduce` 累加。
3. 将结果 ST 到各 peer 的对称 output 地址。
4. `allreduceEnds` 处理首尾未对齐的标量尾巴。

multimem 版本则使用 `applyLoadMultimem` 和 `multimem_st_global`，由 LSA/multimem 硬件完成相应的多播访问。

### 5.3 Symmetric 的同步语义

Symmetric 不使用 Classic `connFifo` 和 `Primitives::process`，但并不意味着“没有同步”。主要机制包括：

- `ncclLsaBarrierSession<ncclCoopCta>`：在 reduce 前后对 LSA team 的 resource/barrier 做 arrive、wait 或 sync，保证 peer memory 的读写阶段有明确边界。
- `ncclLLA2ASession<ncclCoopCta>`：AGxLL variants 使用 epoch、slot 和 mailbox 传递数据。
- `ncclCoopCta` 本身只是 CTA 级 cooperative object：`thread_rank = threadIdx.x`、`size = blockDim.x`、`sync = __syncthreads()`。它不是 `cudaLaunchCooperativeKernel` 意义上的整个 grid cooperative launch。

在支持的 CUDA/架构组合下，host launch 还可能设置 programmatic stream serialization 属性；这同样不等于 cooperative launch。Symmetric 的适用范围是满足窗口、registration、NVL/LSA team 和 kernel capability 条件的域内场景，不是通用跨节点 NET AllReduce 的替代实现。

---

## 6. 算法选择与拓扑（简述）

### 6.1 Classic 选择

Classic 任务的算法/协议选择主要由 `enqueue.cc::ncclGetAlgoInfo`、`initCollCostTable`、`updateCollCostTable` 和 `topoGetAlgoInfo` 完成：

1. topology 初始化阶段构建 ring、tree、CollNet 和 NVLS 相关结构，并根据硬件/transport 能力计算 bandwidth、latency 和可用性。
2. `updateCollCostTable` 过滤不支持的算法/协议，并生成各组合的估算时间。
3. tuner plugin 可以提供或调整 cost table；随后 `topoGetAlgoInfo` 选择估算时间最小的合法 `(algorithm, protocol)`，并确定 channel/thread 参数。
4. `NCCL_ALGO`、`NCCL_PROTO` 等环境变量可以限制选择范围；若没有合法组合，NCCL 会报 no algorithm available。

典型倾向是：

- **Ring**：通用路径，适合中大消息和拓扑较均衡的场景。
- **Tree**：reduce/broadcast 层级更突出，适合部分规模和延迟模型。
- **CollNet**：需要配置启用且有可用的网内归约 transport/plugin。
- **NVLS/NVLS_TREE**：需要 communicator 的 NVLS 支持及满足相应 datatype、拓扑和节点条件。

不要把“Ring 和 Tree 都建好了”理解成一次 AllReduce 同时执行两个算法；一个具体 work 会带有一个已选定的 algorithm/protocol，只有不同任务或不同计划才可能分别使用不同算法。

### 6.2 thread threshold 与协议倾向

`NCCL_LL_THREAD_THRESHOLD`、`NCCL_LL128_THREAD_THRESHOLD` 和 `NCCL_SIMPLE_THREAD_THRESHOLD` 以及 `NCCL_THREAD_THRESHOLDS` 主要影响线程/channel 调整和每个算法协议的工作粒度，不应被理解为一个独立的“Lite”查表入口。真正的算法/协议选择来自 cost table、topology/tuning 结果和环境配置。

### 6.3 Symmetric 选择

Symmetric 在 legacy algorithm selection 之前先尝试从任务队列中筛选可用任务；对于被 fallback 的任务，再回到 Classic cost model。因此，同一进程中可能同时存在 Symmetric plan 和 Classic plan，但每个 plan 的 device args、kernelFn 和同步协议仍然独立。

---

## 7. 关键源码导航

| 关注点 | 文件与函数/符号 |
|--------|----------------|
| AllReduce 算法 kernel 体 | `nccl/src/device/all_reduce.h::runRing/runTreeSplit/runTreeUpDown` |
| CollNet/NVLS 实现 | `nccl/src/device/all_reduce.h::RunWorkColl<...>` 特化 |
| 算法/协议枚举 | `nccl/src/include/plugin/nccl_tuner.h` |
| Classic kernel 入口与 dispatch | `nccl/src/device/common.h::ncclKernelMain/RunWorkBatch`、`common.cu::ncclDevKernel_Generic` |
| SIMPLE step/FIFO 同步 | `nccl/src/device/prims_simple.h::process/loadSendConn/loadRecvConn` |
| LL/LL128 数据格式 | `nccl/src/device/prims_ll.h`、`prims_ll128.h` |
| Classic plan 与 work 上传 | `nccl/src/enqueue.cc::finishPlan/uploadWork` |
| proxy op 上传与启动 | `nccl/src/enqueue.cc::uploadProxyOps/hostStreamPlanTask/ncclLaunchPrepare` |
| kernel 前后 host hook | `nccl/src/enqueue.cc::ncclLaunchKernelBefore_NoUncapturedCuda/ncclLaunchKernelAfter_NoCuda` |
| proxy 主循环 | `nccl/src/proxy.cc::progressOps` |
| NET send/recv progress | `nccl/src/transport/net.cc::sendProxyProgress/recvProxyProgress` |
| P2P/SHM memcpy proxy | `nccl/src/transport/p2p.cc::p2pSendProxyProgress`、`shm.cc::shmSendProxyProgress` |
| Symmetric task split/selection | `nccl/src/scheduler/symmetric_sched.cc::ncclMakeSymmetricTaskList/ncclSymmetricTaskScheduler` |
| Symmetric AllReduce kernel | `nccl/src/device/symmetric/all_reduce.cuh` |
| Symmetric CTA/barrier primitive | `nccl/src/include/nccl_device/coop.h::ncclCoopCta`、`nccl/src/include/nccl_device/impl/lsa_barrier__funcs.h` |
| LL A2A mailbox | `nccl/src/include/nccl_device/impl/ll_a2a__funcs.h`、`nccl/src/include/nccl_device/impl/ll_a2a__types.h` |
| Classic 算法成本模型 | `nccl/src/enqueue.cc::updateCollCostTable/topoGetAlgoInfo`、`nccl/src/graph/tuning.cc::ncclTopoGetAlgoTime` |
| host launch round | `nccl/src/group.cc::doLaunches`、`nccl/src/enqueue.cc::ncclLaunchKernel` |
| NCCL kernel-channel profiler timestamps | `nccl/src/device/common.h::profiler` |
| DCCL GPU timing | `nccl/src/trace/trace_probe.cc` 中的 launch/event/completion 路径 |

---

## 8. 小结：从 launch 到完成的因果链

### Classic

1. **准备与发射**：`ncclLaunchPrepare` 生成 plan；`ncclLaunchKernel` 在 launch stream 上发射 Classic kernel，grid 为 active channel 数。
2. **加载**：每个 block 在 `ncclKernelMain` 中加载 comm、channel 和 work batch，定位实际 `channelId`。
3. **分派**：`RunWorkBatch` 遍历 work；generic kernel 必要时通过 `funcId` 调 `ncclDevFuncTable`。
4. **执行**：`RunWorkColl` 进入 Ring/Tree/CollNet/NVLS 等算法；每一步调用 `Primitives`，按 protocol 执行 payload、reduce 和 step 同步。
5. **搬运**：连接可能使用 P2P/NVLS direct，也可能由 P2P/SHM/NET/CollNet proxy 或 network transport 完成；head 是 credit，tail 是 data-ready，具体 buffer/flag 由 protocol 和 transport 决定。
6. **本地完成**：所有 work/batch/channel block 返回后，kernel 在本地 stream 上完成；NCCL profiler timestamp 只有被激活的 work 才会记录，DCCL timing 另走 CUDA event。
7. **proxy 完成**：proxy op 独立推进到 `None`；它与 kernel return 是不同的完成边界。

### Symmetric

1. **筛选**：`ncclMakeSymmetricTaskList` 按可用性和 registration 分组，`ncclSymkPickKernel` 选择 kernel；不满足条件时 fallback 到 Classic。
2. **构建 plan**：scheduler 设置 `isSymColl`、Symmetric kernelFn 和专用 args，明确不生成 Classic proxy ops。
3. **发射**：仍通过 `ncclLaunchKernel` 包装，但进入 generated `__global__` Symmetric wrapper，再调用 `ncclSymkRun_*` device function，而不是 `ncclKernelMain`。
4. **执行与同步**：RSxLD/Multimem 或 AGxLL variant 直接处理对称内存，使用 LSA barrier 或 LL A2A session；`ncclCoopCta` 只表示 CTA 级协作。
5. **完成**：Symmetric kernel 完成后由本地 launch stream 观察；其没有 Classic proxy 生命周期，但内部 barrier/mailbox 必须完成相应的 peer-memory 可见性。

理解 §2 的算法步骤、§4.4 的 head/tail 双向协议、§4.6 的 proxy control-plane，以及 §5 的 Symmetric 分流，就能准确解释 AllReduce 在 NCCL 中“kernel 启动后到底发生了什么”。协议与 primitive 的通用细节见 [`protocols-primitives-kernel-launch.md`](protocols-primitives-kernel-launch.md)，kernel、proxy、FIFO 和 plan 的不同完成边界见 [`completion-reclaim-profiling.md`](completion-reclaim-profiling.md)。
