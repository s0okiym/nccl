# NCCL：Chunking 与 pattern、Head/Tail/FIFO、Direct 与 registration

本文基于当前仓库 `nccl/src/` 的 host enqueue、transport、proxy 和 device `Primitives` 实现，聚焦三条必须一起理解的链路：任务如何被切成 chunk/step 并映射到 pattern；GPU 与 GPU/proxy 如何用 head、tail 和 FIFO 推进流水线；buffer registration 如何与 direct read/write 协同决定真实数据地址。

## 1. 先记住一条完整链路

```text
task 的 func / algorithm / protocol / registration
  -> scheduleCollTasksToPlan 分 channel 与 count
  -> calcCollChunking 计算 pattern、chunk、loop、proxy steps
  -> ncclDevWorkColl + ncclProxyOp
  -> transport 建立 conn.head/tail/connFifo/ptrExchange
  -> Primitives 等待 credit、选择 FIFO 或 direct 地址、搬运数据
  -> post/fence 发布 tail 或 head
  -> proxy 依据 tail + connFifo.size/offset 驱动网络或 CE
```

这三个主题不能割裂：chunkSize 决定一次 step 搬运多少数据，pattern 决定每个 loop 需要多少 peer step，registration 决定 proxy 是否能绕过临时 FIFO，而 direct flag 决定 GPU kernel 是写入对端 buffer 还是从对端 buffer 拉取。

## 2. Chunking 与 pattern

### 2.1 几个不同的“大小”

| 名称 | 代码含义 | 主要消费者 |
|---|---|---|
| `stepSize` | `comm->buffSizes[protocol] / NCCL_STEPS`，一个协议 FIFO step 的容量 | transport、proxy、Primitives |
| `chunkSize` | 一个逻辑 chunk 的字节数；通常是 `stepSize * chunkSteps`，之后还要按协议修正和 grain 对齐 | kernel、proxy |
| `sliceSize` | 一个 chunk 内每次 primitive slice 的大小 | kernel primitive、proxy 的 `sliceSteps` |
| `loopSize` | 一次 pattern loop 覆盖的总字节数，包含 channel 和 pattern 的 chunk 数 | proxy 计算 loop 次数 |
| `nsteps` | proxy 需要推进的 step 数，不是总字节数 | transport proxy |
| `countLo/Mid/Hi` | continuous-byte-distribution 把 task 划到不同 channel 后的 element 数 | `ncclCollCbdPart`、kernel |
| `chunkGrainsLo/Mid/Hi` | 上述各 channel 的 chunk 数，以 `ncclProtoGrainSize(protocol)` 为单位 | device work |

单位必须按层区分：`calcCollChunking()` 和 `ncclProxyOp` 中的 `stepSize/chunkSize/sliceSize` 以 bytes 表示；`prims_simple.h` 内同名局部量通常已经除以 `sizeof(T)`，以 elements 表示。下文未特别注明时，host chunk/proxy 字段按 bytes 讨论。

`chunkSteps` 和 `sliceSteps` 并不是所有算法都直接使用。`calcCollChunking` 只在 `protocol == SIMPLE && algorithm == RING` 时采用 task 上的 API 参数；其他算法/协议将二者视为 1。初始值为：

```text
stepSize  = buffSizes[protocol] / NCCL_STEPS
chunkSize = stepSize * chunkSteps
LL        -> chunkSize /= 2
LL128     -> 按 128B line 中 15/16 个 uint64 数据元素换算
```

最终 `chunkSize` 还会执行 `chunkSize = floor(chunkSize / grainSize) * grainSize`。当前 grain 定义为：LL 16B、LL128 由 line/data 布局计算得到 1920B、SIMPLE 512B。device 侧用 `ncclCollCbdPart()` 把 grain 数转换回 element 数，因此 host 与 device 必须使用同一协议 grain。

### 2.2 algorithm 与 pattern 的实际映射

`calcCollChunking()` 不是根据 API 名字简单选择 ring；它根据 collective function 和已选 algorithm 生成 device/proxy 都要理解的 `ncclPattern_t`：

| Collective | Algorithm | Pattern | 每 loop 的 step/chunk 结构 |
|---|---|---|---|
| Broadcast | Tree | `TreeDown` | 1 step、1 chunk |
| Broadcast | 非 Tree | `PipelineFrom` | 1 step、1 chunk |
| Reduce | Tree | `TreeUp` | 1 step、1 chunk |
| Reduce | 非 Tree | `PipelineTo` | 1 step、1 chunk |
| ReduceScatter | PAT / NVLS / CollNet Direct / 其他 | `PatUp` / `Nvls` / `CollnetDirect` / `Ring` | 取决于 pattern；Ring 为 `nRanks-1` steps、`nRanks` chunks |
| AllGather | PAT / NVLS / CollNet Direct / 其他 | `PatDown` / `Nvls` / `CollnetDirect` / `Ring` | 同上 |
| AllReduce | NVLS / NVLS_TREE / CollNet Direct / CollNet Chain / Tree / 其他 | `Nvls` / `NvlsTree` / `CollnetDirect` / `CollnetChain` / `TreeUpDown` / `RingTwice` | RingTwice 为 `2*(nRanks-1)` steps、`nRanks` chunks |

特殊 pattern 的 loop 参数：

- `Nvls` 和 `NvlsTree`：每 loop 有 `nHeads` 个 chunk、1 个 step；NVLS 还用 `headRank` 生成 `loopOffset`。
- `CollnetDirect`：每 loop 有 `collnetDirect.nHeads` 个 chunk、1 个 step，并按 `headRank` 设置 `loopOffset`。
- `Tree*`、`PAT`、`Pipeline*`、`CollnetChain`：每 loop 一个 chunk、一个 step。
- `Ring`：一次 loop 遍历所有 rank 的 chunk，并经历 `nRanks-1` 个通信 step。
- `RingTwice`：AllReduce 的 reduce-scatter 和 allgather 两轮合并为两倍 step 数。

随后：

```text
loopSize = nChannels * nchunksPerLoop * chunkSize
nLoops   = ceil(nBytes / loopSize)
nsteps   = nstepsPerLoop * nLoops * chunkSteps
```

其中 `nBytes` 是本次 channel segment 的 global data bytes，不是按 collective traffic 放大的字节数。源码特别注明 AllReduce 的 traffic 通常乘 2，但 `calcCollChunking` 需要原始 global bytes；如果把 traffic bytes 误传入，loop 和 proxy step 会被错误放大。

### 2.3 algorithm-specific chunk size 调整

初始 chunk 只是上限和起点，代码还根据拓扑并发度、depth、head 数和消息规模收缩它：

| Algorithm | 主要规则 |
|---|---|
| CollNet Direct | 以 `nChannels * nHeads * chunkSize` 与 CollNet depth 的 64/8 倍阈值比较，必要时从 128KB、64KB 继续收缩到 32KB |
| CollNet Chain | 强制使用 SIMPLE 的 step，chunk 上限 256KB，再按 chain depth 的 64/8/1 倍阈值收缩 |
| NVLS | 通常受 `comm->nvlsChunkSize` 限制；跨节点低带宽时可限到 32KB；按 `concurrentOps = nChannels*nHeads` 对小消息收缩到 64/32/16KB |
| NVLS_TREE | 使用 `nvlsChunkSize` 和 `NVLSTREE_MAX_CHUNKSIZE`；默认在节点数不少于 4 时上限 64KB，再按 32/16/4/1 倍并发阈值收缩 |
| Tree + LL128 | 依据 node 数、每节点 rank 数和估算的 LL128 step 数，避免消息不足以填满树流水线 |
| PAT AllGather/ReduceScatter | 小消息分别按 32 或 16 倍 channel 并发阈值收缩，最低仍受 64KB 条件限制 |

registered NVLS 的 AllGather/ReduceScatter 是一个重要例外：chunkSize 改用完整的 SIMPLE step，且注册资源查询可能重新设置 `nMaxChannels`。这说明 registration 不只是 proxy 的旁路优化，也会反馈到 channel 和 chunk 规划。

### 2.4 task 如何分到多个 channel

在 `scheduleCollTasksToPlan()` 中，CollNet/NVLS 和普通 collective 的分配方式不同。

**CollNet/NVLS 路径**：每个 task 使用 `nMaxChannels`，各 channel 均匀处理 task 的逻辑 count，生成 `devWork->collnet.count/chunkCount`，并为每个 channel 建 proxy op。

**普通路径**：使用 continuous-byte distribution：

1. 先按 `trafficBytes` 统计每个 kind（普通、NVLS、CollNet）的预计 channel traffic，保证每 channel 至少约 32KB 的 traffic。
2. `trafficPerByte = ncclFuncTrafficPerByte()`；LL 协议额外乘 4，因为 LL 的 wire/traffic 统计与实际 element bytes 不同。
3. 以对齐后的 cell 切分 task，计算 `cellsLo`、`cellsPerChannel`、`cellsHi`。
4. 将结果编码为 `countLo`、`countMid`、`countHi` 和对应 channel 区间；首尾 channel 可以不同，中间 channel 尽量相同。
5. 对 lo/mid/hi 各自调用 `calcCollChunking(..., nChannels=1, globalBytesPerElement*countX, ...)`，得到三个 chunk grain。
6. 为每个 channel 设置 `loopOffset` 和 `channelSize`，并把相应 proxy op、work batch、channel mask 写入 plan。

这就是为什么一个 collective task 的 `channelLo..channelHi` 可能跨多个 channel，而 `countLo/Mid/Hi` 不能简单理解为三个固定语义分片；它们是为了均衡 traffic 和满足 grain 对齐产生的首/中/尾布局。

### 2.5 `ProxyOp` 是 kernel 与 proxy 的共同契约

`calcCollChunking` 写入的 `ncclProxyOp` 至少包含：`pattern`、`protocol`、`algorithm`、`chunkSize`、`sliceSize`、`chunkSteps`、`sliceSteps`、`loopSize`、`loopOffset`、`nsteps`、`nbytes`、`nChannels` 和 `nPeers`。这些字段必须与 `ncclDevWorkColl` 一致：

- kernel 根据 `count*`、`chunkGrains*` 生成 channel 内 primitive 操作；
- proxy 根据 `nsteps/chunkSteps/sliceSteps` 预留和推进 transport step；
- `loopOffset/channelSize` 使 proxy 注册路径访问 task 在 channel 中对应的 buffer 区间；
- `nPeers`/`nChannels` 是 network plugin 的并发和 flow hint。

registered network path 会进一步改写 proxy contract：设置 `reg=1`、传入 `sendMhandle/recvMhandle` 和真实 buffer；CollNet Direct、NVLS、CollNet Chain 还可能把 `nsteps` 压成 1 或按 head 拆分。不能只修改 kernel chunk 而忽略 proxy op。

## 3. Head、tail 与 FIFO

### 3.1 连接状态的语义

`ncclConnInfo` 的注释按本端 connector 视角定义：

| 字段 | 发送侧 | 接收侧 | 语义 |
|---|---|---|---|
| `head` | 本地可见的 credit | 远端/对端的 head | 接收方已经消费并释放到哪个 step，发送方据此复用 FIFO slot |
| `tail` | 远端/对端的 ready | 本地可见的 tail | 发送方已经发布到哪个 step，接收方据此读取 |
| `step` | 本 connector 的下一操作游标 | 本 connector 的下一操作游标 | 单调递增；真正选 buffer slot 时用 `step % NCCL_STEPS` |
| `stepSize` | SIMPLE FIFO 的 step bytes | SIMPLE FIFO 的 step bytes | 将 step 转成 element 数或 buffer offset |
| `connFifo` | GPU 写描述符，proxy 读 | proxy 写描述符，GPU 读 | 携带 size、shared-buffer offset 等附加信息 |

可以用下面的方向记忆：

```text
发送方                         接收方
  等待 head 取得 credit  <----  postRecv 写 head
  写 FIFO / direct 数据
  fence 后发布 tail      ---->  waitRecv 读 tail，消费数据
                                  消费后再发布 head
```

head/tail 是 step counter，不是字节指针；它们必须和 `NCCL_STEPS` 的 slot 循环、协议 flag、proxy 的 `base`/`done` 一致。step 不一致通常表现为 GPU 在 wait 中自旋，或 proxy 永远没有推进 done。

### 3.2 SIMPLE `Primitives` 的 wait/post 顺序

`prims_simple.h` 中的角色由 template flags 区分：`RoleWaitRecv`、`RoleWaitSend`、`RolePostRecv`、`RolePostSend`。典型流程是：

1. `loadRecvConn/loadSendConn` 读取 connector 的 `step`，并把 step 向上取整到 `SlicePerChunk*StepPerSlice`。
2. WaitRecv 使用 `conn->tail`，WaitSend 使用 `conn->head`；wait 条件按 `StepPerSlice` 检查，并在循环中检查 abort。
3. Send 在真正写入数据前，如果启用了 `connFifo`，把本 slice 的字节数写到 `connFifo[step % NCCL_STEPS].size`。
4. 根据 direct/FIFO 模式选择源或目的地址，执行 copy/reduce/multimem 操作。
5. `postPeer` 对发送数据执行 system fence，然后把更新后的 step 写到 `connStepPtr`；接收角色发布 head。
6. primitive 析构时把 post 角色的 step 保存回 `conn->step`，保证下一次 kernel 从正确游标继续。

即使某个 slice 的 `workSize == 0`，kernel 仍要完成对应 wait/post，以维持所有 rank、channel、proxy 的 step 对齐；“没有有效 payload”不等于“可以跳过同步”。

`sendPeerNotify` 直接推进 send step 并发布 tail；`recvPeerNotify` 推进 recv step、发布 head，然后等待 tail 达到目标，常用于不走完整 copy primitive 但仍需维持连接协议的路径。

### 3.3 `connFifo` 的字段与 `NCCL_MODE_OFFSET`

`ncclConnFifo` 定义在 `nccl/src/include/collectives.h`：

```cpp
struct ncclConnFifo {
  int mode;       // NORMAL / OFFSET / PTR
  int offset;     // shared FIFO 中的字节偏移
  ssize_t size;   // 当前 slot 的有效字节数；网络路径用 -1 表示空闲
  void* ptr;      // 预留的 pointer 描述字段
};
```

当前 reviewed device path 的关键分支是：

- `NCCL_MODE_NORMAL`：使用 `connEltsFifo + (step % NCCL_STEPS) * connStepSize`。
- `NCCL_MODE_OFFSET`：使用 `connEltsFifo + connFifo[slot].offset / sizeof(T)`，即 proxy 从 shared buffer allocator 选出的实际区域。
- `NCCL_MODE_PTR` 虽然在结构定义中存在，但当前这些 `Primitives` 分支没有按 `ptr` 字段取地址；不能把它当成当前通用地址选择路径。

NET transport 在 shared map 中通常将 mode 设为 `NCCL_MODE_OFFSET`，并在 proxy post 阶段填写 offset；非 shared buffer 则使用 normal slot。CollNet transport 的连接也把 FIFO mode 初始化为 offset。GPU 必须先读取 offset，再进行数据操作；offset 的可见性由 proxy 的 fence 保证。

### 3.4 LL/LL128：head/tail 之外还有 payload readiness

LL 和 LL128 仍维护单调 step，并在连接提供时使用 head credit，但 payload ready 的判断不同于 SIMPLE。LL device receive 直接等待 line flag，不执行普通 tail wait；LL128 仅在相应 `connFifo`/tail 路径存在时发布 send tail：

- LL 在 `ncclLLFifoLine` 中存 data 和 flag；发送侧按 step 计算 `NCCL_LL_FLAG(step+1)`，接收侧检查 line flag。清理 mask 用于 flag 循环时先清除旧 line，避免新旧轮次混淆。
- LL128 每条 128B line 使用 15 个数据元素和 1 个 flag 元素；flag thread 写入 line flag，proxy 在非 GDR 情况下也会检查所有 line flag。
- LL/LL128 的 `connFifo.size` 仍可能供 proxy 确认有效范围，但它不是 GPU kernel 判断每个 payload line ready 的唯一条件。

因此排查 LL/LL128 hang 时，要同时观察：head credit 是否到达、step 是否推进、当前路径是否使用 tail，以及 line flag 是否匹配；只看 SIMPLE 的 `connFifo.size` 不够。

### 3.5 proxy 如何消费 FIFO

以 NET send proxy 为例：

1. proxy 为 op 分配 `base` 和 `nsteps`，并维护 `posted/transmitted/done`。
2. 在 GPU 已获得/发布对应 step 后，proxy 读取 `connFifo[slot].size`；shared 模式还读取 `offset`。
3. SIMPLE 直接把 shared FIFO 或注册 user buffer 交给 network plugin；LL/LL128 先检查 line flag 或 GDR 条件。
4. network transfer 完成后，将 `size` 重置为 `-1`，再推进发送侧 head，允许 GPU 复用 slot。

SHM、P2P CE path 也使用同一类协议：先检查接收端 tail，再依据 size 做 CUDA memcpy，完成后推进对端可见的 tail。不同 transport 的内存位置不同，但“GPU 发布 → proxy 搬运 → proxy 清理/推进 credit”的逻辑相同。

## 4. Direct 与 registration

### 4.1 两者不是同一个概念

**Registration** 解决的是：buffer 是否能被另一 GPU、NIC 或 proxy 以稳定的映射/handle 访问，访问期间如何保持有效，以及如何按 local/graph registration 的引用与 cleanup 协议释放。

**Direct** 解决的是：在已有连接和注册能力的基础上，数据是否绕过中间 FIFO，采用“写入对端 buffer”（DirectWrite）或“从对端 buffer 拉取”（DirectRead）。

所以存在以下组合：

| registration | direct | 典型路径 |
|---|---|---|
| 无 | 无 | 普通 FIFO copy/reduce |
| IPC/NVLS | Write/Read | GPU 之间直接访问注册地址 |
| NET | proxy direct | NIC 直接访问注册 user buffer，GPU kernel 负责同步/描述 |
| 有 registration | 但不 direct | registration 只为 transport handle 或其他算法服务，kernel 仍走 FIFO |

### 4.2 registration 的类型和数据落点

`ncclTaskColl::regBufType` 是 bitmask：

| flag | 含义 | 进入 device work 的字段 |
|---|---|---|
| `NCCL_REGULAR_BUFFER` | 未注册，值为 0 | `regUsed=0`、`netRegUsed=0` |
| `NCCL_IPC_REG_BUFFER` | CUDA IPC/cuMem peer mapping | `sendbuffOffset/recvbuffOffset`、`sendbuffRmtAddrs/recvbuffRmtAddrs`，device `regUsed=1` |
| `NCCL_NVLS_REG_BUFFER` | NVLS registered downstream buffer | `ncclDevWorkCollReg.dnInputs/dnOutputs`，device `regUsed=1` |
| `NCCL_NET_REG_BUFFER` | NIC/CollNet registered buffer | `sendMhandle/recvMhandle`、按 channel 的 net handles，device/proxy `netRegUsed/reg=1` |

这些 bit 可以组合。例如 NVLS task 可能同时拥有 NVLS registration 和 CollNet network handles；普通 device work 用 `ncclDevWorkColl`，需要 NVLS downstream pointers 时使用 `ncclDevWorkCollReg`。

### 4.3 host registration 的选择过程

`ncclPrepareTasks` 完成 algorithm/protocol 后，调用 `ncclRegisterCollNvlsBuffers`；普通算法则在 `ncclTasksRegAndEnqueue` 中调用 `ncclRegisterCollBuffers`。大致顺序是：

1. 只有启用 local registration，或 persistent CUDA Graph 且启用 graph registration 时才尝试注册。
2. 根据 algorithm、protocol、collective、拓扑和 buffer 是否 in-place 判断 registration eligibility；例如部分 Ring/Tree 的 in-place 情形会主动跳过 IPC registration。
3. NVLS 尝试 local/graph NVLS registration；成功后可能调整 `nMaxChannels`，并把 down-neighbor pointers 写入 `dnInputs/dnOutputs`。
4. 对 direct P2P connector 收集 peer ranks，调用 `ncclIpcLocalRegisterBuffer` 或 `ncclIpcGraphRegisterBuffer`，产出本 rank offset 和对端地址数组。
5. 对支持 GDR 的 NET/CollNet connection 调用 local/graph net registration，产出每 channel 的 mhandle。
6. registration 失败或条件不满足时回退 regular FIFO；不能把“调用了 register 函数”理解为“最终一定使用 direct”。

P2P 的 `addP2pToPlan` 还会按连接 flags 和协议判断是否注册：通常只在允许 UB、payload 非空、SIMPLE、且连接支持 P2P read/write 或 direct NIC 时尝试。IPC registration 返回对端地址后，work 中的 `sendAddr/recvAddr` 会被替换为 registered address；NET registration 则保留 user buffer 和 mhandle，供 proxy 直接提交给 network plugin。

### 4.4 direct flag 的来源与含义

`device.h` 中的 flags 是：

| flag | 来源 | 含义 |
|---|---|---|
| `NCCL_P2P_WRITE` | P2P transport 或 `calcCollChunking` 的 CollNet Direct work | producer 直接写 consumer 的目标 buffer |
| `NCCL_P2P_READ` | P2P transport 或 collective direct pull 场景 | consumer/receiver 直接从 producer 的源 buffer 读取 |
| `NCCL_DIRECT_NIC` | NET setup 根据 GDR 能力设置 | NIC 可直接访问 GPU memory；它不是 GPU-GPU P2P read/write flag |

一个容易误读的点：`calcCollChunking()` 对 `NCCL_ALGO_COLLNET_DIRECT` 设置的是 `outDirectFlags = NCCL_P2P_WRITE`，这是写入 `ncclDevWorkColl::direct` 的 collective work flag；而 `NCCL_DIRECT_NIC` 是 connection flag，来源是 `ncclTopoCheckGdr()`。两者都叫 direct，但一个控制 kernel 的 peer buffer 方向，一个控制 NIC registration/GDR 能力。

### 4.5 `Primitives` 如何选择实际地址

`loadRecvConn/loadSendConn` 先把 `collWork->direct` 或 `p2pWork` 的 registration flags 和 connector flags 合成为 `DirectWrite`、`DirectRead`、`NetRegMode`。随后 `waitPeer()`/`process()` 按优先级选择地址：

1. **NET registered mode**：对支持 direct NIC 的路径，使用 user input/output 或由 proxy 管理的 registered buffer 语义；P2P 某些方向返回 `nullptr`，因为数据由 proxy/NIC 侧完成。
2. **FIFO offset mode**：`connFifo[slot].mode == NCCL_MODE_OFFSET` 时，使用 proxy 提供的 shared-buffer offset。
3. **DirectWrite**：发送侧把目的地址设为 `directBuff + dstIx + offset`；接收侧在需要把数据继续发送时也可能从自己的 output buffer 取地址。
4. **DirectRead**：接收侧把源地址设为 `directBuff + srcIx + offset`；纯发送侧可能是 empty send，因为 consumer 会直接 pull。
5. **普通 FIFO**：使用 `connEltsFifo + slot * connStepSize`。

这套优先级说明了为什么不能只看 `direct` 位：最终地址还取决于 `NetRegMode`、`NCCL_MODE_OFFSET`、P2P/collective、send/recv role 和当前 connector 的 `ptrExchange` 状态。

### 4.6 `ptrExchange`：direct pointer 的 producer/acceptor 握手

对 IPC direct path，`Primitives::setDataPtrs()` 将两端分成 provider 和 acceptor：

- DirectWrite：接收方/provider 发布自己的 output 或 registered remote address；发送方/acceptor 读取该指针，设置 `directBuff`，消费后把 slot 清零。
- DirectRead：发送方/provider 发布 input 或 output buffer 的地址；接收方/acceptor 读取该指针，设置 `directBuff`，直接从该地址拉取。

provider 在写 slot 前等待旧值被消费，acceptor 读取非空值后清零。这避免下一个 primitive 覆盖尚未消费的 pointer。跨 rank collective 的 IPC remote address 来自 `sendbuffRmtAddrs/recvbuffRmtAddrs` 与 offset；NVLS registration 没有普通 P2P slot 时，acceptor 直接使用 `work->dnInputs[index]` 或 `work->dnOutputs[index]`。

DirectRead 还有两个生命周期约束：

- P2P sender 在 primitive 析构时等待 receiver 真正读完 source，避免下一个 kernel 覆盖仍被对端读取的 buffer。
- NET registered send 在析构时等待对应 `connFifo.size` 恢复为空，确保 proxy/NIC 已经取走 user buffer 后才允许复用。

### 4.7 NET registration 与 proxy 的配合

NET registration 不是让 GPU kernel 完全退出通信，而是改变 proxy 的 buffer 来源：

1. host 为 task 保存 `sendMhandle/recvMhandle` 和每 channel 的 handles。
2. `calcCollChunking` 设置 `proxyOp.reg`、registered buffer 指针和 mhandle；对 Ring、CollNet Direct、NVLS、CollNet Chain 有不同的 loop/nbytes 特例。
3. proxy 仍然使用 head/tail/connFifo 做 step protocol，但发送时可以直接把 registered user buffer 交给 `ncclNet->isend()`，而不是先从临时 FIFO 复制。
4. network 完成后 proxy 清理 handle 对应 slot，并推进 head；plan 的 cleanup queue 负责 graph/local registration 的生命周期。

对 shared NET buffer，proxy 会在 `connFifo[offset]` 中写 allocator offset，GPU 通过 `NCCL_MODE_OFFSET` 定位；对非 shared buffer，则可能使用每 step 固定的 local buffer。GDR 支持不满足、`NET_DEVICE_UNPACK` 不兼容或 registration 条件失败时，必须回退到普通 proxy/FIFO 路径。

## 5. 三个典型执行例子

### 5.1 Ring + SIMPLE + 普通 buffer

`calcCollChunking` 选择 `Ring` 或 `RingTwice`，从 SIMPLE step 计算 chunk，按 grain 512B 对齐。planner 将 count 分到 lo/mid/hi channel，device primitive 在每个 slice：先用 head 等待 FIFO credit，写 size descriptor；shared offset 模式还要读取 proxy 已发布的 offset。随后写 SIMPLE FIFO，fence 后发布 tail；接收方等 tail，消费 FIFO，再发布 head。NET/SHM proxy 读取 size 并搬运，完成后清理 size。整个路径没有 directBuff，实际地址是 slot FIFO。

### 5.2 CollNet Direct + registered local buffers

`calcCollChunking` 选择 `CollnetDirect`，输出 `NCCL_P2P_WRITE`，并按 CollNet heads 设置 chunk/loop。local IPC registration 成功时，device work 可携带 `regUsed`、remote address 和 offset；NET/CollNet registration 则独立产生 mhandle，并令 proxy 直接使用注册 buffer。具体 connector 可能通过 `ptrExchange` 或 work 中的注册地址取得 direct buffer，不能从算法名反推出必然使用 IPC。无论地址来自哪种注册方式，head/tail 与 notify/step 仍负责顺序；direct 只改变数据路径，不会取消同步协议。

### 5.3 Ring + NET registration + GDR

NET setup 将连接标为 `NCCL_DIRECT_NIC`，registration 为每个 channel 获取 mhandle。proxy op 设置 `reg=1`，proxy 直接把 registered user buffer 提交给 network plugin。GPU 仍通过 conn head/tail 和 FIFO size 表示 slot readiness；在下一次 kernel 复用该 user buffer 前，primitive 析构会等待 proxy 清空 size。这里的 direct 是 NIC path，不等价于 GPU 端 `DirectWrite`。

## 6. 代码审阅得到的不变量与排障顺序

1. **先确认单位**：`nBytes`、traffic bytes、element count、grain、chunkSize、sliceSize 不能混用；尤其 AllReduce 的 traffic 乘数不能直接传给 `calcCollChunking`。
2. **再确认协议契约**：kernel 的 chunkGrain/count、proxy 的 chunkSize/loopSize/nsteps、transport 的 stepSize 必须匹配。
3. **检查 head/tail 方向**：发送方等 head、发布 tail；接收方等 tail、发布 head。若只看到一侧 counter 变化，通常是角色或 connector index 不匹配。
4. **检查 FIFO slot**：`step % NCCL_STEPS` 是否一致，SIMPLE 的 size 是否写入，proxy 完成后是否恢复 `-1`，shared path 的 offset 是否已发布。
5. **检查协议 payload readiness**：LL 看 line flags，LL128 看 flag line，不能只看 tail；SIMPLE 看 size/tail。
6. **检查 registration bit 与地址**：`regBufType` 是否最终传成 `regUsed/netRegUsed`，`direct` 是否真的设置，`ptrExchange` 是否完成 provider/acceptor 握手，NVLS 是否有 `dnInputs/dnOutputs`。
7. **检查生命周期**：registered handle、IPC mapping、proxy op、graph persistent cleanup 是否早于 GPU/proxy 完成；direct path 的 buffer 复用是否满足析构等待条件。

最常见的误判是把“使用了 registration”当成“必然 direct”，或者跨协议地把“tail 已推进”当成唯一的 payload-ready 条件。实际实现中，地址选择、FIFO 描述、协议 flag、proxy progress 和资源生命周期共同决定一次传输是否安全。

## 7. 代码导航

| 主题 | 主要位置 |
|---|---|
| task 分 channel、countLo/Mid/Hi | `nccl/src/enqueue.cc` 的 `scheduleCollTasksToPlan` |
| pattern/chunk/loop/proxy 参数 | `nccl/src/enqueue.cc` 的 `calcCollChunking` |
| grain 与 device work | `nccl/src/include/device.h` 的 `ncclProtoGrainSize`、`ncclCollCbdPart`、`ncclDevWorkColl` |
| connection 字段与 flags | `nccl/src/include/device.h` 的 `ncclConnInfo` |
| connection FIFO descriptor | `nccl/src/include/collectives.h` 的 `ncclConnFifo` |
| SIMPLE wait/post/direct 地址 | `nccl/src/device/prims_simple.h` 的 `waitPeer`、`postPeer`、`load*Conn`、`setDataPtrs` |
| LL/LL128 flag 与 FIFO | `nccl/src/device/prims_ll.h`、`nccl/src/device/prims_ll128.h` |
| P2P head/tail/ptrExchange | `nccl/src/transport/p2p.cc` |
| SHM/NET head/tail/FIFO | `nccl/src/transport/shm.cc`、`nccl/src/transport/net.cc` |
| registration 类型与 work 构造 | `nccl/src/register/coll_reg.cc`、`nccl/src/register/sendrecv_reg.cc`、`nccl/src/enqueue.cc` |
| proxy op 字段与 progress | `nccl/src/include/proxy.h`、`nccl/src/transport/net.cc`、`nccl/src/transport/coll_net.cc` |
