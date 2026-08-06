# NCCL 用户线程从 enqueue 到 GPU kernel 的机制分析

> 分析对象：本仓库 `nccl/` 当前代码，重点覆盖普通 kernel collective 和 P2P；CE、RMA、symmetric collective 作为旁路说明。本文中的“主线程”指调用 NCCL API 的用户线程，必要处会区分 NCCL 的 group launcher、host callback、proxy 线程和 GPU kernel。

## 1. 一句话概览

NCCL 不会在用户调用 `ncclAllReduce()` 时立即把一个完整 kernel 发射出去。用户线程先把 API 参数封装成 `ncclInfo`，再转换为 host-side task，放入 `comm->planner` 的队列。一次 group 结束时，NCCL 才统一完成算法/协议选择、任务聚合、通道切分、buffer 注册、proxy 操作生成和 kernel plan 构建。最后，plan 被编码为：

```text
ncclDevKernelArgs
  ├── ncclDevWorkBatch[]       # 每个 channel 的 batch 链入口
  └── work data                 # ncclDevWorkColl/P2p/Bcast/CollReg
```

GPU kernel 只接收 `ncclDevKernelArgs`，通过其中的 `comm` 指针访问预先建立的 device communicator/channel/connection 拓扑，再根据 batch 的 `funcId` 和 work 数据执行 collective 或 P2P。

整体调用链可以概括为：

```text
用户 API
  -> ncclInfo
  -> ncclEnqueueCheck
  -> taskAppend
  -> planner task queues
  -> ncclGroupEnd / implicit group end
  -> ncclPrepareTasks
  -> ncclTasksRegAndEnqueue
  -> ncclLaunchPrepare
       -> scheduleCollTasksToPlan / scheduleP2pTasksToPlan
       -> finishPlan
  -> uploadWork
  -> ncclLaunchKernel
  -> ncclKernelMain
       -> loadWorkBatchToShmem
       -> RunWorkBatch / ncclDevFuncTable
```

## 2. 用户线程如何 enqueue

### 2.1 API 参数先变成 `ncclInfo`

`nccl/src/collectives.cc` 中的 `ncclAllReduce`、`ncclAllGather`、`ncclBroadcast`、`ncclReduceScatter` 等 API，主要工作是填充一个栈上的 `ncclInfo`，包括：

- collective 类型 `coll` 和名字 `opName`；
- `sendbuff`、`recvbuff`、`count`、`datatype`、`op`、`root`；
- communicator、用户 CUDA stream；
- collective 默认的 `chunkSteps` 和 `sliceSteps`。

随后统一调用 `ncclEnqueueCheck(&info)`。因此，真正的任务组织入口不是各个 collective API，而是 `ncclEnqueueCheck()`（`nccl/src/enqueue.cc:2992`）。

### 2.2 `ncclEnqueueCheck` 的职责

用户线程在这里完成以下工作：

1. 检查 communicator 是否有效、是否 revoked，并等待 communicator 初始化完成。
2. 执行参数和全局一致性检查 `ArgsCheck(info)`。
3. 开始一个隐式 group：`ncclGroupStartInternal()`。显式 `ncclGroupStart/End` 时，外层 group 已经存在，因此这里只是增加嵌套深度。
4. 调用 DCCL 的 `dccl_probe_enqueue()` 保存 enqueue 时间和当前 collective 信息到 thread-local 状态。
5. 调用 `taskAppend(comm, info)` 将 API 参数转换为 planner task。
6. 调用 `ncclGroupEndInternal()`。如果这是最外层隐式 group，group end 会继续触发任务准备和 launch；如果用户显式包住多个操作，则任务仍留在 planner 中，直到用户调用 `ncclGroupEnd()`。

这解释了一个重要语义：

```cpp
ncclAllReduce(...);
```

看起来是一次调用，但普通多 rank 场景下，其 host-side task 会进入 planner，随后在隐式 group end 中完成 plan 和 kernel launch。显式 group 则允许多个 API 调用共同参与规划。

### 2.3 group 与 planner 的线程本地组织

`ncclGroupDepth`、`ncclGroupCommHead[]`、`ncclGroupCommPreconnectHead` 都是 thread-local（`nccl/src/group.cc:24-29`）。第一次某个 communicator 加入 collective group 时，`ncclGroupCommJoin()` 会：

- 把 communicator 插入当前线程的 group 链表；
- `ncclMemoryStackPush(&comm->memScoped)`，使本 group 创建的 work、batch、stream list 和 kernel 参数等 scoped 临时对象有统一生命周期；task/plan/proxy 结构体则由各自 memory pool 管理；
- 清空并初始化 `comm->planner`，但保留已分配的 `peers` 和 RMA queue 数组；
- 记录 blocking/nonblocking 模式，禁止同一个 group 混用两者。

planner 的核心字段位于 `nccl/src/include/comm.h:424-497`：

| 阶段 | 主要字段 | 作用 |
|---|---|---|
| 接收 API task | `collSorter`、`peers[]`、`rmaTaskQueues` | 按 collective、peer、RMA context 暂存任务 |
| 任务准备 | `collTaskQueue`、`collCeTaskQueue`、`collSymTaskQueue`、`collWorkQueue` | 选完算法后形成待切 plan 的列表 |
| 当前 plan | `wipPlan.channels[]` | 暂存每个 channel 的 batch 和 proxy op |
| 已完成 plan | `planQueue`、`unlaunchedPlansHead` | 保存待上传和待发射的 kernel plan |

planner 同时记录所有涉及的 user stream。`ncclPlannerSetCapturingGraph()` 会把新 stream 加到 `planner->streams`，并要求同一 group 的 stream 要么都不捕获，要么都属于同一个 CUDA graph；捕获状态决定后续 plan 是否为 persistent。

## 3. taskAppend 如何组织不同类型任务

### 3.1 普通 collective：`ncclTaskColl`

`collTaskAppend()` 为普通 collective 分配 `ncclTaskColl`（`comm.h:186-228`），并填入：

- 原始 buffer、count、root、datatype；
- `opHost` 与已经转换好的 `opDev`；
- `chunkSteps`、`sliceSteps`；
- 初始 `trafficBytes`；
- profiler 事件句柄。

AllGather 和 Broadcast 在 task 层会把 count 转换为字节数，并把 datatype 归一化为 `ncclInt8`，这样设备侧只需按字节搬运。task 此时还没有最终的 algorithm、protocol、channel 数和 device function id，这些由 group end 的规划阶段计算。

task 插入 `planner->collSorter`。该 sorter 使用多级 size bin，大致按 `trafficBytes` 降序排列（`comm.h:355-415`），目的是让大任务优先参与后续聚合和 channel 分配。

### 3.2 P2P：按 peer 分方向排队

`ncclSend/Recv` 以及 AlltoAll、Gather、Scatter 的 legacy fallback，都会转成 `ncclTaskP2p`（`comm.h:252-269`）。每个 task 根据方向进入：

```text
planner->peers[peer].sendQueue
planner->peers[peer].recvQueue
```

AlltoAll 会为每个 rank 生成 send/recv 对；Gather/Scatter 只在相应 root 端生成匹配的 P2P task。P2P task 保存 buffer、bytes、peer、datatype、`allowUB` 和 profiler 信息。

第一次看到某个 peer 的 send 或 recv 时，`p2pTaskAppend()` 还会根据 `p2pSchedule` 找到 round，设置 `connectSend/connectRecv` 位图，并把 communicator 放入 preconnect 链表。这样连接建立发生在真正切 plan 之前，而不是由 GPU kernel 临时决定。

### 3.3 Broadcast、CE、RMA 旁路

- Broadcast 在满足 AllGatherV 条件时先进入 `ncclTaskBcast` 的 peer queue；只有一个 broadcast peer 时，`ncclPrepareTasks()` 再将其转换成 `ncclTaskColl`。
- Copy Engine 路径将 task 放入 `collCeTaskQueue`，之后生成 `plan->ceCollArgs`，通过 `ncclLaunchCeColl()` 发射，不走普通 `ncclDevKernelArgs`。
- RMA 的 Put/Signal/Wait 进入按 context 划分的 `rmaTaskQueues`，由 `scheduleRmaTasksToPlan()` 生成 RMA plan，可能走 CE 或 proxy。
- symmetric collective 会从普通列表中分离到 `collSymTaskQueue`，由 symmetric scheduler 生成自己的 device 参数和 kernel。

## 4. group end：从 task 到可切分任务列表

`ncclGroupEndInternal()` 在 blocking group 中直接调用 `groupLaunch()`；nonblocking group 则创建异步 group job，用户线程返回 `ncclInProgress`，后续由 group job 执行同样的流程。

`groupLaunch()` 对 collective group 按 communicator clique 处理：

1. 运行 preconnect 和 runtime connection 任务。
2. 调用 `ncclPrepareTasks()` 计算算法、协议、channel/warp 参数，并准备注册 buffer 的 work 描述。
3. 调用 `ncclTasksRegAndEnqueue()` 补齐普通 collective 的 `ncclDevWorkColl` 或 `ncclDevWorkCollReg`。
4. 调用 `doLaunches()`，按 plan round 在各 sibling communicator 间协调发射。

### 4.1 `ncclPrepareTasks`：排序、聚合、选算法

该函数（`nccl/src/enqueue.cc:364-563`）的关键步骤是：

1. 处理只有一个 root peer 的 broadcast queue。
2. 从 `collSorter` 取出按 traffic 排序的 collective task。
3. 按 `(func, opDev.op, datatype)` 分桶。
4. 同一桶内把相邻且 traffic 在 4 倍范围内的任务组成临时 `agg`，用于算法调优时估计 pipeline 数；原 task 并不会被合并成一个用户语义操作。
5. `ncclGetAlgoInfo()` 基于 topology cost table、tuner plugin、环境变量和 buffer registration 情况选出：

```text
algorithm       Ring / Tree / CollNet / NVLS / PAT ...
protocol        LL / LL128 / Simple
nMaxChannels    该 task 允许使用的最大 channel
nWarps          每个 work 所需的 warp 数
devFuncId       根据 func/op/type/algo/proto 映射出的设备函数编号
```

算法选择在 `ncclGetAlgoInfo()` 和 `topoGetAlgoInfo()`（`enqueue.cc:1946-2099`）中完成。先由 topology/tuner 计算各 algorithm-protocol 组合的时间，再选择最小值；随后根据消息大小、线程阈值、算法特性调整 channel 数和线程数。`calcCollChunking()` 再把 algorithm/protocol 翻译为通信 pattern、chunk/slice、proxy op 参数。

6. 按 `[isCollnet][isNvls]` 将 task 分成四类，最终按 CollNet 优先的顺序放入 `planner->collTaskQueue`。
7. 对 runtime connection 和 NVLS 注册 buffer，生成 `ncclWorkList`，其 payload 是后续要复制给 GPU 的 device work 结构。

### 4.2 buffer registration 与 connection 信息

buffer 注册结果并不是只存在 host task 中。`ncclRegisterCollBuffers()` / `ncclRegisterCollNvlsBuffers()` 将注册后的地址、偏移、IPC/NET/NVLS handle 保存到 task 或 cleanup queue；`ncclTasksRegAndEnqueue()` 把这些内容编码进 `ncclDevWorkColl` 或 `ncclDevWorkCollReg`。

连接信息则在 communicator 初始化或 runtime preconnect 时进入 device memory。host 侧的 `ncclConnector` 包含 transport/proxy 状态，GPU 只需要其精简版 `ncclConnInfo`（`device.h:130-148`），包括协议 buffer、head/tail、step size、direct 指针、proxy FIFO 和 network device handle。

## 5. `ncclLaunchPrepare`：按预算生成 kernel plan

`ncclLaunchPrepare()`（`enqueue.cc:1509-1664`）循环生成一个或多个 `ncclKernelPlan`，直到所有 task 都被消费。每个 plan 开始时清空 `planner->wipPlan`，并设置：

- `persistent`：CUDA graph capture 时为 true；
- 初始 `workStorageType`：普通 plan 为 FIFO，persistent plan 为 Persistent；
- budget：kernel argument 中可用的字节数，以及外部 FIFO 可用的字节数。

plan 的关键字段在 `comm.h:301-348`：

| 字段 | 含义 |
|---|---|
| `kernelFn` / `kernelSpecialized` | 本 plan 的 CUDA kernel 入口及是否为特化 kernel |
| `kernelArgs` / `kernelArgsSize` | host 端构造的 kernel 参数区域 |
| `channelMask` | plan 覆盖的 channel 集合 |
| `threadPerBlock` | plan 中 work 所需的最大 block 线程数 |
| `workQueue` / `workBytes` | channel 共享的 host work payload |
| `nWorkBatches` | batch 描述数量 |
| `collTaskQueue`、`p2pTaskQueue` | plan 持有的 task，由 plan reclaimer 在提交后统一回收 |
| `proxyOpQueue` | 与 GPU work 对应的 proxy 操作 |
| `persistent` | plan 是否被 CUDA graph 捕获 |

规划顺序非常重要：

```text
RMA
  -> CE collective
  -> symmetric collective
  -> ordinary collective
  -> broadcast
  -> P2P
```

普通 plan 中 collective 必须在 P2P 前排空。源码注释指出，这能保证各 rank 在相同位置切断 plan，避免 channel picker 因 task 数不同而产生分歧。

### 5.1 collective 如何填充 plan

`scheduleCollTasksToPlan()` 先按 `[collnet][nvls]` 估计 traffic 和可用 channel，再逐个 task 消费：

- CollNet/NVLS 任务通常把均匀 work 分到 `[channelLo, channelHi]`；
- 普通 Ring/Tree/PAT 任务使用 continuous-byte-distribution，把数据切成 `countLo/countMid/countHi` 三段；边界 channel 和中间 channel 可以拥有不同数据量；
- 为每个 channel 计算 `chunkGrainsLo/Mid/Hi` 或 `collnet.chunkCount`；
- 生成与 channel 对应的 `ncclProxyOp`，设置 `opCount`、buffer、bytes、protocol、algorithm、chunk/slice 和 connection；
- 调用 `ncclAddWorkBatchToPlan()` 记录 work batch；
- 更新 `plan->channelMask`、`plan->threadPerBlock`、`plan->kernelFn`、`plan->collOpCount`；
- 将 task 和 workNode 从 planner 队列移动到 plan 队列。

最终普通 collective 的 device payload 主要是 `ncclDevWorkColl`（`device.h:280-309`）：

- channel 范围和 warp 数；
- send/recv buffer 及对称 buffer offset；
- reduction op 参数；
- root、direct/reg/profiler 标志；
- 每 channel 的 count 和 chunk grain。

注册 buffer 的 NVLS 路径使用更大的 `ncclDevWorkCollReg`，在 `coll` 外增加 `dnInputs/dnOutputs/upOutputs` 指针数组。

### 5.2 P2P 如何填充 plan

`scheduleP2pTasksToPlan()` 按 `comm->p2pSchedule` 的 round 遍历 peer，匹配同一轮的 send/recv。`addP2pToPlan()` 会：

1. 检查各 channel 的连接，判断每个方向能否使用 LL、是否走 network、是否注册；
2. 根据 bytes、chunk size 和 min/max channel 计算每个方向的 channel 数；
3. 计算该 part 的 `sendAddr/recvAddr`、bytes、rank、chunk size 和 registration handle；
4. 写入 `ncclDevWorkP2p`；
5. 生成 send/recv proxy op；
6. 用 `p2pEpoch` 和 `p2pRound` 限制 batch 合并，保证不同 epoch 不会落到同一个 device batch。

`ncclDevWorkP2p`（`device.h:237-254`）携带 send/recv 地址、字节数、peer rank、channel base、方向 channel 数、chunk size 编码以及 LL/NET/IPC registration 标志。P2P plan 使用 P2P 专用 `devFuncId`，block 通常至少为 `NCCL_MAX_NTHREADS`。

### 5.3 batch 组织：`ncclDevWorkBatch`

`ncclAddWorkBatchToPlan()` 不会为每一个 work 单独产生一个 batch。它尝试把相邻 work 合并，只有以下情况会新建 batch：

- `workType` 不同；
- `devFuncId` 不同；
- P2P epoch/round 冲突；
- batch work 数或字节数超限；
- work offset 无法在当前 batch 的 64-bit `offsetBitset` 表示；
- P2P 同一 batch 达到最多 8 个操作。

每个 `ncclDevWorkBatch`（`device.h:387-405`）包含：

- `workType`：P2P、普通 collective、注册 collective 或 broadcast；
- `funcId`：设备端 dispatch 的函数编号；
- `nextExtends`：下一 batch 是否要和当前 batch 融合；
- `nextJump`：当前 channel 的下一 batch 在全局 batch 数组中的跳跃距离；
- `offsetBase`：work payload 的起始偏移；
- `offsetBitset`：当前 batch 中实际存在的 work 槽位。

这种编码允许多个 channel 的 batch 交错存放，同时让 GPU 以 `blockIdx.x` 直接取得每个 channel 的第一 batch。

## 6. `finishPlan`：形成 kernel 参数内存布局

`finishPlan()`（`enqueue.cc:203-277`）是 host plan 到 kernel 参数的关键边界。

### 6.1 选择 work storage

如果：

```text
sizeof(ncclDevKernelArgs) + batchBytes + workBytes <= comm->workArgsBytes
```

则整个 batch 和 work payload 都放在 kernel 参数区，`workStorageType = Args`。否则：

- 非 graph plan 使用 communicator 的循环 work FIFO，`workStorageType = Fifo`；
- CUDA graph plan 使用独立的 persistent device buffer，`workStorageType = Persistent`。

`comm->workArgsBytes` 在 `devCommSetup()` 中被限制为 `ncclMaxKernelArgsSize()`，当前实现为 4 KiB；因此大多数较大 plan 会把 payload 放进 FIFO，而不是参数区。

### 6.2 `ncclDevKernelArgs` 的布局

host 端分配一段 16-byte 对齐的 `kernelArgs`：

```text
offset 0:  ncclDevKernelArgs
           - comm            -> device-side ncclKernelComm
           - channelMask
           - workStorageType
           - workMask
           - workBuf         -> FIFO 或 persistent device buffer

随后:      ncclDevWorkBatch batchZero[nWorkBatches]
```

`finishPlan()` 将每个 channel 的 batch queue 以 round-robin 方式写入 `batchZero[]`。每个 channel 的第一 batch 必须在 `batchZero[blockIdx.x]`，所以 kernel 的 block id 不直接等于 channel id，而是等于 `channelMask` 中第 N 个 set bit 的序号。

同时，`finishPlan()` 把各 channel 的 proxy op queue 按 `opCount` 做归并，collective 使用低位 tag 0，P2P 使用低位 tag 1；这保证 proxy 线程看到的顺序与 GPU work 顺序一致。

## 7. kernel launch 前后：host 与 proxy/FIFO 的衔接

### 7.1 `uploadWork`

`ncclLaunchKernelBefore_NoUncapturedCuda()` 先调用 `uploadWork()`：

- Args：work 已经在 `kernelArgs` 后面，`workBuf=null`；
- FIFO：等待 `workFifoConsumed` 释放足够空间，将 work 写入 host/CUDA-host/GDR FIFO，并设置 `kernelArgs->workBuf=workFifoBufDev`；
- Persistent：分配 device buffer，异步 memcpy work payload，并把 device 指针写入 `kernelArgs->workBuf`。

batch 的 `offsetBase` 会根据 storage 类型加上 FIFO cursor；随后按 16-byte 单元把 `plan->workQueue` 的 payload 复制到目标 storage。FIFO 写完后更新 `comm->workFifoProduced`，GPU kernel 通过 batch offset 读取 work。

### 7.2 proxy op 的时序

GPU work 和 network/proxy 操作是两个相互配合的队列：

- `plan->proxyOpQueue` 保存每个 channel 的 proxy 描述；
- `ncclLaunchPrepare()` 如果 plan 有 proxy op，可能在 host stream 上插入 `cudaLaunchHostFunc`；
- host callback 或 `ncclLaunchKernelAfter_NoCuda()` 调用 `hostStreamPlanTask()`；
- `uploadProxyOps()` 把 plan 内局部 `opCount` 翻译成 communicator shared-resource 上的单调计数，再调用 `ncclProxySaveOp()` 和 `ncclProxyStart()`。

因此，proxy 不是 kernel 参数中一个简单的函数指针；它是由 host plan 产生、按 stream/event 与 kernel 建立先后关系的独立控制面。

### 7.3 `ncclLocalOpAppend`：从 plan 描述符到共享 Proxy 队列

`ncclLocalOpAppend(comm, proxyConn, proxyOp)` 是 proxy 提交路径的生产者端核心。它不执行网络传输，也不直接创建 transport progress 使用的 `ncclProxyArgs`；它把已经选定具体 connector 的 `ncclProxyOp` 复制到目标 proxy progress 线程拥有的共享内存池，先组成生产者私有的待发布链，并在安全边界将链发布给消费者。完整的数据流是：

```text
plan->proxyOpQueue
  -> uploadProxyOps：把 plan-local opCount 转成 shared-resource 历史编号
  -> ncclProxySaveOp：按 pattern 展开具体 send/recv peer
  -> SaveProxy：选择 connector
  -> ncclLocalOpAppend：复制到共享池并组成待发布链
  -> ncclProxyStart/ncclProxyPost：发布给目标 proxy progress thread
  -> ncclProxyGetPostedOps -> ProxyAppend：转换、聚合为 ncclProxyArgs/subs[]
  -> transportComm->proxyProgress：推进实际传输
```

#### 7.3.1 目标 Proxy 与槽位所有者是两个维度

函数开头的两个 local rank 不能混为一谈：

| 表达式 | 作用 |
|---|---|
| `proxyConn->tpLocalRank` | 选择哪个 top-parent local rank 的 proxy 负责该 connector，即选择目标共享池 |
| `comm->topParentLocalRanks[comm->localRank]` | 标识当前提交者在目标池中的生产者分区，用于领取和归还槽位 |

`proxyOps += proxyConn->tpLocalRank` 先定位目标 proxy 的本地映射；该 pool 的 `ops[]` 再按生产者划分为 `tpLocalnRanks` 个连续区域，每区 `MAX_OPS_PER_PEER` 个元素。消费端可用 `opIndex / MAX_OPS_PER_PEER` 恢复槽位所属生产者，并归还到相应的 `pool->freeOps[peer]`。这使不同 local rank 通常不争用同一条空闲链；发布到目标 proxy 的全局工作链时才使用共享 mutex。

pool 在目标 proxy 的 `proxyProgressInit()` 中通过共享内存创建并初始化，其他 local rank 在 connector 初始化后映射同一对象。`ncclProxyConnector::connection` 中保存的是目标 proxy 返回的 connection 标识；生产者只复制它，真正解引用发生在拥有该 connection 的 proxy 侧。

#### 7.3.2 两级 free-list 与固定容量背压

先区分两个名字相近、但所有权不同的链头：

| 链头 | 可见范围 | 含义 |
|---|---|---|
| `proxyOps->freeOp` | 当前生产者的本地状态 | 已由该生产者独占的私人空闲库存 |
| `pool->freeOps[tpLocalRank]` | 生产者和目标 proxy 共享 | 初始化时提供槽位、运行中接收 proxy 归还槽位的共享交接入口；这里 `tpLocalRank` 是当前来源 rank，而不是目标 proxy rank |

`freeOps[]` 的一个元素只是整数下标，不是槽位数组本身。例如 `MAX_OPS_PER_PEER=8`、来源 local rank 为 1 时，该生产者的分区是 `ops[8..15]`，初始化结果为：

```text
pool->freeOps[1]
        |
        v
      ops[8] -> ops[9] -> ... -> ops[15] -> -1
```

这里 `pool->freeOps[1] == 8`，后续节点由每个 `ncclProxyOp::next` 串起来。提交侧首次映射该 pool 时，本地的 `proxyOps->freeOp` 初始化为 `-1`；所以第一次 append 通常没有私人库存，但共享交接入口持有初始化好的完整分区。

快路径从私人库存取链首：

```cpp
opIndex = proxyOps->freeOp;
op = pool->ops + opIndex;
proxyOps->freeOp = op->next;
```

假设本地链是 `9 -> 10 -> 11 -> -1`，本次使用 `ops[9]` 后，只需把本地链头改为 10。整个过程不接触共享链头，也不需要原子操作。

本地链为空时，生产者才执行：

```cpp
freeOp = COMPILER_ATOMIC_EXCHANGE(
    &pool->freeOps[tpLocalRank], -1,
    std::memory_order_acquire);
```

这次原子操作只交换一个链首下标：不可分割地返回共享入口的旧值，同时把共享入口写成 `-1`。但旧下标能够沿 `next` 到达所有后续节点，所以它在所有权意义上转移的是整条链，而不是只转移一个槽位。例如交换前是 `8 -> 9 -> 10 -> 11`，交换返回 8 后：

```text
共享交接入口：pool->freeOps[1] = -1
本次使用槽位：ops[8]
生产者私人链：proxyOps->freeOp -> ops[9] -> ops[10] -> ops[11] -> -1
```

因此一次 atomic exchange 可以摊销后续多次 append：首节点立即使用，其余节点放入 `proxyOps->freeOp`，以后继续走无原子的本地快路径。

当 progress 线程执行 `ncclProxyGetPostedOps()` 时，它已经把描述字段转移到 `ncclProxyArgs/subs[]`，原共享槽位不必等网络完成即可回收。线程先把本批槽位重新串成链，再用 release CAS 将整链前插到对应来源 rank 的 `pool->freeOps[i]`。生产者的 acquire exchange 与这个 release CAS 配对，保证生产者看到新链首时，也能看到消费者此前写好的所有 `next` 链接。

这里不能用普通的“先 load、再 store -1”代替 exchange。否则可能发生：生产者读到旧链 A；proxy 在两条普通操作之间把回收链 B 前插成 `B -> A`；生产者随后写入 `-1`，导致 B 永久丢失。atomic exchange 与消费端 CAS 只会形成两种安全顺序：

- exchange 先发生：生产者取得 A，proxy 随后把 B 放进已经清空的共享入口；
- CAS 先发生：proxy 先形成 `B -> A`，生产者随后一次取得 `B -> A`。

两种情况下都不会让同一槽位同时归两方所有，也不会丢失回收链。可以把槽位的状态迁移概括为：

```text
共享空闲链
  -> 生产者私人空闲链
  -> pending/posted ncclProxyOp 链
  -> progress 线程转换为 ncclProxyArgs
  -> 共享空闲链
```

如果共享入口仍为 `-1`，函数会 `sched_yield()` 后继续等待。这是固定池提供的背压：热路径不做动态分配，但 pool 耗尽时提交线程可能等待 proxy 消费和回收。该循环本身不检查 `abortFlag`，因此正常退出依赖 progress 线程持续运行；排查卡在此处的问题时，应同时检查 proxy thread 的 `asyncResult`、stop/abort 状态和已发布队列。

`MAX_OPS_PER_PEER` 按“两轮、所有 channel、每个 P2P work 最多一收一发”的最坏情况估算。这里的 `peer` 是共享池分区的生产者编号，不等同于本次网络通信的 `proxyOp.peer`。

#### 7.3.3 复制后的字段修正与所有权转移

取得槽位后，函数先 `memcpy` 整个 `ncclProxyOp`，再做三项修正：

1. 若存在 `ringAlgo`，增加引用计数。plan 中原描述符可由 `reclaimPlan()` 回收，而共享副本和后续 `ncclProxySubArgs` 必须保持算法对象存活；NET progress 完成相应 sub 后再减少引用。
2. 把 `op->next` 重置为 `-1`。同一字段在空闲态连接 free-list，在提交态连接 pending/posted op 链，不能沿用被复制对象的链指针。
3. 用 `proxyConn->connection` 覆盖 `op->connection`。一个 pattern 可能把同一模板展开成多个 send/recv peer，只有 `SaveProxy` 当前选中的 connector 才是该共享副本的实际执行连接。

随后，新槽位追加到该生产者针对目标 proxy 的 `nextOps..nextOpsEnd` 待发布链。此时工作尚未出现在共享的 `pool->nextOps` 中，progress 线程也看不到它。正常路径等 `uploadProxyOps()` 完成本 plan 的全部展开后，由 `ncclProxyStart()` 遍历目标 proxy，并调用带 mutex 的 `ncclProxyPost()` 一次性发布各链；共享队列从空变为非空时才通知 condition variable，避免每个 sub-op 都加锁和唤醒。

#### 7.3.4 池满时必须保留链尾的完整 `opCount`

每追加一个槽位，`proxyOps->count` 加一。达到 `MAX_OPS_PER_PEER` 时，函数必须提前发布一部分条目来制造可回收空间，但不能直接发布整条链。原因是同一逻辑 collective 或 P2P batch 展开的多个 channel/peer 条目共享 `opCount`，progress 侧可能把共享 connection lane 上相同 `opCount` 的条目聚合到一个 `ncclProxyArgs.subs[]`。如果链尾分组被拆成两次发布，第一批可能已经从 `Ready` 进入运行态，第二批再追加就会触发 `Proxy append on running operation`，或者破坏该逻辑操作的 sub 聚合。

因此，溢出处理读取尾节点的 `lastOpCount`，扫描到最后一个 `opCount != lastOpCount` 的节点，只发布截至该节点的完整前缀，把所有尾部同编号条目留待后续 append 或 `ncclProxyStart()`：

```text
pending: [20, 20] [22, 22] [24, 24]
post:    [20, 20] [22, 22]
retain:                      [24, 24]
```

如果满池条目全部具有同一个 `opCount`，就不存在安全切分点，函数告警并返回 `ncclInternalError`。这表示单个逻辑分组超过了池容量所依据的上游规模不变量，而不是可以任意拆批继续执行的普通背压。

#### 7.3.5 消费、二级聚合与槽位回收

`ncclProxyPost()` 在 `pool->mutex` 下把生产者链拼接到共享 `nextOps..nextOpsEnd`。`ncclProxyGetPostedOps()` 在同一把锁下取走已发布链，然后分批执行 `ProxyAppend()`；它的 batch 限制按生产者和 `opCount` 边界计数，避免在一个逻辑分组中间停下。

`ProxyAppend()` 才是第二级、执行态聚合：以 `connection->proxyAppendPtr` 定位 append lane；仅当 connection 标记为 shared 且当前 args 的 `opCount` 相同时，才把新描述符转换成另一个 `ncclProxySubArgs` 并追加到同一个 `ncclProxyArgs`。转换还会校验 `sliceSteps`、`chunkSteps`、protocol、datatype、reduction op 和 collective 类型一致，且 args 必须仍处于 `ncclProxyOpReady`。不满足聚合条件时，新建 args 并通过 `nextPeer` 保持该 lane 的执行顺序。

共享池里的 `ncclProxyOp` 只承担跨提交边界的描述符传递：转换完成后，槽位立即按原生产者分区组成回收链并通过 atomic CAS 放回 `freeOps[]`；实际异步状态已经转移到 `ncclProxyArgs/subs[]`，直到 transport progress 完成后由 `removeOp()` 回收。由此要区分三个生命周期：plan 描述符由 `reclaimPlan()` 回收，共享 pool 槽位在 append 成 args 后即可复用，运行中的 transport 状态则独立持续到 proxy completion。更完整的完成与回收关系见 `codex_nccl-m18-completion-reclaim-m21-profiling-observability.md`。

#### 7.3.6 `NCCL_PROXY_CPUSET`：为 Proxy 线程指定 CPU 集合

当前实现用 `ncclStrListToCpuset()` 把 `NCCL_PROXY_CPUSET` 解析成逗号分隔的逻辑 CPU 编号列表。例如，在 CPU 8、9 都在线且属于作业允许集合时，可以在程序启动前设置：

```bash
export NCCL_PROXY_CPUSET=8,9
```

不要写成 `8-9` 或 CPU bitmask：解析器不展开范围，`0x10` 会按 base 0 被解释为 CPU 编号 16，而不是表示 CPU 4 的掩码；也应避免 `08` 之类带前导零的编号。当前解析器没有检查每个 token 是否完整转换，因此非法后缀可能被静默忽略，最安全的格式是无前导零的纯十进制编号，以逗号分隔。

该变量通过进程级 `std::call_once` 只解析一次，因此应在 NCCL 初始化、尤其是第一个 Proxy Service 线程启动前设置。Proxy Service 和 Proxy Service UDS 线程都会调用 `sched_setaffinity()` 应用同一集合；由 Service 创建的 Proxy Progress 线程继承其 affinity。它不是按 GPU 一一分配 CPU：`8,9` 表示相关 proxy 线程都允许在 CPU 8 或 9 上调度。

指定 CPU 还必须满足容器、cgroup 或作业调度器的限制。可先用 `grep Cpus_allowed_list /proc/self/status` 检查，并用以下日志确认 NCCL 读取的旧、新集合；需要精确验证时，对 proxy 线程 TID 执行 `taskset -cp <TID>`：

```bash
NCCL_PROXY_CPUSET=8,9 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=ENV,INIT \
./your_nccl_program
```

##### torchrun 两机八卡：按 `LOCAL_RANK` 为每个 rank 分配一个核

`torchrun --nproc-per-node=8` 的典型 GPU 用法是每节点启动 8 个 worker 进程，每个进程使用一张 GPU；每台机器上的 worker 都获得取值为 0 到 7 的 `LOCAL_RANK`。由于 `NCCL_PROXY_CPUSET` 是进程级环境变量，可以让每个 worker 在初始化 NCCL 前按 `LOCAL_RANK` 选择不同 CPU。两台机器可以重复使用同一组 CPU 编号，因为它们属于不同主机：例如节点 0 的全局 rank 0 和节点 1 的全局 rank 8 都是 local rank 0，都可以绑定各自机器上的 CPU 32。

不能在 `torchrun` 父进程中统一设置 `NCCL_PROXY_CPUSET=32,33,...` 来表达映射；这样 8 个 worker 会继承同一个 CPU 集合，每个 rank 都可以在所有这些核上调度。应使用 worker 包装脚本，根据子进程才具有的 `LOCAL_RANK` 写入单个 CPU。假设 GPU 0–3 使用 CPU 32–35，GPU 4–7 使用 CPU 96–99，可创建：

```bash
#!/usr/bin/env bash
set -euo pipefail

proxy_cpus=(32 33 34 35 96 97 98 99)
local_rank="${LOCAL_RANK:?LOCAL_RANK is not set by torchrun}"

if (( local_rank < 0 || local_rank >= ${#proxy_cpus[@]} )); then
  echo "Invalid LOCAL_RANK=${local_rank}" >&2
  exit 1
fi

export NCCL_PROXY_CPUSET="${proxy_cpus[$local_rank]}"
echo "host=$(hostname) rank=${RANK} local_rank=${LOCAL_RANK} proxy_cpu=${NCCL_PROXY_CPUSET}"

exec python -u train.py "$@"
```

保存为 `run_worker.sh` 并执行 `chmod +x run_worker.sh`。因为训练入口现在是 shell 脚本，两个节点都通过 `--no-python` 启动；只有 `--node-rank` 不同：

```bash
# 节点 0
torchrun --nnodes=2 --nproc-per-node=8 \
  --node-rank=0 --master-addr=192.168.1.10 --master-port=29500 \
  --no-python ./run_worker.sh --your-training-args

# 节点 1
torchrun --nnodes=2 --nproc-per-node=8 \
  --node-rank=1 --master-addr=192.168.1.10 --master-port=29500 \
  --no-python ./run_worker.sh --your-training-args
```

也可以在 `train.py` 最开始读取 `LOCAL_RANK` 并设置 `os.environ["NCCL_PROXY_CPUSET"]`，但必须早于 `dist.init_process_group(backend="nccl")` 和任何其他可能创建 NCCL communicator 的调用。使用 `LOCAL_RANK` 而不是全局 `RANK`，可直接复用每节点映射，也避免依赖 elastic restart 后可能变化的全局 rank。

CPU 表不能机械地按编号连续填写，应先用 `nvidia-smi topo -m`、`numactl -H` 和 `lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE` 选择靠近相应 GPU/NIC 的不同物理核，并确认它们位于容器、cgroup 或调度器允许的集合内；两台机器拓扑不同时，可在脚本中按 hostname 使用不同数组。若设置了 `CUDA_VISIBLE_DEVICES`，映射还应依据其重排后的设备顺序理解 `LOCAL_RANK -> GPU`。

最后，一 rank 一核的实际效果是：该进程中的 Proxy Service、UDS Service，以及继承 affinity 的 Proxy Progress 线程都被限制在这个核上，而不是只绑定某一条 progress 线程。典型的一进程一卡、单个主要 NCCL process group 场景符合预期；如果同一 rank 创建多个 communicator，它们的相关 proxy 线程仍共享该进程的一个 CPU 集合，`NCCL_PROXY_CPUSET` 不能进一步按 communicator 分核。

### 7.4 多 communicator 和多 stream 的 launch 协调

`doLaunches()` 以 intra-process clique 为单位调用 `ncclLaunchPrepare()`，再以 round 方式让各 communicator 轮流发射 plan。`NCCL_LAUNCH_MODE=GROUP` 时使用 intra-process barrier，保证 sibling communicator 在相同 round 前进。

`ncclLaunchPrepare()` 选择 `planner->streams` 链表头作为 launch stream。新发现的 stream 是头插的，因此它通常是最近加入的唯一 stream，不能称为“第一个用户 stream”。launch stream 等待其他 user stream 和 device stream；`ncclLaunchFinish()` 再记录完成 event，让其他 user stream 等待 launch stream。这样多个用户 stream 的任务被统一纳入同一个 plan/launch 顺序，同时支持 CUDA graph capture。

## 8. device communicator：GPU kernel 预先拥有的静态信息

真正的 kernel 参数只带一个 `ncclKernelComm*`，该指针在 communicator 初始化阶段由 `devCommSetup()` 建立，而不是每个 collective 临时拼装。

`devCommSetup()`（`nccl/src/init.cc:549-649`）在 device memory 分配 `ncclKernelCommAndChannels`，并复制：

### 8.1 `ncclKernelComm`

`device.h:433-454` 定义的字段包括：

- rank、nRanks、node、nNodes；
- 各 protocol 的 buffer size 和 P2P chunk size；
- `abortFlag`；
- `channels` 指向 device channel 数组；
- `rankToLocalRank`、CollNet dense rank 映射；
- profiler 的 workStarted/workCompleted 计数数组。

### 8.2 `ncclDevChannel`

每个 channel 的 device 镜像（`device.h:414-423`）包括：

- `peers`：指向 `ncclDevChannelPeer` 的指针数组；
- ring、tree、CollNet chain/direct、NVLS 拓扑；
- `workFifoDone` 和 workCounter。

`initChannel()` 创建 host/device peer 指针数组；transport connect 完成后，将 host connector 中的 `ncclConnInfo` 拷贝进 device peer 的 send/recv connector。于是 GPU kernel 运行时可以直接读取协议 buffer、head/tail、connection FIFO 和 transport-specific device handle。

### 8.3 channel topology

`ncclChannel`/`ncclDevChannel` 中的 ring/tree/direct/NVLS 结构决定设备端如何找 next/up/down/head peer。算法选择只决定使用哪类 topology 和 work 解释方式；具体 peer connection 和 rank order 已经在 communicator 初始化/连接阶段准备好。

## 9. GPU kernel 如何消费 plan

### 9.1 launch 配置

`ncclLaunchKernel()`（`enqueue.cc:1683-1785`）计算：

```text
grid.x  = popcount(plan->channelMask)
block.x = plan->threadPerBlock
smem    = ncclShmemDynamicSize(cudaArch)
fn      = plan->kernelFn
args    = plan->kernelArgs
```

随后通过 CUDA driver API 以 `CU_LAUNCH_PARAM_BUFFER_POINTER` 和 `CU_LAUNCH_PARAM_BUFFER_SIZE` 传入 `kernelArgs`。在支持的 CUDA/架构上，还会设置 cluster dimension、memory sync domain、launch completion event、programmatic serialization 等属性。

`kernelFn` 与 `devFuncId` 是两层 dispatch：

- `kernelFn` 是 kernel 外壳，通常来自 `ncclDevKernelForFunc[devFuncId]`；
- 每个 batch 内的 `funcId` 决定具体 `ncclDevFuncTable[funcId]`，因此一个 kernel 可以按 batch 执行不同 work。

`device/generate.py` 生成 host/device table。`ncclDevFuncId()` 用 collective、device reduction op、datatype、algorithm、protocol 计算 row，再映射到 primary function id。生成器还会把可复用的组合映射到特化 kernel，以减少 device function call 开销。

### 9.2 `ncclKernelMain`

`nccl/src/device/common.h:341-410` 是普通 NCCL kernel 的核心入口：

1. 前若干线程把参数区的 `ncclDevKernelArgs` 复制到 shared memory，避免每个线程把 4 KiB 参数区 spill 到 local memory。
2. 根据 `args->channelMask` 计算当前 `blockIdx.x` 对应的真实 channel id。
3. 第一 warp 把 `ncclKernelComm` 拷进 shared memory，第二 warp 把当前 `ncclDevChannel` 拷进 shared memory。
4. 其余线程调用 `loadWorkBatchToShmem(args, batchIx=blockIdx.x)`。
5. `loadWorkBatchToShmem()` 读取 `batchZero[batchIx]`，利用 `offsetBitset` 找出本 batch 中的 work，按 `workType` 确定结构体大小，并从 Args/FIFO/Persistent storage 中按 16-byte 单元加载到 shared memory 的连续 work storage。
6. batch 的 `workType`、`nWorks`、`funcId`、`nextBatchIx` 被写入 shared state。
7. kernel 执行特化的 `RunWorkBatch`，或者调用 `ncclDevFuncTable[funcId]()`。
8. 如果 `nextBatchIx != -1`，继续加载同一 channel 的下一批；否则该 block/channel 完成。

普通 collective 的 `RunWorkBatch` 按 `nWorks` 遍历 `ncclDevWorkColl`，再按 work 的 `nWarps` 让对应线程参与 `RunWorkColl<Fn,T,RedOp,Algo,Proto>()`。底层 primitives 从 shared 的 `ncclShmem.comm`、`ncclShmem.channel` 和 work 结构中得到 peer connection、buffer、chunk 和 reduction 参数，最终执行 Ring/Tree/NVLS 等算法。

## 10. 生命周期、同步和回收

一个 plan 的生命周期是：

```text
memory pool allocate
  -> planner queue
  -> plan queue
       ├─ upload -> kernel launch
       └─ hostStreamPlanTask -> proxy 描述符完成所有权转移
                                -> reclaimer 入 callbackQueue
  -> 主线程在本轮 launch 后的 safe point 消费 reclaimer

与上面并行：
  GPU event completion -> workFifoConsumed / DCCL completion
  proxy progress completion -> ncclProxyArgs 回收
```

- workNode、batch、stream list 和普通 `kernelArgs` 从 `memScoped` 分配；task、proxy op、plan 结构体从各自 memory pool 分配。group 完成时 scoped stack 可以回收，因为普通 kernel 的参数已被 CUDA launch 接收，work payload 也已上传到 Args/FIFO/Persistent storage。
- 对非 persistent plan，`hostStreamPlanTask()` 在 profiler task/group stop、proxy op 已复制并发布后，把 `plan->reclaimer` 放入 MPSC `callbackQueue`。host-callback 路径中的入队可能先于 CPU 发出 kernel launch；无 host callback 的路径则在 launch 后直接入队。
- 入队只是跨执行上下文通知。communicator 主线程到本轮 launch 后的 safe point 才调用 `reclaimPlan()`，此时 CUDA launch 已接收参数、work 已进入 Args/FIFO，proxy 使用复制到 shared pool 的 op，异步资源另有 event 或 cleanup 生命周期。该回收仍不等待 GPU 或网络整体完成，不能当成 collective completion 信号。
- FIFO 的 consumed cursor 通过完成 event 更新，host 线程在 FIFO 不足时会轮询 event callback 并等待。
- CUDA graph 中的 plan 为 persistent，graph destructor 执行时才回收；若使用 persistent storage，还要释放其 device work buffer。
- `ncclLaunchFinish()` 记录 launch-stream completion dependency；FIFO completion callback 与 DCCL completion probe 也通过各自 CUDA event 挂到该 stream 时序上。

## 11. 最关键的设计不变量

1. **task 与 plan 分离**：task 描述用户意图，plan 描述一次实际 kernel/proxy 发射；一个 API 可被拆到多个 plan，一个 plan 也可承载多个 task。
2. **`ncclDevWork*` 才是 kernel 的操作描述**：`ncclTaskColl` 的 host 字段不会直接被 GPU 使用，必须经过注册、算法选择、channel 切分后编码成 device work。
3. **`ncclDevKernelArgs` 只描述入口和索引**：真正的通信拓扑在 `devComm`，真正的操作参数在 batch 指向的 work storage。
4. **channel mask 与 batch index 一致**：grid 只启动有效 channel 数，`blockIdx.x` 通过 mask 映射到真实 channel，并从 batch 数组对应位置开始遍历。
5. **collective 先于 P2P**：plan 切分顺序和 P2P epoch/round 约束共同保证不同 rank 生成一致的 work/batch 边界。
6. **GPU kernel 与 proxy 是双轨协作**：GPU 执行数据面，proxy 执行 network/异步 transport 控制面；两者通过 connection 的 head/tail/FIFO、proxy op 发布和 stream ordering 协调。`workCounter` 属于 profiler 计数，不是通用 transport 同步量。
7. **连接信息先于 kernel**：transport 初始化把 `ncclConnInfo` 写入 device peer；enqueue 阶段只选择已存在的 connection 和其能力，不在 kernel 中做拓扑发现或连接建立。

## 12. 建议的源码阅读顺序

```text
nccl/src/collectives.cc
  -> nccl/src/enqueue.cc:2898-3046
  -> nccl/src/group.cc:290-362, 632-723
  -> nccl/src/enqueue.cc:364-830
  -> nccl/src/enqueue.cc:840-1178
  -> nccl/src/enqueue.cc:1509-1785
  -> nccl/src/enqueue.cc:1812-1872
  -> nccl/src/init.cc:549-649
  -> nccl/src/include/comm.h:186-497
  -> nccl/src/include/device.h:237-475
  -> nccl/src/device/common.h:130-410
```

如果要继续深入某个具体 collective，应从 `ncclDevFuncId()` 和 `device/generate.py` 找到对应的 generated kernel，再进入 `nccl/src/device/all_reduce.h`、`all_gather.h`、`reduce_scatter.h`、`broadcast.h` 或 `sendrecv.h`，观察 `ncclDevWork*` 字段如何被消费。
