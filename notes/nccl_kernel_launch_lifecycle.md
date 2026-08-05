# NCCL Kernel Launch 生命周期：准备、发射与收尾（v2.30.7）

> 本文聚焦普通 NCCL device kernel 的 launch 路径；任务如何选算法、划 channel 与创建
> `ncclDevWork*` 的前半段见 [nccl_enqueue_pipeline.md](nccl_enqueue_pipeline.md)。源码入口主要是
> `src/group.cc`、`src/enqueue.cc`、`src/include/{comm,device,strongstream}.h` 和 `src/device/common.h`。

## 1. 全景：一个 group 如何变成多个 CUDA kernel

`ncclAllReduce` 等 API 将 task 放进 `comm->planner`；最外层 `ncclGroupEnd()` 的 `groupLaunch()` 先做
`ncclPrepareTasks()`、buffer 注册和 `ncclTasksRegAndEnqueue()`，再调用 `doLaunches()`。后者按同一
`intraComm0` 的 communicator 组成 clique：先为每个 comm 调 `ncclLaunchPrepare()`，随后按轮次逐个发射
每个 comm 的下一个 plan。`NCCL_LAUNCH_MODE=GROUP` 时，intra-process barrier 还会使各 local rank 在每轮
launch 对齐；否则仅在本地尚有 plan 时继续。

```text
task / ncclDevWork*  ──> ncclKernelPlan ──> uploadWork()
                                      │              │
                                      │              ├─ Args / FIFO / Persistent
                                      │              ▼
doLaunches() ──> ncclLaunchKernel() ──> cuLaunchKernelEx()
                                      │
                                      └─> proxy op 提交、event/host callback、plan 回收
```

一个 plan 对应一次 launch（CE、RMA 和 symmetric collective 有各自的 launch 路径）。普通任务按
**collective → batched broadcast → p2p** 的顺序装入 plan；尤其不能先切 p2p，否则各 rank 的预算切分点和
channel 选择可能不同，进而死锁。`finishPlan()` 把每 channel 的 work batch 轮转排入 kernel 参数区，并按
`opCount` 合并各 channel 的 proxy-op 队列。

## 2. clique：进程内 communicator 的协调单元

此处的 **clique** 不是 NVLink 拓扑或图论中的完全子图，而是同一个 NCCL 全局 communicator 中、位于**同一
进程**的一组 local-rank `ncclComm`。初始化扫描 `peerInfo[]`：`hostHash` 与 `pidHash` 都相同的 rank 被归为一组；
其中全局 rank 最小的 comm 成为 leader，所有成员的 `intraComm0` 都指向它，并记录 `intraRank`、`intraRanks`。
leader 还以 `intraNext` 串起本地成员。因此 `intraComm0` 相等就是源码判断“属于同一 clique”的依据。

```text
同一个 8-rank communicator：
  进程 P0: rank 0,1,2,3  → clique A, intraComm0 = comm(rank 0)
  进程 P1: rank 4,5,6,7  → clique B, intraComm0 = comm(rank 4)
```

单进程单 GPU 时 clique 只有一个成员；`ncclCommInitAll` 或一个进程驱动多 GPU 时它才会有多个成员。它存在的原因是
这些 comm 虽各自代表一个 rank，却必须协调共享的 host-side 状态和执行节奏：

- `ncclGroupCommJoin()` 让同 clique 的成员在本次 group 链表中连续，`doLaunches()` 因而能先为全组 prepare、
  再一轮轮地发射各成员的下一个 plan。
- `NCCL_LAUNCH_MODE=GROUP` 下，成员在 leader 的 `intraBarrierCounter/intraBarrierGate` 上汇合。每轮以“是否仍有
  未发射 plan”作为输入，barrier 返回总和；所有成员据此一致地继续或结束，避免本地 GPU 的 collective launch 节奏失配。
- 同一 clique 不能混用 CUDA Graph capture 与非 capture；`doLaunches()` 在所有成员已经进入 barrier 前检查并拒绝这种
  状态，避免无法成立的 stream/graph 依赖。
- group 收尾时的 preconnect 以 clique 为边界处理，源码明确将其用于规避 split 后共享 communicator 的连接竞争；
  finalize 也等待 clique 中全部成员到达后才清理 local resources。

不要混淆两条链：`groupNext` 是某次 `ncclGroupStart/End` 中待处理 comm 的链，`intraNext` 是 clique 内成员链；
前者刻意把后者的同组成员排在一起，后续 launch 才能按 clique 分段。

## 3. 发射前的关键数据结构

| 结构 | 作用 |
| --- | --- |
| `ncclKernelPlanner` | group 生命周期中的队列与状态：待调度 task/work、正在构造的 `wipPlan`、`planQueue`、用户 stream 列表和 capture graph。 |
| `ncclKernelPlan` | 一次 launch 的完整描述：`channelMask`、block 线程数、`kernelFn`、work/proxy/task/cleanup 队列及 persistent 标记。 |
| `ncclDevKernelArgs` | 唯一的 kernel 参数块：device `comm`、启用 channel 位图、work 存储类别、环形缓冲 mask 与 work 地址。 |
| `ncclDevWorkBatch` | 一个 channel 的一批同类 work：`funcId`、类型、`offsetBitset` 和 `nextJump`；首 batch 固定在 `batches[blockIdx.x]`。 |
| `ncclKernelComm` / `ncclDevChannel` | host 初始化后长期驻留 device 的 communicator、拓扑连接与每 channel 的运行状态。 |

`ncclTestBudget()` 同时约束参数区和外部 work 空间。每个 plan 的 `kernelArgs` 至少包含 args 与 batch 描述；若全部 work 也能装下 `comm->workArgsBytes`（当前最多 4 KiB），`finishPlan()` 将其提升为 `Args`，避免额外内存访问。

## 4. work 上传与 stream 编排

`ncclLaunchKernelBefore_NoUncapturedCuda()` 调用 `uploadWork()`。普通 work 有三种存储方式：

- **Args**：`ncclDevWork*` 紧跟在 `ncclDevKernelArgs` 后，device 从 parameter memory 读取。
- **FIFO**：写入 `comm->workFifoBuf`（pinned host memory，或 GDR 映射的 CUDA memory）。`workFifoProduced` 是单调游标；空间不足时 `waitWorkFifoAvailable()` 轮询 event callback、检查 abort flag，直到 `workFifoConsumed` 前进。GDR 路径在更新游标后执行写屏障。
- **Persistent**：仅 graph capture 的大 work 使用。host 先建对齐临时缓冲，再在 NCCL 的 `deviceStream` 上 `cudaMallocAsync` 和 `cudaMemcpyAsync` 到专属 device buffer；完成 event 触发回调释放 host 缓冲。它不复用 FIFO，以保证被捕获的 graph 后续重放仍可访问 work。

一个 group 可以有多个用户 stream。`ncclLaunchPrepare()` 使第一个（`launchStream`）等待其余 stream 与
NCCL `deviceStream`，所以整个 group 保持用户可见的前后依赖。`ncclStrongStream` 为 live stream 和各
capture graph 保留稳定的逻辑身份，通过 acquire/release 与 `serialEvent` 串行化对共享资源的访问。若开启
`NCCL_LAUNCH_ORDER_IMPLICIT`，还会通过每 CUDA context 的 `launchOrder` stream 排列不同 communicator 的
launch。

在 graph capture 中 plan 标为 `persistent`。默认使用 capture-aware `deviceStream`；当
`NCCL_GRAPH_STREAM_ORDERING=0`，kernel 改在 graph origin stream 发射，并以 external event wait 维持跨图顺序。

## 5. CUDA kernel 如何真正发射

`ncclLaunchKernel()` 由 `channelMask` 的 popcount 得到 grid.x（每个启用 channel 一个 block），以 plan 内
任务所需的最大 `nWarps * 32` 得到 block.x，并选择动态 shared memory 大小。若同一 plan 内任务同质，
`kernelFn` 指向 `ncclDevKernelForFunc[devFuncId]` 的特化 kernel；混合任务则是 `ncclDevKernel_Generic`，由
device function table 再分派。

NCCL 先用 `cudaGetFuncBySymbol()` 得到 `CUfunction`，再用 `extra` 的
`CU_LAUNCH_PARAM_BUFFER_POINTER/SIZE` 传递整个 `kernelArgs` buffer：CUDA runtime 与 driver 均不低于 11.8 时走
`cuLaunchKernelEx()`，否则回退 `cuLaunchKernel()`。前者按能力附加 launch attribute：sm90 的 CGA cluster（必须整除
grid）、remote memory-sync domain、CUDA 12.3 的 launch-completion event 和 symmetric kernel 的 programmatic
serialization，以及 CUDA 13/sm100 的 NVLink util-centric scheduling。

在 `ncclKernelMain()` 内，一个 block 从 `channelMask` 找到其第 N 个置位 channel；前两个 warp 将 args、
`ncclKernelComm` 与 `ncclDevChannel` 拷到 shared memory，其余线程加载 `batches[blockIdx.x]`。`offsetBitset`
压缩了本 batch 实际存在的 work；work 可从 parameter space（`ld.param`）或 FIFO/persistent global memory 读取。
随后 kernel 沿 `nextJump` 执行该 channel 的后续 batch，并运行特化 `RunWorkBatch` 或通用 function-table entry。

## 6. Proxy、完成与回收

带 NET transport 的 plan 同时有 `proxyOpQueue`。常规模式下，CUDA launch 已提交后
`ncclLaunchKernelAfter_NoCuda()` 立即在 host 调 `hostStreamPlanTask()`：它把 plan 内局部 `opCount` 平移为
communicator 全局序号，调用 `ncclProxySaveOp()`，再以 `ncclProxyStart()` 唤醒 progress thread。capture、launch-blocking
或已有 persistent graph 尚在执行时，这项 host 工作会以 `cudaLaunchHostFunc()` 排到 `hostStream`，并令 kernel 等待它，
确保 proxy 参数在正确的 stream/graph 顺序中提交。

`ncclLaunchFinish()` 在 launch stream 记录完成 event：NCCL `deviceStream` 快进至该 event，其余用户 stream 都等待它；
随后释放 strong stream，或在 graph 中记录下一次 capture 所需的 serial event。FIFO 每推进超过容量的 1/8，会把
`workFifoProduced` 快照附到该 event；event callback 完成后才更新 `workFifoConsumed`，从而允许环形缓冲复用。

非 persistent plan 的 host 任务会把 `plan->reclaimer` 投递到 MPSC `callbackQueue`，随后由后续 group 的
`ncclCommPollCallbacks()` 释放 task、proxy op、注册清理 callback 和 plan。captured plan 则把
`persistentDestructor` 绑定到 CUDA graph，graph 销毁后才投递回收；这也会释放其 persistent work buffer 并递减引用计数。
这两个路径避免了 GPU 或 graph 仍可能读取 metadata 时过早释放内存。

## 7. 阅读与调试入口

- launch 主线：`src/group.cc:doLaunches` → `src/enqueue.cc:ncclLaunchPrepare` → `ncclLaunchKernel` → `ncclLaunchFinish`。
- metadata：`finishPlan`、`uploadWork`、`src/include/device.h:ncclDevKernelArgs/ncclDevWorkBatch`。
- device 执行：`src/device/common.h:ncclKernelMain` 与 `loadWorkBatchToShmem`。
- 回收与异步边界：`hostStreamPlanTask`、`reclaimPlan`、`src/include/comm.h:ncclCommPoll{,Event}Callbacks`。

排查 launch 顺序或 work FIFO 卡住时，优先同时观察 `plan->channelMask`、`plan->workStorageType`、
`workFifoProduced/workFifoConsumed`、`plan->isHostCbEnq` 及 `planner->capturingGraph`；它们能区分调度、stream
依赖、proxy 提交和缓冲复用四类问题。
