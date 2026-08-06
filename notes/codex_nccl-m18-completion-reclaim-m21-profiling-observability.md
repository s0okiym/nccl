# NCCL M18/M21：Completion/reclaim 与 Profiling/observability

本文基于当前仓库 `nccl/src/` 的实现，聚焦两个容易被混淆、但对理解 NCCL 执行时序和问题定位都很关键的主题：M18 的完成与回收，以及 M21 的 Profiling/observability。文中的“完成”不是单一事件，而是 GPU、proxy、FIFO 和 CPU 资源分别达到可安全推进的状态。

相关主线代码：`nccl/src/enqueue.cc`、`nccl/src/group.cc`、`nccl/src/include/comm.h`、`nccl/src/init.cc`、`nccl/src/proxy.cc`、`nccl/src/plugin/profiler.cc`、`nccl/src/include/plugin/nccl_profiler.h`、`nccl/src/include/plugin/profiler/profiler_v6.h`，以及本 fork 的 `nccl/src/trace/` 和 `nccl/src/transport/{net,p2p,shm}.cc`。

## 1. 先建立正确的完成模型

一次 collective 从 API 返回到所有内部资源释放，至少包含以下边界：

| 边界 | 代表状态/信号 | 谁推进 | 能证明什么 | 不能证明什么 |
|---|---|---|---|---|
| API enqueue | task 进入 planner 队列 | 调用线程 | 参数已被 NCCL 接收 | GPU 已执行、网络已完成 |
| kernel/stream completion | launch stream 上的 CUDA event 已完成 | GPU + host event poll | event 之前的 launch-stream 工作及已显式汇入该 stream 的依赖已完成 | proxy args、plan、host 资源已回收 |
| proxy completion | `op->progress()` 令 `op->state == ncclProxyOpNone` | proxy progress thread | 该 proxy transport op 不再需要 active args | 对应 plan 一定已释放 |
| FIFO reclaim point | `workFifoConsumed` 更新 | `KernelFinishCallback_fn` | FIFO 中相应 work bytes 可重新覆盖 | plan/task 对象已经释放 |
| plan reclaim | `reclaimPlan` 执行完 | 主线程轮询 `callbackQueue` | host plan、task、plan 内 proxy 描述符和 cleanup 对象已按其所有权协议释放 | GPU kernel 或 transport proxy 已完成 |

因此，`ncclCommPollEventCallbacks()` 和 `ncclCommPollCallbacks()` 解决的是两类不同问题：前者等待 CUDA event 并更新 GPU 顺序相关状态，后者执行由 host task、graph destructor 等执行上下文排入的 CPU 资源回收。普通非 persistent plan 在 proxy op 已复制到 shared pool 后即可把 reclaimer 入队；host-callback 路径中的入队可能先于 CPU 发出 kernel launch。真正的消费发生在 communicator 主线程的后续 safe point，此时本轮 launch 已提交，但 GPU 和网络可能仍在运行。安全性来自“入队与消费分离”、所有权转移及异步资源自己的 event/cleanup 协议，而不是一次全局完成同步。

## 2. M18：Completion/reclaim 的完整链路

### 2.1 Plan、work storage 与 launch 的关系

`ncclLaunchPrepare()` 为 planner 中的任务生成 `ncclKernelPlan`，设置 `plan->reclaimer.fn = reclaimPlan`，并根据是否处于 CUDA Graph capture 选择 storage：

- 普通执行初始使用 `ncclDevWorkStorageTypeFifo`；`finishPlan()` 在 work 能放进 kernel arguments 时可提升为 `Args`。
- Graph capture 使用 `ncclDevWorkStorageTypePersistent`，plan 和其 work buffer 的寿命要覆盖 graph 的生命周期。
- `Args` 直接把 work 放在 `plan->kernelArgs` 后面；`Fifo` 写入 `comm->workFifoBuf`；`Persistent` 使用对齐 host buffer，并通过 `cudaMallocAsync` 分配 device buffer。

普通 FIFO 的写入由 `workFifoProduced` 表示，回收位置由 `workFifoConsumed` 表示。`waitWorkFifoAvailable()` 用无符号差值判断 `(desiredProduced - consumed) <= workFifoBytes`；空间不足时轮询 event callback，并检查 `abortFlag`，避免 abort 后永久自旋。每当尚未记录的 produced 增量超过 `workFifoBytes/8`，`ncclLaunchFinish()` 才注册一个 FIFO completion callback；callback 在对应 CUDA event 完成后将 `workFifoConsumed` 更新到当次记录的 produced 值。

Persistent work 的上传顺序更严格：主机 buffer → `cudaMallocAsync` device buffer → `cudaMemcpyAsync` → `memcpyDone` event → 加入 `eventCallbackQueue`。`uploadWork_cleanup_fn` 只有在 event ready 后才释放 host buffer、销毁 event 并释放 callback；这保证异步 copy 尚未完成时不会释放源地址。

### 2.2 Host stream、proxy 与 plan reclaim

`ncclLaunchPrepare()` 会先把 plan 放入 `planner->planQueue`，设置 stream 依赖，然后按条件为包含 proxy op 的 plan 排入 `hostStream` 上的 `cudaLaunchHostFunc`。需要 host callback 的典型情况包括 persistent graph、`CUDA_LAUNCH_BLOCKING=1`，以及 host stream 前序事件尚未完成；`NCCL_LAUNCH_MODE` 是 group/parallel 的提交模式，不等同于 CUDA 的 launch blocking 开关。

host callback 或无 host callback 的直接路径最终都调用 `hostStreamPlanTask()`：

1. 开始 profiler group/task events。
2. `uploadProxyOps()` 把 plan 内的 proxy op 转成 communicator shared-resource 上跨 plan 单调的 collective/p2p `opCount`，写入 profiler context、activation mask、task event handle 和 pid。
3. `ncclProxySaveOp()` 按 ring/tree/CollNet/NVLS/PAT/Send/Recv 等 pattern，把 op 复制到对应 transport 的 proxy pool；`ncclProxyStart()` 将本批 op 发布给 proxy progress thread。
4. 停止 task/group profiler events。
5. 非 persistent plan 将 `plan->reclaimer` 通过 MPSC `callbackQueue` 交给主线程。

这里有一个重要的所有权转换：plan 中的 `ncclProxyOp` 只是提交描述符；`ncclLocalOpAppend()` 会把它复制到 shared proxy op pool，proxy thread 后续使用的是复制后的对象。因此，在 `hostStreamPlanTask()` 已经完成 proxy 提交之后，主线程可以回收 plan 内的 proxy 描述符，而不等于提前释放 proxy thread 正在运行的 `ncclProxyArgs`。

还要区分 reclaimer 的入队和执行：由 CUDA host callback 调用 `hostStreamPlanTask()` 时，callback 可能在调用线程执行 `ncclLaunchKernel()` 前就入队；无 host callback 时则由 `ncclLaunchKernelAfter_NoCuda()` 在 launch 后入队。MPSC queue 只传递通知，`ncclCommPollCallbacks()` 在 group/reclaim safe point 才真正执行 `reclaimPlan()`，因此不会在主线程仍使用 plan 发射 kernel 时提前释放它。

### 2.3 两条 callback 队列

`eventCallbackQueue` 是普通 intrusive FIFO，保存带 CUDA event 的 `ncclCommEventCallback`。`ncclCommPollEventCallbacks(comm, waitSome)` 只检查队首：

- `waitSome=false` 使用 `cudaEventQuery`，队首未完成就立即返回；
- `waitSome=true` 使用 `cudaEventSynchronize`，直到队首完成；
- 完成后出队并执行 callback，callback 通常负责销毁自己的 event 和释放自身；
- 为适应 CUDA Graph capture，函数暂时切换到 relaxed capture mode。

`callbackQueue` 是 MPSC 队列，保存 `ncclCommCallback`。CUDA host callback、graph destructor 或其他线程可以生产，主线程在安全点调用 `ncclCommPollCallbacks()` 消费。它先批量取出 callback，再调用 `cb->fn(comm, cb)`；回调可以在执行过程中释放自身。MPSC 的意义是避免在 CUDA callback 或 graph destructor 执行上下文中直接破坏 communicator 的 task memory pool。

`groupLaunch()` 不会每次 group 都强制完整回收，而是通过 `comm->reclaimSteps` 周期性地非阻塞轮询 callback queue。`ncclCommDestroy()` 则按更强的顺序处理：同步 host/device strong stream，等待 event callbacks，轮询普通 callbacks，并持续等待 `localPersistentRefs` 归零，最后才停止 proxy。这种差异解释了为什么短期内看到 memory pool 仍有对象，不一定是泄漏。

### 2.4 Proxy completion 与 plan reclaim 是两条并行生命周期

proxy progress thread 的核心循环是：

```text
progressOps(active)
  ├─ transport 的 op->progress()
  ├─ 仍在运行：保留 active args
  └─ state == ncclProxyOpNone 或发生错误：removeOp()

ncclProxyGetPostedOps()
  ├─ 从 pool 取出已发布 op
  ├─ 按 peer/opCount 组织聚合
  └─ ProxyAppend() 创建/追加 ncclProxyArgs
```

`removeOp()` 从 active 链摘除 `ncclProxyArgs`；如果存在 `nextPeer`，将后继 peer group 接到原位置，否则清空该 peer 的 append 指针，并把 args 放回 progress state 的 pool。`progressOps()` 在 op 完成或返回错误时都会走这条移除路径；proxy thread 结束前还会 drain active ops。NCCL profiler 的 proxy op stop、DCCL 的 `dccl_probe_proxy_op_end()` 都发生在 transport progress 把 op 置为完成的路径附近。

这意味着四个对象不能混为一谈：

- `plan->proxyOpQueue` 的描述符由 `reclaimPlan()` 回收；
- shared proxy pool 的 `ncclProxyOp` 在 `ncclProxyGetPostedOps()` 中归还给 peer pool；
- active transport 状态位于 `ncclProxyArgs`，由 `removeOp()` 回收；
- GPU work 的 FIFO 空间由 event callback 更新 `workFifoConsumed`。

### 2.5 Persistent graph 的 reclaim

Graph plan 不在 `hostStreamPlanTask()` 中排入普通 reclaimer。`ncclLaunchPrepare()` 为每组 persistent plans 增加 `sharedRes->persistentRefs` 与 `localPersistentRefs`，并把 `persistentDestructor(planHead)` 注册到 CUDA Graph destructor。graph 被销毁时，destructor 遍历 plan 链，将每个 reclaimer 放入 MPSC callback queue；真正的 `reclaimPlan()` 仍由 communicator 侧轮询执行。

`reclaimPlan()` 对 persistent plan 先减少两类 persistent ref；Persistent work storage 还会在 relaxed capture mode 下 `cudaFree(plan->workBufPersistent)`。随后依次释放 symmetric args、collective task 的 net handles、p2p/bcast task、proxy op（包括 `ringAlgo` 引用计数），执行 `plan->cleanupQueue`，最后将 plan 归还 `memPool_ncclKernelPlan`。因此 `localPersistentRefs != 0` 是 communicator 仍被 graph 引用的直接信号，destroy 路径必须等待它归零。

### 2.6 错误、取消和 destroy 的检查点

需要排查 hang 或资源迟迟不回收时，应按以下顺序区分：

1. `workFifoProduced` 是否持续增长而 `workFifoConsumed` 不动：关注 launch stream、队首 event callback 和 GPU 是否完成。
2. `eventCallbackQueue` 队首是否被一个未完成 event 阻塞：队列是有序检查，后面的 callback 不会越过队首执行。
3. `callbackQueue` 是否只是尚未到 group reclaim 周期：检查 `reclaimSteps` 和 `ncclCommPollCallbacks()`。
4. `planQueue` 为空并不代表 plan 已释放；`ncclLaunchFinish()` 会清空 plan queue，plan 仍等待 reclaimer。
5. persistent graph 是否仍持有 `localPersistentRefs`；这是正常引用关系，不应通过强制 free 绕过。
6. proxy 的 `active` 链、`op->done`/`op->state`、`asyncResult` 与 `abortFlag` 是否一致。

失败路径中的 `groupCleanup()` 会清理尚未发射的 plan；已注册到 graph destructor 的 persistent plan 依靠 graph 生命周期完成回收。comm destroy 先 drain CUDA 相关回调和 persistent refs，再 `ncclProxyStop()`，这是避免 UAF 的关键顺序。

## 3. M21：NCCL 官方 Profiler 的层次和数据流

### 3.1 插件、mask 与 context

`ncclProfilerPluginInit()` 在 communicator 初始化时加载 `NCCL_PROFILER_PLUGIN` 指定的 profiler；实现支持从 v6 到 v1 的 ABI 回退，并可使用 net plugin 提供的 profiler。当前 `plugin/nccl_profiler.h` 将通用 `ncclProfiler_t`、event descriptor 和 state args 别名到 v6；v6 在 v5 的 Group API、Coll/P2P API、KernelLaunch 等描述之外增加 CE collective/sync/batch 事件。插件通过 `init()` 得到每个 communicator 的 opaque context，同时写回全局 `ncclProfilerEventMask`。没有插件或插件初始化失败时，NCCL 继续正常运行。

API 线程在 `ncclProfilerStartGroupApiEvent()` 中读取 activation mask；它用 TLS 保存 `profilerGroupDepth`、group/p2p/coll API handles 和状态机。只有最外层 NCCL group 生成 group API event；嵌套 group 通过 group depth 状态更新，而不是重复创建顶层事件。proxy control event 则在 proxy thread 中重新读取全局 mask，允许控制类事件按当前 mask 决定是否记录。

### 3.2 事件层次

| 层次 | 触发位置 | 关键内容 | 父对象/关联 |
|---|---|---|---|
| Group API | group/API 入口 | group depth、是否 graph captured | 顶层 API event |
| Coll API / P2P API | collective 或 P2P API | func、count、datatype、root/peer、stream、graph 状态 | Group API |
| Group event | `hostStreamPlanTask` | 一个 kernel plan 的执行组 | plan 的 group handle |
| Coll/P2P task | plan task event | seqNumber、buffer、count、channels、warps、algo、proto | API event、Group event |
| Kernel launch | `ncclLaunchKernel` | launch stream | Group API handle |
| Proxy op | `ncclProxyOpToArgs`/proxy append | pid、channel、peer、nSteps、chunkSize、send/recv | task event |
| Proxy step | transport progress | 逻辑 step，实际按 `DIVUP(step, sliceSteps)` 归一 | Proxy op |
| Kernel channel | proxy/transport 侧 GPU timer | channel 与 GPU global timer | task event |
| Net plugin | net plugin callback | plugin id、opaque data | Proxy step |
| Proxy control | progress thread | sleep/wakeup、append、idle/active、追加数量 | proxy context |
| CE events | Copy Engine/NVLS 路径 | CE collective、sync、batch | collective API |

事件父子关系不是装饰信息：当前 `profiler_v6.h` descriptor 的 `parentObj` 和各层 parent handle 让插件能把 API、plan、task、proxy op、step 和 net plugin 组合成一棵可查询的执行树。task event 中的 `seqNumber` 来自 communicator 的 collective 序列计数；graph-captured task 对序列计数有专门的处理，以避免不同 rank 因 graph 调用路径差异产生错误的 RAS/Profiler 对齐。

### 3.3 Profiler 如何改变 proxy 行为

`ncclProfilerNeedsProxy()` 在 profiler 已加载且 op 的 activation mask 包含 `ncclProfileKernelCh` 时返回 true；首次需要时建立 profiler transport 的 send/recv proxy connections。于是某些本来只需 kernel 侧记录的观测，会额外拥有 proxy op/step 执行路径。`uploadProxyOps()` 在真正提交 op 前补齐 profiler context、task handle 和 pid，并把 plan-local `opCount` 转成 communicator/shared-resource 维度的单调编号。

proxy event 对 `sliceSteps` 做归一：一个网络传输可能由多个底层 steps 组成，Profiler 的 `nSteps`、`step` 和 `chunkSize` 表达的是面向网络 transfer 的逻辑粒度，而不是简单暴露每个 FIFO slot。step state 还可记录 `transSize`；kernel channel stop 通过 state args 写回 GPU global timer；net plugin 则由 callback 动态 start/update/stop。

Profiler 的成本不只来自 `startEvent`：它可能引入 profiler proxy connection、proxy progress 工作、event handle 和 step state。因此生产环境通常按需打开 event mask，不应把所有细粒度事件默认视为零成本。

## 4. DCCL observability：对 NCCL 生命周期的补充观测

### 4.1 独立于官方 profiler 的数据通路

本 fork 的 DCCL trace 由 `DCCL_OBSERVABILITY=1` 开启，默认关闭时 probe 主要退化为一次 branch check。初始化由 `pthread_once` 完成，在 `/dev/shm`（可由 `DCCL_BUF_DIR` 修改）创建 per-process SPSC arena，路径包含 job、rank、pid 和 epoch。arena 预留最多 64 个 producer shard。probe 先构造 256B 的内部 `dcclMetricEvent`，再转换为 256B、256B 对齐的 SPSC wire record `dcclSpscMetricEvent`；后者包含 64B transport envelope 和 192B business payload。

普通/异常 producer shard 的 SPSC readiness 不使用 per-slot sequence flag：producer 先复制完整 slot，再以 release store 推进 shard 的 `published_head`；consumer 通过 acquire 观察 head，并以 `consumed_tail` 归还容量。`producer_local_seq` 只是用于 producer-local 排序和诊断的 wire metadata，丢弃数量由 control counter 记录；它不是 slot 发布锁。独立的 emergency fallback 是多 producer ring，才使用 `dcclEmergencyRingSlot::sequence` 做逐槽发布；两套内存序协议不能混用。

写入者按来源分 lane：普通 application thread 使用 thread-local writer，proxy thread 通过 `ncclProxyState` 绑定 proxy writer；异常事件走优先级 lane，error/stall 还可使用 emergency ring。普通事件是 best-effort：lane 满时可能丢弃；priority/emergency 丢弃会按幂次告警。因此 trace 是观测系统，不应成为 NCCL 正确性的同步依赖。

当前事件定义与生产覆盖如下：

| 事件 | 当前生产位置/语义 |
|---|---|
| `DCCL_EVT_COLL_ENQUEUE` | 第一个消费 pending TLS 的 `ncclLaunchKernel()` 输出 enqueue/launch 元数据；事件名虽为 COLL，Send/Recv 也可能进入 |
| `DCCL_EVT_COLL_COMPLETE` | 上述一次观测对应的 launch-stream CUDA event 被 host 侧 event polling callback 观察完成 |
| `DCCL_EVT_PROXY_AGG` | proxy op/channel 或 GIN 聚合统计 |
| `DCCL_EVT_PROXY_ANOMALY` | 超阈值且命中采样的 proxy step |
| `DCCL_EVT_ERROR` | enqueue、transport、progress/service 等错误路径 |
| `DCCL_EVT_PROXY_STALL` | 开启 timeout 后对 active proxy sub 的周期检测 |
| `DCCL_EVT_HANG` | schema 与 dump reader 已定义，但当前 `trace_probe.cc` 没有生产该类型的路径 |

### 4.2 `COLL_*` 事件的 enqueue → launch → complete

`dccl_probe_enqueue()` 在 `ncclEnqueueCheck()` 记录 thread-local pending 信息：API 名、消息大小、comm hash、rank、enqueue timestamp 和进程内原子递增的 sequence。它没有按 `info->coll` 过滤，因此 Send/Recv 和由 collective 展开的 P2P fallback 也会写入该槽。`dccl_probe_launch()` 在 `ncclLaunchKernel()` 读取它，并写出 `DCCL_EVT_COLL_ENQUEUE`，附带 kernel launch 时间、channel 数、protocol 和算法短名；P2P-only plan 的 `collTaskQueue` 为空，此时 `graph_pattern="unknown"`、`protocol=0xFF`。当前实现把 collective 算法映射到 `graph_pattern` 字段（如 `tree`、`ring`、`nvls`），这个字段名在阅读数据时应按实现理解，而不是误认为 NCCL graph pattern。消费者不能仅凭 `COLL_*` 类型断定它一定来自普通 collective。

这条路径当前不是严格的 task-level trace：`t_dccl_pending` 只有一个 TLS 槽，每次 enqueue 会覆盖前值，第一次成功的 plan launch 又会将其消费；显式 group 中存在多个 API、多个 plan 或多个 communicator 时，不保证每个 API 都产生一对 enqueue/complete event。空 collective、单 rank、CE/RMA 等不经过 `ncclLaunchKernel()` 的提交还可能只改写 pending，而不产生该事件对。并且 `coll_name/msg_size/coll_sn` 来自最后保留的 pending enqueue，algorithm/protocol 来自被观测 plan 的第一个 `collTaskQueue` task；enqueue event 的 payload rank 来自 pending，而 wire `comm_hash` 来自当前 launch comm，completion callback 又使用 finish comm 的 rank/hash。复杂 group 中这些字段可能并非同一 task。因而事件应视为 best-effort launch 观测；若要 task 级精确关联，需要把 identity 随 task/plan 显式传递，而不能依赖单个 TLS 槽。

非 persistent plan 且 `DCCL_GPU_TIMING=1` 时，launch probe 在 launch stream 上记录 timing-enabled start event；finish probe 记录 stop event，并额外创建独立的 disable-timing completion event。独立 event 很重要：不能复用会被后续 collective 重新 record 的 `scratchEvent`，否则旧 callback 可能永远等不到自己对应的 event。event callback 执行时计算 GPU elapsed time，生成 `DCCL_EVT_COLL_COMPLETE`，并销毁 timing/completion events。

stop event 在该 communicator 的 `ncclLaunchFinish()` 才记录，而不是紧跟被观测 plan 的 kernel 之后；若一次 group 为该 comm 生成多个 plan，`kernel_exec_us` 会覆盖 start 与最终 finish 之间的多个 launch-stream work，不能解释为某一个 CUDA kernel 的独占执行时间。

`complete_ts_us` 是 host 轮询并执行 callback 的时间，代表“观测系统确认 GPU event 完成”的时间，不是硬件产生中断的精确时间；若 callback queue 前面还有未完成 event，观测完成事件还可能被队首顺序延迟。Graph persistent path 不创建 DCCL GPU timing events，因而 `kernel_exec_us` 可能为 0，但仍注册完成 callback。

### 4.3 Proxy aggregation、anomaly 与 stall

DCCL probe 被插入 NET、P2P、SHM transport 的 proxy progress 路径：

- `dccl_probe_proxy_op_start()` 为 op 建立 transport/path/fallback、comm hash、opCount 和 per-sub 统计；
- `dccl_probe_proxy_step_begin/received/end()` 记录 posted、received、transmitted 三阶段时间和 step 延迟；NET receive 可填充 received timestamp，send-side 缺失时按 posted timestamp 处理；
- `dccl_probe_proxy_op_end()` 按 channel/sub 生成聚合指标，统计 step 数、字节数、min/max/avg latency、slow step 数和 total transfer time；
- `dccl_probe_proxy_maintenance()` 首个 proxy loop 立即执行一次，之后每累计 64 个 loop 执行维护；`DCCL_PROXY_STALL_TIMEOUT` 非零时再受约 100ms 时间门控，检查 active op 的 pending steps，并在超过 timeout 后通过 emergency lane 记录一次 stall；
- progressOps、getPostedOps 或 proxy service 出错时调用 `dccl_probe_error()`，错误事件同样走 emergency lane。

`DCCL_PROXY_AGG` 有四种粒度：`off`、`channel`、`step`、`trace`。channel 模式不保存逐 step 时钟，主要给出完成后的计数/字节量；step/trace 模式维护 step latency 和 anomaly。默认 anomaly threshold 为 10ms，sample rate 默认每 1000 个超阈值 step 记录一个异常；这些参数由 `DCCL_ANOMALY_THRESHOLD_US` 和 `DCCL_ANOMALY_SAMPLE_RATE` 控制。transport/path/fallback 字段用于区分 P2P/NVL、SHM/PHB、NET/SYS 以及 P2P 或 SHM 不可用导致的降级。

这里有两个不同的序列号命名空间，而且 wire payload 中两者最终都是 32-bit：collective enqueue/complete 的 `coll_sn` 由 64-bit `g_dccl.global_coll_sn` 截断写入；proxy agg/anomaly/stall 的同名字段通常由 64-bit `args->opCount` 截断写入，GIN 聚合甚至写 0。它们不能仅因字段同名就直接相等关联，长时间运行后的 wraparound、grouped collective 和 P2P 尤其需要注意。跨层关联应同时使用 `comm_hash`、rank、时间窗口、channel、task/profiler parent handle；若需要严格一一映射，应在生产端显式携带映射键。

### 4.4 Snapshot 与配置边界

comm snapshot 是初始化/拓扑侧的静态或准静态观测，仅由 `comm->localRank == 0` 的 node-local rank 写出，通常每个 node 一份；它既不表示每个进程都会写，也不等同于全局 rank 0。输出目录由 `DCCL_SNAPSHOT_DIR` 控制，默认是节点本地的 `/tmp`。文件名只含可选 job ID 和 `commHash`，不含 node/rank；若把目录指向所有节点共享的文件系统，同一 communicator 的各节点可能竞争或覆盖同一路径，应为各节点配置独立目录。snapshot 不能替代运行时 event。常用配置包括：

```bash
DCCL_OBSERVABILITY=1
DCCL_BUF_DIR=/dev/shm
DCCL_JOB_ID=<job-id>
DCCL_GPU_TIMING=1          # observability 开启后的默认值
DCCL_PROXY_AGG=step        # off|channel|step|trace；默认 channel
DCCL_ANOMALY_THRESHOLD_US=10000
DCCL_ANOMALY_SAMPLE_RATE=1000
DCCL_PROXY_STALL_TIMEOUT=0 # 秒；0 表示关闭
```

GPU timing 需要额外 CUDA event；step/trace 需要 proxy 热路径统计；stall 还会周期扫描 active proxy ops。调试时应根据问题选择粒度，避免把 anomaly、step timing 和全量 profiler 同时打开造成不可忽略的干扰。

## 5. 把 M18 与 M21 关联起来排障

### 场景 A：API 返回后 FIFO 很快耗尽

先看 `workFifoProduced` 是否增长、`workFifoConsumed` 是否只在 event callback 中更新；再看 `eventCallbackQueue` 队首 event。若 GPU 已完成但 `workFifoConsumed` 不更新，优先检查 callback 是否被轮询，而不是先怀疑 kernel 没有完成。

### 场景 B：Profiler 显示 task 已结束，但 proxy 仍长时间 active

这是可能的正常现象：NCCL task/group profiler event 在 `hostStreamPlanTask()` 完成提交后停止，proxy op/step 是独立的异步事件。应结合 task parent handle、profiler proxy handle、`op->state`、`sub->done` 和 DCCL `PROXY_AGG/ANOMALY/STALL` 分析；DCCL collective `coll_sn` 与 proxy `opCount` 不属于同一命名空间，不能只用“数值相等 + channel”建立关联。

### 场景 C：plan queue 已清空但内存仍在

检查 `callbackQueue` 是否已经消费 reclaimer，以及是否仍有 persistent graph 引用。`planQueue` 清空只代表 launch finish 不再持有待发射计划；真正的 task/proxy/cleanup 释放在 `reclaimPlan()`。

### 场景 D：collective complete 时间异常偏晚

区分三种时间：DCCL 的 GPU elapsed、completion event 被 host 观察到的时间、callback 实际执行时间。若 event callback 队首被前一事件阻塞，`DCCL_EVT_COLL_COMPLETE` 会晚于 GPU 真正完成；此时结合 NCCL Profiler 的 Kernel launch/Channel 事件和 event queue 状态判断，不能只用 `complete_ts_us` 反推 kernel 执行时间。

### 场景 E：出现 proxy stall 或错误

先确认 `DCCL_PROXY_STALL_TIMEOUT` 已开启且维护路径正在运行，再使用 proxy 侧 `opCount`、channel、transport、fallback、steps_done/total 定位 active sub，并用 `comm_hash`、rank 和时间窗口与 collective 事件交叉验证。`DCCL_EVT_ERROR` 的 source 可区分 progress thread、proxy service、enqueue 和 transport；NCCL 的 `asyncResult` 与 `abortFlag` 则决定后续是否停止推进。

## 6. 源码阅读索引与检查清单

建议按以下顺序继续追踪具体问题：

1. `enqueue.cc`：`ncclLaunchPrepare` → `uploadWork` → `ncclLaunchKernelBefore_NoUncapturedCuda` → `ncclLaunchKernel` → `ncclLaunchKernelAfter_NoCuda` → `ncclLaunchFinish`。
2. `enqueue.cc`：`hostStreamPlanTask` → `uploadProxyOps` → `ncclProxySaveOp` → `reclaimPlan`。
3. `include/comm.h`：`eventCallbackQueue`、`callbackQueue`、`ncclCommPollEventCallbacks`、`ncclCommPollCallbacks`。
4. `proxy.cc` 与 `transport/*.cc`：`ncclProxyStart` → `ncclProxyGetPostedOps` → `ProxyAppend` → `progressOps` → `removeOp`，再看各 transport 的 `done/state` 更新。
5. `plugin/profiler.cc`、`include/plugin/nccl_profiler.h` 与 `profiler/profiler_v6.h`：ABI fallback、activation mask、parent handle、event state、CE events 和 profiler proxy。
6. `trace/trace_probe.cc` 与 `trace_spsc_*`：DCCL producer lane、event callback、proxy aggregation、priority/emergency drop。

最终检查问题时应同时回答：GPU 是否完成、FIFO 是否可复用、proxy 是否完成、plan 是否已入 reclaim queue、reclaim callback 是否执行、Profiler 事件属于哪一层、DCCL 事件是否可能因采样/队列满而缺失。只有把这些问题拆开，才能避免把“API 已返回”“kernel 已结束”“proxy 已结束”和“内存已释放”误认为同一个时刻。
