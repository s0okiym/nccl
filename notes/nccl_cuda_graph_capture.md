# NCCL 对 CUDA Graph Capture 的支持

> 结论来源:`src/include/strongstream.h`、`src/misc/strongstream.cc`(强流与捕获语义)、
> `src/enqueue.cc`(捕获检测、发射排序、持久化)、`src/init.cc`(graphUsageMode / 参数配置)、
> 以及 `docs/dev_guide/nccl_internals.md`。

## 一句话

NCCL 把所有 CUDA 调用(kernel 发射、event、host 回调、内存操作)都做成**可捕获**的;关键是引入
**Strong Stream** 在多次图实例化之间保持内部流身份、用 `serialEvent` + `cudaEventWaitExternal` 把各次
图发射串行化,从而让图拓扑保持稳定、`cudaGraphExecUpdate` 能原地更新;被捕获的 plan 转为
**persistent** 模式并注册图析构器,使同一张图可被反复 `cudaGraphLaunch`。

---

## 关键概念

| 概念 | 含义 | 位置 |
|---|---|---|
| `ncclCudaGraph` | 对 `cudaGraph_t` 的轻量包装,记录 `origin`(捕获起始流)、`graph`、`graphId` | `strongstream.h:30` |
| `ncclStrongStream` | "强流":跨多次图捕获保持身份的内部流(liveStream + 每图 captureStream + serialEvent) | `strongstream.h:116` |
| `capturingGraph` | planner 当前这组调用所属、正在被捕获的图 | `enqueue.cc:2594` |
| `serialEvent` | 用于跨图发射串行化的 CUDA event,捕获时以 `cudaEventWaitExternal` 等待 | `strongstream.cc:127/247` |
| `graphUsageMode` | 0/1/2;2="graph mixing"(默认),允许多 comm 的图混合 | `init.cc:2427` |
| `GRAPH_STREAM_ORDERING` | 0=直接在 origin 上跑;非0(默认)=用 NCCL 自有 captureStream 串行化 | `enqueue.cc:1539` |

**为什么需要 Strong Stream**:普通 CUDA stream 被捕获后,它在某次图实例化中的"被捕获形态"与未捕获形态、
以及和其它次实例化的形态**没有任何关联**。NCCL 的内部流(`deviceStream`/`hostStream`/`launchOrder`)需要
跨图发射序列化地访问**持久资源**(如 persistent 工作缓冲、proxy 参数),普通流做不到,因此引入强流抽象
(`strongstream.h:69-83` 注释)。

---

## 流程与机制

### 1. 捕获检测
- 每次 `ncclAllReduce` 等在 `taskAppend` 阶段调用 `ncclCudaGetCapturingGraph(info->stream, ...)`
  (`enqueue.cc:2585` / `3066`),内部用 `cudaStreamGetCaptureInfo` 判断该流是否处于捕获中。
- 要求 **CUDA runtime ≥ 11.3 且 driver ≥ R465**,否则直接返回 `ncclInvalidUsage`(`strongstream.cc:81-94`)。
- 同一 group 内的所有 NCCL 调用**必须属于同一张图**,否则报错(`ncclCudaGraphSame` 检查,
  `enqueue.cc:2595-2597`)——这是图捕获正确性的硬约束。

### 2. 内部流的强流捕获(核心)
- 发射前 `ncclStrongStreamAcquire(graph, &ss, ...)`(`strongstream.cc:164`)为当前图取一个
  **per-graph 的 `captureStream`**,并通过 `recordEventOnStream` 把 `serialEvent` 作为 record 节点挂到图上,
  使本次图发射排在上一次之后。
- 这样多次 `cudaGraphLaunch` 之间内部流的顺序稳定 → **图拓扑逐次一致**,`cudaGraphExecUpdate` 才能原地更新
  可执行图而不重建(`enqueue.cc:1658` 注释)。
- 非捕获路径(`graph.graphId == ULLONG_MAX`)直接用 `liveStream`,无额外开销。

### 3. 两种流排序模式(GRAPH_STREAM_ORDERING)
`useLaunchStream = capturing && !ncclGraphStreamOrderingSerialize(comm)`(`enqueue.cc:1646`):

- **`=0`(直接在 origin 上跑)**:kernel 直接 `cuLaunchKernelEx` 到 `launchStream`(=图 origin)。
  用 `cudaStreamWaitEvent(launchStream, serialEvent, cudaEventWaitExternal)`(`enqueue.cc:1661`)
  串行化;首次捕获时在 liveStream 上 bootstrap 记录 `serialEvent`(`1660`),使每张图都含 ExternalWait 节点、
  拓扑一致。`graphStreamOrdering=0` **不兼容 graphUsageMode=2**(graph mixing),NCCL 会告警并回退
  (`init.cc:2459`)。
- **`=1`(默认 serialize)**:走强流的 captureStream 路径(`useLaunchStream=false`),由强流机制用
  `serialEvent` 串行化。这是与默认 `graphUsageMode=2` 配套的路径。

### 4. 隐式排序(Implicit Ordering / LAUNCH_COMPLETION_EVENT)
`getImplicitOrder`(`enqueue.cc:1554`)决定如何给"多次发射"定序:
- CUDA ≥ 12.3 且开启隐式 launch order:用 `CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT`
  (绑 `sharedRes->launchEvent`,`enqueue.cc:1812`)——顺序由**kernel 完成**而非发射顺序决定。
- 更早版本:用 kernel 完成事件 `finishedEvent` 串行执行(`enqueue.cc:1940`)。
- 该顺序落在每 CUDA context 的 `launchOrder` 强流上(`comm->context->launchOrder`),且即使用
  `GRAPH_STREAM_ORDERING=0` 也不能省略(`enqueue.cc:1928` 注释)。

### 5. 持久化(Persistent)与工作存储
- 一旦 `capturingGraph` 有效,`planner->persistent = true`,每个 plan 也置 `persistent`
  (`enqueue.cc:1571/1590`)。
- 工作描述符存储从 **work FIFO** 切换为**专用 persistent 缓冲**:`workStorageType =
  ncclDevWorkStorageTypePersistent`(`enqueue.cc:1592`);`uploadWork` 为其 `cudaMallocAsync` 一块持久设备缓冲
  (`enqueue.cc:1270-1348`)。
- 集合通信 kernel 被**录制进图一次**,该图即可反复 `cudaGraphLaunch`;生命周期用
  `sharedRes->persistentRefs` / `comm->localPersistentRefs` 计数(`enqueue.cc:1729-1732`)。
- 清理:用 `ncclCudaGraphAddDestructor`(内部 `cudaUserObjectCreate` + `cudaGraphRetainUserObject`)
  注册 `persistentDestructor`,当图被销毁时释放 persistent 缓冲(`enqueue.cc:1732` /
  `strongstream.cc:117`)。

### 6. Proxy / Host 回调在捕获中
- 需要推送 proxy 参数(网络建连等)时,NCCL 用 `cudaLaunchHostFunc(hostStream, hostStreamPlanCallback, plan)`
  把 host 任务排成**图节点**(`enqueue.cc:1717`)。
- 捕获期间该 host 函数**不执行**,只在图真正 `cudaGraphLaunch` 时才运行——所以网络收发在图执行时发生,
  而非捕获时。
- 是否走 host 回调由 `persistent || ncclCudaLaunchBlocking || proxy 未就绪` 决定(`enqueue.cc:1703`),
  非捕获路径尽量不做以省开销。

---

## 正确性要点与约束

- **同一 group 必须落在同一张图**;跨图会直接报错(`enqueue.cc:2595`)。
- **版本门槛**:CUDA ≥ 11.3 / driver ≥ R465,否则 `ncclCudaGetCapturingGraph` 返回 invalid usage。
- **`GRAPH_STREAM_ORDERING=0` 与 graph mixing(`graphUsageMode=2`)不兼容**,会告警回退。
- **捕获时不真正收发网络**:网络工作由图执行时的 host 节点完成。
- **强流跨线程竞争检测**:`NCCL_LAUNCH_RACE_FATAL`(默认 1),若 Acquire/Release 不是同一线程会告警
  (`strongstream.cc:156/336`)。
- 内存注册进图用 `ncclCommGraphRegister`(`register.cc:166`),由 `NCCL_GRAPH_REGISTER`(默认 1)控制。

## 相关参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `NCCL_GRAPH_STREAM_ORDERING` | undef(默认走 serialize 路径) | 0=origin 直跑;非0=强流 captureStream 串行化 |
| `NCCL_GRAPH_REGISTER` | 1 | 是否把用户缓冲注册进图(`ncclCommGraphRegister`) |
| `NCCL_GRAPH_MIXING_SUPPORT` | 自动探测 | 决定 `graphUsageMode` 是否取 2(graph mixing) |
| `NCCL_LAUNCH_RACE_FATAL` | 1 | 强流被多线程竞争发射时是否 fatal |
| `NCCL_LAUNCH_ORDER_IMPLICIT`(隐式) | 关 | 启用 launch-completion 隐式排序(CUDA≥12.3) |
