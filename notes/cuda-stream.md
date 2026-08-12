# NCCL 如何使用 CUDA Stream

> 结论来源：`src/enqueue.cc`(发射编排)、`src/group.cc`(group 语义)、`src/proxy.cc`(proxy 线程)、
> `src/init.cc`(sharedRes)、`src/include/comm.h` / `strongstream.h`,以及 `docs/dev_guide/nccl_internals.md`。

## 一句话

用户传入的 `cudaStream_t` 是**集合通信 kernel 的归处**:NCCL 在 `ncclGroupEnd` 把多个用户流上的任务
合并成 `ncclKernelPlan`,用 CUDA event 把这些流在 **launchStream(发射流)** 上排好顺序后,把 kernel
launch 到 launchStream;网络/SHM 进度由独立 proxy 线程经 host callback 与 event 联动,**从不占用任何
用户 stream**。

---

## 总体模型

| 角色 | 承载 | 用途 |
|---|---|---|
| 用户流 `info.stream` | 用户的 `cudaStream_t` | 集合通信 device kernel 挂在这里 |
| 内部 `deviceStream` | `sharedRes` | device 内存分配/拷贝(`ncclCudaCallocAsync`、`cudaMemcpyAsync`) |
| 内部 `hostStream` | `sharedRes` | `cudaLaunchHostFunc` 推送 proxy 参数 / 回收 |
| `launchOrder` | 每 context | 保证同 context 内多次发射的程序顺序 |
| proxy 线程 | 独立 CPU 线程(每 GPU 一个) | 网络/SHM 进度,**不在任何 CUDA stream 上** |

---

## 流程与机制

### 1. 入口:stream 被记进任务
`ncclAllReduce(...)`(`src/collectives.cc`)只构造一个 `ncclInfo`,其中带着用户传入的 `stream`。
此时**不发射 kernel**。

### 2. Group 语义:延迟发射、攒成 plan
- 在 `ncclGroupStart` / `ncclGroupEnd`(或隐式 group)之间,NCCL 把请求累积成 task,挂着各自的
  `info.stream`。
- 真正发射发生在 `ncclGroupEnd` → `ncclGroupEndInternal`(`src/group.cc:766`),它把所有 task 组装成
  `ncclKernelPlan`(**一个 plan = 一次 kernel 启动**)。

### 3. 多流聚合与排序(核心机制)
一个 plan 可能包含来自**多个不同用户流**的 task。planner 收集这些流,选第一个作为 **launchStream**,
并在发射前做流间同步(`src/enqueue.cc:1643-1727`):

- **用户流之间**:让 `userStream[0]`(=launchStream)等待每个其它 `userStream[i]` —— 在各流上
  `cudaEventRecord(scratchEvent)` 再 `cudaStreamWaitEvent(launchStream, scratchEvent)`,确保所有
  用户数据就绪。
- **等待内部 deviceStream**:`ncclStreamWaitStream(launchStream, deviceStream, ...)`(`enqueue.cc:1676`),
  deviceStream 负责前序 device 准备。
- **等待每上下文 launchOrder 流**:`ncclStreamWaitStream(launchStream, launchOrder, ...)`(`enqueue.cc:1696`),
  保证同 CUDA context 内多次发射的程序顺序。
- **若有 proxy 任务**:launchStream 还要等待内部 **hostStream** 上的 host callback
  (`enqueue.cc:1722`)。

### 4. Kernel 真正发射到 launchStream
`ncclLaunchKernel`(`src/enqueue.cc:1753`)以 `grid = 通道数`、`block = threadPerBlock` 通过
`cuLaunchKernelEx(..., launchConfig.hStream = launchStream, ...)`(`enqueue.cc:1838`)把集合通信 kernel
挂到 launchStream。因此**集合通信 kernel 与用户的其它 CUDA 工作串在同一 stream 上**,由 CUDA 保证顺序
与并发。

### 5. 内部流与事件(sharedRes)
`sharedRes` 是同一进程内所有 comm 共享的(`ncclCommSplit`/clone 复用,`src/init.cc:466-472`),含:

- **deviceStream**:`ncclCudaCallocAsync` / `cudaMemcpyAsync` 等 device 侧准备。是 strong stream,用
  `ncclStrongStreamAcquire/Release` 跨流同步(`src/include/strongstream.h`)。
- **hostStream**:`cudaLaunchHostFunc(hostStreamPlanCallback, plan)`(`enqueue.cc:1717`)把 host 任务排上
  GPU timeline,launchStream 等它。
- **launchEvent / scratchEvent**:CUDA event,做流间同步的"门闩";`launchEvent` 还作为 kernel 的
  `LAUNCH_COMPLETION_EVENT`(`enqueue.cc:1812`)。
- **`ncclCommIntraBarrierIn/Out`**(`src/include/comm.h:855/874`):同进程内多个 comm 之间的轻量 barrier
  (对齐 padding 防 false sharing),保证多 comm 启动顺序一致。

### 6. Proxy 线程:不在任何 CUDA stream 上
网络/跨进程 SHM 进度由独立 CPU 线程 `ncclProxyProgress`(`src/proxy.cc:954`,每 GPU 一个)驱动,它用条件
变量 `pool->cond` + 轮询推进 `ncclConnFifo`,**不占用用户 stream**。它所需的参数通过 host 侧
`cudaLaunchHostFunc` 回调 `hostStreamPlanCallback` 推到 GPU timeline —— GPU 真正执行到该点时 CPU 才去
准备网络,避免 host 跑飞。这正是第 3 步 `cudaLaunchHostFunc` + `ncclStreamWaitStream(launchStream,
hostStream)` 的由来。

### 7. CUDA Graph 捕获
`ncclPlannerSetCapturingGraph`(`src/enqueue.cc:2585`)检测 `info->stream` 是否处于 capture。捕获时
launchStream 即 graph origin stream,NCCL 用 `cudaEventWaitExternal`(`serialEvent`)保持 graph 结构一致,
使 `cudaGraphExecUpdate` 成功,并可用 `programmaticStreamSerialization`、`LAUNCH_COMPLETION_EVENT` 等
launch attribute。

---

## 时序示意

```
用户代码 ──ncclAllReduce(stream)──▶ 攒成 task(info.stream)
                                       │ (ncclGroupEnd)
                                       ▼
                              组装 ncclKernelPlan(跨多 stream)
                                       │ 用 CUDA event 在 launchStream 上排序:
                                       │   各用户流 / deviceStream / launchOrder / hostStream
                                       ▼
                  cuLaunchKernelEx(..., hStream=launchStream)  ← 集合通信 kernel
                                       │
                  hostStream 上的 callback ──▶ proxy 线程(独立 CPU)驱动网络/SHM
```

## 要点

- **用户 stream 是集合通信 kernel 的归处**,不是 NCCL 用来偷偷干活的私有流。
- **多流合并**:group 内不同 stream 的 task 在 group end 被合成一个 plan,靠 CUDA event 在 launchStream
  上统一排序后再发射。
- **NCCL 的"私活"(device 分配、proxy 参数推送)走内部 deviceStream / hostStream**,通过 event 与
  launchStream 衔接,不污染用户 stream。
- **网络/SHM 进度在独立 proxy 线程**,靠 `cudaLaunchHostFunc` 回调 + event 与 GPU timeline 对齐。
