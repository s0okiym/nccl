# NCCL 核心机制总览与学习索引

> 本文是 NCCL 源码学习的总目录。目标不是一次解释所有实现，而是把机制拆成可以逐项掌握的单元：每次选择一个机制，沿“问题 → 数据结构 → host 流程 → device/proxy 执行 → 同步与完成 → 性能/故障边界”展开。
>
> 当前代码对象为本仓库 `nccl/`（NCCL v2.29.7 分支）。表格中的“已覆盖”只表示已有 notes 入口，不代表该机制已经完成实验或性能验证。

---

## 1. 机制之间的总依赖图

```text
Communicator / rank / topology
          ↓
task enqueue / planner / group
          ↓
algorithm + protocol + registration selection
          ↓
channel / warp / chunk / pattern / devFuncId
          ↓
kernel plan + work batch + proxy op
          ↓
CUDA stream ordering + kernel launch
          ↓
device dispatch + Primitives + transport/proxy
          ↓
FIFO/direct/flag synchronization + collective completion
```

建议先掌握“选择与发射”主链，再分别深入算法、协议、transport 和完成/错误处理。否则容易把算法步骤、协议数据格式和 transport 行为混成同一层。

---

## 2. 核心机制总表

| 编号 | 机制 | 要回答的核心问题 | 关键数据结构/符号 | 主要源码入口 | 当前 notes 入口 | 覆盖状态 |
|---|---|---|---|---|---|---|
| M01 | Communicator 与 rank | 哪些 GPU 参与通信？rank、node、local rank 如何映射？ | `ncclComm`、`rankToNode`、`rankToLocalRank` | `src/init.cc`、`src/include/comm.h` | 待建立专题 | 待深入 |
| M02 | Topology graph | Ring、Tree、NVLS、CollNet 如何根据硬件拓扑构造？ | `ncclTopoGraph`、ring/tree/channel topology | `src/graph/`、`src/transport/` | 待建立专题 | 待深入 |
| M03 | Collective enqueue/task | API 参数如何变成 planner task？ | `ncclInfo`、`ncclTaskColl`、`ncclTaskP2p` | `src/collectives.cc`、`src/enqueue.cc` | [`user-thread-enqueue-kernel-flow.md`](user-thread-enqueue-kernel-flow.md) | 已覆盖 |
| M04 | Planner 与 group | 多个 API、stream、communicator 如何聚合？ | `ncclGroupDepth`、`ncclKernelPlanner`、task queues | `src/group.cc`、`src/include/comm.h` | [`planner-group-aggregation.md`](planner-group-aggregation.md) | 专题覆盖 |
| M05 | 算法/协议选择 | 为什么这次选择 Ring/Tree 和 LL/SIMPLE？ | cost table、bandwidth、latency、`nMaxChannels`、`nWarps` | `enqueue.cc::ncclGetAlgoInfo`、`graph/tuning.cc` | [`protocols-primitives-kernel-launch.md`](protocols-primitives-kernel-launch.md) | 专题覆盖 |
| M06 | Chunking 与 pattern | algorithm/protocol 如何变成 chunk、slice、step 和 proxy nsteps？ | `calcCollChunking`、`ncclPattern_t`、`ncclProtoGrainSize` | `src/enqueue.cc`、`src/include/device.h` | [`chunking-pattern-fifo-registration.md`](chunking-pattern-fifo-registration.md) | 专题覆盖 |
| M07 | Protocol | SIMPLE、LL、LL128 的数据布局和 ready 语义是什么？ | `ProtoSimple`、`ProtoLL`、`ProtoLL128`、`NCCL_STEPS` | `src/device/primitives.h`、`prims_*.h` | [`protocols-primitives-kernel-launch.md`](protocols-primitives-kernel-launch.md) | 专题覆盖 |
| M08 | Primitives | 算法调用如何落实为 wait、地址选择、reduce/copy 和 post？ | `Primitives<T,RedOp,Fan,Direct,Proto,P2p>` | `src/device/primitives.h`、`prims_*.h` | [`protocols-primitives-kernel-launch.md`](protocols-primitives-kernel-launch.md) | 专题覆盖 |
| M09 | Head/tail/FIFO | 如何实现 producer/consumer credit、背压和数据可见性？ | `ncclConnInfo::head/tail`、`ncclConnFifo` | `src/device/prims_simple.h`、`src/transport/net.cc` | [`chunking-pattern-fifo-registration.md`](chunking-pattern-fifo-registration.md) | 专题覆盖 |
| M10 | Direct 与 registration | 何时直读/直写对端 buffer，何时使用 staging/proxy？ | `NCCL_P2P_READ/WRITE`、`NCCL_DIRECT_NIC`、`ptrExchange` | `src/device/prims_simple.h`、`src/register/` | [`chunking-pattern-fifo-registration.md`](chunking-pattern-fifo-registration.md) | 专题覆盖 |
| M11 | Kernel plan | task 如何按 budget、channel 和 work 类型组成 plan？ | `ncclKernelPlan`、`ncclDevWorkBatch`、`channelMask` | `src/enqueue.cc::ncclLaunchPrepare` | [`protocols-primitives-kernel-launch.md`](protocols-primitives-kernel-launch.md) | 已覆盖 |
| M12 | Kernel launch | plan 如何排 stream、上传 args、发射 grid/block？ | `ncclLaunchPrepare`、`doLaunches`、`ncclLaunchKernel` | `src/group.cc`、`src/enqueue.cc` | [`protocols-primitives-kernel-launch.md`](protocols-primitives-kernel-launch.md) | 专题覆盖 |
| M13 | Device dispatch | block 如何找到 channel，batch 如何找到 work/function？ | `ncclKernelMain`、`RunWorkBatch`、`funcId` | `src/device/common.h`、`common.cu` | [`protocols-primitives-kernel-launch.md`](protocols-primitives-kernel-launch.md) | 已覆盖 |
| M14 | Collective algorithms | Ring、Tree、CollNet、NVLS 的数据流分别是什么？ | `RunWorkColl`、`runRing`、`runTreeSplit` | `src/device/all_reduce.h` 等 | [`allreduce-kernel-execution.md`](allreduce-kernel-execution.md) | 已覆盖 |
| M15 | Transport | P2P、SHM、NET、CollNet、NVLS 如何提供连接和数据通路？ | `ncclConnector`、transport resources | `src/transport/` | 待建立专题 | 待深入 |
| M16 | Proxy control plane | proxy op 如何生成、启动、progress、完成？ | `ncclProxyOp`、`ncclProxyArgs`、`progressOps` | `src/proxy.cc`、`src/transport/net.cc` | [`allreduce-kernel-execution.md`](allreduce-kernel-execution.md) | 已覆盖 |
| M17 | Symmetric path | Symmetric kernel 如何选择、fallback 和同步？ | `ncclSymkAvailable`、LSA barrier、LL A2A | `src/scheduler/symmetric_sched.cc`、`src/device/symmetric/` | [`allreduce-kernel-execution.md`](allreduce-kernel-execution.md) | 已覆盖 |
| M18 | Completion/reclaim | kernel、proxy、FIFO、plan 何时完成和回收？ | `workFifoProduced/Consumed`、两类 callback queue、`reclaimPlan` | `src/enqueue.cc`、`src/include/comm.h`、`src/proxy.cc` | [`completion-reclaim-profiling.md`](completion-reclaim-profiling.md) | 专题覆盖 |
| M19 | Stream/order/Graph | 多 stream、clique、CUDA Graph 如何保持顺序？ | strong stream、launch order、persistent plan | `src/group.cc`、`src/enqueue.cc` | [`planner-group-aggregation.md`](planner-group-aggregation.md) | 已覆盖 |
| M20 | Abort/error | hang、abort、异步错误如何传播？ | `abortFlag`、`ncclGroupError`、async error | `src/device/primitives.h`、`src/group.cc` | 待建立专题 | 待深入 |
| M21 | Profiling/observability | API、task、kernel、proxy 和 DCCL timing 如何记录？ | `ncclProfilerEventMask`、`workStarted/workCompleted`、`dcclSpscMetricEvent` | `src/plugin/profiler.cc`、`src/device/common.h`、`src/trace/` | [`completion-reclaim-profiling.md`](completion-reclaim-profiling.md) | 专题覆盖 |
| M22 | Generated device table | function row、primary function、specialized kernel 如何生成？ | `ncclDevFuncId`、`device/generate.py`、function tables | `src/include/device.h`、`src/device/generate.py` | [`protocols-primitives-kernel-launch.md`](protocols-primitives-kernel-launch.md) | 已覆盖 |

---

## 3. 四项当前重点机制的掌握目标

### M05/M06：算法选择与 chunking

掌握后应能回答：

- 当前 collective 的候选算法/协议为什么被过滤或保留？
- cost table 中 bandwidth、latency 和 `numPipeOps` 如何影响结果？
- tuner plugin、`NCCL_ALGO`、`NCCL_PROTO` 和 registration 如何改变选择？
- 为什么最终的 `nMaxChannels`、`nWarps`、chunk size 与算法/协议绑定？
- `calcCollChunking` 如何生成 pattern、proxy nsteps 和 direct flags？

### M07：Protocol

掌握后应能回答：

- SIMPLE 的 `SlicePerChunk`/`StepPerSlice` 如何划分数据？
- LL/LL128 的 line flag 如何表示 data-ready？
- `NCCL_STEPS`、`connFifo.size`、head、tail 分别表示什么？
- 为什么 LL128 受架构、path 和对齐约束？
- 同一个协议在 collective、P2P、NET registration 和 direct path 中有什么差异？

### M08：Primitives

掌握后应能回答：

- `Fan`、`Direct`、`P2p`、`RedOp` 和 `Proto` 分别如何影响实例化？
- wait、worker、post 线程为什么分离？
- `directRecvReduceDirectSend` 如何映射为具体的 reduce/copy/fence？
- direct pointer exchange、registration 和 destructor wait 如何保证 buffer 安全？
- abort 如何从 spin wait 传播到 kernel 退出？

### M11/M12/M13：Kernel plan 与发射

掌握后应能回答：

- 一个 task 如何变成 `ncclDevWorkColl`、batch 和 kernel args？
- 为什么要有 Args/Fifo/Persistent 三种 work storage？
- `channelMask`、`grid.x`、`blockIdx.x` 和 `batchZero` 如何对应？
- specialized kernel、generic kernel 和 `ncclDevFuncTable` 如何协作？
- host stream callback、proxy op、multi-round 和 CUDA Graph 如何参与 launch ordering？

---

## 4. 建议的逐机制学习顺序

```text
第一阶段：M01 Communicator → M02 Topology
第二阶段：M03 Task/Planner → M04 Group
第三阶段：M05 Algorithm selection → M06 Chunking
第四阶段：M07 Protocol → M08 Primitives → M09 Head/Tail → M10 Direct/Registration
第五阶段：M11 Plan → M12 Kernel launch → M13 Device dispatch
第六阶段：M14 Algorithms → M15 Transport → M16 Proxy
第七阶段：M17 Symmetric → M19 Stream/Graph → M18 Completion
第八阶段：M20 Abort/Error → M21 Profiling → M22 Generated tables
```

对于当前重点，建议先阅读：

编号是机制目录的稳定标识，不强制等同于阅读顺序；完成边界依赖 stream/graph 生命周期，因此建议先读 M19，再读 M18。

1. [`protocols-primitives-kernel-launch.md`](protocols-primitives-kernel-launch.md) 的总图；
2. §4 算法选择与 §5 Kernel 发射；
3. §2 Protocol；
4. §3 Primitives；
5. 最后回看 [`allreduce-kernel-execution.md`](allreduce-kernel-execution.md) 的 AllReduce 算法和 proxy 执行细节。

---

## 5. 统一记录模板

后续为每个机制建立专题时，建议固定回答以下问题：

| 维度 | 要记录的内容 |
|---|---|
| 目标 | 该机制解决什么通信/调度问题？ |
| 输入 | 谁创建它？输入参数、拓扑、buffer 或状态是什么？ |
| 核心状态 | 关键 struct、queue、counter、flag、pointer 是什么？ |
| Host 流程 | 从哪个函数进入，经过哪些阶段？ |
| Device/Proxy 流程 | GPU 或 proxy 实际执行哪些步骤？ |
| 同步 | 谁等待谁？head/tail/flag/event/barrier 的方向是什么？ |
| 性能 | 哪些参数影响延迟、带宽、并发度或内存访问？ |
| 完成 | work、kernel、proxy、plan 分别何时算完成？ |
| 失败边界 | 哪些配置会 fallback、报错、abort 或 hang？ |
| 验证 | 最小源码路径、日志、测试或 profiling 方法是什么？ |

这个模板能避免只记录“调用链”，而遗漏 NCCL 最重要的状态机、同步不变量和性能决策。

---

## 6. 当前重点文档与源码索引

| 主题 | 推荐入口 |
|---|---|
| 用户线程到 plan/kernel | [`user-thread-enqueue-kernel-flow.md`](user-thread-enqueue-kernel-flow.md) |
| Planner/group、多 API/stream/comm | [`planner-group-aggregation.md`](planner-group-aggregation.md) |
| Protocol、Primitives、算法选择、kernel 发射 | [`protocols-primitives-kernel-launch.md`](protocols-primitives-kernel-launch.md) |
| Chunking、FIFO、Direct/registration | [`chunking-pattern-fifo-registration.md`](chunking-pattern-fifo-registration.md) |
| AllReduce 算法、device/proxy 执行 | [`allreduce-kernel-execution.md`](allreduce-kernel-execution.md) |
| Completion/reclaim、Profiler/DCCL | [`completion-reclaim-profiling.md`](completion-reclaim-profiling.md) |
| Host planner/task/plan | [`enqueue.cc`](../nccl/src/enqueue.cc)、[`comm.h`](../nccl/src/include/comm.h) |
| Device dispatch/work | [`common.h`](../nccl/src/device/common.h)、[`device.h`](../nccl/src/include/device.h) |
| Protocol/Primitives | [`primitives.h`](../nccl/src/device/primitives.h)、[`prims_simple.h`](../nccl/src/device/prims_simple.h) |
| Tuning/cost model | [`tuning.cc`](../nccl/src/graph/tuning.cc) |

---

## 7. 使用说明

每次深入一个机制时，先从本表找到它在整体依赖图中的上游和下游，再打开对应源码入口。尤其要区分：

- host 选择与 device 执行；
- device data plane 与 proxy control plane；
- algorithm、protocol、transport 三个不同抽象层；
- kernel 返回、proxy 完成、plan 回收三个不同生命周期边界。

只要这四组边界没有混淆，后续无论研究 Ring、LL128、NET registration、NVLS 还是 CUDA Graph，都可以沿同一套方法继续展开。
