# NCCL M04：Planner 与 group——多个 API、stream、communicator 如何聚合

本文基于当前仓库 `nccl/src/` 的实现，聚焦 NCCL enqueue 系统中四个容易混淆的层次：用户 API 如何变成 task、一个 communicator 如何用 Planner 聚合任务、多个 stream 如何被收敛到一次发射、以及多个 communicator 如何通过 group 协同执行。这里的“聚合”不是把所有操作变成一个全局 task，而是先按 communicator 分别建模，再由 group 统一准备、连接、注册和发射。

## 1. 总体结论：group 是执行边界，Planner 是 communicator 内部的任务账本

可以把一次显式或隐式 group 的主链路概括为：

```text
nccl API
  -> ncclEnqueueCheck
  -> taskAppend
  -> 当前 comm 的 ncclKernelPlanner
       ├─ collective sorter / P2P peer queues / bcast queues
       ├─ stream list + capturingGraph
       └─ CE、RMA、symmetric-register 等专用队列
  -> ncclGroupEndInternal（最外层 group 结束时）
  -> 每个 comm：prepare -> preconnect -> registration -> plan/work
  -> 每个 intra-process clique：多轮 kernel / CE / RMA launch
```

关键边界如下：

| 层次 | 所属对象 | 作用 | 是否跨 communicator 融合 |
|---|---|---|---|
| 用户调用 | `ncclInfo` | 描述一次 API 的 buffer、count、dtype、op、root、stream | 不适用 |
| task 累积 | `comm->planner` | 保存该 communicator 在本次 group 中的所有操作 | 否 |
| kernel plan | `ncclKernelPlan` | 按 work/FIFO/args 预算切分设备工作 | 否 |
| group | TLS `ncclGroupCommHead[]` | 把多个 comm 纳入同一准备与发射事务 | 只协同，不共享 task |
| clique | `intraComm0` 相同的一组 comm | 在同一进程内做 barrier 和多轮 launch 协调 | 以 clique 为单位协调 |

因此，同一个 group 中的多个 communicator 不会共享一个 Planner；每个 `comm` 仍然拥有自己的 task、channel、plan 和 stream 列表。group 负责保证这些 communicator 在正确的顺序和同步条件下完成准备与发射。

## 2. API 如何进入 Planner

### 2.1 每个 API 都有隐式 group 包装

`ncclEnqueueCheck()` 首先执行 `ncclGroupStartInternal()`，完成参数检查和 `taskAppend()` 后执行 `ncclGroupEndInternal()`。因此：

- 没有显式 `ncclGroupStart/End` 时，一次 API 调用自身形成深度为 1 的隐式 group，API 返回前完成 enqueue/launch 准备；这不等于通信已经完成。
- 在显式 group 内，隐式 start/end 只增加和减少嵌套深度；只有最外层 `ncclGroupEnd()` 将深度降到 0 时，才真正执行 group launch。
- 若前一个 API 失败，错误保存到线程本地 `ncclGroupError`，最外层结束时统一处理，避免只清理半个 group。

### 2.2 `taskAppend()` 的分流

| 用户操作 | 主要入口 | Planner 中的落点 | 说明 |
|---|---|---|---|
| AllReduce、AllGather、ReduceScatter、Reduce、Broadcast 等普通 collective | `collTaskAppend` | `collSorter` | 先按 traffic size 粗排，之后按 `(func, op, datatype)` 分桶、选算法协议 |
| Send/Recv | `p2pTaskAppend` | `peers[peer].sendQueue/recvQueue` | 按 peer 保存，后续按 P2P schedule 的 round 取出 |
| Broadcast 的 allgatherv 特殊路径 | `collTaskAppend` | `peers[root].bcastQueue` | 多 root 时先保留为 bcast task；只有一个 root 时可转为普通 collective task |
| Copy Engine collective | `ceCollTaskAppend` | `collCeTaskQueue` | 还可能加入 `SymRegister` 类型的 group |
| PutSignal/Signal/WaitSignal | `rmaTaskAppend` | 按 context 的 `rmaTaskQueues` | RMA CE 初始化也可能加入 `SymRegister` group |
| AllToAll/Gather/Scatter fallback | `taskAppend` | 展开为若干 P2P task | 只有满足条件时才走 kernel collective 的专用路径 |
| 单 rank collective | `ncclLaunchOneRank` | 不进入 Planner | 直接在用户 stream 上发起本地 copy/reduction |
| 空 collective | `taskAppend` | 丢弃 | `count == 0` 不创建通信 task |

在 task 创建前，`collTaskAppend` 和 `p2pTaskAppend` 都会执行 `ncclGroupCommJoin(comm, ncclGroupTaskTypeCollective)`，然后记录 stream 和 CUDA Graph 状态。`comm->memScoped` 覆盖 stream/work/batch 等本轮临时存储；task、plan 和 proxy 描述符来自各自的 memory pool，并由 plan/reclaimer 管理，不能把两类生命周期混为一谈。

## 3. 一个 communicator 内的 Planner 如何聚合

`ncclKernelPlanner` 分为三段状态：

1. **Accumulation state**：`peers[]`、`collSorter`、各类 task 数量、`streams`、`capturingGraph`。
2. **Plan construction state**：`collTaskQueue`、`collWorkQueue`、`wipPlan.channels[]`、每个 channel 的 work batch 和 proxy-op queue。
3. **Launch state**：`planQueue` 和 `unlaunchedPlansHead`。

普通 collective 不是简单地按调用顺序追加到一个数组：

1. `collTaskAppend` 计算 `trafficBytes`，把 task 放入 `collSorter`。
2. `ncclPrepareTasks` 取出粗略按大小降序排列的 task，按 `(function, reduction op, datatype)` 分桶。
3. 同一桶中，约按相近规模（代码中为不超过首 task 的 4 倍）构造临时聚合样本，用于 `ncclGetAlgoInfo`；这一步是算法/协议调优的估算，不是把多个用户 collective 语义合并成一次 collective。
4. 得到 `algorithm`、`protocol`、`nMaxChannels`、`nWarps`、`devFuncId` 后，原始 task 逐个写回这些结果，再按 CollNet/NVLS/普通路径分 bin。
5. 完成 buffer registration 后，每个 task 生成对应的 `ncclDevWorkColl` 或注册版本，放入 `collWorkQueue`。
6. `scheduleCollTasksToPlan` 按 kernel args、work buffer/FIFO 预算取 task，分配 channel、chunk、proxy operation 和 work batch；预算不足时切出新的 `ncclKernelPlan`。

因此“多个 API 被融合”通常表示多个 task 的 device work 被放入同一个 kernel plan，或者在同一个 group 中连续发射多个 plan；每个 task 的 buffer、count、op、算法和协议仍然独立保存。

P2P 有额外约束：Planner 按 peer 保存 send/recv 队列，`scheduleP2pTasksToPlan` 按 `p2pSchedule` 的 round 配对，并通过 `p2pEpoch` 防止不同 epoch 的 work batch 混合；同一 channel 的 batch 不能重复使用同一 P2P round，也受 `NCCL_MAX_DEV_WORK_P2P_PER_BATCH` 限制。这些限制是为了让各 rank 的 batching 一致，避免连接/step 不匹配导致 hang。

## 4. 多个 stream 如何聚合

### 4.1 记录阶段：stream list 而不是“选一个 stream”

`ncclPlannerSetCapturingGraph` 用 `streamRecent` 快速处理重复 stream；遇到新 stream 时，将其插入 `planner->streams` 链表，并查询该 stream 的 capture graph。约束是：

- 同一个 communicator 在一次 group 中使用的 stream 必须全部未捕获，或全部捕获于同一个 CUDA Graph。
- 若先后发现两个不同 graph，立即返回 `ncclInvalidUsage`。
- `capturingGraph` 属于整个 planner，不属于单个 task；因此不能把同一 group 的任务拆给不同 graph。

这也解释了为什么 `streams` 要保留完整链表：后续需要为所有参与过本次 group 的用户 stream 建立前后依赖。

### 4.2 发射阶段：链表头 stream 作为 launch stream

`ncclLaunchPrepare` 令 `planner->streams->stream` 成为 launch stream。`ncclPlannerSetCapturingGraph()` 对新 stream 采用头插，因此链表头通常是最近加入的唯一 stream，并不代表 API 调用顺序中的“第一个 stream”：

1. 对链表中的其他 user stream 记录 event。
2. 让 launch stream 等待这些 event，汇合所有 stream 在 group 之前的工作。
3. 让 launch stream 等待 communicator 的 strong `deviceStream`，并按需要等待进程级 `launchOrder`。
4. 准备 host callback/proxy 参数和 kernel plan。

发射完成后，`ncclLaunchFinish` 在 launch stream 上记录完成 event，让其他 user stream 等待该 event，再释放 strong stream。于是多个 stream 共享一批 NCCL 工作，但每个 stream 的前后依赖仍然被保留。这里的聚合是 stream dependency aggregation，不是把不同 CUDA stream 变成一个 CUDA stream。

### 4.3 CUDA Graph 与 persistent plan

当 `capturingGraph` 有效时，planner 的 `persistent` 被置为 true，plan 使用 persistent storage，并将 plan list 交给 CUDA Graph destructor 管理。未捕获路径使用 FIFO/args 存储；`hostStreamPlanTask()` 在 proxy op 已完成复制和发布后即可把 reclaimer 放入 `callbackQueue`，host-callback 路径中的入队甚至可能先于 CPU 发出 kernel launch。真正的 `reclaimPlan()` 由 communicator 主线程在本轮 launch 后的 safe point 消费，无需等待 GPU 或网络整体完成。不能把捕获和未捕获 stream 混在同一个 communicator group 中；`doLaunches()` 还会检查同一 clique 中是否出现部分 captured、部分 uncaptured，若出现则报 `ncclInvalidUsage`。

## 5. 多个 communicator 如何通过 group 聚合

### 5.1 两条线程本地链表、两种 group 类型

`ncclGroupCommHead[]` 和 `ncclGroupCommPreconnectHead` 是 thread-local 状态。`ncclGroupCommJoin` 对每个 communicator、每个 group type 只加入一次：

- `ncclGroupTaskTypeCollective`：承载 collective/P2P/CE/RMA 的通信任务。
- `ncclGroupTaskTypeSymRegister`：承载 symmetric registration 或 CE/RMA 初始化相关任务。

第一次加入 collective group 时，会 push `comm->memScoped`，清空本轮 planner 的运行状态，但保留 `peers` 数组和 RMA queue 存储；离开 group 时 pop scope。stream/work/batch 等 scoped 节点随 group scope 释放；task、plan 和 proxy 描述符则已经转移到 plan/reclaimer 生命周期，不依赖 stack pop。

### 5.2 排序规则：同 clique 相邻，不同 clique 按 commHash

`ncclComm` 初始化时根据同一全局 communicator 中各 rank 的 host/pid 信息计算 `intraComm0`、`intraRank`、`intraRanks`。位于同一进程、且属于同一全局 NCCL communicator 的多个本地 rank handle 共享 leader `intraComm0`，构成一个 intra-process clique；“由同一线程加入 group”只决定 TLS 链表归属，不会让任意两个 communicator 自动成为同一 clique。

加入 TLS group 链表时：

- 若找到相同 `intraComm0`，新 communicator 会插到该 clique 当前首成员之前，使 clique 始终连续；这不是一个稳定保持 communicator 首次出现顺序的队列，各 comm 内部的 task 顺序仍由各自 planner 保持。
- 新 clique 插入时按 `commHash` 升序排列。
- `groupNext[type]` 为真实指针时指向链表后继，`nullptr` 表示链表结束；地址值 `0x1` 是“尚未加入该类型 group”的 sentinel。

这种排列不是装饰性的：`doLaunches()` 依赖同 clique 成员连续出现，才能确定 clique 边界并进行 intra-process barrier。

### 5.3 group end 的真实执行顺序

最外层 `ncclGroupEndInternal` 将 thread-local 状态转移到 `ncclGroupJob`，之后有两条路径：

- blocking group：当前线程直接执行 `groupLaunch`；
- nonblocking group：创建后台 async job，API 可返回 `ncclInProgress`，后续由 communicator 的 async-error/group-job 机制完成或报告错误。

`groupLaunch` 的主要阶段是：

1. 执行显式 P2P preconnect 和异步初始化。
2. 处理 `SymRegister` 类型的 clique 任务。
3. 对 collective 类型按 clique 调用 `ncclPrepareTasks`，必要时做 runtime connection/preconnect。
4. 对所有 collective communicator 调用 `ncclTasksRegAndEnqueue`，完成 registration 和 work node 构造。
5. 调用 `doLaunches()`，逐个 clique 进行 plan 准备和多轮 kernel/CE/RMA 发射。
6. drain 本轮 async jobs，周期性轮询 callback reclaimer，清理 planner/group 状态并 pop 每个 communicator 的 `memScoped`。非 persistent plan 及其 pool 对象可稍后由 `callbackQueue` 回收；proxy progress 已取得独立副本，不随 group scope 一起结束。

`doLaunches()` 对一个 clique 先让所有成员执行 `ncclLaunchPrepare`，然后反复取出各自的 `unlaunchedPlansHead`。在 `NCCL_LAUNCH_MODE=GROUP` 下，每一轮通过 intra-process barrier 判断是否还存在后续 plan；否则依据各 comm 是否还有未发射 plan 推进。每个 comm 的 plan 仍在自己的 CUDA device 和 launch stream 上执行，group 只负责把各成员的轮次对齐。

## 6. 两个典型例子

### 例 1：同一 comm、两个 API、两个 stream

应用在 stream A 上提交 AllReduce，在 stream B 上提交 AllGather，并用显式 group 包住。两个调用分别生成 task，但共用一个 `comm->planner`；planner 记录 A/B 两个 stream，算法选择和 work 生成后可能形成一个或多个 plan。launch stream 等待 A/B 的既有工作，发射结束时记录的 stream dependency event 再让 A/B 等待该批 NCCL 工作。`ncclGroupEnd` 只保证这些操作被正确 enqueue，不保证 CPU 端等待通信完成。

### 例 2：一个 group、同一进程的两个 comm

同一进程持有某个全局 NCCL communicator 的两个本地 rank handle：GPU0 的 `comm0` 和 GPU1 的 `comm1`。两者分别拥有 planner 和 plan，但因共享 `intraComm0` 会被放入同一个 clique。group end 会先分别 prepare，再按 barrier 对齐多轮 launch。若一个 comm 因 work budget 产生三个 plan，另一个产生一个 plan，GROUP launch mode 会让后者参与空轮次同步，避免 clique 内 launch 次序失配。若 `comm0`、`comm1` 来自两个互不相关的 communicator，则即使由同一线程提交也不会因此组成 clique。

## 7. 代码审阅得到的关键不变量

- group 的聚合单位是 communicator 和执行事务，不是跨 communicator 的 task 合并。
- planner 的 task 只在最外层 group end 后进入 prepare；中途不能假设已经完成 algorithm selection、registration 或 kernel launch。
- 同一 communicator 的所有 stream 必须处于相同 capture 状态，且 captured 时必须来自同一 graph。
- 同 clique 的 communicator 必须连续排列；`groupNext[]` 的 sentinel 不能当作普通链表指针解引用。
- work/FIFO budget、P2P epoch/round、channel mask 会把一个 group 切成多个 plan；“一次 group”不等于“一次 kernel”。
- collective、CE、RMA、symmetric-register 是不同调度分支，不能只用“collective task queue”解释整个 group。

## 8. 代码导航

| 主题 | 主要位置 |
|---|---|
| API 入队与 task 分流 | `nccl/src/enqueue.cc:2898` 附近的 `taskAppend`、`ncclEnqueueCheck` |
| stream/capture 聚合 | `nccl/src/enqueue.cc:2464` 的 `ncclPlannerSetCapturingGraph` |
| Planner 数据结构 | `nccl/src/include/comm.h` 的 `ncclKernelPlanner`、`ncclTask*`、`ncclKernelPlan` |
| group join/leave | `nccl/src/include/group.h` |
| group end 与 clique launch | `nccl/src/group.cc` 的 `groupLaunch`、`doLaunches`、`ncclGroupEndInternal` |
| prepare、registration、plan 切分 | `nccl/src/enqueue.cc` 的 `ncclPrepareTasks`、`ncclTasksRegAndEnqueue`、`ncclLaunchPrepare` |
| user stream 前后依赖 | `nccl/src/enqueue.cc` 的 `ncclLaunchPrepare`、`ncclLaunchFinish` |
