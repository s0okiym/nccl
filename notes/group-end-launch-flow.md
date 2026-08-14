# NCCL Group-End → Kernel Launch 流程深度解析（以 AllReduce 为例）

> 本文基于 **NCCL 2.30.7** 源码（写作时的仓库状态），逐环节拆解「用户调用一个集合通信 API」到「GPU 内核真正完成这次通信」之间，
> 主机端（Host）、设备端（Device）、代理线程（Proxy）三层各自做了什么、通过什么数据结构握手。
> 全程以一次 **AllReduce** 作为贯穿例子。所有结论均带 `文件:行号`，可直接跳转核对。
>
> 阅读前置：建议先看 `docs/dev_guide/nccl_internals.md`。本文聚焦「group end 到 launch」这一段的细枝末节。
>
> ⚠️ **时效说明（2026-08-14）**：仓库其后已升级到 **NCCL 2.31.2**（上游 "Enqueue Overhaul" 合并）：
> `src/enqueue.cc` 移至 **`src/enqueue/enqueue.cc`**（并新增 `src/enqueue/{task_prep,task_sched}/`、`mgmt_task_enq.cc`）；
> `groupLaunch` 拆出 `groupLaunchLegacy`（`group.cc:748`）/ `groupLaunchEnqueueRearch`（`group.cc:912`）两个变体，
> 由 `groupLaunch`（`group.cc:1018`）分派。本文行号对应 2.30.7，按新树阅读时需换算路径并偏移行号
> （例：`ncclEnqueueCheck` 3124→3383、`ncclLaunchPrepare` 1568→1661、`ncclLaunchKernel` 1753→1852、
> `ncclGroupEndInternal` 766→1026；proxy.cc 基本未动）。流程主体（task→plan→work/proxyOp 四层抽象）仍一致。

---

## 0. 一张图看懂全流程

```
 用户线程 (Host)
 ────────────────────────────────────────────────────────────────────────────
  ncclAllReduce()                              collectives.cc:168
      │  构造 ncclInfo
      ▼
  ncclEnqueueCheck(&info)                       enqueue.cc:3124
      │  ncclGroupStartInternal()   ← 即便单个 op 也会隐式开一个 depth=1 的 group
      │  taskAppend()  → collTaskAppend()       enqueue.cc:3014 / 2690
      │      └─ 分配 ncclTaskColl，塞进 comm->planner.collSorter（按大小降序桶排）
      │  ncclGroupEndInternal()                 ← depth 减到 0 → 触发整条 launch 流水
      ▼
  ncclGroupEndInternal() → groupLaunch()        group.cc:766 / 598
      │
      ├─ ① ncclPrepareTasks()                   enqueue.cc:363
      │      选算法/协议(Ring/Tree/NVLS…×LL/LL128/SIMPLE)、分箱、构造 ncclDevWorkColl
      ├─ ② ncclTasksRegAndEnqueue()             enqueue.cc:302
      │      注册 buffer、把 work 节点挂到 planner.collWorkQueue
      └─ ③ doLaunches()                         group.cc:309
            └─ 对每个 comm：ncclLaunchPrepare()  enqueue.cc:1568
                 把 task 切成若干 ncclKernelPlan（一个 plan = 一次 kernel launch）
                 scheduleCollTasksToPlan()  ← 按 channel 切数据 + 算 chunkSize + 建 proxyOp
                 finishPlan()               ← 把 work batch 装进 kernelArgs、合并 proxyOp
               然后 per-plan：
                 ncclLaunchKernelBefore_NoUncapturedCuda() → uploadWork()      ← 写 work FIFO
                 ncclLaunchKernel()           → cuLaunchKernelEx()            ← 真正起 GPU kernel
                 ncclLaunchKernelAfter_NoCuda() → hostStreamPlanTask()
                       └─ uploadProxyOps() → ncclProxySaveOp()  ← 把 proxyOp 交给 proxy 线程
                          ncclProxyStart()                       ← 唤醒 proxy 线程
               最后一轮：ncclLaunchFinish()  ← 收尾、stream 依赖、FIFO 回收

 GPU 内核 (Device)              Proxy 线程 (CPU, 每 GPU 一个)
 ─────────────────────────      ────────────────────────────────
 ncclDevKernel_##suffix          ncclProxyProgress()  proxy.cc:954
  └─ ncclKernelMain()            死循环：progressOps() → transport.proxyProgress
      blockIdx.x → channelId            (net_ib: ibv_post_send/recv，推 completion)
      loadWorkBatchToShmem()     通过 connection FIFO(ncclConnFifo) 的 head/tail 计数器
      RunWorkColl<AllReduce,Ring 与 GPU 握手：GPU 写 tail、spin peer head；
        ,Simple>::run()           proxy 把数据在网络/SHM 间搬、推进 head
      └─ runRing():              ─────────────────────────────────────────
          scatter-reduce (nR-1 步) + allgather (nR-1 步)
```

**核心心智模型（四层抽象，记住这四个词就抓住了主线）**：

> **task**（用户的一次集合通信请求）→ **plan**（一次 kernel launch，含若干 task）→
> **work**（device 侧执行单元，写在 work FIFO 里）/ **proxyOp**（proxy 侧执行单元，交给 proxy 线程）。

一次 `ncclGroupEnd` 可能把若干 task 打包成**多个 plan**（即多次 kernel launch）；每个 plan 里，
**work 描述「GPU 要干什么」**，**proxyOp 描述「CPU 要替 GPU 干什么（网络收发）」**。

---

## 1. 关键数据结构（先认人，再读流程）

定义集中在 `src/include/comm.h` 和 `src/include/proxy.h`。

| 结构 | 位置 | 角色 |
|---|---|---|
| `ncclInfo` | `collectives.h` | 一次 API 调用的入参快照（func/send/recv/count/dtype/op/stream…）|
| `ncclTaskColl` | `comm.h:193` | 一个集合通信 task（allreduce/bcast…），含 algo/protocol/count/sendbuff…|
| `ncclKernelPlanner` | `comm.h:429` | 每个 comm 的「任务累积器 + plan 构造器」。group 期间收 task，group end 时切 plan |
| `ncclKernelPlan` | `comm.h:307` | **一次 kernel launch 的全部信息**：channelMask、kernelFn、kernelArgs、work/proxyOp 队列 |
| `ncclDevWorkColl` | device header | device 侧「这个 channel 上这次 coll 的执行参数」：channelLo/Hi、countLo/Mid/Hi、chunkGrains…|
| `ncclDevWorkBatch` | device header | 把若干 work 打包成一组的「批次」容器，offsetBitset 标记哪些 work 在内 |
| `ncclProxyOp` | `proxy.h` | proxy 侧执行单元：pattern/chunkSize/nsteps/loopSize/sendbuff…|
| work FIFO | `comm->workFifoBuf` | host→device 单向环：host 往里写 work，GPU kernel 读出来执行 |
| connection FIFO (`ncclConnFifo`) | `channel.*.conn` | device↔proxy 双向环（深度 `NCCL_STEPS=8`），靠 head/tail 计数器 + fence 握手 |

**`ncclKernelPlanner` 的几个关键队列**（`comm.h:466-501`），后面会反复出现：

- `collSorter` —— group 期间收 task 的入口（按 trafficBytes 降序桶排，`comm.h:361`）。
- `collTaskQueue` —— `ncclPrepareTasks` 把选好算法/协议的 task 串成的最终链表。
- `collWorkQueue` / `tmpCollWorkQueue` —— 与 task 一一对应的 `ncclDevWorkColl` work 节点。
- `wipPlan.channels[c]` —— 正在构造的 plan 中，第 c 个 channel 的「work batch 队列 + proxyOp 队列」。
- `planQueue` / `unlaunchedPlansHead` —— 构造好的 plan 链表，等待 `doLaunches` 逐个发射。

---

## 2. Layer 1：从 `ncclAllReduce` 到 task 收集

### 2.1 API 包装层（极薄）

`ncclAllReduce` 只做两件事：构造 `ncclInfo`，然后调 `ncclEnqueueCheck`（`collectives.cc:168-178`）：

```c
struct ncclInfo info = { ncclFuncAllReduce, "AllReduce", sendbuff, recvbuff, count,
                         datatype, op, 0, comm, stream,
                         ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
return ncclEnqueueCheck(&info);
```

### 2.2 `ncclEnqueueCheck`：隐式 group + 任务挂载（`enqueue.cc:3124`）

这是理解全流程的**第一把钥匙**：

```c
ncclGroupStartInternal();          // 3137  depth: 0→1（若原本不在 group）
...
taskAppend(info->comm, info);      // 3159  把这次 allreduce 挂到 planner
...
ncclGroupEndInternal();            // 3164  depth: 1→0 → 触发完整 launch
```

- `ncclGroupStartInternal()` / `ncclGroupEndInternal()` 操作的是线程局部变量 `ncclGroupDepth`（`group.cc:27`）。
- **关键洞察**：即使你**没有显式** `ncclGroupStart()`，单个 `ncclAllReduce()` 也会被包进一个 depth=1 的隐式 group，于是在 `ncclGroupEndInternal` 里 depth 减到 0，**立即触发整条 launch 流水**。
- 显式 `ncclGroupStart()…ncclAllReduce()…ncclGroupEnd()` 时：用户 `GroupStart` 让 depth=1；每个 `ncclAllReduce` 内部 start→2、end→1（**不**触发，只是累积 task）；最后用户 `GroupEnd` 让 depth 1→0，**一次性** launch 所有累积的 task。

### 2.3 `taskAppend` → `collTaskAppend`：构造 task（`enqueue.cc:3014 / 2690`）

`taskAppend` 按 `info->coll` 分流：send/recv → `p2pTaskAppend`；PutSignal/Signal/Wait → `rmaTaskAppend`；
普通集合通信（allreduce/allgather/…）→ `collTaskAppend`。对 allreduce，还会先 `hostToDevRedOp()` 把
`ncclRedOp_t` 转成 device 友好的 `ncclDevRedOpFull`（`enqueue.cc:3037`）。

`collTaskAppend`（`enqueue.cc:2690`）做的事：

1. **`ncclGroupCommJoin(comm, Collective)`**（`2694`）：把该 comm 链进线程局部的 group 通信头
   `ncclGroupCommHead[ncclGroupTaskTypeCollective]`，这样后面 `groupLaunch` 才能遍历到它。
2. 从内存池分配一个 `ncclTaskColl`（`2725`），填入 `func=AllReduce`、send/recv buff、count、datatype、
   `opDev`、`chunkSteps=ALLREDUCE_CHUNKSTEPS`、`sliceSteps`（`2741-2742`）。
3. 计算通信量：`t->trafficBytes = count*elemSize * ncclFuncTrafficPerByte(AllReduce, nRanks)`（`2738`）。
   trafficPerByte 体现了「每字节数据在网络上要传几遍」（allreduce≈2·(nR-1)/n）。
4. **插入 `planner->collSorter`**（`ncclTaskCollSorterInsert`）——按大小降序桶排，`comm.h:381`。
   排序的目的：后面优先把大 op 切到多 channel，小 op 复用剩余 channel。
5. `planner->nTasksColl += 1`。

> 至此，task 已经躺在 planner 里，但**还没有选算法、没有切 channel、没有起 kernel**。真正的活儿在 group end。

---

## 3. Layer 2 入口：`ncclGroupEnd` → `groupLaunch`

### 3.1 `ncclGroupEndInternal`（`group.cc:766`）

```c
if ((--ncclGroupDepth) > 0) goto exit;     // 788  还在嵌套层，直接返回
...
NEW_NOTHROW_GOTO(groupJob, ncclGroupJob, ...) // 811  快照线程局部的 group 状态
...                                           //     （comm 头、asyncJobs、preconnect 头）
if (ncclGroupBlocking == 0) { ... 起线程跑 groupLaunchNonBlocking ... }  // 非阻塞
else { groupLaunch(&groupJob->base, simInfoPtr); }                       // 阻塞（默认）
```

- 阻塞（默认 `ncclGroupBlocking==1`）：直接在当前线程跑 `groupLaunch`。
- 非阻塞：`STDTHREADCREATE` 一个线程跑 `groupLaunchNonBlocking`，函数立即返回 `ncclInProgress`。

### 3.2 `groupLaunch`（`group.cc:598`）—— group end 的心脏

按顺序做四件事（allreduce 走的是第 ③④ 步，①② 视情况）：

| 步 | 代码 | 作用 |
|---|---|---|
| ① 预连接 | `asyncJobLaunch(asyncJobsMain)` `626` | P2P 预连接、其它已排队 async job（如 split/RMA 注册）|
| ② 对称注册 | `ncclCommGroupRegisterSymmetric` `640` | symmetric memory 注册任务（allreduce 通常不涉及）|
| ③ **准备 + 连接** | `ncclPrepareTasksAndCollPreconnect(comm)` `674` | 调 `ncclPrepareTasks`；按需 runtime 连接 ring/tree/nvls/collnet |
| ④ **注册 + 入队** | `ncclTasksRegAndEnqueue(comm)` `690` | 注册 buffer、构造 work 节点 |
| ⑤ **发射** | `doLaunches(head)` `723` | 真正构造 plan 并 `cuLaunchKernelEx` |

> 注意 ③④ 是「**每个 comm 依次**」做的（`do{...}while(comm)`），⑤ `doLaunches` 则按 **clique**（同一 `intraComm0`
> 的兄弟 comm 们）成组发射，组内用 barrier 同步（`ncclLaunchModeGroup`）。

---

## 4. `ncclPrepareTasks`：算法/协议选择 + 构造 work（`enqueue.cc:363`）

这一步把「裸 task」变成「选好算法、可调度的 task」，并为每个 task 配一个 `ncclDevWorkColl`。

### 4.1 三次遍历

1. **按大小降序出队**：`ncclTaskCollSorterDequeueAll(&planner->collSorter)`（`393`）。
2. **按 (func, op, datatype) 分箱**（`406-416`）：相同 (fn,op,ty) 的 task 串成 LIFO，便于融合。
   > 对 allreduce，(`AllReduce`, op, dtype) 相同的多个 allreduce 会被聚到一起，后续可共享 channel 规划。
3. **逐箱选算法/协议**（`421-471`）：
   - `ncclGetCollNetSupport`、判定 nvls 是否支持（`424-426`）。
   - 把大小相近（4× 内）的 task 聚合成 `agg`，调 **`ncclGetAlgoInfo(comm, &agg, …)`**（`441`）→ 选定
     **algorithm**（`Ring`/`Tree`/`NVLS`/`NVLS_TREE`/`CollNet`/`PAT`）和 **protocol**（`LL`/`LL128`/`SIMPLE`），
     并算出 `nMaxChannels`、`nWarps`。选择依据来自 `topoGetAlgoInfo`（`enqueue.cc:2028`）里的带宽/延迟表
     （`comm->bandwidths[fn][algo][proto]`、`comm->latencies`，这些在 init 时的 graph 搜索中算好）。
   - 计算 `devFuncId = ncclDevFuncId(func, op, dtype, algo, protocol)`（`442`）——这是后面
     **选哪个特化 kernel** 的索引。
   - 回填到每个 task：`algorithm/protocol/nMaxChannels/nWarps/devFuncId`，并按 `isCollnet × isNvls`
     分进 `collBins[2][2]`（`467`）。
4. **拼成最终链表**（`476-480`）：`collBins` 顺序拼进 `planner->collTaskQueue`。
   collnet 作外层维度（因为它影响 channel 切分方式）。
5. **构造 work + 判定连接**（`487-547`）：再遍历一遍，注册 NVLS buffer，为每个 task 构造
   `ncclDevWorkColl`（填 send/recv buff、nWarps、redOpArg、oneNode 等），挂到 `tmpCollWorkQueue`；
   若 `comm->runtimeConn` 且该算法 channel 还没连，置 `algoNeedConnect[algo]=*needConnect=true`
   （`495-506`）——供下一步 preconnect。

### 4.2 小结：到此为止，每个 task 都有了 algo/proto/devFuncId，并配了一个 work 节点；但还没切 channel、还没算 chunkSize。

---

## 5. `ncclTasksRegAndEnqueue`：注册 buffer + 串 work（`enqueue.cc:302`）

- 对 `planner->collTaskQueue` 每个 task：
  - `ncclRegisterCollBuffers()`（`318`）：按 `task->regBufType` 注册 IPC / net / NVLS buffer 句柄，
    产出 `regBufSend/Recv[]`（供 device 直接寻址或供 proxy 网络注册）。
  - 填一个 `ncclDevWorkColl devWork`（`320-336`），并包成 `ncclWorkList` 节点（NVLS 用
    `ncclDevWorkCollReg`），挂到 `planner->collWorkQueue`（`354`）。
- 注意 NVLS/NVLS_TREE 的 work 节点在 4.1 第 5 步已先进 `tmpCollWorkQueue`，这里直接 `goto next`（`315-316`）。

> 现在 `collTaskQueue`（task）与 `collWorkQueue`（work）**一一对应、同序**，下一步把它们一起切进 plan。

---

## 6. `doLaunches` → `ncclLaunchPrepare`：把 task 切成 plan（`group.cc:309 / enqueue.cc:1568`）

### 6.1 `doLaunches` 的发射节奏（`group.cc:309`）

```
对每个 clique（同 intraComm0 的 comm 集合）:
  对 clique 内每个 comm: ncclLaunchPrepare(comm)   ← 构造所有 plan，挂到 planQueue
  多轮循环（useBarrier 时用 intra-barrier 对齐各 comm 进度）:
    每个 comm 弹出 unlaunchedPlansHead 的下一个 plan:
      ncclLaunchKernelBefore_NoUncapturedCuda(comm, plan)   ← uploadWork
      ncclLaunchKernel(comm, plan)                          ← cuLaunchKernelEx
      ncclLaunchKernelAfter_NoCuda(comm, plan)              ← hostStreamPlanTask（proxy ops）
    最后一轮: ncclLaunchFinish(comm)                        ← 收尾
```

「多轮」是因为**一个 group 可能产生多个 plan（多个 kernel）**——当 work 总量超过单个 kernel 的 args/FIFO 预算时，
`ncclLaunchPrepare` 会把 task 切成多个 plan，`doLaunches` 逐个发射。

### 6.2 `ncclLaunchPrepare`：plan 构造主循环（`enqueue.cc:1568`）

```c
do {
  memset(&planner->wipPlan, 0, ...);                 // 1584 清空「正在构造的 plan」
  plan = pool.alloc(ncclKernelPlan);                 // 1586 从池里拿一个空 plan
  plan->workStorageType = persistent ? Persistent : Fifo;  // 1592
  ...
  scheduleCollTasksToPlan(comm, plan, &budget);      // 1617 把 task 灌进 plan，直到预算用满
  finishPlan(comm, plan);                            // 1628 装填 kernelArgs、合并 proxyOp
  if (plan->workBytes != 0) enqueue(planQueue, plan);// 1630
} while (还有 task 没灌完);                          // 1634 → 继续造下一个 plan
planner->unlaunchedPlansHead = planQueue.head;       // 1639
... stream 依赖设置（deviceStream / userStreams / hostStream）...  // 1643+
```

要点：
- 预算 `budget`（`1607-1610`）：`inArgsBytes`（能塞进 kernel args 的字节）、`outArgsBytes`（FIFO 里最多用一半，
  持久化/graph 模式放大到 1<<30）。`ncclTestBudget`（`294`）判断「这批 work 是否放得下」。**具体怎么算见 6.5**。
- **先 drain coll，再 drain bcast，最后 drain p2p**（`1616-1625`）——注释解释：p2p 不是集合操作，
  若先切 p2p，不同 rank 切 kernel 的位置会不一致，导致「最短 channel 优先」的 channel 选择器发散而死锁。
- 持久化（CUDA graph 捕获）走 `ncclDevWorkStorageTypePersistent`，否则走 `Fifo`；`finishPlan` 会尽量把
  work 降级塞进 `Args`（直接随 kernel 传，不走 FIFO，更快）。

### 6.3 `scheduleCollTasksToPlan`：按 channel 切数据 + 算 chunkSize + 建 proxyOp（`enqueue.cc:576`）

这是「**一次 allreduce 的数据怎么落到各个 channel 上**」的核心。以普通（非 collnet）路径（`655`）为例：

1. **估算本 plan 能装几个 coll**（`586-604`）：边遍历边累加 `workBytes`，`ncclTestBudget` 不够就 `plan_full`。
2. **算每 channel 的目标通信量**（`616-622`）：`trafficPerChannel = trafficBytes[kind] / nChannels[kind]`，
   再对齐 16 字节。`kind = 2*isCollnet + isNvls` 区分标准/NVLS channel 池。
3. **把 task 的数据切成 countLo / countMid / countHi**（`659-701`，即 `devWork->cbd`）：
   - 用 `cellSize`（≥32K 通信量对应的元素数）把 `task->count` 切成若干 cell；
   - 按 `trafficPerChannel` 把 cell 分配到连续若干 channel：第一个 channel 拿 `countLo`，中间若干 channel
     各拿 `countMid`（相等），最后一个 channel 拿 `countHi`（余数）。
   - 这就是 device 侧 `ncclCollCbdPart`（channel-balanced division）的来源——**每个 channel 处理一段连续的输入**。
4. **算 chunkSize（每个 ring step 传多少）**：对 Lo/Mid/Hi 各调一次 **`calcCollChunking`**（`725-738`），
   产出 `chunkGrainsLo/Mid/Hi = chunkSize/grainSize`。
5. **构造 proxyOp**（`720-809`）：每个用到 NET 的 channel 配一个 `ncclProxyOp`（Lo/Mid/Hi 三种），
   设 `channelId / opCount / task.coll / channelSize / loopOffset / ringAlgo`。
   对 ring+net 注册场景，会 `new RingARAlgorithm(...)`（`786`）——一个描述「ring allreduce 在网络上怎么走」的对象。
6. **`ncclAddWorkBatchToPlan`**（`803`，定义 `121`）：把一个 work 项塞进该 channel 的 work batch 队列。
   - 关键技巧：batch 用 `offsetBitset`（64 位）记录「这一批里包含哪几个 work」（按 workSize 对齐的位）；
     装不下/换 func/换 channel 就开新 batch，用 `nextExtends`/`nextJump` 把同一 channel 的多个 batch 串起来。
7. **`ncclAddProxyOpIfNeeded`**（`807`）：若该 channel 的连接需要 proxy（NET 传输），把 proxyOp 挂到
   `wipPlan.channels[c].proxyOpQueue`。
8. 填回 `devWork`（`711-715`）、更新 `plan->channelMask / threadPerBlock / kernelFn`（`812-817`）：
   `plan->kernelFn = ncclDevKernelForFunc[devFuncId]`（选定的特化 kernel），
   `plan->threadPerBlock = max(..., task->nWarps * WARP_SIZE)`。

#### `calcCollChunking`（`enqueue.cc:2182`）要点

- **选 pattern**（`2207-2213`）：AllReduce → Ring=`RingTwice`、Tree=`TreeUpDown`、NVLS=`Nvls`…
- **chunkSize 初值**（`2222-2227`）：`stepSize = buffSizes[protocol]/NCCL_STEPS`，再乘 chunkSteps、
  按 LL/LL128 调整；各类算法（collnet/nvls/tree/PAT）有各自的「按 nBytes/concurrentOps 缩小 chunkSize」启发式。
- **ring 的步数**（`2347-2350`）：`RingTwice` → `nstepsPerLoop = 2*(nRanks-1)`，`nchunksPerLoop = nRanks`。
- 填 `proxyOp`：`nsteps = nstepsPerLoop*nLoops*chunkSteps`、`chunkSize`、`loopSize = nChannels*nchunksPerLoop*chunkSize`、
  `pattern`、`redOp`（PreMulSum/SumPostDiv 在网络上当 Sum）、`nbytes` 等（`2363-2432`）。

### 6.4 `finishPlan`（`enqueue.cc:203`）：装填 kernelArgs + 合并 proxyOp

- **尽量塞进 kernel args**（`212-213`）：若 `sizeof(args)+batchBytes+workBytes ≤ workArgsBytes`，则
  `workStorageType = Args`（work 直接随 kernel 启动参数传，免走 FIFO）。
- **轮转排列 work batch**（`227-247`）：把各 channel 的 batch 按 channel 升序轮转铺到 `batchZero[]`，
  保证「每个 channel 的第一个 batch 恰好在 `batchZero[blockIdx.x]`」——这样 device 用 blockIdx.x 就能定位。
- **merge-sort 各 channel 的 proxyOp**（`251-280`）：按 `opCount`（并把 tag 位移到「coll 在 p2p 前」）合并成
  全局有序的 `plan->proxyOpQueue`。

> 至此，plan 完工：它知道用哪个 kernel、开几个 channel（grid 维度）、每个 channel 干什么（work batch）、
> CPU 要替哪些 channel 跑网络（proxyOp）。

### 6.5 预算怎么算 / 一个 group 切几个 plan（深入）

把 task 切成 plan 的唯一依据是 **「这批 work 放不放得下」** 的预算检查。预算由 `ncclKernelPlanBudget`
（`enqueue.cc:289`）描述，核心判定函数是 `ncclTestBudget`（`enqueue.cc:294`）。

#### (1) 预算 = 两类存储资源的容量

Plan 的 work（`ncclDevWorkColl`）可以落在两处之一，预算就是这两处的容量上限（`enqueue.cc:1607-1610`）：

```c
budget.inArgsBytes  = comm->workArgsBytes - sizeof(ncclDevKernelArgs);   // args 区剩余空间
budget.outArgsBytes = persistent ? (1<<30) : comm->workFifoBytes / 2;    // FIFO 区（或持久 buf）
```

| 量 | 取值 | 出处 |
|---|---|---|
| `workArgsBytes`（kernel 启动参数缓冲区总大小） | **4 KiB**（`4<<10`） | `device.h:485` `ncclMaxKernelArgsSize`；`init.cc:610` 与 param 取 min |
| `sizeof(ncclDevKernelArgs)`（args 头：comm 指针/channelMask/workBuf…） | ~32 B | `device.h:473-481` |
| **`inArgsBytes`** | ≈ **4064 B** | = 4096 − 32 |
| `workFifoBytes`（work FIFO 环大小） | 默认 **1 MiB** | `init.cc:394` `NCCL_WORK_FIFO_BYTES_DEFAULT = 1<<20` |
| **`outArgsBytes`**（非 graph） | **512 KiB** | = workFifoBytes/2 |
| **`outArgsBytes`**（graph/persistent） | **1 GiB** | `1<<30` |

> 为什么 FIFO 只用一半？让「上传下一个 plan」和「GPU 消费当前 plan」流水重叠——host 往另一半写时不必等 GPU
> 把前一半排空（`waitWorkFifoAvailable` `enqueue.cc:1216` 只在未消费量超过整环时才 spin）。

#### (2) `ncclTestBudget`：两种「放得下」取并集（`enqueue.cc:294`）

```c
batchBytes = nWorkBatches * sizeof(ncclDevWorkBatch);   // sizeof(ncclDevWorkBatch)=16 B (device.h:392-410)
ok |= (batchBytes + workBytes <= inArgsBytes);                                  // 方式A：batch+work 全进 args
ok |= (batchBytes <= inArgsBytes) && (workBytes <= outArgsBytes);               // 方式B：batch进args, work进FIFO
```

- **batch 头永远在 args 区**（它必须内联在 `ncclDevKernelArgs` 之后，device 用 `blockIdx.x` 直接索引 `batchZero[]`）。
- **work 数据**可进 args（方式A，最快，免走 FIFO）或进 FIFO（方式B）。`finishPlan`（`212-213`）会在最后判定：
  若 `sizeof(args)+batchBytes+workBytes ≤ workArgsBytes`，就把 `workStorageType` 降级成 `Args`——
  即「先按 FIFO 调度，最后能塞进 args 就塞进去」。

#### (3) 预算被什么消耗（关键：work 按 coll 计，batch 按 channel 计）

这是最容易看走眼的地方：

- **`workBytes`**：每个 coll task 增加一次 `workNode->size`（`enqueue.cc:854`），即 **1 个 `ncclDevWorkColl` ≈ 128 B（16 对齐）**，
  **与 channel 数无关**——channel 信息（`channelLo/Hi` + `cbd`）编码在这一个 work 结构里（`device.h:284-316`）。
- **`nWorkBatches`**：`ncclAddWorkBatchToPlan` **每个 channel 调一次**（`enqueue.cc:650/803`），需要时新增一个 16 B 的
  batch 头。所以一个 allreduce 摊到 C 个 channel，就消耗 **C 个 batch 头**。

一句话：**channel 数主要吃 batch 预算，不吃 work 数据预算**。

#### (4) 切 plan 的两轮判定：估算 + 实际 worst-case

`scheduleCollTasksToPlan`（`enqueue.cc:576`）里有两处预算检查：

**① 估算轮**（`586-604`）——粗略估计本 plan 能装几个 coll，只给 `nPlanColls` 一个上限：
```c
int nBatches = divUp(nPlanColls, 4);   // 粗略：假设每 4 个 coll 共用一个 batch（忽略 channel 展开）
if (!ncclTestBudget(budget, nBatches, workBytes + workNode->size)) goto plan_full;
nPlanColls += 1; workBytes += workNode->size;
```

**② 实际 packing 轮**（`610-855`）——真正按 channel 切、真正建 batch，每塞一个 coll 前做**权威检查**（worst-case）：
```c
// 标准路径 enqueue.cc:707（collnet 路径在 627）：
if (!ncclTestBudget(budget, plan->nWorkBatches + nChannels, plan->workBytes + workNode->size))
    return ncclSuccess;          // 放不下 → 本 plan 封口，剩余 coll 留给下一个 plan
```
`plan->nWorkBatches + nChannels` = 「为这个 coll 在它每个 channel 上各留一个新 batch 的最坏情况」。
**一旦这个检查失败，函数立即返回——这才是 plan 边界的真正决定者**；估算轮只是乐观上界，避免多迭代。

#### (5) 具体演算

- **单个 allreduce**（C 个 channel，~128 B 的 1 个 work）：消耗 `workBytes=128`、`nWorkBatches=C`（C×16 B）。
  方式A 检查 `C×16 + 128 ≤ 4064` → C ≤ 246，现实 C 远小于此 → **必然走方式A，work 进 args，1 个 plan**。
- **大 group 里 K 个 allreduce**（同 func，可共享 channel 的 batch）：work 按 ~128 B/coll 线性涨；
  batch 按 `channel × ceil(coll/64)` 涨（offsetBitset 每批 64 位）。args 装不下 work 时切到方式B：
  work 进 512 KiB FIFO → `512K/128 ≈ 4096 个 coll` 的 work 数据才填满。

结论：**一次普通 allreduce 几乎总是 1 个 plan**；只有当 group 里 coll 非常多（或海量 channel 的 batch 头爆掉 args）才切出多个 plan。

#### (6) 一个 group 切几个 plan：没有固定公式，是动态 do-while

`ncclLaunchPrepare`（`enqueue.cc:1583-1636`）就是个循环，**每次循环产出一个 plan，直到 planner 里的 task 全部排空**：

```c
do {
  memset(&planner->wipPlan, 0, ...);
  plan = pool.alloc();
  ...
  scheduleCollTasksToPlan(comm, plan, &budget);   // 尽量塞满这个 plan（被 (4)② 的 worst-case 检查封口）
  finishPlan(comm, plan);
  if (plan->workBytes != 0) enqueue(planQueue, plan);
} while (nTasksColl + nTasksP2p + nTasksBcast != 0 || 还有 sym/ce/rma task);
```

所以 **plan 数 = ceil(总 task / 每 plan 可容纳 task)**，但「每 plan 可容纳 task」是运行时按预算动态算出来的，不是常量。

#### (7) 多 plan 的安排（drain 顺序 + 发射节奏）

**同一 plan 内的 drain 顺序**（`enqueue.cc:1616-1625`）：**先 coll、再 bcast、最后 p2p**。原因：p2p 不是集合操作，
若先切 p2p，不同 rank 切 kernel 的位置会不一致，导致 channel 选择器发散而死锁——所以**先把所有 coll 排空，保证各 rank 切点一致**。

**多 plan 的发射**（`doLaunches` `group.cc:338-379`）：
- `planQueue` 里每个 plan = 一次 `cuLaunchKernelEx`（一个 kernel）。
- 按 **clique**（同 `intraComm0` 的兄弟 comm）成组发射；clique 内各 comm 的 plan **轮流、按轮**发射。
- `ncclLaunchModeGroup` 下用 intra-barrier 把各 comm 的进度对齐：每轮各 comm 各发一个 plan，barrier 同步，
  `moreRounds` 由 barrier 归约结果决定；最后一轮调 `ncclLaunchFinish` 收尾。
- 即 **plan 之间是串行启动的**（依次 `cuLaunchKernelEx`），靠 CUDA stream 顺序 + barrier 保证依赖。

#### (8) CUDA graph（persistent）的差异

- `outArgsBytes = 1<<30`（1 GiB）→ FIFO 几乎不会成为切 plan 的理由，**graph 捕获时通常就 1 个 plan**。
- 但 work 不再走环形 FIFO（捕获时不能动态写环），而是 `cudaMallocAsync` 一块**专用持久 device buf** +
  `cudaMemcpyAsync` 拷过去（`uploadWork` `enqueue.cc:1270-1286`），graph 重放时复用。
- persistent plan 的回收走 graph 析构回调（`persistentDestructor` `enqueue.cc:1528`），不是普通 `reclaimPlan`。

---

## 7. 单个 plan 的发射：uploadWork → cuLaunchKernelEx → proxy ops

### 7.1 `ncclLaunchKernelBefore_NoUncapturedCuda`（`enqueue.cc:1740`）→ `uploadWork`

仅一句：`uploadWork(comm, plan)`。`uploadWork`（`1248`）把 work 真正写到 device 能看到的地方：

- 按 `workStorageType` 分三种落点（`1256-1289`）：
  - **Args**：work 就在 kernelArgs 里，`workBuf=nullptr`。
  - **Fifo**：写到环形 `comm->workFifoBuf`，先 `waitWorkFifoAvailable`（`1267`，FIFO 满就 spin 等 GPU 消费），
    `workBuf=workFifoBufDev`，并记 `workMask=workFifoBytes-1`。
  - **Persistent**：单独 `cudaMallocAsync` 一块 device buf，`cudaMemcpyAsync` 拷过去（graph 模式用）。
- 把每个 work 节点按 16 字节粒度 memcpy 进目标（`1303-1313`），并把 batch 的 `offsetBase` 平移到 FIFO 基址（`1297-1300`）。
- Fifo 模式更新 `comm->workFifoProduced`，必要时 `wc_store_fence()`（`1317-1318`）。

> work FIFO 是 host→device 的**单向生产者/消费者环**：host 生产（`workFifoProduced`），
> GPU 消费（`workFifoConsumed`，由 `KernelFinishCallback_fn` 在 `enqueue.cc:1867` 回收时推进）。

### 7.2 `ncclLaunchKernel`：真正起 kernel（`enqueue.cc:1753`）

```c
int nChannels = countOneBits(plan->channelMask);    // 1756  grid.x = 参与的 channel 数
dim3 grid = {nChannels,1,1}, block = {plan->threadPerBlock,1,1};
void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, plan->kernelArgs,
                 CU_LAUNCH_PARAM_BUFFER_SIZE, &plan->kernelArgsSize, CU_LAUNCH_PARAM_END};
...
cuLaunchKernelEx(&launchConfig, fn, nullptr, extra);  // 1839
```

- **grid = channel 数**（不是 rank 数！每个 channel 一个 block）。
- kernel 参数（`plan->kernelArgs`，类型 `ncclDevKernelArgs`）含：`comm`(device 镜像)、`channelMask`、
  `workStorageType`、`workBuf`、`workMask`，紧跟内联的 `batchZero[]`。
- launch attrs（`1779-1837`）：sm90+ 的 **CGA cluster**（`clusterSize`）、**mem sync domain**（默认 Remote）、
  **launch completion event**（用于 implicit ordering）、symmetric kernel 的 programmatic stream serialization、
  Blackwell 的 nvlink-util-centric scheduling。

### 7.3 `ncclLaunchKernelAfter_NoCuda`（`enqueue.cc:1853`）→ `hostStreamPlanTask`

普通（非 graph）路径下 `plan->isHostCbEnq==false`，直接同步调 `hostStreamPlanTask(comm, plan)`（`1857`）。

`hostStreamPlanTask`（`1436`）：
1. **`uploadProxyOps(comm, plan)`**（`1440`）：
   - 把 plan 内每个 proxyOp 的 `opCount` 平移到 comm 的全局单调序（collOpCount / p2pOpCount，`1397-1432`）。
   - 对每个 op 调 **`ncclProxySaveOp(comm, op)`**（`1422`）。
2. **`ncclProxyStart(comm)`**（`1441`）：把已攒好的 ops `ncclProxyPost` 给 proxy 线程（见下）。
3. 非 persistent：把 `plan->reclaimer`（= `reclaimPlan`，`1462`）挂到 `comm->callbackQueue`，让 GPU 跑完后异步回收 plan 内存。

> 注意顺序：**先 uploadWork、再起 kernel、再提交 proxyOp**。proxyOp 提交得晚没关系——proxy 线程会和 GPU 并发跑，
> 等 GPU 真正请求数据时 proxy 已经在轮询了。

#### `ncclProxySaveOp`（`proxy.cc:602`）—— 把 op 落到 proxy 的 per-peer 池

按 `op->pattern` 分流，对 ring/tree/collnet/nvls 各自的连接调 `SaveProxy`：
- 关键是 `NeedProxy(...)`（`613/620`）：**只有当连接是 NET（需要 CPU 驱动 IB/TCP）时才需要 proxy**。
  纯 intra-node P2P/SHM 的连接 `NeedProxy` 为假，GPU 自己搬数据，不麻烦 proxy。
- 命中则 `SaveProxy` 把 op 追加到 `comm->proxyState->proxyOps[peer].pool`。

#### `ncclProxyStart`（`proxy.cc:1014`）—— 唤醒 proxy 线程

```c
for (int r=0; r<peerArraySize; r++)
  if (ops[r].nextOps != -1) ncclProxyPost(ops[r].pool, ops[r].nextOps, ops[r].nextOpsEnd);
comm->opCount++;
```

`ncclProxyPost` 把 ops 链挂到 pool 的 `nextOps` 并 `cond.notify`，唤醒可能正在 `cond_wait` 的 proxy 线程。

---

## 8. Layer 3：Proxy 线程如何驱动网络（`proxy.cc:954`）

每个 GPU 一个 proxy 进度线程（`ncclProxyProgress`，`954`），死循环：

```c
do {
  int idle = 1;
  progressOps(proxyState, state, state->active, &idle);   // 980  推进所有 active op
  if (idle || ... ++counter == appendFreq()) {
    ncclProxyGetPostedOps(proxyState, &added);            // 997  捡起新 post 的 op
    if (added == 0) std::this_thread::yield();            // 1005 没活儿就 yield
  }
} while (!stop && !abort);
```

- **`progressOps`**（`801`）：遍历 active 的 `ncclProxyArgs`，调 `op->progress(proxyState, op)`——
  这就是**传输 vtable 的 `proxyProgress`**（`net_ib` 实现里：根据 op 的 step/head/tail，`ibv_post_send`/`ibv_post_recv`，
  轮询 completion，推进 GPU 在 spin 的那个 head 计数器）。op 完成则 `removeOp`。
- **`ncclProxyGetPostedOps`**（`835`）：从 pool 取新 post 的 op（`ProxyAppend` 把它们接成可推进的 `ncclProxyArgs`）；
  完全空闲时在 `pool->cond.wait(lock)` 上睡眠，直到 `ncclProxyStart` 唤醒。

**device↔proxy 握手机制**（细节在 `prims_simple.h`）：GPU 端把数据写进 channel 的 net buffer 并推进 `tail` 计数器；
proxy 线程观察到 `tail` 前进，发起网络收发，完成后推进 `head`；GPU 在
`while (peer->recv.tail < step)`（`prims_simple.h:316`）上 spin 等 head 到位。两者通过这套
**head/tail 计数器 + fence** 的无锁环（`ncclConnFifo`，深度 `NCCL_STEPS=8`）解耦，等价于一个 RDMA 式的生产者/消费者环。

---

## 9. Layer 4：GPU 内核如何真正完成 AllReduce

### 9.1 kernel 入口 → `ncclKernelMain`（`device/common.h:438 / 355`）

```c
__global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K const args4K) {
  ncclKernelMain<SpecializedFnId, RunWorkBatch<coll,ty,redop<ty>,algo,proto>>(&args4K.args);
}
```

`ncclKernelMain`（`355`）：

1. **拷贝 args 到 shmem**（`362-364`）——避免编译器把 args 塞进线程栈。
2. **blockIdx.x → channelId**（`370-373`）：`channelId` = channelMask 中第 blockIdx.x 个置位。
   （注释解释：不用 PTX `fns` 指令，因为所有线程查同一个 mask，用 popcount 更快。）
3. **三个 warp 组分工加载 shmem**（`383-414`）：warp0 载 `ncclKernelComm`、warp1 载 `ncclDevChannel[channelId]`、
   其余 warp 调 `loadWorkBatchToShmem(...,batchIx=blockIdx.x)` 载第一个 work batch。
4. **主循环**（`417-431`）：
   ```c
   while (!aborted) {
     profiler(START);
     if (特化命中) SpecializedRunWorkBatch().run();   // 419  编译期特化，最快
     else ncclDevFuncTable[funcId]();                 // 422  通用函数表分发
     if (nextBatchIx == -1) break;                    // 425  本 channel 的 batch 处理完
     loadWorkBatchToShmem(..., nextBatchIx);          // 429  载下一个 batch
   }
   profiler(FINI);
   ```
   一个 block（channel）会依次处理它名下的所有 work batch，直到 `nextBatchIx==-1`。

#### `loadWorkBatchToShmem`（`common.h:140`）怎么解析 work

- 从 `args+1` 处取 `ncclDevWorkBatch`（`145`），用 `offsetBitset` 算出「这一批含哪几个 work」
  （`fnsOfBitset`，`153-163`，按 16 字节 pack 对齐加载）。
- 按 `workStorageType` 从 kernel args（`ld.param`）或 work FIFO（`ld.v2.u64`，带 `workMask` 环回）取 work 字节，
  写进 shmem 的 `workStorage[]`（`228-238`）。
- `batch.nextExtends` 为真则沿 `nextJump` 继续载同一 channel 的后续 batch（`242-255`）；否则登记
  `nWorks/workType/funcId` 并算出 `nextBatchIx`。

### 9.2 `RunWorkBatch → RunWorkColl`（`common.h:284 / 265`）

`RunWorkBatch<Fn,T,RedOp,Algo,Proto>::run()`（`287`）遍历本 batch 的每个 work，
调 `RunWorkColl<Fn,T,RedOp,Algo,Proto>().run(tid,tn,work)`（`314`）。
对 AllReduce+Ring+Simple，特化在 `all_reduce.h:229`，进而调 **`runRing`**。

### 9.3 `runRing`：环形 allreduce 的两阶段（`all_reduce.h:14`）

```c
ncclRing* ring = &ncclShmem.channel.ring;
ncclCollCbdPart(work, channelId, Proto::Id, sizeof(T), nullptr,&gridOffset,&channelCount,&chunkCount);
const ssize_t loopCount = nranks * chunkCount;
Primitives<T,RedOp,FanSymmetric<1>,1,Proto,0> prims(tid,nthreads,&ring->prev,&ring->next,
                                                    work->sendbuff, work->recvbuff, work->redOpArg,0,0,0,work);
for (elemOffset = 0; elemOffset < channelCount; elemOffset += loopCount) {
  // ── 第一阶段 scatter-reduce（nRanks-1 步）：每个 chunk 绕环一周，逐站累加
  step0:  prims.directSend(...)                                  // 把自己的 chunk 推给下一个 rank
  for j in 2..nRanks-1: prims.directRecvReduceDirectSend(...)    // 收上一站、与本地 reduce、再推下一站
  // ── 第二阶段 allgather（nRanks-1 步）：把每个 chunk 的最终结果广播给所有 rank
  prims.directRecvReduceCopyDirectSend(..., postOp=true)         // 本 chunk 收尾 reduce（含 postOp）
  for j in 1..nRanks-2: prims.directRecvCopyDirectSend(...)      // 纯转发
  prims.directRecv(...)                                          // 最后一站落到 recvbuff
}
```

> - `ncclCollCbdPart`：根据 `work->cbd`（countLo/Mid/Hi、chunkGrains）+ `channelId`，算出**本 channel**
>   在全局数据里的 `gridOffset`、本 channel 要处理的 `channelCount`、每个 ring chunk 的 `chunkCount`。
> - `Primitives` 绑定本 channel 的 `ring->prev`/`ring->next` 连接器；`directRecv`/`directSend` 等在
>   `prims_simple.h` 里实现：通过 connection FIFO 与对端（或 proxy）握手搬数据，必要时在 device 内做 reduce。
> - ring allreduce 总通信量 ≈ `2·(nRanks-1)/nRanks · data`（与 2.3 里 trafficPerByte 一致），这就是
>   `RingTwice` pattern 的 `nstepsPerLoop = 2*(nRanks-1)` 的来历。

### 9.4 数据真正移动的地方（`prims_simple.h`）

- 对 **intra-node（P2P/SHM）**：GPU 直接读写对端 GPU 暴露的 buffer，推进 `tail`，对端 spin `tail` 后取走——全程不经 proxy。
- 对 **inter-node（NET）**：GPU 把数据写进 net buffer、推进 `tail`；proxy 线程（第 8 节）发现 `tail` 前进后
  `ibv_post_send` 把数据发到远端 rank，远端 proxy `ibv_post_recv` 收下并推进本地 `head`，远端 GPU spin 到 `head` 后继续。
- 这些 `while (peer->recv.tail < step)` 式的 spin（`prims_simple.h:316/742`）+ `checkAbort`（`primitives.h:154`）
  构成 GPU 与 proxy 之间的无锁同步。

---

## 10. 端到端串联：一次 N-rank AllReduce 的完整时序

以「rank 0 上调用 `ncclAllReduce(s, r, count, Float, Sum, comm, stream)`、Ring/SIMPLE、跨节点需 NET」为例：

```
[rank0 用户线程]
 1. ncclAllReduce → ncclEnqueueCheck → (隐式 group start)
 2. collTaskAppend: planner.collSorter 收到 1 个 AllReduce task；comm 加入 group
 3. (隐式 group end) → groupLaunch:
    a. ncclPrepareTasks: 选 Ring/SIMPLE, devFuncId=AllReduce_Sum_f32_RING_SIMPLE,
       造 ncclDevWorkColl
    b. ncclTasksRegAndEnqueue: 注册 s/r buffer, work 节点入 collWorkQueue
    c. doLaunches → ncclLaunchPrepare:
       scheduleCollTasksToPlan: 把 count 切到 C 个 channel (countLo/Mid/Hi),
         每 channel 算 chunkSize, 建 proxyOp(因跨节点 NET);
       finishPlan: 装 kernelArgs, 合并 proxyOp; → 1 个 plan (数据小时)
    d. 发射该 plan:
       uploadWork → work 写进 workFifoBuf, workFifoProduced++
       cuLaunchKernelEx: grid=C 个 block, kernelFn=ncclDevKernel_AllReduce_..., 挂在用户 stream
       hostStreamPlanTask:
         uploadProxyOps → ncclProxySaveOp(每 NET channel 一个 op) → proxyState.proxyOps[peer]
         ncclProxyStart → ncclProxyPost → 唤醒 proxy 线程

[rank0 GPU 内核] (与下面 proxy 并发)
 4. ncclKernelMain: block c → channelId; 载 comm/channel/workBatch
 5. RunWorkColl<AllReduce,Ring,Simple>::run → runRing:
    对本 channel 的 channelCount 跑 scatter-reduce + allgather;
    每个 directSend/directRecv 经 Primitives + connFifo 与对端/proxy 握手

[rank0 proxy 线程]
 6. ncclProxyGetPostedOps 捡起 op → progressOps → net_ib.proxyProgress:
    ibv_post_send/recv 在 rank 间搬数据, 推进 GPU spin 的 head 计数器

[全局] 所有 rank 的 GPU 各自跑完 2*(nRanks-1) 步/chunk → 每个 rank 的 recvbuff 得到全 reduce 结果
 7. 内核结束 → launchStream 上的 finishedEvent 被 record
 8. ncclLaunchFinish: 串好 stream 依赖; KernelFinishCallback 推进 workFifoConsumed;
    reclaimPlan 回收 plan 内存
```

**rank 间靠什么对齐？** 不是靠 group end 时的 barrier（那只对同进程多 comm 的 clique 有效），而是靠
**ring/tree 拓扑本身**：每个 rank 的 GPU 在 `directRecv` 上 spin 等上一站数据，自然就同步了；
proxy 的网络收发也以同样的 step 序列推进。所以各 rank 只要拓扑/步数一致（由 `calcCollChunking` 保证同构），
就能无显式同步地完成。

---

## 11. 关键 file:line 速查表

| 环节 | 位置 |
|---|---|
| API 包装 | `collectives.cc:168` (ncclAllReduce) |
| 隐式 group + 挂 task | `enqueue.cc:3124` (ncclEnqueueCheck) / `2690` (collTaskAppend) |
| group end 入口 | `group.cc:766` (ncclGroupEndInternal) / `598` (groupLaunch) |
| 算法/协议选择 | `enqueue.cc:363` (ncclPrepareTasks) / `441` (ncclGetAlgoInfo) |
| buffer 注册 + work | `enqueue.cc:302` (ncclTasksRegAndEnqueue) |
| plan 切分 | `enqueue.cc:1568` (ncclLaunchPrepare) / `576` (scheduleCollTasksToPlan) |
| 预算 / 切几个 plan | `enqueue.cc:289/294` (ncclKernelPlanBudget/ncclTestBudget) / `1607-1610` (预算) / `707` (worst-case 封口)；详见 §6.5 |
| channel 数据切分 | `enqueue.cc:659-701` (countLo/Mid/Hi) / `2182` (calcCollChunking) |
| work batch 构造 | `enqueue.cc:121` (ncclAddWorkBatchToPlan) / `203` (finishPlan) |
| 发射编排 | `group.cc:309` (doLaunches) |
| 写 work FIFO | `enqueue.cc:1248` (uploadWork) / `1216` (waitWorkFifoAvailable) |
| 起 kernel | `enqueue.cc:1753` (ncclLaunchKernel → cuLaunchKernelEx:1839) |
| 提交 proxy op | `enqueue.cc:1392` (uploadProxyOps) / `proxy.cc:602` (ncclProxySaveOp) / `1014` (ncclProxyStart) |
| proxy 进度循环 | `proxy.cc:954` (ncclProxyProgress) / `801` (progressOps) |
| 收尾 | `enqueue.cc:1876` (ncclLaunchFinish) / `1462` (reclaimPlan) |
| device kernel 入口 | `device/common.h:438` (DEFINE_ncclDevKernel) / `355` (ncclKernelMain) |
| work batch 解析 | `device/common.h:140` (loadWorkBatchToShmem) |
| allreduce ring 执行 | `device/all_reduce.h:14` (runRing) / `229` (RunWorkColl 特化) |
| device↔proxy 握手 | `device/prims_simple.h` (connFifo head/tail spin) |

---

## 12. 几个容易困惑的点（FAQ）

1. **单个 `ncclAllReduce` 没有 group，也会立刻 launch 吗？**
   会。`ncclEnqueueCheck` 用隐式 `ncclGroupStartInternal/EndInternal` 把它包成 depth=1 的 group，
   `End` 时 depth 减到 0 就走完整 launch（`enqueue.cc:3137/3164`）。

2. **一次 group 一定只 launch 一个 kernel 吗？**
   不一定。work 量超过单 plan 预算时，`ncclLaunchPrepare` 会切出多个 plan（多个 kernel），
   `doLaunches` 多轮逐个发射（`group.cc:338-379`）。**预算怎么算、什么时候切多个 plan，详见 §6.5。**

3. **grid 维度是 rank 数吗？**
   不是。grid.x = **channel 数**（`plan->channelMask` 置位数，`enqueue.cc:1756`）。每个 block 处理一个 channel。

4. **为什么有 work 和 proxyOp 两套？**
   work 给 GPU 看（device 能自己干的：intra-node 搬运 + reduce）；proxyOp 给 CPU proxy 看（device 干不了的：
   跨节点 IB/TCP 收发）。同一 channel 可能两者都有（GPU 负责 reduce 和本地搬运，proxy 负责网络这一段）。

5. **GPU 和 proxy 怎么不撞车？**
   靠 `ncclConnFifo`（深度 `NCCL_STEPS=8`）的 head/tail 计数器 + 内存 fence：生产者推进 tail，消费者 spin tail/head，
   天然构成无锁环。`NeedProxy` 决定某连接是否真要 proxy 参与（纯本地连接不需要）。

6. **kernel 跑完后内存怎么回收？**
   `hostStreamPlanTask` 把 `plan->reclaimer` 挂到 `comm->callbackQueue`（`enqueue.cc:1447`），
   主线程后续 `ncclCommPollCallbacks` 时调 `reclaimPlan`（`1462`）释放 task/work/proxyOp。

---

*本文为 fork-local 分析笔记，非 NVIDIA 官方文档。结论以 NCCL 2.30.7 源码为准（当前仓库已升级至 2.31.2，
路径/行号偏移见文首时效说明）。*
