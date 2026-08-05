# NCCL Enqueue 全流程解析：从用户调用到 kernel 发射（v2.30.7）

> 本文逐行基于 `src/collectives.cc`、`src/enqueue.cc`（3246 行）、`src/group.cc`、`src/include/comm.h`、`src/include/info.h` 整理，所有结论附 `文件:行号`。
> 前置阅读：`notes/nccl_thread_model.md`（线程模型）、`notes/nccl_proxy_internals.md`（proxy 机制）、`notes/nccl_cuda_stream.md`（stream 语义）。

---

## 0. 一句话总览

用户调用 `ncclAllReduce` 后，主线程依次经历 **task 入队（taskAppend）→ group 收尾时统一规划（ncclPrepareTasks）→ 注册与 work 构建（ncclTasksRegAndEnqueue）→ 切分成 kernel plan（ncclLaunchPrepare + scheduleXxxTasksToPlan）→ 逐 plan 发射（doLaunches → uploadWork + cuLaunchKernelEx）→ 异步收尾（proxy op 上传、plan 回收）** 六个阶段。全程无隐藏线程；唯一的异步性是 CUDA stream 语义与 host 回调。

```
ncclAllReduce (用户线程)
  │ collectives.cc:168  填 ncclInfo
  ▼
ncclEnqueueCheck (enqueue.cc:3124)
  │ ncclGroupStartInternal → taskAppend → ncclGroupEndInternal
  ▼
taskAppend (enqueue.cc:3014) ── 任务进 comm->planner 的各队列
  │
  ▼  (ncclGroupEnd 时)  groupLaunch (group.cc)
ncclPrepareTasks (enqueue.cc:363)      ← 排序/聚合/选算法/算线程数
ncclTasksRegAndEnqueue (enqueue.cc:302) ← 注册 buffer + 构建 ncclDevWorkColl
doLaunches (group.cc:309)
  ├─ ncclLaunchPrepare (enqueue.cc:1568)  ← 任务切分成 ncclKernelPlan 队列
  │    └─ scheduleCollTasksToPlan / addP2pToPlan / finishPlan
  ├─ 每 plan: ncclLaunchKernelBefore (uploadWork, enqueue.cc:1248)
  │           ncclLaunchKernel (enqueue.cc:1753)  ← cuLaunchKernelEx
  │           ncclLaunchKernelAfter (hostStreamPlanTask 或直接)
  └─ ncclLaunchFinish (enqueue.cc:1876)   ← stream 依赖缝合 + workFifoConsumed
异步收尾 (CUDA host 回调线程): hostStreamPlanTask → uploadProxyOps + ncclProxyStart → reclaimPlan
```

---

## 1. 阶段一：API 入口与 `ncclInfo`

每个集合 API 是极薄的包装：填一个 `ncclInfo` 后调 `ncclEnqueueCheck`。以 allreduce 为例（`collectives.cc:168-178`）：

```c
struct ncclInfo info = {
  ncclFuncAllReduce, "AllReduce", sendbuff, recvbuff, count, datatype, op, 0, comm, stream,
  ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS
};
return ncclEnqueueCheck(&info);
```

`struct ncclInfo`（`info.h:17-41`）：`coll/opName/sendbuff/recvbuff/count/datatype/op/root/comm/stream` + `chunkSteps/sliceSteps`（SIMPLE+RING 协议的切片参数，每个集合有默认值常量）+ RMA 相关字段（peerWin/sigIdx/ctx/signalDescs）。

`ncclEnqueueCheck`（`enqueue.cc:3124-3172`）做的事：

1. `CommCheck` / revoked 检查；
2. `ncclGroupStartInternal()`——**隐式打开 group**（嵌套深度+1；用户显式 group 时只是加深一层）；
3. `ncclCommEnsureReady`——等 comm 初始化完成（init 可能是异步 job）；
4. `ArgsCheck`（`argcheck.h`）——参数合法性；`checkMode==DebugGlobal` 时仅记录不执行；
5. `taskAppend(info->comm, info)`——真正的入队；
6. `ncclGroupEndInternal()`——深度归零时**触发整个 group 的 launch**（见 §3）；非阻塞 comm 上错误经 `ncclCommGetAsyncError` 上报。

要点：**单个集合调用 = 一个隐式 group**。所有"攒任务"的逻辑（planner）在显式 `ncclGroupStart/End` 下才真正跨多个 API 调用累积。

---

## 2. 阶段二：`taskAppend`——任务进 planner（`enqueue.cc:3014-3122`）

按任务类型分流：

| 类型 | 去向 | 函数 |
|---|---|---|
| `ncclFuncSend/Recv` | p2p 任务 | `p2pTaskAppend`（`enqueue.cc:2615`） |
| `ncclFuncPutSignal/Signal/WaitSignal` | RMA 任务 | `rmaTaskAppend`（`enqueue.cc:2807`） |
| AlltoAll/Gather/Scatter | **拆成 p2p 任务**（每 rank 一对 send+recv） | `p2pTaskAppend` 循环 |
| 集合 + `CTAPolicy==ZERO` 且 CE 可用 | CE collective | `ceCollTaskAppend`（`enqueue.cc:2754`） |
| Blackwell 上 AllGather >8MB 且全 NVLink | CE collective | 同上（`enqueue.cc:3110-3113`） |
| 其他集合 | 普通 kernel 集合 | `collTaskAppend`（`enqueue.cc:2690`） |

特例：`count==0` 直接丢弃；`nRanks==1` 走 `ncclLaunchOneRank` 本地拷贝；FP8 规约要求 sm90+；`hostToDevRedOp`（`enqueue.cc:2479`）把用户 redOp 句柄物化为 `ncclDevRedOpFull`（因为句柄可能在 GroupEnd 前被销毁）。

### 2.1 `collTaskAppend`（`enqueue.cc:2690-2752`）

1. `ncclGroupCommJoin(comm, ncclGroupTaskTypeCollective)`——把 comm 挂进线程本地 group 链表（`groupNext[type]`），group 收尾时按此链表逐 comm 处理；
2. `ncclPlannerSetCapturingGraph`（`enqueue.cc:2585`）——记录用户 stream 到 `planner->streams` 链表，并约束**同 group 内所有 stream 要么都没被 graph 捕获、要么被同一个 graph 捕获**；
3. Broadcast 特例：`NCCL_ALLGATHERV_ENABLE` 且无 CC 时转为 `ncclTaskBcast`（AllGatherV 语义）进 `planner->peers[root].bcastQueue`，并维护 `bcast_info`（min/max peer、peer 数）；
4. 普通路径：从 `comm->memPool_ncclTaskColl` 分配 `ncclTaskColl`，填充字段；**AllGather/Broadcast 统一转成字节流**（`count *= elementSize; datatype = ncclInt8`）；`trafficBytes = count*eltSize * ncclFuncTrafficPerByte(func, nRanks)`（AllReduce=2、AllGather/ReduceScatter=nRanks、其余=1，`enqueue.cc:91-102`）；插入 **`collSorter`**。

`ncclTaskColl`（`comm.h:193-235`）关键字段：`func/sendbuff/recvbuff/count/root/datatype/opHost/opDev/chunkSteps/sliceSteps`；规划后回填：`trafficBytes/nMaxChannels/nWarps/algorithm/protocol/devFuncId/isCollnet/isNvls`；注册相关：`sendWin/recvWin/regBufType/sendMhandle/recvMhandle/sendNetHandles...`；profiler 字段。

### 2.2 `ncclTaskCollSorter`——按流量大小降序的桶排序（`comm.h:361-420`）

- 桶范围 1KB–1GB，每个 2 的幂区间再分 4 桶（`BitsPerPow2=2`），桶号由 `u32fpEncode(size>>10)` 算出，最坏乱序幅度 25%；
- 桶内 LIFO 头插，`bins[]` 保存"指向桶头的指针的指针"使单链表 O(1) 插入；
- `ncclTaskCollSorterDequeueAll` 一次性取出整条降序链表。
- **为什么大任务在前**：大集合先占 channel，小集合填缝隙，channel 利用率更高（见 §5.1 的 traffic 均衡）。

### 2.3 `p2pTaskAppend`（`enqueue.cc:2615-2688`）

- `ncclTaskP2p` 入 `planner->peers[peer].sendQueue/recvQueue`（每 rank 一对队列），`nTasksP2p(Send/Recv)` 计数；
- **预连接标记**：首次见到某 peer 方向时，按 `comm->p2pSchedule` 找到该 peer 的 round，对该 round 覆盖的 channel 置 `connectSend/Recv[peer]` 位图 + 连接器 `hasSeen/p2pOnly=1` + `ncclGroupCommPreconnect(comm)`——group 收尾时统一建连，避免首次 p2p 时串行建连的延迟。`hasSeen` 跨 split 共享 comm 去重（注释 `enqueue.cc:2665-2669`）。

### 2.4 planner 的任务容器（`struct ncclKernelPlanner`，`comm.h:429-502`）

```
积累区（GroupStart~GroupEnd 之间）:
  collSorter            —— 普通集合（按流量降序）
  peers[r].sendQueue / recvQueue / bcastQueue  —— p2p 与 bcast
  streams / streamRecent / capturingGraph      —— 用户 stream 集合
待装配区（PrepareTasks 产出）:
  collTaskQueue / collCeTaskQueue / collSymTaskQueue / rmaTaskQueues
  collWorkQueue（与 collTaskQueue 平行的 ncclDevWork* 队列）
WIP plan（正在装配）:
  wipPlan.channels[64] = { workBatchQueue, proxyOpQueue, wipBatch 状态 }
发射区:
  planQueue + unlaunchedPlansHead
```

---

## 3. 阶段三：group 收尾的统一编排（`group.cc`）

`ncclGroupEndInternal` 深度归零 → `groupLaunch`（阻塞模式直接跑；非阻塞跑在 `groupLaunchNonBlocking` job 线程上）。与 enqueue 相关的顺序（`group.cc:660-724`）：

1. **对称注册 job**（`ncclCommGroupRegisterSymmetric`，异步 job 并行）；
2. **逐 clique**（`intraComm0` 相同的 comm 集合）：`ncclPrepareTasksAndCollPreconnect`——先 `ncclPrepareTasks`（本阶段主角，见 §4），需要建连的算法经 `algoNeedConnect` 汇总，预连接以异步 job 并行执行（`asyncJobLaunch`）。clique 间串行是为避免 split 共享 comm 对同一连接竞争建连（注释 `group.cc:669-672`）；
3. 逐 comm `ncclTasksRegAndEnqueue`（注册 + 建 work，见 §4.4）；
4. DebugGlobal 检查 job；
5. `doLaunches(groupCommHeadMain[Collective])`（见 §6）；
6. 收尾：`ncclGroupCommLeave` 每个 comm（顺带 `ncclCommPollCallbacks` 回收内存池）、planner 清零复位。

`simInfo != NULL`（`ncclCommSimulate`）时只准备不发射。

---

## 4. 阶段四：`ncclPrepareTasks`——排序、聚合、选算法（`enqueue.cc:363-565`）

每个 comm 一次，做四件事：

### 4.1 队列整流

- 单一 bcast peer 时把 `ncclTaskBcast` 转回普通 `ncclTaskColl`（func=Broadcast）进 sorter（`enqueue.cc:368-390`）；
- `ncclTaskCollSorterDequeueAll` 取出**流量降序**的任务链；
- 支持 symmetric 且无跨 clique P2P 时，`ncclMakeSymmetricTaskList` 把可走对称 kernel 的任务切到 `collSymTaskQueue`（`enqueue.cc:401-403`）。

### 4.2 按 (fn, op, ty) 分桶 + 4x 聚合（`enqueue.cc:405-470`）

- 任务按 `(func, opDev.op, datatype)` 组合分桶（桶内 LIFO，此时同类任务**按流量升序**——因为取出时是降序）；
- 每桶内做**聚合扫描**：从 `aggBeg` 出发，把 `trafficBytes < 4 * aggBeg->trafficBytes` 的后续任务聚成一个 `agg`——**一次算法选择服务一整组尺寸相近的任务**，摊薄调优开销；
- 对每个聚合体调 `ncclGetAlgoInfo`（见 §4.3），然后把结果（`algorithm/protocol/nMaxChannels/nWarps/devFuncId/isCollnet/isNvls`）回写到组内每个任务；LL 协议任务的 `trafficBytes *= 4`（LL 有效负载减半、旗标开销）；
- 按 `(isCollnet, isNvls)` 四个桶重新排队，最后拼成 `collTaskQueue`——**collnet 是外层序**（影响 channel 划分，`enqueue.cc:473-480`）。

### 4.3 `ncclGetAlgoInfo` → `topoGetAlgoInfo`（`enqueue.cc:2123/2028`）

算法+协议选择：

1. `initCollCostTable` + `updateCollCostTable` 填 `[algo][proto]` 代价表（拓扑带宽模型 + 环境变量 `NCCL_ALGO/NCCL_PROTO` 过滤）；
2. 有 tuner 插件时先 `tuner->getCollInfo`（可改代价表并给 `nMaxChannels`），然后 `topoGetAlgoInfo` 在表上取**最小时间**的 (algo, proto)；
3. `topoGetAlgoInfo` 的通道/线程调优（`enqueue.cc:2070-2115`）：
   - channel 数 `nc`：Ring/Tree 按 `nBytes < nc*nt*threadThreshold` 递减（小消息少通道）；NVLS 用 `nvlsChannels`；CollNetDirect 有专门的 ncSwitch 阶梯；
   - 线程数 `nt`：取自 `comm->maxThreads[algo][proto]`（`graph/tuning.cc` 预调表），小消息再对半减；**RING+SIMPLE 加 1 个同步 warp、TREE+SIMPLE 加 4 个**（`enqueue.cc:2107-2109`）；下限 3 warp；**TREE 与 PAT 恒为 `NCCL_MAX_NTHREADS=640`**；
   - 产出 `nMaxChannels` 与 `nWarps`；
4. `devFuncId = ncclDevFuncId(func, op, ty, algo, proto)`——决定用哪个特化 kernel（`ncclDevKernelForFunc[]` 表）。

另外 `NCCL_CTA_POLICY_EFFICIENCY` 且无 tuner 时，已注册 buffer 的 AllGather/ReduceScatter 会被强制改写为 NVLS+SIMPLE（`enqueue.cc:2152-2175`）。

### 4.4 `ncclTasksRegAndEnqueue`（`enqueue.cc:302-359`）

逐个任务：`ncclRegisterCollBuffers`（用户 buffer 注册，含 graph 捕获下的延迟注册；NVLS 任务此前已在 `ncclPrepareTasks` 里建好 work 进 `tmpCollWorkQueue`，这里直接出队）→ 构建 **`ncclDevWorkColl`**（或 NVLS 注册的 `ncclDevWorkCollReg`）封装进 `ncclWorkList`，与任务一一对应地挂到 `collWorkQueue`。注册产生的清理动作用 `collCleanupQueue` 暂存，随 plan 走。

---

## 5. 阶段五：`ncclLaunchPrepare`——把任务切成 kernel plan（`enqueue.cc:1568-1738`）

### 5.1 plan 切分循环

```c
do {
  memset(&planner->wipPlan, 0, ...);
  plan = ncclMemoryPoolAlloc(memPool_ncclKernelPlan);
  plan->workStorageType = persistent ? Persistent : Fifo;   // finishPlan 可提升为 Args
  if (有 RMA)              scheduleRmaTasksToPlan;
  else if (有 CE)          scheduleCeCollTaskToPlan;
  else if (有 sym)         ncclSymmetricTaskScheduler;
  else {
    budget.inArgsBytes  = comm->workArgsBytes - sizeof(ncclDevKernelArgs);
    budget.outArgsBytes = persistent ? 1GB : comm->workFifoBytes / 2;  // 单 kernel 最多用半个 FIFO
    if (nTasksColl)     scheduleCollTasksToPlan(comm, plan, &budget);
    if (coll 空 && bcast) ncclScheduleBcastTasksToPlan;
    if (coll/bcast 空 && p2p) scheduleP2pTasksToPlan(comm, &p2pEpoch, &p2pRound, plan, &budget);
  }
  finishPlan(comm, plan);
  if (plan->workBytes) planQueue 入队;
} while (还有任务);
```

两个设计要点：

- **调度顺序：先集合、再 bcast、最后 p2p**（`enqueue.cc:1612-1624`）。注释明确：plan 的切分点由 budget 决定，若先排 p2p，切分点会随 rank 任务分布不同而发散，导致"最短 channel 优先"的选取在各 rank 不一致（挂死风险）。**先排集合保证所有 rank 的 plan 切分一致**。
- **budget 双约束**（`ncclTestBudget`，`enqueue.cc:294-300`）：`(batchBytes+workBytes ≤ inArgsBytes)` 或 `(batchBytes ≤ inArgsBytes && workBytes ≤ outArgsBytes)`——能整个塞进 kernel 参数（4KB）最好，否则 batch 进参数、work 进 FIFO/persistent buffer。budget 用尽是 plan 切分的唯一原因（一个 plan = 一次 kernel 发射）。

### 5.2 `scheduleCollTasksToPlan`——集合任务的 channel 分配（`enqueue.cc:576-857`）

这是最复杂的调度函数，分两步：

**预估**：在 budget 内数出本 plan 能装几个集合（`nPlanColls`），按 `[collnet][nvls]` 四类累计 `trafficBytes` 与 `nChannels`（每通道至少计 `MinTrafficPerChannel=32KB`，`enqueue.cc:586-604`）。

**逐个任务分配 channel**（`enqueue.cc:610-855`）：

- 每个 kind 的 `trafficPerChannel = divUp(trafficBytes[kind]/nChannels[kind], 16)*16`——channel 间的**流量均衡目标**；
- 非 collnet 任务（`enqueue.cc:656-755`）：把任务数据切成 `cellSize`（由 32KB 最小流量推出）的 cell，然后按 Lo/Mid/Hi 三段摊到 channel：
  - 当前 channel 剩余流量配额装不下的零头进 **Lo**（最后 channel 全进 Lo）；
  - 中间 channel 各分 `cellsPerChannel` 个 cell（**Mid**，均匀）；
  - 余数进最后一个 channel 的 **Hi**；
  - 溢出可用 channel 时重新均摊；`channelId/currentTraffic` 逐任务推进，实现**跨任务的连续流量均衡**（"最短 channel 优先"的效果）。
- 产出 `devWork->cbd = {countLo, countMid, countHi, chunkGrainsLo/Mid/Hi}` 与 `devWork->channelLo/channelHi`；
- 对 Lo/Mid/Hi 三段分别 `calcCollChunking`（见 §5.3）得到三个 `ncclProxyOp` 模板；
- **每 channel 一份 proxy op**：`opCount = (plan->collOpCount++) << 1`（偶数=集合；p2p 用奇数——与 `uploadProxyOps` 的重编号规则呼应），填 `channelId/loopOffset/channelSize`；注册 buffer 的 RING 算法还要 `new RingAG/AR/BCAlgorithm(...)`（`enqueue.cc:778-799`）供 progress 线程直收直发用户 buffer；
- 每个 channel 调 `ncclAddWorkBatchToPlan` + `ncclAddProxyOpIfNeeded` + `addProfilerProxyOpIfNeeded`；
- 汇总：`plan->channelMask |= (2<<hi)-(1<<lo)`；`threadPerBlock = max(threadPerBlock, nWarps*32)`；若 plan 内任务同质则选定**特化 kernel**（`ncclDevKernelForFunc[devFuncId]`），否则退回 `ncclDevKernel_Generic`；
- 任务与 workNode 从 planner 队列搬进 `plan->collTaskQueue/workQueue`，`plan->workBytes` 累计。

### 5.3 `calcCollChunking`——chunk 与 proxy op 参数（`enqueue.cc:2182-2477`）

1. **func+algo → pattern**（`enqueue.cc:2188-2218`）：如 AllReduce = RingTwice/TreeUpDown/CollnetChain/CollnetDirect/Nvls/NvlsTree；ReduceScatter = Ring/PatUp/Nvls/CollnetDirect……这个 pattern 就是 `ncclProxySaveOp` 的分解依据；
2. **chunk 基准**：`stepSize = comm->buffSizes[proto] / NCCL_STEPS`；`chunkSize = stepSize * chunkSteps`；LL 减半、LL128 按 120/128 数据线折算；SIMPLE+RING 才用任务里的 `chunkSteps/sliceSteps`，其他恒 1；
3. **按算法收缩 chunk**（`enqueue.cc:2231-2299`）：CollNet/NVLS/PAT/TREE_LL128 各自的经验公式循环减半，保证流水线级数；
4. tuner 插件可覆写 chunkSize（钳到 buffer 上限，`enqueue.cc:2308-2318`）；
5. **nsteps 计算**（`enqueue.cc:2322-2364`）：按 pattern 定 `nstepsPerLoop`（Ring=nRanks-1、RingTwice=2(nRanks-1)、Nvls=1 但 nchunksPerLoop=nHeads……），`loopSize = nChannels*nchunksPerLoop*chunkSize`，`nLoops = DIVUP(nBytes, loopSize)`，`proxyOp->nsteps = nstepsPerLoop*nLoops*chunkSteps`；
6. 填充 `ncclProxyOp` 全字段（`sliceSize/loopOffset/protocol/dtype/redOp`（PreMulSum/SumPostDiv 对网络呈现为 Sum）`/pattern/coll/root`）；`NCCL_NET_REG_BUFFER` 时按算法填 `sendbuff/recvbuff/sendMhandle/recvMhandle` 与直发参数；CollnetDirect/Nvls 填 `specifics.collnetDirect`；`nPeers` 提示插件。

### 5.4 `ncclAddWorkBatchToPlan`——batch 组织（`enqueue.cc:121-201`）

把 work 按 `(workType, funcId)` 聚合到 channel 的 batch 链：

- **开新 batch 的条件**：队空；类型/funcId 变化；p2p 的 epoch/round/分组冲突（防挂死：同 round 不同组不能同 batch，注释 `enqueue.cc:142-149`）；p2p 超 `NCCL_MAX_DEV_WORK_P2P_PER_BATCH=8`；bcast 超单 batch 容量；workBytes 超 `NCCL_MAX_DEV_WORK_BATCH_BYTES`；
- **extension batch**：offset 不是 workSize 对齐时开 `nextExtends=1` 的扩展批（device 侧会把它们融合执行，且共享同一 proxyOpCount，`enqueue.cc:178-189`）；
- `offsetBitset |= 1<<(offset/workSize)` 记录本批哪些 work 槽有效（device 端 `loadWorkBatchToShmem` 依此装载）。

### 5.5 `finishPlan`——plan 定稿（`enqueue.cc:203-281`）

1. **Args 提升**：`sizeof(ncclDevKernelArgs) + batchBytes + workBytes ≤ comm->workArgsBytes` 时 `workStorageType = Args`（全部嵌进 kernel 参数，免 FIFO）；
2. 分配 `kernelArgs`（memScoped），填 `comm=comm->devComm / channelMask / workStorageType`；
3. **batch 的 round-robin 排布**（`enqueue.cc:224-247`）：各 channel 的 batch 链按 channel 升序交错排进 `batchZero[]`（紧跟 kernelArgs 之后），用 `nextJump`（相对索引）串起同 channel 的后续 batch——保证**每个 channel 的第一个 batch 正好在 `batchZero[blockIdx.x]`**，kernel 内按 blockIdx 直接定位；
4. **proxy op 归并**（`enqueue.cc:249-280`）：把 64 个 channel 的 `proxyOpQueue` 按 opCount **归并排序**进 `plan->proxyOpQueue`（比较前把 tag 位旋到高位：`id>>1 | id<<63`，让集合排在 p2p 前），并置 `plan->hasProxyOps`。归并保持同 channel 内 opCount 顺序——这是 proxy 端聚合 subs 的顺序保证。

### 5.6 p2p 调度（`scheduleP2pTasksToPlan` + `addP2pToPlan`，`enqueue.cc:866-1216`）

- 按 `p2pEpoch/p2pRound`（跨 plan 延续）逐 round 配对 send/recv，同 pair 融合进**一个 `ncclDevWorkP2p`**（kernel 内一个 work 同时处理收发）；
- channel 数按消息大小分档（`minPartSize/maxPartSize`，单节点允许 32x 超订）；协议按每 channel 负载选 LL（≤`NCCL_P2P_LL_THRESHOLD=16KB`）或 SIMPLE；网络 chunk 有专门收缩规则；
- `bytes=-1` 编码 no-op（占住 round 位置，保证各 rank batch 结构一致）；
- `allowUB`（user buffer）时尝试注册直传：网络经 `ncclRegisterP2pNetBuffer`（SIMPLE+同进程 proxy+非 PXN），IPC 经 `ncclRegisterP2pIpcBuffer`（要求双方注册，`NCCL_P2P_WRITE/READ`）。

---

## 6. 阶段六：`doLaunches`——逐 plan 发射（`group.cc:309-384`）

逐 clique 执行：

1. 每个 comm 先 `ncclLaunchPrepare`（§5），然后 `ncclCommIntraBarrierIn(comm, 1)`（`NCCL_LAUNCH_MODE=GROUP` 时）——**进程内多 GPU 的 barrier 发射模式**：所有本地 comm 对齐后才发 kernel，避免先后发射造成的跨 rank 偏斜；
2. 轮转发射循环：每轮每个 comm 弹出 `unlaunchedPlansHead` 一个 plan：
   - `ncclLaunchKernelBefore_NoUncapturedCuda`（barrier 内禁 CUDA 段）：`uploadWork`（见 §7.1）；
   - `ncclLaunchKernel` / `ncclLaunchCeColl` / `ncclLaunchRma`；
   - barrier 交换"是否还有 plan"；
   - `ncclLaunchKernelAfter_NoCuda`：若 proxy op 没走 host stream 回调则**就地调 `hostStreamPlanTask`**（同步模式常见路径）；
3. 无更多 plan 时 `ncclLaunchFinish` 收尾。

### 6.1 `ncclLaunchKernel`（`enqueue.cc:1753-1851`）

- `grid = {popcount(channelMask),1,1}`，`block = {threadPerBlock,1,1}`，`smem = ncclShmemDynamicSize(cudaArch)`（sym kernel 用 `plan->kernelDynSmem`）；stream = **第一个用户 stream**（`planner->streams->stream`）；
- 参数通过 `CU_LAUNCH_PARAM_BUFFER_POINTER/SIZE` 传 `plan->kernelArgs`（≤4KB，`ncclDevKernelArgs4K`，`device.h:496`）；
- 优先 `cuLaunchKernelEx`，按硬件代际加属性（`enqueue.cc:1774-1839`）：
  - sm90+：`CGA cluster`（`config.cgaClusterSize`，SPREAD 策略）、`MEM_SYNC_DOMAIN`（默认 Remote，`NCCL_MEM_SYNC_DOMAIN`）；
  - CUDA 12.3+：隐式序模式为 Launch 时挂 `LAUNCH_COMPLETION_EVENT`（`sharedRes->launchEvent`）；sym 集合开 `PROGRAMMATIC_STREAM_SERIALIZATION`；
  - sm100+：`NVLINK_UTIL_CENTRIC_SCHEDULING`（`config.nvlinkCentricSched`）；
- 老驱动回退 `cuLaunchKernel`。

### 6.2 `ncclLaunchFinish`（`enqueue.cc:1876-1954`）

- `finishedEvent` 记录到 launchStream；若 `workFifoProduced` 前进超过 `workFifoBytes/8`，挂一个 `KernelFinishCallback` 到 `eventCallbackQueue`（kernel 完成事件触发后回填 `comm->workFifoConsumed`，供 `waitWorkFifoAvailable` 节流——**每 1/8 FIFO 一次，摊薄事件开销**）；
- `deviceStream` 快进到 finishedEvent（`ncclStreamAdvanceToEvent`，graph 友好）；
- 每个用户 stream wait finishedEvent（多 stream 汇聚语义）；
- 隐式序：`launchOrder` wait `launchEvent`（CUDA 12.3+ 用 launch-completion 事件而非全完成事件，重叠度更高）；
- 释放 strong stream（deviceStream / launchOrder）。

---

## 7. work 上传与异步收尾

### 7.1 `uploadWork`——三种存储模式（`enqueue.cc:1248-1370`）

| 模式 | 何时 | 动作 |
|---|---|---|
| `Args` | 全部 work 塞进 4KB kernel 参数 | `workBuf=nullptr`，work 紧跟 batch 之后，`workMask=~0` |
| `Fifo`（默认非捕获） | 常态 | `waitWorkFifoAvailable(produced+workBytes)` 节流；16B 对齐 memcpy 进 `workFifoBuf` 环形位；推进 `workFifoProduced`；GDR 映射时 `wc_store_fence`；`kernelArgs->workBuf = workFifoBufDev` |
| `Persistent`（graph 捕获） | plan persistent | 16B 对齐 host 暂存 + `cudaMallocAsync`/`cudaMemcpyAsync` 到 deviceStream（graph 捕获期间禁 FIFO 复用） |

所有模式先把每个 batch 的 `offsetBase += fifoCursor`（把 plan 内零基偏移翻译成目标存储的绝对偏移，`enqueue.cc:1297-1300`）。

### 7.2 stream 依赖缝合（`ncclLaunchPrepare` 后半，`enqueue.cc:1643-1734`）

- launchStream（首个用户 stream）wait 其余用户 stream 与 deviceStream（非 graph 时经 strong stream 获取）；
- 隐式序非 None 时再 wait `context->launchOrder`（进程级发射序）；
- **proxy op 的 host 回调**：任一 plan `hasProxyOps` 且（persistent / `ncclCudaLaunchBlocking` / hostStream 忙）时，对每个 plan 在 `hostStream` 上 `cudaLaunchHostFunc(hostStreamPlanCallback, plan)`，并让 launchStream wait hostStream——**proxy op 上传被排成 stream 上的一个节点**，与 kernel 有序；
- persistent plan：`persistentRefs += nPlans` + `ncclCudaGraphAddDestructor(persistentDestructor)`。

### 7.3 异步收尾：`hostStreamPlanTask` 与 `reclaimPlan`

`hostStreamPlanTask`（`enqueue.cc:1436-1450`，在 CUDA host 回调线程或 `ncclLaunchKernelAfter` 里执行）：

1. profiler group/task 事件；
2. `uploadProxyOps`（`enqueue.cc:1392`）：重编号 opCount（最低位 p2p 标记；集合 `(collOpCount<<1)+id`、p2p `(p2pOpCount[channel]<<1)+id`）→ 正式 `ncclProxySaveOp` → 恢复旧编号（persistent 重放要用）；
3. `ncclProxyStart`——proxy op 进 SHM 池（详见 `notes/nccl_proxy_internals.md` §5.2）；
4. 非 persistent plan：把 `plan->reclaimer` 挂到 `comm->callbackQueue`，主线程下次进 group 时 `reclaimPlan`（`enqueue.cc:1462`）回收 plan/proxyOp/work 内存到各自 pool；persistent plan 由 graph 析构回调 `persistentDestructor`（`enqueue.cc:1528`）统一回收。

---

## 8. 关键数据结构清单

| 结构 | 位置 | 作用 |
|---|---|---|
| `ncclInfo` | `info.h:17` | API 调用的全部参数 + chunkSteps/sliceSteps |
| `ncclTaskColl` / `ncclTaskP2p` / `ncclTaskBcast` | `comm.h:193/258/237` | 用户任务（集合/p2p/bcast），规划后回填算法、warp 数、注册句柄 |
| `ncclTaskCollSorter` | `comm.h:361` | 按 trafficBytes 降序的桶排序器（1KB–1GB，每幂 4 桶） |
| `ncclKernelPlanner` | `comm.h:429` | 任务积累区（sorter/peers/streams）→ 待装配区（各 task/work 队列）→ WipPlan（64 channel 的 batch/proxyOp 队列）→ planQueue 的全周期状态 |
| `ncclKernelPlan` | `comm.h:307` | 一次 kernel 发射的全部信息：channelMask、threadPerBlock、kernelFn、kernelArgs(4K)、workStorageType、workQueue、proxyOpQueue、coll/p2p/bcast/rma 任务队列、persistent/isCeColl/isRma 标记；首成员 `reclaimer` 使 plan 自我回收 |
| `ncclDevWorkColl` / `ncclDevWorkP2p` | `device.h` | device 侧消费的 work（buffer、count、channelLo/Hi、cbd Lo/Mid/Hi 三段、redOpArg 等） |
| `ncclDevWorkBatch` | `device.h:473+` | 一个 channel 一批同 funcId work：`workType/funcId/offsetBase/offsetBitset/nextExtends/nextJump` |
| `ncclDevKernelArgs(4K)` | `device.h:473-519` | kernel 参数头：`comm/channelMask/workStorageType/workBuf/workMask` + 紧随的 batch 数组（≤4KB 经 `CU_LAUNCH_PARAM_BUFFER` 传入） |
| `ncclProxyOp` | `proxy.h:73` | 每 channel 一份的 proxy 消息（见 proxy 文档） |
| `ncclKernelPlanBudget` | `enqueue.cc:289` | plan 切分约束：kernel 参数内/外两组字节预算 |

---

## 9. 机制要点速记

1. **隐式 group**：单集合调用也是 `ncclGroupStartInternal + taskAppend + ncclGroupEndInternal`；所有规划/发射都发生在最外层 `GroupEnd`。
2. **确定性切分**：先集合后 p2p 的调度顺序、budget 切 plan、Lo/Mid/Hi 流量均衡，都是为了**所有 rank 产生逐位一致的 plan/batch 结构**（batch 结构不一致会挂死；注释 `enqueue.cc:1612-1615`、`142-149`）。
3. **三级内存管理**：`memPermanent`（跨 group 的 pool 后备）、`memScoped`（本 group 内有效，GroupEnd 弹栈）、`memPool_ncclTaskColl/P2p/KernelPlan/ProxyOp`（对象池，回收靠 `reclaimPlan`/`groupCleanup`）。
4. **plan = kernel 发射单元**：`channelMask` 决定 grid.x（一 block 一 channel）、`threadPerBlock` 取任务最大 `nWarps*32`（下限 128）；任务同质时用 `devFuncId` 特化 kernel，否则 `ncclDevKernel_Generic`。
5. **batch 排布保证**：round-robin 交错 + `nextJump` 相对索引，使 `batchZero[blockIdx.x]` 恒为该 channel 首 batch；`offsetBitset` 标记有效 work 槽。
6. **proxy op 的两次 life**：规划期进 channel `proxyOpQueue`（inquire 判定）→ finishPlan 归并排序 → 发射后由 host 回调重编号 opCount 并 `ncclProxySaveOp` 进 SHM 池（细节见 proxy 文档 §5.2）。
7. **发射对齐**：`NCCL_LAUNCH_MODE=GROUP` 时进程内 barrier 齐发；`doLaunches` 轮转逐 plan 发射；`ncclLaunchFinish` 用 `finishedEvent/launchEvent` 缝合用户 stream、deviceStream 与 launchOrder。
8. **graph 捕获约束**：同 group 全部 stream 同 graph 或都无捕获；persistent plan 的 work 走 `cudaMallocAsync` 而非 FIFO；proxy op 上传经 host stream 节点串进 graph；plan 生命周期挂到 graph 析构。
