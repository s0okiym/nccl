# NCCL 内部实现深度指南

> 面向有 **GPU 架构、C 系统编程、RDMA 通信**基础的工程师,目标是读懂 NCCL 2.30 的实现细节、
> 执行流程与关键机制。文中所有结论都附 `文件:行号`,可直接跳转源码核对。
> 阅读顺序建议:先 §0 建立心智模型 → §1 熟悉数据结构 → §3 跟随一次 all-reduce 走完全流程 →
> 再按需深入 §4–§11。源码阅读顺序见 §13。

NCCL 是 NVIDIA 的 GPU 间通信原语库(all-reduce / all-gather / reduce / broadcast /
reduce-scatter / send-recv / all-to-all / RMA)。它的本质是一个 **host 编排 + device 执行
+ CPU proxy 驱动网络** 的三层协作系统。掌握 NCCL 的关键,是理解这三层之间的**数据结构与
握手协议**,而不是单看某个函数。

---

## 0. 心智模型

一次 `ncclAllReduce()` 从用户调用到数据真正流动,经过三个角色:

```
 用户线程 (host)          GPU (device)              Proxy 线程 (CPU, 每 GPU 一个)
 ┌─────────────┐         ┌──────────────┐          ┌──────────────────┐
 │ ncclAllReduce│        │              │          │                  │
 │  → enqueue   │  把工作 │  集合通信内核 │  读共享环 │ ncclProxyProgress│
 │  → group end │ ─work──▶│  (ring/tree) │ ◀─buffer▶│ 驱动 NIC/SHM 进度│
 │  → launch    │  FIFO   │              │  生产/消费│                  │
 └─────────────┘         └──────────────┘          └──────────────────┘
       │                       ▲                          │
       │   ① 规划: 把请求翻译成 task / plan              │
       │   ② 把 plan 上传到 device(work FIFO)+proxy     │
       └───────────────────── ③ cuLaunchKernelEx ─────────┘
                               (网格 = 通道数,挂到用户 stream)
```

- **Host** 负责:校验参数、把请求切成 task、选算法/协议、把 work 描述符写进一个与 GPU 共享的
  FIFO、把 proxy 操作描述符投递给 proxy 线程、最后 `cuLaunchKernelEx` 启动 GPU kernel。
- **Device (GPU)** 负责:真正的数据搬运与归约。kernel 从共享 FIFO 读 work 描述符,通过
  per-channel 的 primitives 在 NVLink/PCIe/SHM 上搬运数据。
- **Proxy 线程** 负责:凡是涉及**网络**(IB/TCP)或需要 CPU 介入的传输(如 SHM 跨进程),GPU
  自己搬不动,就由 proxy 线程在 CPU 上驱动 NIC 收发、搬运 SHM buffer。它通过一个**与 GPU 共享
  的环形 buffer**(`ncclConnFifo`)与 GPU 协调进度。

这三者通过两类共享缓冲通信:
- **work FIFO**(`comm->workFifoBuf`,host↔device):host 把 work 描述符写进去,GPU kernel 读出来执行。
- **connection FIFO**(`ncclConnFifo`,device↔proxy):GPU 把数据填进一块固定大小的环形 buffer,
  推进 `sendHead`;proxy 轮询这个 head,把数据搬到 NIC 发出去。反过来 proxy 收到网络数据填进 buffer,
  推进 `tail`,GPU 轮询 `tail` 取数据。

> 类比你已知的 RDMA:NCCL 的 GPU↔proxy 协作本质上是**一套用门(head/tail 计数器)控制的环形
> 队列**,类似 verbs 的 send/recv WR ring,只不过生产者/消费者一个是 GPU 一个是 CPU,门是用
> 普通 store/load + fence 维护的。真正的 IB verbs 调用在 proxy 线程里完成(见 §9)。

---

## 1. 核心数据结构

掌握以下结构,就掌握了 NCCL 的"骨架"。均定义在 `src/include/`。

### 1.1 `struct ncclComm` —— 通信子中心状态

`src/include/comm.h` 的 `struct ncclComm { ... }`(从第 523 行起)。按功能分组:

- **拓扑与图**:`topo`(`ncclTopoSystem*`,系统拓扑树)、`graphs[NCCL_NUM_ALGORITHMS]`
  (每种算法一张图,`ncclTopoGraph`)、`rank`/`nRanks`/`nNodes`/`localRank`/`localRanks`、
  `rankToNode`/`rankToLocalRank`/`localRankToRank`、`nodeRanks`。
- **通道**:`channels[MAXCHANNELS]`(`ncclChannel`)、`nChannels`(已连接通道数)、
  `collChannels`(供集合通信用的通道数)、`nvlsChannels`、`p2pnChannels`。
- **task / plan 规划**:`planner`(`ncclKernelPlanner`,见 §3)、若干 `ncclMemoryPool`
  (`memPool_ncclTaskColl` 等,用于 task/plan/proxyOp 对象池)。
- **device 通信状态**:`devComm`(`ncclKernelComm*`,驻留显存,被 GPU kernel 直接读)、
  `workFifoBuf`/`workFifoBufDev`(host/device 两侧的 work FIFO 指针)、
  `workFifoBufGdrHandle`(GDR 优化时用)。
- **proxy**:`proxyState`(`ncclProxyState*`)。
- **共享资源**:`sharedRes`(`ncclSharedResources*`,同进程内 fork/clone 出来的 comm 共享一份;
  含 `ginState`/`rmaState`/`deviceStream`/`hostStream`/`launchEvent` 等)。
- **intra-process 协调**:`intraComm0`(同进程内多个 comm 的 leader)、`intraNext`、
  `intraBarrierCounter`/`intraBarrierGate`(用 64 字节对齐的 padding 避免假共享,实现同进程
  内 kernel 的轻量 barrier)。
- **插件上下文**:`netContext`/`ginContext`/`rmaContext`/`collNetContext`/`tunerContext`/
  `profilerContext`。
- **bootstrap**:`bootstrap`(out-of-band 握手状态)。
- **tuning 表**:`maxThreads[ALGO][PROTO]`、`bandwidths[coll][algo][proto]`、
  `latencies[coll][algo][proto]`(§4 用)。

### 1.2 `ncclChannel` —— 一条并行通信上下文

`src/include/comm.h:150`(`struct ncclChannel`)。一个 collective 被切成多份,在多个 channel 上并行流动。每个 channel(`comm.h:150-170`):
- 有一组 `ncclChannelPeer** peers`(环/树的邻居);
- 带 `collnetPeers`、`nvlsPeers`(供 SHARP / NVSwitch 集合用);
- 每个 peer 关系配一对 connector(send + recv)。

### 1.3 connector 与 transport vtable

`src/include/transport.h:117` 的 `ncclTransportComm` 是每个传输的**虚表**:既含 host 侧的
`setup`/`connect`/`free`,也含 proxy 侧的 `proxySetup`/`proxyConnect`/`proxyProgress`/
`proxyRegister`/`proxyDeregister`。`ncclTransport`(同文件)把 send/recv 两个 vtable 打包。
全局数组 `ncclTransports[]`(transport.h:34)索引:`p2pTransport` / `shmTransport` /
`netTransport` / `collNetTransport` / `profilerTransport`。

`ncclConnector`(transport.h:37 附近)持有一个传输的资源指针(`resources`)、host/pipe buffer
指针、以及 `proxyAppend[]`(proxy 端挂载点)。**connector 是把"逻辑 channel 邻居"绑定到"具体传输
实现"的粘合剂**。

### 1.4 task / plan / work / proxyOp 四级

这是 enqueue 路径的核心抽象(均见 `comm.h`):

- **`ncclTaskColl` / `ncclTaskP2p` / `ncclTaskBcast` / `ncclTaskRma`**(`comm.h:193/258/237/277`):
  用户一次请求翻译成的逻辑任务。`ncclTaskColl` 含 `func`、`op`、`datatype`、`count`、
  `sendbuff/recvbuff`、`algorithm`、`protocol`、`nChannels`、`nWarps`、`trafficBytes` 等。
- **`ncclKernelPlan`**(`comm.h:~315`):一组 task 被调度成一个 plan,对应**一次 kernel 启动**。
  含 `channelMask`(哪些 channel 参与本次启动)、`kernelFn`(device kernel 指针)、`kernelArgs`、
  `threadPerBlock`、work 队列、proxyOp 队列等。
- **`ncclDevWorkColl` / `ncclDevWorkP2p` / `ncclDevWorkBcast`**:下发到 device 的 work 描述符
  (一个 task 可能切成多个 work,跨多个 channel)。被串成 `ncclDevWorkBatch` 链表。
- **`ncclProxyOp`**(`proxy.h:73`):下发到 proxy 的操作描述符。

### 1.5 device 侧:`ncclKernelComm` 与 `ncclDevChannel`

驻留显存,被 GPU kernel 直接读:`ncclKernelComm`(全局通信状态,如 `workStarted`/`workCompleted`
profiler 环)和 `ncclKernelCommAndChannels`(comm + 所有 channel 数组)。GPU kernel 启动时把整个
这个结构 memcpy 进 shared memory(`src/device/common.h:382-413`)。

---

## 2. 初始化全流程

入口 `ncclCommInitRank` → `ncclCommInitRankDev`(`init.cc:2477`)→ 构造一个
`ncclCommInitRankAsyncJob` → `ncclAsyncLaunch` 异步执行 `ncclCommInitRankFunc`(`init.cc:1831`)。
真正干活的是 `ncclCommInitRankFunc`,它依次:

1. **`ncclInitKernelsForDevice`**(`enqueue.cc:34`):按当前 GPU 架构预编译/选择 device kernel。
2. **`commAlloc`**(`init.cc:441`):分配 `ncclComm`,填基础字段。若从 parent split 出来则继承 sharedRes。
3. **`bootstrapInit`**(`bootstrap.cc:674`):见下。
4. **`initTransportsRank`**(`init.cc:965`):见下。
5. **`devCommSetup`**(`init.cc:574`):分配 device 侧 `ncclKernelCommAndChannels` 并 memcpy 到显存。

### 2.1 UniqueId 与 bootstrap(out-of-band 握手)

RDMA/集合通信建立前,各 rank 互相不认识,需要一条**带外**信道交换连接信息。NCCL 用 TCP socket:

- `ncclGetUniqueId`(`init.cc:183`):由 rank 0 调用,内部 `bootstrapGetUniqueId`
  (`bootstrap.cc:430`)。它 fork 一个 **root 进程**(`bootstrapRoot`,`bootstrap.cc:288` 的线程函数),
  root 监听一个 socket,把地址塞进 `ncclUniqueId`(≤ `NCCL_UNIQUE_ID_BYTES`)。
- 所有 rank 拿到同一个 UniqueId 后调 `ncclCommInitRank`。`bootstrapInit`(`bootstrap.cc:674`)让每个 rank
  连到 root;root 给每个 rank 分配正式 rank 号,并把"全员的 socket 地址表"发回给每个人。
- 之后 `bootstrapAllGather`(`bootstrap.cc:1194`)、`bootstrapSend/Recv`(999/1094)就能在任意 rank 间
  **直接**点对点通信,无需再经过 root。这套机制只用于初始化阶段交换元数据,**不参与**正式数据通路。

### 2.2 initTransportsRank 的三阶段(`init.cc:965`)

#### 阶段 A:拓扑建立与图搜索
- `ncclTopoGetSystem`(`init.cc:1141`):探测系统拓扑树(GPU/NIC/PCI switch/CPU NUMA,见 §4)。
- `ncclTopoComputePaths`(`init.cc:1143`):在拓扑里为每对 rank 计算最优路径,决定用哪种 connector。
- `ncclTopoCompute`(`init.cc:1178` 起,每种算法一次):对 ring / tree / collnet-chain / collnet-direct /
  nvls 各跑一次图搜索,得到每张图的 `nChannels`、`bwIntra/bwInter`、`typeIntra/typeInter` 等。

#### 阶段 B:多轮 AllGather + channel 分配
初始化需要全 rank 同步多次,NCCL 把它命名为 AllGather1/2/3:
- **AllGather1**(`init.cc:1037`,`bootstrapAllGather` 交换 `ncclPeerInfo`):每 rank 的 hostHash、
  busId、CUDA 版本、`cuMemSupport`、`supportedGinType` 等;据此统计 `nNodes`/`localRanks`、
  校验版本一致、判断全局 GIN/跨 NIC 支持位(`globalGinSupport`,`init.cc:1019/1062`)。
- **AllGather3**(`init.cc:1283`,交换 `allGatherInfo`):每 rank 的图信息、`ncclTopoRanks`、
  CPU arch、`minNetBw`、`localNetDeviceCount` 等。`ncclTopoPreset`(`init.cc:1281`)用这些
  统一各 rank 对环/树拓扑的认识(谁是谁的 ring next/prev、tree parent/child)。

之后 `ncclTopoComputeP2pChannels`(`init.cc:1500`)决定 p2p 用的通道数,`ncclProxyCreate`
(`init.cc:1529`)起 proxy 线程,`ncclTransportP2pSetup`(`init.cc:1631`)建立 peer 间 connector。

#### 阶段 C:device 状态就位
`devCommSetup`(`init.cc:574`)分配显存里的 `ncclKernelCommAndChannels`,把 channel 信息拷过去,
`cudaMemcpyAsync` 到 device。最后 intra-node barrier(`init.cc:1694`)确保所有 rank 就绪。

### 2.3 `sharedRes` 与 comm 生命周期

`sharedRes`(`ncclSharedResources`)在**进程内第一个 comm** 创建时分配(`init.cc:~465`),含
`deviceStream`/`hostStream`/`launchEvent`/`scratchEvent`/`ginState`/`rmaState`。同进程内通过
`ncclCommSplit`/clone 出来的子 comm **共享同一份** `sharedRes`(引用计数 `refCount`)。这避免每个
comm 重复起 proxy、重复分配 device stream。

生命周期:`ncclCommFinalize`(`init.cc:2783`)→ `ncclCommDestroy`(`init.cc:2879`)→
`commReclaim`(真正释放)。释放按"逆分配"顺序,且 `commReclaim` 保证只有最后一个 rank 的最后
一次调用才真正回收 sharedRes/proxy(见 `init.cc:281` 注释)。

### 2.4 devcomm 版本兼容(`src/devcomm/`)

`devcomm_v22902.cc`/`devcomm_v22907.cc`/`devcomm_v23000.cc` 是 **per-CUDA-driver-version** 的
device communicator 兼容垫片。因为 device 侧的 `ncclKernelComm` 内存布局随 driver 版本变化,但
一个 libnccl 要能在不同 driver 版本上工作。`enhcompat.cc` 处理 enhanced-compat(同代 driver 间
的兼容)。`initTransportsRank` 末尾按探测到的 driver 版本选择对应 shim。

---

## 3. 一次集合通信的完整旅程(核心)

以 `ncclAllReduce` 为例,这是理解 NCCL 最重要的一节。

### 3.1 入口 → taskAppend

`ncclAllReduce`(`collectives.cc:168`)极薄,只构造一个 `ncclInfo`(含 func、buffer、count、
datatype、op、以及两个常量 `ALLREDUCE_CHUNKSTEPS`/`ALLREDUCE_SLICESTEPS`),然后
`ncclEnqueueCheck(&info)`(`collectives.cc:177`)。所有 collective 入口都是这个模式(`collectives.cc`
里十几个 `return ncclEnqueueCheck(&info)`)。

`ncclEnqueueCheck`(`enqueue.cc:3124`):
1. `CommCheck` 校验 comm;`ncclCommEnsureReady` 确保初始化完成。
2. **隐式包一层 group**:`ncclGroupStartInternal()`(enqueue.cc:3137)…… `ncclGroupEndInternal()`
   (enqueue.cc:3164)。所以即使用户没显式 `ncclGroupStart/End`,每次调用也是一个长度为 1 的 group。
3. `taskAppend(comm, info)`(`enqueue.cc:3159`):把请求转成一个 `ncclTaskColl`,塞进
   `comm->planner` 的 `collTaskQueue`(按 (func,op,datatype) 分桶)。
4. 若 group depth 归零,`ncclGroupEndInternal` 触发真正的规划与启动。

### 3.2 group end:规划三步曲

`ncclGroupEndInternal`(`group.cc:766`)→ `groupLaunch`(`group.cc` 内)对 group 里每个 comm:

**① `ncclPrepareTasksAndCollPreconnect`**(`group.cc:553`)→ `ncclPrepareTasks`(`enqueue.cc:363`):
- 把 bcast task 转成 coll task(若只有一个 bcast peer);用 `ncclTaskCollSorter`(按 size 分桶,
  `comm.h:361`)把 task 按**大小降序**取出。
- 按 `(func, op, datatype)` 分桶(`enqueue.cc:406-416`),对每个桶:
  - `ncclGetAlgoInfo`(`enqueue.cc:441`):**根据 task 大小 + §4 的 bandwidths/latencies 表,选出最优
    algorithm + protocol**,并算出 `nMaxChannels`、`nWarps`、`devFuncId`。
  - 大小相近(4×以内)的 task 聚合,共享算法选择(`enqueue.cc:431-470`)。
  - LL 协议会把 `trafficBytes *= 4`(因为 LL 带冗余 flag/size 头,`enqueue.cc:461`)。
- 按 `(isCollnet, isNvls)` 二维分桶(`collBins[2][2]`,`enqueue.cc:420`),决定 channel 划分。
- 对每个 task 调 `ncclRegisterCollNvlsBuffers` 等,必要时触发**运行时连接**(`runtimeConn`,
  `enqueue.cc:495`:第一次用到某算法才建对应 channel)。
- 若 `*needConnect`,这一步先去**预连接**(preconnect)channel 再返回(惰性建连)。

**② `ncclTasksRegAndEnqueue`**(`enqueue.cc:302`):遍历 `collTaskQueue`,为每个 task 构造
`ncclDevWorkColl`(填 sendbuff/recvbuff/root/nWarps/redOpArg 等,enqueue.cc:320-335),按
`regBufType`(NVLS / IPC / NET 注册缓冲)选 `ncclDevWorkColl` 或带注册信息的 `ncclDevWorkCollReg`,
挂进 `planner->collWorkQueue`。

**③ `doLaunches`**(`group.cc:309`):真正调度 + 上传 + 启动。对每个 plan:
- `ncclLaunchPrepare` → `scheduleCollTasksToPlan`(`enqueue.cc:576`):把 work 分配到各 channel,
  形成 `ncclKernelPlan`(带 `channelMask`)。budget 控制单个 plan 的大小,超了就开新 plan
  (`enqueue.cc:1617`),所以一个 group 可能产生多次 kernel 启动。
- `ncclLaunchKernelBefore_NoUncapturedCuda`(`enqueue.cc:1740`)→ **`uploadWork`**(见 3.3)。
- `ncclLaunchKernel`(`enqueue.cc:1753`)→ **`cuLaunchKernelEx`**(见 3.5)。
- `ncclLaunchKernelAfter_NoCuda` → **`hostStreamPlanTask`**(见 3.4)→ `uploadProxyOps`。

### 3.3 uploadWork:把 work 描述符写进共享 FIFO(`enqueue.cc:1248`)

work 描述符(`ncclDevWorkBatch` 链)有三种存储方式(`workStorageType`):
- **`ncclDevWorkStorageTypeArgs`**:work 直接打包进 kernel 参数(小消息)。
- **`ncclDevWorkStorageTypeFifo`**:写进 `comm->workFifoBuf`(一个与 device 共享的环形 FIFO,
  `enqueue.cc:1263-1269`)。host 是生产者(推进 `workFifoProduced`),GPU 是消费者。
  `waitWorkFifoAvailable`(`enqueue.cc:1216`)在快满时等 GPU 消费掉。
- **`ncclDevWorkStorageTypePersistent`**:CUDA Graph 捕获时,host 一次性 `cudaMallocAsync` 一块
  device buffer,`cudaMemcpyAsync` 把 work 拷进去,之后重放 graph 时无需再走 host(enqueue.cc:1270-1365)。

写完后,每个 `ncclDevWorkBatch` 的 `offsetBase` 被翻译成相对于 FIFO 基址的偏移(enqueue.cc:1297-1300)。

> GDR 优化:若 work FIFO 启用了 GDR(`workFifoBufGdrHandle != nullptr`),写完后要
> `wc_store_fence()`(enqueue.cc:1318)冲掉 WC(Write-Combining)缓冲,保证 GPU 通过 PCIe 读到最新值。

### 3.4 uploadProxyOps:把 proxy 操作投递给 proxy 线程(`enqueue.cc:1392`)

凡是需要 proxy 介入的传输(网络、SHM),plan 里会生成 `ncclProxyOp`。`hostStreamPlanTask`
(`enqueue.cc:1436`)先 `uploadProxyOps`,再 `uploadWork`(`enqueue.cc:1440` 顺序)。proxy op 经
`ncclProxySaveOp`(`proxy.h:422`)判重后,塞进 `proxyOps` 池,proxy 线程会在 §7 的循环里取走。

### 3.5 cuLaunchKernelEx:启动 GPU kernel(`enqueue.cc:1753`)

- **网格维度** `grid = {nChannels, 1, 1}`(每个 block = 一个 channel);`block = threadPerBlock`。
- 用 `cuLaunchKernelEx`(不是 `<<<>>>`,见 CONTRIBUTING 风格)配一堆 launch attribute:
  - **CGA / Thread Block Cluster**(`enqueue.cc:1792`,sm90+):`cgaClusterSize`,把多个 block 绑成一个
    cluster 跨 SM 协作,`CU_CLUSTER_SCHEDULING_POLICY_SPREAD`。
  - **Mem Sync Domain**(`enqueue.cc:1801`,sm90+,CUDA12+):默认 `Remote` 域,
    `NCCL_MEM_SYNC_DOMAIN` 可调。
  - **Launch Completion Event**(`enqueue.cc:1810`):把 `launchEvent` 绑定,用于 TMA 隐式序。
  - **Programmatic Stream Serialization**(`enqueue.cc:1816`,symmetric collectives)。
  - **NVLink Util-Centric Scheduling**(`enqueue.cc:1823`,sm100+ / Blackwell)。
- kernel 挂到 `planner->streams->stream`(即用户传的 stream)。

### 3.6 GPU kernel 执行(详见 §5、§6)

kernel 入口 `ncclKernelMain`(`device/common.h:356`):block→channel 映射、把 comm/channel/work
载入 shared memory、循环执行 work batch,每个 work dispatch 到
`RunWorkColl<Fn,T,RedOp,Algo,Proto>::run()`,后者调 `Primitives`(§6)做实际搬运。

### 3.7 小结:谁动谁的数据

| 阶段 | host | device | proxy |
|---|---|---|---|
| 规划 | taskAppend→PrepareTasks→Reg | — | — |
| 上传 | uploadWork(FIFO)、uploadProxyOps | — | — |
| 启动 | cuLaunchKernelEx | 载入 work | — |
| 执行(p2p/shm/nvls) | — | primitives 直接搬 | (不参与) |
| 执行(网络) | — | 填环形 buffer、推进 head | 轮询 head、发 IB verbs、推进 tail |

---

## 4. 拓扑发现与算法/协议选择

### 4.1 拓扑树

`src/graph/topo.cc` 构建一棵多类型节点树:`ncclTopoSystem` 为根,下挂 GPU / NIC / PCI / CPU(NUMA)
节点,节点间用 `ncclTopoLink`(带宽 `bw`、类型)相连。探测来源:
- **GPU/NIC/PCI**:走 sysfs(`topo.cc:436` 起读 `/sys/.../pci`),识别 PCI switch 拓扑(`getBcmGen` 判
  Broadcom 芯片代,`topo.cc:207-257` 做坍缩)。
- **XML**:`xml.cc` 解析拓扑 XML;`NCCL_TOPO_FILE` 可加载人工拓扑覆盖自动探测
  (调试无硬件环境时极有用)。
- 带 nvml / IB verbs 查 NIC 能力。

不同 link 类型的带宽估值(`bwIntra/bwInter` 单位 GB/s)决定后续选择。

### 4.2 图搜索

`search.cc` 的核心是一组递归 `ncclTopoSearch*` 函数:为目标算法(ring/tree/collnet/nvls)在拓扑上
**搜索一组通道**,使得每条通道上,链路带宽被"除以"经过它的数据流条数后(所谓 `RecvDivBw`/
`SendDivBw`)尽量高。`ncclTopoCompute`(`topo.cc`)对每种算法跑一遍,产出 `ncclTopoGraph`:
`pattern`、`nChannels`、`sameChannels`、`bwIntra`、`bwInter`、`typeIntra`、`typeInter`、`crossNic`。

- **ring**(`rings.cc`):构造环,每 rank 有 prev/next,数据沿环转一圈完成 all-reduce/reduce-scatter+
  allgather。带宽利用最优(2(n-1)/n),但延迟 ∝ n。
- **tree**(`trees.cc`):构造**双工二叉树**(tree0 上行、tree1 下行,`maxTreePattern`),延迟 O(log n),
  适合小消息。
- **collnet-direct / collnet-chain**:SHARP 网内归约(交换机做 reduction)。direct 要求每 GPU 一个本地 NIC。
- **nvls / nvls-tree**:NVSwitch 集合(NVLink SHARP),单节点内 NVSwitch 全互连时极快。

`paths.cc`/`connect.cc`:决定每对 rank 间用 p2p(NVLink/PCIe 直连)、shm(同进程/同节点)还是 net。

### 4.3 tuning:离线填表 + 在线查表

`tuning.cc` 的 `ncclTopoTuneGraph`(`tuning.cc:~230` 起)在初始化末尾,**对每个
(coll, algo, proto) 组合**预算一张表:
- `comm->bandwidths[coll][a][p]`(`tuning.cc:383`):由 `bwIntra/bwInter` 推 bus bandwidth,再按算法
  比例换算成 algo bandwidth。含大量经验修正:LL 协议打 0.5 折(`tuning.cc:327`)、tree LL 打 1/3.8
  (`tuning.cc:332`)、PAT 打 0.75(`tuning.cc:337`)、collnet-direct 的 GPU/NIC 比因子(`tuning.cc:345`)等。
- `comm->latencies[coll][a][p]`(`tuning.cc:384`):由 base + hw latency + 环/树步数 × link latency 推
  延迟。环:intra/inter step 分开算(`tuning.cc:411`);树:`2×(intraSteps + log2(nNodes)×interLat)`
  (`tuning.cc:416`)。

**在线查表**发生在 `ncclGetAlgoInfo`(`enqueue.cc:441`,在 §3.2 步骤①被调):给定 task 大小,遍历合法的
(algo, proto),用 `bandwidths`/`latencies` 估算完成时间(`time = size/bw + latency`),取最快的。算法合法
性约束见 `tuning.cc:294-308`(如 bcast/reduce 只用 ring;allgather/reducescatter 可用 PAT/RING/NVLS/collnet;
allreduce 不用 PAT;NVLS 只配 SIMPLE)。

算法枚举:`NCCL_ALGO_RING` / `_TREE` / `_COLLNET_DIRECT` / `_COLLNET_CHAIN` / `_NVLS` / `_NVLS_TREE` / `_PAT`。
协议枚举:`NCCL_PROTO_LL` / `_LL128` / `_SIMPLE`(协议含义见 §6)。可用环境变量 `NCCL_ALGO`/`NCCL_PROTO`
强制限定选择集。

---

## 5. 设备端执行引擎

### 5.1 kernel 主循环(`src/device/common.h:356` `ncclKernelMain`)

1. **把 kernel 参数 memcpy 进 shared memory**(`common.h:362-364`,用前若干线程协同拷),避免编译器
   把参数放进 thread-local stack。
2. **blockIdx → channelId 映射**(`common.h:370-373`):`channelMask` 是个位图,第 n 个置位 bit 对应
   blockIdx.x=n。用 `__popcll` 算"前 n 位有几个 1"。
3. **三组 warp 协同载入**(充分利用 warp 并行,`common.h:382-413`):
   - warp 0:载入 `ncclKernelComm`(全局 comm 状态);
   - warp 1:载入当前 `ncclDevChannel`;
   - 其余 warp:载入 work batch(`loadWorkBatchToShmem`)。
4. **主循环**(`common.h:417-431`):反复执行 work batch——若 `SpecializedFnId` 匹配则直接调特化版
   `SpecializedRunWorkBatch().run()`(编译期特化,快路径),否则走通用 `ncclDevFuncTable[funcId]()`
   函数表(`common.h:422`,表在 `common.cu:23` 附近,由 `generate.py` 生成大量 `<coll,redop,ty,algo,proto>`
   组合的 `DEFINE_ncclDevFunc`)。处理完一个 batch 看 `nextBatchIx` 是否还有下一个(扩展 batch 机制,
   见 §3.2 work batch)。

### 5.2 work batch 执行(`RunWorkBatch`,`common.h:287`)

对 batch 里每个 `ncclDevWorkColl`(`common.h:303-315`):按 `work->nWarps` 决定多少 warp 参与
(`subtn = nWarps * WARP_SIZE`),调 `RunWorkColl<Fn,T,RedOp,Algo,Proto>().run(tid, subtn, work)`。
不同 nWarps 的 work 之间 `__syncthreads()`(`common.h:308`)。

### 5.3 kernel 变体生成(`generate.py`)

NCCL 的 device kernel 是**模板全特化**:`RunWorkColl<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE>`
等。`generate.py` 对所有合法组合 × 数据类型 × 归约算子生成 `DEFINE_ncclDevKernel` / `DEFINE_ncclDevFunc`
(`common.h:438/446`),靠编译器死代码消除 + 链接器去重压体积。`ncclInitKernelsForDevice`(`enqueue.cc:34`)
在运行时按 `cudaArch` 选可用 kernel 集。

### 5.4 CE Collectives / RMA / symmetric

除主路径外还有:
- **CE (Copy Engine) collectives**(`ceColl`、`ncclLaunchCeColl`,`enqueue.cc:359`):用 CE 异步搬运,
  和 compute kernel 并行。
- **RMA**(`rma/`,one-sided put/signal/wait):`ncclLaunchRma`(`enqueue.cc:361`)。
- **Symmetric collectives**(`scheduler/`、`isSymColl`):对称内存调度,launch 配 programmatic stream
  serialization(`enqueue.cc:1816`)。

---

## 6. Primitives 与三种协议

集合算法(ring/tree/...)本身是高层逻辑,真正的"把字节从 A 搬到 B 并做归约"由 **primitives** 完成。

### 6.1 `Primitives` 模板(`src/device/primitives.h:116`)

`Primitives<T, RedOp, Fan, Direct, Proto, P2p>` 是核心抽象,`Fan` 指定收发扇出(如
`FanSymmetric<1>` = 环上一个 prev 一个 next;`FanAsymmetric<N,1>` = 树上 N 个 child 1 个 parent)。
它封装:从 pipe buffer 收数据、做归约、发数据、与 proxy 协调进度。集合 kernel(如
`all_reduce.h:31` 的 `runRing`)拿到 prev/next connector,构造 Primitives,循环 `send/recv/reduce`。

### 6.2 三种协议(prims_ll / prims_ll128 / prims_simple)

协议决定**数据在 pipe buffer 里的格式与同步方式**,是延迟/带宽的权衡:

- **`NCCL_PROTO_LL`(Low Latency, `prims_ll.h`)**:每个 64-bit 数据附带 64-bit flag + size 头(所以
  `trafficBytes *= 4`,`enqueue.cc:461`)。用 flag 做生产者-消费者同步,**延迟最低但带宽最差**,
  用于小消息。带宽打折到 ~0.5(tuning.cc:327)。
- **`NCCL_PROTO_LL128`(128-byte Low Latency, `prims_ll128.h`)**:128 字节(8×128bit)为一个单元,
  用 `ldmatrix`/`stmatrix` 等 PTX + 一个 flag 字(`sendFlag/recvFlag`)同步。**大带宽且低延迟**,
  是现代 GPU 上的主力协议(要求 sm80+)。关键函数:`postSend`/`postRecv`(`prims_ll128.h:84/87`)、
  `waitSend`(`prims_ll128.h:70`)、128-bit 对齐的 shmem load/store(`prims_ll128.h:127-170`)。
- **`NCCL_PROTO_SIMPLE`(prims_simple)**:大块搬运,用完整 NCCL_STEPS 深度的环形 buffer + head/tail
  门同步。**纯带宽导向**,用于大消息。proxy 侧也最简单(`sendProxyProgress` 的三段流水,§9)。

> 类比 RDMA:LL 像 small-message send/recv(每条带元数据,低延迟低带宽);LL128/SIMPLE 像
> 大块 RDMA write/read(批量、流控靠门)。

### 6.3 一个例子:all-reduce ring(`src/device/all_reduce.h`)

`RunWorkColl<ncclFuncAllReduce,T,RedOp,NCCL_ALGO_RING,NCCL_PROTO_SIMPLE>`(`all_reduce.h:229`)
→ `runRing`(`all_reduce.h:14`):构造
`Primitives<T,RedOp,FanSymmetric<1>,1,Proto,0>(tid,nthreads,&ring->prev,&ring->next,sendbuff,...)`
(`all_reduce.h:31`),然后对每个 chunk:`recv` from prev → `reduce`(与 sendbuff)→ `send` to next。
树算法见 `runTreeUpDown`/`runTreeSplit`(`all_reduce.h:86/146`)。

---

## 7. Proxy 线程引擎(`src/proxy.cc`)

每个 GPU 起一个 proxy 线程(`ncclProxyProgressCreate`,`proxy.cc:1032` 起 `std::thread`)。

### 7.1 主循环(`ncclProxyProgress`,`proxy.cc:954`)

```
do {
  progressOps(... active op 链 ...);        // 推进已有 op
  if (idle 或到 append 频率) ncclProxyGetPostedOps();  // 取新投递的 op
  yield if nothing added;
} while (!stop || active);
```

- **`progressOps`**(`proxy.cc:801`):遍历 `state->active` 链表,对每个 `ncclProxyArgs` 调
  `op->progress(proxyState, op)`(`proxy.cc:810`)—— 这个函数指针就是**传输 vtable 的
  `proxyProgress`**(net / shm / p2p 各自的实现,§8)。完成(`state == ncclProxyOpNone`)则 `removeOp`。
- **`ncclProxyGetPostedOps`**(`proxy.cc:835`):从 `opsPool`(生产者=host 线程,消费者=proxy 线程,
  用 mutex+cond 协调)取出新 `ncclProxyOp`,经 `ProxyAppend`(`proxy.cc:441`)挂到对应 connector 的
  `proxyAppend[]`,并把 op 归还 pool(无锁 freelist,`proxy.cc:897-909`)。`proxyOpAppendCounter`
  (`proxy.cc:977`)控制 append 频率以减少对小消息的扰动。

### 7.2 op 生命周期状态机

`ncclProxyArgs.state`:`ncclProxyOpNone` → `ncclProxyOpReady` → `ncclProxyOpProgress` → 完成。
每次 `proxyProgress` 把当前步推进一点(idle=1 表示本轮没活干)。`NCCL_STEPS`(默认 8)是环形 buffer
深度;`sliceSteps`/`chunkSteps` 控制每步大小。

### 7.3 与 GPU 的握手

proxy 不直接调 GPU,而是读写**与 GPU 共享的环形 buffer**(`ncclConnFifo` + 收发缓冲)。GPU 往缓冲写
数据并推进 `sendHead`,proxy 读 `sendHead` 取数据发网络;proxy 收到网络数据写缓冲推进 `tail`,GPU 读
`tail` 取数据。这套 head/tail 门就是 §0 说的"device↔proxy 环形队列"。

---

## 8. 传输层抽象与各后端

`ncclTransportComm` vtable(`transport.h:117`)的 host 方法在 `connect` 阶段建好 connector,proxy 方法
在 §7 的循环里被回调。各后端:

- **p2p(`transport/p2p.cc`)**:GPU↔GPU 直连(NVLink / PCIe P2P / CUDA IPC)。通常**不经 proxy**,
  GPU kernel 直接读写对端显存(`Direct=1` 的 Primitives)。`canConnect` 判 `NCCL_P2P_LEVEL`。
- **shm(`transport/shm.cc`)**:同节点跨进程,用共享内存 + 命名信号量。需 proxy 搬运(或 GPU 直接
  map)。
- **nvls(`transport/nvls.cc`)**:NVSwitch / NVLink SHARP 集合,reduce 由交换硬件做。
- **net(`transport/net.cc` + `net_ib/` / `net_socket.cc`)**:跨节点。**必经 proxy**(proxy 调 IB verbs)。
  `net.cc` 是**网络层抽象**,底下可插外部 plugin 或内置 IB/socket 实现。
- **coll_net(`transport/coll_net.cc`)**:SHARP in-network reduction 的 NCCL 侧适配。
- **profiler(`transport/profiler.cc`)**:插桩"伪传输"。

> 网络插件机制:`plugin/net/` 允许外部 `libnccl-net.so`。`netContext` 持有选中的 net vtable。
> 真实 IB/socket 实现在 Linux 上编译,非 Linux 用 `*_stub.cc`(`src/Makefile` 的 `TRANSPORT_CC` 规则)。

---

## 9. 网络 / RDMA 实现细节

`src/transport/net.cc` 的 `sendProxyProgress`(`net.cc:1304`)/`recvProxyProgress`(`net.cc:1470`)
是网络传输的心脏。它们与底层 IB verbs(`transport/net_ib/`)配合,实现**三段流水**:

```
posted ──▶ transmitted ──▶ done
 (GPU 往缓冲填)   (proxy 发 NIC)   (收到 ACK)
```

### 9.1 sendProxyProgress 三段(`net.cc:1304`)

1. **`ncclProxyOpReady` → 初始化**(`net.cc:1306-1319`):对每个 sub,算 `base`(对齐到 chunkSteps)、
   清 `posted/transmitted/done`、取已注册的 memory handle(`sendMhandle`)。
2. **posted 段(把缓冲让给 GPU)**(`net.cc:1335-1356`):在深度未超 `maxDepth`(`NCCL_STEPS`,
   `net.cc:1323`)时,推进 `posted`,**写 `sendHead`**(`net.cc:1348`:
   `*sendHead = base + posted - NCCL_STEPS`)告诉 GPU"你可以往这个 slot 填了"。GDR 优化路径用
   `gdcSync` + `wc_store_fence()`(`net.cc:1349`)。
3. **transmitted 段(发 NIC)**(`net.cc:1358` 起):`transmitted < posted` 说明 GPU 已填好,proxy 调
   底层 `isend`(`ncclNet->isend`,最终是 `ibv_post_send` RDMA-write/send)把它发出去,推进 `transmitted`。
4. **done 段**:收到对端 ACK(`ipull`/`ibv_poll_cq`),推进 `done`;`done == nsteps` 时 op 完成。

recv 侧对称:`recvProxyProgress`(`net.cc:1470`)预 post 接收缓冲,收到数据后推进 `tail` 让 GPU 取。

### 9.2 关键 RDMA / GDR 点

- **GPUDirect RDMA(GDR)**:NIC 直接 DMA 显存,绕开 host bounce buffer。`NCCL_IB_GDR_LEVEL` 控制
  何时启用(各级 `PhB/Pxi/Rel/Sys`)。proxy 侧 `NCCL_NET_MAP_GET_POINTER(&map, cpu, buffs)` 取到的缓冲
  可能在 host 也可能在 device。
- **GDRCOPY**(`net.cc:339-341`):用 GDRCOPY 做 GPU→CPU 的小量同步(flag/tail 通知),
  `NCCL_GDRCOPY_SYNC_ENABLE` / `NCCL_GDRCOPY_FLUSH_ENABLE` 控制(`net.cc:1642` 附近)。这是降低小消息
  延迟的关键——比走 PCIe MMIO 读 GPU 内存快得多。
- **memory registration**:`proxyRegister`/`proxyDeregister` vtable 把显存/主机内存注册给 NIC
  (`ibv_reg_mr` / dma-buf)。NCCL 缓存注册结果(`regCache`,`comm.h:252`)避免重复 reg/dereg。
- **IBverbs 封装**:`transport/net_ib/common.cc` 封装 QP/CQ/MR;`net_ib/init.cc` 探测 IB 设备;
  `net_ib/gdr.cc` GDR 支持;`net_ib/p2p.cc` NIC↔GPU P2P;`net_ib/p2p_resiliency*.cc` 链路容错恢复。

### 9.3 共享环形缓冲(`ncclConnFifo`)

每个 send/recv connector 有一块深度 `NCCL_STEPS`、每槽 `stepSize` 字节的环形缓冲 + 一个
`ncclConnFifo[NRANKS]` 数组(每槽存 `offset`/`tail`/`head`/`sizes`/`mhandles`)。GPU 和 proxy 通过
这个结构协调"第 k 步数据在哪个 slot、谁生产谁消费"。

---

## 10. GIN 机制(GPU-Initiated Network)

GIN 让 GPU 直接发起 NIC 工作,绕开 CPU proxy,进一步降延迟。三层条件(`src/plugin/gin.cc`、
`src/transport/net_ib/gin.cc`、`src/include/nccl_device/gin/`):

1. **编译期**:仅 Linux + IB。两种后端各有架构门槛(`nccl_device/gin/gin_device_common.h:20-37`):
   GDAKI 需 `CUDA≥12.2 && sm≥7.0`;GPI 同;PROXY 后端默认开启。
2. **运行期探测** `ncclGinInit`(`init.cc:477` 调,`plugin/gin.cc:269`):GPU 计算能力 ≥ 70 才继续;
   按优先级尝试插件(外部 `libnccl-gin.so` → NET 插件 GIN → 内置 GDAKI `ncclGinIbGdaki`
   需 MLX5+GDR → 内置 PROXY `ncclGinProxy` fallback)。成功则置 `ginState.ginType` 非 NONE。
3. **通信子级激活**(`dev_runtime.cc:~1138` 置 `devr->ginEnabled = true`):**必须显式请求**
   `ginConnectionType = FULL/RAIL`(或已废弃 `ginForceEnable`)才真正激活,且要求通信子内**所有 rank
   支持同类型**(`init.cc:1062` `globalGinSupport`)。

**普通 PyTorch 训练不会启用 GIN**:标准 NCCL API(`ncclCommInitRank` 等)不暴露 `ginConnectionType`,
默认 `NCCL_GIN_CONNECTION_NONE`。GIN 的真实用户是 `contrib/`(`nccl_ubx`/`nccl_ep`/`nccl_m2n`/
`nccl_checkpoint`)和 RMA/symmetric 路径。相关环境变量:`NCCL_GIN_PLUGIN`/`NCCL_GIN_TYPE`/`NCCL_GIN_IB_TC`/
`NCCL_GIN_PROXY_QUEUE_SIZE`。GIN 详见 `docs/contrib/GIN/`。

---

## 11. 流、group、CUDA Graph 集成

### 11.1 group 机制(`src/group.cc`)

`ncclGroupStart/End` 用一个全局 `groupState`。Start 时 depth++,End 时 depth--,**depth 归零才触发
`groupLaunch`**(`group.cc` 内)。多 comm 在 group 内通过 `groupNext[ncclGroupTaskTypeNum]` 串成链表,
分 collective / p2p 两类任务类型。group 的意义:**把多个 collective 合并规划、共享一次 kernel 启动
开销、允许 send/recv 配对**。

### 11.2 与 CUDA stream 的绑定

用户传的 `stream` 被存进 `planner->streams`。NCCL 把 kernel launch 挂到这个 stream,于是
NCCL 操作与用户其它 stream 操作有正确的 CUDA 流序。`sharedRes->deviceStream`(NCCL 自己的)用于
内部 memcpy(如 persistent work 上传);`sharedRes->hostStream` 用于 host-side callback(proxy op
投递、reclaim)。`strongstream`(`include/strongstream.h`)抽象"在 graph 捕获下也能正确获取的 stream"。

### 11.3 CUDA Graph / persistent plan

若检测到 stream capture(`ncclCudaGraphValid(planner->capturingGraph)`),走 persistent 路径:
work 用 `ncclDevWorkStorageTypePersistent`(`enqueue.cc:1270`)一次性拷到 device buffer,plan 复用;
`groupLaunch` 里 `capturingYes && capturingNo` 会报错——一个 group 内要么全捕获要么全不捕获
(`group.cc:329`)。persistent plan 用 `localPersistentRefs`/`persistentRefs` 引用计数管理
(`comm.h:140/193`)。

### 11.4 intra-process barrier(`group.cc:309` `doLaunches`)

同进程内多个 comm(多个 GPU)的 kernel 要协同(尤其 group launch 模式)。NCCL 用 `intraBarrierCounter`
/`intraBarrierGate`(64 字节对齐,`comm.h:156-159`)实现一个**无系统调用的轻量 barrier**,避免
pthread barrier 的开销。`ncclParamLaunchMode`(group vs parallel)决定是否用它。

---

## 12. 调试与环境变量速查

- `NCCL_DEBUG=INFO`(或 VERSION/SUBSYS):日志。常用 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH`
  看初始化与图选择;`COLL` 看每次 collective。
- `NCCL_ALGO` / `NCCL_PROTO`:限定算法/协议(如 `NCCL_PROTO=LL128`)。
- `NCCL_TOPO_FILE`:加载人工拓扑 XML。
- `NCCL_P2P_LEVEL` / `NCCL_P2P_DISABLE`:控制 P2P。
- `NCCL_IB_*`(`_GDR_LEVEL`/`_DISABLE`/`_HCA`/`_TC`/`_SL`/`_ADAPTIVE_ROUTING`):IB 行为。
- `NCCL_NET`:选网络后端(如 `NCCL_NET=IB`)。
- `NCCL_GIN_*`:见 §10。
- `NCCL_LAUNCH_MODE`(`GROUP`/`PARALLEL`):见 §11.4。
- `NCCL_PROXY_DUMP_SIGNAL`:给 proxy 线程发信号 dump 状态(`proxy.cc:964-965`,调试卡死用)。
- `ncclparam` 二进制(`make` 后 `build/bin/ncclparam`):列出所有 `NCCL_*` 参数及默认值。
- `ncclras` 二进制:RAS 遥测客户端。

排查"卡住"的典型路径:看 proxy 线程(`NCCL Progress*`)是否在跑 → 看 work FIFO 是否被消费 →
看 `sendHead/tail` 是否推进 → 看 IB CQ 是否有未完成 WR。

---

## 13. 推荐源码阅读顺序

1. **建立全局观**:`src/include/comm.h`(ncclComm)、`src/include/transport.h`(vtable)、
   `src/include/proxy.h`(ProxyOp)。先只读结构定义,不看实现。
2. **初始化**:`init.cc:1831` `ncclCommInitRankFunc` → `bootstrap.cc:674` `bootstrapInit` →
   `init.cc:965` `initTransportsRank`(重点 1137–1690,拓扑/图/AllGather/连接)。
3. **数据路径(最重要)**:`collectives.cc:168` `ncclAllReduce` → `enqueue.cc:3124` `ncclEnqueueCheck`
   → `enqueue.cc:363` `ncclPrepareTasks` → `enqueue.cc:302` `ncclTasksRegAndEnqueue` →
   `group.cc:309` `doLaunches` → `enqueue.cc:576` `scheduleCollTasksToPlan` →
   `enqueue.cc:1248` `uploadWork` → `enqueue.cc:1753` `ncclLaunchKernel`。
4. **device 执行**:`device/common.h:356` `ncclKernelMain` → `device/common.h:287` `RunWorkBatch` →
   `device/all_reduce.h` `runRing` → `device/primitives.h` → `device/prims_ll128.h`(看
   `postSend`/`postRecv`/`waitSend`)。
5. **proxy 与网络**:`proxy.cc:954` `ncclProxyProgress` → `proxy.cc:801` `progressOps` →
   `net.cc:1304` `sendProxyProgress` → `transport/net_ib/common.cc`(verbs 封装)。
6. **选型**:`graph/topo.cc` → `graph/search.cc` → `graph/tuning.cc`(`ncclTopoTuneGraph`)。
7. **进阶**:GIN(`plugin/gin.cc`)、RMA(`rma/`)、scheduler(`scheduler/`)、devcomm 兼容(`devcomm/`)。

配套实操:用 nccl-tests 跑一个 `all_reduce_perf`,`NCCL_DEBUG=INFO` 对照日志走一遍上面的函数链;
再用 `NCCL_ALGO`/`NCCL_PROTO` 强制切换,观察 `ncclGetAlgoInfo` 选型的差异。

---

## 14. 术语速查

| 术语 | 含义 |
|---|---|
| channel | 一条并行通信上下文,collective 被切到多 channel 上并行 |
| connector | 把 channel 邻居绑定到具体 transport 的粘合对象(send/recv 各一) |
| task | 用户一次请求翻译成的逻辑任务(Coll/P2p/Bcast/Rma) |
| plan | 一组 task 调度成的一次 kernel 启动(`ncclKernelPlan`) |
| work | 下发到 device 的描述符(`ncclDevWorkColl` 等),串成 work batch |
| proxyOp | 下发到 proxy 线程的操作描述符 |
| work FIFO | host→device 的 work 描述符环形队列(`comm->workFifoBuf`) |
| connFifo | device↔proxy 的数据/进度环形队列(`NCCL_STEPS` 深) |
| algorithm | ring/tree/collnet/nvls/pat —— 数据怎么在 rank 间流转 |
| protocol | LL/LL128/SIMPLE —— 数据在 pipe buffer 里的格式与同步方式 |
| LL / LL128 | 低延迟协议,带 flag 同步 |
| SIMPLE | 纯带宽协议,大块 + head/tail 门 |
| sharedRes | 同进程内 comm 共享的资源(stream/event/gin/rma state) |
| GDR | GPUDirect RDMA,NIC 直接 DMA 显存 |
| GDRCOPY | GPU→CPU 小量同步拷贝,降小消息延迟 |
| GIN | GPU-Initiated Network,GPU 直接发 NIC 工作 |
| bootstrap | 初始化用的带外 socket 握手,不参与数据通路 |
| CE coll | Copy Engine collective,用 CE 异步搬运 |
| CGA | Cooperative Group Array / Thread Block Cluster(sm90+) |
