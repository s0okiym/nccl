# NCCL 核心机制深度详解

本文是 [`nccl_internals.md`](./nccl_internals.md) 的配套深度文档。概览文档给出"全景",
本文针对四个最重要的机制,展开到 **数据结构字段、状态机、时序图、代码逐行** 的深度。

阅读前提:已读过概览文档,了解 host/device/proxy 三层、channel/connector/transport、
algorithm/protocol 的基本概念。本文所有 `文件:行号` 引用均针对当前 master 代码。

涉及的两个系统级抽象,贯穿全文:

| 抽象 | 含义 | 关键常量 |
|------|------|----------|
| **Work FIFO** | host → device 的单向命令流 | `NCCL_MAX_DEV_WORK_BATCH_COLLS` |
| **connFifo + head/tail** | device ↔ proxy(或 device↔device)的数据握手 | `NCCL_STEPS = 8` (`src/include/device.h:26`) |

---

## 机制一:数据通路同步机制

NCCL 的同步要解决两类问题:(1) host 如何把"要做什么"安全地交给 GPU;
(2) GPU/代理之间如何在不加锁的情况下,对一个深度为 8 的环形缓冲区做生产—消费。
这两者各有一条独立的"轨道"。

### 1.1 核心数据结构

#### host 侧 —— communicator 的连接信息 `ncclConnInfo`

`src/include/device.h:133`

```c
struct ncclConnInfo {
  char*  buffs[NCCL_NUM_PROTOCOLS]; // Local for recv, remote for send
  void*  mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t* tail;   // Local for recv, remote for send
  uint64_t* head;   // Local for send, remote for recv
  int    flags;     // Direct 通信等标志
  int    shared;    // 缓冲区是否共享
  int    stepSize;  // SIMPLE 协议单 step 的字节数
  void** ptrExchange;          // direct 通信时交换指针
  uint64_t* redOpArgExchange;  // direct pull 时交换预缩放系数
  struct ncclConnFifo* connFifo; // device↔proxy 控制通道
  uint64_t step;     // "我推进到哪了"
  uint64_t llLastCleaning;
  ncclNetDeviceHandle_t netDeviceHandle;
};
```

`tail` / `head` 是两个 `uint64_t*` 指针。**这是整个同步机制最易被误读的地方**:
变量名叫 head/tail,但它们的真实语义不是"FIFO 头/尾",而是:

> - **`tail` = "数据就绪"计数器**:由数据**生产方**推进,表示"数据已填充到第 tail 步";
>   **消费方**在其上自旋等待。
> - **`head` = "信用/可用"计数器**:由**消费方**推进,表示"我已消费/释放到第 head 步";
>   **生产方**在其上自旋等待(等缓冲槽空出来)。

后面 1.3 节会从 connect() 的指针布线和 device 端的读/写角色两条独立证据链证明这一点。
`ncclConnInfo` 注释里的 "Local for recv / remote for send" 描述的是**指针落在谁的内存里**,
与"谁推进它"是两回事 —— 不要把两者混淆。

#### 两条共享内存管道 `ncclSendMem` / `ncclRecvMem`

`src/include/comm.h:53`

```c
struct ncclSendMem {
  uint64_t head;                 // 生产方推进
  uint64_t offsFifo[NCCL_STEPS]; // 每 step 的偏移(LL 用)
  ...
};
struct ncclRecvMem {             // comm.h:67
  uint64_t tail;                 // 消费方推进
  struct ncclConnFifo connFifo[NCCL_STEPS]; // device↔proxy 控制槽
  int flush;                     // GDRCopy flush 用
  ...
};
```

#### device↔proxy 控制槽 `ncclConnFifo`

`src/include/collectives.h:72`

```c
struct ncclConnFifo {
  int     mode;   // NCCL_MODE_NORMAL / OFFSET / PTR
  int     offset; // 共享缓冲区时的偏移
  ssize_t size;   // GPU 写入本 step 字节数;proxy 完成后写 -1
  void*   ptr;    // NCCL_MODE_PTR 时的地址
};
```

`connFifo` 是 device 与 proxy 之间的"小纸条":GPU 在某 step 写入数据后,把这一步的
`{size, offset, ...}` 塞进 `connFifo[step % NCCL_STEPS]`;proxy 读到它后才知道"这一步
数据多大、放在共享缓冲区的哪个偏移",发完 RDMA 后把 `size` 写成 `-1` 通知 GPU"这块
可以复用了"。

### 1.2 角色:同一个 kernel block 内的线程分工

NCCL 不为每个连接起一个 kernel。一个 channel 的 kernel block 内,线程被切成若干
"同步角色",每个角色绑定一个 peer 连接。角色在 `Primitives` 构造函数里按 `tid` 分配
(`src/device/prims_simple.h:611`):

```c
if (tid < nrecv)              { flags |= RoleWaitRecv;  index = tid; }
else if (tid < nrecv+nsend)   { flags |= RoleWaitSend;  index = tid-nrecv; }
else if (nthreads-nsend<=tid) { flags |= RolePostSend;  index = tid-(nthreads-nrecv-nsend); }
else if (nthreads-nrecv<=tid) { flags |= RolePostRecv;  index = ...; }
```

四个角色的语义(结合 1.1 的计数器定义):

| 角色 | 做什么 | 读/写的计数器 |
|------|--------|---------------|
| **WaitRecv** | 消费"收到的数据":等数据就绪 | 等 `tail` |
| **PostRecv** | 消费完毕:释放缓冲槽 | 推进 `head` |
| **WaitSend** | 生产"待发的数据":等有空槽 | 等 `head` |
| **PostSend** | 生产完毕:标记数据就绪 | 推进 `tail` |

读法记忆:**Wait\* 都在"等"(读计数器),Post\* 都在"发布"(写计数器)**;
Recv 侧与 Send 侧的 head/tail 互换 —— 这正是 1.3 节 connect() 的布线结果。

### 1.3 证据链:connect() 如何把指针接上

#### 证据一:p2p 传输(节点内,无 proxy)

`src/transport/p2p.cc:556-600`,以 sender A → receiver B 为例:

```c
// A 的 send connector
send->conn.tail = &remDevMem->tail;      // 指向 B 的 recvMem.tail  (568)
send->conn.head = &sendDevMem->head;     // 指向 A 自己的 sendMem.head (569)
// B 的 recv connector
recv->conn.tail = &devMem->tail;         // 指向 B 自己的 recvMem.tail (599)
recv->conn.head = &remDevMem->head;      // 指向 A 的 sendMem.head    (600)
```

节点内 p2p,A 可以直接写 B 的共享内存(含 B 的 `recvMem.tail`)。把"谁推进谁"代入:

- A 写完数据后**推进 B 的 `recvMem.tail`**(=A 的 send connector 的 `tail` 字段指向处)。
  → `tail` 由生产方 A 推进 ✓。
- B 消费完后**推进 A 的 `sendMem.head`**(=B 的 recv connector 的 `head` 字段指向处)。
  → `head` 由消费方 B 推进 ✓。

#### 证据二:device 端角色 ↔ 计数器

`src/device/prims_simple.h` 的 `loadRecvConn`(473)/`loadSendConn`(527):

```c
// loadRecvConn: RECV connector
if (flags & RolePostRecv) { connStepPtr = conn->head; *connStepPtr = step; }  // 486: 写 head
if (flags & RoleWaitRecv) { connStepPtr = conn->tail; ... }                    // 493: 读 tail
// loadSendConn: SEND connector
if (flags & RolePostSend) { connStepPtr = conn->tail; }                        // 537: 写 tail
if (flags & RoleWaitSend) { connStepPtr = conn->head; ... }                    // 544: 读 head
```

把证据一的"RECV connector: head=远端, tail=本地"代入:

- `WaitRecv` 读 `tail`(本地 `recvMem.tail`) → 即"等生产方把数据推进到这里"。✓
- `PostRecv` 写 `head`(远端 `sendMem.head`) → 即"通知对端生产方:我消费到这一步了"。✓

两条证据链完全自洽。`tail`=数据就绪、`head`=信用,这个结论**与传输无关、与是否有 proxy 无关**,
是 NCCL 数据通路的统一不变量。

> PAT 模式(`prims_simple.h:670`)用更直白的写法做了同一件事:
> recv peer `stepCache = loadStepValue(tailPtr = conn->tail)`(686),
> send peer `stepCache = loadStepValue(headPtr = conn->head)`(697)——印证了
> "recv 等 tail、send 等 head"。

### 1.4 状态机:一步数据的生命周期

下面用一个时间轴展示**一个 channel 上,两个相邻 step 在深度 8 的环形缓冲里**的握手
(p2p 场景;proxy 场景把"对端 GPU"换成"本端 proxy"即可,见机制二):

```
时刻   生产方(A, RolePostSend/WaitSend)              消费方(B, RoleWaitRecv/RolePostRecv)
----   ------------------------------------------      ------------------------------------------
 t0    WaitSend: while(head + NCCL_STEPS < step)      (B 上一轮还在处理 step-8)
        自旋读 A 的 sendMem.head(被 B 推进)
 t1    head 终于 >= step-NCCL_STEPS:有空槽了            B 完成处理, PostRecv: 写 A.sendMem.head = step
 t2    把数据 DMA 写进 B 的 recv 缓冲(槽 step%8)
 t3    threadfence(发布顺序)
 t4    PostSend: 写 B.recvMem.tail = step              ─(内存可见)→
 t5                                                     WaitRecv: 自旋读 tail, 发现 tail >= step
 t6                                                     从 recv 缓冲(槽 step%8)读数据并 reduce/copy
 t7    (A 已继续处理 step+1 ...)                         PostRecv: 写 A.sendMem.head = step+1 (下一轮信用)
```

对应的 device 代码在 `prims_simple.h` 的 `process<>()`(322),核心三段:

**(a) 等待(Wait)** —— 331:

```c
while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
  connStepCache = loadStepValue(connStepPtr);   // send 等 head(+NCCL_STEPS 偏移即"留 8 槽余量")
  if (checkAbort(flags, Aborted, spins)) break; // 死锁/abort 保护,spins 计数后退避
}
```

注意 send 侧要 `+NCCL_STEPS`:即"head 至少要落后 posted 8 步以上我才停",这正是深度 8
环形缓冲的背压条件 —— **缓冲区不能被生产方超前消费方 8 步以上**。

**(b) 搬运**:375 调用 `fn(...)`(由各 collective 模板提供的 reduceCopy),从 `srcs[]`
拷到 `dsts[]`。`srcs/dsts` 由 `connFifo[step%8]` 的 mode 决定(336-361):

- `NCCL_MODE_OFFSET`:`ptrs = connEltsFifo + offset/sizeof(T)`(proxy 共享缓冲);
- `regUsed`(直接读写对端用户缓冲):`ptrs = directBuff`;
- 普通:`ptrs = connEltsFifo + (step%8)*connStepSize`(本端环形 step 缓冲)。

**(c) 发布(Post)** —— 381-397:

```c
if (flags & Send*RolePostSend) {
  dstSize = dstSizes[index];
  connFifo[step % NCCL_STEPS].size = dstSize * sizeof(T);  // 386: 写"小纸条"
}
barrier();                                                   // 388: block 内发布
step += StepPerSlice;                                        // 390
if (flags & (...RolePostRecv | ...RolePostSend)) {
  if (Send && ...) fence_acq_rel_sys();                     // 394: 跨设备 acquire/release 屏障
  st_relaxed_sys_global(connStepPtr, step);                  // 396: 推进 tail/head
}
```

`fence_acq_rel_sys()` + `st_relaxed_sys_global` 是关键:`fence` 保证"数据写入"在
"计数器推进"之前对**整个系统**(含其他 GPU、NIC)可见;`_sys_` 后缀即 system-scope,
跨 PCIe/NVLink/网络都成立。这就是无锁同步能在异构内存间工作的根基。

### 1.5 host → device:Work FIFO 与常驻内核

device↔proxy 之外,还有 host→device 这条轨道。host 把一个 collective 拆成若干
`ncclDevWorkColl`(`comm.h`),打包成 `ncclDevWorkBatch`,通过 `uploadWork()`
(`enqueue.cc:1248`)写进 device 可见的 work 区域。

device 侧常驻内核的调度核心在 `src/device/common.h:356` 的 `ncclKernelMain`:

1. **block ↔ channel 映射**(370):用 `__popcll(channelMask & ((1<<tid)-1))` 反查
   `channelMask` 的第 n 个置位 bit → 得到本 block 负责的 `channelId`。比 PTX `fns` 快。
2. **协作加载**(382):前两个 warp 分别把 `ncclKernelComm` 和本 channel 的
   `ncclDevChannel` 载入 shmem;其余 warp 用 `loadWorkBatchToShmem` 载入 work batch。
3. **workCounter 门控**(378):每个 channel 有单调 `workCounter`,host 写新 work 时
   `+1`,device 据此判断有无新任务。
4. **批链接循环**(417-431):

```c
while (ncclShmem.aborted == 0) {
  profiler(START);
  if (funcId == SpecializedFnId) SpecializedRunWorkBatch().run();
  else                           ncclDevFuncTable[funcId]();   // 422: 函数表派发
  if (ncclShmem.nextBatchIx == -1) break;                      // 425: 没有后续批,退出
  loadWorkBatchToShmem(..., ncclShmem.nextBatchIx);            // 429: 链式载入下一批
}
```

`nextBatchIx` 是**常驻内核(CUDA Graph / persistent plan)的关键**:一次 kernel launch
可以串联处理多个 work batch,避免每个 collective 都付 launch 开销。`run()` 内(303)
再对 `nWorks` 个 work 逐个执行 `RunWorkColl`(按 `work->nWarps` 动态切线程)。

### 1.6 小结

```
            ┌──────── host ────────┐
            │  uploadWork(): 写 ncclDevWorkBatch, workCounter++   (enqueue.cc:1248)
            └──────────┬───────────┘
                  Work FIFO(单向)
            ┌──────────▼───────────┐
            │  device: ncclKernelMain → func dispatch → RunWorkColl (common.h:356)
            │         每个 work 内用 Primitives 跑 ring/tree 步进
            └──────────┬───────────┘
              connFifo + tail(数据) + head(信用)   深度 NCCL_STEPS=8
            ┌──────────▼───────────┐
            │  proxy: ncclProxyProgress → progressOps → op->progress() (proxy.cc:954)
            │         网络:send/recvProxyProgress 三/四段流水 (net.cc)
            └──────────────────────┘
```

三条轨道各有自己的同步原语:Work FIFO 用 `workCounter` 单调门控;数据通路用
`tail`(就绪)/`head`(信用)双计数器 + system-scope fence;connFifo 用 `size` 值
(>0 = 待发,-1 = 完成)做细粒度控制。**全程序几乎没有锁**,靠内存序 + 单调计数器 + 自旋。

---

## 机制二:网络 RDMA 流水线

跨节点时,device↔proxy 的 `tail/head` 仍然成立,但中间多了一个 **CPU proxy 线程**
负责把数据搬过网卡。proxy 用 IB Verbs 的 RDMA write/send 把 device 写好的数据发出去,
用 RDMA read / `iflush` 保证 GPU 可见性。这一节拆解 proxy 的多段流水。

### 2.1 总体结构

每个 network connector 在 proxy 侧有一份 `sendNetResources` / `recvNetResources`
(`src/transport/net.cc`),内含:

- `netSendComm` / `netRecvComm`:底层 verbs 通信句柄;
- `map`:CUDA IPC / GDR 映射的缓冲;
- `sendMem` / `recvMem`:device 可见的 `ncclSendMem`/`ncclRecvMem`(即机制一里
  `conn->head/tail` 指向的地方);
- `useGdr` / `gdcFlush` / `gdcSync`:GPUDirect RDMA 与 GDRCopy flush 的开关。

proxy 主循环 `ncclProxyProgress`(`src/proxy.cc:954`)不断调用 `progressOps`(801),
后者对每个 `ncclProxyOp` 调它的 `progress` 回调 —— 网络发送就是 `sendProxyProgress`,
网络接收就是 `recvProxyProgress`。

### 2.2 发送侧三段流水 `sendProxyProgress`

`src/transport/net.cc:1304`。三个本地游标 `posted / transmitted / done` 推进同一批
`nsteps` 个 step。状态机:

```
ncclProxyOpReady ── 初始化 base/posted/trans/done=0 (1306-1318) ──▶ ncclProxyOpProgress
                                                                   │
                 ┌──────────────────────────────────────────────────┘
                 ▼ 每轮循环依次尝试三段(只要其一有进展就把 idle=0)
   ┌─────────────────────────┐  ┌──────────────────────────┐  ┌────────────────────────┐
   │ ① POSTED (1335-1356)    │  │ ② TRANSMITTED (1358-1430)│  │ ③ DONE (1432-1458)     │
   │ 预登记 NIC 接收缓冲,    │  │ 检查 GPU 是否已把数据填入,│  │ 检查 NIC 发送是否完成, │
   │ 给 GPU 发"信用":        │  │ 发 isend():              │  │ 回收缓冲:             │
   │ *sendHead = base+posted │  │  (rdma write / send)     │  │ connFifo[slot].size=-1 │
   │            - NCCL_STEPS │  │ transmitted += sliceSteps│  │ *sendHead = base+done  │
   └─────────────────────────┘  └──────────────────────────┘  └────────────────────────┘
```

逐段解读:

**① POSTED**(1335):proxy 还没看到 GPU 数据,先向 NIC 登记好"我准备好的接收槽位"
(对端会把数据 RDMA-write 进来,或本端用 send/recv)。登记后立即把 `sendMem->head`
推进到 `base + posted - NCCL_STEPS`(1348)。这正是机制一里 **GPU 的 `WaitSend` 自旋
等待的 `head`** —— proxy 通过推进 head 告诉 GPU"你可以往回填 NCCL_STEPS 步了"。
共享缓冲模式下还把获得的共享偏移写进 `connFifo[slot].offset`(1343),让 GPU 知道把
数据写到共享池的哪个位置。

**② TRANSMITTED**(1358):proxy 检查两个条件都满足才能发(1362):

```c
if (connFifo[buffSlot].size != -1 && (*recvTail > tail || p==NCCL_PROTO_LL)) {
```

- `connFifo[size] != -1`:GPU 的 PostSend 已经写过"小纸条"(有效 size);
- `*recvTail > tail`:GPU 已经把 `recvMem.tail` 推过这一步(即机制一里 send 侧
  `PostSend` 写的 `tail`);"LL 协议例外"是因为 LL 用 flag 行而不是 tail 计数。

然后按协议校验"数据是否真的全部就位":
- **LL128**(1368):逐行读缓冲里每个 128B 行的 flag uint64,要求等于 `base+transmitted+1`;
- **LL**(1384):读每个 `ncclLLFifoLine` 的 `flag1/flag2`,要求等于 `NCCL_LL_FLAG(base+transmitted+1)`;
- **SIMPLE**:直接信任 `tail` + connFifo(若 `reg` 则用 `ringAlgo->getNextSendAddr` 取
  用户缓冲地址做 zero-copy 发送,1402)。

校验通过后调用 verbs 层 `isend()`(1414)发起 RDMA 写/发,`transmitted += sliceSteps`。

**③ DONE**(1432):`test(request)` 查 NIC 完成队列。完成后:
- `connFifo[slot].size = -1`(1439)→ 通知 GPU 这个槽可复用(机制一 destructor 里
  `NetRegMode` 正是在等这个 `-1`,`prims_simple.h:720`);
- `*sendHead = base + done`(1448)→ 进一步推进信用(对非共享缓冲)。

### 2.3 接收侧四段流水 `recvProxyProgress`

`src/transport/net.cc:1470`。比发送多一段 **flush**,因为 RDMA 把数据写进 GPU 内存后,
需要"冲刷"让 GPU 看到一致性视图。四段:`posted → received → transmitted → done`。

```
① POSTED(1531):  按 GPU 信用(done+maxDepth)限流,irecv() 登记 NIC 接收请求
                  (数据将被 DMA 进 recv 缓冲)
② RECEIVED(1613): test(irecv) 完成 → 数据已到 GPU 内存
                  → connFifo[slot].size=-1; 若 useGdr 且 needFlush,进入 flush 路径
③ FLUSH(1641-1687): 保证 NIC 的 DMA 写对 GPU 全局可见(见 2.4)
④ TRANSMITTED(1696): test(flush) 完成 → 推进 recvMem->tail = base+transmitted
                  ★ 这就是 GPU 的 WaitRecv 自旋等待的"数据就绪"信号
⑤ DONE(1727):     读 sendMem->head(GPU 的 PostRecv 推进)→ 知道 GPU 已消费,回收接收槽
```

最关键的一行在 1711-1712:

```c
volatile uint64_t* recvTail = resources->gdcSync ? resources->gdcSync : &resources->recvMem->tail;
*recvTail = sub->base + sub->transmitted;
```

—— **`recvMem.tail`(数据就绪计数器)只有在 flush 通过后才推进**。这意味着 GPU 看到的
数据一定是一致的。注意 `transmitted` 这里名不副实(它其实是"flush 完成"游标),命名沿用
了发送侧模板。

### 2.4 GDRCopy flush:为什么需要,如何实现

GPUDirect RDMA(GDR)让 NIC 直接 DMA GPU 显存,绕过 CPU 拷贝,极大降延迟。但有一个
微妙问题:**NIC 的 DMA 写是 PCIe posted write,GPU 端可能"还没全部落地"**。如果 proxy
此时就把 `recvMem.tail` 推进,GPU 读到的可能是旧数据。

NCCL 的解法是在推进 `tail` 之前做一次 **flush**(1641 起),两条路径:

**路径 A:`gdcFlush` 可用(GDRCopy Async / DOCA GDAKI,x86)** —— 1645-1652:

```c
asm volatile("mfence" ::: "memory");                            // 1649: 先 mfence
asm volatile("mov (%0), %%eax" ::"l"(resources->gdcFlush)...);  // 1652: 从 GPU 内存读一字节
```

注释解释得很清楚(1646-1651):`mfence` 保证前面的 CQE 轮询读排在前面(防止 WC 读被
推测提前到 PCIe);随后那个 `mov (%gdcFlush)` 是一次 **PCIe 读**,会**阻塞 CPU 直到
该 endpoint 上所有先前 posted 写(含 NIC DMA)都提交**。这是一招经典的"用读来等写"。

**路径 B:回退 `iflush`** —— 1684:发一个 RDMA read 请求读 GPU 内存,等其完成同样
能强制 ordering;之后 `test()` 通过才进入第④段。

**发送侧对称地有 GDR 写的 fence**:`sendProxyProgress` 推进 `sendHead` 时若用 `gdcSync`,
跟一句 `wc_store_fence()`(1349/1449)冲刷 WC(write-combining)写。

### 2.5 参数与环境

`net.cc:339-341` 附近定义了若干 flush 相关 param(`NCCL_NET_GDRCOPY_ENABLE`、
`NCCL_IB_GPU_DIRECT_RDMA`、`NCCL_GDRSYNC_ENABLE` 等)。运行时若探测不到 GDR
支持,`useGdr=0`,proxy 会回退到"NIC ↔ host 内存 ↔ GPU"的 bounce-buffer 路径
(慢但兼容)。这也是为什么开 GDR 前要确认驱动、`nvidia-peermem`、IB 卡都支持。

`static_assert(NCCL_STEPS <= NCCL_NET_MAX_REQUESTS)`(net.cc:1302)保证 verbs 层
的请求槽位够覆盖环形缓冲深度 —— 否则流水会卡死。

### 2.6 小结:发送/接收的计数器对照

| 计数器 | 谁推进 | 谁等 | 语义 |
|--------|--------|------|------|
| `sendMem.head` | proxy(posted/done 两处) | GPU WaitSend | 发送信用(可填几步) |
| `recvMem.tail`(send 侧) | GPU PostSend | proxy transmitted 段 | 数据已就绪(发几步) |
| `recvMem.tail`(recv 侧) | proxy transmitted 段 | GPU WaitRecv | 数据已到达(可消费) |
| `sendMem.head`(recv 侧) | GPU PostRecv | proxy done 段 | 消费信用(可回收) |
| `connFifo[slot].size` | GPU 写 size / proxy 写 -1 | 互相 | 本 step 控制 + 完成 |

机制一的 `tail`=就绪 / `head`=信用 在网络路径下被 proxy 完整复刻,proxy 本质上是
"一个用 RDMA 搬数据的、替对端 GPU 做计数器推进的中继"。

---

## 机制三:调度与切分

用户调用一次 `ncclAllReduce(...)` 等,host 要回答几个问题:用几个 channel?每个 channel
分多少数据?切成多大的 chunk?多个 collective 能否合并进同一个 kernel plan?这些都由
`enqueue.cc` 的调度层决定。

### 3.1 任务抽象

每个 collective 请求被翻译成一个 `ncclTaskColl`(`comm.h:193`),关键字段:
`func`(allreduce/...),`count`,`datatype`,`sendbuff/recvbuff`,`algorithm`,`protocol`,
`chunkSteps/sliceSteps`,`nMaxChannels`,`trafficBytes`,`devFuncId`。一次 `ncclGroupEnd`
会把组内所有 task 组织进 `comm->planner`。

`ncclFuncTrafficPerByte`(`enqueue.cc:91`)给出"每字节流量系数":allreduce=2(n-1)/n
(bus count),allgather/reducescatter=(n-1)/n,broadcast/reduce=1。用于把用户数据量换算
成链路流量,是后面 channel 分配和带宽建模的基础。

### 3.2 chunk 切分:`calcCollChunking`

`enqueue.cc:2182`。核心是把"用户数据"切成能在环形缓冲里流动的 **chunk**。

**(a) 基础尺寸**(2222-2229):

```c
int stepSize  = comm->buffSizes[info->protocol] / NCCL_STEPS;  // 单 step = 缓冲/8
int chunkSteps = (proto==SIMPLE && algo==RING) ? info->chunkSteps : 1;
int sliceSteps = (proto==SIMPLE && algo==RING) ? info->sliceSteps : 1;
int chunkSize  = stepSize * chunkSteps;                          // 一个 chunk 字节数
if (proto==LL)   chunkSize /= 2;                                 // LL: data+flag 各占一半
if (proto==LL128) chunkSize = (chunkSize/16)*15;                 // LL128: 15/16 是数据
int bufferMaxChunkSize = chunkSize;                              // 缓冲允许的上限
```

只有 **RING + SIMPLE** 会用 `chunkSteps > 1`(把多个 step 合成一个更大的 chunk,减少
循环开销);其余协议/算法都是 `chunkSteps=1`。`chunkSteps/sliceSteps` 在拓扑调优时
写入 task(`ncclPrepareTasks` 里 `BROADCAST_CHUNKSTEPS/SLICESTEPS` 是默认,见 380-382)。

**(b) 模式选择**(2188-2218):按 `(func, algorithm)` 选 `ncclPattern*`。例如
allreduce+RING→`ncclPatternRingTwice`(跑两圈:reduce-scatter + allgather),
allreduce+TREE→`ncclPatternTreeUpDown`,reduceScatter+PAT→`ncclPatternPatUp`。
pattern 决定 device 内核的步进逻辑。

**(c) 自适应缩小**(2231-2296):**若数据太小、chunk 太大,流水深度不够,会主动把
chunkSize 减半**,直到"在途 step 数"够覆盖算法深度。以 COLLNET_DIRECT 为例(2231):

```c
while (nBytes/(nChannels*nHeads*chunkSize) < depth*64 && chunkSize>131072) chunkSize /= 2;
while (...                                              < depth*8  && chunkSize>65536)  chunkSize /= 2;
while (...                                              < depth*8  && chunkSize>32768)  chunkSize /= 2;
```

思想:并发量 `nChannels*nHeads*chunkSize` 要能撑起 `depth*64` 等级的流水气泡,否则缩小
chunk、增加在途步数来掩盖延迟。NVLS / NVLS_TREE / TREE+LL128 / PAT 各有类似的
"按 nBytes/(nChannels*chunkSize) 比例收半"逻辑(2276-2296)。

### 3.3 channel 条带化:`scheduleCollTasksToPlan`

`enqueue.cc:576`。这一步决定"这个 plan 里放哪些 collective、每个 collective 用哪些 channel"。

**(a) 容量预估**(586-604):先扫一遍待调度的 task,按预算 `ncclTestBudget`(294)
预估能塞下多少个 collective(`nPlanColls`)。预算分两块:

```c
struct ncclKernelPlanBudget { ssize_t inArgsBytes, outArgsBytes; };   // 289
bool ncclTestBudget(budget, nWorkBatches, workBytes) {               // 294
  // work batch 头 + work 体要么全塞进 kernel args(inArgsBytes),
  // 要么 batch 头进 args、work 体进 fifo/persistent buf(outArgsBytes)
}
```

这就是 NCCL **fused kernel** 的由来:多个 collective 共用一个 kernel launch,只要总
work 字节数不超预算。`NCCL_MAX_DEV_WORK_BATCH_COLLS` 限制单个 batch 内的 collective 数。

**(b) 按流量给 channel 分配数据**(616-680)。对每个 collective,先算

```c
trafficPerChannel = divUp(trafficBytes[kind]/nChannels[kind], 16)*16;   // 618
cellSize          = divUp(divUp(MinTrafficPerChannel, trafficPerByte),16)*16; // 659
cells             = divUp(count*elementSize, cellSize);                  // 661
cellsPerChannel   = min(cells, divUp(trafficPerChannel, trafficPerCell));     // 664
```

然后把 `cells` 切成三段铺到连续 channel 上(665-684):

```
channelId →  [ ... | nMidChannels 个"满" channel (cellsPerChannel) | cellsHi | cellsLo ]
                   ▲ cellsHi: 不足一整 channel 的余数
                                                                      ▲ cellsLo: 当前 channel 累计未满的尾部
```

`devWork->channelLo/channelHi`(636/712)记录这个 collective 占用的 channel 区间,
device 内核据此把 work 派发到对应 channel。`MinTrafficPerChannel=32K`(585)是"低于
这个量不值得单独占一个 channel"的下限,避免 channel 过度碎片化。

**(c) 产出 work 节点**:`ncclTasksRegAndEnqueue`(302)把每个 task 物化成一个
`ncclDevWorkColl`(或带注册信息的 `ncclDevWorkCollReg`,343),挂到 `collWorkQueue`。
`devWork` 里携带 `sendbuff/recvbuff`、`channelLo/Hi`、`nWarps`、`regUsed/netRegUsed`、
`direct` 标志等 device 执行所需的一切(320-352)。

### 3.4 plan、常驻 plan、CUDA Graph

一个 `ncclKernelPlan`(`enqueue.cc` 的 `ncclKernelPlanner`)聚合:work batches、proxy ops、
kernel args。`ncclAddWorkBatchToPlan`(121)把 work 按 channel 归并成 batch;
`uploadWork`(1248)再把 plan 上传 device;`ncclLaunchKernel`(1753)用
`cuLaunchKernelEx` 发射(带 CGA / MemSyncDomain 属性)。

**常驻计划 / CUDA Graph**:当 communicator 处于 graph 捕获(`comm->planner.capturingGraph`)
或 persistent plan 模式时,机制一提到的 `nextBatchIx` 链式批处理被激活 —— 一次 launch
处理整条 batch 链,replay 时不重新调度。这正是 NCCL 在 CUDA Graph 训练里几乎零开销的
原因:调度与切分只发生一次(capture 时),replay 时只重放 device 内核 + proxy ops。

### 3.5 小结

```
ncclGroupEnd
   │
   ▼  ncclPrepareTasks (enqueue.cc:363)
   │   - 为每个 task 选 algo/proto(见机制四),写 chunkSteps/sliceSteps
   │   - 算 trafficBytes = count * ncclFuncTrafficPerByte
   ▼  scheduleCollTasksToPlan (576)
   │   - 预算内 fused 多个 collective → nPlanColls
   │   - 每个 collective:calcCollChunking 算 chunkSize
   │   - 按 trafficPerChannel 把 cells 铺到 channelLo..channelHi
   ▼  ncclTasksRegAndEnqueue (302) → ncclDevWorkColl
   ▼  uploadWork (1248) + ncclAddProxyOpIfNeeded → device work + proxy ops
   ▼  ncclLaunchKernel (1753)  ──▶  机制一的 device/proxy 数据通路
```

调度层把"语义层 collective"翻译成"通道并行度 × 流水深度 × 缓冲预算"的三维规划,
目标是:在 work 预算内尽量 fuse、按链路带宽均衡地跨 channel 条带、给每个算法配合适
的 chunk 粒度以填满流水。

---

## 机制四:拓扑 / 算法 / 协议选择

NCCL 在 communicator 初始化时做一次离线**拓扑调优**,为每个 `(collective, algorithm, protocol)`
三元组预算出 `(bandwidth, latency)`;运行时对每次调用按消息大小选时间最短的三元组。
两个阶段分别在 `src/graph/tuning.cc` 和 `src/enqueue.cc`。

### 4.1 第一阶段:离线带宽/延迟建模 `ncclTopoTuneGraph`

`tuning.cc:245`。前提是图搜索(`src/graph/search.cc` 等)已经为每个算法 `a` 算出了
`graphs[a]`,含 `bwIntra / bwInter`(每条 channel 的链路带宽)、`nChannels`、
`latencyInter`、`typeIntra`(NVL/PCI)。

**线程数表**(245-257):为每个 `(algo, proto)` 定 `maxThreads`(如 RING+SIMPLE 在
PCIe 带宽受限时只用 256 线程,否则 512;LL128 固定 640)。线程数与寄存器上限、
`-maxrregcount` 联动(见 CLAUDE.md 的构建说明)。

**合法性过滤**(294-308):三元组必须满足约束才进入建模,例如:
- broadcast/reduce 只能用 RING;
- allgather/reducescatter 只能用 PAT/RING/NVLS/COLLNET_DIRECT;
- allreduce 不能用 PAT;NVLS/NVLS_TREE 只能用 SIMPLE;
- PAT 还需 `ncclPatEnable(comm)`。

**带宽建模**(300-382):对每个合法三元组,

```c
float bw = (nNodes<=2 || collnet) ? graphs[a]->bwIntra : graphs[a]->bwInter;   // 306
... // 各算法的效率修正(见下)
float busBw = graphs[a]->nChannels * bw;                                        // 324
```

关键的**效率修正因子**(326-339,直接体现各协议/算法的物理特性):

| 修正 | 含义 |
|------|------|
| RING+LL: `busBw*0.5` | LL 每 16B 里一半是 flag,有效数据减半 |
| RING+LL128: `busBw*120/128` | LL128 每 128B 行里 120B 数据(15/16) |
| TREE+AllReduce: `busBw*0.92` | 树形 allreduce 的工程效率 |
| TREE+LL: `/3.8` | LL 在树上的额外开销 |
| PAT: `*0.75` | PAT(halving-doubling)的效率折损 |
| NVLS: `*nvlsEfficiency*(nChannels-1)/nChannels` | NVSwitch 多播的开销 |

NVLS 还要按 `ppn` 把 intra 带宽折算:allreduce 串联两个操作故 `*2`(315),其余
`*(ppn-1)/ppn`(317)。

**busBW → algoBW**(376-382):链路带宽要换成"用户视角的算法带宽"。

```c
if (a==RING||NVLS||NVLS_TREE) ratio *= (1.0*nRanks)/nsteps;   // ring: 2(n-1)/n 系数
else                          ratio *= .5;                     // tree: ~0.5
comm->bandwidths[coll][a][p] = busBw * ratio;                  // 383
```

`nsteps` = allreduce 是 `2(n-1)`,allgather/reducescatter 是 `n-1`,broadcast 是 `n`(289)。
这一步把"链路上搬的字节"折算回"用户数据的字节",正是 `ncclFuncTrafficPerByte` 的逆运算。

**延迟建模**(384-434):`baseLatencies + hwLatencies × 跳数`。不同算法的跳数模型:

- **RING**(393):`(nsteps-nInterSteps)*intraLat + nInterSteps*interLat`,单节点全是 intra,
  多节点把跨节点那几跳用更贵的 `interLat`(`interLat` 还含 NIC flush 额外一次,391)。
  大消息 inter-node allreduce 还有"plateau 效应"修正(649 处 `lat *= 1.4`)。
- **TREE AllReduce**(416):`2*((nRanks/nNodes-1)*intraLat + log2(nNodes)*interLat)`,
  体现树的 `log2(nNodes)` 深度。
- **COLLNET_DIRECT**(419):`+0.4us` 的 arity 序列化延迟。
- **NVLS_TREE**(428):`intraLat + 2*log2(nNodes)*interLat`。
- **PAT**(430):`log2(nNodes)*(interLat/3.5) + nRanks*2.8`(对数项 + 残留线性项)。

结果存进 `comm->bandwidths[coll][a][p]` / `comm->latencies[coll][a][p]`,作为运行时选型的
查表依据。可由 `NCCL_TUNING_FILE` 覆盖,或用 tuner plugin(`comm->tuner->getCollInfo`)替换。

### 4.2 第二阶段:运行时选型

每次 collective,`ncclPrepareTasks` 调 `ncclGetAlgoInfo`(`enqueue.cc:2123`)。

**(a) 造代价表**(1989 `updateCollCostTable`):对每个合法三元组调
`ncclTopoGetAlgoTime`(`tuning.cc:630`)算时间:

```c
float bw  = comm->bandwidths[coll][algorithm][protocol];
float lat = comm->latencies [coll][algorithm][protocol];
if (bw == 0) { *time = -1.0; return; }                 // 635: 不合法,跳过
int logSize = log2i(nBytes >> 6);
if (algo==TREE && logSize<23) bw *= treeCorrectionFactor[proto][logSize];  // 640: 经验小消息修正
if (algo==RING && nNodes>1 && nBytes/(nChannels*nRanks)>=64) lat *= 1.4;   // 647: ring plateau
int latCount = algo==RING ? numPipeOps : DIVUP(numPipeOps, NCCL_MAX_DEV_WORK_BATCH_COLLS); // 652
*time = lat * latCount + nBytes / (1000.0 * bw);       // 653 ★ 核心公式
```

代价公式就是经典的 **`T = 延迟 + 数据量/带宽`**:`lat`(us)+ `nBytes`/(1000·bw)
(bw 单位 MB/s → 化成 us/byte)。`latCount` 让 fused(pipeline)的多个操作分摊延迟:
RING 是逐个承担,TREE 等可按 batch 分摊。

**(b) 取最小**(2028 `topoGetAlgoInfo`):遍历代价表取 `minTime` 对应的
`(algorithm, protocol)`(2038),写回 `info->algorithm/protocol`。`NCCL_ALGO_PROTO_IGNORE`
表示被环境变量或约束强制排除。

**(c) tuner / 特殊覆盖**(2140-2174):若加载了 tuner plugin,先让 plugin 改写代价表;
`NCCL_CTA_POLICY_EFFICIENCY` 下,若用户缓冲已注册且是 allgather/reducescatter,会强制
NVLS(2163-2174)以走 zero-copy direct 路径。

### 4.3 算法谱

| 算法 | 结构 | 典型场景 |
|------|------|----------|
| **RING** | 数据沿逻辑环流动,每 rank 收上一段、发下一段 | 通用,中大消息 |
| **TREE** | 二叉树(可 2-arity),上下两趟 | 小—中消息 allreduce,低延迟 |
| **COLLNET_DIRECT / CHAIN** | 经 SHARP/交换机网内归约 | 多节点 + SHARP 交换机 |
| **NVLS / NVLS_TREE** | NVSwitch 多播(NVLink SHARP) | 单节点 NVSwitch 机器,allreduce/allgather |
| **PAT** | halving-doubling(Pattern),`±2^k` 通信 | allgather/reducescatter,大规模 |

### 4.4 协议谱

三种 device 侧原语协议(决定"数据如何在 step 缓冲里摆放、如何判定就绪"):

| 协议 | 线程上限 | 数据布局 | 就绪判定 | 适用 |
|------|----------|----------|----------|------|
| **SIMPLE** | 512(`device.h:95`) | 纯数据,每 step = `stepSize` 字节 | `tail` 计数器 | 大消息 |
| **LL** | 512(`97`) | 16B 行:`{data1,data2,flag1,flag2}` | 每 `ncclLLFifoLine` 的 flag | 小消息(低延迟) |
| **LL128** | 640(`114`) | 128B 行:`15×uint64 数据 + 1×uint64 flag` | 每 `NCCL_LL128_LINEELEMS` 行的 flag | 节点内大消息(NVLink 满带宽) |

关键常量(`device.h`):
- `NCCL_STEPS = 8`(环形深度);
- `NCCL_LL128_LINESIZE=128`,`LINEELEMS=16`,`DATAELEMS=15`(110-112);
- `NCCL_LL_FLAG(a)`(102/105):LL 的 flag 用步号生成,`NCCL_LL_CLEAN_MASK=0x7ffffff8`
  保证清理掩码能覆盖至少 `NCCL_STEPS`(108 的 static_assert);
- `ncclProtoGrainSize`(`device.h:328`):不同协议的"颗粒度",用于 chunk 计数。

LL/LL128 的 flag 机制解释了为什么 `sendProxyProgress` 对 LL/LL128 要逐行校验 flag
(机制二 2.2 的 ② 段)—— 它们不依赖 `tail` 计数器判就绪,而依赖每行内嵌的 flag,
代价是有效带宽打折(LL ×0.5,LL128 ×120/128),换来更低的每步延迟。

### 4.5 小结

```
                       ┌──── 离线(comm init)─────┐
   graph search ──▶ graphs[a] ──▶ ncclTopoTuneGraph (tuning.cc:245)
                                   合法性过滤 + 效率修正 + busBw→algoBW + 跳数延迟
                                   └─▶ comm->bandwidths/latencies [coll][a][p]
                       └──────────────────────────┘
                       ┌──── 运行时(每次 collective)────┐
   ncclGetAlgoInfo (enqueue.cc:2123)
     updateCollCostTable → ncclTopoGetAlgoTime: time = lat*latCount + nBytes/(1000*bw)
     topoGetAlgoInfo → argmin (a,p)
     └─▶ task.algorithm / task.protocol / nMaxChannels
                       └───────────────────────────────┘
```

整个选型是个**两阶段查表 + 线性时间模型**:离线把硬件拓扑烧成带宽/延迟表,运行时
对每个消息大小用 `T=lat+size/bw` 挑最优。简单、可解释,也正因为如此,任何一个修正因子
(如 ring plateau 的 1.4、tree 的 correctionFactor)都对应一个真实可复现的硬件现象 ——
调参时改动它们要有实测依据。

---

## 附:关键文件速查

| 主题 | 文件:行 |
|------|---------|
| `NCCL_STEPS`/协议常量 | `src/include/device.h:26,90-118` |
| `ncclConnInfo`(head/tail/connFifo) | `src/include/device.h:133` |
| `ncclSendMem`/`ncclRecvMem` | `src/include/comm.h:53,67` |
| `ncclConnFifo` | `src/include/collectives.h:72` |
| 角色分配 / `loadRecvConn`/`loadSendConn` | `src/device/prims_simple.h:473,527,611` |
| Wait/Post 循环 / fence | `src/device/prims_simple.h:322,381-397` |
| host→device work 循环 | `src/device/common.h:356,417` |
| proxy 主循环 | `src/proxy.cc:954,801` |
| 发送/接收 proxy 流水 | `src/transport/net.cc:1304,1470` |
| GDRCopy flush | `src/transport/net.cc:1641-1687` |
| connect() 指针布线 | `src/transport/p2p.cc:556-600`,`net.cc:507-588`,`shm.cc:168-194` |
| chunk 切分 | `src/enqueue.cc:2182` |
| channel 条带化 / 预算 | `src/enqueue.cc:576,294` |
| work 物化 | `src/enqueue.cc:302` |
| 离线带宽/延迟建模 | `src/graph/tuning.cc:245` |
| 运行时代价公式 | `src/graph/tuning.cc:630`,`src/enqueue.cc:2028,2123` |

> 建议读法:先按本文顺序通读一遍建立模型,再对照概览文档
> [`nccl_internals.md`](./nccl_internals.md) 的全流程图把四个机制"接线"。
> 遇到行为不符预期时,优先怀疑三个地方:(1) `tail/head` 语义是否搞反;
> (2) `NCCL_STEPS` 背压与 proxy 多段游标是否对齐;(3) 代价模型里某个修正因子
> 是否在该拓扑下生效。
