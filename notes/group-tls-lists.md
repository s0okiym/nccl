# Group 路径三条 thread-local 链表：`ncclGroupCommHead` / `PreconnectHead` / `ncclAsyncJobs`

> 基于 **NCCL 2.31.2** 源码核对。结论附 `文件:行号`。
> 对象：`src/group.cc:30-32` 定义、`src/include/group.h:56-61` 声明的三张 TLS 表。
>
> 它们是当前**用户线程**上、一次 `ncclGroupStart`/`ncclGroupEnd` 周期里的待办登记表。
> 不存 AllReduce 的 buffer/count（那些在 `comm->planner`），只回答：这次 group 有哪些 comm、要不要先 P2P connect、有哪些要异步跑的 job。

---

## 0. 作用域与哨兵

| 名字 | 作用域 | 生命周期 |
|---|---|---|
| `ncclGroupDepth` | 本线程 | `GroupStart++` / 最外层 `GroupEnd--`（`group.h:63-65`，`group.cc:28,1048`） |
| 三张 TLS 表 | **本线程** | 只活在一次最外层 group；`groupLocalResetJobState()`（`group.cc:505-511`）清空 |
| 真正的 task/plan | **每个 `ncclComm`** | `planner`、`rawTaskQueue`、`mgmtTaskQueue` |

NCCL 允许一个线程同时操作多个 comm（多 GPU 单进程）。三张表把这些 comm **按线程、按这次 group** 聚在一起。别的线程有自己的一份 TLS。

每个 `ncclAllReduce` 内部都会 `ncclGroupStartInternal()` 再 `ncclGroupEndInternal()`（`enqueue.cc:3396` 起）。用户没写 `ncclGroupStart` 时 depth 从 0→1→0，三张表仍然会用，只是周期极短。

底层是 **intrusive 单链表**：结点自带 `next`，队列不另分配结点。

```370:403:src/include/utils.h
template <typename T, T* T::* next>
struct ncclIntruQueue {
  T *head, *tail;
  int nElems;
};
// enqueue: x->next = nullptr; (head ? tail->next : head) = x; tail = x;
```

`ncclGroupCommHead` / `PreconnectHead` 更简单：不用 `ncclIntruQueue`，直接用 `comm->groupNext[type]` / `comm->preconnectNext`。

**哨兵（不是空指针）**：

- `groupNext[type] == 0x1`（`NCCL_COMM_GROUP_INVALID`，`group.h:17`）→ comm **还没** 加入该 type 的 group
- `preconnectNext == 0x1` → **不需要** / 已经不在 preconnect 链表里
- 链表**尾**才是 `nullptr`

`calloc` 出来的 comm 在 init 里写成 `0x1`（`init.cc:2701-2703`、`init.cc:549`），避免和 `nullptr` 混淆。

---

## 1. `ncclGroupCommHead[ncclGroupTaskTypeNum]`：按任务类型挂着的 comm 链表

```546:551:src/include/comm.h
typedef enum ncclGroupTaskType {
  ncclGroupTaskTypeCollective = 0,  // AllReduce / SendRecv 等
  ncclGroupTaskTypeSymRegister = 1, // 对称窗注册、CE init、mem suspend/resume
  ncclGroupTaskTypeRawTask     = 2, // enqueue rearch 路径的 raw task
  ncclGroupTaskTypeMgmtTask    = 3, // init/destroy/finalize/split
  ncclGroupTaskTypeNum = 4,
} ncclGroupTaskType_t;
```

每个 comm 有 `groupNext[4]`（`comm.h:749-751`），四种 type **各一条独立链表**。同一个 comm 可以同时出现在多条上。

### 插入：`ncclGroupCommJoin`（`group.h:80-116`）

1. 若 `groupNext[type] != 0x1`：已在链表，直接返回（幂等）。
2. 否则按 **`intraComm0` 相同的 sibling 挨在一起** 插（同一进程 split、共享资源的 comm）。
3. 找不到 clique，再按 **`commHash` 升序** 插新 clique。

`doLaunches()`（`group.cc:427`）依赖这个顺序：外层按 clique 切，内层 round-robin 发射 kernel；同一 clique 还可能走 `ncclCommIntraBarrier`。

Join Collective / RawTask 时还会：

- `ncclMemoryStackPush(&comm->memScoped)` —— 这次 group 的 task 从这块 scoped 内存出
- `memset` planner（保留 `peers` / `rmaTaskQueues` 指针）

```text
本线程 ncclGroupCommHead[Collective]
  │
  ▼
 commA ──groupNext[Coll]──► commB ──► commC ──► nullptr
 intraComm0=X                intraComm0=X         intraComm0=Y
          └── 同一个 clique ──┘              └── 下一个 clique

ncclGroupCommHead[MgmtTask] 可能是另一条：commA → commD → nullptr
```

Join 调用点（入队的是 **comm**，不是 buffer）：

| type | 谁 Join |
|---|---|
| Collective | `p2pTaskAppend` / `collTaskAppend` 等（`enqueue.cc:2615` 等） |
| RawTask | rearch 路径（`enqueue.cc:3237`） |
| SymRegister | `dev_runtime.cc`、`mem_manager.cc`、CE init |
| MgmtTask | `ncclMgmtTaskEnqueue`（`mgmt_task_enq.cc:24`） |

真正的工作挂在 comm 内部：

- Collective → `comm->planner` 的 coll/p2p/rma 队列
- RawTask → `comm->rawTaskQueue`
- MgmtTask → `comm->mgmtTaskQueue`
- SymRegister → `devrState.regTaskQueue` / `ceInitTaskQueue` 等

`Head` 只回答：**GroupEnd 时扫哪些 comm、按什么 clique 顺序扫。**

---

## 2. `ncclGroupCommPreconnectHead`：需要先建 P2P 连接的 comm 子集

这是 Collective 链表的 **子集**，只登记「这次 group 里第一次发现某个 P2P peer 的 channel 还没 connect」的 comm。

```119:124:src/include/group.h
inline void ncclGroupCommPreconnect(struct ncclComm* comm) {
  if (comm->preconnectNext == reinterpret_cast<struct ncclComm*>(0x1)) {
    comm->preconnectNext = ncclGroupCommPreconnectHead;
    ncclGroupCommPreconnectHead = comm;
  }
}
```

**头插**单链表，不排序。同一 comm 只进一次。

触发点在 `p2pTaskAppend`（`enqueue.cc:2643-2674`）：给 `peer` 标 `connectSend/Recv[peer] |= (1ULL << channelId)` 后调用。

```text
ncclGroupCommPreconnectHead
  │
  ▼
 commB ──preconnectNext──► commA ──► nullptr     （0x1 表示不在这条链）
```

`groupLaunchLegacy`（`group.cc:757-776`）：

1. 沿这条链为每个 comm 造 `ncclPreconnectJob`（`func = ncclP2PPreconnectFunc`）
2. **Enqueue 进 `asyncJobs`**（和 init/destroy job 同一队列）
3. 把 `preconnectNext` 恢复成 `0x1`
4. `asyncJobLaunch`：一个 job 就本线程跑；多个就每 comm 一条线程去 connect

Collective **算法通道**的 preconnect 不走这条 TLS 链，而是在 `ncclPrepareTasksAndCollPreconnect` 里另建 `asyncCollJobs`（`group.cc:811-828`）。`PreconnectHead` 专指 **P2P send/recv 的运行时连接**。

---

## 3. `ncclAsyncJobs`：本线程 group 里要异步执行的 job 队列

类型：`ncclIntruQueue<ncclAsyncJob, &ncclAsyncJob::next>`，FIFO。

```22:43:src/include/enqueue/mgmt_task_enq.h
struct ncclAsyncJob {
  struct ncclAsyncJob* next;
  std::thread thread;
  ncclResult_t result;
  ncclResult_t (*func)(struct ncclAsyncJob*);
  void (*undo)(struct ncclAsyncJob*);
  void (*destructor)(void*);
  ncclGroupJobState_t state;   // Running / Done / Joined
  uint32_t* abortFlag; ...
  ncclComm_t comm;
  int destroyFlag;
  bool isThreadMain;
};
```

Job 是基类。真实对象是 `ncclCommInitRankAsyncJob`、`ncclPreconnectJob` 等，第一字段 `ncclAsyncJob base`，靠 `next` 链进队列。

入队入口 `ncclAsyncLaunch`（`group.cc:36-74`）：

- **`ncclGroupDepth == 0`**：立刻 `func(job)`，然后 destructor。不进队列。
- **在 group 里**：填 func/undo/destructor/abortFlag，`ncclIntruQueueEnqueue(&ncclAsyncJobs, job)`。  
  同时检查 group 里不能混 blocking / nonblocking comm。

典型来源：`ncclCommInitRank` / `Destroy` / `Finalize` / `Split` / `Grow`（`init.cc` 的 `ncclAsyncLaunch`）。用户可以：

```c
ncclGroupStart();
ncclCommInitRank(&c0, ...);
ncclCommInitRank(&c1, ...);
ncclGroupEnd();  // 两个 init job 在 ncclAsyncJobs 里并行
```

**不要和 `comm->mgmtTaskQueue` 搞混。** 新路径 `ncclMgmtTaskEnqueue` 把 job 挂在 **comm 自己的** `mgmtTaskQueue` 上，只把 comm Join 到 `Head[MgmtTask]`。`ncclAsyncJobs` 是线程级「游离 job 袋」，主要是老 `ncclAsyncLaunch` + GroupEnd 时从 PreconnectHead **转化出来的** preconnect job。

---

## 4. 业务流程

```text
 用户线程
 ────────────────────────────────────────────────────────────────
  GroupStart (depth++)
      │
      ├─ AllReduce / Send：Join Head[Collective]，task 进 planner
      ├─ P2P 未 connect：PreconnectHead 头插
      ├─ InitRank：job 进 ncclAsyncJobs
      └─ MgmtTaskEnqueue：Join Head[MgmtTask]，job 进 comm->mgmtTaskQueue
      │
  最外层 GroupEnd
      │  memcpy Head + PreconnectHead
      │  transfer AsyncJobs → heap 上的 ncclGroupJob
      │  groupLocalResetJobState() 立刻清空 TLS
      ▼
  groupLaunch（本线程或非阻塞后台线程）
      ① PreconnectHead → 生成 job 并入 asyncJobs
      ② 并行跑 asyncJobs（init / P2P connect）
      ③ Head[SymRegister] 按 clique 注册
      ④ Head[Collective] prepare + coll connect + enqueue
      ⑤ doLaunches(Head[Collective]) 按 clique 发 kernel
      ⑥ Leave 各 Head，pop memScoped
```

### 阶段 A：收集（depth > 0）

用户线程只登记，不 launch kernel（除 depth==0 的 `ncclAsyncLaunch` 同步跑）。

```text
TLS:
  Head[Collective]  →  comm0 → comm1
  Head[MgmtTask]    →  comm0
  PreconnectHead    →  comm1
  AsyncJobs         →  [InitJob_c2] → [InitJob_c3]

comm0.planner:  AllReduce task × N
comm1.planner:  Send/Recv task，connectSend[] 已置位
comm0.mgmtTaskQueue: Finalize job（若走新路径）
```

### 阶段 B：最外层 GroupEnd 交接（`group.cc:1026-1136`）

1. `depth--`，若仍 >0（嵌套 group）直接返回，表继续攒。
2. 若 `ncclGroupError` 已失败 → `groupCleanup`。
3. 堆上分配 `ncclGroupJob`（`group.h:34-44`）：
   - `memcpy(groupJob->groupCommHead, ncclGroupCommHead, ...)`
   - `groupJob->groupCommPreconnectHead = ncclGroupCommPreconnectHead`
   - `ncclIntruQueueTransfer(&groupJob->asyncJobs, &ncclAsyncJobs)` —— **整链搬走，TLS 队列变空**
4. 三张表都空（空 group）→ 丢掉 groupJob。
5. **blocking**：本线程 `groupLaunch()`。  
   **nonblocking**：`groupJob` 丢到新线程，`GroupEnd` 返回 `ncclInProgress`，comm 上挂 `groupJob` 引用计数。
6. **立刻** `groupLocalResetJobState()`：TLS 清零，好接下一次 group。  
   后面真正干活用的是 **groupJob 里的拷贝**。

### 阶段 C：`groupLaunch` 消化顺序

`groupLaunch`（`group.cc:1018`）按 `NCCL_ENQUEUE_REARCH_ENABLE` 分派 `groupLaunchLegacy` / `groupLaunchEnqueueRearch`。

Legacy（`group.cc:750+`）对 **groupJob 副本**：

1. **PreconnectHead** → 为每个 comm 造 P2P preconnect job，并入 `asyncJobs`，哨兵改回 `0x1`
2. **`asyncJobLaunch(asyncJobs)`**（`group.cc:76`）  
   - 1 个 job：本线程跑 `ncclAsyncJobMain`  
   - N 个：每 job 一条 `std::thread`，轮询 `state` 直到 Done，再 join
3. **Head[SymRegister]**：按 clique 造 `ncclGroupSymmetricJob`，并行注册窗口 / CE / mem
4. **Head[Collective]**：clique 内 `ncclPrepareTasksAndCollPreconnect` → 再 `asyncJobLaunch` 连 collective 通道 → `ncclTasksRegAndEnqueue`
5. **`doLaunches(Head[Collective])`**：按 clique、按 round 调 `ncclLaunchKernel`
6. 析构 asyncJobs；沿各 `Head[type]` `ncclGroupCommLeave`（`groupNext` 写回 `0x1`，pop memScoped）

Rearch 路径：先 `Head[MgmtTask]` 的 `mgmtTaskQueue`，再 RawTask prepare/schedule，最后 `doLaunches(Head[RawTask])`。TLS 三张表角色不变。

### 阶段 D：失败

`groupCleanup`（`group.cc:514`）沿 Head 把 planner/task 退回内存池，Preconnect 哨兵复位，asyncJobs destructor，nonblocking comm 设 async error。

---

## 5. 三者关系

```text
一次 GroupEnd 看到的 TLS：

ncclGroupCommHead[Collective]          ncclGroupCommPreconnectHead
        │                                        │
        ▼                                        ▼
     comm0 ──► comm1                        comm1 ──► nullptr
        │         │                           │
        │         └── planner 里有 Send        └── Head 的子集
        └── planner 里有 AllReduce                 （需要 P2P connect）

ncclAsyncJobs
        │
        ▼
   InitJob(comm2) ──► InitJob(comm3) ──► nullptr
        （不经过 Head；job.comm 指向新 comm）

GroupEnd 之后：
  PreconnectHead 的 comm1  ──转化为──►  PreconnectJob  ──并入──► asyncJobs
  asyncJobs 先被并行执行
  再扫 Head[Collective] 做 prepare / launch
```

| 结构 | 节点 | next 在哪 | 回答什么问题 |
|---|---|---|---|
| `Head[type]` | `ncclComm*` | `comm->groupNext[type]` | 这次 group 有哪些 comm、clique 顺序 |
| `PreconnectHead` | `ncclComm*` | `comm->preconnectNext` | 哪些 comm 要先做 P2P connect |
| `ncclAsyncJobs` | `ncclAsyncJob*` | `job->next` | GroupEnd 时要（并行）跑哪些函数 |

`Head` 是 comm 的索引；工作在 comm 内部队列。`PreconnectHead` 是索引的索引，GroupEnd 时 **物化成 job 再并进 `ncclAsyncJobs`**。`ncclAsyncJobs` 才是可执行对象队列。

---

## 6. 读代码时容易混的点

1. **Join 幂等，Preconnect 也幂等。** 同一 comm 同一 type 多次 AllReduce 只 Join 一次；planner 里会有多个 task。
2. **空指针 vs `0x1`。** 链表尾是 `nullptr`；「不在链表」是 `0x1`。判断要不要 Join 看 `== 0x1`。
3. **TLS 在 GroupEnd 一开始就被搬走/清空。** 之后只看 `ncclGroupJob`。非阻塞 group 里用户线程的 TLS 已经空了，工作在后台线程的 `groupJob` 上。
4. **`ncclAsyncJobs` ≠ `mgmtTaskQueue`。** 一个在线程，一个在 comm；新管理任务走后者 + `Head[MgmtTask]`。
5. **隐式 group 同样走这三张表。** 单次 `ncclAllReduce` 也会 Join Head、可能 Preconnect、GroupEnd 立刻 launch。

一句话：`Head` 登记「谁要干活、按 clique 怎么排」，`PreconnectHead` 登记「谁要先接线」，`AsyncJobs` 登记「GroupEnd 时要跑哪些函数」；最外层 `GroupEnd` 按 **异步 job（含 P2P connect）→ 对称注册 → collective 准备/连接 → `doLaunches`** 把它们吃掉。

---

## 7. Intrusive 链表：next 长在对象里，不是 `void*` 容器

`ncclIntruQueue<T, &T::next>`（`utils.h:369`）是 **侵入式** 单链表：`next` 是 `T` 自己的字段，队列只有 `head/tail/nElems`。Enqueue 改两个指针，不另 `new` 包装结点。

这和 C 里常见的「容器塞 `void*`」相反：

```text
非侵入: Queue → [Node|void*] → [Node|void*]  再 chase 到 Job（常要为 Node 再分配）
侵入:   Queue.head ──────────► Job { next ──► Job }
```

热路径要零分配、dequeue 直接得到 `T*`、同一 comm 用不同字段同时挂多条链（`groupNext[4]`、`preconnectNext`）。代价：每个 next 字段同时只能在一条链上；「不在链上」不能用 `nullptr`（链尾已经是），所以用 `0x1`。

---

## 8. Group TLS 在哪条线程

**调用 NCCL API 的那条用户线程** 的 C++ `thread_local`，NCCL 不为它 `pthread_create` / `malloc`。多数程序里就是 main / MPI rank 主循环；多 pthread 分别调 NCCL 则各有一份互不可见的表。

`GroupStart` 与配对的 `GroupEnd` 必须在同一线程。最外层 End 把表 `memcpy`/`transfer` 进堆上的 `ncclGroupJob` 后立刻 `groupLocalResetJobState()`。之后 blocking 仍在调用线程跑 `groupLaunch(groupJob)`；nonblocking 才另开线程，新线程有自己的空 TLS，吃的是 `groupJob` 拷贝。Proxy / Bootstrap / RAS 不填这三张表。

---

## 9. Clique：同进程、同世界的一组 `ncclComm`

此处 **不是** 图论完全子图，也不是 NVLink/MNNVL 拓扑 clique。

`doLaunches`（`group.cc:432-435`）：clique = `groupNext` 上 **`intraComm0` 相同** 的连续一段。`intraComm0` 是本进程里该世界 communicator 的 leader（`hostHash+pidHash` 相同、global rank 最小的那个 `ncclComm*`，`init.cc:1160-1212`）。

```text
2 进程 × 4 GPU：P0 rank 0–3 → clique A（leader=comm0）；P1 rank 4–7 → clique B
```

用途：`NCCL_LAUNCH_MODE=GROUP` 时在 `intraComm0` 上做进程内 barrier，各卡同一 round 发一个 plan；同一 clique 不能混 CUDA Graph capture；split 共享连接时 preconnect 按 clique 切开。`Join` 故意把同 `intraComm0` 的 comm 插在一起。`intraNext` 是 clique 的永久成员链；`groupNext` 只是这一次 group 的待办链。

---

## 10. `PreconnectHead` 只因 P2P 接线，comm 上仍可有 collective

全库只有 `p2pTaskAppend` 调用 `ncclGroupCommPreconnect`（`enqueue.cc:2664,2672`）：某 `Send`/`Recv` 发现该 peer 的 P2P connector 尚未建立。

Collective 通道连接走 GroupEnd 的 `ncclCollPreconnect`（`group.cc:226,811-828`），**不登记** 到 `PreconnectHead`。

`p2pTaskAppend` 先 `Join(..., Collective)`，所以在 `PreconnectHead` ⇒ 一定也在 `Head[Collective]`。反过来不成立（纯 AllReduce，或 P2P 早已连好）。同一 comm、同一次 group 可以既有 AllReduce 又有 Send：planner 里两种 task 都有；End 先按 `PreconnectHead` 跑 P2P connect，再扫 `Head[Collective]` prepare/launch。

---

## 11. `ncclMgmtTaskEnqueue`：默认关着的管理任务入队口

2.31.2 enqueue 重构路径（`src/enqueue/mgmt_task_enq.cc`）。`NCCL_ENQUEUE_REARCH_ENABLE` 默认 **0**（`enqueue.cc:33`），所以按 2.30 习惯读 `init.cc` 的 `else` 只会看到 `ncclAsyncLaunch`。

打开后：管理类 job（init/destroy/finalize/split、对称窗注册、CE init、mem suspend）不进 TLS `ncclAsyncJobs`，而是 `Join Head[MgmtTask]` + 入 `comm->mgmtTaskQueue`。`groupLaunch` 走 `groupLaunchEnqueueRearch`（`group.cc:1018`）：先按 clique 跑 `ncclMgmtTaskJobFunc` 消化各 comm 的管理队列，再 `Head[RawTask]` prepare，再 `doLaunches`。

| | `ncclAsyncLaunch`（默认） | `ncclMgmtTaskEnqueue`（rearch=1） |
|---|---|---|
| 队列 | TLS `ncclAsyncJobs` | `comm->mgmtTaskQueue` |
| GroupEnd | `groupLaunchLegacy` 开头一锅端 | 固定阶段 ① Mgmt ② Raw prepare ③ launch |

数据面 collective 不走 MgmtTask。Destroy 时 `ncclMgmtTaskJobFunc` 只 dequeue **一个** 任务（`group.cc:312-315`）。
