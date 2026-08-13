# NCCL `src/transport/net.cc` 生命周期导读

> **给谁看**：把你当小白——读完应能回答：这个文件在 NCCL **整段生命**里干什么、**谁在什么时候**按什么顺序调用、**各自起什么作用**。  
> **对应源码**：`src/transport/net.cc`（约 2133 行）、`src/transport.cc`（`ncclTransportP2pSetup`）、`src/proxy.cc`（RPC）  
> **本文合并自**：原 `nccl_net_sendsetup_rpc.md` + `nccl_transport_net_cc_functions.md`  
> **还可对照**：[`net-cc-beginner-guide.md`](./net-cc-beginner-guide.md)（shared 比喻）、[`net-cc-analysis.md`](./net-cc-analysis.md)、[`proxy-internals.md`](./proxy-internals.md)

---

## 目录

1. [先建立三张图](#1-先建立三张图)
2. [术语（务必先读）](#2-术语务必先读)
3. [完整时间线：从 init 到销毁](#3-完整时间线从-init-到销毁)
4. [阶段展开：主线程建连 `P2pSetup`](#4-阶段展开主线程建连-p2psetup)
4.1. [建连总览：一条 NET 边要过几道门](#41-建连总览一条-net-边要过几道门)
4.2. [`sendSetup` 逐步流程](#42-sendsetup-逐步流程)
4.3. [`recvSetup` 逐步流程](#43-recvsetup-逐步流程)
4.4. [`sendProxySetup` / `recvProxySetup`](#44-sendproxysetup--recvproxysetup)
4.5. [Bootstrap 交换 connectInfo](#45-bootstrap-交换-connectinfo)
4.6. [`sendConnect` 逐步流程](#46-sendconnect-逐步流程)
4.7. [`recvConnect` 逐步流程](#47-recvconnect-逐步流程)
4.8. [`sendProxyConnect` / `recvProxyConnect`](#48-sendproxyconnect--recvproxyconnect)
5. [RPC 是什么：Init / Setup / Connect](#5-rpc-是什么init--setup--connect)
6. [稳态：几乎只剩 Progress](#6-稳态几乎只剩-progress)
6.1. [Progress 入门：args / sub / 三个计数器](#61-progress-入门args--sub--三个计数器)
6.2. [`sendProxyProgress` 逐步流程](#62-sendproxyprogress-逐步流程)
6.3. [`recvProxyProgress` 逐步流程](#63-recvproxyprogress-逐步流程)
7. [注册、销毁、例外](#7-注册销毁例外)
8. [关键注意事项](#8-关键注意事项)
9. [重要数据结构表](#9-重要数据结构表)
10. [全部函数表（含义 + 何时调用）](#10-全部函数表含义--何时调用)
11. [调用关系次数梳理](#11-调用关系次数梳理)

---

## 1. 先建立三张图

### 1.1 `net.cc` 在 NCCL 里是谁？

```text
用户: ncclAllReduce / ncclSend
        │
        ▼
算法在 GPU 上跑（Ring/Tree…）
        │  跨节点时需要「走网卡」
        ▼
┌───────────────────────────────────┐
│  net.cc = 运输调度室 + 货站规章    │
│  不负责真正开车（那是网卡插件）    │
└─────────────┬─────────────────────┘
              │
     ┌────────┴────────┐
     ▼                 ▼
 GPU 写中转区      Proxy 调 isend/irecv
                       ▼
                 net_ib / socket 插件
                       ▼
                    物理网络
```

**一句话：** GPU 不能直接调网卡库；`net.cc` 规定 **如何建连、货放哪、GPU 和 proxy 怎么对账**，真正发包的是插件。

### 1.2 三个角色

| 角色 | 是谁 | 稳态干什么 |
|------|------|------------|
| **主线程** | 你调 `ncclCommInit` / `ncclAllReduce` 的线程 | init 建连、enqueue；**稳态一般不再进 Setup** |
| **Proxy 服务线程** | 收主线程 RPC（Setup/Connect/Reg） | **建连阶段**忙，稳态很少 |
| **Proxy 进度线程** | 循环推进网络 op | **稳态核心**：`send/recvProxyProgress` |

（实现上服务与进度可能同属 proxy 体系；你只需记住：**控制面 RPC** vs **数据面 Progress**。）

### 1.3 两条「带外」通道（别混）

| 通道 | 谁跟谁 | 干什么 |
|------|--------|--------|
| **Proxy RPC** | 主线程 ↔ **本机（或 PXN）proxy** | 在 proxy 上建资源、listen/connect、regMr |
| **Bootstrap** | 本 rank ↔ **对端 rank** | 交换 handle、proxyRank、useGdr |

---

## 2. 术语（务必先读）

| 词 | 在本文 / `net.cc` 里的意思 | 不是 |
|----|---------------------------|------|
| **P2P** | **走 NET 的 send/recv 点对点边** | 本机 GPU NVLink peer（那是 `TRANSPORT_P2P` / `p2p.cc`） |
| **graph** | **`ncclTopoGraph*`**：拓扑搜索出的 Ring/Tree 图 | CUDA Graph |
| **shared（缓冲）** | 多条**网络 P2P** 边共用一块中转池 | SHM 传输 |
| **shared comms** | 多条流共用一个插件 `sendComm` | 同上 |
| **RPC Setup** | 主线程发 `ncclProxyMsgSetup`，proxy 跑 `*ProxySetup` | 不是 bootstrap，也不是连上网卡 |
| **NCCL_STEPS** | 环形货位个数，一般 **8** | — |

`sendSetup` 里：

```c
shared = graph || connIndex == 0 ? 0 : /* 默认网络 P2P */ 1;
```

有 **topo graph** 或主连接槽 0 → **专用缓冲**；网络 send/recv 边 → **默认可共享池**。

---

## 3. 完整时间线：从 init 到销毁

```text
时间 ──────────────────────────────────────────────────────────────►

① bootstrapInit
   带外握手 uniqueId / 地址
   ★ 还不进入 net.cc

② initTransportsRank
   拓扑、AllGather、起 proxy
   ★ 仍几乎不进 net.cc 的 setup
   可能：ProxyMsgSharedInit → proxySharedInit（预建 P2P 池）

③ 集体建边  ncclTransportP2pSetup(comm, &graphs[RING/TREE], 0)
   主线程:
     canConnect → sendSetup/recvSetup → bootstrap 交换 → sendConnect/recvConnect
   proxy（被 RPC 叫醒）:
     send/recvProxySetup → send/recvProxyConnect
   ★ 每条「跨节点 NET 集体边」走一遍 Setup+Connect

④ （可选）NVB 预连接 / 其它 graph
   P2pSetup(NULL, 1) 或 NVLS/CollNet 的 P2pSetup
   再走 ③ 同一套路

⑤ 第一次 ncclAllReduce（边已连好）
   主线程: enqueue + launch kernel   ★ 一般不再进 sendSetup
   GPU: 写 buff / tail / size
   proxy 进度: sendProxyProgress / recvProxyProgress  ← 热路径开始

⑥ 运行中第一次对「还没连的 peer」ncclSend
   再 P2pSetup(NULL,1) → Setup+Connect   ★ 例外：runtime connect

⑦ 可选 Register / Graph 捕获
   ncclNetLocal/GraphRegisterBuffer → RPC Reg

⑧ Abort / Destroy
   sendFree / recvFree（主）+ send/recvProxyFree（proxy）
```

---

## 4. 阶段展开：主线程建连 `P2pSetup`

源码：`src/transport.cc` `ncclTransportP2pSetup`。  
**线程：主线程。** bootstrap 此时 **已经可用**。

下面把 **建连核心函数** 按「一条跨节点发送边 + 对端接收边」走完。进度函数见 §6。

### 4.1 建连总览：一条 NET 边要过几道门

```text
① canConnect          这条边能不能走 NET
② sendSetup           发送 rank 主线程：决策 + RPC Init + RPC Setup
   └ sendProxySetup   发送 proxy：建 sendNetResources 空壳
③ recvSetup           接收 rank 主线程：决策 + RPC Init + RPC Setup
   └ recvProxySetup   接收 proxy：listen，返回 handle
④ bootstrapSend/Recv  两端主线程交换 connectInfo（handle / proxyRank / useGdr）
⑤ sendConnect         发送主线程：RPC Connect（可多次 poll）
   └ sendProxyConnect 发送 proxy：插件 connect + 分配货位 + regMr
⑥ recvConnect         接收主线程：RPC Connect
   └ recvProxyConnect 接收 proxy：插件 accept + 货位 + regMr
⑦ cudaMemcpy conn → GPU
⑧ 再一条 bootstrap 栅栏，清 connect mask
```

同机常被 `TRANSPORT_P2P`/`SHM` 的 `canConnect` 抢走，**整段 NET 建连都不会发生**。

---

### 4.2 `sendSetup` 逐步流程

**谁调：** `selectTransport` 选中 NET 之后，主线程立刻调。  
**还不连网卡。** 只「想清楚 + 通知本侧 proxy 开户」。

**（1）定 shared**

```c
shared = graph || connIndex == 0 ? 0 : 默认 1;
```

集体 topo 边 / 槽 0 → 专用货位；网络 send/recv 边 → 默认可共享池。

**（2）选网卡和谁当快递员（proxy）**

```c
ncclTopoGetNetDev(..., peerRank, &netDev, &proxyRank);
```

- `netDev`：插件里的网卡号。  
- `proxyRank`：由哪个 rank 的 proxy 出网。  
  - 等于自己：本 GPU 的 proxy 发。  
  - **不等于自己 = PXN**：旁边更靠近 NIC 的 GPU 的 proxy 发。  
- 集体主连接且 PXN → `comm->useNetPXN = true`。

**（3）GDR**

```c
ncclTopoCheckGdr(..., isSend=1, &useGdr);
if (useGdr) flags |= NCCL_DIRECT_NIC;
```

NIC 能否直接 DMA GPU。集体主连接若不能，会把 comm 级 `useGdr` 关掉。

**（4）挂上 proxy 电话：`ncclProxyConnect(NET, send=1, proxyRank)`**

内部（`proxy.cc`）：

1. 算 `sameProcess`（同机同 pid？）。  
2. 没有到该 proxy 的 socket → 连上。  
3. **RPC Init**（`ncclProxyMsgInit`）：proxy 上登记「这是一条 NET **发送** connection」，返回 `connection*` cookie。  
4. 若该传输有 Progress，打开共享 ops 池。  

**（5）填 `setupReq` 并发 RPC Setup**

字段：shared、netDev、useGdr、channel、connIndex、本端/远端 top-parent rank、`sameDevice`（kernel 的 GPU 和 **发包那个 proxy 所在 GPU** 是否同一块）。

```c
ncclProxyCallBlocking(..., ncclProxyMsgSetup, &req, sizeof(req), NULL, 0);
```

无响应体。等到 `sendProxySetup` 做完才返回。

**（6）写 `connectInfo` 给对端**

- 开头：`topParentRanks[proxyRank]`（对端要知道是谁连过来；PXN 时不是数据 GPU 自己）。  
- 偏移 `sizeof(ncclNetHandle_t)`：`useGdr`。  

发送 Setup **没有** listen handle；那一块先拿来塞 proxyRank。

打日志：`Channel xx : me -> peer [send] via NET/IB/dev /GDRDMA /Shared (proxyRank)`。

---

### 4.3 `recvSetup` 逐步流程

和 send **不对称** 的几点：

| | sendSetup | recvSetup |
|--|-----------|-----------|
| 选网卡的 peer | **对端 rank**（为发出去选路） | **自己**（接收用本机 NIC） |
| proxyRank | 可以是别人（PXN） | **必须本 rank**（「We don't support PXN on receive」） |
| 额外 | — | GDR 时算 `needFlush` |
| RPC Setup 响应 | 无 | **`ncclNetHandle_t`**（listen 句柄） |

流程：同样定 shared → 选本端网卡/GDR/flush → `ProxyConnect(NET, send=0, **myRank**)` → Blocking Setup，响应写进 `connectInfo` 前部 → 再附 useGdr。

对端 send 后面用这个 handle 去 `connect`。

---

### 4.4 `sendProxySetup` / `recvProxySetup`

都在 **proxy 服务线程**，被 `proxyProgressAsync` 看到 `ncclProxyMsgSetup` 后调用。

**`sendProxySetup`**

1. 检查 `reqSize`。  
2. `calloc sendNetResources`，挂 `connection->transportResources`。  
3. 把 req 全部抄进 resources。  
4. `ncclNet->getProperties(netDev)`：DMA-BUF？`maxRecvs`？device-net 类型？`maxP2pBytes` 是否合法？  
5. **到此结束。** 不 listen、不 connect、不分配数据大缓冲。  
6. `*done=1`，connection 标 `connSetupDone`。

**`recvProxySetup`**

前几步相同，然后：

```c
ncclNet->listen(netDev, respBuff, &netListenComm);
```

`respBuff` 就是主线程拿到的 **handle**。listen 句柄留在 `resources->netListenComm`，等 Connect 时 `accept`。

---

### 4.5 Bootstrap 交换 connectInfo

仍在 `ncclTransportP2pSetup` 主线程，**不进 net.cc**：

```text
Rank A sendSetup 写出:  [proxyRank | .... | useGdr_A]
Rank B recvSetup 写出:  [listen handle........ | useGdr_B]
        bootstrapSend/Recv 对调
Rank A 之后 sendConnect 读到: B 的 handle + useGdr_B
Rank B 之后 recvConnect 读到: A 的 proxyRank + useGdr_A
```

同一 peer 的多个 channel 的 `ncclConnect` 打成一包交换。

---

### 4.6 `sendConnect` 逐步流程

**谁调：** `P2pSetup` 在 bootstrap 交换之后，对 `connected==0` 的 send connector。  
**可重入：** 插件 connect 未完成时返回 `ncclInProgress`，外层 while 再进来。

**第一次进来（还没有 map）：**

1. 读对端 `useGdr`；对端没有 GDR → **清掉本端 `DIRECT_NIC`**（必须两边都能直 DMA）。  
2. `calloc connectMap` 挂在 `send->transportResources`（主线程这份是「地图副本」）。  
3. `populateCommNetAttrs`：按集体/网络 P2P 填并发 hint。  
4. 把对端 **handle** + netAttr 打成 `netSendConnectArgs`。  
5. **`ncclProxyCallAsync(Connect)`**（注意是 Async，不是 Blocking）。  
6. `PollProxyResponse` 把 proxy 填好的 map 拷回来。

**再进来（map 已有）：** 只 Poll，不再发 RPC。

**Poll 成功之后（主线程把地图变成 GPU 能用的指针）：**

- 同进程、不同 GPU、老 IPC：可能 `cudaDeviceEnablePeerAccess`。  
- 跨进程（PXN）：`netMapShm` 导入 HOST；DEVMEM / 共享池 `ncclP2pImportShareableBuffer`。  
- 从 map **解出 GPU 指针**，填：

| `send->conn` | 指向 |
|--------------|------|
| `head` | sendMem->head 或 GDRCopy 映射 |
| `tail` | recvMem->tail |
| `connFifo` | recvMem->connFifo |
| `buffs[p]` | 各协议货位 |
| `stepSize` | SIMPLE 缓冲 / 8 |

- shared 则 `connFifo.mode = OFFSET`。  
- 挂 `proxyProgress = sendProxyProgress`（设备网且不需 proxy 时为 NULL）。

外层成功则 `cudaMemcpyAsync` 整份 `conn` 到 device。

---

### 4.7 `recvConnect` 逐步流程

与 send 对称：

1. 读对端 send 的 useGdr，必要时清 DIRECT_NIC。  
2. 第一次：Async Connect，载荷是 **对端 proxyRank** + netAttr（不是 handle；本端已经 listen 过了）。  
3. Poll 拿 map。  
4. Recv **禁止 remote proxy**（map.sameProcess 必须为真，在 ProxyConnect 里就会查）。  
5. 填 `recv->conn`：`head` 来自 sendMem，`tail` 可能是 gdcSync。  
6. 挂 `recvProxyProgress`。

---

### 4.8 `sendProxyConnect` / `recvProxyConnect`

**线程：proxy。** 这才是「连上网卡 + 准备货位」。

**`sendProxyConnect`**

1. `setNetAttrs`。  
2. `ncclNetGetDeviceHandle`（多数情况不用）。  
3. **`ncclNet->connect(netDev, 对端 handle, &netSendComm)`**  
   - shared + `NET_SHARED_COMMS` + `maxRecvs>1`：按 netDev×远端 rank **复用** 一个 sendComm。  
   - `netSendComm==NULL` → `*done=0`，主线程下次再 Poll（异步 connect）。  
4. `*done=1` 后分配 **connectMap**：  
   - 非 shared：每协议一块专用缓冲（GDR 则放 GPU）。  
   - shared：挂上共享池。  
   - 永远有 sendMem/recvMem（HOST；跨进程则 `netCreateShm`）。  
   - 可选 GDRCopy `gdcSync`。  
5. `head = shared ? -NCCL_STEPS : 0`；`connFifo[].size = -1`。  
6. 对各协议缓冲 `regMr` 或 `regMrDmaBuf`。  
7. 整份 map **拷回**主线程。

**`recvProxyConnect`**

1. 记下对端 `proxyRank`。  
2. **`ncclNet->accept(netListenComm, &netRecvComm)`**（也可复用 recvComm）。  
3. 未完成 → InProgress。  
4. **立刻 `closeListen`**（这条 listen 已用完）。  
5. `sameProcess==0` → 内部错误（recv 无 PXN）。  
6. 分配货位 + host 控制面 + 可选 `gdcSync/gdcFlush`。  
7. `regMr`，返回 map。

Connect 成功后 connection 标 **`connConnected`**。此后稳态只走 Progress。

### 4.9 建连结束时你手里有什么

```text
主线程 / GPU:
  connector.conn = { buffs, head, tail, connFifo, flags }
  proxyConn.proxyProgress = send/recvProxyProgress

Proxy:
  send/recvNetResources = { netSend/RecvComm, map, mhandles, step, ... }

对端对称的一套
```

**④ 再一条 bootstrap 栅栏**，清 `connectSend/Recv` mask，防止有人已经开始拆连接、别人还在 import 缓冲。

---

## 5. RPC 是什么：Init / Setup / Connect

主线程和 proxy 可能不同进程，所以用 **socket 请求–应答**，叫 RPC。

```text
主线程 CallBlocking/Async
  socket: 类型 + connection* + setupReq + opId
proxy 服务线程
  tcomm->proxySetup / proxyConnect
  socket 回: opId + 结果
```

| 消息 | 主线程入口 | Proxy 干活 | 插件 |
|------|------------|------------|------|
| **Init** | `ncclProxyConnect` 内部 | 登记 connection | 无 |
| **Setup** | `send/recvSetup` | `*ProxySetup` | send 仅 getProperties；recv **listen** |
| **Connect** | `send/recvConnect` | `*ProxyConnect` | **connect/accept** + 缓冲 + regMr |
| Register | `netRegisterBuffer` | `*ProxyRegBuffer` | regMr 用户缓冲 |
| Deregister | `ncclNetDeregBuffer` | `*ProxyDeregBuffer` | deregMr |

**Setup ≠ 连上网卡。** Setup = 在正确的 proxy 上创建并配置资源空壳。

`CallBlocking` = `CallAsync` + 自旋 `Poll` 直到完成。Setup 很快；Connect 才可能多次 InProgress。

---

## 6. 稳态：几乎只剩 Progress

边已连上、缓冲已注册之后：

| 谁 | 每步集体/sendrecv |
|----|-------------------|
| **主线程** | enqueue + launch；**一般不再进 Setup/Connect** |
| **GPU** | 写/读货位，改 tail、`connFifo.size`（不在 net.cc） |
| **Proxy 进度** | **`sendProxyProgress` / `recvProxyProgress`** → 插件 isend/irecv/test |

**`sendProxyProgress` 三步（每个切片）：** posted → isend → test。  
**`recvProxyProgress` 五段：** 分组 → irecv → test → flush → 通知 GPU → 等 GPU 还 credit。  

下面把两个函数按「小白能跟着走完一遍」写细。源码约 `net.cc:1324` / `1493`。

### 6.1 Progress 入门：args / sub / 三个计数器

每次集体或 send/recv，enqueue 会给 **这条 NET 边**（或一组边）挂一个 `ncclProxyArgs`：

| 字段 | 白话 |
|------|------|
| `args->nsubs` | 这次要推进几条「子流」（常 = 几条连接/channel） |
| `args->subs[s]` | 第 s 条子流 |
| `args->protocol` | LL / LL128 / SIMPLE |
| `args->sliceSteps` / `chunkSteps` | 一次推进多少「步」（切片粒度） |
| `args->state` | Ready → Progress → None |
| `args->idle` | 本轮有没有干成活；0=有进展 |
| `args->done` | 已完成的 sub 个数 |

每个 **sub** 像一条流水线上的计数器：

```text
0  ≤  done  ≤  transmitted  ≤  posted  ≤  nsteps

posted      ：已经「安排」到第几步（send：给 GPU 放了多少空位；recv：发了多少 irecv）
transmitted ：已经交给网络 / 已经让 GPU 看见的步数
done        ：网络和 GPU 都收工、可以忘掉的步数
nsteps      ：这次 op 总共多少步
base        ：本 op 在这条连接上的步号起点（接上一次的 resources->step）
```

货位下标：`buffSlot = (base + 当前步) % NCCL_STEPS`（一般 8 个槽转圈）。

**谁调用：** proxy **进度线程**循环 `progressOps` → `args->progress(args)`，即这两个函数。  
**一次调用通常只推进一步**：有进展就 `idle=0` 并 `continue`/`return`，下次再来，避免饿死别的 op。

---

### 6.2 `sendProxyProgress` 逐步流程

角色：本端 GPU 已（或将要）把数据放进货位，**proxy 负责发出去**。

#### Ready（每个 op 只做一次）

对每个 sub：

```text
base = 把连接上的 step 上对齐到 chunkSteps
resources->step = base + nsteps     // 给下一次 op 接着编号
posted = transmitted = done = 0
若不是用户缓冲注册：sendMhandle = 协议缓冲的 MR
state = Progress
```

#### Progress：对每个还没 `done==nsteps` 的 sub，按顺序尝试三件事

**（1）posted：给 GPU 空货位**

条件：`posted < nsteps` 且 `posted < done + maxDepth`  
`maxDepth = min(8, 16/nsubs)`，防止一次占满所有槽。

- **非 shared**：只把 `posted += sliceSteps`（专用货位地址 GPU 早就知道）。  
- **shared**：  
  1. `sharedBuffersGet` 算出池子里的 **offset**；  
  2. 写入 `connFifo[slot].offset`，fence；  
  3. `posted += sliceSteps`；  
  4. **立刻** `head = base + posted - NCCL_STEPS`（放行 GPU 去写这块）。  
- GDRCopy：写 `gdcSync` 再 `wc_store_fence`。  
- 然后 **`continue`**（本轮这个 sub 先做到这）。

**（2）transmitted：货齐了就 `isend`**

条件：`transmitted < posted`（GPU 已被允许写到这）且未超出 8 槽。

「货齐了」要同时：

- `connFifo[slot].size != -1`（GPU 写下了字节数）  
- `recvMem->tail > 当前步`（GPU 发布了；**LL 协议**可放宽，靠线内 flag）  

数据地址：

| 模式 | `buff` 从哪来 |
|------|----------------|
| 非 shared 普通 | `localBuff + slot * stepSize` |
| shared 普通 | `localBuff + connFifo.offset` |
| shared + 用户注册 | `sendbuff + transmitted * NCCL_MAX_NET_SIZE` |
| 非 shared + 注册 | `ringAlgo->getNextSendAddr`（须和 fifo size 一致） |

还要 **ready**：

- **LL128 + 非 GDR**：扫每一行末尾 flag 是否等于 `step+1`（GPU 只 threadfence，proxy 必须自己确认写完）。  
- **LL**：扫每行 `flag1/flag2`。  
- **SIMPLE + GDR**：一般 `size`+`tail` 就够。

ready 后：

```text
setXferNetAttrs（本次 op 的并发 hint，每轮最多一次）
ncclNet->isend(netSendComm, buff, size, tpRank, mhandle, phandle, &request[slot])
request 非空 → 插件收下了 → transmitted += sliceSteps → continue
request 为空 → 插件暂时吃不下，下次再试（不报错）
```

**（3）done：`test` 完成，还货位**

条件：`done < transmitted`（网上还有没完成的 isend）。

```text
ncclNet->test(request[slot], &done, &size)
若完成:
  connFifo[slot].size = -1
  fence                         // 必须先于 head
  非 shared: head = base + done // 让 GPU 复用该槽
  shared: 不在这里改 head（信用在 posted 已发）
  done += sliceSteps
  若 done==nsteps: args->done++，丢掉 ringAlgo
```

全部 sub 完成 → `args->state = None`，这个 proxy op 从进度队列消失。

#### 发送侧和 GPU 对账（再看一眼）

```text
GPU:  等 head 有空位 → 写 buff → size=N → tail++
Proxy: posted 放空位 → 看见 size/tail → isend → test → size=-1 → head++
```

---

### 6.3 `recvProxyProgress` 逐步流程

角色：proxy **先向网卡收数据**，收齐并保证 GPU 可见后，再让 GPU 读。

比 send 多两步：要 **分组 multi-recv**，GDR 还要 **flush**。

#### Ready：按 `recvComm` 分组

插件一次 `irecv` 可以收多路（`maxRecvs`）。Ready 时把 **同一个 `netRecvComm`** 的 sub 换到一起，设 `groupSize`。

每个 sub 同样设 `base/step`，并清 `posted/received/transmitted/done`，`regBufferReady=0`。

之后循环按 **组** 走（`s += groupSize`）。

#### （1）posted：组好地址，发 `irecv`

组内每个还没 post 满的 sub，在不超过 `maxDepth` 时，填一组数组：

| 模式 | 收数据写到哪 |
|------|----------------|
| SIMPLE 专用 | `localBuff + slot*stepSize`，长度 `stepSize*sliceSteps` |
| SIMPLE 共享 | 池子 offset，写入 `connFifo.offset` |
| SIMPLE + 用户注册 | **先等 kernel 启动**：`connFifo[base%8].size==-1` 才 `regBufferReady`，再直收用户 `recvbuff` |
| LL / LL128 | 协议 FIFO 槽 |

然后对整组：

```text
可选：LL/LL128 且只有 1 路 → OPTIONAL_RECV_COMPLETION（少一次严格 completion）
ncclNet->irecv(recvComm, subCount, ptrs, sizes, tags, mhandles, phandles, &request)
成功（request 非空）→ 组内每个 sub posted += sliceSteps
```

本轮若刚 irecv 成功（`idle==0`），**先 return**，下次再 test。

#### （2）received：`test` 这组 irecv 是否完成

`posted > received` 时 `test(request)`。

完成则：

- 每个 sub：`connFifo[slot].size = -1`，`received += sliceSteps`  
- 若 SIMPLE 且 GDR 且 `needFlush` 且本次真有数据 → **做 flush**  

**Flush 为什么要：** NIC 经 PCIe 写 GPU 是 posted write，CPU 看见 CQE 不代表 GPU 已看见数据。

- 有 `gdcFlush`：x86 上 `mfence` + 读一下 GPU 映射地址，卡住直到写落盘。  
- 否则 `ncclNet->iflush`（常见是一次 RDMA Read 当刷）。  

flush 发起后也 `idle=0` 先 return。

#### （3）transmitted：通知 GPU「可以读了」

`received > transmitted`：若刚才的 flush request 也 test 完（或没有 flush）：

```text
fence
recvTail（或 gdcSync）= base + transmitted
transmitted += sliceSteps
```

GPU 原语看到 tail 前进，才去读货位。

#### （4）done：等 GPU 读完，还网络 credit

GPU 读完会推进 **`sendMem->head`**（接收侧这条 conn 上，head 表示消费进度）。

proxy：

```text
读 sendHead
while (GPU 已消费超过 done) 且 (不超过 transmitted):
  可选 irecvConsumed（告诉插件这组 multi-recv 已被消费）
  done += sliceSteps
  若 done==nsteps: args->done++
```

全部 sub 完成 → `state = None`。

#### 接收侧和 GPU 对账

```text
Proxy: irecv 到货位 → test 到 → flush → 推 tail
GPU:   看见 tail → 读数据 → 推 head
Proxy: 看见 head → done++，必要时 irecvConsumed
```

---

### 6.4 两个 Progress 对照（帮助记忆）

| | **sendProxyProgress** | **recvProxyProgress** |
|--|----------------------|------------------------|
| 谁先动 | GPU 先写 | Proxy 先 irecv |
| 步数名 | posted / transmitted / done | posted / **received** / transmitted / done |
| 网络 API | `isend` → `test` | `irecv` → `test` → 可选 `iflush`/`irecvConsumed` |
| 分组 | 每个 sub 独立 | **同 recvComm 合成一组** multi-recv |
| 放行 GPU | posted（shared）或 done（专用）更新 **head** | transmitted 更新 **tail** |
| 等 GPU | transmitted 等 size+tail | done 等 GPU 的 **head** |
| 用户缓冲 | isend 直接用 sendbuff / ringAlgo | irecv **前必须等 kernel 启动** |

Progress 里还可能调用：`sharedBuffersGet`、`setXferNetAttrs`。

---

## 7. 注册、销毁、例外

| 情况 | 主线程还会进 |
|------|----------------|
| 运行时 **新 peer** send/recv | 再整段 P2pSetup（Setup+Connect） |
| `ncclCommRegister` / Graph 捕获 | `ncclNetLocal/GraphRegisterBuffer` → `netRegisterBuffer` |
| Graph 结束 | `cleanupNet` |
| Destroy/Abort | `sendFree` / `recvFree`（proxy 侧 `*ProxyFree`） |
| Device-net 且不需 proxy | `proxyProgress` 可为 NULL，**稳态甚至不调 Progress** |

`proxySharedInit`：init 里 `ProxyMsgSharedInit`，给 NVB/P2P **先建共享池**，不是 sendSetup 的一部分。

---

## 8. 关键注意事项

1. **先 `connFifo.size=-1`，再改 `head`**，否则 GPU 会踩未完成的槽。  
2. **两端 useGdr 必须协商**；任一侧不行就清 `DIRECT_NIC`。  
3. **PXN 只用于 send**；recv 的 proxy 必须是本 rank。  
4. **有 topo `graph` 的边用专用缓冲**；网络 P2P 默认 shared 池。  
5. **Setup 在 bootstrap 之后**，不是 bootstrapInit。  
6. **热路径不在主线程的 net.cc**，在 Progress + 插件。  
7. 用户缓冲直发需 **GDR + Register**；多物理段无 DMA-BUF 会注册失败。  
8. shared 模式 **head 初值 `-NCCL_STEPS`**，等 posted 再放行 GPU。

---

## 9. 重要数据结构表

| 名字 | 活在哪 | 含义 |
|------|--------|------|
| **`ncclConnector` / `conn`** | 主线程 + 拷到 GPU | kernel 只认：`buffs/head/tail/connFifo/flags` |
| **`ncclProxyConnector`** | 主线程 | 到某个 proxy 的 RPC 句柄、`connection*` cookie |
| **`setupReq`** | Setup RPC 载荷 | shared、netDev、GDR、channel、rank、sameDevice |
| **`sendNetResources`** | **Proxy** | 发送侧：map、netSendComm、缓冲、mhandle、step |
| **`recvNetResources`** | **Proxy** | 接收侧：另有 listen/recvComm、needFlush、gdcFlush |
| **`connectMap`** | Proxy 填、主线程 import | 各「银行」内存布局；跨进程靠这份描述对齐指针 |
| **银行 HOST/DEV/SHARED/GDC** | map 里 | 控制结构 / 专用或共享数据 / GDRCopy head |
| **`ncclSendMem` / `ncclRecvMem`** | HOST（或 SHM） | head、tail、`connFifo[8]` |
| **`ncclConnFifo`** | 上者之内 | `size`（-1=空）、`offset`（shared 池内位移） |
| **`ncclNetHandle_t`** | recv listen 产出 | 对端 send 用来 `connect` |
| **`ncclNetAttr_t`** | Connect/Progress | 告诉插件并发 peer/流、op/algo |
| **`netRegInfo`** | Register RPC | 用户缓冲地址、大小、物理段数 |
| **`netTransport`** | 全局 | 把上面函数挂进传输表 |

环形货位：`NCCL_STEPS=8`。共享池深度：`NCCL_SHARED_STEPS=16`。

---

## 10. 全部函数表（含义 + 何时调用）

### 10.1 主线程会直接进入

| 函数 | 何时（主线程） | 完成什么 |
|------|----------------|----------|
| `canConnect` | P2pSetup 选传输 | 这对 peer 能否走 NET |
| `sendSetup` | 选中 NET 发送边 | 决策 + **RPC Setup** + 写 connectInfo |
| `recvSetup` | 选中 NET 接收边 | 决策 + **RPC Setup（listen）** |
| `populateCommNetAttrs` | Connect 里 | 填插件 hint |
| `sendConnect` | bootstrap 交换后 | **RPC Connect** + 填 GPU send conn |
| `recvConnect` | 同上 | **RPC Connect** + 填 GPU recv conn |
| `netMapShm` | sendConnect 跨进程 | 导入 HOST 映射 |
| `ncclNetLocalRegisterBuffer` | 用户 Register | 入口 → `netRegisterBuffer` |
| `ncclNetGraphRegisterBuffer` | Graph 捕获 | 注册 + cleanup 入队 |
| `netRegisterBuffer` | 上两者内部 | 对每个 NET peer **RPC Reg** |
| `ncclNetDeregBuffer` | 显式注销 | **RPC Dereg** |
| `cleanupNet` | Graph 结束回调 | Graph deregister |
| `sendFree` / `recvFree` | Destroy | 释放主线程侧 map/IPC |

### 10.2 仅 Proxy（RPC 或进度）

| 函数 | 何时 | 完成什么 |
|------|------|----------|
| `sendProxySetup` | RPC Setup | calloc 发送资源 + getProperties |
| `recvProxySetup` | RPC Setup | 同上 + **listen** |
| `sendProxyConnect` | RPC Connect | **connect** + 分配货位 + regMr |
| `recvProxyConnect` | RPC Connect | **accept** + 货位 + regMr；关 listen |
| `proxySharedInit` | SharedInit RPC | 预建 P2P 共享池 |
| `sharedNetBuffersInit/Get/Destroy` | Connect/Progress/Free | 共享池生命周期与 offset |
| `sendProxyProgress` | **稳态每次发送切片** | credit → isend → test |
| `recvProxyProgress` | **稳态每次接收切片** | irecv → test → flush |
| `setNetAttrs` / `setXferNetAttrs` | Connect/Progress | 把 hint 给插件 |
| `printNetAttrs` | 调试 TRACE | 打印 hint |
| `ncclNetGetDeviceHandle` | Connect | 是否要 GPU 侧 net handle |
| `getHandleForAddressRangeFlags` | Connect/Reg | DMA-BUF PCIe 标志 |
| `send/recvProxyRegBuffer` | RPC Register | 用户缓冲 regMr |
| `send/recvProxyDeregBuffer` | RPC Deregister | deregMr |
| `send/recvProxyFree` | Destroy | 关插件连接、释放缓冲 |
| `netCreateShm` | Proxy Connect 跨进程 | 创建 HOST SHM |
| `netDumpMap` / `netHandleCmp` / `ncclTopoFlushTypeStr` | 调试/队列比较/日志 | 辅助 |

vtable：`netTransport` 把 setup/connect/free/progress/reg 填进 `TRANSPORT_NET`。

---

## 11. 调用关系次数梳理

「次数」按 **一条 NET 边 / 一次 comm 生命** 理解（集体边在 init 建；P2P 边可能延后）。

| 事件 | 主线程 net.cc | Proxy net.cc | 大约次数 |
|------|---------------|--------------|----------|
| comm init，每条跨节点 **集体** NET 边 | 1× canConnect + 1× Setup + 1× Connect | 1× ProxySetup + 1× ProxyConnect | **每边各一次** |
| 同上，bootstrap 交换 | （不在 net.cc） | — | 每 peer 一轮 Send/Recv |
| NVB / runtime 新 P2P 边 | 再 1 套 Setup+Connect | 再 1 套 ProxySetup+Connect | **每个新 peer/channel 一次** |
| SharedInit | — | 0～1× `proxySharedInit` | 每 comm / PXN proxy 少量 |
| 稳态 AllReduce **一步**（边已在） | **0** | 每个切片多次 Progress | **热路径，极频繁** |
| 一次 Progress 推进 | — | 可能 isend/irecv/test 各 1；shared 则 Get | 每切片 |
| Register 每块用户缓冲 | 1× Local/GraphRegister | 每 peer 1× ProxyReg | 按注册次数 |
| Destroy | 每边 1× Free | 每边 1× ProxyFree | **每边一次** |

```text
次数直觉:

  canConnect / Setup / Connect / Free     =  O(边数)     冷
  Progress                                 =  O(步数×切片) 热
  Register                                 =  O(用户注册)  偶发
```

**完整一次「建边 → 干活 → 拆边」：**

```text
主: canConnect
主: sendSetup ──RPC──► proxy: sendProxySetup
主: recvSetup ──RPC──► proxy: recvProxySetup (listen)
主: bootstrap 对调 connectInfo
主: sendConnect ──RPC──► proxy: sendProxyConnect (connect+缓冲)
主: recvConnect ──RPC──► proxy: recvProxyConnect (accept+缓冲)
        [cudaMemcpy conn → GPU]
──────── 稳态 ────────
GPU kernel  ⇄  proxy: send/recvProxyProgress ⇄ 插件
──────── 结束 ────────
主: sendFree / recvFree
proxy: sendProxyFree / recvProxyFree
```

---

读完若只记四句：

1. **`net.cc` = GPU 与网卡插件之间的调度层。**  
2. **建连在主线程 + RPC；干活在 Progress。**  
3. **Setup 不是连网卡；Connect 才是。**  
4. **稳态主线程几乎不再进这个文件。**
