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

对每个要对齐的 peer、每个置位的 channel：

### ① 选传输 + Setup

```text
selectTransport
  按 P2P → SHM → NET → COLLNET 顺序 canConnect
  同机常被 P2P/SHM 抢走，走不到 NET
  跨节点: canConnect 成功
    → sendSetup 或 recvSetup
```

**`canConnect`**：默认允许 NET；同机再问拓扑是否禁用节点内 NET。

**`sendSetup`（主线程自己）**

1. 定 `shared`（见术语）。  
2. `ncclTopoGetNetDev`：网卡 + **谁当 proxy**（PXN 可能是旁边 GPU）。  
3. GDR？置 `NCCL_DIRECT_NIC`。  
4. `ncclProxyConnect`：连上该 proxy（内部先 **RPC Init**，登记一条 NET 发送 connection）。  
5. **`ncclProxyCallBlocking(ncclProxyMsgSetup, setupReq)`** → proxy 上 **`sendProxySetup`**。  
6. 写 `connectInfo`：proxyRank + useGdr，供对端。

**`sendProxySetup`（proxy，被 RPC 调用）**  
calloc `sendNetResources`，抄 req，`getProperties`（DMA-BUF、maxRecvs…）。  
**不 listen、不 connect 网卡、不分配数据大缓冲。**

**`recvSetup` / `recvProxySetup`**  
对称；proxy 上 **`listen`**，响应带回 **`ncclNetHandle_t`**。Recv **不做 PXN**（proxy 必须是本 rank）。

### ② Bootstrap 交换

主线程 **不进** net.cc，用 bootstrap 把双方 `connectInfo` 对调。  
发送侧给出 proxyRank+useGdr；接收侧给出 listen handle。

### ③ Connect（可异步多圈 poll）

```text
transportComm->connect == sendConnect / recvConnect
  内部: populateCommNetAttrs
        ProxyCallAsync(Connect)
        Poll 直到成功
        必要时 netMapShm / import IPC
        填 conn.head/tail/buffs
        挂 proxyProgress
成功则 cudaMemcpy conn → GPU
```

**`sendProxyConnect` / `recvProxyConnect`（proxy）**  
插件 `connect`/`accept`；按 shared 分配货位；`regMr`；把 `connectMap` 传回主线程。  
`netSendComm==NULL` 时返回 InProgress，主线程再 poll。

**`populateCommNetAttrs`**：按集体/网络 P2P 填插件并发 hint。  
**`netMapShm`**：PXN 时主线程导入 proxy 的 HOST 映射。

### ④ 再一条 bootstrap 栅栏，清 mask

然后这条 NET 边对主线程来说 **建完了**。

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
