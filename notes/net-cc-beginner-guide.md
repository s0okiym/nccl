# NCCL `net.cc` 小白详解：它在干什么、为何这样设计、shared 是什么

> **面向读者**：不熟悉 NCCL 内部的读者；先建立直觉，再对照代码。  
> **对应源码**：`src/transport/net.cc`  
> **生命周期与函数时序（推荐）**：[`nccl_transport_net_cc_lifecycle.md`](nccl_transport_net_cc_lifecycle.md)  
> **进阶文档**：[`net-cc-analysis.md`](net-cc-analysis.md)（偏源码结构）  
> **相关**：`proxy-internals.md`、`ib-isend-analysis.md`

---

## 目录

0. [术语先钉死：P2P 与 graph 是什么](#0-术语先钉死p2p-与-graph-是什么)
1. [先忘掉代码：问题是什么](#1-先忘掉代码问题是什么)
2. [net.cc 一句话 + 一张总图](#2-netcc-一句话--一张总图)
3. [设计是如何一步步长出来的](#3-设计是如何一步步长出来的)
4. [核心协同：GPU 写、CPU proxy 发网](#4-核心协同gpu-写cpu-proxy-发网)
5. [shared 与非 shared：到底共享什么](#5-shared-与非-shared到底共享什么)
6. [两种 shared 别搞混：缓冲 vs 连接](#6-两种-shared-别搞混缓冲-vs-连接)
7. [何时用 shared、何时用非 shared](#7-何时用-shared何时用非-shared)
8. [对照代码：shared 如何写进 net.cc](#8-对照代码shared-如何写进-netcc)
9. [把整文件功能串起来](#9-把整文件功能串起来)
10. [常见疑问](#10-常见疑问)
11. [读完后建议的学习路径](#11-读完后建议的学习路径)

---

## 0. 术语先钉死：P2P 与 graph 是什么

本文和 `net.cc` 里两个词最容易误解，先对齐含义。

### 0.1 本文说的「P2P」是哪种？

**是：走 NET 的 point-to-point（网络点对点）**——例如 `ncclSend` / `ncclRecv`，以及 AlltoAll 等基于 send/recv 的模式在**网卡路径**上的连接。

**不是：本机 GPU↔GPU 的 NVLink/PCIe peer access。**

| 说法 | 含义 | 典型代码 | 和 net.cc shared 的关系 |
|------|------|----------|-------------------------|
| **网络 P2P**（本文默认） | 跨节点（或走网）的 send/recv 边 | `TRANSPORT_NET`，常 `connIndex ≠ 0` | **shared 默认用在这类边上** |
| **本机 GPU P2P** | 同机 GPU 直接访存 | `TRANSPORT_P2P`（`p2p.cc`），NVLink/PCIe | **不归 net.cc 的 shared 管** |

> 一句话：beginner 文档里的 P2P = **NET 上的 point-to-point**，不是本机 GPU-to-GPU。

### 0.2 「有 graph」指的是拓扑算出来的 graph 吗？

**是的。** 这里的 `graph` 是 **`struct ncclTopoGraph*`**——初始化阶段由 **拓扑搜索** 得到的集体通信图（Ring / Tree / NVLS / CollNet…），**不是** CUDA Graph。

来源关系：

```text
拓扑探测 (topo)
  → 图搜索 (search/rings/trees…)
  → 得到 ncclTopoGraph（如 graphs[NCCL_ALGO_RING]）
  → ncclTransportP2pSetup(comm, graph, connIndex)
  → selectTransport(..., graph, ...)
  → net 的 sendSetup/recvSetup(comm, graph, ...)
```

在 `sendSetup` / `recvSetup` 里：

```c
shared = graph || connIndex == 0 ? 0 : /* 默认 P2P 用 shared */ 1;
```

| `graph` 指针 | 含义 | shared 倾向 |
|--------------|------|-------------|
| **非 NULL** | 正在为**某集体算法拓扑边**建 NET 连接（Ring/Tree 等跨节点边） | **强制非 shared**（专用缓冲） |
| **NULL** | 不是“带着 topo graph 建 collective 边”，常见于 **send/recv 类网络 P2P 连接** | 再看 `connIndex` 等，**默认可 shared** |

补充：

- **`connIndex == 0`**：连接槽 0，集体路径常用主连接，也倾向 **非 shared**。  
- **CUDA Graph**（捕获 kernel 图）是另一回事，和这里的 `ncclTopoGraph* graph` **无关**。

> 一句话：代码里的 `graph` = **拓扑/算法算出来的 `ncclTopoGraph`**，有它 ⇒ 这条 NET 边是集体拓扑的一部分 ⇒ 不用 shared 缓冲池。

---

## 1. 先忘掉代码：问题是什么

分布式训练里，GPU 之间经常要做 **AllReduce** 等操作。  
同一台机器上的 GPU 可以走 **NVLink**；跨机器时，数据必须走 **网卡（IB/RoCE/以太网）**。

于是出现三个角色：

| 角色 | 谁 | 能做什么 | 不能轻松做什么 |
|------|-----|----------|----------------|
| **GPU** | CUDA kernel | 极快地搬 GPU 内存、做 reduce | 直接调用网卡驱动（`ibv_post_send` 等） |
| **CPU 上的 proxy 线程** | NCCL 为每 GPU 起的后台线程 | 调用网卡插件 API、轮询完成 | 代替 GPU 做大规模计算 |
| **网卡插件** | `net_ib` / socket 等 | 真正发收网络包 | 不知道 NCCL 的 ring/tree 算法细节 |

**矛盾：**

- 算法跑在 **GPU** 上；
- 发包 API 在 **CPU/网卡库** 里；
- 两边要 **流水线并行**：GPU 准备下一块时，上一块已经在网上飞。

`net.cc` 就是为了解决这个矛盾而写的 **“适配层 + 流水线调度”**。

---

## 2. net.cc 一句话 + 一张总图

### 2.1 一句话

> **`net.cc` = NCCL 的「跨节点点对点网络运输公司」的总部规章：规定谁建连接、用哪块中转货场、如何用 head/tail 发货收货、何时通知网卡发车。真正开车的是网卡插件（如 `net_ib`）。**

### 2.2 总图

```text
        用户程序: ncclAllReduce(...)
                    │
                    ▼
        NCCL 算法层 (Ring/Tree...) 在 GPU 上跑
                    │
        需要跨节点时，选中 NET 传输
                    │
        ┌───────────┴───────────┐
        │      net.cc 管这些     │
        │  · 建网连接 (setup/connect)
        │  · 准备中转缓冲        │
        │  · GPU↔Proxy 信用协议  │
        │  · 调插件 isend/irecv  │
        └───────────┬───────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   GPU kernel              Proxy 线程
   写/读中转区              调网卡插件
         │                     │
         └──────────┬──────────┘
                    ▼
              物理网络 (RoCE/IB/…)
```

**它不负责：** 拓扑选路细节（主要在 `graph/`）、IB QP 细节（在 `net_ib/`）、集合算法数学（在 `device/`）。

---

## 3. 设计是如何一步步长出来的

下面用「如果只有你一个人设计」的思路，解释为何最终长成现在这样。

### 第 0 步：最笨但能用的方案

```text
GPU 算完一整块 → 拷到 host → 主线程同步 isend → 等完成 → 再算下一块
```

问题：GPU 和网卡 **串行**，极慢。  
→ 需要 **流水线**。

### 第 1 步：多块缓冲 + 异步发包

准备 **N 块中转缓冲**（NCCL 里 `NCCL_STEPS`，一般是 **8**）：

```text
槽0 槽1 槽2 ... 槽7  围成一圈（环形）

GPU 写槽0 → proxy 发槽0 → GPU 写槽1 → proxy 发槽1 → ...
GPU 写槽0（要等槽0 发完才能复用）
```

这就是 **depth=8 的流水线**。  
→ 代码里大量 `buffSlot = (base + step) % NCCL_STEPS`。

### 第 2 步：GPU 和 proxy 如何同步？

需要两个计数器（类似“还有多少货没发完”）：

| 名字 | 谁写 | 含义（直觉） |
|------|------|----------------|
| **tail** | GPU 写 | “我已经准备好到第几步” |
| **head** | proxy 写 | “网络已经处理完到第几步” |
| **connFifo[slot].size** | GPU 写 size，proxy 清回 -1 | “这一槽有多少字节 / 是否空闲” |

```text
空闲槽数量 ≈ NCCL_STEPS - (tail - head)

GPU:  有空位就写 → 写 size → 增加 tail
Proxy: 看到 size 有效就 isend → 完成 → size=-1 → 增加 head
```

→ 这就是 `sendMem` / `recvMem` / `connFifo` 存在的原因。  
→ 代码里 **必须先 `size=-1` 再改 `head`**，否则 GPU 会踩到还在用的槽。

### 第 3 步：发包不能在 GPU 上做 → 独立 proxy 线程

每个 GPU 一个 **proxy 线程** 死循环：

```text
while (有活干):
  看看哪些槽数据好了
  isend / irecv
  test 是否完成
  更新 head，放行 GPU
```

→ `sendProxyProgress` / `recvProxyProgress`。  
→ 主线程只负责 **建连、把进度函数挂上去**。

### 第 4 步：真正的网卡代码不该写死在 NCCL 里

NCCL 支持 IB、Socket、第三方插件 → 抽象成 **`ncclNet` 接口**：

```text
listen / connect / isend / irecv / test / regMr / close...
```

→ `net.cc` **只调用接口**，不写 `ibv_post_send`。  
→ `net_ib` 等是“司机”；`net.cc` 是“调度室规章”。

### 第 5 步：建连要分“主线程”和“proxy 进程/线程”

因为：

- 选网卡、交换 handle 在 **初始化主路径**；
- `listen/connect/regMr` 最好在 **真正跑进度的 proxy 上下文**（且可能在 **另一个 rank 的进程** 上，见 PXN）。

→ 拆成：

| 阶段 | 谁 | 干什么 |
|------|-----|--------|
| Setup | 主线程 + Proxy RPC | 选网卡、listen 出 handle |
| Connect | 主线程 + Proxy RPC | connect、分配缓冲、regMr |
| Progress | 仅 Proxy | 收发数据 |
| Free | 两边 | 释放 |

→ `sendSetup` / `sendProxySetup` / `sendConnect` / `sendProxyConnect` / `sendProxyProgress` 这一长串名字就来自 **“主线程规章 + proxy 执行”** 的拆分。

### 第 6 步：集体通信 vs 点对点，缓冲策略不同

- **Ring AllReduce**：每条跨节点边 **长期、大流量** → 每条连接 **自备一整套大缓冲**（**非 shared**）。  
- **大量 P2P send/recv**：边极多、突发 → 若每条边都自备大缓冲，**显存爆炸** → **多条边共用一个缓冲池**（**shared**）。

→ 这就是本文重点：**shared / 非 shared** 的设计动机。

### 第 7 步：GDR、注册用户缓冲、PXN…

在 1–6 能跑之后，再加优化：

- **GDR**：网卡直接 DMA 到 GPU 内存，少一次拷贝；  
- **用户缓冲注册**：大消息甚至不进中转区；  
- **PXN**：用更靠近网卡的 GPU 的 proxy 出网。  

这些都是“主干设计上的枝叶”，仍挂在同一套 head/tail 流水线上。

---

## 4. 核心协同：GPU 写、CPU proxy 发网

用寄快递打比方：

```text
GPU = 打包车间
中转缓冲 = 月台货位（只有 8 个货位）
Proxy = 快递员
网卡插件 = 货车公司
head/tail = 货位使用登记本
```

### 发送侧（本机 A → 远端 B）

```text
1. GPU 看登记本：有空货位吗？(tail - head < 8)
2. GPU 把数据搬到货位 k，写下“本货位 N 字节”(size=N)
3. GPU 更新 tail：“我装到第 tail 票了”
4. Proxy 看到货位 k 有货且 size 有效
5. Proxy 叫货车公司 isend(货位地址, N)
6. 货车运走、完成 test
7. Proxy 把 size 清成 -1，再更新 head：“货位 k 空了”
8. GPU 以后可以再用货位 k
```

### 接收侧（远端 → 本机）

```text
1. Proxy 先向网卡 irecv 到某个货位（或用户缓冲）
2. 数据到了、必要时 flush 让 GPU 看见
3. 通知 GPU：货位可读
4. GPU 读走数据，归还信用
```

**`net.cc` 的 progress 函数，就是把上面 4–7 步写成状态机。**

---

## 5. shared 与非 shared：到底共享什么

这里的 **shared 只指：多条 NET 连接是否共用同一块「数据中转池」**。  
（不是“共享内存 SHM 传输”，也不是“多 rank 共享一个进程”。）

### 5.1 非 shared（专用缓冲）——每人一个仓库

```text
连接 A→B 的 channel0:
  自己的 8 个大货位 [################]  例如几 MB 级 / 槽

连接 A→C 的 channel0:
  又一套 8 个大货位 [################]

连接 A→D ...
  再来一套 ...
```

**特点：**

| 优点 | 缺点 |
|------|------|
| 每条边互不抢货位，吞吐稳 | 边一多，内存/显存线性涨 |
| 实现简单：槽号 = step % 8 | 不适合“很多 peer 偶尔发一下” |
| 适合 Ring/Tree 长期打满 | |

**直觉：** 高速公路专用车道——贵，但一路畅通。

### 5.2 shared（共享缓冲池）——大家共用一个停车场

```text
同一个本地 rank 的 proxy 上，有一块大池子:

  池子大小 ≈ nChannels × NCCL_SHARED_STEPS × p2pChunkSize
            （约 16 个“时间槽” × 通道数 × 每块 chunk）

多条 P2P 连接不再各建一套 8 大槽，而是:
  每次要发时，从池子里领一块 → 记下 offset
  发完归还（通过 head/done 与流水线深度限制）
```

**特点：**

| 优点 | 缺点 |
|------|------|
| 边再多，缓冲总量可控 | 多条流 **抢同一池子** |
| 适合 send/recv、AlltoAll 等边多场景 | 需要 `offset` 告诉 GPU “用池子哪一段” |
| | 流水线深度受 `SHARED_STEPS/nsubs` 限制 |

**直觉：** 共享停车位——省地，忙时可能要等空位。

### 5.3 一张对比图

```text
【非 shared】每条连接
  Conn1: [s0][s1][s2][s3][s4][s5][s6][s7]   ← 独占
  Conn2: [s0][s1][s2][s3][s4][s5][s6][s7]   ← 独占
  Conn3: [s0][s1]...

【shared】一个池 + 多条连接租用
  大池: [c0-slot0][c0-slot1]...[c1-slot0]...
           ▲              ▲
           │              │
         Conn1 用这段   Conn2 用那段
         (记在 connFifo.offset 里)
```

### 5.4 控制协议有何不同

**非 shared：**

```text
GPU 知道: 数据总在「本连接的 buff + slot * stepSize」
Proxy 还 credit: 主要更新 head
connFifo[slot].size = 本次字节数
```

**shared：**

```text
GPU 不能假设固定地址
Proxy 在 posted 阶段就:
  1) 从池子算出 offset
  2) 写入 connFifo[slot].offset
  3) 提前推进 head（放行 GPU 去写这个 offset）
GPU 写到 (共享池基址 + offset)
Proxy isend 时同样用 offset 取地址
```

代码注释也写了：共享模式 **一开始 head = -NCCL_STEPS**，表示“先别写，等 proxy 通过 posted 放 credit”。

### 5.5 用数字感受内存差异

假设：

- 每槽 512KB，`NCCL_STEPS=8` → 每连接每方向约 **4MB** 级 SIMPLE 缓冲（量级示意）  
- 某 rank 有 **64 条** P2P NET 边  

| 模式 | 粗算缓冲 |
|------|----------|
| 非 shared | 64 × 4MB ≈ **256MB** 仅一侧协议缓冲量级 |
| shared | 与 channel、SHARED_STEPS、chunk 相关，**不随 64 线性涨到同量级** |

这就是为什么 P2P 默认倾向 shared，而 Ring/Tree 用非 shared。

---

## 6. 两种 shared 别搞混：缓冲 vs 连接

`net.cc` 里有 **两个不同的“共享”开关**，名字都像 shared，含义完全不同。

### 6.1 Shared **Buffers**（`resources->shared` / `NCCL_NET_SHARED_BUFFERS`）

- **共享的是：中转数据内存池**  
- 上文 §5 讲的就是这个  
- 日志里常见：`/Shared`

### 6.2 Shared **Comms**（`NCCL_NET_SHARED_COMMS`，默认 1）

- **共享的是：网卡上的“连接对象” `sendComm`/`recvComm`**  
- 场景：插件支持 **一次 irecv 收多路**（`maxRecvs > 1`）  
- 同一块网卡、同一远端 rank 的多条逻辑流，**复用一个底层连接**，减少 QP/连接数  

```text
缓冲 shared:  多条逻辑连接 → 共用内存池
连接 shared:  多条逻辑流   → 共用插件 connect 句柄
```

可以：

- 缓冲 shared + 连接不 shared  
- 缓冲不 shared + 连接 shared（少见组合取决于配置）  
- 两者都开（P2P + multi-recv 插件时常见）

---

## 7. 何时用 shared、何时用非 shared

代码判定（`sendSetup` / `recvSetup`）：

```c
shared = graph || connIndex == 0 ? 0
       : (NCCL_NET_SHARED_BUFFERS 若用户设了就用用户值)
       : 1;   // 默认：网络 P2P 用 shared
```

翻译成人话：

| 条件 | shared | 原因 |
|------|--------|------|
| **`graph != NULL`**（带着**拓扑算出的** `ncclTopoGraph` 建边，如 Ring/Tree 跨节点边） | **0 非 shared** | 集体拓扑边：边相对少、流量大、要专用通道 |
| **`connIndex == 0`**（主连接槽，集体常用） | **0** | 同上 |
| **网络 P2P**（`graph == NULL` 且 `connIndex != 0`，send/recv 类） | **默认 1 shared** | 边可能很多，省内存 |
| 用户设 `NCCL_NET_SHARED_BUFFERS=0/1` | 强制 | 调优/调试 |

```text
AllReduce 跨节点 ring 边（有 topo graph）  →  非 shared（专用大缓冲）
ncclSend/Recv 很多 peer（网络点对点）     →  shared（池化）
本机 NVLink GPU peer（TRANSPORT_P2P）      →  不走 net.cc 这套 shared 逻辑
```

再次强调：

- 这里的 **P2P = 网络 point-to-point**，不是本机 GPU P2P。  
- 这里的 **graph = `ncclTopoGraph`（拓扑/算法图）**，不是 CUDA Graph。

---

## 8. 对照代码：shared 如何写进 net.cc

### 8.1 建立连接时谁决定

`sendSetup` 算出 `req.shared` → 经 Proxy Setup 存进 `resources->shared`。

### 8.2 Connect 时如何分配内存

**非 shared（`sendProxyConnect`）：**

```text
for 每个协议 LL/LL128/SIMPLE:
  给「这条连接」单独 ADD_POINTER 一块 buff
```

**shared：**

```text
sharedNetBuffersInit(...)  // 池子只建一次，refcount++
把池子登记为 SHARED_DEVMEM 或 SHARED_HOSTMEM
buffs[SIMPLE] 指向整池（用时再算 offset）
```

池大小：

```text
size = nChannels * NCCL_SHARED_STEPS * p2pChunkSize
// NCCL_SHARED_STEPS = 16
```

取块：

```text
sharedBuffersGet(channel, slot, &offset)
offset = p2pChunkSize * (channel * 16 + slot)
```

### 8.3 Progress 时行为差异（发送）

| 阶段 | 非 shared | shared |
|------|-----------|--------|
| posted | 只加 `posted` 计数 | 分配 offset 写入 fifo，并 **提前** 更新 head |
| isend 地址 | `localBuff + slot*stepSize` | `localBuff + connFifo.offset`（或 reg 用户地址） |
| done 后 head | 更新 head 还专用槽 | 共享模式下 done 路径对 head 处理不同（信用在 posted 已部分发放） |

### 8.4 日志里怎么认

```text
via NET/IB/0/GDRDMA          ← 无 /Shared → 多半非 shared
via NET/IB/0/GDRDMA/Shared   ← shared 缓冲
```

---

## 9. 把整文件功能串起来

按「你要完成跨节点发数据」需要的能力，对应 `net.cc` 里的模块：

### 能力 1：判断能不能走网

- `canConnect`：同机是否允许 NET  

### 能力 2：建一条逻辑链路

- Setup：选哪块网卡、要不要 GDR、谁当 proxy（PXN）  
- Listen/Connect：插件建真实连接  
- 交换 handle  

### 能力 3：准备“货位”

- 非 shared：每连接自建  
- shared：进池子  
- 控制结构 sendMem/recvMem 始终要有  

### 能力 4：让 GPU 看见货位地址

- `connectMap` 描述布局  
- 主线程 import 后填 `conn.buffs/head/tail/connFifo`  
- kernel 只认 `conn`，不认 proxy  

### 能力 5：运行时发车

- `sendProxyProgress`：posted → isend → test → 还 credit  
- `recvProxyProgress`：irecv → test → flush → 通知 GPU  

### 能力 6：可选加速

- GDR / DMA-BUF / 用户缓冲注册 / netAttr 提示插件  

### 能力 7：收工

- Free、deregMr、减共享池引用  

**整文件 ≈ 能力 1–7 的实现清单**；不是杂乱函数堆，而是 **建连 → 准备缓冲 → 流水线 progress → 释放** 的故事。

---

## 10. 常见疑问

### Q1：shared 是不是“用 SHM 传输”？

**不是。**  
SHM 是另一种 transport。这里 shared 是 **NET 路径上的缓冲池共享**。跨进程时池子可能用 CUDA IPC/SHM **映射**，但传输仍是 NET 插件。

### Q2：非 shared 是不是更快？

**集体 Ring/Tree（有 topo graph 的 NET 边）通常用非 shared，因为更稳、不抢池。**  
**网络** P2P 边极多时 shared 往往 **整体更优**（省内存、缓存更友好）。不是绝对谁更快，是 **场景不同**。

### Q3：没有 shared 能不能跑 AlltoAll？

能，但每条边一套缓冲可能 **OOM 或占满显存**。所以网络 P2P 默认 shared。

### Q4：和 `net_ib` 什么关系？

```text
net.cc:  “何时 isend、缓冲区在哪、head/tail 怎么走”
net_ib:  “isend 内部如何 ibv_post_send、QP、CTS”
```

你之前看的 `ibv_post_send` 在 **插件里**；`net.cc` 在它的上一层。

### Q5：PXN 和 shared 有关吗？

正交。PXN = **换一个 rank 的 proxy 出网**；shared = **缓冲是否池化**。  
但 PXN 跨进程时，数据缓冲映射更复杂（代码禁止某些 host 缓冲组合）。

### Q6：为什么要 8 个槽？

在 **延迟** 和 **占内存** 之间折中：太少流水线填不满网；太多浪费内存。8 是 NCCL 全局常数 `NCCL_STEPS`。

### Q7：文档里的 P2P 是本机 GPU peer 还是网络点对点？

**是网络点对点（NET 上的 send/recv 边），不是本机 GPU-to-GPU。**  
本机 NVLink/PCIe peer 属于 `TRANSPORT_P2P`（`p2p.cc`），与 net.cc 的 shared 缓冲逻辑无关。详见 [§0.1](#01-本文说的p2p是哪种)。

### Q8：`graph != NULL` 是不是拓扑算出来的 graph？是 CUDA Graph 吗？

**是拓扑/算法图 `ncclTopoGraph*`（Ring/Tree/… 搜索结果），不是 CUDA Graph。**  
初始化时 `ncclTransportP2pSetup(comm, graph, …)` 把该指针传进 `sendSetup`；有 graph 的 NET 边按**集体拓扑边**处理，强制非 shared。详见 [§0.2](#02-有-graph指的是拓扑算出来的-graph吗)。

---

## 11. 读完后建议的学习路径

1. **本文**建立直觉：流水线、head/tail、shared/非 shared  
2. 看一次发送日志：有没有 `/Shared`  
3. 读 `sendProxyProgress` 三个 `if`（posted / transmitted / done），对照 §4 快递比喻  
4. 再读 [`net-cc-analysis.md`](net-cc-analysis.md) 的函数表  
5. 若关心 IB：再读 `ib-isend-analysis.md`  

---

## 附录：一句话速记卡

| 概念 | 一句话 |
|------|--------|
| `net.cc` | GPU 与网卡插件之间的调度与缓冲管理层 |
| 非 shared | 每条 NET 连接自备中转货位（集体 topo graph 边常用） |
| shared buffers | 多条**网络 P2P** 连接租用同一货位池，用 offset 区分 |
| shared comms | 多条流复用同一插件连接句柄 |
| 本文 P2P | **NET 点对点**，不是本机 GPU peer |
| graph 参数 | **`ncclTopoGraph`（拓扑算法图）**，不是 CUDA Graph |
| head/tail | proxy 与 GPU 之间的环形信用账本 |
| proxy progress | 快递员：看见货就 isend，到了就还货位 |
| 插件 | 真正开货车的人 |

---

## 修订记录

| 日期 | 内容 |
|------|------|
| 2026-07-10 | 初稿：面向小白的 net.cc 功能与设计演化；详细解释 shared/非 shared 与 shared buffers/comms 区别 |
| 2026-07-10 | 增补 §0：明确 P2P=网络点对点（非本机 GPU peer）；graph=`ncclTopoGraph`（非 CUDA Graph）；FAQ Q7/Q8 |
