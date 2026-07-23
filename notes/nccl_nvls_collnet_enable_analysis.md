# NCCL `NCCL_NVLS_ENABLE` 与 `NCCL_COLLNET_ENABLE` 深度分析

> **状态**：基于 NCCL 源码与 HGX 类拓扑的归纳  
> **目标机型**：8× NVIDIA H200（NV18）+ 8× 400G RoCE（mlx5 / eth400g）+ 双 NUMA  
> **相关代码**：`src/init.cc`、`src/transport/nvls.cc`、`src/transport/coll_net.cc`、`src/graph/tuning.cc`、`src/graph/search.cc`、`src/graph/connect.cc`、`src/device/all_reduce.h`、`src/include/coll_net.h`  
> **相关笔记**：`nccl_params.md`、`nccl_thread_model.md`、`ibv_post_send_ms_stall_analysis.md`、`allreduce-channel-c8-pair-slow-analysis.md`

---

## 目录

1. [执行摘要](#1-执行摘要)
2. [目标机器拓扑](#2-目标机器拓扑)
3. [两个开关分别管什么](#3-两个开关分别管什么)
4. [`NCCL_NVLS_ENABLE` 源码路径](#4-nccl_nvls_enable-源码路径)
5. [`NCCL_COLLNET_ENABLE` 源码路径](#5-nccl_collnet_enable-源码路径)
6. [算法矩阵：何时启用 / 禁用](#6-算法矩阵何时启用--禁用)
7. [在本机拓扑上的 1 vs 0 本质差异](#7-在本机拓扑上的-1-vs-0-本质差异)
8. [组合矩阵（单机 / 多机 / 有无插件）](#8-组合矩阵单机--多机--有无插件)
9. [数据路径对照](#9-数据路径对照)
10. [与 `ibv_post_send` / 训练性能的关系](#10-与-ibv_post_send--训练性能的关系)
11. [如何用日志验证是否生效](#11-如何用日志验证是否生效)
12. [**节点间 CollNet：原理、机制与实现流程**](#12-节点间-collnet原理机制与实现流程)（含 [12.7 插件 + 交换机硬件](#127-打开-collnet除了插件还需要交换机硬件吗)）
13. [**节点内 NVLS AllReduce：原理、机制与实现流程**](#13-节点内-nvls-allreduce原理机制与实现流程)
14. [推荐配置与排查清单](#14-推荐配置与排查清单)
15. [结论](#15-结论)
16. [修订记录](#16-修订记录)
17. [附录](#附录-a关键源码索引)

---

## 1. 执行摘要

| 环境变量 | 默认 | 语义 | 作用域 |
|----------|------|------|--------|
| **`NCCL_NVLS_ENABLE`** | **2（自动）** | 是否允许 **NVLink SHARP（NVLS）**：NVSwitch 上的 CUDA Multicast 规约路径 | **主要影响节点内**（多机时还可组成 NVLS_TREE） |
| **`NCCL_COLLNET_ENABLE`** | 配置默认 **0**；env 可覆盖 | 是否尝试 **CollNet 插件**（常见 IB SHARP / 智能交换机 offload） | **主要影响跨节点集合通信** |

**二者正交，不要混为一谈：**

```text
NVLS     = 节点内 NVSwitch 硬件多播（GPU 侧，通常不经 NIC）
CollNet   = 跨节点集合 offload（CollNet 插件，不是普通 net_ib 点对点）
```

**在「8×H200 + 8×400G RoCE」上的务实结论：**

| 场景 | `NVLS_ENABLE` 1 vs 0 | `COLLNET_ENABLE` 1 vs 0 |
|------|----------------------|-------------------------|
| **单机 8 卡** | **有本质差别**（NVLS 多播 vs Ring/Tree@NVLink） | **无差别**（节点数 < threshold，强制关 CollNet） |
| **多机 + 无 CollNet 插件**（RoCE 常见） | **有差别**（可走 NVLS_TREE；机间仍 RoCE） | **无实质差别**（无插件则 init 打回 0） |
| **多机 + 有 SHARP/CollNet 插件** | 机内可 NVLS | **有本质差别**（机间 offload vs 纯 Ring/Tree+RoCE） |

**常见误区：**

1. `NCCL_NVLS_ENABLE=1` **不会**让跨节点流量走 NVSwitch。  
2. `NCCL_COLLNET_ENABLE=1` **不等于** 打开了普通 IB/RoCE；没有 `ncclCollNet` 插件时等于没开。  
3. **仅有 CollNet 插件不够**：标准路径（如 IB SHARP）还需要 **交换机（或对等网侧）具备 in-network 集体 offload 能力**；见 [§12.7](#127-打开-collnet除了插件还需要交换机硬件吗)。  
4. 默认 `NVLS_ENABLE=2` 已是自动探测；显式写 `1` 是强制，写 `0` 才是彻底关闭。  
5. 多机训练里的 `ibv_post_send` 热路径属于 **NET 传输**；仅开 NVLS/CollNet **不保证** 消除 RoCE 门铃尖峰。

---

## 2. 目标机器拓扑

### 2.1 GPU

- **8× NVIDIA H200**
- 彼此 **NV18**（18 条 NVLink，典型 HGX / NVSwitch 全互联）
- 双 NUMA：
  - GPU0–3 → NUMA 0（CPU 0–47, 96–143）
  - GPU4–7 → NUMA 1（CPU 48–95, 144–191）

### 2.2 网卡（PIX 亲和）

| GPU | NUMA | PIX NIC | 训练网口（示例命名） |
|-----|------|---------|----------------------|
| GPU0 | 0 | mlx5_0 | eth400g0 |
| GPU1 | 0 | mlx5_1 | eth400g1 |
| GPU2 | 0 | mlx5_2 | eth400g2 |
| GPU3 | 0 | **mlx5_5**（注意不是 mlx5_3） | eth400g3 |
| GPU4 | 1 | mlx5_6 | eth400g4 |
| GPU5 | 1 | mlx5_7 | eth400g5 |
| GPU6 | 1 | mlx5_8 | eth400g6 |
| GPU7 | 1 | mlx5_9 | eth400g7 |

另有 **mlx5_3 / mlx5_4**（eth0 / eth1，管理网段，如 `10.93.x`），与 400G 训练网段（如 `10.129.x`）不同。  
**建议：** `NCCL_IB_HCA` 排除管理口，避免误入 NET 选路。

### 2.3 对 NVLS / CollNet 的硬件含义

| 能力 | 本机是否具备 |
|------|----------------|
| NVSwitch（NVS）拓扑 | NV18 全互联 → 图搜索侧通常具备 NVS |
| SM90+（Hopper/H200） | 是 → 满足 NVLS 搜索 `ccMin >= 90` |
| 400G RoCE 点对点 | 是 → **NET / net_ib** 路径 |
| CollNet / SHARP 交换机 offload | **不一定** → 取决于是否加载 CollNet 插件与交换机能力 |

---

## 3. 两个开关分别管什么

### 3.1 概念分层

```text
┌─────────────────────────────────────────────────────────────┐
│                     NCCL 集合通信                             │
├────────────────────────────┬────────────────────────────────┤
│  节点内 (intra-node)        │  节点间 (inter-node)            │
├────────────────────────────┼────────────────────────────────┤
│  NVLS（NVSwitch multicast）│  CollNet 插件（可选 offload）   │
│  Ring/Tree over NVLink P2P │  Ring/Tree over NET (RoCE/IB)  │
│  SHM / P2P 等              │  NVLS_TREE 的机间边也是 NET    │
└────────────────────────────┴────────────────────────────────┘
         ▲                              ▲
         │                              │
   NCCL_NVLS_ENABLE              NCCL_COLLNET_ENABLE
```

### 3.2 参数定义位置

| 参数 | 源码 | 默认 |
|------|------|------|
| `NCCL_NVLS_ENABLE` | `src/transport/nvls.cc`：`NCCL_PARAM(NvlsEnable, "NVLS_ENABLE", 2)` | **2** |
| `NCCL_NVLS_NCHANNELS` | `src/init.cc` | 未定义时由 tuning 决定 |
| `NCCL_NVLS_CHUNKSIZE` | `nvls.cc` | 128 KiB |
| `NCCL_NVLSTREE_MAX_CHUNKSIZE` | `nvls.cc` | -2（自动） |
| `NCCL_COLLNET_ENABLE` | `src/init.cc`：环境覆盖 `comm->config.collnetEnable` | 配置默认 0 |
| `NCCL_COLLNET_NODE_THRESHOLD` | `src/init.cc` | **2**（节点数少于此则关 CollNet） |
| `NCCL_IGNORE_COLLNET_MISMATCH` | `src/init.cc` | 0 |

### 3.3 取值语义速查

**NVLS_ENABLE**

| 值 | 含义 |
|----|------|
| **0** | 禁用 NVLS：不探测、不建 NVLS graph、无 NVLS 算法 |
| **1** | **强制** 认为支持 NVLS（不查 multicast 属性）；多 rank 同 NVML 设备会 `ncclInvalidUsage` |
| **2** | **自动**：有 `cuMulticastCreate` 时查询 `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` |

**COLLNET_ENABLE**

| 值 | 含义 |
|----|------|
| **0** | 不使用 CollNet 算法 / 不 `ncclCollNetSetup` |
| **1** | 尝试启用；仍受插件、节点数、graph 搜索结果等否决 |

非法值会在 config 校验里打回 0（CollNet）或忽略（NVLS channels 等）。

---

## 4. `NCCL_NVLS_ENABLE` 源码路径

### 4.1 初始化：`ncclNvlsInit`（`src/transport/nvls.cc`）

伪流程：

```text
ncclNvlsInit(comm):
  nvlsSupport = 0
  if hasMultiRankNvml:
    if NVLS_ENABLE==1 → ERROR 并失败
    else return（不启用）
  if NVLS_ENABLE==0 or gpuCount<2 → return
  if NVLS_ENABLE==2:
    若存在 cuMulticastCreate:
      cuDeviceGetAttribute(MULTICAST_SUPPORTED) → nvlsSupport
  else:  # ==1
    nvlsSupport = 1
  log: "NVLS multicast support is [not] available"
```

**要点：**

- 需要 **至少 2 个 GPU**。  
- 默认 **2** 依赖 **CUDA 12.1+ 用户态 API** 与驱动上报的 multicast 能力。  
- H200 硬件通常具备；最终以日志 `available / not available` 为准。

### 4.2 图搜索：`NCCL_TOPO_PATTERN_NVLS`（`src/graph/search.cc`）

```text
if pattern==NVLS and (NVS.count==0 or ccMin < 90):
  return 空图   // 不建 NVLS channels

// 单机：强制拉满 channel，使各 GPU 均匀参与
if system.inter==0 and pattern==NVLS:
  minChannels = maxChannels
```

与本机的关系：

- H200 → SM90+，`ccMin >= 90` 满足。  
- NV18 → 通常存在 NVS 节点。  
- **单机** NVLS 搜索会倾向 **满 heads / 满 channels** 的机内结构。

### 4.3 初始化管线中的挂接点（`src/init.cc`）

```text
ncclNvlsInit()
  → if nvlsSupport: ncclTopoCompute(nvlsGraph)
  → AllGather 后若 nvlsGraph.nChannels==0: 清掉 nvlsSupport
  → if nvlsSupport: ncclNvlsTuning()   // nChannels, chunkSize, treeMaxChunkSize
  → ncclNvlsSetup / ncclNvlsBufferSetup
  → if nNodes>1: ncclNvlsTreeConnect()  // 机间 tree 的 P2P 连接
```

### 4.4 Tuning 量级（`ncclNvlsTuning`）

- 按 SM 架构选默认 channel 数（如 SM90 约 16；SM100 单机/多机另有表）。  
- 多机 SM100 路径会按 **节点数、ppn、`minNetBw`、GPU–NIC path** 调整 channel 与 chunk。  
- 用户可用 `NCCL_NVLS_NCHANNELS`、`NCCL_NVLS_CHUNKSIZE` 覆盖。  
- 最终 channel 数 clamp 到 `[minCTAs, maxCTAs]`。

本机 **400G + PIX** 属于高 `minNetBw`、近路径，多机 NVLS_TREE 会按高带宽表走。

### 4.5 设备侧算法（`src/device/all_reduce.h`）

`NCCL_ALGO_NVLS` 的 AllReduce 在 GPU 上按 warp 角色切分，例如：

- scatter / gather  
- reduce / bcast（走 NVLS up/down）  
- 多机时可能含与 `nvls->out`（网络侧）的交互  

**机内 NVLS 段不经过 proxy 的 `ibv_post_send`。**  
机间边仍可能走 NET（Tree）或 CollNet。

### 4.6 多机 NVLS Tree 连接（`ncclNvlsTreeConnect`）

仅当 `nvlsSupport && nNodes > 1`：

- 为每个 NVLS channel 配置 treeUp / treeDown  
- `ncclTransportP2pConnect` + `ncclTransportP2pSetup`  
- 日志：`Connected NVLS tree`

---

## 5. `NCCL_COLLNET_ENABLE` 源码路径

### 5.1 环境变量注入（`src/init.cc`）

```text
每次 comm init 重新读 NCCL_COLLNET_ENABLE
  → 覆盖 comm->config.collnetEnable
  → 仅 0/1 合法，其它打回 0
```

### 5.2 否决链（必须全部通过才真正启用）

```text
① collNetSupport(comm)?
     = (comm->ncclCollNet != nullptr)
     否 → collnetEnable = 0

② collnetEnable==1 ?
     是 → ncclTopoCompute(COLLNET_CHAIN)
          ncclTopoCompute(COLLNET_DIRECT)

③ collNetChainGraph.nChannels == 0 ?
     是 → collnetEnable = 0

④ nNodes < NCCL_COLLNET_NODE_THRESHOLD (默认 2) ?
     是 → collnetEnable = 0
          log: less than CollNet node threshold

⑤ 连接阶段
     ncclCollNetSetup / BufferSetup
```

`collNetSupport` 定义（`src/include/coll_net.h`）：

```c
static int collNetSupport(struct ncclComm* comm) {
  return comm->ncclCollNet != nullptr ? 1 : 0;
}
```

**没有 CollNet 插件，环境变量写 1 也会在 init 早期被清零。**

### 5.3 CollNet vs 普通 NET

| 维度 | NET（`net_ib` / RoCE） | CollNet |
|------|------------------------|---------|
| 传输表 | `TRANSPORT_NET` | `TRANSPORT_COLLNET` |
| 典型用途 | 点对点 RDMA、Ring/Tree 的边 | 集合 offload（`iallreduce` 等） |
| API | `isend/irecv/test` | 插件 `iallreduce` / gather 等 |
| 硬件 | 任意 IB/RoCE NIC | 常需 SHARP 等交换机能力 |
| 算法名 | RING、TREE、NVLS_TREE 机间边… | **COLLNET_CHAIN**、**COLLNET_DIRECT** |

### 5.4 COLLNET_DIRECT 与 NVSwitch

`src/graph/tuning.cc`：

```text
if nvsCount==0 and algo==COLLNET_DIRECT → disable
```

本机有 NV18，**若 CollNet 插件可用**，DIRECT 在拓扑上合格；  
**无插件时该分支根本走不到。**

### 5.5 连接与 Proxy

`ncclCollNetSetup` 成功后：

- 建立 collnet chain / direct 的 peer 关系  
- 通过 **proxy** 推进 CollNet 进度（与 NET proxy 类似，但是 CollNet 传输 vtable）  
- 日志形态：`CollNet ... via COLLNET/<plugin>/<dev>`

---

## 6. 算法矩阵：何时启用 / 禁用

核心逻辑在 `src/graph/tuning.cc`（简化）：

```text
for each function f, algorithm a:
  disable = 0

  // 单机不用 NVLS_TREE
  if nNodes==1 and a==NVLS_TREE:
    disable = 1

  // collnet 关闭时：
  //   - 禁用 COLLNET_DIRECT / COLLNET_CHAIN
  //   - 多机时禁用「纯 NVLS」（注意：不是 NVLS_TREE）
  if collnetEnable==0 and
     (a==COLLNET_DIRECT or a==COLLNET_CHAIN or
      (a==NVLS and nNodes>1)):
    disable = 1

  if collnetEnable and a==COLLNET_CHAIN and collNetChainSupport==0:
    disable = 1

  if nvsCount==0 and a==COLLNET_DIRECT:
    disable = 1
```

### 6.1 解读表

| 算法 | 单机 | 多机 collnet=0 | 多机 collnet=1（且插件 OK） |
|------|------|----------------|---------------------------|
| RING / TREE | ✅ | ✅ | ✅ |
| **NVLS** | ✅（需 nvlsSupport） | ❌（被上面规则禁用） | ✅（可与 CollNet 组合） |
| **NVLS_TREE** | ❌ | ✅（需 nvlsSupport） | ✅ |
| **COLLNET_CHAIN** | ❌（节点数/插件） | ❌ | ✅（需 chain support） |
| **COLLNET_DIRECT** | ❌ | ❌ | ✅（需 NVS） |

**关键细节：**

- `collnetEnable=0` **不会** 关掉多机的 **NVLS_TREE**。  
- `collnetEnable=0` **会** 关掉多机的 **纯 NVLS** 算法条目。  
- 单机 **NVLS** 与 CollNet **无关**。

---

## 7. 在本机拓扑上的 1 vs 0 本质差异

### 7.1 `NCCL_NVLS_ENABLE`

| | **=1 或 =2 且 support=1** | **=0** |
|--|---------------------------|--------|
| 图搜索 | 计算 NVLS graph | 不计算 / 无 NVLS channels |
| 资源 | multicast buffer、channel、可能 tree 连接 | 无 |
| 单机 AllReduce 候选 | **NVLS** + Ring/Tree | **仅 Ring/Tree（NVLink P2P）** |
| 多机候选 | **NVLS_TREE**（机内 NVLS + 机间 tree@NET）；若 CollNet 开还可 NVLS+CollNet | Ring/Tree@NVLink + NET |
| 是否自动用 NIC 做机内规约 | **否**（NVLS 机内不经 NIC） | **否**（单机 Ring 也不经 NIC） |
| 与 400G 网卡 | 多机机间仍用 | 多机机间仍用 |

**本质差别：**  
是否允许走 **NVSwitch 硬件多播规约路径**，从而改变 **节点内** 的算法、channel 数、kernel 形态与带宽/延迟曲线。  
**不是**「开了就不用网卡做多机通信」。

### 7.2 `NCCL_COLLNET_ENABLE`

| | **=1 且插件+多机+graph 成功** | **=0 或否决链失败** |
|--|------------------------------|---------------------|
| CollNet graph | 搜索并连接 | 无 |
| 算法 | COLLNET_* 可进 tuner | 全部禁用 |
| 多机纯 NVLS | 可启用 | 禁用（NVLS_TREE 仍可） |
| 机间数据面 | 可能 offload 到交换机 | 标准 NET RoCE（proxy + `ibv_post_send` 等） |

**在本机（典型纯 RoCE、无 CollNet 插件）上的本质差别：**

```text
COLLNET_ENABLE=1  →  init 中 collNetSupport==0 → 强制 0
COLLNET_ENABLE=0  →  本来就是 0
→ 运行时无实质区别
```

**单机 8 卡：**

```text
nNodes=1 < COLLNET_NODE_THRESHOLD(2) → 强制 0
→ 无论 env 写 0 还是 1，CollNet 均关闭
```

---

## 8. 组合矩阵（单机 / 多机 / 有无插件）

记：

- **N**：节点数  
- **V**：`nvlsSupport==1`  
- **C**：CollNet 插件存在且 setup 成功  

### 8.1 单机（N=1，8×H200）

| NVLS env | CollNet env | 实际 NVLS | 实际 CollNet | 主路径 |
|-----------|-------------|-----------|--------------|--------|
| 0 | 0/1 | 关 | 关 | Ring/Tree @ NVLink |
| 1/2 且 V | 0/1 | 开 | 关 | **NVLS** 优先候选 + Ring/Tree 备选 |
| 1/2 但 !V | 0/1 | 关 | 关 | 同 NVLS=0 |

**结论：单机只需要关心 NVLS；CollNet 开关可忽略。**

### 8.2 多机（N≥2）且无 CollNet 插件（!C，RoCE 常见）

| NVLS | CollNet env | 实际 | 机内 | 机间 |
|------|-------------|------|------|------|
| 0 | 0 或 1 | CollNet 无效 | Ring/Tree NVLink | Ring/Tree **RoCE** |
| 1/2 且 V | 0 或 1 | CollNet 无效 | **NVLS** | **NVLS_TREE 的 tree 边 @ RoCE** 或 Ring/Tree |

**结论：CollNet 1/0 无差；NVLS 改变机内与是否 NVLS_TREE。**

### 8.3 多机且 CollNet 真可用（C）

| NVLS | CollNet | 机内 | 机间 |
|------|---------|------|------|
| 0 | 0 | Ring/Tree | Ring/Tree RoCE |
| 0 | 1 | CollNet 拓扑相关 | **CollNet offload** |
| 1 | 0 | NVLS | Tree/Ring RoCE |
| 1 | 1 | NVLS | CollNet 和/或 Tree；**tuner 选择** |

---

## 9. 数据路径对照

### 9.1 单机 AllReduce

```text
NVLS 开启且被选中:
  GPU ranks ──► NVSwitch multicast reduce/bcast ──► GPU ranks
  （无 eth400g，无 net_ib post_send）

NVLS 关闭:
  GPU ranks ──NVLink P2P── Ring/Tree 邻居 ──► …
  （仍无 NIC）
```

### 9.2 多机 AllReduce（无 CollNet 插件）

```text
NVLS 开启:
  节点内: NVLS (NVSwitch)
  节点间: Tree/Ring 边 → TRANSPORT_NET → mlx5 RoCE
          → proxy → ncclIbIsend → ibv_post_send

NVLS 关闭:
  节点内: Ring/Tree over NVLink
  节点间: 同上 RoCE 路径
```

### 9.3 多机 + CollNet 可用

```text
COLLNET_ENABLE 生效:
  节点 head 与 CollNet 设备建连
  规约可走插件 iallreduce（交换机侧聚合）
  主机侧 post 模式与「每条 ring 边一次 RDMA」不同
```

---

## 10. 与 `ibv_post_send` / 训练性能的关系

详见 `ibv_post_send_ms_stall_analysis.md`。与本开关交叉结论：

| 设置变化 | 对 `ibv_post_send` 的影响 |
|----------|---------------------------|
| 单机 + NVLS=1 | 机内 AllReduce **几乎不** 走 IB post |
| 单机 + NVLS=0 | 机内仍不走 NIC；post 不是主路径 |
| 多机 + 任意 NVLS | **机间 NET 仍大量 post**；NVLS 只优化机内 |
| CollNet=1 但无插件 | 与 0 相同，**不改变** post 行为 |
| CollNet=1 且 SHARP 工作 | 机间可能从多边 ring RDMA 变为 offload，post 模式变化 |

**因此：**

- 不要用「两个 ENABLE 都设 1」来解释或指望消除 **RoCE 上 post_send 毫秒尖峰**。  
- 若 profiling 显示热在 `ncclIbMultiSend` / `ibv_post_send`，应优先查 **NET 路径、NIC 亲和、计时口径、平台/驱动**，而不是只拧 CollNet/NVLS。

---

## 11. 如何用日志验证是否生效

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,NET
# 可选排除管理网卡
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
```

| 日志关键词 | 含义 |
|------------|------|
| `NCCL_COLLNET_ENABLE set by environment to 1` | env 已读入 |
| `NVLS multicast support is available` | nvlsSupport=1 |
| `NVLS multicast support is not available` | 硬件/驱动未开 multicast 或 ENABLE 逻辑未通过 |
| `NVLS tuning: nChannels … chunkSize …` | NVLS 进入 tuning |
| `Connected NVLS tree` | 多机 NVLS tree 连接完成 |
| `Rank x: N CollNet devices` | CollNet 设备数；**0 则插件侧不可用** |
| `less than CollNet node threshold` | 节点数不够被关 |
| `CollNet … via COLLNET/…` | CollNet 传输建连成功 |
| Graph dump / algo 相关 INFO | 最终是否出现 NVLS / COLLNET_* |

**判定口诀：**

```text
想确认 NVLS 真开了:
  available + (单机有 NVLS tuning / 多机有 Connected NVLS tree)

想确认 CollNet 真开了:
  CollNet devices > 0
  + 无 threshold/support 否决
  + 出现 COLLNET 连接日志
  仅有 COLLNET_ENABLE=1 的 ENV 日志不够
```

---

## 12. 节点间 CollNet：原理、机制与实现流程

> 代码主线：`src/transport/coll_net.cc`、`src/graph/connect.cc`、`src/device/all_reduce.h`（COLLNET_*）、`src/plugin/net/*`（插件适配）、`src/init.cc`（setup 挂接）。

### 12.1 原理：为什么需要 CollNet

普通多机 AllReduce（Ring / Tree + **NET**）把集合通信拆成大量 **点对点 RDMA**：

```text
每个 ring 边 / tree 边:
  GPU → (可选 bounce) → NIC ibv_post_send → 网络 → 对端 NIC → GPU
  主机 proxy 为每条边做 progress
```

当节点数、channel、消息规模上升时：

- 边数与 `post_send` 次数爆炸  
- 交换机只做转发，**不参与 reduce**  
- 带宽受「多跳点对点」与 host 进度线程限制  

**CollNet（Collective Network）** 的目标是：把 **跨节点的集合语义**（至少 AllReduce，以及 AllGather / ReduceScatter 等）交给 **专用插件 + 常配合智能交换机（如 IB SHARP）** 完成一次 offload 规约，而不是在主机上拼很多 P2P 边。

```text
逻辑对比:

  NET 模式:   多 rank × 多边 × 多次 RDMA Write/WithImm
  CollNet:    节点 head 参与 collNetComm → 插件 iallreduce(…)
              （交换机/网卡侧完成跨节点 reduce+broadcast 语义）
```

**注意：** CollNet **不是** `net_ib` 的别名。`TRANSPORT_COLLNET` 有独立 vtable；数据面核心是插件的 `iallreduce` / `iallgather` / `ireducescatter`，不是 `ncclIbIsend`。

### 12.2 角色与拓扑抽象

#### 12.2.1 Head（轨 / rail）

每个节点选出若干 **head rank**（与 channel / NVLS heads 对齐）：

- **有 NVLS 时**（`ncclCollNetSetup`）：head 直接复用 `comm->nvlsHeads`（与 NVLS 多播 heads 一致）。  
- **无 NVLS 时**：从 `COLLNET_DIRECT` graph 的 `intra[c*localRanks+0]` 收集 unique heads。

Head 是 **本节点对接 CollNet 网络的“出口/入口 GPU”**：

- 机内：其它 GPU 把数据 reduce/scatter 到 head（P2P / NVLS / chain）  
- 机间：head 的 **proxy 线程** 调 `ncclCollNet->iallreduce` 等  

#### 12.2.2 两种算法形态

| 算法 | 机内结构 | 机间 | 适用 |
|------|----------|------|------|
| **COLLNET_CHAIN** | 节点内链（类似 tree/chain，`collnetChain.up/down`） | 链上特定 rank 接 CollNet | 更通用；需 chain heads 是 CollNet heads 的子集 |
| **COLLNET_DIRECT** | 叶子 ↔ 多个 heads（`collnetDirect.up/down/heads`） | 每个 head 一条 CollNet 轨 | 要求 **NVSwitch**（`nvsCount>0`）；`maxLocalRanks ≤ NCCL_MAX_DIRECT_ARITY+1` |

`connectCollNet`（`src/graph/connect.cc`）为 DIRECT 设置：

- `headRank`：本 rank 是否为第 i 个 head  
- `out = nRanks`：伪 peer，表示 **CollNet 网络侧**（kernel 里 `direct->out`）  
- `down[]`：head 连接的同节点 peer  
- `up[]` / `heads[]`：叶子连向的 heads  
- `shift`：错开头发送时刻，减轻 head 热点  

### 12.3 初始化与建连流程

```text
[1] 插件加载
    ncclCollNet != nullptr  (collNetSupport)

[2] env / config
    NCCL_COLLNET_ENABLE=1
    nNodes >= NCCL_COLLNET_NODE_THRESHOLD (默认 2)

[3] 图搜索 (init.cc)
    ncclTopoCompute(COLLNET_CHAIN)   pattern TREE, collNet=1
    ncclTopoCompute(COLLNET_DIRECT)  pattern COLLNET_DIRECT
    graph 需 net 设备 collSupport

[4] 拓扑后处理 (connect.cc)
    可能增加 channel 以吃满机内带宽
    connectCollNet() 填 collnetDirect 字段
    chain depth = nRanks/nNodes

[5] ncclCollNetSetup (coll_net.cc)
    确定 collNetHeads / collNetHeadsNum
    collNetChainSupport = heads 是否匹配 chain
    分配 collNetSharedRes (nChannels, buffSize)
    collNetInitRailRankMap
    对每个 channel × 每个 head:
      ncclTransportCollNetSetup(..., collNetRecv)
      ncclTransportCollNetSetup(..., collNetSend)
    首 channel 后 AllGather 校验，失败则整 comm 关 CollNet

[6] Buffer setup
    ncclCollNetChainBufferSetup
    ncclCollNetDirectBufferSetup (localRanks 足够小时)
```

#### 12.3.1 传输 setup（`sendSetup` / `recvSetup`）

每个 head 上的 connector：

1. `ncclTopoGetNetDev` 选 CollNet 网卡 / rail  
2. `ncclTopoCheckGdr` 决定是否 GDR  
3. `ncclProxyConnect(TRANSPORT_COLLNET, …)` 把 **proxy 绑到本 rank**  
4. `ncclProxyCallBlocking(..., ncclProxyMsgSetup, …)`：  
   - **Recv 侧** 得到 `collNetHandle`（供 connect 交换）  
   - **Send 侧** 完成发送资源准备  

日志示例：

```text
CollNet 00/0 : <rank> [send] via COLLNET/<plugin>/<dev>/GDRDMA
CollNet 00/0 : <rank> [receive] via COLLNET/<plugin>/<dev>
```

Proxy 侧资源（`sendResources` / `recvResources`）包括：

- `collNetComm`：插件集合通信句柄  
- `connectMap`：host/dev/shared 缓冲布局  
- `sendMem` / `recvMem`：与 GPU 的 head/tail、`connFifo` 握手  
- protocol 缓冲 mhandle、可选 GDR sync/flush  
- `reqFifo`：collnet 请求槽，控制 send/recv 轮次  

### 12.4 运行时数据面流程（以 AllReduce 为例）

#### 12.4.1 端到端分层

```text
应用: ncclAllReduce
  → enqueue 选中 COLLNET_DIRECT 或 COLLNET_CHAIN
  → 启动 device kernel (GPU)
  → 同时 proxy 线程 progress TRANSPORT_COLLNET

机内 (GPU kernel):
  非 head: scatter/reduce 到 head(s)  [P2P / NVLS / chain]
  head:    把本轨规约结果交给 collnet peer (out)
           再从 collnet 收回结果并 bcast/gather 给叶子

机间 (CPU proxy @ head):
  等 GPU 通过 connFifo / tail 交付数据
  → ncclCollNet->iallreduce(send, recv, count, dtype, redOp, mhandles, &req)
  → test(req) 直到完成
  → 更新 head/credit，让 GPU 继续
```

#### 12.4.2 Kernel 侧（COLLNET_DIRECT AllReduce 摘要）

`RunWorkColl<…, NCCL_ALGO_COLLNET_DIRECT, …>`（`all_reduce.h`）按 tid 分段：

| 线程角色 | 条件 | 行为 |
|----------|------|------|
| **Scatter** | 有 up | 把 `sendbuff` 分片送到 heads（`prims.scatter`，带 `headRank/shift`） |
| **Reduce→Net** | `out != -1` 且有 down | 收叶子 + reduce，`recvReduceDirectSend` 发到 **collnet out** |
| **直送 Net** | head 无 down | 本 head 直接 `send` / UB 下 notify |
| **Gather** | 有 up | 从 heads `directGather` 回 `recvbuff` |
| **Bcast from Net** | 有 out | 从 collnet 收结果 再广播给 down |

`direct->out == comm->nRanks` 表示 **网络伪 peer**，由 CollNet 传输的 conn 承接，而不是另一个 GPU rank。

#### 12.4.3 Proxy 侧 progress（`sendProxyProgress`）

状态机概要（`coll_net.cc`）：

```text
ncclProxyOpReady
  → 初始化 sub->base/posted/received/transmitted/done
  → ncclProxyOpProgress

Progress 循环 (每 sub / group):
  [posted]  通知 GPU 可用 slot（写 sendHead / GDR sync）
  [received] 等 GPU 写完 (recvTail / connFifo.size)
  [transmitted] 有序地发起插件调用:
       collNetIallreduce 或 collNetRegIallreduce
       (以及 iallgather / ireducescatter 变体)
  [done]    test 完成，归还 credit，推进 step
```

要点：

1. **`COLLNET_GROUP_NSUBS`（8）**：多个 sub 组合成 group，控制 outstanding 与偏移计算。  
2. **强顺序**：`ordered` 保证 collnet op 跨 sub 的 transmitted 顺序，满足插件/交换机有序假设。  
3. **注册缓冲（UB）**：`sub->reg` + `isOneRPN` 时，插件可 **直接碰用户 GPU buffer**（`collNetRegIallreduce`），减少中间 `region` 拷贝；多 RPN 时常需中间缓冲保证连续性。  
4. **与 NET 的差异**：progress 热点是 **`iallreduce` + test**，不是 `ibv_post_send` 链；底层插件内部可能仍用 verbs，但对 NCCL 透明。

### 12.5 机制要点小结

| 机制 | 说明 |
|------|------|
| **Offload** | 跨节点 reduce 语义下沉到 CollNet 插件/交换机 |
| **Head/Rail** | 每节点少量 GPU 对接网络，降低 NIC 争用与边数 |
| **GPU–Proxy 握手** | `connFifo` + head/tail（可 GDR）实现无锁流水 |
| **双算法** | Chain（深机内链）vs Direct（NVSwitch 机内扁平 + 多 head） |
| **与 NVLS 协同** | 有 NVLS 时 heads 对齐 NVLS；`nvls.out` 可接 CollNet（`connectNvls`） |
| **失败回退** | setup 任一环节失败 → `collnetEnable=0`，退回 Ring/Tree+NET |

### 12.6 在 8×H200 + 400G RoCE 上的含义

- **有 SHARP 等 CollNet 后端（插件 + 交换机能力）**：多机 AllReduce 可显著减少主机侧点对点 RDMA 与 proxy 负担。  
- **仅有 RoCE、无 CollNet 插件**：本节流程 **整条不发生**；`COLLNET_ENABLE=1` 被否决。  
- **有插件但交换机无集体 offload**：通常 setup/运行失败或不可用，**不能**当作有效 CollNet（见 §12.7）。  
- CollNet **不替代** 机内 NVLS；机内仍可由 NVLS / P2P 完成到 head 的聚合。

### 12.7 打开 CollNet：除了插件，还需要交换机硬件吗？

**工程结论（针对标准部署）：是的。**  
有效 CollNet ≈ **NCCL CollNet 插件** + **网侧 in-network 集体 offload 能力**（最常见是 **IB 交换机 SHARP**）。  
仅主机上有一个 CollNet `.so`、而交换机只做普通转发时，**不能**指望真正的 CollNet 收益。

#### 12.7.1 NCCL 源码强制什么、不强制什么

| 层级 | NCCL 是否检查 | 说明 |
|------|----------------|------|
| `ncclCollNet != nullptr` | **是** | `collNetSupport()` 仅此条件 |
| `NCCL_COLLNET_ENABLE=1`、节点数、graph、`ncclCollNetSetup` | **是** | 软件否决链（§5） |
| 交换机是否支持 SHARP / 网内 reduce | **否** | 库内 **不探测** 交换机集体功能 |

因此：

```text
软件门槛（NCCL）:  插件能加载 + setup/connect 成功 + 能调 iallreduce/…
硬件门槛（后端）:  由 CollNet 插件实现决定；标准 SHARP 路径依赖交换机 ASIC
```

硬件是否够用，往往在 **插件 connect 或首次 collective** 时才失败，而不是 NCCL 在 init 里打印「no SHARP switch」。

#### 12.7.2 标准路径：插件 ≈ SHARP，交换机必须具备集体功能

常见生产组合（NVIDIA 生态）：

| 层级 | 需要什么 |
|------|----------|
| NCCL | CollNet 插件（coll 类 net 插件） |
| 网卡 | 支持 SHARP 的 IB HCA + 匹配 OFED |
| **交换机** | **支持 SHARP 的 IB 交换机**（聚合在交换机侧完成） |
| 子网 | SHARP 使能、固件/SM 配置、版本匹配等 |

| 只有… | 结果 |
|--------|------|
| 只有插件，交换机不会 in-network reduce | 建连/运行失败，或无法形成有效 CollNet |
| 只有交换机 SHARP，无 CollNet 插件 | NCCL 仍走普通 NET Ring/Tree，用不上交换机聚合 |
| 插件 + SHARP（或等价 offload） | 才可能真正走 §12.4 的 `iallreduce` 路径 |

**记忆：标准 SHARP CollNet = 主机插件 + 交换机集体功能，缺一不可。**

#### 12.7.3 理论上「只有插件、没有交换机 SHARP」可能吗？

**可能，但取决于插件后端，不是 NCCL 默认保证。**

CollNet 对 NCCL 只是一组 API（`listen`/`connect`/`iallreduce`/`iallgather`/…）。后端可以是：

| 实现方式 | 是否需要「交换机集体功能」 |
|----------|----------------------------|
| **IB SHARP**（最常见） | **需要** SHARP 交换机 |
| 其它交换机 in-network compute | **需要** 对应交换机能力 |
| SmartNIC / DPU 上做 reduce | 需要 DPU 能力；不一定是「交换机 SHARP」 |
| 纯主机软件再开一套集合（极少见） | 不需要交换机聚合，但通常 **无官方高性能方案**，也失去 offload 意义 |

NCCL 树内 **不提供**「无硬件也能假 CollNet」的生产级内置后端。

#### 12.7.4 与本机（8×H200 + 400G RoCE）对照

```text
有 net_ib / RoCE          →  普通 NET（Ring/Tree）可用
有 CollNet 插件吗？        →  许多 RoCE 集群没有，或只对 IB SHARP 域提供
交换机有 SHARP 吗？        →  以太网 RoCE 交换机多数 **没有** IB SHARP
```

因此：

```text
NCCL_COLLNET_ENABLE=1
  + 无 CollNet 插件              → init 直接当没开
  + 有插件但交换机无聚合能力    → setup/运行失败或不可用
  + 插件 + SHARP（或等价硬件）  → 才可能真正走 CollNet
```

**纯 RoCE、无 SHARP/无对等 offload 时：CollNet 开关打开也通常等于没开；应依赖 NET +（可选）NVLS 机内。**

#### 12.7.5 一句话

> **CollNet = NCCL 插件接口 + 后端 offload 能力。**  
> 后端在真实集群里几乎总是「带集体功能的网络硬件」（最常见 **IB SHARP 交换机**），不是只靠主机上的插件 `.so`。  
> NCCL 源码只强制插件与 setup；**交换机能力由插件/硬件栈保证。**

---

## 13. 节点内 NVLS AllReduce：原理、机制与实现流程

> 代码主线：`src/transport/nvls.cc`、`src/graph/connect.cc`（`connectNvls`）、`src/device/all_reduce.h`（`NCCL_ALGO_NVLS`）、CUDA Multicast API（`cuMulticast*`）。

### 13.1 原理：NVLS 解决什么问题

节点内传统 AllReduce：

```text
Ring:  O(N) 步 NVLink 单向传递 + reduce
Tree:  O(log N) 步，但仍是多次 P2P
```

在 **NVSwitch 全互联（如 H200 NV18）** 上，硬件支持 **Multicast（多播）内存**：

- 一次写到 **MC 地址**，NVSwitch 把数据分发给订阅该 multicast group 的 GPU  
- 结合 reduce 语义，可实现 **近似一步的机内规约/广播**（逻辑上远少于 ring 的 N-1 步）

**NVLS（NVLink SHARP）** 在 NCCL 中的含义：

> 利用 **CUDA Multicast + NVSwitch**，在节点内用 UC（unicast）缓冲与 MC（multicast）视图协作，完成 AllReduce 的 reduce + broadcast，而 **不经过 NIC / proxy**。

```text
Ring 机内:   GPU0 → GPU1 → … → GPU7 → … (多步 P2P)
NVLS 机内:  各 GPU ─scatter→ head 相关 MC/UC 路径 ─reduce/bcast→ gather
             数据面在 GPU kernel + NVSwitch，无 ibv_post_send
```

### 13.2 核心机制

#### 13.2.1 Multicast Group（CUDA）

`ncclNvlsGroupCreate` / 相关分配路径：

```text
cuMulticastCreate(mcHandle, prop)     // 创建 MC 对象
cuMulticastAddDevice(mcHandle, dev)   // 本 GPU 加入
cuMulticastBindMem / BindAddr         // 把 UC 物理内存绑到 MC
cuMemMap(mcptr, …, mcHandle)          // 得到 GPU 可访问的 MC 映射
```

节点内 ranks 通过 **shareable handle**（导出/导入或 UDS）共享同一 MC group：

- rank0 create + export  
- 其它 rank `ncclNvlsGroupConnect` import  
- 再 `AddDevice` + `Bind*`  

结果：同一逻辑缓冲存在：

| 视图 | 含义 |
|------|------|
| **UC（unicast）** | 本 GPU 私有映射，用于本地读写、credit |
| **MC（multicast）** | 写一次、多 GPU 可见（经 NVSwitch） |

#### 13.2.2 Heads 与伪 peer

`connectNvls`（`connect.cc`）：

```text
channel->nvls.nHeads = nHeads
channel->nvls.up[h]  = nRanks + 1 + h     // 伪 peer：NVLS head 槽
channel->nvls.down   = nRanks + 1 + headRank
channel->nvls.headRank = 本 rank 是否为某 head，否则 -1
channel->nvls.out    = -1 或 nRanks（接 CollNet 时）
```

- **`up[]`**：逻辑上连到各个 head 的 NVLS 连接  
- **`down`**：head 侧对应的下行  
- 真实 peer 编号 ≥ `nRanks` 表示 **NVLS 传输对象**，不是另一个 MPI rank  

#### 13.2.3 Credit / head-tail 同步（无大数据缓冲的 conn）

`ncclNvlsSetup` 为每个 head×channel 配置：

```text
Reduce  方向: UC credit  (send head/tail)  ↔  MC credit (recv，带 NCCL_NVLS_MIN_POLL)
Broadcast 方向: MC → UC 对称的一对 conn
buffs[SIMPLE] = NULL   // 数据走 multicast 映射的用户/注册缓冲语义，不靠常规 ring buffer
stepSize = nvlsChunkSize
```

GPU kernel 用 **head/tail 轻量握手** 控制 reduce/bcast 步，而不是像 NET 那样大块 staging + proxy。

#### 13.2.4 注册用户缓冲（可选加速）

`ncclNvls` 路径支持把 **用户 buffer 绑进 MC**（`cuMulticastBindAddr` 等），`work->regUsed` 时 scatter/gather 可走 **零拷贝/最小拷贝** 语义（kernel 中 `nelem=0` 的分支表示注册路径）。

### 13.3 初始化流程（节点内）

```text
[1] ncclNvlsInit
      NVLS_ENABLE、gpuCount、MULTICAST_SUPPORTED
      → nvlsSupport

[2] ncclTopoCompute(PATTERN_NVLS)
      要求 NVS 存在且 ccMin≥90
      单机 force minChannels=maxChannels（均匀 heads）

[3] AllGather 后 nvlsChannels；失败则 nvlsSupport=0

[4] ncclNvlsTuning
      nChannels / chunkSize / treeMaxChunkSize

[5] connectNvls
      填 channel.nvls 字段；多机再算 treeUp/Down

[6] ncclNvlsSetup
      可与 parent 共享 nvlsResources
      分配 UC/MC credit 内存 (nvlsAllocateMem)
      initNvlsChannel
      配置 peers[nRanks+1+h] 的 send/recv conn
      H2D 拷贝到 devPeers

[7] ncclNvlsBufferSetup
      更大块 UC/MC 数据缓冲（按 channel/head/协议）

[8] 多机: ncclNvlsTreeConnect
      机间 tree 用 P2P transport 连接 treeUp/Down
```

### 13.4 单机 NVLS AllReduce 运行时流程（`oneNode`）

`RunWorkColl<AllReduce, T, RedOp, NCCL_ALGO_NVLS, SIMPLE>` 且 `work->oneNode`（`all_reduce.h`）：

线程块按 warp 组划分（总数约 `NCCL_MAX_NTHREADS`）：

| 角色 | tid 区间（逻辑） | 做什么 |
|------|------------------|--------|
| **Scatter** | 前段 | 从 `sendbuff` 按 head 分片 `prims.scatter` → `nvls->up` |
| **Gather** | 中段 | 从 `nvls->up` `prims.gather` → `recvbuff` |
| **Reduce/Bcast** | 仅 `headRank != -1` | 通过 `nvls->down` 做 `directRecvDirectSend`：MC 路径上的 reduce+broadcast |

```text
单机逻辑流水（简化）:

  所有 rank:
    Scatter:  用户 sendbuff 分片送到各 head 的 NVLS 连接

  Head ranks:
    在 MC/UC 上完成 reduce，再 broadcast 结果

  所有 rank:
    Gather:  从 NVLS 连接收集到用户 recvbuff

同步: 组间 named barrier / Primitive 内部 head-tail
无:   proxy 线程、ibv_post_send、CollNet
```

数据划分：

- `loopCount = nHeads * chunkSize`  
- 每 head 负责交错的 chunk（`headRank * chunkSize` 偏移）  
- `regUsed` 时减少显式拷贝量  

### 13.5 多机时 NVLS 如何嵌入（对照，非纯节点内）

`work->oneNode == false` 时同一 kernel 模板扩展为：

```text
Scatter / Gather: 仍机内 NVLS
Head Reduce:  directRecvDirectSend 目标含 nvls->out
             → 把机内规约结果送给「网络侧」(CollNet 或 NVLS_TREE 的上联)
Head Bcast:   从网络收回后再经 NVLS 广播
```

另有 **`NCCL_ALGO_NVLS_TREE`**：

- 机内：NVLS  
- 机间：经典 tree（P2P/NET），**不要求 CollNet**  
- 单机被 tuning 禁用  

这与「纯节点内 NVLS AllReduce」的差异：节点内段机制相同，多了机间边。

### 13.6 机制要点小结

| 机制 | 说明 |
|------|------|
| **CUDA Multicast** | NVSwitch 上硬件多播内存，支撑高效 reduce/bcast |
| **UC + MC 双映射** | 本地控制面 + 多播数据面 |
| **Heads** | 并行多轨规约，提高 NVSwitch 利用率 |
| **Credit 同步** | 轻量 head/tail，可 `NCCL_NVLS_MIN_POLL` |
| **全 GPU 数据面** | 节点内不经 proxy；与 CollNet/NET 的 CPU progress 模型本质不同 |
| **注册缓冲** | 可选把用户内存绑入 MC，降低拷贝 |
| **资源共享** | `ncclCommSplit` 等同节点 parent 可共享 `nvlsResources` |

### 13.7 与 Ring 机内 AllReduce 的对比

| 维度 | Ring（机内） | NVLS（机内） |
|------|--------------|---------------|
| 步数 | O(N) 环步 | 逻辑上 O(1) 量级的 scatter/reduce/gather 结构 |
| 链路 | NVLink P2P peer | NVSwitch multicast + head 并行 |
| CPU | 无 proxy（纯机内） | 无 proxy |
| 依赖 | 通用 | SM90+、multicast、NVS、驱动 |
| 调优 | nChannels、LL/LL128/Simple | `nvlsChannels`、`nvlsChunkSize`、nHeads |

### 13.8 在 8×H200 NV18 上的含义

- 硬件与拓扑 **高度契合** NVLS（NVSwitch + Hopper）。  
- 单机训练/单机 nccl-tests：开 NVLS 后 AllReduce 应优先走本节路径，**网卡空闲**。  
- 多机：节点内仍用本节机制，节点间另接 Tree 或 CollNet。  
- 若 `MULTICAST_SUPPORTED=0` 或 `NVLS_ENABLE=0`，退回 Ring/Tree@NVLink，仍快，但不是硬件多播规约。

---

## 14. 推荐配置与排查清单

### 14.1 本机推荐

```bash
# 训练网卡（排除 eth0/1）
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export NCCL_IB_GID_INDEX=3          # 按 show_gids 中 RoCEv2+IPv4 调整
export NCCL_IB_ROCE_VERSION_NUM=2

# NVLS：默认 2 即可；A/B 时用 0 对比
# export NCCL_NVLS_ENABLE=2
# export NCCL_NVLS_ENABLE=0   # 对比实验

# CollNet：确认有插件/SHARP 再开
# export NCCL_COLLNET_ENABLE=1
# 纯 RoCE 无插件时可保持 0 或省略
```

### 14.2 A/B 实验建议

| 实验 | 目的 |
|------|------|
| 单机 `NVLS=2` vs `0` | 看机内 AllReduce 带宽/延迟与 algo |
| 多机 `NVLS=2` vs `0` | 看是否出现 NVLS_TREE、机间是否仍占满 RoCE |
| `COLLNET=1` vs `0` + 查 CollNet devices | 验证插件是否真实介入 |
| 固定其它变量，只改一个开关 | 避免与 GDR/HCA/DEBUG 混杂 |

### 14.3 检查清单

- [ ] `nvidia-smi topo -m` 确认 NV18 与 PIX 配对  
- [ ] `show_gids` 确认 400G 与管理口分离  
- [ ] INIT 日志中 NVLS available / CollNet devices  
- [ ] 单机不要期望 CollNet  
- [ ] 多机 RoCE 不要假设 CollNet=1 有收益  
- [ ] 性能问题区分「机内 NVLS」与「机间 CollNet/NET/post_send」  
- [ ] 确认 CollNet 时看 `iallreduce` 路径而非仅 `COLLNET_ENABLE=1`  

---

## 15. 结论

1. **`NCCL_NVLS_ENABLE`** 控制是否启用 **NVSwitch 上的 NVLS（CUDA Multicast）路径**；在 8×H200 NV18 上，这是 **节点内算法的核心开关**（默认 2=自动探测）。  
2. **`NCCL_COLLNET_ENABLE`** 控制是否尝试 **CollNet 插件型跨节点 offload**；受插件、节点数、graph 多重否决。  
3. **节点间 CollNet**：head/rail + 插件 `iallreduce` + GPU–proxy 流水；与点对点 `ibv_post_send` 模型本质不同（见第 12 节）。  
4. **CollNet 有效前提**：除插件外，标准路径还需要 **交换机/网侧集体 offload（如 IB SHARP）**；仅插件或仅普通 RoCE 交换机通常不够（见 §12.7）。  
5. **节点内 NVLS AllReduce**：UC/MC 双映射 + heads + GPU kernel scatter/reduce/gather；不经 proxy（见第 13 节）。  
6. **单机**：CollNet 恒关；NVLS 1/0 决定多播 vs Ring/Tree@NVLink。  
7. **多机纯 RoCE 无插件/无 SHARP**：CollNet 1/0 无实质差别；NVLS 影响机内与 NVLS_TREE，机间仍是 400G NET。  
8. **多机 + 插件 + SHARP（或等价）**：CollNet 1/0 有机间路径质变；可与 NVLS 组合（机内 NVLS + 机间 CollNet）。  
9. 两个开关 **不能替代** 对 NET 路径（`ibv_post_send`、NIC 亲和、平台抖动）的分析；见 `ibv_post_send_ms_stall_analysis.md`。

```text
总结一张图:

  8×H200 NV18 + 8×400G RoCE
           │
     ┌─────┴─────┐
     ▼           ▼
  NVLS_ENABLE  COLLNET_ENABLE
  节点内多播    跨节点插件 offload
  (第13节)      (第12节)
     │           │
     │           ├─ 无插件 / 单机 → 设 1 也等于 0
     │           └─ 有 SHARP+多机 → 1 vs 0 质变
     │
     ├─ 0 → Ring/Tree@NVLink（机内）
     └─ 1/2 成功 → NVLS（机内）；多机 + NVLS_TREE 或 +CollNet
```

---

## 16. 修订记录

| 日期 | 内容 |
|------|------|
| 2026-07-10 | 初稿：两开关语义、否决链、组合矩阵与验证方法 |
| 2026-07-10 | 增补第 12 节 CollNet 节点间机制/流程/原理；第 13 节 NVLS 节点内 AllReduce 机制/流程/原理 |
| 2026-07-10 | 增补 §12.7：CollNet 除插件外还需交换机/网侧集体 offload（SHARP 等） |

---

## 附录 A：关键源码索引

| 主题 | 位置 |
|------|------|
| NVLS 参数与 init | `src/transport/nvls.cc`：`ncclNvlsInit` / `ncclNvlsTuning` / `ncclNvlsSetup` / `ncclNvlsGroupCreate` / `ncclNvlsTreeConnect` |
| CollNet env / threshold | `src/init.cc`：`CollnetEnable`、`CollNetNodeThreshold`、graph 计算与 setup |
| collNetSupport | `src/include/coll_net.h` |
| 算法 enable 矩阵 | `src/graph/tuning.cc`（约 496–511 行逻辑） |
| NVLS 图搜索条件 | `src/graph/search.cc`：`NCCL_TOPO_PATTERN_NVLS` |
| CollNet / NVLS 图连接 | `src/graph/connect.cc`：`connectCollNet` / `connectNvls` |
| CollNet 传输与 proxy | `src/transport/coll_net.cc`：`ncclCollNetSetup`、`sendProxyProgress`、`collNetIallreduce` |
| AllReduce NVLS kernel | `src/device/all_reduce.h`：`NCCL_ALGO_NVLS` |
| AllReduce CollNet Direct kernel | `src/device/all_reduce.h`：`NCCL_ALGO_COLLNET_DIRECT` |
| 参数表 | `notes/nccl_params.md` |

## 附录 B：与其它笔记的关系

| 笔记 | 关系 |
|------|------|
| `ibv_post_send_ms_stall_analysis.md` | 机间 NET 门铃尖峰；NVLS/CollNet 不能直接当根因或银弹 |
| `nccl_thread_model.md` | CollNet / NET 走 proxy；NVLS 正常路径无 proxy progress |
| `ncclIbIsend_analysis.md` | NET 数据面 isend 链路（与 CollNet `iallreduce` 对照） |
| `allreduce-channel-c8-pair-slow-analysis.md` | 含 NVLS 多播可用时的案例观察 |
| `nccl_ib_send_recv_cts_flow.md` | 普通 NET CTS 流程（CollNet 不走此路径） |
