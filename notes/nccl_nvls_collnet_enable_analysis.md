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
12. [推荐配置与排查清单](#12-推荐配置与排查清单)
13. [结论](#13-结论)
14. [修订记录](#14-修订记录)

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
3. 默认 `NVLS_ENABLE=2` 已是自动探测；显式写 `1` 是强制，写 `0` 才是彻底关闭。  
4. 多机训练里的 `ibv_post_send` 热路径属于 **NET 传输**；仅开 NVLS/CollNet **不保证** 消除 RoCE 门铃尖峰。

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

## 12. 推荐配置与排查清单

### 12.1 本机推荐

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

### 12.2 A/B 实验建议

| 实验 | 目的 |
|------|------|
| 单机 `NVLS=2` vs `0` | 看机内 AllReduce 带宽/延迟与 algo |
| 多机 `NVLS=2` vs `0` | 看是否出现 NVLS_TREE、机间是否仍占满 RoCE |
| `COLLNET=1` vs `0` + 查 CollNet devices | 验证插件是否真实介入 |
| 固定其它变量，只改一个开关 | 避免与 GDR/HCA/DEBUG 混杂 |

### 12.3 检查清单

- [ ] `nvidia-smi topo -m` 确认 NV18 与 PIX 配对  
- [ ] `show_gids` 确认 400G 与管理口分离  
- [ ] INIT 日志中 NVLS available / CollNet devices  
- [ ] 单机不要期望 CollNet  
- [ ] 多机 RoCE 不要假设 CollNet=1 有收益  
- [ ] 性能问题区分「机内算法」与「机间 NET/post_send」

---

## 13. 结论

1. **`NCCL_NVLS_ENABLE`** 控制是否启用 **NVSwitch 上的 NVLS（CUDA Multicast）路径**；在 8×H200 NV18 上，这是 **节点内算法的核心开关**（默认 2=自动探测）。  
2. **`NCCL_COLLNET_ENABLE`** 控制是否尝试 **CollNet 插件型跨节点 offload**；受插件、节点数、graph 多重否决。  
3. **单机**：CollNet 恒关；NVLS 1/0（或 2 探测成败）决定能否用 NVLS 多播 vs Ring/Tree@NVLink。  
4. **多机纯 RoCE 无插件**：CollNet 1/0 **无实质差别**；NVLS 影响机内与 NVLS_TREE，**机间仍是 400G NET**。  
5. **多机 + SHARP/CollNet**：CollNet 1/0 才有机间路径质变；可与 NVLS 组合。  
6. 两个开关 **不能替代** 对 NET 路径（`ibv_post_send`、NIC 亲和、平台抖动）的分析；见 `ibv_post_send_ms_stall_analysis.md`。

```text
总结一张图:

  8×H200 NV18 + 8×400G RoCE
           │
     ┌─────┴─────┐
     ▼           ▼
  NVLS_ENABLE  COLLNET_ENABLE
  节点内多播    跨节点插件 offload
     │           │
     │           ├─ 无插件 / 单机 → 设 1 也等于 0
     │           └─ 有 SHARP+多机 → 1 vs 0 质变
     │
     ├─ 0 → Ring/Tree@NVLink（机内）
     └─ 1/2 成功 → NVLS（机内）；多机 + NVLS_TREE 或 +CollNet
```

---

## 14. 修订记录

| 日期 | 内容 |
|------|------|
| 2026-07-10 | 初稿：结合 H200+400G 拓扑与 init/nvls/coll_net/tuning 源码，完整说明两开关语义、否决链、组合矩阵与验证方法 |

---

## 附录 A：关键源码索引

| 主题 | 位置 |
|------|------|
| NVLS 参数与 init | `src/transport/nvls.cc`：`ncclNvlsInit` / `ncclNvlsTuning` / `ncclNvlsTreeConnect` |
| CollNet env / threshold | `src/init.cc`：`CollnetEnable`、`CollNetNodeThreshold`、graph 计算与 setup |
| collNetSupport | `src/include/coll_net.h` |
| 算法 enable 矩阵 | `src/graph/tuning.cc`（约 496–511 行逻辑） |
| NVLS 图搜索条件 | `src/graph/search.cc`：`NCCL_TOPO_PATTERN_NVLS` |
| CollNet 图连接 | `src/graph/connect.cc`：`connectCollNet` |
| CollNet 传输 | `src/transport/coll_net.cc` |
| AllReduce NVLS kernel | `src/device/all_reduce.h`：`NCCL_ALGO_NVLS` |
| 参数表 | `notes/nccl_params.md` |

## 附录 B：与其它笔记的关系

| 笔记 | 关系 |
|------|------|
| `ibv_post_send_ms_stall_analysis.md` | 机间 NET 门铃尖峰；NVLS/CollNet 不能直接当根因或银弹 |
| `nccl_thread_model.md` | CollNet / NET 走 proxy；NVLS 正常路径无 proxy progress |
| `ncclIbIsend_analysis.md` | NET 数据面 isend 链路 |
| `allreduce-channel-c8-pair-slow-analysis.md` | 含 NVLS 多播可用时的案例观察 |
