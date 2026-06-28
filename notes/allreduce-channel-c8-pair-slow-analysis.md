# NCCL All-Reduce「c / c+8 channel 成对变慢」定位笔记

> 场景：单机 8 卡 × 16 机器 = 128 rank 的分布式训练，PyTorch DDP，底层 all-reduce 通信。
> 每个 all-reduce kernel 下有 16 个 channel；发现**某些** all-reduce（不是全部）会让
> **所有 rank** 同时变慢。进一步观察：慢的 channel 总是**偶数个**（如 2 个或 4 个），
> 且慢的 channel 总是 **`c` 与 `c+nChannels/2`（即 `c+8`）成对出现**。
>
> 本文记录问题背景、代码层结论、日志解读、诊断地图与根因候选。
> 配套可读：[`allreduce-training-slow-analysis.md`](./allreduce-training-slow-analysis.md)
> 与 [`docs/dev_guide/nccl_internals_indepth.md`](../docs/dev_guide/nccl_internals_indepth.md)。

---

## 1. 现象摘要

| 维度 | 观察 |
|------|------|
| 集群 | 128 rank（16 节点 × 8 卡），PyTorch DDP，all-reduce |
| 每个 AR kernel | 16 个 channel（channelId 0..15） |
| 慢的 collective | **部分** all-reduce；慢的 collective 上 **所有 rank** 同时变慢 |
| 慢的 channel 数 | 总是**偶数**（实测 2 或 4） |
| 慢的 channel 编号 | 总是 **`c` 与 `c+8` 成对**（nChannels/2 = 8） |

核心待解问题：**NCCL 在划分 channel 时，`c` 和 `c+8` 有什么共性？为何瓶颈恰好成对出现？**

---

## 2. 结论速览

1. **`c` 与 `c+8` 是「克隆对」**：NCCL 在建 communicator 时有一道 channel 翻倍
   （duplication）步骤，把每条物理通道克隆成 2 个逻辑 channel。8 条物理 ring
   翻倍为 16，产生偏移 = 8 的配对。
2. **克隆对是镜像环**：相同环序、相同 prev/next 邻居、相同方向（单向环 prev→me→next）、
   相同物理 inter-node 路径（同一张 NIC）。
3. **镜像的是「路由」，不是「数据」**：两个 channel 各搬不同的数据 slice、各自独立状态
   （buffer/head/tail/connFifo/workCounter/proxy op 互不依赖）。
4. **瓶颈是物理共享，不是逻辑互锁**：一对 channel 同向压同一条物理 NIC 路径；
   该 NIC 变窄时，其上 2 个克隆 channel **同时**变慢 → 表现为偶数个、成对。
5. **「全 rank 一起慢」由 PXN 解释**：每张 NIC 的跨节点流量全部汇聚到它的 home GPU，
   故一条 NIC 瓶颈会同时体现在本机 8 个 rank 上；而 AR kernel 要等全部 16 channel 到齐，
   最慢的一对就拖垮整个 kernel。
6. **真正要修的是物理层**（某张 NIC 的链路/速率/GDR/NUMA），不是 NCCL 本身。

---

## 3. 代码层证据（推理依据）

### 3.1 channel 翻倍产生 `(c, c+nChannels)` 克隆对

`src/graph/connect.cc:447-460` —— ring 的 prev/next 克隆：

```c
// Duplicate ringPrev/ringNext for ncclBuildRing
memcpy(ringPrev + nChannels * nranks, ringPrev, nChannels * nranks * sizeof(int));   // 448
memcpy(ringNext + nChannels * nranks, ringNext, nChannels * nranks * sizeof(int));   // 449

// Set ring prev/next for my rank
for (int c = 0; c < nChannels; c++) {
  struct ncclChannel* channel0 = comm->channels + c;
  struct ncclChannel* channel1 = channel0 + nChannels;                              // 454 ★
  channel0->ring.prev = channel1->ring.prev = ringPrev[c * nranks + comm->rank];     // 455
  channel0->ring.next = channel1->ring.next = ringNext[c * nranks + comm->rank];     // 456
}
nChannels = comm->nChannels = std::min(MAXCHANNELS, nChannels * 2);                 // 460 ★
```

`c` 与 `c+nChannels` 被赋了**完全相同**的 prev/next，随后 `nChannels *= 2`。
翻倍后新 nChannels=16，原偏移 `nChannels`(=8) 即为 `nChannels/2` → **`(c, c+8)` 配对**。

### 3.2 拓扑层路径同样克隆（保证物理路径相同）

`src/graph/search.cc:1036-1048` —— `ncclTopoDupChannels`：

```c
int dupChannels = std::min(graph->nChannels * 2, graph->maxChannels);                 // 1042
memcpy(graph->intra + graph->nChannels * ngpus, graph->intra, ...);                  // 1043 ★ intra 路径复制
memcpy(graph->inter + graph->nChannels * 2, graph->inter, ...);                      // 1044 ★ inter 路径复制
graph->nChannels = dupChannels;                                                       // 1047
```

`intra[]`（节点内 GPU 顺序）与 `inter[]`（节点间路径）被原样拷贝 → 克隆对不仅 ring 邻居相同，
**物理路径也相同**。

### 3.3 后续翻倍同样用克隆手法（`copyChannels`）

`src/graph/connect.cc:361-366` + 调用处 `481-483`：

```c
static int copyChannels(struct ncclComm* comm, int start, int end, int* ringPrev, int* ringNext) {
  ...
  memcpy(ringPrev + c * nranks, ringPrev + (c - start) * nranks, nranks * sizeof(int));  // 365
  memcpy(ringNext + c * nranks, ringNext + (c - start) * nranks, nranks * sizeof(int));  // 366
}
// connect.cc:481-483 — sm90+ 多节点，若 nChannels<16 再翻一倍到 16
if (comm->minCompCap >= 90 && comm->nNodes > 1 && graphs[NCCL_ALGO_RING]->bwIntra > 45.0 && nChannels < 16) {
  nChannels = comm->nChannels = copyChannels(comm, nChannels, 2 * nChannels, ringPrev, ringNext);  // 482
}
```

`copyChannels` 把 `[0, start)` 拷到 `[start, 2*start)`，即 channel `c+start` ≡ channel `c`。
**最后一次 8→16 翻倍产生的就是 offset=8 的克隆对 `(c, c+8)`**。

### 3.4 基数 8 的来源（NIC 数）

`src/graph/paths.cc:947-952`：

```c
int nNetChannels = comm->config.nChannelsPerNetPeer;
...
NCCLCHECK(ncclTopoGetLocalNetCountByBw(system, g, &netCount, &netBw));   // netCount = 本节点数据面 NIC 数
nNetChannels = 2;
if (netCount > 0) nNetChannels = std::max(netCount, divUp((int)netBw, (int)ncclParamP2pPerChannelNetBw()));
```

本场景 8 张数据面 NIC → 基数 8 → 翻倍到 **16**。每条物理路径对应 2 个克隆 channel = `(c, c+8)`。

### 3.5 不要和 crossNic 的 even/odd 混淆

`src/graph/connect.cc:405-411` 另有一个 **even/odd 交换**（`c ^ 1`，即 (0,1)(2,3)...），
用于把相邻 channel 摊到不同 NIC。**其偏移是 1，不是 8**。本场景的成对慢是克隆对（偏移 8），
不是这个。

### 3.6 翻倍是全算法的共性

tree 算法同理：`src/graph/connect.cc:149-151`、日志 `170` 直接打印
`Tree %d : ... c + nChannels`。所以 ring/tree allreduce 都有 `(c, c+8)` 配对 ——
成对慢在两种算法下都会复现。

---

## 4. 克隆对的方向：相同，且单向

### 4.1 方向由 prev/next 决定，克隆对二者相同

环的方向完全由 `ring.prev/ring.next` 决定（`rings.cc:29` 的 `ncclBuildRings` 只验证环闭合，
不引入方向反转）。克隆对 prev/next 相同（§3.1），故方向不可能不同。

### 4.2 allreduce 是单向环（prev→me→next）

`src/device/all_reduce.h:31`，`Primitives` 构造参数顺序为 `(recvPeers, sendPeers, …)`：

```c
Primitives<…> prims(tid, nthreads, &ring->prev, &ring->next, …);   // recvPeers=prev, sendPeers=next
```

随后整个 ring allreduce（`all_reduce.h:34-82`）始终同一方向步进：

```c
prims.directSend(offset, offset, nelem);                    // step 0: 发 next
prims.directRecvReduceDirectSend(offset, offset, nelem);    // k-2 步: 收 prev、reduce、发 next
prims.directRecvReduceCopyDirectSend(...);                  // step k-1
prims.directRecvCopyDirectSend(offset, offset, nelem);      // k-2 步: 收 prev、发 next
prims.directRecv(offset, nelem);                            // 最后: 收 prev
```

数据流始终 **`prev → me → next`**。reduce-scatter 与 allgather 靠 chunk 偏移
`modRanks(ringIx + nranks - j)`（43/51/68 行）在**同一个单向环里**合并完成，**没有反向第二圈**。

### 4.3 全代码库无反向/双向 ring

搜索 `reverse / bidirect / flushP / nChannels/2 / recvPeers=next / sendPeers=prev`，
在 `all_reduce.h`、`primitives.h`、`prims_ll.h`、`prims_ll128.h`、`run_coll.cc` 中
**全部零命中**。NCCL 的 ring 是纯单向的，靠增加 channel 数（含克隆）来弥补带宽损失，
**不做双向 ring**。

### 4.4 方向相同正是成对慢的成因之一

`c` 与 `c+8` 同环、同向、同路径 → 在**同一条物理链路上是同向并行的两个数据流**。
不会对冲（方向一致），但会**共享/竞争该 NIC 的出口带宽**：

- 正常：NIC 带宽充足，两个克隆 channel 各跑各的，靠"路径复用 × 2"堆并发吃满带宽（翻倍本意）。
- 异常：NIC 带宽打折 → 该路径变窄 → 两个克隆 channel **同时**变慢 → 偶数个、成对。

---

## 5. 「镜像」的边界：路由相同，数据/状态独立

| 维度 | c 与 c+8 是否相同 |
|------|------------------|
| 环序（intra[]）、prev/next、方向 | ✅ 完全相同（镜像） |
| 物理 NIC / PCIe/NUMA 路径 | ✅ 完全相同（同一条） |
| 携带的数据 chunk | ❌ **不同**（各自一条带，见 `enqueue.cc:576` scheduleCollTasksToPlan） |
| 执行状态（buffer/head/tail/connFifo/workCounter/proxy op） | ❌ **独立**（`comm.h:534` channels[MAXCHANNELS] 各项独立） |

**两个 channel 之间没有任何同步依赖**：c 慢不会直接让 c+8 卡住，反过来也是。
成对慢的成因是**第三方的共同瓶颈**（共享 NIC 变窄），不是镜像关系本身的逻辑互锁。
这个区分很重要：若是逻辑耦合（共享 counter/buffer）应在 NCCL 内部找 bug；
既然是物理耦合（共享 NIC），只能去硬件找。

---

## 6. 日志解读（rank 26 视角，NCCL 2.29.3 + CUDA 12.9）

### 6.1 环境与硬件画像

| 项 | 值 | 解读 |
|---|---|---|
| NCCL | 2.29.3 + CUDA 12.9, driver 12080 | 正常 |
| rank | 26, cudaDev 2, busId `4c000`, 32 ranks | rank 26 的视角 |
| 网络 | **RoCE**（非 IB），9 张 mlx5 | 数据面 8 张 + mlx5_0 未用 |
| `NCCL_IB_PCI_RELAXED_ORDERING=2` | RO 强制开 | ✅ 利好 GDR |
| `NCCL_IB_ADAPTIVE_ROUTING=1` | AR 开 | ✅ |
| `NCCL_CROSS_NIC=2` | 强制 cross-NIC | 但 pattern 落 `crossNic 0`（见 §6.5） |
| `NCCL_IGNORE_CPU_AFFINITY=1` | NCCL 不绑 CPU 亲和 | ⚠️ 风险点（见 §7.3） |
| `NCCL_NVLS_ENABLE=1` | NVLS 多播可用, 16 ch | ✅ |
| `NCCL_SOCKET_IFNAME=eth0` | OOB/bootstrap 走 eth0 | ✅ 与数据面分离 |

### 6.2 ⚠️ 硬件异常一：mlx5_0 是 100G，其余 8 张 400G

```
[0] mlx5_0 ... speed=100000    ← 100G，异常
[1..8] mlx5_2..mlx5_9 ... speed=400000
```

**但好消息**：NCCL 建环时**完全没用 mlx5_0**（见 §6.4）。所以它不是 c/c+8 变慢的直接元凶，
但它是红旗。**`speed=` 是标称速率，不等于实际协商速率** —— 8 张"400G"的实际链路速率
必须用 `ibstat` 逐张核对，任何一张掉到 100G/200G 就会正好表现为「一对 channel 慢」。

### 6.3 ⚠️ 硬件异常二：libmlx5 偏旧

```
dlvsym failed on mlx5dv_get_data_direct_sysfs_path   ← Data Direct 不可用
dlvsym failed on mlx5dv_reg_dmabuf_mr                ← dmabuf MR 注册不可用
```

`/usr/lib/.../libmlx5.so` 缺符号，**libmlx5/rdma-core 版本旧于 NCCL 期望**。后果：GDR 的
dmabuf 注册回退到普通路径（更慢、可能不稳），Data Direct 低延迟优化拿不到。**全节点统一影响**，
不会单独拖某对 channel，但会整体抬高 baseline，让单点瓶颈更显眼。建议升级到与 NCCL 2.29 匹配版本。

### 6.4 拓扑结构（NUMA + NIC + home GPU）

树形日志把 NUMA 结构讲得很清楚。8 张**数据面 NIC** 与各自"home GPU"：

| NUMA | NIC 设备号 | home GPU (rank/bus) | PCI 根 |
|------|-----------|---------------------|--------|
| **0** (CPU/0-0) | mlx5_2 → NET/0-1 | rank 24 / `2a000` | PCI/0-27000 |
| 0 | mlx5_3 → NET/0-2 | rank 25 / `3a000` | PCI/0-38000 |
| 0 | mlx5_4 → NET/0-3 | rank 26 / `4c000` | PCI/0-49000 |
| 0 | mlx5_5 → NET/0-4 | rank 27 / `5c000` | PCI/0-5a000 |
| **1** (CPU/0-1) | mlx5_6 → NET/0-5 | rank 28 / `aa000` | PCI/0-a8000 |
| 1 | mlx5_7 → NET/0-6 | rank 29 / `ba000` | PCI/0-b8000 |
| 1 | mlx5_8 → NET/0-7 | rank 30 / `ca000` | PCI/0-c8000 |
| 1 | mlx5_9 → NET/0-8 | rank 31 / `da000` | PCI/0-d8000 |
| 0 | mlx5_0 → NET/0-0 | （无 home GPU，100G，未使用） | — |

典型 8 GPU : 8 NIC 全互联（HGX Hopper 级，NVSwitch 全互联）。**rank 24-27 在 NUMA 0，
rank 28-31 在 NUMA 1**。

**GDR/PXN**：日志中每张 NIC 恰好绑一张 home GPU（distance=4 的那张），其余 7 个 rank 全部
经 PXN 走过去（如 NET/0-1 home=rank 24；rank 25/26/27/28/29/30/31 全 PXN 经 rank 24）。
NET/0-0（mlx5_0）在 GDR 列表里**未出现** → 确认被排除在数据面之外。

**PXN 语义**（`src/transport/net.cc:314`、`src/graph/paths.cc:515-647`）：某 rank 不直连某 NIC 时，
先把数据用 **NVLink** 发到 home GPU，由 home GPU 做 GPUDirect RDMA 写到网卡。
`comm->useNetPXN = true` 由此置上。

### 6.5 channel → NIC 映射（诊断地图）

Pattern 4 = RING（`src/include/graph.h:167`），nChannels 8，逐行 8 条环：

```
ch 0: NET/0-1 ... = mlx5_2  (home rank 24, NUMA 0)
ch 1: NET/0-2 ... = mlx5_3  (home rank 25, NUMA 0)
ch 2: NET/0-3 ... = mlx5_4  (home rank 26, NUMA 0)
ch 3: NET/0-4 ... = mlx5_5  (home rank 27, NUMA 0)
ch 4: NET/0-5 ... = mlx5_6  (home rank 28, NUMA 1)
ch 5: NET/0-6 ... = mlx5_7  (home rank 29, NUMA 1)
ch 6: NET/0-7 ... = mlx5_8  (home rank 30, NUMA 1)
ch 7: NET/0-8 ... = mlx5_9  (home rank 31, NUMA 1)
```

经 `connect.cc:481-483` 的 sm90+ 翻倍（8→16）产生 `(c, c+8)` 克隆对：

| 克隆对 (c, c+8) | 物理 NIC | home GPU | NUMA |
|------|------|------|------|
| **(0, 8)** | mlx5_2 / NET/0-1 | rank 24 | 0 |
| **(1, 9)** | mlx5_3 / NET/0-2 | rank 25 | 0 |
| **(2, 10)** | mlx5_4 / NET/0-3 | rank 26 | 0 |
| **(3, 11)** | mlx5_5 / NET/0-4 | rank 27 | 0 |
| **(4, 12)** | mlx5_6 / NET/0-5 | rank 28 | 1 |
| **(5, 13)** | mlx5_7 / NET/0-6 | rank 29 | 1 |
| **(6, 14)** | mlx5_8 / NET/0-7 | rank 30 | 1 |
| **(7, 15)** | mlx5_9 / NET/0-8 | rank 31 | 1 |

**这张表是诊断地图**：把 profiling 观测到的"慢 channel 对编号"填进去，即可直接定位到
具体的 mlx5_NIC、home rank、所属 NUMA。例如：
- 慢 (2,10) → mlx5_4 / home rank 26（即本机发日志的 rank 26 自己，本地重点查）；
- 慢 (4,12) → mlx5_6 / NUMA 1 / home rank 28。

### 6.6 次要点

1. **`NCCL_CROSS_NIC=2` 与 `crossNic 0` 不矛盾**：cross-NIC 只在每条环跨多种 NIC 类型时才有意义；
   本场景每条环只用一张 NIC、且只有一种 NIC 类型（mlx5，忽略 mlx5_0），所以 `crossNic` 没东西可 cross，落到 0。非问题。
2. **Pattern 3 vs Pattern 4 都跑出**：
   - Pattern 4 (RING)：`bw 24/24, type NVL/PXN` —— inter 走 PXN。
   - Pattern 3 (TREE)：`bw 48/24, type NVL/PIX` —— bwIntra 更高、inter 更近。
   
   两者都各 8 ch、都会克隆到 16，故 `(c, c+8)` 配对在 RING 和 TREE 下都成立 —— 成对慢两种算法都复现。
3. **MNNVL**：`MNNVL busId 0x4c000 fabric UUID ... state 3` —— 检测到 Multi-Node NVLink fabric。
   若集群确有跨节点 NVLink 而当前 allreduce 走 RoCE，可考虑让拓扑参与以分摊 NIC 压力。
4. **NVLS 多播可用**：`NVLS multicast support is available on dev 2 (NVLS_NCHANNELS 16)`。
   单节点内小-中消息 allreduce 可能走 NVLS（NVSwitch 多播，不经 NIC），不受 NIC 瓶颈影响；
   你慢的是大消息走 RING+RoCE 的路径，与 NVLS 无关。

---

## 7. 根因候选（按概率排序）

### 7.1 NIC 实际链路速率掉了
报告 400G，协商可能 200G/100G。→ 逐张 `ibstat` 核对 `Rate` / `State` / `Physical state`。
重点看与慢对对应的那张。

### 7.2 home GPU 的 NVLink/GPU 本身异常（PXN 汇聚点）
那张 GPU 既要算自己的、又要替另外 7 个 rank 转发，最易成为热点。
→ `nvidia-smi nvlink -s` / 该卡 ECC 降速 / 过热降频。

### 7.3 NUMA/CPU 亲和错位 ⚠️
`NCCL_IGNORE_CPU_AFFINITY=1` 意味着 NCCL 不绑 proxy 线程。若训练进程把 CPU 绑死在某个 NUMA，
proxy 线程可能跨 NUMA 访问 NIC 缓冲区 → 特定 NUMA 侧的 channel 对变慢。
**判据**：若慢的几对集中在 NUMA 0（0-3, 8-11）或 NUMA 1（4-7, 12-15）的一侧，高度怀疑此项。
→ 去掉 `NCCL_IGNORE_CPU_AFFINITY=1`，或给训练进程正确的 `numactl` / `--cpus-per-gpu` 亲和。

### 7.4 RoCE/ECMP/AR 路径抖动
AR 开着但若交换机侧 hash 或 PFC 配置不均，某张卡的特定流可能持续丢包/重传，
表现为该对 channel 偶发慢。→ `ibquerycounter` / `ethtool -S` 看丢包与重传计数。

---

## 8. 行动清单（按优先级）

1. **回报慢的 channel 对编号**（从 profiling 工具看是哪几个 c），对照 §6.5 诊断地图，
   直接定位 mlx5 名 + home rank + NUMA。
2. **逐张 `ibstat`**：确认 8 张数据面 NIC 的 `Rate` 真为 400G、`State: Active`、
   `Physical state: LinkUp`。重点看慢对对应的那张。
3. **逐张 `ibquerycounter` / `ethtool -S`**：看慢对对应 NIC 的 `rx/tx_errors`、`retry`、
   `ecn`、PFC 丢包。
4. **去掉 `NCCL_IGNORE_CPU_AFFINITY=1`**，或给训练进程正确的 NUMA 亲和，
   让 proxy 线程落在对应 NUMA。
5. **升级 libmlx5 / rdma-core**，消除两个 `dlvsym failed`（dmabuf MR + Data Direct），
   恢复 GDR 最优路径。
6. **临时验证** `NCCL_MAX_NCHANNELS=8`：只用 8 个非克隆 channel，看慢对是否消失。
   - 若消失 → 确属克隆对的物理共享瓶颈；
   - 若仍慢 → 单 channel 即慢，问题更深（须查该 NIC 单路径）。

### 兜底调参（治标，硬件修不了时临时用）
- `NCCL_IB_HCA=` / `NCCL_NET_FORCE_MERGE`：排除或合并问题 NIC。
- `NCCL_MAX_NCHANNELS=8`：每条物理路径只跑 1 个 channel，消除克隆对相互影响（总带宽减半）。
- `NCCL_CROSS_NIC=0/1`：改变 even/odd 在 NIC 上的摊布，可能让坏路径不被选中。
- `NCCL_ALGO=TREE` 或 `RING`：换算法看是否同样卡同一对（都受同一克隆结构影响，但步数/扇出不同，
  瓶颈点可能转移）。

---

## 9. 关键文件速查

| 主题 | 文件:行 |
|------|---------|
| ring 翻倍产生 (c, c+nChannels) | `src/graph/connect.cc:447-460` |
| `copyChannels` 克隆 | `src/graph/connect.cc:361-366, 481-483` |
| 拓扑路径克隆 (`ncclTopoDupChannels`) | `src/graph/search.cc:1036-1048` |
| crossNic even/odd 交换（偏移 1，非本次） | `src/graph/connect.cc:405-411` |
| 基数 = NIC 数 | `src/graph/paths.cc:947-952` |
| ring 单向、prev→me→next | `src/device/all_reduce.h:31, 34-82` |
| 环闭合校验（不反转方向） | `src/graph/rings.cc:29` |
| PXN 置位 | `src/transport/net.cc:314`、`src/graph/paths.cc:515-647` |
| Pattern 枚举（RING=4, TREE=3, NVLS=5） | `src/include/graph.h:160-168` |
| 通道条带化（数据 slice 各不同） | `src/enqueue.cc:576` |
| channel 独立结构 | `src/include/comm.h:534` |

---

## 10. 一句话结论

`c` 与 `c+8` 不是巧合 —— 它是 `connect.cc:447-460` / `search.cc:1042-1047` 的 channel
翻倍逻辑制造的**克隆对**：同环、同向（单向 prev→me→next）、同条物理 inter-node 路径
（同一张 NIC），但数据与状态各自独立。「偶数个、成对、卡住全 rank」是单条物理 NIC 路径
瓶颈在克隆结构下的标准指纹，叠加 PXN 把一张 NIC 的流量汇聚到 home GPU、AR kernel 要全 channel
到齐，于是表现为全 rank 同时慢。真正要修的是那一两张 NIC 的链路/速率/GDR/NUMA 摆放，
不是 NCCL 本身。
