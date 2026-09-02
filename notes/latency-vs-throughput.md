# 在 NCCL 上进一步降延迟 / 提吞吐：分层思路

> 基于 **NCCL 2.31.2** 源码与官方 tuning 文档。不是 nccl-tests 实测报告。
> 对照：`src/tuning/` 成本模型、`src/device/prims_{ll,ll128,simple}.h`、`src/enqueue/`、`src/gin/`、`src/register/`、`docs/userguide/source/troubleshooting/performance_and_tuning.rst`。
>
> NCCL 已经把延迟和吞吐拆成两套解，tuner 用 `cost ≈ latency + size/bandwidth` 在协议与算法之间切换。所谓「进一步」是认清时间花在哪一层再动那一层；两套目标经常互斥。

---

## 0. 一次 collective 的时间构成

```text
host enqueue / GroupEnd / 选算法          ← 小消息可观，大消息可忽略
cuLaunchKernelEx + kernel 启动
  GPU: 协议握手（LL flag / SIMPLE FIFO credit）
  GPU: 算法 hop（Ring ~2(N-1) step，Tree ~2 log N）
  节点内: NVLink / P2P / SHM
  跨节点: proxy + NIC（或 GIN 绕过 proxy）
  可能的 staging copy（未注册 user buffer）
```

- **延迟敏感**：小消息、TP allreduce、短 step。瓶颈多半是 hop 数、协议握手、kernel launch、proxy 往返。
- **吞吐敏感**：大 AllReduce / AllGather、DP gradient。瓶颈是 bus 带宽、SM、NIC 注入、是否零拷贝。

Cost model 已写死这种张力：

- LL 的 busBw 被压到 `llMaxBw` 且再乘约 0.5（`tuning/ring.cc:38-40`）；Tree+LL 还有 `/3.8`（`tuning/tree.cc:33-35`）。
- Tree AllReduce 延迟 `2*((ppn-1)*intraLat + log2(nNodes)*interLat)`（`tuning/tree.cc:55-57`）。
- SIMPLE 再加一份 `latencyInter` 和 `NET_OVERHEAD`（`tuning/ring.cc:12-18,63,77+`）。

经验切分（须在目标硬件上用 nccl-tests 校准）：

| 尺度 | 倾向 |
|---|---|
| 节点内 几 KB～几十 KB | LL + 少 channel；再小考虑 device API oneshot |
| 中等、多节点 AllReduce | Tree + LL128 |
| 大、吃满 NVLink/IB | SIMPLE + Ring/NVLS + 注册 |

---

## 1. 降延迟

目标：缩短 **第一次数据可见** 的时间，可以牺牲峰值 GB/s。

### 1.1 协议：GPU 自己完成握手

| 协议 | 完成语义 | 延迟 | 带宽税 |
|---|---|---|---|
| **LL** | 128-bit line 里 data+flag，`ld.volatile` 等到 flag（`prims_ll.h:108-121`） | 最低 | tuner 用 `/3.8`、`*0.5` |
| **LL128** | 128B 线里约 120B 数据 | 中 | Ring 上约 `*0.92` |
| **SIMPLE** | `NCCL_STEPS=8` FIFO + proxy credit | 最高 | 接近峰值 |

思路：

- 小消息落在 LL/LL128（`NCCL_PROTO`、tuner、`P2P_LL_THRESHOLD` 默认 16KiB，`enqueue.cc:934`）。
- 跨节点小消息避免 SIMPLE：否则必经 `SendGPUWait → PeerWait → NIC`。
- LL 热路径已有 warp 特化：`nthreads==WARP_SIZE` 只用 `__syncwarp`（`prims_ll.h:63-68`）。再压可考虑更短的 step 环、更少空转；**不要**在 `waitSend`/`readLL` 里再 stamp。

### 1.2 算法：减少 hop

- **Tree AllReduce**：~`log N` hop。NCCL 只给 AllReduce 开 Tree（`tuning/tree.cc:22-26`）。
- **PAT**（`tuning/pat.cc`）：AllGather / ReduceScatter 的延迟型路径；多节点 + NVLS 时默认可能开（`PAT_ENABLE=2`）。
- Ring 的 `nSteps` 随 `nRanks` 线性涨，大消息才划算。

可做：更准的 small-size Tree/PAT 切换；单节点极小消息 **NVLS oneshot**（`contrib/custom_algos` 有参考），绕过 ring 的 N-1 步。

### 1.3 砍 host / launch 税

单次 AllReduce 也会隐式 `GroupStart/End`（`enqueue.cc:3396`）。小消息上这段可与 GPU 时间同量级。

- CUDA Graph：迭代里只 `cudaGraphLaunch`（`GRAPH_REGISTER`，`enqueue.cc:331`）。
- 用户把多个小 op 放进同一 group，摊掉一次 prepare。
- `NCCL_LAUNCH_MODE=GROUP` 用 `intraBarrier` 锁步 clique，利于 overlap，但可能增加单 collective 的 host 等待；延迟基准常对照非 GROUP。
- Device API 融合（`docs/examples/07_kernel_fusion`、`contrib/nccl_ubx`）：通信不再单独 `cuLaunchKernelEx`，消掉一次 launch + 一次全局 barrier。对「训练 step 暴露延迟」往往比再抠 LL 更有效。
- Symmetric / LSA：节点内 load/store + barrier，无 proxy。

### 1.4 跨节点：拿掉 proxy 一跳

- GDR + user buffer 注册：少 GPU↔host staging。
- **GIN / GDAKI**（`src/gin/`、`src/transport/net_ib/gdaki`）：GPU 写 WQE，不经 proxy progress。
- `GDRCOPY_FLUSH_ENABLE`：收端 CPU 可见更快，是否默认要按平台测。
- Proxy 贴 NIC 的 NUMA（`NCCL_PROXY_CPUSET`、官方 affinity 文档）。错 NUMA 直接进尾延迟。

### 1.5 拓扑

- 同节点走 P2P/NVLink，避免 SHM bounce（`P2P_DIRECT_DISABLE`、`SHM_DISABLE`）。
- 小消息不要默认上 NVLS_TREE/CollNet 长链路：NVLS 在 tuner 里绑 SIMPLE（`tuning/nvls.cc:28-30`），小消息未必赢 LL+Tree。

**延迟方向不该做的**：盲目加 CTA、加 QP、加 chunk——那是吞吐旋钮，会拉长启动、多占 SM。

---

## 2. 提吞吐

目标：接近 NVLink/NIC 峰值 busBw，可以多占 SM、多 pipeline。

### 2.1 协议回到 SIMPLE，打满 pipeline

`NCCL_STEPS=8` 的 SIMPLE 是带宽协议：slice/chunk 流水，GPU 与 proxy 重叠。大消息应离开 LL（tuner 已封顶 `llMaxBw`）。

- `NCCL_CHUNK_SIZE`：Send/Recv、AlltoAll 偏大可改善「带宽被延迟限制」（官方 tuning 文档）。
- 更多 channel / CTA（`NCCL_MIN_CTAS` / `MAX_CTAS`）：平行搬运，代价是抢计算 SM。
- LL128 作中间档。

### 2.2 算法：NVLS / CollNet / Ring

- 单节点大消息：**NVLS**（Hopper+ multicast；`nvls.cc` efficiency Hopper 0.85 / Blackwell 0.74）。要 `nChannels>=2`，且 NVLS/NVLS_TREE 只配 SIMPLE。
- 多节点大 AllReduce：NVLS_TREE 或 CollNet Direct/Chain（依赖 NIC SHARP 一类）。
- Ring：大消息 busBw ≈ 链路 × `nRanks/nSteps`。
- 大 AllGather：**CE**（`SYM_CE_THRESHOLD` 默认 8MB，`enqueue.cc:34`），把 SM 还给计算。

### 2.3 零拷贝：注册是第一杠杆

未注册时 SIMPLE 常走 staging FIFO，多一次 copy。

- `ncclCommRegister`（`src/register/`，`LOCAL_REGISTER`）
- 对称窗 `ncclCommWindowRegister`：NVLS / GIN / device API 的前提
- NIC GDR + `ibv_reg_mr`
- Graph 内 `GRAPH_REGISTER`

### 2.4 网络注入

官方点名（`performance_and_tuning.rst`）：

- `NCCL_IB_QPS_PER_CONNECTION`：ECMP / 多端口 / RoCE 上加 QP
- `NCCL_CROSS_NIC`：ring/tree 是否跨 NIC
- rail 对齐：GPU–NIC 1:1
- `NET_SHARED_BUFFERS` / `NET_SHARED_COMMS`

代码向：proxy `progress` 少空转、`ibv_post_send` 批量、WQE 深度与 `NCCL_STEPS` 匹配。再优化应 **加大并行度**，不要在 step 状态机上再插一层。

### 2.5 有效吞吐：和计算重叠

训练看 step 时间，不是 nccl-tests 的 busBw。

- 通信用 **更少 CTA**，给 GEMM 留 SM（与「加 CTA 提 NCCL busBw」相反）。
- Device API 融合（UB-X、PACE、`07_kernel_fusion`）：RS/AG 与 RMSNorm 同 kernel。
- CUDA Graph + 多 stream。
- GIN 对称 AG/RS：kernel 内 put，host 不再串行 launch 两次 collective。

### 2.6 拓扑与 channel 数

`src/graph/` 决定 `nChannels` 和 intra/inter bw。吞吐差常见于 channel 太少或环跨了差链路。`NCCL_MIN_NCHANNELS` / 拓扑 XML / `NCCL_ALGO` 可对症；长期让 tuner 吃实测而不是表常数。

---

## 3. 对照表

```text
小消息延迟                    大消息吞吐
  LL / LL128  Tree / PAT         SIMPLE  Ring / NVLS / CollNet
  少 CTA、少 hop                  多 channel、注册、GDR
  绕过 proxy（GIN / LL）          打满 proxy / NIC pipeline
  融合 / Graph 去 launch          CE 把 SM 还给计算
```

| 层 | 延迟 | 吞吐 |
|---|---|---|
| 热路径 | 复用 LL flag，不要第二套 stamp | 复用 SIMPLE FIFO，不要每 slice 打日志 |
| 最值得改的代码 | enqueue/launch 税、Tree/PAT 切换、GIN 小消息、融合 kernel | 注册/GDR 覆盖率、NVLS 效率、chunk/QP、CE 阈值 |
| 最值得先调的环境 | `PROTO=LL`、affinity、Graph | `MIN_CTAS`、`CHUNK_SIZE`、`IB_QPS_PER_CONNECTION`、register |
| 和计算的关系 | 融合掉一次 barrier+launch | 少占 SM、和 GEMM overlap |

在本仓库加功能：延迟优先碰 **enqueue/launch 与 LL/GIN/融合**；吞吐优先碰 **register、NVLS、net_ib 注入和 CTA/chunk**。两边都不要在 `prims_*.h` 的 wait/post 里再加工作——那是已经付过代价的握手。

官方也声明：生产上不要随便锁死 `NCCL_MIN_CTAS` 等（`performance_and_tuning.rst` Tuning 节）；一处基准的增益常在别处变成回退。改代码前用 nccl-tests 在目标规模上把 LL/Tree vs SIMPLE/Ring/NVLS 的交叉点测出来，再决定动哪一层。
