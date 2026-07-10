# NCCL `ibv_post_send` 毫秒级耗时分析笔记

> 状态：分析结论（含计时方法论纠偏）  
> 场景：分布式训练中，`src/transport/net_ib/p2p.cc` 的 `ncclIbMultiSend` / `wrap_ibv_post_send` 出现数 ms～数十 ms 尖峰  
> 相关代码路径：`ncclIbIsend` → `ncclIbMultiSend` → `wrap_ibv_post_send` → `ibv_post_send`  
> 相关既有笔记：`ncclIbIsend_analysis.md`、`nccl_ib_send_recv_cts_flow.md`

---

## 1. 问题现象

### 1.1 业务侧观察

- 分布式训练过程中，怀疑 IB 发送路径过慢。
- 在 `wrap_ibv_post_send` 前后打点，观测到 **数 ms～40ms+** 的墙钟时间。
- 初期认为「`ibv_post_send` API 本身就要这么久」。

### 1.2 预期基线（正常 mlx5 用户态 verbs）

| 操作 | 正常量级 |
|------|----------|
| `ibv_post_send`（写 WQE + UAR doorbell） | **亚微秒～数微秒～十余 μs** |
| `ncclIbMultiSend` 填 WR / 选 QP | **通常 < 10～几十 μs** |
| 真正的网络 RTT / 对端完成 | 在后续 `ibv_poll_cq` / `ncclIbTest`，**不在** `post_send` 返回路径上 |

`ibv_post_send` 语义是 **异步提交**，不负责等数据到达对端。稳定出现 **数 ms** 属于异常量级（约 10³～10⁴ 倍）。

### 1.3 调用链（数据面）

```text
ncclProxyProgress
  └── sendProxyProgress          [src/transport/net.cc]
        └── ncclNet->isend
              └── ncclIbIsend    [src/transport/net_ib/p2p.cc]
                    └── ncclIbMultiSend
                          └── wrap_ibv_post_send  [src/include/ibvwrap.h]
                                └── qp->context->ops.post_send  // 用户态 mlx5 快路径
```

同文件中其它 `post_send` 用途（易混淆）：

| 位置 | 用途 |
|------|------|
| `ncclIbMultiSend` | 数据 RDMA Write (+ With Imm) |
| `ncclIbPostFifo` | 接收端写 CTS 到发送端 `ctsFifo` |
| `ncclIbIflush` | GDR flush（RDMA Read） |

---

## 2. 机器拓扑（关键上下文）

实测节点示例（`nvidia-smi topo -m` + `show_gids`）：

- **8 GPU**，NV18 全互联；双 NUMA：
  - GPU0–3 → NUMA0（CPU 0–47, 96–143）
  - GPU4–7 → NUMA1（CPU 48–95, 144–191）
- **10 个 mlx5 设备**：
  - **训练 400G**：`mlx5_0/1/2/5/6/7/8/9` → `eth400g0–7`，网段 `10.129.x`
  - **管理/杂用**：`mlx5_3/4` → `eth0/eth1`，网段 `10.93.x`（与训练口不同）

### 2.1 理想 PIX 配对（门铃与 DMA 最近）

| GPU | NUMA | PIX NIC | 网口 |
|-----|------|---------|------|
| GPU0 | 0 | mlx5_0 | eth400g0 |
| GPU1 | 0 | mlx5_1 | eth400g1 |
| GPU2 | 0 | mlx5_2 | eth400g2 |
| GPU3 | 0 | **mlx5_5**（不是 mlx5_3） | eth400g3 |
| GPU4 | 1 | mlx5_6 | eth400g4 |
| GPU5 | 1 | mlx5_7 | eth400g5 |
| GPU6 | 1 | mlx5_8 | eth400g6 |
| GPU7 | 1 | mlx5_9 | eth400g7 |

跨 NUMA 的 GPU↔NIC 为 **SYS**（经 UPI）。门铃 MMIO 跨 socket 会变差，通常仍是 μs 级，但在满载 + 争用下可与其它问题叠加。

### 2.2 软件环境摘录

- OFED：`MLNX_OFED_LINUX-23.10-3.2.2.0`
- 运行环境：训练日志出现在 **K8s 作业目录**（`k8s-vj-...`），分析时需考虑容器/共机噪声。
- 日志样例 rank：`ml-b1-ser104:2265:3584 [7]` → pid=2265, **proxy tid=3584**, **本地 GPU/rank 7**。

### 2.3 建议强制排除管理网卡

```bash
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
# 或
export NCCL_IB_HCA=^mlx5_3,mlx5_4
export NCCL_IB_GID_INDEX=3          # 以 show_gids 中 RoCEv2+IPv4 为准
export NCCL_IB_ROCE_VERSION_NUM=2
```

---

## 3. 理论背景：`post_send` 慢可能意味着什么

### 3.1 正常快路径在做什么

1. 组装/链接 WQE（用户态 SQ）
2. 内存屏障
3. 写 HCA **UAR doorbell**（PCIe UC MMIO）

**不**同步等待 CQ 完成，**不**等对端 CTS（CTS 在 `ncclIbIsend` 更早返回/重试）。

### 3.2 若「严格包住 post_send」仍见 ms

在 **未排除计时污染** 前，候选包括：

| 类别 | 说明 |
|------|------|
| 计时口径错误 / 日志污染 | 见第 5 节（本次最大坑） |
| 线程占 CPU 的平台 stall | wall≈thread_cpu；build/align 也会中枪 |
| UAR MMIO / PCIe 卡住 | wall≈cpu，perf 热点在门铃写 |
| libmlx5 BF/spinlock 争用 | 同 HCA 多线程狂 post |
| 跨 NUMA 门铃 + 满载 | 放大尖峰 |
| 虚拟化 / 容器噪声 | K8s 共机、限流等 |
| NVRM / GPU 驱动异常 | dmesg 刷屏，间接拖垮系统 |

### 3.3 容易误判为 post_send 的路径

| 现象 | 实际卡点 |
|------|----------|
| CTS 未到 | `slots[0].idx != idx` 快速返回，上层重试 |
| multi-recv 忙等 | `while (slots[r].idx != idx)` |
| 等完成 | `ncclIbTest` / `ibv_poll_cq` |
| IB 重传 | timeout/retry，体现在完成延迟 |

详见 `nccl_ib_send_recv_cts_flow.md`、`ncclIbIsend_analysis.md`。

---

## 4. 实测日志模式（修正计时之前）

### 4.1 `Mbreakdown` 样本形态

```text
build=0     align=0     wrap_ibv_post_send=14906
build=2290  align=1     wrap_ibv_post_send=3
build=0     align=17820 wrap_ibv_post_send=3
build=18185 align=11    wrap_ibv_post_send=1580
build=0     align=1     wrap_ibv_post_send=40740
build=20909 align=2     wrap_ibv_post_send=106
```

归类：

| 类型 | 特征 | 含义（在计时无 bug 时） |
|------|------|-------------------------|
| B | build 大，post 小 | 与 RDMA API 无关 |
| A | align 大，post≈0 | t1→t2 段 |
| P | post 大，build≈0 | 落在 post 计时窗 |
| 混合 | 多段都大 | 连续 stall |

**关键观察：B / A / P 都存在** → 即使不计时 bug，也不支持「只有 ibv_post_send 有病」。

### 4.2 BUILD 的 wall vs thread CPU（有口径问题，见第 5 节）

```text
POSTSTALL zzzkind=BUILD wall_us=18189 thread_cpu_us=18041 cpu_pct=99.2
POSTSTALL zzzkind=BUILD wall_us=20910 thread_cpu_us=20806 cpu_pct=99.5
```

- `cpu_pct ≈ 99%` → **wall ≈ thread CPU**  
- 含义倾向：**线程在跑着消耗时间**，不是简单 sleep 等网络。  
- 但该 BUILD 标签在错误打点下 **并非纯 build 段**（见下节）。

### 4.3 `NCCL_NET_GDR_LEVEL=LOC` 实验

- 设置为 `LOC` 后尖峰仍在。  
- 在 NCCL 路径距离中：`LOC < PIX < ... < SYS`。  
- GPU–NIC 多为 **PIX**，`GDR_LEVEL=LOC` 往往对 PIX **关闭 GDR**，并非「完全无关的对照」。  
- **更干净的关 GDR：** `NCCL_NET_GDR_LEVEL=0` 或 `DISABLE`。  
- 即便如此，因 build/align 也会爆，**主因更不像「纯 GDR 数据面」**。

### 4.4 内核 NVRM 报错

大量：

```text
NVRM: refcntRequestReference_IMPL: Failed to enter state 1
      (current state: 0, status: 0x00000056)
```

| 字段 | 含义 |
|------|------|
| `0x00000056` | `NV_ERR_NOT_SUPPORTED`（Call not supported） |
| 函数 | GPU RM 对 request 对象做引用/状态迁移失败 |
| 大量刷屏 | 热路径失败重试或异常状态，不是偶发 |

- 对 `mlx5|aer|pcie` 的 egrep 可能较干净，主噪声在 **NVIDIA 驱动**。  
- 与 post_send **无直接调用关系**，但可通过 PCIe/锁/IPI 造成 **系统级抖动**。  
- 建议并行收集：Xid、`nvidia-smi`、驱动/peermem 版本、关 GDR/dma-buf A/B。

---

## 5. 计时方法论纠偏（核心结论）

一次排查中使用的 `ncclIbMultiSend` 打点存在 **系统性问题**，会导致错误归因到 `ibv_post_send`。

### 5.1 错误时间轴（示意）

```text
t0, wall00, cpu00
  [填 WR]
t1
  for (i = 0; i < nqps; i++) {
      [选 QP / rkey / length / profiler]
      wall10, cpu10, t2
      if (wall10 - wall00 >= 1ms)
          INFO(zzzkind=BUILD)     // ← 在 t2 之后！
      [wrap_ibv_post_send]
      t3
      INFO(Mbreakdown: build=t1-t0, align=t2-t1, post=t3-t2)
      ...
  }
```

### 5.2 Bug 清单

| # | 问题 | 后果 |
|---|------|------|
| **A** | `zzzkind=BUILD` 使用 `wall10 - wall00`（函数入口 → post 前） | **不是**纯 build；含 align；`i>0` 时还含 **前几轮 post** |
| **B** | `INFO(BUILD)` 写在 `t2` 之后、`post_send` 之前 | **`Mbreakdown` 的 post 含日志时间**（锁 + 格式化 + 写文件） |
| **C** | `t1` 在 loop 外，`t2` 在 loop 内 | **`nqps>1` 时 align 吞掉上一轮 post+offset** |
| **D** | 热路径高频 `INFO` | 全局 `ncclDebugMutex` + IO，**本身制造 ms 尖峰**，污染邻段与其它 rank |

### 5.3 被污染后的读法

| 日志 | 不可直接得出 |
|------|----------------|
| `post=1580` 且刚打过 BUILD INFO | 不能 = 真 post_send |
| `zzz BUILD wall=18ms cpu_pct=99%` | 不能 = memset 烧了 18ms |
| `align=17820 post=3` 且 nqps>1 | 不能 = 选 QP 要 18ms |

**仍可保留的判断：** 系统上确实出现了 ms 级异常；但 **定罪 `ibv_post_send` 需要干净计时**。

### 5.4 正确打点原则

1. **先量完一段，再打日志**；日志永远在该段 timer 之外。  
2. **每段独立起止**，禁止「从函数入口累计」冒充单段。  
3. **多 QP：align/post 的 timer 必须在 for 循环内每轮重置**。  
4. 排查期热路径 **禁止 INFO**；慢样本进 **TLS ring buffer**，周期/结束再 flush。  
5. 同时打 `CLOCK_MONOTONIC_RAW`（wall）与 `CLOCK_THREAD_CPUTIME_ID`（thread cpu）。

### 5.5 推荐分段

| 段名 | 包含 |
|------|------|
| **build** | 填 `wrs/sges/lastWr` 仅一次（loop 外） |
| **align**（每 QP） | `GetQpForRequest`、rkey/length、（可选单独）profiler |
| **post**（每 QP） | **仅** `wrap_ibv_post_send`，前后无任何 INFO |

日志字段建议：

```text
qp_i/nqps, build_w/build_c, align_w/align_c, post_w/post_c,
qpn, devIndex, nreqs, size, hca_name
```

### 5.6 wall vs cpu 读法

| 关系 | 含义 |
|------|------|
| wall ≈ cpu，且都很大 | 忙等 / 在跑 / MMIO 等待仍记在线程上 / 大量可计费工作 |
| wall ≫ cpu | 阻塞、睡眠、被换出、部分 throttle |
| build/align/post **都会** wall≈cpu 尖峰 | 平台级 stall，非单一 API |
| 仅 post 大且计时干净 | 再查门铃 / libmlx5 / PCIe |

---

## 6. 综合因果模型（当前最佳判断）

```text
                    ┌─────────────────────────────┐
                    │  平台 / 驱动 / 运行环境抖动   │
                    │  (NVRM、PCIe、共机、电源等)   │
                    └─────────────┬───────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
   落在 build 计时窗        落在 align 计时窗        落在 post 计时窗
   (填 WR 等)               (选 QP / 或计时 bug)     (门铃 或 INFO 污染)
          │                       │                       │
          └───────────────────────┴───────────────────────┘
                                  │
                    业务表现：训练 step 偶发/持续变慢
```

叠加放大因子：

1. **错误/过密的 NCCL_DEBUG INFO**（热路径）  
2. **GPU–NIC 亲和错误** / 管理网卡混入 / 跨 NUMA SYS  
3. **多 QP、多 channel** 提高 post 频率，尖峰更易被命中  
4. **K8s 多租户** 噪声  

**较弱假设（已被部分实验削弱）：**

- 「只有 GDR 打开才慢」——`GDR_LEVEL=LOC` 后仍见尖峰，且 build 类尖峰与 RDMA 无关。  
- 「CPU 调度换出」——初期称 proxy 未被 schedule out；且 wall≈cpu 更像在跑。仍建议用正确计时 + `perf` 复核。

---

## 7. 排查清单（后续查阅用）

### 7.1 立刻做

- [ ] 按第 5.5 节重做打点；热路径去掉 INFO  
- [ ] 确认 `nqps`、`NCCL_IB_QPS_PER_CONNECTION`  
- [ ] `NCCL_IB_HCA` 排除 mlx5_3/4；核对每个 GPU 的 PIX NIC  
- [ ] 新日志同时输出 `post_w/post_c`、`build_w/build_c`、`nqps`

### 7.2 环境与驱动

```bash
# 拓扑与网卡
nvidia-smi topo -m
show_gids
ibstat | egrep 'CA type|Firmware|Rate|State'

# GPU 驱动噪声
dmesg -T | egrep -i 'Xid|NVRM|refcntRequest|GSP|fallen'
nvidia-smi

# peermem
lsmod | egrep 'nvidia|peermem|nv_peer'
```

### 7.3 A/B 实验

| 实验 | 设置 | 看什么 |
|------|------|--------|
| 真关 GDR | `NCCL_NET_GDR_LEVEL=0` | 尖峰是否消失 |
| 关 dma-buf | `NCCL_DMABUF_ENABLE=0` | 与 peermem 路径 |
| 单 QP | `NCCL_IB_QPS_PER_CONNECTION=1` | 简化归因 |
| 降日志 | `NCCL_DEBUG=WARN` + 无热路径 INFO | 尖峰频率是否暴跌 |
| 绑核 NUMA | proxy/计算与 NIC 同节点 | SYS 门铃是否改善 |

### 7.4 perf / 平台

```bash
# 盯 proxy tid（示例 3584）
perf record -t 3584 -e cycles,instructions,page-faults --call-graph dwarf -- sleep 15
perf report
perf stat -t 3584 -e page-faults,minor-faults,major-faults -- sleep 10

# 平台延迟（与 RDMA 无关的对照）
# cyclictest / turbostat smi 等
```

| perf 结果 | 方向 |
|-----------|------|
| 热点 `ncclDebugLog` / 写文件 | 日志问题 |
| 大量 page-faults | 缺页导致「假慢」 |
| 热点 mlx5 / UAR | 真门铃路径 |
| 热点在 memset 但 cycles 爆炸 | 核/内存/虚拟化 stall |

---

## 8. wrap_ibv_post_send 实现备注

```c
// src/include/ibvwrap.h（逻辑）
static inline ncclResult_t wrap_ibv_post_send(...) {
  int ret = qp->context->ops.post_send(qp, wr, bad_wr);
  if (ret != IBV_SUCCESS) {
    WARN(...);
    return ncclSystemError;
  }
  return ncclSuccess;
}
```

- **无重试、无 sleep**。  
- SQ 满时 mlx5 通常立即 `ENOMEM`，而不是堵 20ms。  
- 故「成功返回但耗时 20ms」更像 **执行路径被拖慢 / 计时窗含其它工作**，而非 verbs 同步等网络。

---

## 9. 结论摘要（供快速回顾）

1. **正常 `ibv_post_send` 应为 μs 级**；ms 级是异常。  
2. 初版分段日志显示 **build / align / post 均可出现 ms 尖峰**，不支持「单一 API 固有慢」。  
3. **打点实现存在严重口径错误**（累计时间冒充 BUILD、INFO 污染 post、多 QP 污染 align），在修正前 **不能**把账算死在 `ibv_post_send`。  
4. 热路径 `INFO` 会通过全局锁与 IO **主动制造** ms 级延迟。  
5. `NCCL_NET_GDR_LEVEL=LOC` ≠ 关 GDR；PIX 拓扑上需 `0/DISABLE` 做对照。  
6. 机器为 **8×GPU + 8×400G + 2 管理口 + 双 NUMA**；必须 `NCCL_IB_HCA` 绑训练口并核对准 PIX。  
7. dmesg 中大量 `refcntRequestReference` + `0x56` 指向 **GPU RM 异常**，宜与尖峰时间对齐排查。  
8. **下一步**：干净计时 → 再分「真 post MMIO」vs「平台 stall」vs「日志自扰」；并行查 NVRM/Xid/亲和/K8s 噪声。

---

## 10. 修订记录

| 日期 | 内容 |
|------|------|
| 2026-07-10 | 初稿：汇总问题现象、拓扑、日志模式、计时 bug、NVRM、排查清单与结论 |

---

## 11. 参考

- 源码：`src/transport/net_ib/p2p.cc`、`src/include/ibvwrap.h`、`src/transport/net.cc`、`src/debug.cc`
- 笔记：`notes/ncclIbIsend_analysis.md`、`notes/nccl_ib_send_recv_cts_flow.md`、`notes/nccl_params.md`
- NVIDIA status `0x00000056`：`NV_ERR_NOT_SUPPORTED`
