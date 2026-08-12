# NCCL 训练场景 All-Reduce 慢定位笔记

> 场景：128 GPU（TP+DP）分布式训练，DDP 梯度 all-reduce。相同 size、相同 128 卡下，
> `nccl-tests all_reduce_perf` benchmark 仅需 ~1ms/op，而训练中单个 AR kernel 在
> torch.profiler 里观察到 20ms+。本文记录定位方法论与结论（NCCL 2.29.3，`NCCL_DEBUG=info`）。

## 1. 现象与基线

- 同一 size（如 40–48MB）、同一组 128 卡：
  - **benchmark**（`all_reduce_perf`，warmup 后紧 loop）：~1ms/op
  - **训练**（DDP）：单个 AR kernel 4ms ~ 40ms+，且同一 size 的不同 op 之间差 10×
- 训练日志本身无 NCCL error / watchdog / 超时（见下：自旋在秒级 watchdog 阈值之下，不会触发）

## 2. 关键源码事实（推理依据）

- **NCCL device kernel 不是纯计算**：`src/device/prims_ll128.h:70` 的 `waitSend` 是
  `while (sendConnHeadCache + NCCL_STEPS < sendConnHead + 1)` 自旋等待对端/代理推进。
  所以 profiler 里 kernel 的 wall-clock 时长 = `[到达等待] + [代理/网络等待] + [真正 reduce]`。
  长 kernel 时长里很大一部分是自旋。
- **kernel 时长 ≠ reduce 真实延迟**；它度量的是 kernel 在 GPU 流上从启动到整个 collective
  完成的墙钟时长。
- 因此用 `dur`（kernel 时长）排查时，长 `dur` 的根因在 kernel **执行期间**（自旋/被饿死/网络慢），
  而不是 host enqueue 的 gap（host 慢会表现为 kernel 启动前的空隙，而非 kernel 时长）。

## 3. 方法论：跨 rank、按 op 统计

对每个 AR op，统计 128 个 rank 各自的 kernel 时长，得到
`dur_min / dur_max / dur_avg / p50 / p90 / p99 / max-min 比`。
**同一 op 内跨 rank 的离散度（max/min）** 是分辨"隐式 barrier / straggler" 与
"全员同步慢" 的关键判据。

## 4. 三类现象（实测可清晰分开）

### Type A — 长尾型（高 max/min，p50 ≪ p99）
- 形态：max/min 2×–15×，p50 很低但 p99 很高。
- 典型：12KB 小 op、以及少数大 op 出现少数 rank 拖尾。
- 成因：**少数 rank 慢 / 晚到**（隐式 barrier）或个别慢节点 / 慢链路。早到的 rank
  自旋等晚到的 → 时长差大。
- 这是经典 collective 同步开销，**不是本次要追的主因**。

### Type B — 均匀慢（max/min ≈ 1.0–1.1，但 dur 20–42ms）★ 真正要解的
- 形态：所有 128 个 rank 时长几乎相同（离散度仅 1.0×–1.1×），但绝对值 20–42ms。
- **关键含义**：全员同步地一起慢，却没有跨 rank 时长差。这直接排除了 barrier / straggler
  ——barrier 必然让快 rank dur 更大、慢 rank dur 更小，造出离散度。
- 它说明：所有 rank 几乎同时算完、同时进 AR，而 **AR 的数据搬运阶段对所有人一样慢**。
  瓶颈在"所有 rank 共享、且同时承受"的东西上：**GPU 计算争用、代理线程、或网络**。

### Type C — 同 size 跨 op 的巨大方差
- 同样 48MB：有的 op 4ms、有的 42ms，差 10×；且慢 op 在执行序上**成簇**出现。
- 提示慢 op 集中在迭代的某个相位（如 backward 前段）。

## 5. 头号假设：AR kernel 与反向计算争抢 SM/HBM

一次解释 Type A 之外的全部特征（均匀性 + 10× 跨 op 方差 + 成簇）：

- **均匀性**：DDP 下各 rank 跑同一模型、被 collective 同步，进某 AR 时大家都在同一层 →
  同样的并发计算负载 → 同样被饿死 → max/min ≈ 1.0。
- **跨 op 10× 方差 + 成簇**：DDP overlap 模式下，**backward 前段的梯度桶** overlap 着大量
  剩余反向计算 → 重争用 → 慢；**backward 末段的桶** overlap 的计算少 → 轻争用 → 快。
  慢 op 因此成簇在 backward 前段。
- **benchmark = 1ms**：benchmark 时 GPU 只做 AR、零并发计算 → 不争用 → 飞快。
- **连最快 training op（~4ms）也是 benchmark 的 4×**：即使轻争用窗口，AR kernel 仍与部分
  计算共抢 GPU。

ring / tree all-reduce 的 AR kernel 吃 SM 和 HBM 带宽；与反向 matmul 共抢时，发 RDMA 的速率
下降 → 带宽塌缩（48MB@42ms ≈ 1GB/s 量级，而 4ms 时 ~12GB/s）。

## 6. 基线 4× 差距是另一个独立问题：comm / channel / algo 配置

即使最快 training AR 也只有 benchmark 的 1/4，这可能与争用无关，而是 **DDP comm 与 benchmark
comm 的结构性差异**：

- channel 数（`NCCL_MAX_CHANNELS` / tuner 实际分配）
- algo（Ring / Tree / NVLS-SHARP / CollNet）
- protocol（LL128 / Simple）
- 是否走 GDR、是否用 SHARP

benchmark 通常用满 channel、tuner 选吞吐最优；DDP comm 可能 channel 更少或 algo 不同。
**需要把每个 op 标注它属于哪个 comm**——Type B 的均匀慢 op 是否都集中在某个 channel 少的 comm 上。

## 7. 次要候选

- **代理线程被注册 / UDS 打断**：communicator init 日志里可见海量 `ProxyCall UDS` /
  `ClientGetFd` / `Register`（对应 `ncclProxyMsgType` 的 Register=11 / GetFd=9）。若每个 iter
  梯度 buffer 反复 register / deregister（`gradient_as_bucket_view=False` / 未开 `static_graph`
  / 无持久注册），代理线程周期性被占住，AR kernel 自旋等代理；注册风暴在所有 rank 同步发生
  → 均匀慢，与 Type B 吻合。
- **网络 / IB 拥塞或单条慢链路**：对称 all-reduce 通常表现为跨 rank 离散（Type A），
  完美均匀较少；除非整网同步拥塞。

## 8. 一次定位（nsys）

挑一个 Type B 的 op（均匀慢、max/min ≈ 1.0），在**单 rank** 上放大看那一帧：

1. AR kernel 期间，**同一 GPU 的计算流上在跑什么**？有大 backward matmul 并发 → 计算争用实锤。
2. AR kernel 期间，**proxy 线程**在干嘛？忙 `Register / UDS` → 注册饱和；闲但 kernel 仍慢 → 网络。
3. AR kernel 是**独自占 GPU 还是与计算 kernel 共占 SM**（SM occupancy / concurrent kernels 视图）。
4. 该 op 走**哪个 comm、几条 channel、什么 algo / protocol**？与 fast op 对比。

## 9. 快速实验（按性价比）

1. **关掉 / 置零计算，只对同 buffer 跑 AR**：若 Type B 的 op 全部掉回 ~1ms → 计算争用确认。
   最干脆的一刀。
2. **强制匹配 benchmark 的 NCCL 配置**：`NCCL_ALGO` / `NCCL_PROTO` / `NCCL_MAX_CHANNELS` /
   `NCCL_MIN_NRINGS` 设成与 benchmark 一致，看基线 4× 是否消失 → 区分"配置差异"还是"争用"。
3. **DDP 旋钮**：`bucket_cap_mb`（更大桶 → 更少更大 AR，更易打满网络）、
   `gradient_as_bucket_view=True`、`static_graph=True`、持久注册 → 削注册 churn + 减碎片 AR。
4. **TP 并发**：该模型有 TP（`src/include/proxy.h` 中 `tpRank / tpLocalRank / tpnRanks`）。
   确认 Type B 的 DP all-reduce 是否与 TP 的 all-reduce / all-gather **同时**跑、抢 NVLink / SM。

## 10. 结论

- "20ms+"**不是 all-reduce 本身慢**（同 size benchmark 1ms 已证）。
- 跨 rank 按 op 的统计把"隐式 barrier / straggler"（Type A / C 长尾）与"全员同步慢"（Type B）
  分开；**真凶在 Type B**：max/min ≈ 1.0 说明所有 rank 被同一东西同时卡住。
- 头号可能：**AR kernel 与反向计算抢 SM / HBM**（解释均匀性、10× 跨 op 方差、成簇）；其次是
  代理线程注册饱和、comm / channel 配置导致的 4× 基线。
- 下一步：用"关计算跑纯 AR"验证计算争用，再用 nsys 在 Type B 的 op 上看并发计算 / proxy / comm，
  决定往哪挖。
