# NCCL Fault Tolerance（容错）支持与机制分析

> **状态**：基于 NCCL 2.30.x 源码与官方 User Guide 的归纳  
> **范围**：应用级 API 恢复、设备/代理中止、IB 传输弹性、RAS 诊断；**不含** 框架级 checkpoint/restart  
> **相关代码**：`src/init.cc`、`src/group.cc`、`src/device/primitives.h`、`src/transport/net_ib/p2p_resiliency*.cc`、`src/ras/`  
> **官方文档**：`docs/userguide/source/usage/communicators.rst`（Fault Tolerance / Error handling）  
> **相关笔记**：`thread-model.md`、`ibv-post-send-stall.md`、`env-params.md`

---

## 目录

1. [执行摘要与能力边界](#1-执行摘要与能力边界)
2. [分层总览](#2-分层总览)
3. [应用级容错 API 与错误模型](#3-应用级容错-api-与错误模型)
4. [中止传播机制（abortFlag）](#4-中止传播机制abortflag)
5. [Communicator Shrink / Revoke](#5-communicator-shrink--revoke)
6. [IB 传输弹性：Port Failover 与 Recovery](#6-ib-传输弹性port-failover-与-recovery)
7. [RAS：可靠性/可服务性诊断网络](#7-ras可靠性可服务性诊断网络)
8. [底层加固（非完整 FT，但降低故障率）](#8-底层加固非完整-ft但降低故障率)
9. [端到端故障场景对照](#9-端到端故障场景对照)
10. [关键环境变量](#10-关键环境变量)
11. [应用集成建议](#11-应用集成建议)
12. [结论](#12-结论)
13. [源码索引与修订记录](#13-源码索引与修订记录)

---

## 1. 执行摘要与能力边界

### 1.1 NCCL 容错在做什么

NCCL 的 fault tolerance **不是** “进程挂了自动无感续训”，而是提供一套 **可观测 + 可中止 + 可重建通信域** 的机制，让上层（训练框架 / 作业系统）在节点/进程/网络故障后：

1. **发现** 异步错误（不永远卡死）  
2. **中止** 当前 communicator 上未完成的集体通信  
3. **释放** 资源  
4. **可选地** 排除故障 rank，**重建** 更小的 communicator 继续  
5. **可选地** 在 IB 多网卡场景下对 **端口/设备级** 错误做传输层 failover/recovery  

### 1.2 能力边界（重要）

| NCCL **有** | NCCL **没有 / 不做** |
|-------------|----------------------|
| `ncclCommGetAsyncError` 异步错误查询 | 透明的进程故障替换（process FT） |
| `ncclCommAbort` 中止并释放 | 自动 checkpoint / 自动 restart 训练 |
| 非阻塞 comm + abort 可打断阻塞路径 | 单网卡、单路径故障的完整透明恢复（默认） |
| `ncclCommShrink` 排除故障 rank 建新 comm | 保证 shrink 后集体语义与业务状态一致（应用负责） |
| IB multi-rail **resiliency**（可选） | 默认开启的网卡热切换（默认 env=0） |
| RAS 诊断 / dead peer 广播（观测） | RAS 自动修复数据面 |
| 设备 kernel 轮询 `abortFlag` 退出忙等 | 所有错误都可恢复（多数 fatal 需重建 comm） |

**一句话：** NCCL FT = **通信层“死得干净 + 可重建”**；业务级容错由 **应用/调度器** 编排。

---

## 2. 分层总览

```text
┌─────────────────────────────────────────────────────────────────┐
│ L4 应用 / 框架                                                   │
│   检测超时 / 轮询 asyncError / 决策 restart 或 shrink            │
│   同步健康 rank 的 abort 意图（globalFlag）                      │
└────────────────────────────▲────────────────────────────────────┘
                             │ ncclCommGetAsyncError / Abort /
                             │ Shrink / Revoke / InitRankConfig
┌────────────────────────────┴────────────────────────────────────┐
│ L3 Communicator 生命周期                                         │
│   abortFlag(host+device)  ·  asyncResult  ·  nonblocking         │
│   revoke 静默  ·  shrink 排除 rank  ·  destroy 释放               │
└────────────────────────────▲────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│ L2 执行引擎                                                       │
│   Device: checkAbort / aborted 短路拷贝                           │
│   Proxy / Bootstrap socket: abortFlag 打断阻塞                    │
│   Group job: 错误时置 abortFlag                                   │
└────────────────────────────▲────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│ L1 传输弹性（可选，主要 IB）                                       │
│   multi-rail QP  ·  CQE 错误 → probe → 重传 / 换 QP               │
│   Port failover  ·  Port recovery 线程 alive/ack                  │
└────────────────────────────▲────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│ L0 可观测性                                                       │
│   RAS 线程网络  ·  deadPeer 广播  ·  ncclras 客户端查询             │
│   NCCL_DEBUG / asyncError 语义                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 应用级容错 API 与错误模型

### 3.1 官方 FT 流程（User Guide）

典型致命故障（网卡挂、节点挂、对端进程挂）时，文档推荐：

```text
1) 健康 rank 通过 ncclCommGetAsyncError / 超时发现异常
2) 所有仍存活的 rank 调用 ncclCommAbort(comm)
3) 清理 / 重建：ncclCommInitRank* 新通信域
   或 ncclCommShrink(..., NCCL_SHRINK_ABORT) 排除故障 rank
4) 继续训练（应用负责状态一致性）
```

**硬性建议：**

- Communicator 设为 **nonblocking**（`config.blocking = 0`）  
- **Abort 期间没有任何线程** 再调其它 NCCL 操作  
- 阻塞模式下，网络错误可能导致线程卡在 NCCL 内部，**abort 无法介入**  

### 3.2 核心 API

| API | 作用 | 容错角色 |
|-----|------|----------|
| **`ncclCommGetAsyncError`** | 取 `asyncResult` / 异步错误 | **探测** 通信是否已死 |
| **`ncclCommAbort`** | 置 abort、打断未完成 op、释放资源 | **止损**，comm 不可再用于正常集体 |
| **`ncclCommDestroy`** | 正常销毁（可能阻塞） | 优雅退出；故障场景优先 Abort |
| **`ncclCommRevoke`** | 撤销 comm：静默、禁 launch、可再 destroy/split/shrink | 中间态管控，禁资源共享 |
| **`ncclCommShrink`** | 排除 rank 列表，创建子 comm | **降级续跑**（去掉坏节点） |
| **`ncclCommInitRankConfig`** | 带 `ncclConfig_t` 初始化 | 启用 nonblocking / 配置 shrinkShare 等 |

### 3.3 错误码与是否“可恢复”

| 错误 | 含义 | 对 comm | 典型处理 |
|------|------|---------|----------|
| `ncclSuccess` | 正常 | 可用 | — |
| `ncclInProgress` | 非阻塞未完成 | 可用 | 继续 poll |
| `ncclInvalidArgument` | 参数错 | **通常仍可用** | 改参数；可不 abort |
| `ncclInvalidUsage` | API 误用 / 动态条件 | **常 fatal** | Abort + 重建 |
| `ncclSystemError` | 系统/驱动调用失败 | **fatal** | Abort + 查环境 |
| `ncclUnhandledCudaError` | CUDA 失败 | **fatal** | Abort + 查 CUDA |
| `ncclInternalError` | NCCL 内部 bug | **fatal** | Abort + 报 bug |
| `ncclRemoteError` | 远端/链路相关（含 resiliency 放弃） | 视路径 | 常 Abort 或 shrink |

Group 内动态错误：由 **`ncclGroupEnd`** 汇总，影响组内所有 op；需对 **组内所有 comm** abort。

### 3.4 异步错误轮询模式（原理）

网络错误往往 **不在** 发起 `ncclAllReduce` 的返回值里立即出现，而在：

- proxy 进度失败 → 写入 comm 的 **async 状态**  
- kernel 永不结束 / stream 不 ready  

因此应用应：

```text
while (!done):
  cudaStreamQuery(stream)
  ncclCommGetAsyncError(comm, &asyncErr)
  if asyncErr != success:
    ncclCommAbort(comm)
    break / restart
  sched_yield()
```

这是 NCCL **官方推荐的“可中止等待”** 模型，也是上层 FT 的基础。

---

## 4. 中止传播机制（abortFlag）

### 4.1 双副本 abort 标志

Comm 初始化时（`src/init.cc`）：

```text
abortFlag      // 主机侧 volatile / atomic
abortFlagDev   // cudaHostAlloc，设备可访问
abortFlagRefCount
```

设备侧 `ncclKernelComm` / `devComm` 持有 `abortFlag` 指针（见 `devcomm_*`、`dev_runtime.cc`）。

### 4.2 设备侧：忙等可退出

`src/device/primitives.h`：

```c
__device__ inline int checkAbort(int& abortCache, const int abortValue, int& spins) {
  if (++spins < NCCL_SPINS_BEFORE_CHECK_ABORT) return 0;
  spins = 0;
  int abort = *ncclShmem.comm.abortFlag;
  if (abort) {
    ncclShmem.aborted = abort;
    abortCache |= abortValue;
  }
  return abort;
}
```

用于 LL / LL128 / Simple 原语的 spin 循环。`prims_simple.h` 中若已 abort，可将 workSize 置 0，**跳过实际拷贝/reduce**，尽快结束 kernel，避免 GPU 永久空转。

**原理：** Abort 不是“撤销已发出的 IB 包”，而是 **停止本地执行与等待**，使 host 能回收 comm。

### 4.3 主机侧：Socket / IPC / Group

| 路径 | 行为 |
|------|------|
| Bootstrap / socket | `abortFlag` 置位后 accept/connect/读写循环退出 |
| IPC socket | 同样检查 abort，避免卡在管道 |
| Group 异步 job | 失败时 `atomic_store(abortFlag/abortFlagDev, 1)` |

`ncclCommAbort` 会驱动这些路径收敛，并释放 proxy、注册内存等资源。

### 4.4 非阻塞与 Abort 的配合原理

```text
blocking=1:
  线程可能深陷 proxy 忙等 / 阻塞 socket
  其它线程调 Abort 时，卡死线程未必能及时看到 abortFlag
  → 文档：可能 hang forever

blocking=0:
  NCCL 调用尽快返回 ncclInProgress
  应用线程保持可调度，可随时 Abort
  → FT 的前提条件
```

---

## 5. Communicator Shrink / Revoke

### 5.1 `ncclCommShrink`：排除故障 rank

**用途：** 节点/进程故障后，**存活 rank** 创建一个 **不含故障 rank** 的新 communicator，继续集体通信。

| 标志 | 行为 |
|------|------|
| **`NCCL_SHRINK_DEFAULT` (0)** | 正常 shrink；父 comm 上 **不应有** outstanding NCCL op（防死锁）；可按 `shrinkShare` 共享资源 |
| **`NCCL_SHRINK_ABORT`** | 先中止父 comm 上进行中的操作，再 shrink；**不共享** 父资源；用于父 comm 已错误/挂起的恢复 |

约束与原理：

- **只有将成为新 comm 成员的 rank** 调用 shrink  
- 排除列表由应用给出（应用需知道谁挂了）  
- Shrink **不恢复** 模型权重/优化器状态——只恢复 **NCCL 通信域**  
- 父 comm 资源：ABORT 路径下仍可能未完全 free，应用可再 `ncclCommAbort(parent)`  

实现挂接：`src/init.cc` 中 `ncclCommInitChildComm(..., isShrink=true, shrinkFlags)`。

### 5.2 `ncclCommRevoke`

**用途：** 将 communicator **撤销** 到安全静默态：

- 停止后续 launch（再集体 → `ncclInvalidUsage`）  
- 可再 `Destroy` / `Split` / `Shrink`  
- **禁用** `splitShare` / `shrinkShare` 资源共享  
- **不支持** revoke 后再 `Finalize` 的常规路径  

适合：已确定 comm 不可信，但希望受控地拆分/收缩，而不是立刻完整 abort 语义。

### 5.3 与 Split 的对比

| | Split | Shrink |
|--|-------|--------|
| 目的 | 按 color 划分子通信域 | **去掉** 故障/不需要的 rank |
| FT 场景 | 次要 | **主要**（节点故障降级） |
| 错误父 comm | Split 可能需 Abort 打断 | **`NCCL_SHRINK_ABORT`** 专为错误恢复 |

---

## 6. IB 传输弹性：Port Failover 与 Recovery

> 代码：`src/transport/net_ib/p2p_resiliency.cc`、`p2p_resiliency_recovery.cc`、`p2p.cc` 错误 CQE 分支。  
> **默认关闭**（需环境变量显式打开）。

### 6.1 解决什么问题

多网卡 / multi-rail（`vProps.ndevs > 1` 或 multi-QP）时，**单端口/单设备** 出错（CQE error）若直接 fail 整个 comm，代价过高。

Resiliency 目标：

1. **识别** 是否仍有健康设备可继续  
2. **探测** 出错请求是否其实已送达  
3. **重传** 未送达部分（selective retransmission，与 `sentData[qpIndex]` 配合）  
4. **替换 QP** 到健康设备  
5. **可选恢复** 故障端口（独立 recovery 线程 + alive/ack 协议）  

### 6.2 使能条件

```text
NCCL_IB_RESILIENCY_PORT_FAILOVER=1
  → 连接路径上分配 resiliency 上下文
  → 常与 multi-dev / OOO RQ 等能力协同（见 common.cc）

NCCL_IB_RESILIENCY_PORT_RECOVERY=1
  → recoveryEnabled，故障后进入 RecoveryInProgress
```

无 multi-rail / 单设备时，failover 空间有限（没有“换到另一块网卡”）。

### 6.3 设备状态机

```text
Ok
  → (CQE/设备错误) Error
  → ReplaceQps 把流量迁走
  → [若 recovery 开] RecoveryInProgress
       → Recovered  或  RecoveryFailed / ErrorPermanent
```

（`ncclIbResiliencyDevState*` in `p2p_resiliency.h`）

### 6.4 Failover 数据面流程（发送侧）

```text
[1] poll CQ 得到 error WC
[2] ncclIbResiliencyHandleCompletionErrorSender
      登记 failedRequests[slot]
      state = Pending
[3] 等待 ProbeDelay 后 RDMA Read probe
      （发端用 probing QP 读对端 cmpls / 状态相关内存）
[4] ProbeCompleted:
      - 各 qpIndex 上 probingResults 全 true
          → 数据已到，清 events，当作请求成功
      - 有 missing
          → ncclIbResiliencyRepostRequest（跳过已 sentData 的 QP）
[5] 同时 HandleDeviceFailure:
      标记设备 Error
      ReplaceQps（改走仍 Ok 的 dev）
      可选启动 port recovery
```

接收侧：错误 CQE 上可能 **repost recv**，并配合设备状态，避免整 comm 立刻死。

**原理要点：**

- **Probe**：区分 “真丢了” vs “CQE 错但数据已交付”，避免盲目双发导致数据损坏语义  
- **Selective retransmit**：multi-QP 下只补未确认的 rail  
- **Generation id**：忽略过期 CQE，防止乱序完成干扰恢复逻辑  

### 6.5 Port Recovery 协议（可选）

`p2p_resiliency_recovery.cc`：

- 独立 **recovery 线程**  
- 故障后延迟 `RECOVERY_START_DELAY`  
- 通过 **portRecoveryQps** 发 **alive 消息批** / 等 **ack**  
- 超时与最大尝试次数可配  
- 成功 → 设备 Recovered，可再参与；失败 → Permanent  

这是 **链路/端口级** 自愈，不是进程级 FT。

### 6.6 与应用级 FT 的关系

```text
IB resiliency 成功:  上层可能完全无感（asyncError 不出现）
IB resiliency 失败:  仍上报错误 → 应用 Abort/Shrink
IB resiliency 关闭:  单设备 CQE 错误更易直接导致 comm 失败
```

---

## 7. RAS：可靠性/可服务性诊断网络

> 代码：`src/ras/`（`ras.cc`、`rasnet.cc`、`peers.cc`、`collectives.cc`、`client*.cc`）  
> 二进制：`ncclras` 客户端（默认端口 **28028**）  
> 平台：以 **Linux** 为主；Windows 为 stub。

### 7.1 定位

RAS = **Reliability, Availability, Serviceability** 辅助面：

- **不是** 数据面热备  
- **是** 进程间旁路控制网 + 状态采集 + dead peer 通知  

### 7.2 架构原理

```text
每个 NCCL 进程:
  首次 comm init → ncclRasCommInit
    → 监听随机端口的 RAS socket
    → 启动 “NCCL RAS” 线程
    → pipe 通知：本地 NCCL 线程 ↔ RAS 线程

RAS 线程:
  poll 多路 socket
  维护 peers[]（所有 NCCL 进程地址）
  处理 KEEPALIVE / PEERSUPDATE / COLLREQ/RESP
  可 BC_DEADPEER 广播“发现某 peer 死亡”

外部:
  ncclras 客户端连入查询连接与 communicator 健康信息
```

### 7.3 消息与集合类型（摘要）

| 类型 | 用途 |
|------|------|
| `CONNINIT` / `ACK` | RAS 连接握手 |
| `KEEPALIVE` | 保活 |
| `PEERSUPDATE` | 同步 peer 列表 |
| `COLLREQ` / `COLLRESP` | 类广播/收集 |
| `RAS_BC_DEADPEER` | 广播死节点 |
| `RAS_COLL_CONNS` | 收集 RAS 连接信息 |
| `RAS_COLL_COMMS` | 收集各 comm 状态（含 asyncError 等） |

超时：`RAS_COLLECTIVE_LEG_TIMEOUT_SEC` 等，可用 `NCCL_RAS_TIMEOUT_FACTOR` 放大。

### 7.4 对 FT 的价值

| 能做 | 不能做 |
|------|--------|
| 帮助运维/脚本发现 hung rank、死 peer | 自动替换进程 |
| 为应用/调度提供旁路证据 | 替代 `GetAsyncError` |
| 与数据面故障关联排查 | 保证数据面继续正确前进 |

---

## 8. 底层加固（非完整 FT，但降低故障率）

这些机制 **提高鲁棒性**，但不构成“故障后自动恢复业务”：

| 机制 | 位置/参数 | 作用 |
|------|-----------|------|
| **IB RC timeout / retry** | `NCCL_IB_TIMEOUT`、`NCCL_IB_RETRY_CNT` | 链路瞬时错误硬件重传 |
| **Adaptive Routing** | `NCCL_IB_ADAPTIVE_ROUTING` | 减轻拥塞丢包（与 FT 互补） |
| **Multi-rail / multi-QP** | `NCCL_IB_QPS_PER_CONNECTION`、merge NICs | 提供 resiliency 迁移目标 |
| **Socket abortFlag** | bootstrap / proxy | 配合 Abort 打断连接建立 |
| **cuMem / 现代分配** | 文档建议 | Abort 时资源释放更可靠 |
| **Proxy 进度失败 → asyncError** | proxy / net | 把传输错误抬到应用可观测层 |

---

## 9. 端到端故障场景对照

| 故障 | NCCL 层表现 | 推荐处理 |
|------|-------------|----------|
| 单 IB 端口 CQE 错误（多 rail + failover=1） | 传输层 probe/重传/换 QP | 可能无感；失败则 asyncError |
| 单 IB 端口故障（failover=0 或单网卡） | asyncError / hang | Abort；查网卡 |
| 对端进程崩溃 | 本地 op 不完成 / asyncError；RAS 可能 deadPeer | **所有存活 rank Abort**；重建或 Shrink |
| 整节点宕机 | 同上 | Shrink 排除该节点 ranks 或全局 restart |
| CUDA 错误 | `ncclUnhandledCudaError` | Abort；查 GPU/驱动 |
| 应用 API 误用 | `ncclInvalidUsage` 等 | 修用法；常需 Abort |
| 想排除慢/坏 rank 继续 | — | **Shrink**（DEFAULT 或 ABORT） |
| 只想停 launch 再拆分 | — | **Revoke** 再 Split/Shrink |

```text
决策简图:

  错误可局部到网卡 rail 且 resiliency 开?
      是 → 等传输自愈；超时仍失败 → 走下支
  整个 comm 不可用?
      是 → 所有健康 rank Abort
           ├─ 换一批人继续 → Shrink(ABORT) 或 新 Init
           └─ 作业级重启 → 框架 checkpoint + 新进程 Init
```

---

## 10. 关键环境变量

### 10.1 应用/Comm 行为

| 变量/配置 | 默认 | 含义 |
|-----------|------|------|
| `config.blocking` | 阻塞 | **FT 应设 0（nonblocking）** |
| `NCCL_COMM_SHRINK_SHARE_RESOURCES` | undef | shrink 是否共享父资源（ABORT 时忽略） |

### 10.2 IB Resiliency

| 变量 | 默认 | 含义 |
|------|------|------|
| `NCCL_IB_RESILIENCY_PORT_FAILOVER` | **0** | 端口/设备 failover |
| `NCCL_IB_RESILIENCY_PORT_FAILOVER_MAX_ATTEMPTS` | 1 | probe 最大次数 |
| `NCCL_IB_RESILIENCY_PORT_FAILOVER_PROBE_DELAY` | 10 ms | 错误后 probe 前等待 |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY` | **0** | 故障端口恢复协议 |
| `NCCL_IB_RESILIENCY_PORT_RECOVERY_*` | 见源码 | 延迟、alive 批、超时、最大尝试 |

### 10.3 IB 基础可靠性

| 变量 | 默认 | 含义 |
|------|------|------|
| `NCCL_IB_TIMEOUT` | 20 | RC 超时指数 |
| `NCCL_IB_RETRY_CNT` | 7 | 硬件重试次数 |
| `NCCL_IB_MQP_RETRY_SLEEP_MSEC` | 100 | 多 QP 相关重试睡眠 |

### 10.4 RAS

| 变量 | 默认 | 含义 |
|------|------|------|
| `NCCL_RAS_TIMEOUT_FACTOR` | 1 | 放大 RAS 集合超时 |

---

## 11. 应用集成建议

### 11.1 最小 FT 骨架

```text
1. ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
   config.blocking = 0;

2. ncclCommInitRankConfig(..., &config);
   while GetAsyncError == InProgress && !timeout;

3. 每次集体后 / 等待完成时:
   轮询 cudaStreamQuery + ncclCommGetAsyncError

4. 任一 rank 判定失败:
   同步 global abort 意图（应用自己的 allreduce/广播）
   所有健康 rank: ncclCommAbort(comm)

5. 恢复策略:
   A) 全员重启 job + 新 Init
   B) Shrink(ABORT) 去掉坏 rank，继续（业务需支持）
```

### 11.2 多机多网卡集群

```text
# 在可接受“重传/换轨”开销时评估打开:
export NCCL_IB_RESILIENCY_PORT_FAILOVER=1
# 需要端口自愈时再开:
# export NCCL_IB_RESILIENCY_PORT_RECOVERY=1
```

需保证拓扑上 **多设备/多 rail** 才有意义。

### 11.3 运维观测

- 训练侧：asyncError + 超时  
- 节点侧：`ncclras` / RAS 日志（`NCCL_DEBUG` 含 RAS）  
- 网络侧：IB 计数、链路 down、CQE vendor_err  

### 11.4 常见误区

1. **只 Abort 故障 rank、其它 rank 不 Abort** → 易死锁/资源泄漏。  
2. **blocking comm + 网络 hang 时期望 Abort** → 不可靠。  
3. **Shrink 后以为梯度状态自动一致** → 否，框架要处理。  
4. **打开 resiliency = 进程 FT** → 否，只是传输层弹性。  
5. **RAS 会自动修数据面** → 否，主要是诊断与通知。

---

## 12. 结论

NCCL 的 fault tolerance 是 **分层组合**：

```text
① 错误模型 + 异步探测     →  应用能发现“comm 已坏”
② abortFlag 全栈传播       →  GPU/proxy/socket 能停下来
③ Abort / Revoke / Shrink  →  资源释放与通信域降级重建
④ IB resiliency（可选）    →  多 rail 下设备/端口级自愈
⑤ RAS（可选旁路）          →  可观测、dead peer、运维查询
```

**设计哲学：**

- 数据面追求性能，**不在关键路径做重型共识 FT**  
- 故障时强调 **快速失败（fail fast）+ 干净中止 + 上层重建**  
- 传输层 resiliency 是 **增值、默认关** 的局部自愈  

对训练作业：NCCL FT 是 **通信容错积木**；完整 job FT 还需要 **弹性调度、checkpoint、rank 映射与状态恢复**。

---

## 13. 源码索引与修订记录

### 13.1 源码索引

| 主题 | 路径 |
|------|------|
| Abort / Destroy / Revoke / Shrink | `src/init.cc` |
| Group 与 abortFlag | `src/group.cc` |
| 设备 checkAbort | `src/device/primitives.h`、`prims_*.h` |
| abortFlag 分配 | `src/init.cc`（commAlloc 路径） |
| IB resiliency | `src/transport/net_ib/p2p_resiliency.cc` |
| IB port recovery | `src/transport/net_ib/p2p_resiliency_recovery.cc` |
| resiliency 挂接 | `src/transport/net_ib/common.cc`、`p2p.cc` |
| RAS | `src/ras/ras.cc`、`rasnet.cc`、`peers.cc`、`collectives.cc` |
| 用户文档 FT | `docs/userguide/source/usage/communicators.rst`（Fault Tolerance） |
| API 说明 | `docs/userguide/source/api/comms.rst`、`flags.rst` |

### 13.2 修订记录

| 日期 | 内容 |
|------|------|
| 2026-07-10 | 初稿：汇总 NCCL 应用级 FT API、abort 传播、shrink/revoke、IB resiliency、RAS 与能力边界 |
