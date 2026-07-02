# NCCL 环境变量参数汇总

本文档整理自 NCCL 源码中所有 `NCCL_PARAM(...)` / `DEFINE_NCCL_PARAM(...)` 定义。这些参数均通过环境变量 `NCCL_<NAME>` 进行配置，用于调试、调优、功能开关以及故障恢复等场景。

> 生成说明：基于 `src/` 目录下所有 `NCCL_PARAM`/`DEFINE_NCCL_PARAM` 定义逐条阅读 surrounding code 与注释整理，按功能模块分组。

---

## 目录

1. [Debug / 日志](#debug--日志)
2. [初始化 / Runtime / 通用配置](#初始化--runtime--通用配置)
3. [拓扑 / 图搜索 / 调优](#拓扑--图搜索--调优)
4. [传输层通用（P2P / SHM / NVLS / Socket / Net）](#传输层通用)
5. [InfiniBand 传输](#infiniband-传输)
6. [GIN / GDAKI](#gin--gdaki)
7. [RMA](#rma)
8. [Proxy / 调度 / 入队 / 内核](#proxy--调度--入队--内核)
9. [内存 / 注册 / 分配器](#内存--注册--分配器)
10. [插件引用计数](#插件引用计数)
11. [Bootstrap / RAS / 杂项](#bootstrap--ras--杂项)

---

## Debug / 日志

| 环境变量名 | 源文件:行号 | 默认值 | 作用/含义 |
|---|---|---|---|
| NCCL_DEBUG | src/debug.cc:45 | NCCL_LOG_NONE | 设置 NCCL 调试日志输出级别（VERSION/WARN/INFO/ABORT/TRACE），级别包含所有更低 verbosity 的输出 |
| NCCL_DEBUG_SUBSYS | src/debug.cc:55 | NCCL_INIT \| NCCL_BOOTSTRAP \| NCCL_ENV | 按子系统过滤调试输出，逗号分隔的 INIT/COLL/P2P/SHM/NET/GRAPH/TUNING/ENV/ALLOC 等位掩码 |
| NCCL_WARN_ENABLE_DEBUG_INFO | src/debug.cc:80 | false | 为 true 时，在输出 WARN 级别日志后自动将调试级别提升为 INFO |
| NCCL_DEBUG_TIMESTAMP_LEVELS | src/debug.cc:85 | (1u << NCCL_LOG_WARN) | 控制哪些日志级别前会打印时间戳，按位掩码选择 VERSION/WARN/INFO/ABORT/TRACE/ALL |
| NCCL_DEBUG_TIMESTAMP_FORMAT | src/debug.cc:99 | "[%F %T] " | 设置调试日志时间戳的打印格式 |
| NCCL_DEBUG_FILE | src/debug.cc:103 | nullptr | 将 NCCL 调试日志重定向到文件；支持 filename.%h.%p 格式替换 hostname 与 pid |
| NCCL_SET_THREAD_NAME | src/debug.cc:441 | false | 为 true 时允许 NCCL 通过 pthread_setname_np 给 CPU 线程设置有意义的名称 |
| NCCL_NVTX_DISABLE | src/init_nvtx.cc:16 | 0 | 是否禁用 NVTX 枚举注册 |
| NCCL_PARAM_DUMP_ALL | src/param/param.cc:16 | false | 为 true 时，ncclParamDumpAll()/ncclparam 工具会列出包括私有参数在内的所有参数 |
| NCCL_NO_CACHE | src/param/param.cc:20 | nullptr | 逗号分隔的参数键列表（或 ALL），用于禁用对应参数的缓存，使每次读取都重新解析环境变量 |

---

## 初始化 / Runtime / 通用配置

| 环境变量名 | 源文件:行号 | 默认值 | 作用/含义 |
|---|---|---|---|
| NCCL_GROUP_CUDA_STREAM | src/init.cc:57 | NCCL_GROUP_CUDA_STREAM (0/1) | 控制组调用时是否使用内部 CUDA stream（CGMD 相关） |
| NCCL_CHECK_POINTERS | src/init.cc:59 | 0 | 是否启用指针检查（debug/local 模式） |
| NCCL_COMM_BLOCKING | src/init.cc:60 | NCCL_CONFIG_UNDEF_INT (INT_MIN) | 覆盖 communicator 的 blocking 配置 |
| NCCL_RUNTIME_CONNECT | src/init.cc:61 | 1 | 在支持 cuMem 时启用 runtime connect |
| NCCL_WIN_ENABLE | src/init.cc:62 | 1 | 是否启用 symmetric memory window（对称内存窗口） |
| NCCL_COLLNET_ENABLE | src/init.cc:63 | NCCL_CONFIG_UNDEF_INT (INT_MIN) | 是否启用 CollNet/SHARP 插件（覆盖 config.collnetEnable） |
| NCCL_NVLS_NCHANNELS | src/init.cc:64 | NCCL_CONFIG_UNDEF_INT (INT_MIN) | 覆盖 NVLS 算法使用的通道数（CTAs） |
| NCCL_NUM_RMA_CTX | src/init.cc:65 | NCCL_CONFIG_UNDEF_INT (INT_MIN) | 设置每个 communicator 的 RMA 上下文数量 |
| NCCL_P2P_MAX_PEERS | src/init.cc:66 | NCCL_CONFIG_UNDEF_INT (INT_MIN) | 覆盖最大 P2P peer 数量 |
| NCCL_SET_CPU_STACK_SIZE | src/init.cc:67 | 1 | 是否在 OS 初始化时设置 pthread 默认栈大小 |
| NCCL_MULTI_RANK_GPU_ENABLE | src/init.cc:68 | 0 | 是否允许多个 rank 使用同一 GPU/分区 |
| NCCL_GDRCOPY_ENABLE | src/init.cc:130 | 0 | 是否初始化 GDRCOPY 库 |
| NCCL_IGNORE_NET_MISMATCH | src/init.cc:131 | 1 | 是否忽略各 rank 间 Net 设备数量不一致 |
| NCCL_IGNORE_COLLNET_MISMATCH | src/init.cc:132 | 0 | 是否忽略各 rank 间 CollNet 设备数量不一致 |
| NCCL_GRAPH_HELPER_DISABLE | src/init.cc:391 | 0 | 当前源码树中未见实际使用（保留/废弃参数） |
| NCCL_GDRCOPY_FIFO_ENABLE | src/init.cc:393 | 1 | 是否将 work FIFO 分配到 GDR 映射的 CUDA 内存 |
| NCCL_WORK_FIFO_BYTES | src/init.cc:395 | NCCL_WORK_FIFO_BYTES_DEFAULT (1 MiB) | work FIFO 的字节大小 |
| NCCL_WORK_ARGS_BYTES | src/init.cc:396 | INT64_MAX | 传给内核的 work 参数最大字节数 |
| NCCL_DMABUF_ENABLE | src/init.cc:399 | 1 | 是否使用 DMA-BUF 进行内存注册 |
| NCCL_MNNVL_UUID | src/init.cc:707 | -1 | 覆盖 MNNVL 128-bit fabric UUID（-1 表示不覆盖） |
| NCCL_MNNVL_CLIQUE_ID | src/init.cc:708 | -1 | 覆盖/派生 MNNVL clique ID（-2 派生，-1 不覆盖） |
| NCCL_MNNVL_CROSS_CLIQUE | src/init.cc:709 | 0 | 是否启用跨 clique 的 P2P |
| NCCL_BUFFSIZE | src/init.cc:814 | -2 | 覆盖 SIMPLE 协议的 buffer 大小（-2 使用默认 4 MiB） |
| NCCL_LL_BUFFSIZE | src/init.cc:815 | -2 | 覆盖 LL 协议的 buffer 大小（-2 使用内部默认） |
| NCCL_LL128_BUFFSIZE | src/init.cc:816 | -2 | 覆盖 LL128 协议的 buffer 大小（-2 使用内部默认） |
| NCCL_P2P_NET_CHUNKSIZE | src/init.cc:818 | 1<<17 (128 KiB) | 跨节点 P2P 使用的 chunk 大小 |
| NCCL_P2P_PCI_CHUNKSIZE | src/init.cc:819 | 1<<17 (128 KiB) | PCIe P2P 使用的 chunk 大小 |
| NCCL_P2P_NVL_CHUNKSIZE | src/init.cc:820 | 1<<19 (512 KiB) | NVLink P2P 使用的 chunk 大小 |
| NCCL_GRAPH_DUMP_FILE_RANK | src/init.cc:856 | 0 | 指定哪个 rank 将拓扑图 dump 到文件 |
| NCCL_COLLNET_NODE_THRESHOLD | src/init.cc:857 | 2 | 启用 CollNet 所需的最小节点数 |
| NCCL_NVB_PRECONNECT | src/init.cc:858 | 1 | 是否对 NVB 路径预建立 P2P 连接 |
| NCCL_ALLOC_P2P_NET_LL_BUFFERS | src/init.cc:859 | 0 | 是否为 P2P 网络分配 LL 缓冲区 |
| NCCL_MNNVL_ENABLE | src/init.cc:862 | 2 | 是否启用 Multi-Node NVLink（0 禁用，1 强制，2 自动） |
| NCCL_P2P_SCHEDULE_GROUP_SIZE | src/init.cc:888 | NCCL_MAX_DEV_WORK_P2P_PER_BATCH (8) | MNNVL 下 P2P 调度分组大小，用于 PXN 聚合 |
| NCCL_SET_STACK_SIZE | src/init.cc:1722 | 0 | 是否将 CUDA 内核栈大小设为最大局部内存大小 |
| NCCL_CGA_CLUSTER_SIZE | src/init.cc:1723 | NCCL_CONFIG_UNDEF_INT (INT_MIN) | 覆盖 CGA cluster 大小配置 |
| NCCL_MAX_CTAS | src/init.cc:1725 | NCCL_CONFIG_UNDEF_INT (INT_MIN) | 覆盖最大 CTA/通道数配置 |
| NCCL_MIN_CTAS | src/init.cc:1726 | NCCL_CONFIG_UNDEF_INT (INT_MIN) | 覆盖最小 CTA/通道数配置 |
| NCCL_NCHANNELS_PER_NET_PEER | src/init.cc:1729 | NCCL_CONFIG_UNDEF_INT (INT_MIN) | 覆盖每个网络 peer 的通道数 |
| NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE | src/init.cc:1730 | NCCL_CONFIG_UNDEF_INT (INT_MIN) | 是否启用面向 NVLink 利用率的调度 |
| NCCL_GRAPH_MIXING_SUPPORT | src/init.cc:1731 | NCCL_CONFIG_UNDEF_INT (INT_MIN) | 是否启用 CUDA Graph 混合支持（1 启用，0 禁用） |
| NCCL_COMM_SPLIT_SHARE_RESOURCES | src/init.cc:1765 | NCCL_CONFIG_UNDEF_INT (INT_MIN) | 覆盖 comm split 资源共享配置 |
| NCCL_COMM_SHRINK_SHARE_RESOURCES | src/init.cc:1766 | NCCL_CONFIG_UNDEF_INT (INT_MIN) | 覆盖 comm shrink 资源共享配置 |
| NCCL_WIN_STRIDE | src/dev_runtime.cc:28 | -1 | 对称内存窗口每 rank 的 VA stride（-1 按最大 peer 显存自动计算） |
| NCCL_ENABLE_VERSION_CHECK | src/dev_runtime.cc:29 | 1 | 当应用编译版本高于运行时版本时是否报错 |
| NCCL_ELASTIC_BUFFER_REGISTER | src/dev_runtime.cc:30 | 1 | 是否允许注册包含 CPU 物理段（sysmem）的弹性缓冲区 |
| NCCL_SYM_REUSE_SYSMEM_HANDLES | src/dev_runtime.cc:31 | 0 | 对称内存导入时是否复用本地 sysmem allocation handles |
| NCCL_RMA_DISABLE | src/dev_runtime.cc:32 | 0 | 是否禁用 RMA proxy 路径 |
| NCCL_LSA_TEAM_SIZE | src/dev_runtime.cc:76 | 0 | 覆盖 LSA team 大小（0 表示从节点大小 gcd 自动计算） |
| NCCL_CUMEM_ENABLE | src/misc/cudawrap.cc:16 | -2 | 控制是否使用 cuMem API；-2 自动检测平台支持，-1 强制禁用，≥0 直接覆盖 |
| NCCL_CUMEM_HOST_ENABLE | src/misc/cudawrap.cc:17 | -1 | 控制是否使用 cuMem 进行 host 内存分配；-1 根据驱动版本自动决定（CUDA 12.60+ 默认启用），0/1 强制关闭/开启 |
| NCCL_LAUNCH_RACE_FATAL | src/misc/strongstream.cc:156 | 1 | 为 1 时，当多个 host 线程竞争在同一个 device/捕获流上发起 NCCL launch 时返回 ncclInvalidUsage 并警告；为 0 时仅继续执行 |

---

## 拓扑 / 图搜索 / 调优

| 环境变量名 | 源文件:行号 | 默认值 | 作用/含义 |
|---|---|---|---|
| NCCL_CROSS_NIC | src/graph/search.cc:16 | 2 | 是否允许 ring/tree 跨节点使用不同 NIC（0 禁用，1 允许，2 自动优化） |
| NCCL_MNNVL_SCATTER_NETS_ENABLE | src/graph/search.cc:540 | 1 | MNNVL 选网时是否按 GPU 优先、通道次之的顺序分散 NIC |
| NCCL_MNNVL_RAIL_PER_HOST | src/graph/search.cc:591 | 0 | MNNVL 系统中是否按主机区分 rail（使用 PCI id/port 匹配） |
| NCCL_P2P_PXN_LEVEL | src/graph/search.cc:1398 | 2 | P2P 操作中使用 PXN 的级别（0 禁用，1 需要时，2 最大化聚合） |
| NCCL_TOPO_SPLIT_MLOPART | src/graph/topo.cc:588 | 1 | 拓扑发现时是否按 MLO partition 拆分 GPU |
| NCCL_TOPO_DUMP_FILE_RANK | src/graph/topo.cc:992 | 0 | 指定哪个 rank 将拓扑 XML dump 到 NCCL_TOPO_DUMP_FILE |
| NCCL_IGNORE_CPU_AFFINITY | src/graph/topo.cc:2148 | 0 | 获取 CPU affinity 时是否忽略当前 CPU affinity 掩码 |
| NCCL_NTHREADS | src/graph/tuning.cc:14 | -2 | 覆盖 SIMPLE/LL 协议的线程数（-2 使用内部默认值） |
| NCCL_LL128_NTHREADS | src/graph/tuning.cc:15 | -2 | 覆盖 LL128 协议的线程数（-2 使用内部默认值） |
| NCCL_PAT_ENABLE | src/graph/tuning.cc:215 | 2 | 是否启用 PAT 算法（0 禁用，1 启用，2 自动） |
| NCCL_NET_OVERHEAD | src/graph/tuning.cc:226 | -2 | 网络后处理开销（ns，-2 按 CPU 架构取默认值） |
| NCCL_LL128_C2C | src/graph/tuning.cc:235 | 1 | 在 Hopper+ 上是否允许 LL128 使用 C2C/PxN 路径 |
| NCCL_NVB_DISABLE | src/graph/paths.cc:36 | 0 | 是否禁用 NVLink Bridge (NVB) 路径 |
| NCCL_IGNORE_DISABLED_P2P | src/graph/paths.cc:282 | 0 | 当 NVML 报告 P2P 被禁用时是否忽略（0 警告/报错，1 忽略，2 强制检查） |
| NCCL_NET_GDR_READ | src/graph/paths.cc:460 | -2 | 是否对 send/read 启用 GDR（0 禁用，1 启用，-2 自动） |
| NCCL_NET_GDR_C2C | src/graph/paths.cc:465 | 1 | 是否允许 C2C 连接的 NIC 使用 GDR |
| NCCL_NET_GDR_MLOPART | src/graph/paths.cc:466 | 0 | 是否允许 MLO-partitioned GPU 使用 GDR |
| NCCL_NET_FORCE_FLUSH | src/graph/paths.cc:572 | 0 | 是否在 GDR recv 路径上强制 flush |
| NCCL_NET_DISABLE_INTRA | src/graph/paths.cc:599 | 0 | 是否禁用网络作为 intra-node 路径 |
| NCCL_PXN_DISABLE | src/graph/paths.cc:668 | 0 | 是否禁用跨节点 PXN（proxy-X）机制 |
| NCCL_PXN_C2C | src/graph/paths.cc:719 | 1 | 是否允许通过 C2C+PCIe 使用 PXN |
| NCCL_P2P_PER_CHANNEL_NET_BW | src/graph/paths.cc:918 | 14 (GB/s) | 计算 P2P 网络通道数时假设的每通道网络带宽 |
| NCCL_MIN_P2P_NCHANNELS | src/graph/paths.cc:957 | 1 | P2P 通道数下限 |
| NCCL_MAX_P2P_NCHANNELS | src/graph/paths.cc:958 | MAXCHANNELS (64) | P2P 通道数上限 |
| NCCL_MIN_NRINGS | src/graph/connect.cc:328 | -2 | 遗留命名：最小 channel/ring 数量（-2 未设置） |
| NCCL_MAX_NRINGS | src/graph/connect.cc:329 | -2 | 遗留命名：最大 channel/ring 数量（-2 未设置） |
| NCCL_MIN_NCHANNELS | src/graph/connect.cc:331 | -2 | 最小 channel 数量（-2 未设置） |
| NCCL_MAX_NCHANNELS | src/graph/connect.cc:332 | -2 | 最大 channel 数量（-2 未设置） |
| NCCL_UNPACK_DOUBLE_NCHANNELS | src/graph/connect.cc:378 | 1 | 使用 unpack 网络且多节点时是否将 channel 数翻倍 |
| NCCL_CONNECT_ROUND_MAX_PEERS | src/transport.cc:75 | 128 | P2P 连接建立时每轮处理的最大 peer 数 |
| NCCL_REPORT_CONNECT_PROGRESS | src/transport.cc:76 | 0 | 是否在 rank 0 上打印 P2P 连接进度 |

---

## 传输层通用

| 环境变量名 | 源文件:行号 | 默认值 | 作用/含义 |
|---|---|---|---|
| NCCL_LEGACY_CUDA_REGISTER | src/transport/p2p.cc:104 | 0 | 当 cuMem 句柄保留/导出失败或走 legacy IPC 路径时，决定是否回退到 legacy `cudaIpcGetMemHandle` 进行缓冲区注册；若为 0 且 directMode 或 cuMem 失败则直接注册失败 |
| NCCL_P2P_USE_CUDA_MEMCPY | src/transport/p2p.cc:123 | 0 | 启用基于 CE (Copy Engine) `cudaMemcpyAsync` 的 P2P 传输路径，替代直接 GPU P2P 访问；代理线程通过共享内存执行拷贝 |
| NCCL_P2P_READ_ENABLE | src/transport/p2p.cc:327 | -2 | 控制 P2P 是否使用读操作而非写操作；-2 为自动模式（由拓扑决定，如 Ampere+NVLink 默认启用 read），其他值覆盖 |
| NCCL_P2P_DIRECT_DISABLE | src/transport/p2p.cc:328 | 0 | 非零时禁用同 PID 进程内直接指针 P2P 路径，强制使用 IPC/cuMem 句柄交换 |
| NCCL_SHM_DISABLE | src/transport/shm.cc:55 | 0 | 设为 1 时禁用共享内存 (SHM) 传输 |
| NCCL_SHM_LOCALITY | src/transport/shm.cc:56 | 2 | 决定共享内存缓冲区由哪一侧分配：1 为发送方侧，2 为接收方侧（默认） |
| NCCL_NVLS_ENABLE | src/transport/nvls.cc:159 | 2 | 控制 NVLink SHARP (NVLS) 启用：0=禁用，1=强制启用，2=通过 `CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED` 自动检测 |
| NCCL_NVLS_CHUNKSIZE | src/transport/nvls.cc:160 | 131072 | NVLS 操作的默认分块大小（字节） |
| NCCL_NVLSTREE_MAX_CHUNKSIZE | src/transport/nvls.cc:161 | -2 | NVLS Tree 算法的最大分块大小；-2 表示使用调优/回退默认值，其他值显式覆盖 |
| NCCL_MULTI_SEGMENT_REGISTER | src/transport/generic.cc:12 | 1 | 允许 P2P/网络/NVLS 注册跨多个物理内存段的缓冲区；为 0 时遇到多段缓冲区直接跳过注册 |
| NCCL_NET_SHARED_BUFFERS | src/transport/net.cc:171 | -2 | 控制 P2P 网络缓冲区是否共享：-2 自动（默认共享），0 专用，1 共享 |
| NCCL_NET_SHARED_COMMS | src/transport/net.cc:172 | 1 | 当网卡 `maxRecvs > 1` 时，允许同一网卡/对端秩的通道间复用共享网络连接 |
| NCCL_NET_OPTIONAL_RECV_COMPLETION | src/transport/net.cc:187 | 1 | 对 LL/LL128 协议，当只下发单个子请求时，代理可跳过等待网络 recv 完成 |
| NCCL_GDRCOPY_SYNC_ENABLE | src/transport/net.cc:339 | 1 | GDRCOPY 支持：将 RX 代理 tail 放在 CUDA 内存中 |
| NCCL_GDRCOPY_FLUSH_ENABLE | src/transport/net.cc:341 | 0 | GDRCOPY 支持：使用 PCIe read 刷新 GDRDMA 缓冲区 |
| NCCL_SOCKET_INLINE | src/transport/net_socket.cc:196 | 128 | Socket 传输内联阈值：小于等于该大小的数据随控制消息一起内联发送 |
| NCCL_SOCKET_MIN_TASKSIZE | src/transport/net_socket.cc:197 | 65536 | 将一次 socket 传输拆分到多个 socket/线程时的最小任务大小 |
| NCCL_NSOCKS_PERTHREAD | src/transport/net_socket.cc:198 | -2 | 每个辅助线程使用的 socket 数；-2 根据网卡 vendor 自动检测 |
| NCCL_SOCKET_NTHREADS | src/transport/net_socket.cc:199 | -2 | Socket 辅助线程数；-2 根据网卡 vendor 自动检测 |
| NCCL_IPC_USE_ABSTRACT_SOCKET | src/os/linux_ipcsocket.cc:21 | 1 | 使用 Linux 抽象 Unix Domain Socket 进行 IPC handle/fd 交换；0 则使用基于文件系统的普通 UDS |
| NCCL_OOB_NET_ENABLE | src/bootstrap.cc:103 | 0 | 使用网络插件（而非 socket）进行带外 bootstrap 环网通信 |
| NCCL_UID_STAGGER_RATE | src/bootstrap.cc:669 | 7000 | 控制 bootstrap 连接到 root 时的错峰速率（messages/sec） |
| NCCL_UID_STAGGER_THRESHOLD | src/bootstrap.cc:670 | 256 | 当某个 root 下的 rank 数超过该阈值时才启用 bootstrap 连接错峰 |
| NCCL_RAS_ENABLE | src/bootstrap.cc:672 | 1 | bootstrap 阶段启用 RAS (Remote Access Service / telemetry) 客户端初始化 |

---

## InfiniBand 传输

| 环境变量名 | 源文件:行号 | 默认值 | 作用/含义 |
|---|---|---|---|
| NCCL_IB_AR_THRESHOLD | src/transport/net_ib/p2p.cc:13 | -2 | 自适应路由（AR）下触发“先发送 bulk RDMA Write，再发带 immediate 的尾包”的大小阈值；`-2` 表示使用内部默认值 `8192` |
| NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME | src/transport/net_ib/p2p.cc:17 | -2 | 接收端请求匹配方式；`-2` 表示默认按索引（`BY_INDEX`）匹配，可覆盖为 `BY_ID`；启用 OOO RQ 或端口故障转移时强制为 `BY_ID` |
| NCCL_IB_PCI_RELAXED_ORDERING | src/transport/net_ib/init.cc:11 | 2 | 控制是否使用 PCI Relaxed Ordering；`0` 禁用，`1` 强制启用，`2` 自动探测能力 |
| NCCL_IB_ADAPTIVE_ROUTING | src/transport/net_ib/init.cc:12 | -2 | 控制 IB 自适应路由；`-2` 表示按链路类型自动决定（IB 默认启用，RoCE 默认禁用），其它值强制开关 |
| NCCL_IB_DATA_DIRECT | src/transport/net_ib/init.cc:13 | 1 | 控制 mlx5 Data Direct DMA 接口探测；`0` 禁用，`1` 只暴露 Data Direct NIC，`2` 同时暴露 C2C+PCIe 与 Data Direct |
| NCCL_IB_OOO_RQ | src/transport/net_ib/init.cc:16 | 0 | 是否启用 mlx5 乱序接收队列（Out-of-Order RQ）；启用要求 mlx5 provider 且 AR 开启 |
| NCCL_IB_DISABLE | src/transport/net_ib/init.cc:27 | 0 | 设为 `1` 时直接禁用整个 IB 网络插件 |
| NCCL_IB_MERGE_VFS | src/transport/net_ib/init.cc:28 | 1 | 是否将同一物理 NIC 的多个 Virtual Function 合并为同一设备 |
| NCCL_IB_MERGE_NICS | src/transport/net_ib/init.cc:29 | 1 | 是否将多个物理 NIC 合并成虚拟设备（vDevice） |
| NCCL_IB_DEVICE_PCI_ORDER | src/transport/net_ib/init.cc:30 | 1 | 是否按 PCI 路径排序设备，以保证多节点间设备顺序一致 |
| NCCL_IB_GID_INDEX | src/transport/net_ib/connect.cc:12 | -1 | RoCE 下使用的 GID 索引；`-1` 表示自动探测 |
| NCCL_IB_ROUTABLE_FLID_GID_INDEX | src/transport/net_ib/connect.cc:13 | 1 | IB 网络下优先尝试该 GID 索引以获取可路由 FLID |
| NCCL_IB_ROCE_VERSION_NUM | src/transport/net_ib/connect.cc:14 | 2 | RoCE GID 自动选择时匹配的 RoCE 版本号 |
| NCCL_IB_TIMEOUT | src/transport/net_ib/connect.cc:15 | 20 | IB QP RTS 阶段的 ACK 超时值 |
| NCCL_IB_RETRY_CNT | src/transport/net_ib/connect.cc:16 | 7 | IB QP RTS 阶段的重试次数 |
| NCCL_IB_PKEY | src/transport/net_ib/connect.cc:17 | 0 | IB QP INIT 阶段的 PKey 索引 |
| NCCL_IB_USE_INLINE | src/transport/net_ib/connect.cc:18 | 0 | 是否为 CTS FIFO 发送启用 inline 数据 |
| NCCL_IB_SL | src/transport/net_ib/connect.cc:19 | -1 | IB Service Level；`-1` 表示从 traffic class 推导或使用默认值 `0` |
| NCCL_IB_TC | src/transport/net_ib/connect.cc:20 | -1 | IB Traffic Class；`-1` 表示从 NCCL 网络上下文推导或使用默认值 `0` |
| NCCL_IB_FIFO_TC | src/transport/net_ib/connect.cc:21 | -1 | RoCE 下 FIFO QP 的 Traffic Class；`-1` 表示使用对端传来的 `tc` |
| NCCL_IB_ECE_ENABLE | src/transport/net_ib/connect.cc:22 | 1 | 是否启用 Enhanced Connection Establishment（ECE）协商 |
| NCCL_IB_QPS_PER_CONNECTION | src/transport/net_ib/connect.cc:60 | 1 | 每个连接在每个设备上创建的 QP 数量 |
| NCCL_IB_SUBNET_AWARE_ROUTING | src/transport/net_ib/connect.cc:61 | 0 | 是否启用多子网 RoCE 的子网感知路由，自动选择能到达对端子网的本地设备 |
| NCCL_IB_SUBNET_PREFIX_LEN | src/transport/net_ib/connect.cc:62 | 24 | 子网感知路由中用于匹配本地/远端 GID 子网的前缀长度 |
| NCCL_IB_WARN_RAIL_LOCAL | src/transport/net_ib/connect.cc:1099 | 0 | 当本地与远端 vDevice 物理设备不一致时是否打印提示信息 |
| NCCL_GDR_FLUSH_DISABLE | src/transport/net_ib/connect.cc:1354 | 0 | 设为 `1` 时禁用 GPU Direct RDMA 的 flush 机制 |
| NCCL_IB_SPLIT_DATA_ON_QPS | src/transport/net_ib/common.cc:22 | 0 | 是否将单个发送请求的数据切分到多个 QP 上传输 |
| NCCL_IB_PREPOST_RECEIVE_WORK_REQUESTS | src/transport/net_ib/common.cc:23 | -2 | 是否在接收端预先 post receive WQE；`-2` 表示默认不预 post，启用 OOO RQ 或弹性时会强制开启 |
| NCCL_IB_RETURN_ASYNC_EVENTS | src/transport/net_ib/common.cc:24 | 1 | 是否将 async fatal error 返回给上层调用者；`0` 时可能忽略 |
| NCCL_IB_RESILIENCY_PORT_FAILOVER | src/transport/net_ib/p2p_resiliency.cc:13 | 0 | 是否启用 IB 端口故障转移弹性机制 |
| NCCL_IB_RESILIENCY_PORT_FAILOVER_MAX_ATTEMPTS | src/transport/net_ib/p2p_resiliency.cc:14 | 1 | 故障发送请求探测失败的最大尝试次数，超过则报 fatal error |
| NCCL_IB_RESILIENCY_PORT_FAILOVER_PROBE_DELAY | src/transport/net_ib/p2p_resiliency.cc:15 | 10 | 发送请求出错后到发起探测的最小等待时间（毫秒） |
| NCCL_IB_RESILIENCY_PORT_RECOVERY | src/transport/net_ib/p2p_resiliency_recovery.cc:10 | 0 | 是否启用 IB 端口恢复后台异步线程 |
| NCCL_IB_RESILIENCY_PORT_RECOVERY_START_DELAY | src/transport/net_ib/p2p_resiliency_recovery.cc:11 | 200 | 端口恢复上下文初始化后到真正开始恢复协议的延迟（毫秒） |
| NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_INTERVAL | src/transport/net_ib/p2p_resiliency_recovery.cc:13 | 500 | 两次 alive 消息批次之间的最小间隔（毫秒） |
| NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE | src/transport/net_ib/p2p_resiliency_recovery.cc:14 | 5 | 恢复 QP 上 alive 消息批次的容量（实际受内部上限 10 限制） |
| NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE | src/transport/net_ib/p2p_resiliency_recovery.cc:15 | 5 | 接收端判定端口恢复所需连续收到的 alive 消息数量 |
| NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_TIMEOUT | src/transport/net_ib/p2p_resiliency_recovery.cc:17 | 4000 | 接收端等待 alive 消息的超时时间（毫秒） |
| NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_TIMEOUT | src/transport/net_ib/p2p_resiliency_recovery.cc:18 | 5000 | 等待 ack / final ack 回复的超时时间（毫秒） |
| NCCL_IB_RESILIENCY_PORT_RECOVERY_ATTEMPTS_MAX | src/transport/net_ib/p2p_resiliency_recovery.cc:19 | 5 | 端口恢复协议失败重试次数上限，超过则标记恢复失败 |
| NCCL_IB_MQP_RETRY_ALL | src/misc/ibvwrap.cc:98 | 0 | `ibv_modify_qp` 失败时是否对所有 errno 都重试；`0` 时仅对 `ETIMEDOUT` 重试 |
| NCCL_IB_MQP_RETRY_CNT | src/misc/ibvwrap.cc:99 | 34 | `ibv_modify_qp` 失败后的最大重试次数 |
| NCCL_IB_MQP_RETRY_SLEEP_MSEC | src/misc/ibvwrap.cc:100 | 100 | `ibv_modify_qp` 重试间隔的基准睡眠时间（毫秒），实际睡眠时长为 `timeout * attempts` |
| NCCL_IB_QUERY_PORT_SPEED | src/misc/ibvwrap.cc:101 | 1 | 是否使用 `ibv_query_port_speed` 获取链路速度 |

---

## GIN / GDAKI

| 环境变量名 | 源文件:行号 | 默认值 | 作用/含义 |
|---|---|---|---|
| NCCL_GIN_ENABLE | src/gin/gin_host.cc:19 | 1 | 为 0 时禁用 GIN；ncclGinConnectOnce 会直接返回错误 |
| NCCL_DEV_API_JIT | src/gin/gin_host.cc:20 | 0 | 为 1 时 GIN 设备端代码版本选择最新版本（JIT 模式）；为 0 时根据请求版本匹配 |
| NCCL_GIN_NCONNECTIONS | src/gin/gin_host.cc:78 | -2 | 覆盖 GIN 连接数；-2 表示使用内部默认逻辑（旧版本固定为 NCCL_GIN_MAX_CONNECTIONS），其他值会覆盖 ginCommCount 并受最大连接数限制 |
| NCCL_GIN_PROXY_QUEUE_SIZE | src/gin/gin_host_proxy.cc:20 | -1 | GIN proxy 队列大小；-1 表示使用 NCCL_NET_MAX_REQUESTS * maxRecvs 的默认值，且不能超过该上限 |
| NCCL_GIN_TYPE | src/transport/net_ib/gin.cc:46 | -1 | 强制指定 GIN IB 后端类型；`-1` 表示使用默认/自动选择，非 `-1` 且非 GDAKI 时会跳过 GDAKI 设备初始化 |
| NCCL_GIN_IB_TC | src/transport/net_ib/gin.cc:47 | -1 | 覆盖 GIN IB 的 InfiniBand Traffic Class；`-1` 时回退到 `NCCL_IB_TC`，用于 QP 连接和 GDAKI/RMA 代理上下文创建 |
| NCCL_GIN_GDAKI_NIC_HANDLER | src/transport/net_ib/gdaki/gin_host_gdaki.cc:54 | 0 | 设置 DOCA GPU verbs QP 的 NIC handler 模式 |
| NCCL_GIN_GDAKI_QP_DEPTH | src/transport/net_ib/gdaki/gin_host_gdaki.cc:55 | 128 | 设置 GDAKI QP 的 send queue 深度；当传入的 `queueDepth <= 0` 时生效 |
| NCCL_GIN_GDAKI_MAX_DEST_RD_ATOMIC | src/transport/net_ib/gdaki/gin_host_gdaki.cc:56 | -2 | 设置 GDAKI QP 的 `max_dest_rd_atomic`；`>0` 时生效，否则使用设备属性中的最大值 |
| NCCL_GIN_GDAKI_MAX_QP_RD_ATOMIC | src/transport/net_ib/gdaki/gin_host_gdaki.cc:57 | -2 | 设置 GDAKI QP 的 `max_rd_atomic`；`>0` 时生效，否则使用设备属性中的最大值 |
| NCCL_GIN_ERROR_QUERY_SEC | src/transport/net_ib/gdaki/gin_host_gdaki.cc:58 | 10 | 设置 GDAKI QP 错误查询的最小时间间隔（秒），用于限制 `query_last_error` 调用频率 |
| NCCL_GDAKI_USE_RELIABLE_DB | src/transport/net_ib/gdaki/gin_host_gdaki.cc:458 | 0 | 控制 GDAKI QP 是否使用可靠门铃（reliable doorbell）模式；`0` 关闭，`1` 启用并允许 SW 模拟回退，`2` 可进一步回退到 valid DBR |

---

## RMA

| 环境变量名 | 源文件:行号 | 默认值 | 作用/含义 |
|---|---|---|---|
| NCCL_RMA_PROXY_DUMP_SIGNAL | src/rma/rma_proxy.cc:26 | -1 | 设置为信号编号时，RMA proxy 进度线程注册 dump 处理函数以调试挂起；-1 表示禁用 |
| NCCL_RMA_PROXY_QUEUE_SIZE | src/rma/rma_proxy.cc:27 | -1 | RMA proxy 队列大小；-1 表示使用 NCCL_NET_MAX_REQUESTS * maxRecvs 的默认值，且不能超过该上限 |

---

## Proxy / 调度 / 入队 / 内核

| 环境变量名 | 源文件:行号 | 默认值 | 作用/含义 |
|---|---|---|---|
| NCCL_PROXY_APPEND_BATCH_SIZE | src/proxy.cc:833 | 16 | 限制每次从 proxy op pool 取出的连续操作批量大小，达到该数量后停止追加，避免单次处理过多操作 |
| NCCL_PROXY_DUMP_SIGNAL | src/proxy.cc:925 | -1 | 设置为 SIGUSR1/SIGUSR2 等信号时，proxy 进度线程会在挂起时注册信号处理函数以 dump proxy 状态；-1 表示禁用 |
| NCCL_PROGRESS_APPENDOP_FREQ | src/proxy.cc:926 | 8 | 控制 proxy 线程调用 ncclProxyGetPostedOps() 的频率；proxyOpAppendCounter 累加到该值时才去拉取新的 proxy 操作，降低小消息通信的性能开销 |
| NCCL_SYM_NOWIN_ENABLE | src/scheduler/symmetric_sched.cc:21 | 0 | 对称内存调度中，当为 0 时对于非注册窗口（non-reg send/recv）会 fallback 到传统调度；为 1 时允许其走对称路径 |
| NCCL_L1_SHARED_MEMORY_CARVEOUT | src/enqueue.cc:29 | 0 | 初始化 CUDA kernel 时通过 cudaFuncSetAttribute 设置的 preferred shared memory carveout 百分比；0 表示不设置 |
| NCCL_ALLGATHERV_ENABLE | src/enqueue.cc:30 | 1 | 为 1 时，Broadcast 集合通信在特定条件下会转换为 AllGatherV 任务执行 |
| NCCL_SYM_CE_THRESHOLD | src/enqueue.cc:31 | 8388608 (8 * 1024 * 1024) | AllGather 在满足对称内存支持、Blackwell、全 NVLink 直连等条件时，count 大于该阈值（字节数）则使用 Copy Engine (CE) 路径 |
| NCCL_P2P_EPOCH_ENABLE | src/enqueue.cc:119 | 1 | 为 1 时，P2P 操作在 batch 合并阶段会按 p2pEpoch 分隔，避免不同 epoch 的操作进入同一 batch 导致 hang |
| NCCL_GRAPH_REGISTER | src/enqueue.cc:283 | 1 | 为 1 时，CUDA Graph 捕获期间可将 buffer 视为已注册，从而影响 tuner 的 regBuff 判断与 NVLS/COLLNET 注册路径 |
| NCCL_P2P_LL_THRESHOLD | src/enqueue.cc:859 | 16384 | P2P 操作按每通道 payload 大小选择协议：小于等于该阈值时使用 LL 协议，否则使用 SIMPLE 协议 |
| NCCL_CHUNK_SIZE | src/enqueue.cc:860 | 0 | P2P 调度中用户覆盖的 chunk 大小；0 表示使用内部默认值 |
| NCCL_LAUNCH_ORDER_IMPLICIT | src/enqueue.cc:1538 | 0 | 为 1 时启用隐式 launch 顺序序列化；CUDA Graph 捕获且驱动低于 12090 时使用串行化，否则根据 CUDA/驱动版本选择 launch 顺序 |
| NCCL_GRAPH_STREAM_ORDERING | src/enqueue.cc:1539 | NCCL_CONFIG_UNDEF_INT | 覆盖 comm config 的 graphStreamOrdering；0 表示在 origin stream 上通过 external event 序列化，1 表示使用 captureStream 并内部序列化 |
| NCCL_MEM_SYNC_DOMAIN | src/enqueue.cc:1750 | cudaLaunchMemSyncDomainRemote | 在 sm90+、CUDA 12.0+ 时设置 kernel launch 的 mem sync domain，默认使用 Remote domain |
| NCCL_SYM_CTAS | src/sym_kernels.cc:131 | 0 | 用户覆盖对称 kernel 的 CTA（block）数量；0 表示由内部模型自动决定，≥1 时强制使用指定 block 数 |
| NCCL_SYM_GIN_KERNELS_ENABLE | src/sym_kernels.cc:132 | 1 | 为 1 时允许使用 GIN 相关的对称 kernel；在 kernel 掩码过滤时保留 GIN 路径 |
| NCCL_SYM_RS_GIN_CHUNK_SIZE | src/sym_kernels.cc:133 | -1 | ReduceScatter GIN 路径的 chunk 大小；-1 表示使用默认值 128 KiB，正数则覆盖并被限制在 [128, 1GiB] 范围内且取 2 的幂 |
| NCCL_SYM_TMA_ENABLE | src/sym_kernels.cc:134 | 0 | 为 1 且 compute capability ≥100 时允许对称 kernel 使用 TMA 指令相关的 kernel 变体 |

---

## 内存 / 注册 / 分配器

| 环境变量名 | 源文件:行号 | 默认值 | 作用/含义 |
|---|---|---|---|
| NCCL_SINGLE_PROC_MEM_REG_ENABLE | src/group.cc:536 | 0 | 是否为单进程内存注册启用独立预连接任务 |
| NCCL_DISABLE_MEM_MANAGER | src/mem_manager.cc:28 | 0 | 内部测试参数；为 1 时跳过 ncclMemManager 的初始化与销毁逻辑 |
| NCCL_SHADOW_MEMPOOL_MAX_SIZE | src/allocator.cc:14 | 1073741824 (1LL << 30) | 创建 shadow mempool 时设置的 maxSize，限制 mempool 可增长的最大字节数 |
| NCCL_LOCAL_REGISTER | src/register/register.cc:16 | 1 | 为 1 时允许 ncclCommRegister 执行本地 buffer 注册；为 0 或 P2P 使用 memcpy 时跳过注册 |
| NCCL_MLOPART_RDMA_ENABLE | src/register/coll_reg.cc:14 | 0 | 当 communicator 具有 MLOPart 且 buffer 非空时，决定是否将 buffer 标记为 RDMA capable；影响 NVLS + CollNet 跨节点注册路径 |

---

## 插件引用计数

| 环境变量名 | 源文件:行号 | 默认值 | 作用/含义 |
|---|---|---|---|
| NCCL_NET_PLUGIN_REF_COUNT | src/plugin/net.cc:38 | 0 | 外部 net 插件加载时的初始引用计数；非零值可阻止插件在引用计数归零前被卸载 |
| NCCL_GIN_PLUGIN_REF_COUNT | src/plugin/gin.cc:23 | 0 | 外部 GIN 插件加载时的初始引用计数；非零值可阻止插件在引用计数归零前被卸载 |
| NCCL_RMA_PLUGIN_REF_COUNT | src/plugin/rma.cc:23 | 0 | 外部 RMA 插件加载时的初始引用计数；非零值可阻止插件在引用计数归零前被卸载 |

---

## Bootstrap / RAS / 杂项

| 环境变量名 | 源文件:行号 | 默认值 | 作用/含义 |
|---|---|---|---|
| NCCL_OOB_NET_ENABLE | src/bootstrap.cc:103 | 0 | 使用网络插件（而非 socket）进行带外 bootstrap 环网通信 |
| NCCL_UID_STAGGER_RATE | src/bootstrap.cc:669 | 7000 | 控制 bootstrap 连接到 root 时的错峰速率（messages/sec） |
| NCCL_UID_STAGGER_THRESHOLD | src/bootstrap.cc:670 | 256 | 当某个 root 下的 rank 数超过该阈值时才启用 bootstrap 连接错峰 |
| NCCL_RAS_ENABLE | src/bootstrap.cc:672 | 1 | bootstrap 阶段启用 RAS (Remote Access Service / telemetry) 客户端初始化 |
| NCCL_RAS_TIMEOUT_FACTOR | src/ras/ras.cc:80 | 1 | RAS 各类超时时间（keepalive、连接重试、stuck、idle、peer dead、collective leg 等）的统一乘数因子 |
| NCCL_SOCKET_RETRY_CNT | src/misc/socket.cc:19 | 34 | socket 连接/accept 遇到可重试错误时的最大重试次数 |
| NCCL_SOCKET_RETRY_SLEEP_MSEC | src/misc/socket.cc:20 | 100 | socket 重试时的基础休眠毫秒数，实际休眠时间为 errorRetries * 该值 |
| NCCL_SOCKET_POLL_TIMEOUT_MSEC | src/misc/socket.cc:21 | 0 | socket 等待可读/可写时 poll() 调用的超时毫秒数；0 表示不阻塞 |
| NCCL_SOCKET_RCVBUF | src/misc/socket.cc:22 | -1 | 设置 socket 接收缓冲区大小（SO_RCVBUF）；-1 表示保持系统默认 |
| NCCL_SOCKET_SNDBUF | src/misc/socket.cc:23 | -1 | 设置 socket 发送缓冲区大小（SO_SNDBUF）；-1 表示保持系统默认 |

---

## 统计

- 本文档共整理 **约 180+** 个 NCCL 环境变量参数。
- 覆盖范围：`src/` 目录下所有 `NCCL_PARAM(...)` / `DEFINE_NCCL_PARAM(...)` 定义。
- 未覆盖：部分在源码中定义但未被实际引用的参数（如 `NCCL_GRAPH_HELPER_DISABLE`）已标注为“当前源码树中未见实际使用”。
