# Engram 主机侧卸载 代码理解与页面口径

来源：xLLM 仓内 `docs/src/content/docs/zh/features/engram_host_offload.md`，以及
`xllm/python/models/engram_config.py`、`deepseek_v41_engram.py`、`engram_shared_memory.py`。

## 它换的是什么

V4.1 的 Engram n-gram 表在 W8A8（Eco-Tech int8）导出下每卡约 **27.7 GiB** 常驻显存。
`cpu_offload` 把它搬到 host 内存（**可分页**，不是锁页），换 HBM 预算。

代价：每步一次 NPU→CPU 的 id 同步 + CPU int8 去量化 + 命中行 H2D 回传。

前提：查表 id 只取决于输入 token，进层前就能算好 —— 所以表的访问可以和前面的计算重叠。

## 每步的路径

设备算哈希 id → D2H 到锁页 host 缓冲 → `xllm_ops.engram_int8_lookup_cpu` 融合算子
（`index_select` + `repeat_interleave` 分组 scale + bf16 转换合成一次 CPU 调用，直接写进锁页 staging）
→ H2D 到设备 staging 块。

代码注释里的两条硬约束：

- 这是**图外的 host staging pass**（`_local_block_offload`），不是图安全的 host 工作，
  **绝不能移进 ACL 图捕获**。
- 图可见的设备 staging 块地址在捕获时被烘焙进图，之后不能移动或增长；按最大 bucket 预先分配。

融合算子是卸载路径**唯一**的去量化实现。没注册直接 `RuntimeError` 提示重新编译，
不静默退回逐算子的 eager host 路径。锁页分配失败同样报真实原因，不悄悄降级成未锁页 staging。

## DP 共享

`dp_shared_memory`：节点内映射同一 TP 分片的 DP 副本共用**一份物理 host 表**（每 rank 约省 27.7 GB host 内存）。

**共享的是只读的 TP 分片表数据**（host 上的 int8 codes 与 fp32 scales），**不是查表结果**。
文档给的理由：各 rank 的 hash id 本就不同，要共享结果得先汇总 id 再广播回来，多一轮通信与同步，
所以让每个 rank 直接查本地那份共享表。**lookup 仍每 rank 各执行一次。**

机制：vllm-ascend 用 CANN 注册的 UVA（`SharedUvaBuffer`）；xLLM 在 CPU 上读表，去掉了设备指针 / host 注册层，
就是 `multiprocessing.shared_memory` 段 + `torch.frombuffer` CPU 视图。
每个「(engram 层, TP 分片)」两个段：`codes` `[rows, dim]` int8、`scales` `[rows, dim // quant_block]` float32。

建段顺序（对齐 vllm-ascend 的时序）：leader 随机命名创建 → gloo 广播描述符 →
非 leader attach 并校验 size/shape/dtype/token → `engram_all_gather_object` 收错 + `engram_barrier` 栅栏 →
leader unlink（自己和所有 attacher 的映射仍有效）→ ATTACHED → leader 加载 → LOADED。

所有集合通信走 gloo `engram_dp` 组（HCCL 没有 object 集合通信）。

### 节点身份这条要写进页面

`hostname` **只用于预检分组**（如 `/dev/shm` 容量估算），**它本身不是共享成功的证明**：

- 不同物理节点必须可区分；
- 同一物理机但处于不同 `/dev/shm` namespace（各容器挂独立 tmpfs）也不能共享。

真正的证明是 attach 成功**且**从段尾读回的 token 与 leader 写入的一致（`_verify_segment_token`）。
`/dev/shm` 容量检查同样只是预检。**不要求**每个 rank 的 hostname 唯一。

## 配置

原始 JSON 字符串经 C++ gflag `engram_config` 透传到 Python 解析，命令行写 `--engram-config`。

xLLM 自身**默认全关**：`EngramConfig()` = `{"cpu_offload":false,"dp_shared_memory":false}`（等价 `{}`），走 HBM 路径。
要开启必须显式传。部署脚本若要默认开启，由脚本自行下发，xLLM 不感知。

解析**严格**：字段必须是真布尔值，`"true"` / `0` / `1` 一律拒绝；未知字段、重复 JSON key、非法 JSON 都报错。
设计意图是「一个拼写错误不能静默关掉开关」。`dp_shared_memory=true` 而 `cpu_offload=false` 在解析期 ValueError。

## 约束

- **TP ∈ {1,2,4,8}**，且 hash 列数被 TP 整除。hash 列数 = `(max_ngram_size - 1) * n_heads`；
  V4.1 默认 `n_heads=8`、`max_ngram_size=4` ⇒ **24 列**，1/2/4/8 均整除。
- **PP = 1、DCP = 1**。
- 权重格式 safetensors / dummy。
- **int8 专属**：只有 W8A8 表有 CPU 解码路径。fp8/bf16 表请求 `cpu_offload` **回退 HBM**，请求 `dp_shared_memory` **直接报错**。
- `dp_shared_memory` 需要节点内 gloo 组与 `/dev/shm` 空间（按聚合 footprint 预检）。
- **PD 分离 TODO**：接入后该拓扑将要求 `dp_shared_memory=true`，当前运行期未强制。

## 页面表达边界

- 明确说这是**拿时延换显存**，不是纯收益。给出判断方法：先算多出的 HBM 能让 KV 容量 / 并发涨多少，
  再在目标负载上 A/B 测时延代价；如果显存没转化成更大 batch 或更长上下文，就不该开。
- 有明确口径的两个数字照写：HBM 释放约 27.7 GiB/卡、host 内存节省约 27.7 GB/rank。
- **时延代价和端到端吞吐影响标为未量化 / 未验证**，不给数字。
- 强调 `dp_shared_memory` 共享的是表不是结果，lookup 仍每 rank 一次 —— 这是最容易被误读的一点。
