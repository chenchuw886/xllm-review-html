# FlashComm 代码理解与页面口径

分析基线：`xLLM-AI/xllm` main，FlashComm for DeepSeek V4 对应 PR #2022 / commit `9549ee93`。

## 系统级链路

请求文本 → Tokenizer → Scheduler/Batch → Worker/Executor → DeepSeek V4 Model Forward → Embedding → Decoder Layers → Final Gather → HC Head / Norm → LM Head / Logits → Sampler → Token ID → Incremental Decoder → 流式文字。

FlashComm 只介入长输入的 Prefill Model Forward。它缩短首轮 prompt 计算，因此影响 TTFT；采样后的 Decode 循环不启用 FlashComm，因此不应宣称直接降低 TPOT。

## 模型内数据流

模块：`FlashComm1Context`

- 职责：根据开关、Prefill 阶段、Token 阈值和 CP 拓扑决定是否启用；保存 TP rank、补齐后 token 数、本地 shard 大小、TP group 与 MMRS 选项。
- 入口：`DeepseekV4ModelImpl::forward()` 构造 context，以 thread-local scope 覆盖 decoder layer 执行。
- 数据流：完整 hidden states → `shard_sequence()` → TP-local rows → decoder layers → `gather_sequence()` → 完整 hidden states → final head。
- 保护条件：NPU、Prefill、`enable_flashcomm1=true`、`cp_size=1`、tokens 达阈值、TP group 可用。

DeepSeek V4 单层边界：

1. Attention 前恢复完整序列。
2. DSAttention 内 row-parallel 输出把 AllReduce 改为 ReduceScatter，输出 local shard。
3. HC residual 区段在 local shard 上执行。
4. MoE gate 前恢复完整序列，确保路由语义不变。
5. MoE 输出后按序列重新 local slice，进入下一层。

## 通信与算子

- Baseline：local Matmul partial → TP AllReduce → 每个 TP rank 都持有 `[S, H]`。
- FlashComm core：local Matmul partial → TP ReduceScatter → 每个 rank 持有 `[S/TP, H]`。
- Optional MMRS：把 Matmul + ReduceScatter 融合为 `npu_mm_reduce_scatter_base`；BF16 和 w8a8_dynamic 已接入。
- Fallback：MMRS 的 dtype、shape、bias 或运行条件不满足时，退回普通 Matmul + ReduceScatter，FlashComm 核心语义仍成立。
- Padding：全局 token 数向上补齐到 `TP × 16` 的倍数，保证各 rank collective 等长；最终 gather 后截回真实 token 数。

## 收益闭环

AllReduce 复制完整序列 → 多个 TP rank 重复处理相同 token → 长 Prefill 的通信和可分片计算成本偏高 → ReduceScatter 在归并时直接完成序列分片 → 安全区段每 rank 只保留/处理约 `S/TP` rows → Prompt forward 更早结束 → LM Head 和采样更早得到首 token → 用户更早看到第一个文字。

参考数据：DeepSeek-V4-Flash W8A8C16，A3，EP16 / DP4 / TP4，8K input，并发 32：TTFT 10211.2 ms → 8840.9 ms（约 -13.4%）。其中序列分片约贡献 -11.8%，MMRS 额外约 -1.8%。该数据只在相同测试条件下成立。

## 主要代码证据

- `xllm/core/common/flash_comm1_context.{h,cpp}`：启用门槛、padding、shard/gather/reduce-scatter。
- `xllm/models/llm/deepseek_v4.h`：模型级 context、首尾 shard/gather、final head 边界。
- `xllm/core/layers/deepseek_v4_decoder_layer.cpp`：Attention / MoE 完整序列边界。
- `xllm/core/layers/common/linear.cpp`：row-parallel reduce mode、BF16 / W8A8 dynamic MMRS 与 fallback。
- `xllm/core/runtime/llm_worker_impl.cpp`：model forward → logits → sampler。
- `xllm/core/framework/request/incremental_decoder.cpp`：token ID → 增量文字。

## 页面表达边界

- 使用“减少重复 token work / 中间激活复制”，不表述为所有算力严格下降到 `1/TP`。
- 使用“TTFT 参考结果”，不泛化为所有模型、shape 或 workload 的固定收益。
- 明确 MMRS 是可选增量且默认关闭，FlashComm 核心能力不依赖 MMRS。
- 区分 decoder 末尾的 token-layout gather 与 LM Head 的词表并行 logits 归并。

## 补充：当前仓里哪些模型真的受这个开关影响（2026-11 复核）

FlashComm 的上下文由模型 forward 用 `FlashComm1ContextScope` 建立。全仓建立该 scope 的只有两处：

- `xllm/models/llm/deepseek_v4.h`（DeepSeek-V4，C++ 路径）
- `xllm/models/llm/qwen3_next_hybrid_base.h`（Qwen3-Next）

**Python 路径的 DeepSeek-V4.1 没有建立这个 scope**，因此给 V4.1 传 `--enable_flashcomm1` 不会生效。
页面在「结论与适用条件」一节用黄色 callout 显式写了这一点，并在「风险与待确认」里重申：
如果在 V4.1 上观察到差异，原因不在 FlashComm。

核对方式：`grep -rn "FlashComm1ContextScope" xllm/` 看谁构造了它。

## 补充：页面重做后保留与新增的内容

保留：机制（AllReduce → ReduceScatter）、单层五个边界、padding 对齐到 `TP × 16`
（常量 `kFc1LocalTokenAlignment = 16`，在 `flash_comm1_context.cpp`）、MMRS 的 fallback 语义、
TTFT 参考数据与拆分（序列分片约 −11.8%，MMRS 额外约 −1.8%）。

新增：四个运行条件的完整列表（NPU / prefill / `cp_size=1` / token 达阈值 / TP group 可用）、
`flashcomm1_min_prefill_tokens` 默认 8192 是保守起点而非推荐值、`mmrs_comm_mode` 的取值。

删除：旧页的 hero 指标条里「0 Decode 路径侵入」这类口号式表述，改为在适用条件里直接写
「它不改 decode 路径，不要指望 TPOT 直接受益」。
