# xLLM DSpark feature page notes

## Scope

The page separates three layers that must not be conflated:

1. Speculative decoding as a lossless draft-and-verify algorithm.
2. The full DSpark design from the paper: semi-autoregressive drafting plus
   confidence-scheduled verification.
3. The current xLLM DeepSeek-V4-Flash-0731 implementation: a fixed five-token
   proposal using the parallel backbone and low-rank Markov head, followed by
   target validation.

The xLLM implementation does not currently load the checkpoint's confidence
head or implement the hardware-aware prefix scheduler. The page therefore does
not attribute the paper's online scheduling gains to the current benchmark.

## Primary references

- DSpark paper: <https://arxiv.org/abs/2607.05147>
- DFlash paper: <https://arxiv.org/abs/2602.06036>
- DeepSpec training repository: <https://github.com/deepseek-ai/DeepSpec>
- xLLM adaptation PR: <https://github.com/xLLM-AI/xllm/pull/2172>
- DeepSeek-V4-Flash-0731 W8A8 weights:
  <https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4-Flash-0731-w8a8>

## Architecture facts used on the page

- Speculative decoding accepts the longest draft prefix consistent with the
  target distribution and appends a target-generated bonus/correction token.
- DSpark uses a parallel backbone to produce all block positions in one model
  forward, then adds prefix dependence with a lightweight sequential module.
- The Markov variant uses a rank-256 factorization of a token-to-token
  transition matrix.
- The DeepSeek-V4 deployment described by the paper uses three MoE draft
  layers, sliding-window attention of 128, maximum block size five, and a
  Markov head.
- Training freezes the target, vocabulary embedding, and LM head, while
  optimizing the draft backbone, sequential module, and confidence head.
- The training objective combines token cross entropy, target-distribution
  total variation, and confidence prediction, with greater weight on earlier
  block positions.

## Benchmark source and arithmetic

Source: local `results.md` and the acceptance counters collected during the
DeepSeek-V4-Flash-0731 xLLM adaptation.

Test setup for the main performance chart:

- Atlas A3, one host, 16 devices
- DP=4
- DeepSeek-V4-Flash-0731 W8A8 for DSpark
- approximately 8,192 input tokens and 1,024 output tokens
- concurrency 32

| Mode | Average TPOT | Output throughput |
|---|---:|---:|
| Baseline | 70.1 ms | 422.9513 tok/s |
| MTP, gamma=1 | 45.1 ms | 613.7594 tok/s |
| DSpark, gamma=5 | 28.0 ms | 845.7383 tok/s |

Derived values:

- TPOT reduction versus baseline:
  `(70.1 - 28.0) / 70.1 = 60.1%`
- TPOT reduction versus MTP:
  `(45.1 - 28.0) / 45.1 = 37.9%`
- Output-throughput ratio versus baseline:
  `845.7383 / 422.9513 = 2.00x`

The page deliberately calls this a reference result. DSpark increases TTFT in
this test, so it is presented as a Decode/long-output optimization rather than
an unconditional end-to-end win.

## Acceptance metrics

Acceptance data comes from four requests with approximately 30K input tokens
and 1K output tokens each. There were 951 proposals at gamma five.

| Position | Prefix reaches position | Prefix survival | Conditional acceptance |
|---:|---:|---:|---:|
| 0 | 798 / 951 | 83.9% | 83.9% |
| 1 | 708 / 951 | 74.4% | 708 / 798 = 88.7% |
| 2 | 625 / 951 | 65.7% | 625 / 708 = 88.3% |
| 3 | 569 / 951 | 59.8% | 569 / 625 = 91.0% |
| 4 | 520 / 951 | 54.7% | 520 / 569 = 91.4% |

Additional derived metrics:

- total draft-token acceptance: `3220 / 4755 = 67.7%`
- mean accepted draft length: `3220 / 951 = 3.39`
- full five-token block survival: `520 / 951 = 54.7%`
- mean committed length including the target bonus/correction: approximately
  `4.39` tokens per decode step

The default chart uses prefix survival because this is what the xLLM
per-position counters report. The interactive conditional view divides each
position by the number of prefixes that reached it and is labeled separately.

## Current implementation limits called out on the page

- fixed five-token validation; no confidence scheduler yet
- `cp_size > 1` not supported
- end-to-end validation currently centered on Atlas A3 single-host DP4 W8A8
- native DSpark sparse-attention mode requires explicit operator support;
  compatibility mode is the default
- extra draft weights, KV cache, graph state, and workspace increase memory
- reduced-vocabulary DSpark checkpoints are not supported
- short-output break-even and broader deployment combinations still require
  workload-specific validation
