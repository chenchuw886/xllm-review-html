# xLLM 特性解读

面向开发者的 xLLM 推理特性可视化解读，重点在昇腾 NPU 上的 DeepSeek-V4.1-Flash。

站点：<https://chenchuw886.github.io/xllm-review-html/>

## 页面

| 页面 | 内容 |
|---|---|
| [`arch/`](./arch/) | **DeepSeek-V4.1-Flash 模型结构**。40 层 CED、CSA2 跨层共享 KV、分层稀疏索引、Single-Pass mHC、Engram、多模态、KV 缓存归属与量化分配、V4→V4.1 结构差异。全文分开写「论文口径」与「实现口径」。其余页面的术语基础。 |
| [`flashcomm/`](./flashcomm/) | **FlashComm1**。长 prefill 把 row-parallel 层后的 `AllReduce` 换成 `ReduceScatter`，每个 TP rank 只处理约 `S/TP` 行 token。仓里有 C++ 与 Python 两条独立实现线，门禁与收益口径不同。 |
| [`dspark/`](./dspark/) | **DSpark**。块并行草稿 + Markov 条件头 + 置信度头；一次 decode step 的完整链路、V4→V4.1 差异，以及昇腾 A3 上按时间顺序的三项优化（draft 成块 → target MoE MC2 → draft 入图）与下一步 prelaunch。 |
| [`dsa-cp/`](./dsa-cp/) | **DSA CP**。V4.1 稀疏注意力在 TP 组内按 query token 行切分，KV 整份复制，出口一次 all-to-all 换轴。 |
| [`swa-replay/`](./swa-replay/) | **SWA Bounded Replay**。CED 下 decoder 层 prefill 只回放末尾 140 行；对照 DeepInfer 的无损实现说明当前这版为什么有损。 |
| [`engram-offload/`](./engram-offload/) | **Engram 主机侧卸载**。int8 n-gram 表搬到 host 内存，融合 CPU 算子查表去量化，换 HBM 预算。 |

## 写作口径

特性页（flashcomm / dsa-cp / swa-replay / engram-offload）固定五段：

1. 一句话结论与适用条件（含「什么时候别开」）
2. 机制：输入 → 处理 → 输出，配图
3. 代码证据：文件与符号
4. 启用方式与约束
5. 收益与验证强度，外加风险与待确认

`arch/` 和 `dspark/` 是知识分享型长文，不套这个骨架，但同样遵守下面的状态标签与数字纪律。
它们的内容以 `000_DSV4.1/模型结构&知识分享/` 下的内部讲稿为准。

状态标签的含义：

- **已实现** — 代码在仓里
- **已实测** — 我们跑过并留了数据，页面上会写明硬件、并行配置、输入输出长度与并发
- **未验证** — 没跑过，页面上不写数字
- **有损** — 会改变数值结果

性能数字一律绑定测试条件，换条件不成立。没在当前配置下实测过的收益不写数字。
来自不同测试条件的百分比不相加、不相乘；压测值与 profile 值不互相换算。
代码引用只写文件名与字段 / 函数名，不写行号 —— 行号随版本漂移。

## 技术实现

零依赖静态页面，GitHub Pages 直接发布。

- `assets/site.css` — 共享设计系统（浅色默认，深色跟随系统或手动切换）
- `assets/site.js` — 只做两件事：记住配色选择、高亮当前小节的目录项。禁用 JS 页面照常可读。
- 图表全部是内联 SVG，用 CSS 变量取色，自动适配深浅色。

本地预览：

```bash
python3 -m http.server 8731 --bind 127.0.0.1
```

## 分析口径记录

页面结论的推导过程和取舍记录在 [`ANALYSIS/`](./ANALYSIS/)，按特性分文件。

---

个人整理，非官方文档。结论以 [xLLM-AI/xllm](https://github.com/xLLM-AI/xllm) 仓库代码为准。
