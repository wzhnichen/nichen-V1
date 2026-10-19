---
license: apache-2.0
base_model: Qwen/Qwen3-4B
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
frameworks:
  - pytorch
tasks:
  - text-generation
language:
  - en
  - zh
tags:
  - agent
  - function-calling
  - tool-use
  - qwen3
---

# nichen-V1

**English** | [中文](#中文说明)

nichen-V1 is a 4B-parameter tool-use / agent model fine-tuned from [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B).
It was trained with QLoRA distillation over 10 rounds, using labels from the open-weight teacher **DeepSeek-V4.1-Flash-1017**.
The training data is synthetic: agent tasks, function calling with invented tool schemas, and `web.search` / `web.fetch` style browsing.
The LoRA adapter (r=8, alpha=16, on q_proj/v_proj) was merged into the bf16 base weights. This repository holds the full merged model in safetensors, so no PEFT is needed at inference.

## Download

| source | link |
|---|---|
| ModelScope | https://modelscope.cn/models/wzhnichen/nichen-V1 |
| Hugging Face | https://huggingface.co/wzhnichen/nichen-V1 |
| GitHub Release (all model files as assets) | https://github.com/wzhnichen/nichen-V1/releases/tag/v1.0 |

```bash
# ModelScope
modelscope download --model wzhnichen/nichen-V1 --local_dir ./nichen-V1
# Hugging Face
hf download wzhnichen/nichen-V1 --local-dir ./nichen-V1
# GitHub Release
gh release download v1.0 -R wzhnichen/nichen-V1 -D ./nichen-V1
```

## Training summary

| item | value |
|---|---|
| base model | Qwen/Qwen3-4B (Apache-2.0) |
| method | QLoRA (NF4 base, LoRA r=8, alpha=16, q_proj/v_proj), 10 distillation rounds |
| teacher | DeepSeek-V4.1-Flash-1017 (open weights), non-thinking mode |
| data | synthetic agent / tool-use / web.search + web.fetch dialogues with invented tool schemas; teacher outputs were accepted or rejected by validators and never hand-edited |
| export | adapter merged into the bf16 base (`merge_and_unload`); tokenizer, chat template and generation config copied unchanged from Qwen3-4B |

## Evaluation: BFCL v4, single-turn (self-run)

| metric | Qwen3-4B (base) | nichen-V1 |
|---|---|---|
| non-live AST | 82.60 | **86.15** |
| live AST (weighted) | 72.39 | **73.28** |
| live_multiple | **73.22** | 71.98 |
| irrelevance (total) | 83.48 | **84.68** |
| relevance | 81.25 | 81.25 |
| live_parallel_multiple | 58.33 | **62.50** |

Conditions: we ran these ourselves with the BFCL v4 harness, single-turn only (13 categories, 3,641 items), `qwen_fc` style function-calling prompts, and thinking disabled. Both models were loaded in **NF4 4-bit**, and nichen-V1 was evaluated as NF4 base + LoRA adapter.
These numbers are **not** official leaderboard results and cannot be compared directly with them. For reference, the public BFCL leaderboard lists Qwen3-4B (FC, full precision, thinking enabled) at 88.58 / 81.42 (non-live AST / live AST) under different conditions.
The released weights are the bf16 merge, so their scores may differ slightly from the NF4 + adapter numbers above.

## Usage (transformers, tool calling)

```python
import json, re
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "wzhnichen/nichen-V1"   # ModelScope: from modelscope import snapshot_download; model_id = snapshot_download("wzhnichen/nichen-V1")
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="bfloat16", device_map="auto")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]
messages = [{"role": "user", "content": "What's the weather in Hangzhou in celsius?"}]

text = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                               tokenize=False, enable_thinking=False)
inputs = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

calls = [json.loads(c) for c in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", reply, re.S)]
print(calls or reply)
# -> [{'name': 'get_weather', 'arguments': {'city': 'Hangzhou', 'unit': 'celsius'}}]
# To continue, append the assistant turn (with tool_calls) and a {"role": "tool", "content": ...} message, then generate again.
```

The model uses the standard Qwen3 chat template (`<tools>` in the system prompt, `<tool_call>{...}</tool_call>` in replies), so OpenAI-compatible servers such as vLLM and SGLang can serve it with their Qwen/Hermes tool parsers. It was trained and evaluated with `enable_thinking=False`, which we recommend.

## Limitations

- Small model (4B): it can pick the wrong tool, hallucinate argument values, or miss calls in long multi-step tasks. Validate tool calls before you execute them, especially side-effecting ones.
- Trained and evaluated in non-thinking mode only. Thinking mode was not tuned and may behave differently from the base model.
- Evaluation covers BFCL single-turn categories only; multi-turn, memory and agentic BFCL categories were not measured. `live_multiple` is slightly below the base model.
- The training data is synthetic and English-centric. Quality on other languages and on real-world APIs with unusual schemas may be lower.
- Inherits the knowledge cutoff, biases and safety properties of Qwen3-4B. No additional safety tuning was done.
- Scores were measured on NF4-quantized weights; the bf16 release may differ slightly.

## License

Apache-2.0, the same as the base model Qwen3-4B. The teacher model DeepSeek-V4.1-Flash-1017 was used only to produce training labels.

---

<a id="中文说明"></a>

# nichen-V1（中文说明）

nichen-V1 是基于 [Qwen/Qwen3-4B](https://modelscope.cn/models/Qwen/Qwen3-4B) 微调的 4B 参数工具调用 / Agent 模型。
训练方式是 QLoRA 蒸馏，共 10 轮，标注来自开源权重教师模型 **DeepSeek-V4.1-Flash-1017**。
训练数据全部为合成数据，涵盖 Agent 任务、使用虚构工具 schema 的函数调用，以及 `web.search` / `web.fetch` 式的网页检索。
LoRA 适配器（r=8，alpha=16，作用于 q_proj/v_proj）已合并进 bf16 基座权重。本仓库提供完整的 safetensors 模型，推理时不需要 PEFT。

## 下载

ModelScope：https://modelscope.cn/models/wzhnichen/nichen-V1 ｜ Hugging Face：https://huggingface.co/wzhnichen/nichen-V1 ｜ GitHub Release（全部模型文件作为附件）：https://github.com/wzhnichen/nichen-V1/releases/tag/v1.0 。下载命令见上文英文部分的 Download。

## 训练概要

| 项目 | 内容 |
|---|---|
| 基座模型 | Qwen/Qwen3-4B（Apache-2.0） |
| 方法 | QLoRA（NF4 基座，LoRA r=8，alpha=16，q_proj/v_proj），10 轮蒸馏 |
| 教师模型 | DeepSeek-V4.1-Flash-1017（开源权重），非思考模式 |
| 数据 | 合成的 Agent / 工具调用 / web.search + web.fetch 对话，工具 schema 均为虚构；教师输出只经校验器接受或拒绝，从未人工修改 |
| 导出 | 适配器合并进 bf16 基座（`merge_and_unload`）；分词器、对话模板和生成配置原样沿用 Qwen3-4B |

## 评测：BFCL v4 单轮（自测）

| 指标 | Qwen3-4B（基座） | nichen-V1 |
|---|---|---|
| non-live AST | 82.60 | **86.15** |
| live AST（加权） | 72.39 | **73.28** |
| live_multiple | **73.22** | 71.98 |
| irrelevance（总计） | 83.48 | **84.68** |
| relevance | 81.25 | 81.25 |
| live_parallel_multiple | 58.33 | **62.50** |

评测条件：我们自己使用 BFCL v4 评测框架运行，只测单轮（13 个类别，共 3,641 条），采用 `qwen_fc` 风格的函数调用提示，关闭思考模式。两个模型都以 **NF4 4-bit** 加载，nichen-V1 的评测形式是 NF4 基座 + LoRA 适配器。
以上**并非**官方排行榜成绩，不能与之直接比较。作为参考，公开 BFCL 排行榜上 Qwen3-4B（FC、全精度、开启思考）为 88.58 / 81.42（non-live AST / live AST），评测条件与我们不同。
本仓库发布的是 bf16 合并权重，实际分数可能与上表的 NF4 + 适配器结果略有差异。

## 使用方法

用法与上文英文部分的 transformers 示例相同。从 ModelScope 下载：

```python
from modelscope import snapshot_download
model_dir = snapshot_download("wzhnichen/nichen-V1")
```

然后按英文示例调用 `tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)`，再从输出中解析 `<tool_call>{...}</tool_call>`。
模型沿用标准 Qwen3 对话模板，可以用 vLLM、SGLang 等 OpenAI 兼容服务部署，配合 Qwen/Hermes 工具解析器使用。训练和评测都在 `enable_thinking=False` 下进行，推荐保持关闭思考模式。

## 局限性

- 模型较小（4B），可能选错工具、编造参数值，或在长的多步任务中漏掉调用。执行工具调用前请先校验，有副作用的操作尤其要注意。
- 只在非思考模式下训练和评测。思考模式未经调优，表现可能与基座不同。
- 评测只覆盖 BFCL 单轮类别，未测多轮、记忆和 Agent 类别。`live_multiple` 略低于基座。
- 训练数据为合成数据，并以英文为主。在其他语言或 schema 特殊的真实 API 上，效果可能更差。
- 继承 Qwen3-4B 的知识截止时间、偏差和安全特性，没有做额外的安全对齐。
- 分数是在 NF4 量化权重上测得的，bf16 版本可能略有差异。

## 许可证

Apache-2.0，与基座模型 Qwen3-4B 相同。教师模型 DeepSeek-V4.1-Flash-1017 只用于生成训练标注。
