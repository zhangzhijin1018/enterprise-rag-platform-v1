# 本地生成式模型微调实施方案

## 1. 结论

如果当前项目下一步要补的是**本地高敏生成能力**，最稳的第一条路线不是全量微调，而是：

**先对 `Qwen3-14B-Instruct` 这类本地 instruct 模型做第一轮 `QLoRA + SFT` 微调。**

原因是：

1. 当前项目已经有比较完整的检索、精排、治理和拒答骨架；
2. 本地生成模型要承担的不是通用闲聊，而是 grounded answer、拒答和冲突说明；
3. `QLoRA` 成本更低、回滚更容易，也更适合当前项目第一代本地生成主模型。

## 2. 当前项目里什么时候才值得微调本地生成模型

建议先满足这几个前提：

1. `Milvus + BGEM3 + reranker` 检索链路已经相对稳定；
2. `/eval` 和 badcase 已经能区分“召回问题”和“生成问题”；
3. 你要解决的主问题已经变成：
   - 回答风格不够企业化
   - 拒答不稳定
   - 冲突解释不稳定
   - 高敏场景需要本地闭环生成

如果现在最主要的问题还是：

- 根本没召回到
- 正确 chunk 没排上来

那优先级仍然应该放在：

1. `bge-reranker-large` 微调
2. `bge-m3` 领域微调

而不是直接上生成模型训练。

## 3. 推荐基础模型

当前最稳的起点是：

- `Qwen/Qwen3-14B-Instruct`

原因：

1. 中文能力和指令跟随能力更适合企业 RAG 场景
2. 资源门槛明显低于 `32B`
3. 更适合作为 `restricted / local_only` 场景的第一代本地生成模型

如果预算更紧，可以先用：

- `Qwen/Qwen3-8B`

如果质量优先、资源更充足，后续可以升级：

- `Qwen/Qwen3-32B`

## 4. 训练目标不要搞错

这轮训练的目标不是“把模型训得更会聊天”，而是让模型更稳地完成这些企业 RAG 特定任务：

1. 基于证据回答
2. 证据不足时明确拒答
3. 冲突场景下输出保守说明
4. 输出更适合企业问答页面的结构化文字

所以训练数据不应该主要是“闲聊对话”，而应该优先来自：

1. grounded answer 样本
2. refusal 样本
3. conflict explanation 样本
4. citation/结构化回答样本

## 5. 数据格式

最推荐的数据格式是 `messages`：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是新疆能源集团企业知识智能副驾。必须基于已检索证据回答；证据不足时明确拒答；不要编造制度、流程和数值。"
    },
    {
      "role": "user",
      "content": "已检索证据：\n[文档1] 《采购报销管理办法（2025版）》第4.2条：单笔报销金额超过5000元时，需经部门负责人、财务负责人和分管领导依次审批。\n\n用户问题：采购报销超过5000元需要哪些审批？"
    },
    {
      "role": "assistant",
      "content": "根据已检索证据，单笔报销金额超过5000元时，需要依次经过部门负责人、财务负责人和分管领导审批。"
    }
  ]
}
```

当前仓库也提供了一份最小样例：

- `/Users/zhangzhijin/study/黑马学习/rag/RAG- project/enterprise-rag-platform/train/examples/llm_sft.sample.jsonl`

## 6. 为什么推荐 QLoRA，而不是一开始就全量微调

`QLoRA` 的核心思路是：

1. 基础模型以低比特量化方式加载
2. 不直接更新整套大模型权重
3. 只训练少量 LoRA 适配器参数

这样做的收益是：

1. 显著降低显存要求
2. 训练和回滚都更轻
3. 更适合当前项目这种“先验证收益，再决定是否做更重训练”的阶段

## 7. 资源建议

当前按 `Qwen3-14B-Instruct` 做第一轮 `QLoRA/SFT`，建议按 Linux + CUDA 服务器来准备资源。

| 场景 | CPU | RAM | GPU | 存储 |
| --- | --- | --- | --- | --- |
| 最低可跑验证 | 16 核 | 64GB | 1 x 48GB | 300GB+ NVMe |
| 推荐 | 24 核 | 128GB | 1 x 80GB 或 2 x 48GB | 500GB+ NVMe |

说明：

1. 当前脚本默认 `4bit + batch_size=1 + gradient_accumulation=8`
2. 24GB 显卡更适合跑推理 PoC，不建议直接承担 14B QLoRA 训练
3. macOS 开发机不建议直接训练，适合做：
   - 样本准备
   - 脚本整理
   - 命令校验

## 8. 训练脚本

当前仓库已提供训练脚本：

- `/Users/zhangzhijin/study/黑马学习/rag/RAG- project/enterprise-rag-platform/train/train_local_llm_qlora.py`

脚本特点：

1. 优先支持标准 `messages` 格式
2. 也兼容简化结构：
   - `system`
   - `question`
   - `context`
   - `answer`
3. 优先走 tokenizer 自带 `apply_chat_template`
4. 回退时使用脚本内置简化模板

## 9. 训练依赖

这条训练链不建议直接混在当前线上服务基础依赖里，而建议在训练机单独安装：

```bash
python -m pip install \
  "transformers>=4.51.0" \
  "datasets>=2.20.0" \
  "peft>=0.14.0" \
  "trl>=0.16.0" \
  "bitsandbytes>=0.45.0" \
  "accelerate>=1.4.0"
```

## 10. 最小试跑命令

先用样例数据 smoke test：

```bash
cd /Users/zhangzhijin/study/黑马学习/rag/RAG-\ project/enterprise-rag-platform

python train/train_local_llm_qlora.py \
  --model-name /path/to/Qwen3-14B-Instruct \
  --train-path ./train/examples/llm_sft.sample.jsonl \
  --output-dir ./artifacts/qwen3-14b-qlora-demo \
  --max-seq-length 2048 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --num-train-epochs 1
```

## 11. 第一轮真实训练命令模板

```bash
cd /Users/zhangzhijin/study/黑马学习/rag/RAG-\ project/enterprise-rag-platform

python train/train_local_llm_qlora.py \
  --model-name /data/models/Qwen3-14B-Instruct \
  --train-path ./data/train/llm_sft_train.jsonl \
  --output-dir ./artifacts/qwen3-14b-qlora-v1 \
  --max-seq-length 4096 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --num-train-epochs 2 \
  --logging-steps 10 \
  --save-steps 200 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05
```

## 12. 上线方式

当前更推荐的上线方式不是把 LoRA 直接塞进应用进程，而是：

1. 用本地 OpenAI-compatible 推理服务加载基础模型 + LoRA adapter
2. 让当前项目通过统一的模型路由调用本地服务

也就是说：

```text
当前项目应用
  -> 本地 OpenAI-compatible 服务（vLLM / SGLang）
  -> base model + LoRA adapter
```

这样做的好处是：

1. 与当前项目现有云模型调用方式更一致
2. 切换基础模型和 adapter 更方便
3. 更适合后续做 `local_only / restricted` 模型路由

## 13. 这轮最值得关注的指标

不要只看 loss。更重要的是：

1. grounded answer 准确率
2. refusal 合理率
3. conflict 解释稳定性
4. 引用占位/结构化输出稳定性
5. 高敏场景本地闭环率

建议训练后至少做两类验证：

1. `/eval` 回放
2. 高敏场景人工验收

## 14. 当前方案的边界

这条路线的定位是：

- 第一轮本地生成模型对齐方案

不是：

- 最终形态的全量训练平台

当前边界包括：

1. 还没自动从 badcase/trace 抽取生成模型训练样本
2. 还没内建 refusal/conflict 专项 evaluator
3. 还没把 LoRA adapter 自动接回当前运行时

所以当前最稳的执行顺序是：

1. 先人工或半自动整理一批 grounded/refusal/conflict 样本
2. 先训一版 `Qwen3-14B-Instruct` QLoRA
3. 再做人审和 `/eval` 回放
4. 最后再考虑把它正式接入 `local_only` 路由

## 15. 为什么这条路线比“先上更大的 32B”更稳

因为当前项目第一代本地生成主模型最关键的是：

1. 先把链路跑通
2. 先把高敏本地生成能力接起来
3. 先验证 grounded answer / refusal / conflict 这三类能力

而不是一开始就追求：

- 更大的参数
- 更长的思维链
- 更高的硬件成本

所以当前最稳的顺序仍然是：

1. `Qwen3-14B-Instruct + QLoRA`
2. 跑通本地闭环
3. 再根据 badcase 和吞吐决定是否升级 `32B`

## 16. 如何从当前项目的 `/eval` 报告自动整理 SFT 训练集

如果你现在还没有人工精标的大规模 `messages` 数据集，最稳的第一步不是手工从零写几千条对话，而是先复用当前项目已经有的：

1. 原始评测集 `enterprise_eval.jsonl`
2. `/eval` 产出的 JSON 报告

当前仓库已经新增了自动整理脚本：

- `/Users/zhangzhijin/study/黑马学习/rag/RAG- project/enterprise-rag-platform/train/build_local_llm_dataset.py`

### 16.1 这份脚本怎么构造一条 SFT 样本

这份脚本使用的是一条务实的规则：

1. `question`
   - 直接使用评测题目的 `question`

2. `contexts`
   - 优先使用 `/eval` 报告里的 `contexts`
   - 如果报告里没有，再退回原始评测集里的 `contexts`

3. `assistant answer`
   - 优先使用报告里的 `answer`
   - 如果报告里没有，再退回 `ground_truth`
   - 如果是拒答题且没有标准答案，再按 `refusal_reason` 生成保守模板

4. `conflict_summary`
   - 如果当前题命中了冲突，脚本会把治理提示写进 user message，帮助模型学习“冲突时应该保守输出”

### 16.2 输出格式长什么样

输出仍然是当前训练脚本直接能吃的 `messages` JSONL：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是新疆能源集团企业知识智能副驾。必须基于已检索证据回答；证据不足时明确拒答；不要编造制度、流程、金额和时间。"
    },
    {
      "role": "user",
      "content": "已检索证据：\n[文档1] 《采购报销管理办法》规定：单笔报销超过 5000 元需部门负责人和财务负责人审批。\n\n用户问题：采购报销超过 5000 元需要哪些审批？"
    },
    {
      "role": "assistant",
      "content": "系统应基于制度文件回答审批节点，并给出来源章节。"
    }
  ],
  "question": "采购报销超过 5000 元需要哪些审批？",
  "scenario": "policy_qa",
  "tags": ["policy", "single_doc", "citation"],
  "source": "eval_report",
  "refusal": false,
  "refusal_reason": "",
  "conflict_detected": false,
  "conflict_summary": "",
  "data_classification": "",
  "model_route": "",
  "answer_mode": ""
}
```

### 16.3 先生成第一版训练集

假设你已经有一个 `/eval` JSON 报告，比如：

- `./data/eval_reports/enterprise_eval_latest.json`

那么可以直接执行：

```bash
cd /Users/zhangzhijin/study/黑马学习/rag/RAG-\ project/enterprise-rag-platform

conda run -n tmf_project python train/build_local_llm_dataset.py \
  --eval-report-path ./data/eval_reports/enterprise_eval_latest.json \
  --dataset-path ./core/evaluation/datasets/enterprise_eval.jsonl \
  --output-path ./data/train/llm_sft.auto.jsonl
```

如果你想先抽一小批预览：

```bash
cd /Users/zhangzhijin/study/黑马学习/rag/RAG-\ project/enterprise-rag-platform

conda run -n tmf_project python train/build_local_llm_dataset.py \
  --eval-report-path ./data/eval_reports/enterprise_eval_latest.json \
  --dataset-path ./core/evaluation/datasets/enterprise_eval.jsonl \
  --output-path ./data/train/llm_sft.preview.jsonl \
  --max-rows 30
```

### 16.4 再直接接上 QLoRA 训练脚本

生成后的训练集可以直接喂给当前脚本：

```bash
cd /Users/zhangzhijin/study/黑马学习/rag/RAG-\ project/enterprise-rag-platform

python train/train_local_llm_qlora.py \
  --model-name /data/models/Qwen3-14B-Instruct \
  --train-path ./data/train/llm_sft.auto.jsonl \
  --output-dir ./artifacts/qwen3-14b-qlora-v1 \
  --max-seq-length 4096 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --num-train-epochs 2
```

### 16.5 这条自动整理链路的边界

这份脚本是“第一轮可落地方案”，但不是最终数据工程。

当前边界有：

1. 如果 `/eval` 成功报告里没有 `answer`，脚本会回退到 `ground_truth`
2. 这意味着第一轮样本里，部分 assistant 目标更接近“参考答案”，不一定等于线上真实回答
3. 当前主要覆盖：
   - grounded answer
   - refusal
   - conflict
4. 还没有自动把 `app.log / audit.log` 里的请求级信息深度并入训练样本

所以最推荐的做法是：

1. 先用这版脚本生成第一批 SFT 数据
2. 训练出第一版 QLoRA adapter
3. 再从 badcase 里补人工高价值样本
4. 第二轮再逐步引入更细的拒答、冲突和企业表达风格数据

## 17. 先不要盲目开训：先做一次 SFT 数据质量检查

自动整理脚本能帮你快速构造第一版 `messages` 数据，但这不代表拿到文件就应该直接开训。  
更稳的做法是先检查：

1. `messages` 结构是否完整
2. `system / user / assistant` 是否齐
3. grounded / refusal / conflict 的样本比例是否失衡
4. user 和 assistant 文本长度是否异常
5. 是否有明显重复样本

当前仓库已经新增质量检查脚本：

- `/Users/zhangzhijin/study/黑马学习/rag/RAG- project/enterprise-rag-platform/train/check_local_llm_dataset.py`

### 17.1 最小使用方式

```bash
cd /Users/zhangzhijin/study/黑马学习/rag/RAG-\ project/enterprise-rag-platform

conda run -n tmf_project python train/check_local_llm_dataset.py \
  --dataset-path ./data/train/llm_sft.auto.jsonl
```

如果你希望把完整检查结果也落盘：

```bash
cd /Users/zhangzhijin/study/黑马学习/rag/RAG-\ project/enterprise-rag-platform

conda run -n tmf_project python train/check_local_llm_dataset.py \
  --dataset-path ./data/train/llm_sft.auto.jsonl \
  --report-path ./data/train/llm_sft.auto.report.json
```

### 17.2 你应该重点看哪些指标

最值得优先看的有：

1. `invalid_rows`
   - 说明是否有缺失 `messages` 或缺失 `assistant` 的样本
2. `grounded_ratio / refusal_ratio / conflict_ratio`
   - 用来判断样本分布是否极度偏科
3. `duplicate_user_assistant_rows`
   - 如果 user 和 assistant 一样，说明这条样本明显有问题
4. `duplicate_sample_rows`
   - 样本重复过多会让第一轮训练收益失真
5. `avg_user_chars / avg_assistant_chars`
   - 长度异常通常意味着模板拼接或上下文灌入有问题

### 17.3 什么样的数据可以先开第一轮训练

一个比较务实的标准是：

1. `invalid_rows` 尽量为 0
2. `duplicate_user_assistant_rows` 为 0
3. grounded / refusal / conflict 三类样本至少不要完全失衡
4. 平均 user 长度不要大得离谱
5. 最好保留 `scenario / tags / source`

### 17.4 如果质量一般，先怎么修

建议按下面顺序修：

1. 先删掉无效样本
2. 再删掉 user/assistant 明显重复的样本
3. 再控制样本分布，避免只有 grounded 没有 refusal/conflict
4. 最后再补人工高价值样本

## 18. 一条最稳的本地生成模型训练顺序

建议完整顺序是：

1. 跑 `/eval`
2. 用 `build_local_llm_dataset.py` 生成第一版 SFT 数据
3. 用 `check_local_llm_dataset.py` 做质量检查
4. 再跑 `train_local_llm_qlora.py`
5. 用本地 OpenAI-compatible 服务加载 base model + adapter
6. 再跑 `/eval` 和高敏 badcase 回放

也就是：

```bash
cd /Users/zhangzhijin/study/黑马学习/rag/RAG-\ project/enterprise-rag-platform

conda run -n tmf_project python train/build_local_llm_dataset.py \
  --eval-report-path ./data/eval_reports/enterprise_eval_latest.json \
  --dataset-path ./core/evaluation/datasets/enterprise_eval.jsonl \
  --output-path ./data/train/llm_sft.auto.jsonl

conda run -n tmf_project python train/check_local_llm_dataset.py \
  --dataset-path ./data/train/llm_sft.auto.jsonl \
  --report-path ./data/train/llm_sft.auto.report.json

python train/train_local_llm_qlora.py \
  --model-name /data/models/Qwen3-14B-Instruct \
  --train-path ./data/train/llm_sft.auto.jsonl \
  --output-dir ./artifacts/qwen3-14b-qlora-v1 \
  --max-seq-length 4096 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --num-train-epochs 2
```
