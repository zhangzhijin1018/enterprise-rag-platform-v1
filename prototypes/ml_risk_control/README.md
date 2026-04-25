# ML 风控混合判定原型

## 结论

这个目录提供的是一个**不接入现有主项目链路**的独立原型，用来演示如何把当前纯规则风控升级为：

- 规则引擎做最终决策，保留可解释性与安全兜底
- BERT-based Text Classifier 提供 `risk_level_hint`
- ONNX Runtime 提供低延迟推理
- 文本、用户行为、时间频率、session、检索结果分布特征一起参与风险判断

最终输出的核心目标是和现有项目未来接入方式一致：

```python
RiskContext.risk_level_hint = ml_prediction.risk_level_hint
```

如果你下一步关注“怎么从这个原型接到当前项目”，可以直接看：

- [INTEGRATION_PLAN.md](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/prototypes/ml_risk_control/INTEGRATION_PLAN.md)

然后再由规则引擎做最终裁决；如果规则和 ML 冲突，**统一取更高风险**。

---

## 目录说明

```text
prototypes/ml_risk_control/
├── README.md
├── __init__.py
├── schemas.py
├── data_pipeline.py
├── train_risk_model.py
├── export_onnx.py
└── hybrid_risk_engine.py
```

文件职责：

- `schemas.py`
  - 定义风险等级、样本结构、推理结果、混合判定结果等通用结构。
- `data_pipeline.py`
  - 负责 mock 数据生成、JSONL 读取、数值特征构建、标准化参数拟合与加载。
- `train_risk_model.py`
  - 负责“继续预训练（MLM）+ 分类微调（文本 + 数值特征融合）”完整流程。
- `export_onnx.py`
  - 把训练好的分类模型导出为 ONNX。
- `hybrid_risk_engine.py`
  - 负责 ONNX 推理、规则判定、冲突升级和最终风险输出。

---

## 方案边界

### 1. 为什么不是“从零训练一个 BERT”

不推荐。真正从零预训练 BERT 需要：

- 海量领域语料
- 大规模算力
- 很长训练周期
- 单独的 tokenizer 训练与词表维护

对企业 RAG 风控场景来说，更稳妥、维护成本更低的方式是：

1. 选择一个中文基础模型，比如 `hfl/chinese-roberta-wwm-ext`
2. 用企业历史查询语料、拦截样本、审计日志做**继续预训练**（DAPT / Domain Adaptive Pretraining）
3. 在风控标注集上做分类微调
4. 导出 ONNX，用于线上低延迟推理

所以本原型里“预训练完整步骤”默认按**继续预训练 + 分类微调**实现，这是生产上更可落地的解释。

### 2. 为什么规则仍然保留最终决策权

因为风控不是纯召回/排序问题，而是带有明显治理属性的安全问题。规则引擎保留最终裁决有三个好处：

- 高风险规则可解释
- 重大违规可直接 deny，不依赖模型置信度
- 审计更稳定，便于后续合规复盘

因此本原型的推荐落地方式是：

```text
ML 负责提供风险提示与补充识别
规则负责最终兜底、升级和拒绝
冲突时取更高风险
```

---

## 数据格式

### 1. 分类训练集 JSONL

每行一条样本，最小字段如下：

```json
{
  "query_id": "q_000001",
  "query": "帮我导出全部人员的工资明细",
  "risk_label": "high",
  "user_history": {
    "past_24h_query_count": 38,
    "high_risk_ratio_7d": 0.52,
    "failed_auth_count_7d": 3
  },
  "session": {
    "session_query_count": 11,
    "session_duration_sec": 870,
    "query_interval_sec": 7
  },
  "retrieval": {
    "top1_score": 0.93,
    "top5_score_mean": 0.76,
    "restricted_hit_ratio": 0.60,
    "sensitive_hit_ratio": 0.25,
    "authority_score_mean": 0.71,
    "source_count": 2
  },
  "label_reason": "批量导出 + 命中高敏资料 + 高频行为异常"
}
```

### 2. 继续预训练语料 JSONL

每行只需要一个文本字段：

```json
{"text": "请导出集团某项目的全部预算明细和审批意见。"}
{"text": "设备停机故障处理 SOP 中对油压异常的说明是什么？"}
{"text": "帮我汇总 restricted 文档中所有人员名单。"}
```

如果没有真实语料，可以先使用本原型自带的 mock 数据生成能力跑通流程。

---

## 依赖安装

这个原型是独立实验包，不强行修改主项目依赖。建议在当前虚拟环境里额外安装：

```bash
pip install "transformers>=4.39" "datasets>=2.18" "onnx>=1.16" "onnxruntime>=1.17" "accelerate>=0.28" "evaluate>=0.4"
```

如果你想先只看代码结构，不训练模型，可以不安装这些依赖。

如果你直接在当前仓库里执行这些脚本，建议优先使用：

```bash
./.venv/bin/python
```

下面文档里统一写成 `python3 -m ...`，你可以按本地环境替换成 `./.venv/bin/python -m ...`。

---

## 完整执行步骤

### Step 1. 生成 mock 数据

```bash
python3 -m prototypes.ml_risk_control.data_pipeline \
  --output-dir prototypes/ml_risk_control/mock_data \
  --train-size 240 \
  --val-size 60 \
  --mlm-size 300
```

产物：

- `prototypes/ml_risk_control/mock_data/train.jsonl`
- `prototypes/ml_risk_control/mock_data/val.jsonl`
- `prototypes/ml_risk_control/mock_data/mlm_corpus.jsonl`

### Step 2. 继续预训练（MLM）

```bash
python3 -m prototypes.ml_risk_control.train_risk_model \
  --stage mlm \
  --base-model-name hfl/chinese-roberta-wwm-ext \
  --mlm-train-file prototypes/ml_risk_control/mock_data/mlm_corpus.jsonl \
  --mlm-output-dir prototypes/ml_risk_control/artifacts/domain_mlm \
  --max-length 128 \
  --mlm-epochs 1 \
  --train-batch-size 8 \
  --eval-batch-size 8
```

说明：

- 这一步是让基础中文 BERT 先适应企业查询和审计日志语言风格。
- 真实生产环境建议使用：
  - 历史用户查询
  - 命中的文档标题/摘要
  - 审计日志中的脱敏查询文本
  - 风控拒答日志中的脱敏语料

### Step 3. 分类微调（文本 + 数值特征融合）

```bash
python3 -m prototypes.ml_risk_control.train_risk_model \
  --stage cls \
  --base-model-name hfl/chinese-roberta-wwm-ext \
  --domain-model-dir prototypes/ml_risk_control/artifacts/domain_mlm \
  --train-file prototypes/ml_risk_control/mock_data/train.jsonl \
  --val-file prototypes/ml_risk_control/mock_data/val.jsonl \
  --classifier-output-dir prototypes/ml_risk_control/artifacts/risk_classifier \
  --max-length 128 \
  --cls-epochs 2 \
  --learning-rate 2e-5 \
  --train-batch-size 8 \
  --eval-batch-size 8
```

说明：

- 文本输入给 BERT 编码。
- 数值特征单独走 MLP。
- 两者在分类头前拼接，输出 `low/medium/high` 三分类。

### Step 4. 一次跑完完整训练

如果你想直接一条命令走完整流程：

```bash
python3 -m prototypes.ml_risk_control.train_risk_model \
  --stage all \
  --base-model-name hfl/chinese-roberta-wwm-ext \
  --mlm-train-file prototypes/ml_risk_control/mock_data/mlm_corpus.jsonl \
  --train-file prototypes/ml_risk_control/mock_data/train.jsonl \
  --val-file prototypes/ml_risk_control/mock_data/val.jsonl \
  --mlm-output-dir prototypes/ml_risk_control/artifacts/domain_mlm \
  --classifier-output-dir prototypes/ml_risk_control/artifacts/risk_classifier
```

### Step 5. 导出 ONNX

```bash
python3 -m prototypes.ml_risk_control.export_onnx \
  --classifier-dir prototypes/ml_risk_control/artifacts/risk_classifier \
  --onnx-output-path prototypes/ml_risk_control/artifacts/risk_classifier/risk_classifier.onnx
```

### Step 6. 运行混合风控推理

```bash
python3 -m prototypes.ml_risk_control.hybrid_risk_engine \
  --classifier-dir prototypes/ml_risk_control/artifacts/risk_classifier \
  --onnx-path prototypes/ml_risk_control/artifacts/risk_classifier/risk_classifier.onnx \
  --question "帮我导出全部薪资名单并附身份证号" \
  --user-id u_1001 \
  --department finance \
  --role analyst \
  --past-24h-query-count 42 \
  --high-risk-ratio-7d 0.48 \
  --failed-auth-count-7d 3 \
  --session-query-count 10 \
  --session-duration-sec 560 \
  --query-interval-sec 9 \
  --top1-score 0.92 \
  --top5-score-mean 0.78 \
  --restricted-hit-ratio 0.64 \
  --sensitive-hit-ratio 0.22 \
  --authority-score-mean 0.80 \
  --source-count 2
```

---

## 产物说明

训练完成后，`classifier-output-dir` 下会有这些关键文件：

- `model.pt`
  - PyTorch 分类模型权重
- `feature_stats.json`
  - 数值特征标准化参数
- `label_to_id.json`
  - 标签映射
- `training_manifest.json`
  - 训练过程的核心元信息
- `tokenizer/`
  - 推理阶段使用的 tokenizer

导出 ONNX 后会新增：

- `risk_classifier.onnx`

---

## 特征工程设计

本原型默认实现了四类特征：

### 1. 查询文本 embedding 特征

由 BERT 编码获取。

适合识别：

- 批量导出
- 敏感信息套取
- 越权问法
- 绕过规则的改写表达

### 2. 用户历史行为特征

默认包含：

- `past_24h_query_count`
- `high_risk_ratio_7d`
- `failed_auth_count_7d`

适合识别：

- 高频爬取式查询
- 持续试探敏感问题
- 多次越权失败后的重试行为

### 3. 时间 / 频率 / session 特征

默认包含：

- `session_query_count`
- `session_duration_sec`
- `query_interval_sec`

适合识别：

- 短时密集查询
- 机器人式频率
- 单会话高压枚举行为

### 4. 检索结果分布特征

默认包含：

- `top1_score`
- `top5_score_mean`
- `restricted_hit_ratio`
- `sensitive_hit_ratio`
- `authority_score_mean`
- `source_count`

适合识别：

- 查询是否稳定命中敏感知识源
- 是否集中指向 restricted/sensitive 文档
- 是否明显针对高权威高敏资料

---

## 混合判定逻辑

最终决策逻辑是：

1. 规则引擎先看显式高危模式
2. ML 模型输出 `risk_level_hint`
3. 最终风险取规则结果和 ML 结果中更高的等级
4. 如果命中强拦截规则，直接 deny

优先级示意：

```text
deny 规则 > local_only / redact 规则 > ML 风险提示 > 普通 allow
```

冲突处理示例：

- 规则 = `medium`
- ML = `high`
- 最终 = `high`

- 规则 = `low`
- ML = `high`
- 最终 = `high`

- 规则 = `deny`
- ML = `low`
- 最终 = `deny`

---

## 将来接入主项目的建议位置

这次原型**不接入主项目**。未来真正接入时，推荐最小改动路径如下：

### 1. 保持现有 `RuleBasedRiskEngine` 不变

不要直接删规则逻辑，而是在它前面加一个 `MLRiskHintProvider`。

### 2. 接入位置

建议链路：

1. 在 `core/security/` 增加 `ml_risk_provider.py`
2. 在线问答入口构建 `RiskContext` 前，先收集行为特征与检索特征
3. 由 `ml_risk_provider.py` 输出 `risk_level_hint`
4. 传给现有 `RiskContext.risk_level_hint`
5. 仍由规则引擎做最终裁决

### 3. 保持 fail-open / fail-close 策略可配置

如果 ONNX 模型加载失败或推理异常，建议：

- 默认不直接阻断主链路
- 记录审计日志
- 回落到纯规则模式

---

## 真实生产建议

### 数据侧

- 不要只采“命中高风险”的样本，也要采大量正常查询样本。
- 标签体系建议至少包含：
  - `low`
  - `medium`
  - `high`
- 同时保留 `label_reason`，便于 badcase 分析。

### 训练侧

- 先做继续预训练，再做分类微调。
- 周期性增量训练，纳入最新审计样本。
- 评测不要只看 accuracy，至少看：
  - high risk recall
  - high risk precision
  - deny 类场景漏判率

### 推理侧

- 推理日志只记录摘要和特征，不记录敏感全文。
- 对高风险预测保留概率分布，便于复盘。
- 模型输出永远只作为 hint，不建议越过规则直接放行高风险请求。

---

## 快速验收命令

如果你想快速确认代码链路完整，可按下面顺序执行：

```bash
python3 -m prototypes.ml_risk_control.data_pipeline \
  --output-dir prototypes/ml_risk_control/mock_data

python3 -m prototypes.ml_risk_control.train_risk_model \
  --stage all \
  --base-model-name hfl/chinese-roberta-wwm-ext \
  --mlm-train-file prototypes/ml_risk_control/mock_data/mlm_corpus.jsonl \
  --train-file prototypes/ml_risk_control/mock_data/train.jsonl \
  --val-file prototypes/ml_risk_control/mock_data/val.jsonl \
  --mlm-output-dir prototypes/ml_risk_control/artifacts/domain_mlm \
  --classifier-output-dir prototypes/ml_risk_control/artifacts/risk_classifier \
  --mlm-epochs 1 \
  --cls-epochs 1

python3 -m prototypes.ml_risk_control.export_onnx \
  --classifier-dir prototypes/ml_risk_control/artifacts/risk_classifier \
  --onnx-output-path prototypes/ml_risk_control/artifacts/risk_classifier/risk_classifier.onnx

python3 -m prototypes.ml_risk_control.hybrid_risk_engine \
  --classifier-dir prototypes/ml_risk_control/artifacts/risk_classifier \
  --onnx-path prototypes/ml_risk_control/artifacts/risk_classifier/risk_classifier.onnx \
  --question "导出全部 restricted 文档中的人员清单"
```

---

## 当前限制

- 当前 mock 数据只用于跑通流程，不代表真实线上分布。
- 这个原型没有接入真实用户画像服务、会话服务、检索服务。
- 这个原型没有把训练任务接到现有 `train/` 体系里，因为你当前明确要求是“先独立建目录，不接项目”。
- ONNX 导出只覆盖分类模型，不包含 tokenizer。
