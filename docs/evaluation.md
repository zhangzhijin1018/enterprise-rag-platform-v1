# Evaluation

## Dataset format

JSONL，每行：

```json
{
  "question": "...",
  "ground_truth": "...",
  "contexts": ["可选参考上下文片段"]
}
```

当前默认评测集：`core/evaluation/datasets/enterprise_eval.jsonl`。  
保留最小样例：`core/evaluation/datasets/sample_eval.jsonl`，适合 smoke / demo。

当前样例集已经开始覆盖这些企业场景：

- 基础 FAQ / 错误码
- 排班与结构化事实查询
- 多文档对比
- 多版本制度冲突
- 权限拒答
- 无依据拒答

推荐字段可以进一步扩到：

```json
{
  "question": "...",
  "ground_truth": "...",
  "contexts": ["..."],
  "scenario": "policy_conflict",
  "tags": ["policy", "conflict", "versioning"],
  "expected_refusal": false,
  "expected_conflict": true
}
```

## Running

- API：`POST /eval`，可选 body `{"dataset_path": "/path/to.jsonl"}`。
- 代码：`from core.evaluation.ragas_runner import run_ragas_eval`。

需要配置 `OPENAI_API_KEY`（RAGAS 与部分指标依赖 LLM）。报告输出目录由 `EVAL_OUTPUT_DIR` 控制。

当前 `/eval` 会同时产出两份报告：

- `ragas_report_*.json`
  - 完整机器可读结果
- `ragas_report_*.md`
  - 面向 badcase 回放的 explainability 报告

当前 JSON 报告还会额外包含：

- `query_understanding_report`
  - `top_scenes`
  - `top_guardrail_scenarios`
  - `top_llm_enhanced_scenarios`
  - `recommendations`

API 返回里：

- `report_path` 指向 JSON
- `analysis_path` 指向 Markdown explainability report

当前报告除了 RAGAS 的逐题结果，还会额外保留每题的治理信号：

- `refusal / refusal_reason`
- `answer_mode`
- `data_classification`
- `model_route`
- `conflict_detected / conflict_summary`
- `audit_id`
- `matched_routes`
- `metadata_boosted`
- `enterprise_entity_boosted`
- `enterprise_entity_matches`
- `governance_boosted`
- `explainable_citation_count`
- `analysis_confidence`
- `analysis_source`
- `analysis_reason`

## Metrics

- **Faithfulness** — 答案是否可由上下文支撑  
- **Answer relevancy**（RAGAS 字段名 `answer_relevancy`）— 答案与问题的相关性  
- **Context recall / precision** — 上下文覆盖与精确度  

评测汇总均值会写入 Prometheus Gauge `erp_ragas_faithfulness_avg`（faithfulness 列存在时）。

同时 `summary` 还会额外统计：

- `sample_count`
- `refusal_rate`
- `conflict_detected_rate`
- `classification:*`
- `model_route:*`
- `matched_route:*`
- `entity_match:*`
- `metadata_boost_hit_rate`
- `enterprise_entity_boost_hit_rate`
- `governance_boost_hit_rate`
- `avg_explainable_citations`
- `avg_analysis_confidence`
- `analysis_source:*`

这几项不是 RAGAS 指标，但对企业知识副驾很重要，因为它们能反映：

- 系统是否过度拒答
- 多版本冲突是否被显式暴露
- 不同数据分级和模型路由策略出现的频率
- 哪些 query route 最常真正命中最终证据
- metadata boost 和治理排序到底有没有在实际样本里起作用
- 企业实体归一后的 `department / site / system / business_domain` 有没有真正影响最终证据排序
- 当前回答里有多少证据已经能提供“为什么被选中”的解释
- 当前 query understanding 有多少题是高置信规则直走、多少题依赖 LLM 补判、多少题被 guardrail 保守回退

建议后续在 badcase 回放里继续追加两类审计相关字段：

- `risk_level`
- `egress_strategy`

因为当前数据分级已经真正影响生成链路，这两项能帮助判断：

- 某题为什么被拒答
- 某题为什么只给了最小片段
- 某题是不是因为高风险而触发了更严格日志与审计

当前 Markdown explainability report 会优先按这些维度挑选最值得看的 badcase：

- `expected_refusal / expected_conflict` 是否与实际不一致
- `faithfulness / answer_relevancy / context_recall / context_precision` 是否偏低
- 是否触发 `refusal / conflict_detected`
- 命中了哪些 `matched_routes`
- 是否触发 `metadata_boost / governance_boost`
- `analysis_source` 是 `heuristic / llm_enhanced / *_guardrail` 中哪一种
- `analysis_reason` 是否能解释当前 route 为什么这样选

当前 Markdown 报告还会专门增加一节 `Query Understanding Tuning`，用来回答：

- 哪些 `query_scene` 在当前评测里最常出现
- 哪些 `scenario` 最常触发 `guardrail`
- 哪些 `scenario` 最常依赖 `llm_enhanced`
- 当前更适合“继续补规则”还是“继续补 badcase 样本”

## Recommended coverage

这次查询路由与检索策略重构后，离线评测数据建议至少覆盖以下问题类型：

- 多轮承接问题：上一轮给出部门/班次，本轮只问“今天是谁值班”“那如果是 3 号线呢”
- 结构化事实查询：时间 / 部门 / 人员 / 班次 / 联系人 / 排班类问题
- 精确实体查询：制度编号、设备名、错误码、字段名、版本号
- 复杂问题：原因 + 处理、方案说明、对比分析
- 需澄清问题：缺目标对象、缺部门、缺时间、缺运行环境、缺比较对象
- 权限问题：普通用户问 restricted 文档、项目组不匹配、跨部门资料访问
- 冲突问题：同主题多版本制度、会议纪要与正式制度不一致
- 无依据问题：知识库中没有证据，系统应明确拒答

建议在 badcase 分析里单独观察：

- `resolved_query` 是否真的补全了上下文，而不是简单拼接噪声
- `structured_filters` 是否提取到了部门 / 时间 / 班次等关键信息
- `keyword` 路线是否提升了精确术语命中
- `HyDE` 是否只在抽象解释类问题中开启，避免污染精确事实查询
