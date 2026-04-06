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

默认样例：`core/evaluation/datasets/sample_eval.jsonl`。

## Running

- API：`POST /eval`，可选 body `{"dataset_path": "/path/to.jsonl"}`。
- 代码：`from core.evaluation.ragas_runner import run_ragas_eval`。

需要配置 `OPENAI_API_KEY`（RAGAS 与部分指标依赖 LLM）。报告输出目录由 `EVAL_OUTPUT_DIR` 控制。

## Metrics

- **Faithfulness** — 答案是否可由上下文支撑  
- **Answer relevancy**（RAGAS 字段名 `answer_relevancy`）— 答案与问题的相关性  
- **Context recall / precision** — 上下文覆盖与精确度  

评测汇总均值会写入 Prometheus Gauge `erp_ragas_faithfulness_avg`（faithfulness 列存在时）。
