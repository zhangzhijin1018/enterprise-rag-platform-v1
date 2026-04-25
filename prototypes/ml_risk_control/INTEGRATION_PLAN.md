# ML 风控原型接入设计稿

## 结论

这套 ML 风控原型未来接入当前项目时，推荐采用**最小侵入式接入**：

1. 保留现有 `RuleBasedRiskEngine` 作为最终裁决器，不改最终决策权。
2. 新增一个 `MLRiskHintProvider`，只负责产出 `risk_level_hint`。
3. 在现有 3 个风控阶段中，按阶段补特征并注入 `risk_level_hint`：
   - 请求入口
   - 检索后
   - 生成前
4. 规则与 ML 冲突时统一取更高风险。
5. ML 推理失败时回落到纯规则模式，不阻断主链路。

一句话说，就是：

```text
ML 负责补风险识别能力
规则负责最终安全裁决
先加 hint，不改主干
```

---

## 步骤

### 第一步：只加 ML Hint Provider，不改现有规则引擎职责

当前项目里，`risk_level_hint` 已经在 [core/security/risk_engine.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/security/risk_engine.py) 的 `RiskContext` 中预留好了。

现状：

- `build_risk_context(...)` 会构造 `RiskContext`
- `safe_evaluate_risk(...)` 会调用 `risk_engine.evaluate(context)`
- `RuleBasedRiskEngine.evaluate(...)` 内部已经会做：
  - `context.risk_level_hint`
  - `assess_query_risk(context.question)`
  两者取高

所以第一轮最稳的接入方式不是改 `RuleBasedRiskEngine`，而是：

1. 在 `core/security/` 下新增 `ml_risk_provider.py`
2. 对外暴露统一接口：

```python
class MLRiskHintProvider(Protocol):
    def predict(self, context: RiskContext, feature_bundle: dict[str, Any]) -> str | None:
        ...
```

3. 在风控执行前先尝试拿到 `risk_level_hint`
4. 再把 hint 塞回 `RiskContext`
5. 继续走原有 `RuleBasedRiskEngine`

这样做的好处：

- 不改 `RiskDecision` 结构
- 不改 route / graph / response 的既有约定
- 不破坏现有拒答、`local_only`、`minimize` 等规则动作

---

### 第二步：明确未来会改哪些文件

后续真正接入时，建议优先改这几个文件，按顺序推进：

1. `core/config/settings.py`
   - 新增 ML 风控相关配置
2. `core/security/ml_risk_provider.py`
   - 新增 ONNX 推理 provider
3. `core/security/risk_features.py`
   - 新增在线特征构造逻辑
4. `core/services/runtime.py`
   - 把 ML provider 装到 runtime
5. `apps/api/routes/chat.py`
   - 在请求级风控前先补请求阶段特征

第二轮再考虑：

6. `core/orchestration/nodes/retrieve_docs.py`
   - 在检索后补 retrieval 分布特征
7. `core/orchestration/nodes/generate_answer.py`
   - 在生成前复用已有特征和状态，再跑一次风险 hint

注意：

- 第一轮不要同时大改 7 个文件。
- 推荐先做“请求级接入”，因为改动最小，也最容易灰度。

---

### 第三步：新增配置项，但保持默认关闭

当前配置里只有：

- `enable_risk_engine`
- `risk_engine_provider="rule_based"`
- `risk_engine_fail_open`

建议后续新增以下配置，默认都偏保守：

```python
enable_ml_risk_hint: bool = False
ml_risk_hint_provider: Literal["onnx", "mock", "disabled"] = "disabled"
ml_risk_model_dir: str = "modes/ml-risk"
ml_risk_onnx_path: str = "modes/ml-risk/risk_classifier.onnx"
ml_risk_timeout_ms: int = 50
ml_risk_fail_open: bool = True
ml_risk_request_stage_enabled: bool = True
ml_risk_retrieval_stage_enabled: bool = True
ml_risk_generation_stage_enabled: bool = False
ml_risk_log_probabilities: bool = True
```

设计原因：

- 默认关闭，避免原型代码未成熟时影响线上链路
- 可单独控制不同阶段是否启用 ML hint
- 推理超时和失败策略显式可配

---

### 第四步：在线特征怎么映射到当前项目

当前原型里的数值特征可以直接映射到现有项目链路，但要分阶段补齐。

#### 1. 请求阶段可直接拿到的特征

来源：

- `apps/api/routes/chat.py`
- 请求体 `body`
- 用户上下文 `user_context`

建议首批接入特征：

- `past_24h_query_count`
- `high_risk_ratio_7d`
- `failed_auth_count_7d`
- `session_query_count`
- `session_duration_sec`
- `query_interval_sec`

说明：

- 这些特征不依赖检索结果，可以最先接入。
- 如果当前项目还没有用户行为服务，第一轮可以先走默认值或 mock provider。

#### 2. 检索阶段可补的特征

来源：

- [core/orchestration/nodes/retrieve_docs.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/orchestration/nodes/retrieve_docs.py)
- `fused` / `reranked_hits`
- `data_classification`
- `model_route`

建议补这些特征：

- `top1_score`
- `top5_score_mean`
- `restricted_hit_ratio`
- `sensitive_hit_ratio`
- `authority_score_mean`
- `source_count`

说明：

- 这些特征是原型里最有价值的一组，因为它们和企业数据分级、权限命中、知识来源分布直接相关。
- 检索后再跑一次 ML hint，比单纯在入口判断更稳。

#### 3. 生成阶段是否需要再跑一次

结论：第一阶段不建议默认开启。

原因：

- 当前生成前的风险规则已经比较明确
- 生成阶段再推理一次，会增加一次额外模型开销
- 如果 retrieval 阶段已经拿到了大部分检索分布特征，收益不一定大

推荐策略：

- 第一轮：请求阶段 + 检索阶段
- 第二轮：观测 badcase 后再决定要不要加生成前 hint

---

### 第五步：推荐的运行时装配方式

当前 [core/services/runtime.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/services/runtime.py) 里是：

```python
self.risk_engine = RuleBasedRiskEngine(self.settings)
```

未来建议变成：

```python
self.ml_risk_provider = build_ml_risk_provider(self.settings)
self.risk_engine = RuleBasedRiskEngine(self.settings)
```

注意不要直接把 `risk_engine` 替换成“纯 ML 风控引擎”，而是让它们分别承担不同职责：

- `ml_risk_provider`
  - 提供 hint
- `risk_engine`
  - 做最终 allow / deny / local_only / minimize / redact 决策

---

### 第六步：推荐的数据流

#### 1. 请求入口

当前入口位于 [apps/api/routes/chat.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/apps/api/routes/chat.py)。

推荐顺序：

1. 构建 `user_context`
2. 构建 `access_filters`
3. 构建 request-stage 特征
4. 调用 `ml_risk_provider.predict(...)`
5. 把结果写入 `RiskContext.risk_level_hint`
6. 调用现有 `safe_evaluate_risk(...)`

#### 2. 检索后

当前接入点位于 [core/orchestration/nodes/retrieve_docs.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/orchestration/nodes/retrieve_docs.py)。

推荐顺序：

1. 检索融合后得到 `fused`
2. 计算分级分布和来源分布特征
3. 更新 `feature_bundle`
4. 再跑一次 `ml_risk_provider.predict(...)`
5. 重新构建 `risk_context`
6. 继续用 `RuleBasedRiskEngine` 做最终决策

#### 3. 生成前

当前接入点位于 [core/orchestration/nodes/generate_answer.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/orchestration/nodes/generate_answer.py)。

建议第一轮先不加，只保留扩展位。

---

### 第七步：`risk_level_hint` 的合并策略

未来正式接入时，推荐保持当前项目已有语义不变：

```text
最终 risk_level = max(规则启发式风险, ML risk_level_hint)
```

更具体一点：

1. ML 只输出：
   - `low`
   - `medium`
   - `high`
2. 不直接输出：
   - `deny`
   - `local_only`
   - `minimize`
3. 最终动作仍由规则引擎决定

这样做的价值是：

- ML 不直接控制策略动作，风险更低
- 线上审计更容易解释
- 规则系统依然是稳定的兜底器

---

### 第八步：失败回退策略

ML hint 接入后，一定要显式设计失败回退。

推荐策略：

1. ONNX 模型文件缺失
   - 记录 warning
   - 回落到纯规则模式
2. 推理超时
   - 记录 timeout 指标
   - 回落到纯规则模式
3. 特征缺失
   - 对缺失特征填默认值
   - 不直接抛异常阻断问答
4. provider 异常
   - 根据 `ml_risk_fail_open=True` 回退

注意：

- `ml_risk_fail_open` 指的是“ML hint 失败不阻断主链路”
- 不等于“风控失败直接放行高风险”
- 纯规则风控仍然会继续执行

---

### 第九步：审计和可观测建议

当前项目已经有审计与告警能力，所以 ML 接入后不要另起一套日志体系。

建议新增但只记录最小必要信息：

- `ml_risk_hint`
- `ml_risk_confidence`
- `ml_risk_provider`
- `ml_risk_latency_ms`
- `ml_risk_fallback`
- `ml_feature_version`

不建议记录：

- 原始敏感全文
- 完整用户画像明细
- 检索全文内容

建议记录位置：

- 复用现有审计日志链路
- 通过 `state` 或 `extra` 传入 observability

---

### 第十步：推荐实施顺序

按你当前项目约束，推荐分 4 轮落地：

#### 第 1 轮

目标：

- 加 settings
- 加 `ml_risk_provider.py`
- runtime 装配 provider
- 请求入口支持 request-stage hint

特点：

- 改动最小
- 不依赖 retrieval 特征
- 最容易先灰度

#### 第 2 轮

目标：

- retrieval 节点补分布特征
- 检索后再跑一次 hint

特点：

- 风险识别质量会明显提升
- 但会开始触碰检索状态传递

#### 第 3 轮

目标：

- 审计字段补齐
- 监控和 badcase 回放补齐
- 开始离线评测

#### 第 4 轮

目标：

- 小流量灰度
- 比较纯规则 vs 规则+ML
- 根据 badcase 决定是否启用 generation-stage hint

---

## 示例

### 1. 推荐的 provider 接口草案

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from core.security.risk_engine import RiskContext


@dataclass
class MLRiskHintResult:
    risk_level_hint: str | None
    confidence: float | None = None
    provider: str = "disabled"
    latency_ms: float | None = None
    fallback: bool = False
    metadata: dict[str, Any] | None = None


class MLRiskHintProvider(Protocol):
    def predict(self, context: RiskContext, feature_bundle: dict[str, Any]) -> MLRiskHintResult:
        ...
```

### 2. 请求入口最小接入伪代码

```python
feature_bundle = build_request_risk_features(
    question=body.question,
    user_context=user_context,
    request=request,
)

hint_result = runtime.ml_risk_provider.predict(
    context=request_risk_context,
    feature_bundle=feature_bundle,
)

request_risk_context.risk_level_hint = hint_result.risk_level_hint
request_risk_decision = safe_evaluate_risk(
    runtime.risk_engine,
    request_risk_context,
    runtime.settings,
)
```

### 3. retrieval 阶段补特征的伪代码

```python
retrieval_feature_bundle = build_retrieval_risk_features(
    fused_hits=fused,
    data_classification=data_classification,
    model_route=model_route,
)

hint_result = runtime.ml_risk_provider.predict(
    context=risk_context,
    feature_bundle=retrieval_feature_bundle,
)

risk_context.risk_level_hint = hint_result.risk_level_hint
risk_decision = safe_evaluate_risk(runtime.risk_engine, risk_context, runtime.settings)
```

### 4. 第一轮建议不要做的事情

```text
不要直接把 risk_engine_provider 从 rule_based 改成 ml
不要让 ML 直接输出 deny / allow
不要为了接入 hint 修改 response schema
不要在生成后再做“敏感内容裁剪式风控”
```

---

## 影响范围

如果未来按这个设计接入，第一轮实际影响范围会很有限：

- 新增少量配置
- 新增 1 到 2 个 `core/security` 文件
- 修改 runtime 装配
- 修改请求入口风控前置逻辑

不会改变：

- 现有 API 对外接口
- 现有 response 字段约定
- 现有规则风控主体行为
- 现有 ACL / 数据分级 / 模型路由主链路

---

## 风险

1. 如果用户行为特征来源暂时不稳定，第一轮建议全部兜底默认值，不要阻塞主链路。
2. 如果检索分布特征构造不稳定，先只在请求阶段上 ML hint。
3. 如果 ML 误报较多，先只把它作为 `medium/high` 提示，不要直接影响 deny 规则。
4. 如果 ONNX 推理开销超出预期，优先只在请求阶段启用。

---

## 验证建议

未来真正接入时，至少做这 4 类验证：

1. 单元测试
   - provider 正常 / 超时 / 异常回退
   - 特征构造缺失值兜底
2. 集成测试
   - request-stage hint 注入后不影响原有接口返回
   - retrieval-stage hint 注入后不破坏拒答链路
3. 离线评测
   - high risk recall
   - false positive rate
4. 灰度验证
   - 纯规则 vs 规则+ML 的 badcase 对比

