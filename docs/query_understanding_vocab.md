# Query Understanding Vocab Guide

## 目标

这份文档只解决一件事：

> 当你要给“新疆能源（集团）有限责任公司 企业知识智能副驾”补业务词时，应该往哪里加、按什么格式加、优先加哪些词。

默认词典文件：

- [data/config/query_understanding_vocab.json](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/data/config/query_understanding_vocab.json)

配置入口：

- [core/config/settings.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/config/settings.py)
- [.env.example](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/.env.example)

加载逻辑：

- [core/orchestration/query_understanding_vocab.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/orchestration/query_understanding_vocab.py)

使用入口：

- [core/orchestration/nodes/analyze_query.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/orchestration/nodes/analyze_query.py)

## 性能说明

当前不是“每次请求读一次 JSON 再现场做所有准备”，而是：

1. 本地 JSON 文件仍然作为词典来源
2. 运行时先读取并 merge 默认词典和外部词典
3. 再把词典预编译成内存索引
4. 后续请求直接复用：
   - scene regex
   - department regex
   - equipment regex
   - business domain 索引
   - alias 索引

所以当前性能瓶颈不在“每次读文件”，而更可能在“词典本身变得极大后，匹配成本会上升”。

当前这套做法的优点：

- 结构简单
- 仍然是本地文件，方便 git 管理
- 运行时不会重复编译大部分规则

如果后面词典继续变大，最合理的升级顺序是：

1. 继续保持本地文件
2. 优先优化索引结构
3. 再考虑热加载
4. 最后才考虑数据库或管理后台

## 当前词典结构

当前词典主要分 5 类。

### 1. 通用信号词表

这类字段的值是 `list[str]`，用于触发 query understanding 的基础规则。

当前常见字段：

- `department_suffixes`
- `shift_keywords`
- `followup_prefixes`
- `policy_keywords`
- `procedure_keywords`
- `meeting_keywords`
- `project_keywords`
- `equipment_keywords`
- `system_keywords`

适合放什么：

- 明显的类别词
- 稳定的高频词
- 不需要做“规范名归一”的词

不适合放什么：

- 临时口语
- 很长的完整句子
- 很容易和别的词混淆的词

### 2. 业务域词表

字段：

- `business_domain_keywords`

结构：

```json
{
  "business_domain_keywords": {
    "fuel_management": ["燃料", "煤场", "入厂煤"]
  }
}
```

规则：

- key 必须是稳定的、代码里可长期使用的业务域名
- value 是能代表这个业务域的词列表

推荐命名：

- `equipment_maintenance`
- `dispatch`
- `project_management`
- `it_ops`
- `safety_production`
- `procurement`
- `environmental_protection`
- `fuel_management`
- `power_generation`
- `emergency_response`
- `contract_management`

### 3. 部门别名字典

字段：

- `department_aliases`

结构：

```json
{
  "department_aliases": {
    "安全环保部": ["安全环保部", "安环部", "安全部", "环保部"]
  }
}
```

规则：

- key 是规范部门名
- value 是所有常见叫法
- 建议把规范名本身也放进去

命中后会进入：

- `department`
- `owner_department`

### 4. 场站/区域别名字典

字段：

- `site_aliases`

结构：

```json
{
  "site_aliases": {
    "准东二矿": ["准东二矿", "二矿", "准东矿区二矿"]
  }
}
```

命中后会进入：

- `plant`
- `applicable_site`

### 5. 系统别名字典

字段：

- `system_aliases`

结构：

```json
{
  "system_aliases": {
    "安全生产管理平台": ["安全生产管理平台", "安生平台", "安监平台"]
  }
}
```

命中后会进入：

- `system_name`

## 最推荐的维护顺序

如果你每次只想补最值钱的内容，按这个顺序来：

1. `department_aliases`
2. `site_aliases`
3. `system_aliases`
4. `business_domain_keywords`
5. `equipment_keywords`

原因很简单：

- 部门、场站、系统这三类最容易直接影响 retrieval filter 和排序
- 业务域词会影响 `query_scene / preferred_retriever / metadata_intent`
- 设备词虽然重要，但歧义通常会更多一点

## 加词原则

### 原则 1：优先加“规范名 + 常见别名”

不要只加简称。

推荐：

```json
{
  "燃料管理部": ["燃料管理部", "燃料部", "燃管部"]
}
```

不推荐：

```json
{
  "燃料管理部": ["燃管部"]
}
```

### 原则 2：优先加高频、稳定、对检索有价值的词

优先加：

- 部门简称
- 场站简称
- 系统简称
- 设备简称
- 业务域典型术语

先别加：

- 只出现一次的随口说法
- 太长的自然语言短句
- 含义不稳定的模糊词

### 原则 3：一个别名尽量只归到一个规范名

如果同一个别名可能映射到多个实体，优先：

1. 先不加
2. 或者改成更明确的别名形式

不要把这种词典做成“模糊猜测库”。

### 原则 4：业务域词不要贪多

业务域词太多、太泛，会导致 domain 误判。

推荐：

- 放能强代表该域的词
- 放 5 到 12 个核心词

不推荐：

- 把所有相关词都塞进去

## 最小修改模板

### 1. 新增部门别名

```json
{
  "department_aliases": {
    "燃料管理部": ["燃料管理部", "燃料部", "燃管部"]
  }
}
```

### 2. 新增场站别名

```json
{
  "site_aliases": {
    "翻车机卸煤区": ["翻车机卸煤区", "翻车机区域", "卸煤区"]
  }
}
```

### 3. 新增系统别名

```json
{
  "system_aliases": {
    "设备缺陷管理系统": ["设备缺陷管理系统", "缺陷系统", "缺陷平台"]
  }
}
```

### 4. 新增业务域

```json
{
  "business_domain_keywords": {
    "fuel_management": ["燃料", "煤场", "入厂煤", "采样", "煤质", "配煤"]
  }
}
```

## 推荐的提交流程

每次加词，建议最少做这 3 步：

1. 改 [data/config/query_understanding_vocab.json](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/data/config/query_understanding_vocab.json)
2. 补一个 `tests/unit/test_query_planning.py` 的最小用例
3. 同步更新这份文档或 `README` 中的词典说明

## 建议的测试问题模板

### 部门

```text
燃管部的设备台账在哪里看
```

### 场站

```text
一号输煤线的巡检记录在哪里
```

### 系统

```text
在燃料平台怎么查入厂煤采样记录
```

### 业务域

```text
入厂煤采样记录怎么查
```

## 什么时候该改词典，什么时候该改代码

优先改词典：

- 新增部门简称
- 新增场站简称
- 新增系统简称
- 新增业务术语
- 新增设备名

考虑改代码：

- 词典已经命中，但 `query_scene` 还是总错
- 词典已经归一，但 retrieval route 明显不合理
- 多个业务域长期相互误伤
- 某类 query 需要新规则而不是新词

## 当前最值得继续补的词

如果你后面继续扩新疆能源词典，优先补：

1. 下属单位/厂站简称
2. 机组和生产装置简称
3. 调度、集控、燃料、环保相关系统简称
4. 安全生产、两票、工作票、操作票相关术语
5. 技改、立项、可研、初设、招采、合同相关术语
