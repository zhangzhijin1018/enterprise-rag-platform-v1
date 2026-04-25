# Ingestion Filetype Matrix

## 目标

这份文档只解决一件事：

> 当前项目里，不同文件类型是怎么 `parse -> chunk -> 入索引` 的，各自的优缺点是什么，后续应该优先补哪里。

相关代码入口：

- 解析器注册：[core/ingestion/parsers/registry.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/ingestion/parsers/registry.py)
- 入库主流程：[core/ingestion/pipeline.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/ingestion/pipeline.py)
- 统一切块器：[core/ingestion/chunkers/semantic_chunker.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/ingestion/chunkers/semantic_chunker.py)

## 当前总策略

当前不是“每种文件一个完全独立的 chunker”，而是：

```text
不同 parser 先增强结构
  -> 统一 SemanticChunker
  -> 按文件类型选择 chunk profile
  -> parent / child 分层切块
  -> 写入本地索引与 Milvus
```

这样做的优点：

1. 主链路统一，维护成本低
2. 不同文件类型仍然能吃到更贴合自身形态的切块策略
3. 后续调优时可以先改 parser 输出或 chunk profile，不必先重写整套 chunking 系统

## 文件类型对照表

| 文件类型 | 当前 parser 做什么 | 当前 chunk profile | 当前优点 | 当前短板 | 下一步最值得补 |
|------|------|------|------|------|------|
| `PDF` | 逐页抽文本，写入 `<!-- page:n -->` | 优先按页级 section，`child` 更保守 | 页码能继承到 citation；跨页混块减少 | 标题层级识别仍弱，目录/表格结构容易丢 | 补 PDF 标题检测、目录识别、表格感知 |
| `DOCX` | 保留 `Heading 1/2/...`、列表项、基础表格 | 结构感知切分，继续吃 heading | 章节路径更稳；制度/SOP 更适合当前 chunker | 复杂表格、嵌套列表、页眉页脚还没特别处理 | 补更强的标题/表格清洗和段落角色识别 |
| `PPTX` | 保留 slide 标题和 bullet level | slide 型更紧凑参数 | 适合课件、方案评审、培训材料 | shape 间语义关系还较弱，图表说明容易丢 | 补 slide 内 block 分组和备注区处理 |
| `CSV` | 每行转 `字段名: 值`，并补主键标题 | 行级更小参数、无 overlap | 行级检索锚点强，适合错误码、FAQ、配置项 | 多列表格、跨列依赖关系表达较弱 | 补主键列配置、表头语义映射 |
| `Markdown` | 基本保留原始标题结构 | 标题驱动切分 | 最贴当前 chunker，效果通常最好 | 表格和代码块没有专门 profile | 补表格/代码块的局部结构标签 |
| `HTML` | 去噪后把 `h1~h4/p/li` 转成 Markdown 风格文本 | 标题驱动切分 | 网页知识库很容易接入统一链路 | 网页导航、面包屑、重复栏目偶尔仍会混入 | 补网页主内容提取和重复块去重 |
| `TXT` | 保留纯文本，轻量清洗 | 通用文本 profile，参数更保守 | 最简单稳妥，日志/草稿可直接入库 | 几乎没有结构信号，召回更多依赖正文和 metadata | 补轻量标题检测、日志块识别 |

## 各类型细化说明

### 1. PDF

对应文件：

- [core/ingestion/parsers/pdf_parser.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/ingestion/parsers/pdf_parser.py)

当前行为：

1. 用 `pypdf` 逐页抽文本
2. 每页前写 `<!-- page:n -->`
3. `SemanticChunker` 遇到 `pdf` profile 时，优先按页切成页级 section

当前最适合：

- 制度 PDF
- 手册 PDF
- 导出的规范文档

当前最弱的点：

- 如果 PDF 本身抽不到标题层级，section 结构仍偏弱
- 表格、页眉页脚、页脚注释可能带噪声

### 2. DOCX

对应文件：

- [core/ingestion/parsers/docx_parser.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/ingestion/parsers/docx_parser.py)

当前行为：

1. 识别 `Heading n` 并转成对应层级的 Markdown 标题
2. 列表样式转成 `- item`
3. 基础表格转成 `## Table n` 和 `字段: 值`

当前最适合：

- SOP
- 制度文件
- 方案文档
- 流程文档

当前最弱的点：

- 复杂 Word 表格和嵌套列表语义还不够强
- 页眉页脚、批注、修订痕迹还没专门处理

### 3. PPTX

对应文件：

- [core/ingestion/parsers/pptx_parser.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/ingestion/parsers/pptx_parser.py)

当前行为：

1. 每个 slide 都输出 `# Slide n: 标题`
2. 正文按 shape 提取
3. bullet paragraph 会保留 level，输出成缩进列表

当前最适合：

- 培训课件
- 方案评审
- 汇报材料

当前最弱的点：

- 图表、图片里的信息仍然看不到
- 同一 slide 内多个 shape 的逻辑关系还比较弱

### 4. CSV

对应文件：

- [core/ingestion/parsers/csv_parser.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/ingestion/parsers/csv_parser.py)

当前行为：

1. 首行作为 header
2. 每一行输出成一个小节
3. 每列转成 `字段名: 值`
4. 自动从 `code / name / title / 编号 / 名称` 里挑主键标题

当前最适合：

- 错误码对照表
- FAQ 导出
- 配置项清单
- 名录类结构化数据

当前最弱的点：

- 多表头、多层列名不够友好
- 某些表的“主键列”可能需要业务侧显式指定

### 5. Markdown

对应文件：

- [core/ingestion/parsers/markdown_parser.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/ingestion/parsers/markdown_parser.py)

当前行为：

1. 基本保留原始 Markdown 标题结构
2. 交给统一 chunker 做 section -> parent -> child

当前最适合：

- 技术文档
- FAQ
- 运维手册

它通常是当前项目里最“天然适配”这套 chunker 的格式。

### 6. HTML

对应文件：

- [core/ingestion/parsers/html_parser.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/ingestion/parsers/html_parser.py)

当前行为：

1. 去掉 `script/style/noscript`
2. 把 `h1~h4` 转成 Markdown 风格标题
3. 把 `p/li` 收成正文文本

当前最适合：

- Wiki 页面
- 帮助中心
- 内部知识库网页

当前最弱的点：

- 网页模板噪声偶尔仍然会残留
- 页面主内容和边栏内容还没有强分离

### 7. TXT

对应文件：

- [core/ingestion/parsers/text_parser.py](/Users/zhangzhijin/study/黑马学习/rag/RAG-%20project/enterprise-rag-platform/core/ingestion/parsers/text_parser.py)

当前行为：

1. 轻量清洗
2. 直接进入统一 chunker
3. 使用更保守的文本 profile

当前最适合：

- 日志片段
- 草稿
- 故障记录

当前最弱的点：

- 缺少结构边界
- 更依赖 metadata 和正文本身质量

## 当前最推荐的调优顺序

如果后面你要继续把 ingestion 做强，我建议优先顺序是：

1. `PDF`
   原因：真实企业里最常见，也最容易脏
2. `DOCX`
   原因：制度和 SOP 很多来自 Word
3. `CSV`
   原因：错误码、FAQ、台账类数据价值高，且容易做出稳定收益
4. `PPTX`
   原因：结构增强已经有基础，再往前做收益会更明显
5. `HTML`
6. `TXT`

## 面试时怎么讲

最稳的说法不是：

> 我给每种文件都写了一套复杂 chunker。

而是：

> 当前 ingestion 采用“parser 先增强结构、统一 chunker 再按文件类型选择 profile”的方案。这样既能让 PDF、PPTX、CSV、TXT 的切块行为更贴各自文档形态，又不会把系统拆成很多难维护的专用 chunker。后续如果某一类文档 badcase 特别多，就优先补 parser 输出和该类 profile，而不是一上来推翻整套切块链路。
