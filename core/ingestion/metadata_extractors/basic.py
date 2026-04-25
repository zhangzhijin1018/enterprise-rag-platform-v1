"""基础元数据提取模块。

目标不是做“完美信息抽取”，而是优先把最影响企业检索和治理的字段补齐：
- 文档身份
- 组织归属
- 业务域
- 流程阶段
- 数据分级 / 权威级别
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from core.models.document import Document


class BasicMetadataExtractor:
    """基础 metadata 提取器。

    这个类的定位不是“高精度信息抽取器”，而是一个稳定、便宜、可维护的第一层增强器：

    - 先把最重要的企业治理字段补起来
    - 让 retrieval / ACL / citation 尽快有可用 metadata
    - 后续如果要接更复杂的抽取器，也可以在它之后逐步增强
    """

    # 这一组 regex 主要负责从文件名、标题、正文前部抽“稳定锚点”：
    # - 文号 / 版本 / 日期
    # - 集团 / 子公司 / 厂矿 / 部门
    # - 设备 / 系统 / 项目
    # 设计上偏启发式，是为了在不引入重模型的前提下，先把企业 metadata 做厚。
    _DOC_NUMBER_RE = re.compile(
        r"(?:文号|文件编号|制度编号|文档编号|编号)\s*[:：]?\s*([A-Za-z0-9\u4e00-\u9fff\-/()（）〔〕\[\]第号]{2,64})"
    )
    _GROUP_COMPANY_RE = re.compile(r"(新疆能源（集团）有限责任公司|新疆能源集团)")
    _SUBSIDIARY_RE = re.compile(
        r"([\u4e00-\u9fffA-Za-z0-9_-]{2,40}(分公司|子公司|煤矿|电厂|选煤厂|事业部|运营公司|管理公司))"
    )
    _PLANT_RE = re.compile(
        r"([\u4e00-\u9fffA-Za-z0-9_-]{2,40}(厂|矿|站|基地|园区|中心))"
    )
    _DEPARTMENT_RE = re.compile(
        r"([\u4e00-\u9fffA-Za-z0-9_-]{2,30}(车间|部门|班组|小组|中心|事业部|工段|科室|仓库|产线|号线|线))"
    )
    _SHIFT_RE = re.compile(r"(白班|夜班|早班|中班|晚班)")
    _LINE_RE = re.compile(r"(\d+号线|\d+线)")
    _ENV_RE = re.compile(
        r"(生产|测试|预发|本地|线上|线下|docker|k8s|kubernetes|macos|linux|windows)",
        re.IGNORECASE,
    )
    _VERSION_RE = re.compile(r"(\b\d+\.\d+(?:\.\d+)?\b|v\d+\.\d+(?:\.\d+)?)", re.IGNORECASE)
    _TIME_RE = re.compile(
        r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}月\d{1,2}[日号]?|今天|明天|本周|下周|本月)"
    )
    _DATE_RE = re.compile(r"\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2})\b")
    _EXPIRY_DATE_RE = re.compile(r"(?:失效日期|废止日期|截止日期)\s*[:：]?\s*(20\d{2}[-/]\d{1,2}[-/]\d{1,2})")
    _PERSON_RE = re.compile(
        r"(联系人|负责人|值班人|审批人|经理|主管|老师)\s*[:：]?\s*([\u4e00-\u9fff]{2,4})"
    )
    _ISSUED_BY_RE = re.compile(
        r"(?:发布部门|印发部门|起草部门|发布单位)\s*[:：]?\s*([\u4e00-\u9fffA-Za-z0-9_-]{2,40})"
    )
    _APPROVED_BY_RE = re.compile(
        r"(?:批准人|审批人|审核人|签发人)\s*[:：]?\s*([\u4e00-\u9fff]{2,8})"
    )
    _OWNER_ROLE_RE = re.compile(
        r"(?:责任角色|责任岗位|岗位)\s*[:：]?\s*([\u4e00-\u9fffA-Za-z0-9_-]{2,30})"
    )
    _EQUIPMENT_TYPE_RE = re.compile(
        r"([\u4e00-\u9fffA-Za-z0-9_-]{2,30}(锅炉|汽轮机|发电机|输煤皮带|磨煤机|风机|泵|压缩机|变压器|阀门|机组))"
    )
    _EQUIPMENT_ID_RE = re.compile(
        r"(?:设备编号|设备编码|设备ID|机组编号|设备号)\s*[:：]?\s*([A-Za-z0-9_-]{2,40})"
    )
    _SYSTEM_NAME_RE = re.compile(
        r"(?:系统名称|平台名称|系统平台)\s*[:：]?\s*([A-Za-z0-9_\-/\u4e00-\u9fff]{2,60})"
    )
    _PROJECT_NAME_RE = re.compile(
        r"(?:项目名称|工程名称|课题名称)\s*[:：]?\s*([A-Za-z0-9_\-/\u4e00-\u9fff（）()《》]{2,80})"
    )
    _APPLICABLE_REGION_RE = re.compile(
        r"(?:适用区域|适用范围|适用地区)\s*[:：]?\s*([A-Za-z0-9_\-/\u4e00-\u9fff、，,]{2,80})"
    )
    _APPLICABLE_SITE_RE = re.compile(
        r"(?:适用场站|适用厂站|适用地点|适用现场)\s*[:：]?\s*([A-Za-z0-9_\-/\u4e00-\u9fff、，,]{2,80})"
    )
    _LEADING_TIME_PREFIX_RE = re.compile(r"^(今天|明天|后天|昨天|本周|下周|本月|上月|本季度|今年|去年)")
    _DOC_TYPE_BY_SUFFIX = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".pptx": "pptx",
        ".md": "markdown",
        ".markdown": "markdown",
        ".html": "html",
        ".htm": "html",
        ".txt": "text",
        ".csv": "csv",
    }
    _CLASSIFICATION_RULES = (
        (re.compile(r"(绝密|restricted|机密)", re.IGNORECASE), "restricted"),
        (re.compile(r"(秘密|敏感|sensitive)", re.IGNORECASE), "sensitive"),
        (re.compile(r"(公开|public)", re.IGNORECASE), "public"),
        (re.compile(r"(内部|internal)", re.IGNORECASE), "internal"),
    )
    _AUTHORITY_RULES = (
        (re.compile(r"(制度|规范|标准|条例|办法|规定)"), "high"),
        (re.compile(r"(SOP|流程|作业指导书|操作手册|应急预案)", re.IGNORECASE), "medium"),
        (re.compile(r"(会议纪要|讨论记录|复盘|草稿|汇报)"), "low"),
    )
    _VERSION_STATUS_RULES = (
        (re.compile(r"(已废止|已作废|已失效|废止执行)"), "obsolete"),
        (re.compile(r"(试行|试运行|暂行)"), "trial"),
        (re.compile(r"(现行|有效|生效|执行中)"), "active"),
    )
    _STATUS_RULES = (
        (re.compile(r"(已废止|已作废|已失效|废止执行)"), "inactive"),
        (re.compile(r"(草稿|征求意见稿|讨论稿)"), "draft"),
        (re.compile(r"(试行|试运行|暂行)"), "trial"),
        (re.compile(r"(现行|有效|生效|执行中)"), "active"),
    )
    _BUSINESS_DOMAIN_RULES = (
        (re.compile(r"(安全生产|安全检查|隐患排查|事故|应急预案)"), "safety_production"),
        (re.compile(r"(巡检|检修|保养|点检|维修|设备)"), "equipment_maintenance"),
        (re.compile(r"(调度|值班|排班|交接班)"), "dispatch"),
        (re.compile(r"(采购|招采|供应商|合同)"), "procurement"),
        (re.compile(r"(报销|预算|财务|付款|结算)"), "finance"),
        (re.compile(r"(人员|薪酬|考勤|培训|人力|招聘)"), "hr"),
        (re.compile(r"(项目|里程碑|方案变更|会议纪要|立项)"), "project_management"),
        (re.compile(r"(接口|错误码|数据库|平台|系统|发布)"), "it_ops"),
    )
    _PROCESS_STAGE_RULES = (
        (re.compile(r"(巡检|点检|巡查)"), "inspection"),
        (re.compile(r"(检修|维修|保养)"), "maintenance"),
        (re.compile(r"(应急|故障处理|事故处置)"), "emergency"),
        (re.compile(r"(审批|报批|签批)"), "approval"),
        (re.compile(r"(报销|付款|结算)"), "reimbursement"),
        (re.compile(r"(招标|采购|比价|询价)"), "procurement"),
        (re.compile(r"(调度|值班|排班|交接班)"), "dispatch"),
    )
    _PROJECT_PHASE_RULES = (
        (re.compile(r"(立项|可研)"), "initiation"),
        (re.compile(r"(设计|方案)"), "design"),
        (re.compile(r"(实施|建设|上线|部署)"), "implementation"),
        (re.compile(r"(验收|试运行|试运)"), "acceptance"),
        (re.compile(r"(复盘|总结|收尾)"), "closure"),
    )

    def ensure_doc_id(self, doc: Document) -> Document:
        """确保文档有稳定可追踪的 doc_id。

        parser 阶段可能只关心把内容读出来，不一定生成正式主键；
        这里补齐后，后续 chunk 才能稳定引用同一个文档来源。
        """

        if doc.doc_id:
            return doc
        return doc.model_copy(update={"doc_id": str(uuid.uuid4())})

    def infer_title_from_filename(self, path: str | Path, doc: Document) -> Document:
        """在缺标题时，用文件名兜底生成标题。

        这样做主要是为了：
        - 避免后续引用、检索调试、审计日志里出现空标题
        - 给没有内嵌标题的原始文件一个稳定可读的展示名
        """

        if doc.title:
            return doc
        name = Path(path).stem
        return doc.model_copy(update={"title": name})

    def enrich_retrieval_metadata(self, path: str | Path, doc: Document) -> Document:
        """从文档标题、文件名和正文中抽取适合检索过滤的轻量 metadata。

        当前策略偏“稳定优先”：
        - 先从标题、文件名和正文前 4000 字抽取
        - 尽量补企业检索最常用字段
        - 已存在的 metadata 不会被随意覆盖

        这里的关键词是“轻量、够用、低成本”：
        - 不追求一次就把所有实体都抽准
        - 优先补最影响过滤、排序、路由、引用的字段
        - 已有值优先信任上游，避免把人工标注好的 metadata 冲掉
        """

        path_obj = Path(path)
        # 这里故意不直接吃全文，而是优先用“标题 + 文件名 + 前部正文”，
        # 因为很多制度、SOP、纪要的关键治理信息通常集中在前面。
        merged_text = "\n".join(
            [
                doc.title or "",
                path_obj.stem,
                doc.content[:4000],
            ]
        )
        metadata = dict(doc.metadata)

        doc_number = self._first_group(self._DOC_NUMBER_RE, merged_text)
        group_company = self._first_group(self._GROUP_COMPANY_RE, merged_text)
        subsidiary = self._first_group(self._SUBSIDIARY_RE, merged_text)
        plant = self._first_group(self._PLANT_RE, merged_text)
        department = self._first_group(self._DEPARTMENT_RE, merged_text)
        shift = self._first_group(self._SHIFT_RE, merged_text)
        line = self._first_group(self._LINE_RE, merged_text)
        environment = self._first_group(self._ENV_RE, merged_text)
        version = self._normalize_version(self._first_group(self._VERSION_RE, merged_text))
        time_value = self._first_group(self._TIME_RE, merged_text)
        effective_date = self._first_group(self._DATE_RE, merged_text)
        expiry_date = self._first_group(self._EXPIRY_DATE_RE, merged_text)
        person = self._extract_person(merged_text)
        issued_by = self._first_group(self._ISSUED_BY_RE, merged_text)
        approved_by = self._first_group(self._APPROVED_BY_RE, merged_text)
        owner_role = self._first_group(self._OWNER_ROLE_RE, merged_text)
        equipment_type = self._first_group(self._EQUIPMENT_TYPE_RE, merged_text)
        equipment_id = self._first_group(self._EQUIPMENT_ID_RE, merged_text)
        system_name = self._first_group(self._SYSTEM_NAME_RE, merged_text)
        project_name = self._first_group(self._PROJECT_NAME_RE, merged_text)
        applicable_region = self._first_group(self._APPLICABLE_REGION_RE, merged_text)
        applicable_site = self._first_group(self._APPLICABLE_SITE_RE, merged_text)
        doc_type = self._DOC_TYPE_BY_SUFFIX.get(path_obj.suffix.lower(), "unknown")
        data_classification = self._infer_data_classification(merged_text, metadata)
        authority_level = self._infer_authority_level(merged_text, metadata)
        version_status = self._infer_version_status(merged_text, metadata)
        status = self._infer_status(merged_text, metadata)
        business_domain = self._infer_business_domain(merged_text, metadata)
        process_stage = self._infer_process_stage(merged_text, metadata)
        project_phase = self._infer_project_phase(merged_text, metadata)

        if doc_number and not metadata.get("doc_number"):
            metadata["doc_number"] = doc_number
        if group_company and not metadata.get("group_company"):
            metadata["group_company"] = group_company
        if subsidiary and not metadata.get("subsidiary"):
            metadata["subsidiary"] = subsidiary
        if plant and not metadata.get("plant"):
            metadata["plant"] = plant
        if department and not metadata.get("department"):
            metadata["department"] = department
        if department and not metadata.get("owner_department"):
            metadata["owner_department"] = department
        if shift and not metadata.get("shift"):
            metadata["shift"] = shift
        if line and not metadata.get("line"):
            metadata["line"] = line
        if environment and not metadata.get("environment"):
            metadata["environment"] = environment
        if version and not metadata.get("version"):
            metadata["version"] = version
        if time_value and not metadata.get("time"):
            metadata["time"] = time_value
        if person and not metadata.get("person"):
            metadata["person"] = person
        if effective_date and not metadata.get("effective_date"):
            metadata["effective_date"] = effective_date
        if expiry_date and not metadata.get("expiry_date"):
            metadata["expiry_date"] = expiry_date
        if not metadata.get("doc_type"):
            metadata["doc_type"] = doc_type
        if not metadata.get("data_classification"):
            metadata["data_classification"] = data_classification
        if not metadata.get("authority_level"):
            metadata["authority_level"] = authority_level
        if not metadata.get("version_status"):
            metadata["version_status"] = version_status
        if not metadata.get("status"):
            metadata["status"] = status
        if not metadata.get("source_system"):
            metadata["source_system"] = "local_file"
        if issued_by and not metadata.get("issued_by"):
            metadata["issued_by"] = issued_by
        if approved_by and not metadata.get("approved_by"):
            metadata["approved_by"] = approved_by
        if owner_role and not metadata.get("owner_role"):
            metadata["owner_role"] = owner_role
        if business_domain and not metadata.get("business_domain"):
            metadata["business_domain"] = business_domain
        if process_stage and not metadata.get("process_stage"):
            metadata["process_stage"] = process_stage
        if applicable_region and not metadata.get("applicable_region"):
            metadata["applicable_region"] = applicable_region
        if applicable_site and not metadata.get("applicable_site"):
            metadata["applicable_site"] = applicable_site
        if equipment_type and not metadata.get("equipment_type"):
            metadata["equipment_type"] = equipment_type
        if equipment_id and not metadata.get("equipment_id"):
            metadata["equipment_id"] = equipment_id
        if system_name and not metadata.get("system_name"):
            metadata["system_name"] = system_name
        if project_name and not metadata.get("project_name"):
            metadata["project_name"] = project_name
        if project_phase and not metadata.get("project_phase"):
            metadata["project_phase"] = project_phase

        if not metadata.get("doc_category"):
            # doc_category 更偏“检索视角下的粗分类”，
            # 便于 query understanding / retrieval 提前做 route 和 boost。
            if re.search(r"(值班|排班|安排表)", merged_text):
                metadata["doc_category"] = "schedule"
            elif re.search(r"(制度|规范|要求|办法)", merged_text):
                metadata["doc_category"] = "policy"
            elif re.search(r"(流程|步骤|SOP|作业指导书)", merged_text):
                metadata["doc_category"] = "procedure"
            elif re.search(r"(会议纪要|复盘|记录)", merged_text):
                metadata["doc_category"] = "meeting"

        return doc.model_copy(update={"metadata": metadata})

    def _infer_data_classification(self, text: str, metadata: dict[str, object]) -> str:
        """根据已有 metadata、标题和正文推断数据分级。"""

        existing = metadata.get("data_classification")
        if existing:
            return str(existing).strip().lower()
        for pattern, level in self._CLASSIFICATION_RULES:
            if pattern.search(text):
                return level
        return "internal"

    def _infer_authority_level(self, text: str, metadata: dict[str, object]) -> str:
        """根据文档类型和标题关键词推断权威级别。"""

        existing = metadata.get("authority_level")
        if existing:
            return str(existing).strip().lower()
        for pattern, level in self._AUTHORITY_RULES:
            if pattern.search(text):
                return level
        return "medium"

    def _infer_version_status(self, text: str, metadata: dict[str, object]) -> str:
        """推断版本状态。"""

        existing = metadata.get("version_status")
        if existing:
            return str(existing).strip().lower()
        for pattern, status in self._VERSION_STATUS_RULES:
            if pattern.search(text):
                return status
        return "active"

    def _infer_status(self, text: str, metadata: dict[str, object]) -> str:
        """推断文档当前状态。"""

        existing = metadata.get("status")
        if existing:
            return str(existing).strip().lower()
        for pattern, status in self._STATUS_RULES:
            if pattern.search(text):
                return status
        return "active"

    def _infer_business_domain(self, text: str, metadata: dict[str, object]) -> str:
        """推断业务域。

        business_domain 是后续 retrieval boost、explainability 和评测里都很重要的字段。
        """

        existing = metadata.get("business_domain")
        if existing:
            return str(existing).strip().lower()
        for pattern, value in self._BUSINESS_DOMAIN_RULES:
            if pattern.search(text):
                return value
        return "general_ops"

    def _infer_process_stage(self, text: str, metadata: dict[str, object]) -> str:
        """推断流程阶段。"""

        existing = metadata.get("process_stage")
        if existing:
            return str(existing).strip().lower()
        for pattern, value in self._PROCESS_STAGE_RULES:
            if pattern.search(text):
                return value
        return "general"

    def _infer_project_phase(self, text: str, metadata: dict[str, object]) -> str | None:
        """推断项目阶段。"""

        existing = metadata.get("project_phase")
        if existing:
            return str(existing).strip().lower()
        for pattern, value in self._PROJECT_PHASE_RULES:
            if pattern.search(text):
                return value
        return None

    @classmethod
    def _extract_person(cls, text: str) -> str | None:
        """按业务优先级抽取核心人员字段。"""

        priorities = {
            "值班人": 5,
            "联系人": 4,
            "负责人": 3,
            "审批人": 2,
            "经理": 1,
            "主管": 1,
            "老师": 1,
        }
        best_score = -1
        best_person: str | None = None
        for match in cls._PERSON_RE.finditer(text):
            label = match.group(1).strip()
            person = match.group(2).strip()
            score = priorities.get(label, 0)
            if score > best_score:
                best_score = score
                best_person = person
        return best_person

    @staticmethod
    def _normalize_version(value: str | None) -> str | None:
        if not value:
            return None
        normalized = value.strip()
        normalized = re.sub(r"^(version|VERSION)\s*", "", normalized)
        normalized = re.sub(r"^[vV]\s*", "", normalized)
        return normalized or None

    @staticmethod
    def _first_group(pattern: re.Pattern[str], text: str) -> str | None:
        match = pattern.search(text)
        if not match:
            return None
        value = match.group(1) if match.groups() else match.group(0)
        value = BasicMetadataExtractor._LEADING_TIME_PREFIX_RE.sub("", value).strip()
        return value or None

    @staticmethod
    def _second_group(pattern: re.Pattern[str], text: str) -> str | None:
        match = pattern.search(text)
        if not match or len(match.groups()) < 2:
            return None
        value = match.group(2).strip()
        return value or None
