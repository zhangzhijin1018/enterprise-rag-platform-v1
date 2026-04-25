/*
  前端演示控制台主页面。

  这个组件承担三个职责：
  1. 提供问答、入库、FAQ 管理、评测、连接配置五个操作页签。
  2. 管理 API 调用与流式响应状态。
  3. 展示答案、引用、检索片段和后台任务状态。
*/

import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  Activity,
  BookOpen,
  ChevronRight,
  Database,
  FlaskConical,
  Loader2,
  MessageSquare,
  RefreshCw,
  Send,
  Settings2,
  Sparkles,
  Upload,
  Zap,
} from "lucide-react";
import { MarkdownView } from "./MarkdownView";

type Tab = "chat" | "ingest" | "faq" | "eval" | "settings";

type Citation = {
  doc_id: string;
  chunk_id: string;
  title: string;
  source: string;
  page: number | null;
  section: string | null;
  doc_type?: string | null;
  owner_department?: string | null;
  data_classification?: string | null;
  version?: string | null;
  effective_date?: string | null;
  authority_level?: string | null;
  source_system?: string | null;
  business_domain?: string | null;
  process_stage?: string | null;
  section_path?: string | null;
  matched_routes?: string[];
  retrieval_score?: number | null;
  semantic_score?: number | null;
  governance_rank_score?: number | null;
  selection_reason?: string | null;
};

type RetrievedChunk = {
  chunk_id: string;
  score: number;
  content: string;
  metadata: Record<string, unknown>;
};

type FaqImportResponse = {
  imported: number;
  status: string;
};

type FaqItem = {
  id: number;
  question: string;
  answer: string;
  keywords: string;
  category: string;
  enabled: boolean;
};

const LS_API = "erp_api_base";

function apiPath(path: string, base: string): string {
  // 没填自定义地址时，直接走 Vite 的同源代理。
  if (!base) return path;
  // 去掉尾部 `/`，避免和下方 path 拼接成双斜杠。
  const b = base.replace(/\/$/, "");
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${b}${p}`;
}

export default function App() {
  const envBase = import.meta.env.VITE_API_BASE ?? "";
  // 首次加载优先读 localStorage，这样刷新页面后还能记住上次的 API 地址。
  const [apiBase, setApiBase] = useState(() => {
    if (typeof window === "undefined") return envBase;
    return window.localStorage.getItem(LS_API) ?? envBase;
  });

  // 顶层页面状态。
  const [tab, setTab] = useState<Tab>("chat");
  const [health, setHealth] = useState<"ok" | "down" | "checking">("checking");

  // 问答页状态。
  const [question, setQuestion] = useState("错误码 E-1001 是什么？应如何处理？");
  const [topK, setTopK] = useState(8);
  const [stream, setStream] = useState(false);
  const [busy, setBusy] = useState(false);
  const [answer, setAnswer] = useState("");
  /** 流式过程中用纯文本，避免半段 Markdown 闪烁；结束后改为 false 再渲染 MD */
  const [answerStreamPlain, setAnswerStreamPlain] = useState(false);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [chunks, setChunks] = useState<RetrievedChunk[]>([]);
  const [fastPathSource, setFastPathSource] = useState<string | null>(null);
  const [refusal, setRefusal] = useState(false);
  const [refusalReason, setRefusalReason] = useState<string | null>(null);
  const [answerMode, setAnswerMode] = useState<string | null>(null);
  const [dataClassification, setDataClassification] = useState<string | null>(null);
  const [modelRoute, setModelRoute] = useState<string | null>(null);
  const [analysisConfidence, setAnalysisConfidence] = useState<number | null>(null);
  const [analysisSource, setAnalysisSource] = useState<string | null>(null);
  const [analysisReason, setAnalysisReason] = useState<string | null>(null);
  const [conflictDetected, setConflictDetected] = useState(false);
  const [conflictSummary, setConflictSummary] = useState<string | null>(null);
  const [traceId, setTraceId] = useState<string | null>(null);
  const [auditId, setAuditId] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  // 入库任务状态。
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [jobDetail, setJobDetail] = useState<string | null>(null);

  // FAQ 导入状态。
  const [faqFile, setFaqFile] = useState<File | null>(null);
  const [faqImported, setFaqImported] = useState<number | null>(null);
  const [faqImportStatus, setFaqImportStatus] = useState<string | null>(null);
  const [faqItems, setFaqItems] = useState<FaqItem[]>([]);
  const [faqListLoading, setFaqListLoading] = useState(false);

  // 离线评测状态。
  const [evalBusy, setEvalBusy] = useState(false);
  const [evalOut, setEvalOut] = useState<string | null>(null);
  const [evalAnalysisOut, setEvalAnalysisOut] = useState<string | null>(null);
  const [evalSummary, setEvalSummary] = useState<Record<string, number> | null>(null);

  // 页面右上角显示一个更友好的 API 来源文本。
  const baseLabel = useMemo(
    () => (apiBase.trim() ? apiBase : "同源代理 · 127.0.0.1:8000"),
    [apiBase],
  );

  const ping = useCallback(async () => {
    // 健康检查结果会映射到右上角状态灯。
    setHealth("checking");
    try {
      const r = await fetch(apiPath("/healthz", apiBase));
      if (!r.ok) throw new Error(String(r.status));
      const j = await r.json();
      setHealth(j.status === "ok" ? "ok" : "down");
    } catch {
      setHealth("down");
    }
  }, [apiBase]);

  useEffect(() => {
    void ping();
  }, [ping]);

  useEffect(() => {
    // API 地址变化后立刻持久化，方便本地调试。
    window.localStorage.setItem(LS_API, apiBase);
  }, [apiBase]);

  useEffect(() => {
    // 只有切到 FAQ 页时才拉列表，避免平时多发一个管理接口请求。
    if (tab === "faq") {
      void loadFaqItems();
    }
  }, [tab, apiBase]);

  async function runChat() {
    // 发新请求前先清空旧结果，避免页面混入上次回答。
    setErr(null);
    setBusy(true);
    setAnswer("");
    setAnswerStreamPlain(false);
    setCitations([]);
    setChunks([]);
    setFastPathSource(null);
    setConfidence(null);
    setRefusal(false);
    setRefusalReason(null);
    setAnswerMode(null);
    setDataClassification(null);
    setModelRoute(null);
    setAnalysisConfidence(null);
    setAnalysisSource(null);
    setAnalysisReason(null);
    setConflictDetected(false);
    setConflictSummary(null);
    setTraceId(null);
    setAuditId(null);
    const useStream = stream;
    try {
      if (useStream) {
        setAnswerStreamPlain(true);
        // 流式模式接收 NDJSON 事件流，而不是一次性 JSON。
        const res = await fetch(apiPath("/chat", apiBase), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: question.trim(),
            top_k: topK,
            stream: true,
          }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        setTraceId(res.headers.get("X-Trace-ID"));
        const reader = res.body?.getReader();
        if (!reader) throw new Error("无响应流");
        const dec = new TextDecoder();
        let buf = "";
        let acc = "";
        // 自己按行切开 NDJSON，再逐条解析事件。
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const lines = buf.split("\n");
          buf = lines.pop() ?? "";
          for (const line of lines) {
            const t = line.trim();
            if (!t) continue;
            const evt = JSON.parse(t) as {
              type: string;
              data: unknown;
            };
            if (evt.type === "token" && typeof evt.data === "string") {
              // token 事件负责渐进拼接答案正文。
              acc += evt.data;
              setAnswer(acc);
            }
            if (evt.type === "final" && evt.data && typeof evt.data === "object") {
              const d = evt.data as {
                answer?: string;
                confidence?: number;
                citations?: Citation[];
                fast_path_source?: string | null;
                refusal?: boolean;
                refusal_reason?: string | null;
                answer_mode?: string | null;
                data_classification?: string | null;
                model_route?: string | null;
                analysis_confidence?: number | null;
                analysis_source?: string | null;
                analysis_reason?: string | null;
                conflict_detected?: boolean;
                conflict_summary?: string | null;
                trace_id?: string | null;
                audit_id?: string | null;
              };
              if (d.answer) setAnswer(d.answer);
              if (typeof d.confidence === "number") setConfidence(d.confidence);
              if (Array.isArray(d.citations)) setCitations(d.citations);
              if (typeof d.fast_path_source === "string") {
                setFastPathSource(d.fast_path_source);
              }
              if (typeof d.refusal === "boolean") setRefusal(d.refusal);
              if (typeof d.refusal_reason === "string" || d.refusal_reason === null) {
                setRefusalReason(d.refusal_reason ?? null);
              }
              if (typeof d.answer_mode === "string" || d.answer_mode === null) {
                setAnswerMode(d.answer_mode ?? null);
              }
              if (
                typeof d.data_classification === "string" ||
                d.data_classification === null
              ) {
                setDataClassification(d.data_classification ?? null);
              }
              if (typeof d.model_route === "string" || d.model_route === null) {
                setModelRoute(d.model_route ?? null);
              }
              if (typeof d.analysis_confidence === "number") {
                setAnalysisConfidence(d.analysis_confidence);
              }
              if (typeof d.analysis_source === "string" || d.analysis_source === null) {
                setAnalysisSource(d.analysis_source ?? null);
              }
              if (typeof d.analysis_reason === "string" || d.analysis_reason === null) {
                setAnalysisReason(d.analysis_reason ?? null);
              }
              if (typeof d.conflict_detected === "boolean") {
                setConflictDetected(d.conflict_detected);
              }
              if (typeof d.conflict_summary === "string" || d.conflict_summary === null) {
                setConflictSummary(d.conflict_summary ?? null);
              }
              if (typeof d.trace_id === "string" || d.trace_id === null) {
                setTraceId(d.trace_id ?? null);
              }
              if (typeof d.audit_id === "string" || d.audit_id === null) {
                setAuditId(d.audit_id ?? null);
              }
            }
            if (evt.type === "meta" && evt.data && typeof evt.data === "object") {
              // meta 事件通常先于最终答案到达，适合提前渲染检索信息。
              const d = evt.data as {
                retrieved_chunks?: RetrievedChunk[];
                confidence?: number;
                citations?: Citation[];
                fast_path_source?: string | null;
                refusal?: boolean;
                refusal_reason?: string | null;
                answer_mode?: string | null;
                data_classification?: string | null;
                model_route?: string | null;
                analysis_confidence?: number | null;
                analysis_source?: string | null;
                analysis_reason?: string | null;
                conflict_detected?: boolean;
                conflict_summary?: string | null;
                trace_id?: string | null;
                audit_id?: string | null;
              };
              if (Array.isArray(d.retrieved_chunks)) setChunks(d.retrieved_chunks);
              if (typeof d.confidence === "number") setConfidence(d.confidence);
              if (Array.isArray(d.citations)) setCitations(d.citations);
              if (typeof d.fast_path_source === "string") {
                setFastPathSource(d.fast_path_source);
              }
              if (typeof d.refusal === "boolean") setRefusal(d.refusal);
              if (typeof d.refusal_reason === "string" || d.refusal_reason === null) {
                setRefusalReason(d.refusal_reason ?? null);
              }
              if (typeof d.answer_mode === "string" || d.answer_mode === null) {
                setAnswerMode(d.answer_mode ?? null);
              }
              if (
                typeof d.data_classification === "string" ||
                d.data_classification === null
              ) {
                setDataClassification(d.data_classification ?? null);
              }
              if (typeof d.model_route === "string" || d.model_route === null) {
                setModelRoute(d.model_route ?? null);
              }
              if (typeof d.analysis_confidence === "number") {
                setAnalysisConfidence(d.analysis_confidence);
              }
              if (typeof d.analysis_source === "string" || d.analysis_source === null) {
                setAnalysisSource(d.analysis_source ?? null);
              }
              if (typeof d.analysis_reason === "string" || d.analysis_reason === null) {
                setAnalysisReason(d.analysis_reason ?? null);
              }
              if (typeof d.conflict_detected === "boolean") {
                setConflictDetected(d.conflict_detected);
              }
              if (typeof d.conflict_summary === "string" || d.conflict_summary === null) {
                setConflictSummary(d.conflict_summary ?? null);
              }
              if (typeof d.trace_id === "string" || d.trace_id === null) {
                setTraceId(d.trace_id ?? null);
              }
              if (typeof d.audit_id === "string" || d.audit_id === null) {
                setAuditId(d.audit_id ?? null);
              }
            }
          }
        }
      } else {
        const res = await fetch(apiPath("/chat", apiBase), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: question.trim(),
            top_k: topK,
            stream: false,
          }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        setTraceId(res.headers.get("X-Trace-ID"));
        const j = (await res.json()) as {
          answer: string;
          confidence: number;
          fast_path_source?: string | null;
          citations: Citation[];
          retrieved_chunks: RetrievedChunk[];
          refusal?: boolean;
          refusal_reason?: string | null;
          answer_mode?: string | null;
          data_classification?: string | null;
          model_route?: string | null;
          analysis_confidence?: number | null;
          analysis_source?: string | null;
          analysis_reason?: string | null;
          conflict_detected?: boolean;
          conflict_summary?: string | null;
          trace_id?: string | null;
          audit_id?: string | null;
        };
        setAnswer(j.answer);
        setConfidence(j.confidence);
        setFastPathSource(j.fast_path_source ?? null);
        setCitations(j.citations ?? []);
        setChunks(j.retrieved_chunks ?? []);
        setRefusal(Boolean(j.refusal));
        setRefusalReason(j.refusal_reason ?? null);
        setAnswerMode(j.answer_mode ?? null);
        setDataClassification(j.data_classification ?? null);
        setModelRoute(j.model_route ?? null);
        setAnalysisConfidence(
          typeof j.analysis_confidence === "number" ? j.analysis_confidence : null,
        );
        setAnalysisSource(j.analysis_source ?? null);
        setAnalysisReason(j.analysis_reason ?? null);
        setConflictDetected(Boolean(j.conflict_detected));
        setConflictSummary(j.conflict_summary ?? null);
        setTraceId(j.trace_id ?? res.headers.get("X-Trace-ID"));
        setAuditId(j.audit_id ?? null);
      }
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      if (useStream) {
        // 流式结束后再切回 Markdown 渲染，避免半截 Markdown 抖动。
        setAnswerStreamPlain(false);
      }
    }
  }

  async function runFaqImport() {
    // FAQ 导入是“结构化问答”的单独入口，不走普通文档入库。
    setErr(null);
    setFaqImported(null);
    setFaqImportStatus(null);
    if (!faqFile) {
      setErr("请选择 FAQ CSV 文件");
      return;
    }
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("file", faqFile);
      const res = await fetch(apiPath("/faq/import", apiBase), {
        method: "POST",
        body: fd,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const j = (await res.json()) as FaqImportResponse;
      setFaqImported(j.imported);
      setFaqImportStatus(j.status);
      await loadFaqItems();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function loadFaqItems() {
    // FAQ 管理页读取的是全量列表，包含启用和停用状态。
    setErr(null);
    setFaqListLoading(true);
    try {
      const res = await fetch(apiPath("/faq", apiBase));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const j = (await res.json()) as { items: FaqItem[] };
      setFaqItems(j.items ?? []);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setFaqListLoading(false);
    }
  }

  async function toggleFaqItem(item: FaqItem) {
    // FAQ 管理先支持启用/停用，避免一开始就把在线编辑做得过重。
    setErr(null);
    try {
      const res = await fetch(apiPath(`/faq/${item.id}`, apiBase), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !item.enabled }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await loadFaqItems();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  async function runIngest() {
    // 入库前先做最小校验。
    setErr(null);
    setJobStatus(null);
    setJobDetail(null);
    if (!file) {
      setErr("请选择文件");
      return;
    }
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      // multipart/form-data 由浏览器自动补齐边界，不要手写 Content-Type。
      const res = await fetch(apiPath("/ingest", apiBase), {
        method: "POST",
        body: fd,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const j = (await res.json()) as { job_id: string; status: string };
      setJobId(j.job_id);
      setJobStatus(j.status);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function pollJob() {
    // 手动轮询后台任务状态，方便观察异步作业执行情况。
    if (!jobId) return;
    setErr(null);
    try {
      const res = await fetch(apiPath(`/jobs/${jobId}`, apiBase));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const j = (await res.json()) as {
        status: string;
        detail?: string | null;
      };
      setJobStatus(j.status);
      setJobDetail(j.detail ?? null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  async function runReindex() {
    // 重建索引用于根据已有 chunks 重新生成 embedding。
    setErr(null);
    setBusy(true);
    try {
      const res = await fetch(apiPath("/reindex", apiBase), { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const j = (await res.json()) as { job_id: string; status: string };
      setJobId(j.job_id);
      setJobStatus(j.status);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function runEval() {
    // 评测可能耗时较长，所以单独维护一套 loading / 结果状态。
    setErr(null);
    setEvalOut(null);
    setEvalAnalysisOut(null);
    setEvalSummary(null);
    setEvalBusy(true);
    try {
      const res = await fetch(apiPath("/eval", apiBase), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const j = (await res.json()) as {
        report_path: string;
        analysis_path?: string | null;
        summary?: Record<string, number> | null;
      };
      setEvalOut(j.report_path);
      setEvalAnalysisOut(j.analysis_path ?? null);
      setEvalSummary(j.summary ?? null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setEvalBusy(false);
    }
  }

  // 导航项单独抽成数据，渲染更直观。
  const tabs: { id: Tab; label: string; icon: ReactNode }[] = [
    { id: "chat", label: "智能问答", icon: <MessageSquare className="h-4 w-4" /> },
    { id: "ingest", label: "知识接入", icon: <Upload className="h-4 w-4" /> },
    { id: "faq", label: "FAQ 导入", icon: <Database className="h-4 w-4" /> },
    { id: "eval", label: "离线评测", icon: <FlaskConical className="h-4 w-4" /> },
    { id: "settings", label: "连接", icon: <Settings2 className="h-4 w-4" /> },
  ];

  return (
    <div className="min-h-full">
      {/* 顶部栏：项目名、健康状态、当前 API 地址。 */}
      <header className="border-b border-white/[0.06] bg-ink-950/80 backdrop-blur-md sticky top-0 z-20">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-5 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent-glow ring-1 ring-sky-400/25">
              <Sparkles className="h-5 w-5 text-sky-300" />
            </div>
            <div>
              <h1 className="text-sm font-semibold tracking-tight text-white">
                Enterprise RAG
              </h1>
              <p className="text-xs text-zinc-500">企业知识库 · 编排与可观测控制台</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => void ping()}
              className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs text-zinc-300 transition hover:bg-white/[0.07]"
            >
              <Activity
                className={`h-3.5 w-3.5 ${
                  health === "ok"
                    ? "text-emerald-400"
                    : health === "down"
                      ? "text-rose-400"
                      : "text-amber-300 animate-pulse"
                }`}
              />
              {health === "ok"
                ? "API 正常"
                : health === "down"
                  ? "API 不可用"
                  : "检测中"}
            </button>
            <span className="hidden sm:block max-w-[220px] truncate text-[11px] text-zinc-600 font-mono">
              {baseLabel}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-5 py-8">
        {/* 页签导航。 */}
        <nav className="mb-8 flex flex-wrap gap-2">
          {tabs.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm transition ${
                tab === t.id
                  ? "bg-white/[0.1] text-white ring-1 ring-sky-400/30 shadow-soft"
                  : "text-zinc-400 hover:bg-white/[0.05] hover:text-zinc-200"
              }`}
            >
              {t.icon}
              {t.label}
            </button>
          ))}
        </nav>

        {err && (
          <div className="mb-6 rounded-2xl border border-rose-500/25 bg-rose-500/[0.08] px-4 py-3 text-sm text-rose-100">
            {err}
          </div>
        )}

        {tab === "chat" && (
          <>
            {/* 问答页分左右两栏：左边交互，右边观察引用和检索片段。 */}
          <div className="grid gap-6 lg:grid-cols-[1fr_380px]">
            <section className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-6 shadow-panel backdrop-blur-xl">
              <div className="mb-4 flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-zinc-500">
                <Zap className="h-3.5 w-3.5 text-sky-400" />
                检索增强生成
              </div>
              <label className="mb-2 block text-sm text-zinc-400">问题</label>
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                rows={5}
                className="mb-4 w-full resize-y rounded-xl border border-white/10 bg-ink-900/80 px-4 py-3 text-sm text-zinc-100 outline-none ring-0 placeholder:text-zinc-600 focus:border-sky-500/40 focus:ring-2 focus:ring-sky-500/20"
                placeholder="输入自然语言问题…"
              />
              <div className="mb-6 flex flex-wrap items-center gap-6">
                <label className="flex items-center gap-3 text-sm text-zinc-400">
                  <span className="w-24 shrink-0">Top-K</span>
                  <input
                    type="range"
                    min={1}
                    max={24}
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    className="h-1 w-40 accent-sky-400"
                  />
                  <span className="w-6 font-mono text-xs text-zinc-300">{topK}</span>
                </label>
                <label className="flex cursor-pointer items-center gap-2 text-sm text-zinc-400">
                  <input
                    type="checkbox"
                    checked={stream}
                    onChange={(e) => setStream(e.target.checked)}
                    className="h-4 w-4 rounded border-white/20 bg-ink-900 text-sky-500 focus:ring-sky-500/40"
                  />
                  流式输出（NDJSON）
                </label>
              </div>
              <button
                type="button"
                disabled={busy || !question.trim()}
                onClick={() => void runChat()}
                className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-cyan-500 px-5 py-2.5 text-sm font-medium text-white shadow-soft transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {busy ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
                发送
              </button>

              <div className="mt-8 border-t border-white/[0.06] pt-6">
                <h3 className="mb-3 flex items-center gap-2 text-sm font-medium text-zinc-300">
                  <BookOpen className="h-4 w-4 text-sky-400/80" />
                  回答
                </h3>
                <div className="min-h-[120px] rounded-xl border border-white/[0.06] bg-ink-900/50 p-4 text-sm leading-relaxed">
                  {!answer ? (
                    <p className="text-zinc-500">
                      {busy ? "生成中…" : "等待请求"}
                    </p>
                  ) : answerStreamPlain ? (
                    <div>
                      {busy && (
                        <p className="mb-2 text-[10px] font-medium uppercase tracking-wider text-zinc-600">
                          流式输出中 · 完成后渲染 Markdown
                        </p>
                      )}
                      <pre className="whitespace-pre-wrap break-words font-mono text-[13px] leading-relaxed text-zinc-200">
                        {answer}
                      </pre>
                    </div>
                  ) : (
                    <MarkdownView content={answer} />
                  )}
                </div>
                {confidence !== null && (
                  <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-zinc-500">
                    <p>
                      置信度{" "}
                      <span className="font-mono text-sky-300/90">
                        {(confidence * 100).toFixed(1)}%
                      </span>
                    </p>
                    {fastPathSource && (
                      <span className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2.5 py-1 font-mono text-[11px] text-emerald-200">
                        {fastPathSource}
                      </span>
                    )}
                    {answerMode && (
                      <span className="rounded-full border border-sky-500/20 bg-sky-500/10 px-2.5 py-1 font-mono text-[11px] text-sky-200">
                        {answerMode}
                      </span>
                    )}
                    {dataClassification && (
                      <span className="rounded-full border border-amber-500/20 bg-amber-500/10 px-2.5 py-1 font-mono text-[11px] text-amber-200">
                        {dataClassification}
                      </span>
                    )}
                    {modelRoute && (
                      <span className="rounded-full border border-fuchsia-500/20 bg-fuchsia-500/10 px-2.5 py-1 font-mono text-[11px] text-fuchsia-200">
                        {modelRoute}
                      </span>
                    )}
                    {analysisSource && (
                      <span className="rounded-full border border-violet-500/20 bg-violet-500/10 px-2.5 py-1 font-mono text-[11px] text-violet-200">
                        {analysisSource}
                      </span>
                    )}
                    {analysisConfidence !== null && (
                      <span className="rounded-full border border-cyan-500/20 bg-cyan-500/10 px-2.5 py-1 font-mono text-[11px] text-cyan-200">
                        analysis {(analysisConfidence * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                )}
                {(conflictDetected || refusalReason || traceId || auditId || analysisReason) && (
                  <div className="mt-4 space-y-3">
                    {analysisReason && (
                      <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/10 p-4 text-sm text-cyan-100">
                        <p className="text-[11px] font-semibold uppercase tracking-wider text-cyan-300/90">
                          Query Understanding
                        </p>
                        <p className="mt-2 leading-relaxed">{analysisReason}</p>
                      </div>
                    )}
                    {conflictDetected && conflictSummary && (
                      <div className="rounded-xl border border-amber-500/20 bg-amber-500/10 p-4 text-sm text-amber-100">
                        <p className="text-[11px] font-semibold uppercase tracking-wider text-amber-300/90">
                          冲突提示
                        </p>
                        <p className="mt-2 leading-relaxed">{conflictSummary}</p>
                      </div>
                    )}
                    {refusal && refusalReason && (
                      <div className="rounded-xl border border-rose-500/20 bg-rose-500/10 p-4 text-sm text-rose-100">
                        <p className="text-[11px] font-semibold uppercase tracking-wider text-rose-300/90">
                          拒答原因
                        </p>
                        <p className="mt-2 font-mono text-xs">{refusalReason}</p>
                      </div>
                    )}
                    {traceId && (
                      <div className="rounded-xl border border-white/[0.06] bg-ink-900/40 p-4 text-xs text-zinc-400">
                        <p className="text-zinc-500">请求链路 Trace ID</p>
                        <p className="mt-2 font-mono break-all text-zinc-200">{traceId}</p>
                      </div>
                    )}
                    {auditId && (
                      <div className="rounded-xl border border-white/[0.06] bg-ink-900/40 p-4 text-xs text-zinc-400">
                        <p className="text-zinc-500">审计追踪 ID</p>
                        <p className="mt-2 font-mono break-all text-zinc-200">{auditId}</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </section>

            <aside className="space-y-6">
              <div className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-5 shadow-panel backdrop-blur-xl">
                <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                  引用
                </h3>
                {citations.length === 0 ? (
                  <p className="text-sm text-zinc-600">暂无结构化引用</p>
                ) : (
                  <ul className="space-y-3">
                    {citations.map((c) => (
                      <li
                        key={c.chunk_id}
                        className="rounded-lg border border-white/[0.06] bg-ink-900/40 p-3 text-xs"
                      >
                        <p className="font-medium text-zinc-200">{c.title || "未命名"}</p>
                        <p className="mt-1 font-mono text-[10px] text-zinc-500 break-all">
                          {c.source}
                        </p>
                        <p className="mt-1 text-zinc-500">
                          chunk <span className="font-mono text-zinc-400">{c.chunk_id}</span>
                          {c.section ? ` · ${c.section}` : ""}
                          {c.page != null ? ` · p.${c.page}` : ""}
                        </p>
                        {(c.doc_type ||
                          c.owner_department ||
                          c.business_domain ||
                          c.process_stage ||
                          c.version ||
                          c.effective_date ||
                          c.authority_level ||
                          c.data_classification) && (
                          <div className="mt-2 flex flex-wrap gap-2">
                            {c.doc_type && (
                              <span className="rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[10px] text-zinc-300">
                                {c.doc_type}
                              </span>
                            )}
                            {c.owner_department && (
                              <span className="rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[10px] text-zinc-300">
                                {c.owner_department}
                              </span>
                            )}
                            {c.business_domain && (
                              <span className="rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[10px] text-zinc-300">
                                {c.business_domain}
                              </span>
                            )}
                            {c.process_stage && (
                              <span className="rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[10px] text-zinc-300">
                                {c.process_stage}
                              </span>
                            )}
                            {c.version && (
                              <span className="rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[10px] text-zinc-300">
                                v{c.version}
                              </span>
                            )}
                            {c.effective_date && (
                              <span className="rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[10px] text-zinc-300">
                                {c.effective_date}
                              </span>
                            )}
                            {c.authority_level && (
                              <span className="rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[10px] text-zinc-300">
                                {c.authority_level}
                              </span>
                            )}
                            {c.data_classification && (
                              <span className="rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[10px] text-zinc-300">
                                {c.data_classification}
                              </span>
                            )}
                          </div>
                        )}
                        {(c.selection_reason ||
                          c.section_path ||
                          (c.matched_routes && c.matched_routes.length > 0) ||
                          c.governance_rank_score != null) && (
                          <div className="mt-3 rounded-md border border-amber-500/15 bg-amber-500/5 p-2 text-[10px] text-zinc-400">
                            {c.selection_reason && (
                              <p className="leading-relaxed text-zinc-300">{c.selection_reason}</p>
                            )}
                            {c.section_path && (
                              <p className="mt-1">
                                章节路径: <span className="text-zinc-200">{c.section_path}</span>
                              </p>
                            )}
                            {c.matched_routes && c.matched_routes.length > 0 && (
                              <p className="mt-1">
                                命中路线: <span className="text-zinc-200">{c.matched_routes.join(" / ")}</span>
                              </p>
                            )}
                            {(c.retrieval_score != null ||
                              c.semantic_score != null ||
                              c.governance_rank_score != null) && (
                              <p className="mt-1">
                                分数:
                                {c.retrieval_score != null && (
                                  <span className="ml-1 text-zinc-200">
                                    retrieval {c.retrieval_score.toFixed(3)}
                                  </span>
                                )}
                                {c.semantic_score != null && (
                                  <span className="ml-2 text-zinc-200">
                                    semantic {c.semantic_score.toFixed(3)}
                                  </span>
                                )}
                                {c.governance_rank_score != null && (
                                  <span className="ml-2 text-zinc-200">
                                    governance {c.governance_rank_score.toFixed(3)}
                                  </span>
                                )}
                              </p>
                            )}
                          </div>
                        )}
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              <div className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-5 shadow-panel backdrop-blur-xl">
                <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                  检索片段
                </h3>
                {chunks.length === 0 ? (
                  <p className="text-sm text-zinc-600">无返回片段</p>
                ) : (
                  <ul className="max-h-[420px] space-y-3 overflow-y-auto pr-1">
                    {chunks.map((ch) => (
                      <li
                        key={ch.chunk_id}
                        className="rounded-lg border border-white/[0.06] bg-ink-900/40 p-3"
                      >
                        <div className="mb-1 flex items-center justify-between gap-2">
                          <span className="font-mono text-[10px] text-sky-300/80">
                            {ch.chunk_id}
                          </span>
                          <span className="text-[10px] text-zinc-500">
                            score {ch.score.toFixed(4)}
                          </span>
                        </div>
                        <div className="max-h-40 overflow-y-auto text-xs leading-relaxed">
                          <MarkdownView content={ch.content} compact />
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </aside>
          </div>
          </>
        )}

        {tab === "ingest" && (
          <>
            {/* 入库页：上传文件并跟踪后台任务状态。 */}
          <section className="max-w-xl rounded-2xl border border-white/[0.08] bg-white/[0.03] p-8 shadow-panel backdrop-blur-xl">
            <h2 className="mb-2 text-lg font-medium text-white">文档入库</h2>
            <p className="mb-6 text-sm text-zinc-500">
              支持 PDF / DOCX / PPTX / HTML / Markdown / TXT / CSV。任务异步执行，可轮询状态。
            </p>
            <input
              type="file"
              accept=".pdf,.docx,.pptx,.html,.htm,.md,.markdown,.txt,.csv"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              className="mb-6 block w-full cursor-pointer text-sm text-zinc-400 file:mr-4 file:rounded-lg file:border-0 file:bg-white/10 file:px-4 file:py-2 file:text-sm file:text-zinc-200 hover:file:bg-white/[0.14]"
            />
            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                disabled={busy}
                onClick={() => void runIngest()}
                className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-cyan-500 px-5 py-2.5 text-sm font-medium text-white shadow-soft transition hover:brightness-110 disabled:opacity-40"
              >
                <Upload className="h-4 w-4" />
                上传并入库
              </button>
              <button
                type="button"
                disabled={busy}
                onClick={() => void runReindex()}
                className="inline-flex items-center gap-2 rounded-xl border border-white/15 bg-white/[0.04] px-5 py-2.5 text-sm text-zinc-200 transition hover:bg-white/[0.07]"
              >
                <RefreshCw className="h-4 w-4" />
                重建索引
              </button>
            </div>
            {jobId && (
              <div className="mt-8 rounded-xl border border-white/[0.06] bg-ink-900/50 p-4">
                <p className="text-xs text-zinc-500">Job ID</p>
                <p className="font-mono text-sm text-sky-300/90">{jobId}</p>
                <p className="mt-2 text-sm text-zinc-400">
                  状态：{" "}
                  <span className="text-white">{jobStatus ?? "—"}</span>
                  {jobDetail ? (
                    <span className="block mt-1 text-rose-300/90 text-xs">{jobDetail}</span>
                  ) : null}
                </p>
                <button
                  type="button"
                  onClick={() => void pollJob()}
                  className="mt-3 inline-flex items-center gap-1 text-xs text-sky-400 hover:text-sky-300"
                >
                  刷新状态
                  <ChevronRight className="h-3 w-3" />
                </button>
              </div>
            )}
          </section>
          </>
        )}

        {tab === "faq" && (
          <>
            {/* FAQ 页同时包含“导入”和“已导入 FAQ 管理”。 */}
            <div className="grid gap-6 xl:grid-cols-[420px_1fr]">
              <section className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-8 shadow-panel backdrop-blur-xl">
                <h2 className="mb-2 text-lg font-medium text-white">FAQ 导入</h2>
                <p className="mb-3 text-sm text-zinc-500">
                  把结构化 FAQ CSV 导入 MySQL。导入完成后，后续问答会先走
                  Redis / MySQL FAQ 快速通道，再决定是否进入 RAG。
                </p>
                <div className="mb-6 rounded-xl border border-white/[0.06] bg-ink-900/45 p-4 text-xs text-zinc-400">
                  <p className="mb-2 font-medium text-zinc-300">推荐 CSV 表头</p>
                  <code className="font-mono text-[11px] text-sky-300/90">
                    question,answer,keywords,category
                  </code>
                </div>
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => setFaqFile(e.target.files?.[0] ?? null)}
                  className="mb-6 block w-full cursor-pointer text-sm text-zinc-400 file:mr-4 file:rounded-lg file:border-0 file:bg-white/10 file:px-4 file:py-2 file:text-sm file:text-zinc-200 hover:file:bg-white/[0.14]"
                />
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => void runFaqImport()}
                  className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-500 px-5 py-2.5 text-sm font-medium text-white shadow-soft transition hover:brightness-110 disabled:opacity-40"
                >
                  <Database className="h-4 w-4" />
                  导入 FAQ
                </button>

                {(faqImportStatus || faqImported !== null) && (
                  <div className="mt-6 rounded-xl border border-white/[0.06] bg-ink-900/50 p-4 text-sm">
                    <p className="text-xs text-zinc-500">导入状态</p>
                    <p className="mt-1 text-zinc-200">{faqImportStatus ?? "—"}</p>
                    <p className="mt-2 text-xs text-zinc-400">
                      导入条数：{" "}
                      <span className="font-mono text-emerald-300/90">
                        {faqImported ?? 0}
                      </span>
                    </p>
                  </div>
                )}
              </section>

              <section className="rounded-2xl border border-white/[0.08] bg-white/[0.03] p-8 shadow-panel backdrop-blur-xl">
                <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h2 className="text-lg font-medium text-white">已导入 FAQ</h2>
                    <p className="mt-1 text-sm text-zinc-500">
                      查看当前 FAQ 列表，并快速启用或停用某条 FAQ。
                    </p>
                  </div>
                  <button
                    type="button"
                    disabled={faqListLoading}
                    onClick={() => void loadFaqItems()}
                    className="inline-flex items-center gap-2 rounded-xl border border-white/15 bg-white/[0.04] px-4 py-2 text-sm text-zinc-200 transition hover:bg-white/[0.07] disabled:opacity-40"
                  >
                    <RefreshCw className={`h-4 w-4 ${faqListLoading ? "animate-spin" : ""}`} />
                    刷新列表
                  </button>
                </div>

                {faqItems.length === 0 ? (
                  <p className="text-sm text-zinc-600">
                    {faqListLoading ? "加载 FAQ 列表中…" : "暂无 FAQ 数据"}
                  </p>
                ) : (
                  <ul className="space-y-4">
                    {faqItems.map((item) => (
                      <li
                        key={item.id}
                        className="rounded-xl border border-white/[0.06] bg-ink-900/45 p-4"
                      >
                        <div className="mb-2 flex flex-wrap items-start justify-between gap-3">
                          <div>
                            <p className="text-sm font-medium text-zinc-100">
                              {item.question}
                            </p>
                            <p className="mt-1 text-[11px] font-mono text-zinc-500">
                              faq:{item.id}
                              {item.category ? ` · ${item.category}` : ""}
                            </p>
                          </div>
                          <button
                            type="button"
                            onClick={() => void toggleFaqItem(item)}
                            className={`rounded-full border px-3 py-1 text-[11px] font-medium transition ${
                              item.enabled
                                ? "border-emerald-500/25 bg-emerald-500/12 text-emerald-200 hover:bg-emerald-500/18"
                                : "border-zinc-500/25 bg-zinc-500/10 text-zinc-300 hover:bg-zinc-500/15"
                            }`}
                          >
                            {item.enabled ? "已启用" : "已停用"}
                          </button>
                        </div>
                        <div className="rounded-lg bg-black/20 p-3 text-xs leading-relaxed text-zinc-300">
                          <MarkdownView content={item.answer} compact />
                        </div>
                        {item.keywords && (
                          <p className="mt-3 text-[11px] text-zinc-500">
                            关键词：{item.keywords}
                          </p>
                        )}
                      </li>
                    ))}
                  </ul>
                )}
              </section>
            </div>
          </>
        )}

        {tab === "eval" && (
          <>
            {/* 评测页：触发后端 RAGAS 评测并展示摘要结果。 */}
          <section className="max-w-xl rounded-2xl border border-white/[0.08] bg-white/[0.03] p-8 shadow-panel backdrop-blur-xl">
            <h2 className="mb-2 text-lg font-medium text-white">RAGAS 评测</h2>
            <p className="mb-6 text-sm text-zinc-500">
              调用后端 <code className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-xs">POST /eval</code>
              ，需配置 <code className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-xs">OPENAI_API_KEY</code>。
            </p>
            <button
              type="button"
              disabled={evalBusy}
              onClick={() => void runEval()}
              className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-500 to-fuchsia-500 px-5 py-2.5 text-sm font-medium text-white shadow-soft transition hover:brightness-110 disabled:opacity-40"
            >
              {evalBusy ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <FlaskConical className="h-4 w-4" />
              )}
              运行评测
            </button>
            {evalOut && (
              <div className="mt-6 rounded-xl border border-white/[0.06] bg-ink-900/50 p-4 text-sm">
                <p className="text-xs text-zinc-500">报告路径</p>
                <p className="font-mono text-xs text-sky-300/90 break-all">{evalOut}</p>
                {evalAnalysisOut && (
                  <>
                    <p className="mt-3 text-xs text-zinc-500">Explainability 报告</p>
                    <p className="font-mono text-xs text-amber-300/90 break-all">
                      {evalAnalysisOut}
                    </p>
                  </>
                )}
                {evalSummary && Object.keys(evalSummary).length > 0 && (
                  <ul className="mt-4 space-y-1 text-xs text-zinc-400">
                    {Object.entries(evalSummary).map(([k, v]) => (
                      <li key={k} className="flex justify-between gap-4">
                        <span>{k}</span>
                        <span className="font-mono text-zinc-200">{v.toFixed(4)}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </section>
          </>
        )}

        {tab === "settings" && (
          <>
            {/* 设置页：解决“前端到底连哪个后端”这个常见本地开发问题。 */}
          <section className="max-w-xl rounded-2xl border border-white/[0.08] bg-white/[0.03] p-8 shadow-panel backdrop-blur-xl">
            <h2 className="mb-2 text-lg font-medium text-white">API 地址</h2>
            <p className="mb-6 text-sm text-zinc-500">
              开发模式下留空即可通过 Vite 代理访问后端。跨域部署时填写完整 URL，并确保后端{" "}
              <code className="rounded bg-white/10 px-1 font-mono text-[11px]">CORS_ORIGINS</code>{" "}
              包含本页来源。
            </p>
            <input
              value={apiBase}
              onChange={(e) => setApiBase(e.target.value)}
              placeholder="例如 https://api.example.com 或留空"
              className="mb-4 w-full rounded-xl border border-white/10 bg-ink-900/80 px-4 py-3 text-sm text-zinc-100 outline-none focus:border-sky-500/40"
            />
            <button
              type="button"
              onClick={() => void ping()}
              className="rounded-xl border border-white/15 bg-white/[0.04] px-4 py-2 text-sm text-zinc-200 hover:bg-white/[0.07]"
            >
              测试连接
            </button>
          </section>
          </>
        )}

        <footer className="mt-16 border-t border-white/[0.06] pt-8 text-center text-[11px] text-zinc-600">
          Enterprise RAG Platform · 流程验证与演示控制台
        </footer>
      </main>
    </div>
  );
}
