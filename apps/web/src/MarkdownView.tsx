/*
  Markdown 渲染组件。

  后端答案和检索片段都可能返回 Markdown，
  这里统一处理渲染、样式映射与基础安全净化。
*/

import ReactMarkdown from "react-markdown";
import rehypeSanitize from "rehype-sanitize";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";

type Props = {
  content: string;
  /** 检索卡片等紧凑排版 */
  compact?: boolean;
  className?: string;
};

// 把 Markdown 标签映射到项目自己的视觉风格，而不是浏览器默认样式。
const mdComponents = (compact: boolean): Components => ({
  h1: ({ children }) => (
    <h1
      className={
        compact
          ? "mb-2 text-base font-semibold tracking-tight text-white"
          : "mb-3 mt-6 text-lg font-semibold tracking-tight text-white first:mt-0"
      }
    >
      {children}
    </h1>
  ),
  h2: ({ children }) => (
    <h2
      className={
        compact
          ? "mb-1.5 mt-3 text-sm font-semibold text-zinc-100 first:mt-0"
          : "mb-2 mt-5 text-base font-semibold text-zinc-100 first:mt-0"
      }
    >
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3
      className={
        compact
          ? "mb-1 mt-2 text-xs font-semibold text-zinc-200"
          : "mb-2 mt-4 text-sm font-semibold text-zinc-200"
      }
    >
      {children}
    </h3>
  ),
  p: ({ children }) => (
    <p className={compact ? "mb-2 last:mb-0 text-zinc-400" : "mb-3 last:mb-0 text-zinc-300"}>
      {children}
    </p>
  ),
  ul: ({ children }) => (
    <ul
      className={
        compact
          ? "mb-2 list-disc space-y-1 pl-4 text-zinc-400 last:mb-0"
          : "mb-3 list-disc space-y-1.5 pl-5 text-zinc-300 last:mb-0"
      }
    >
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol
      className={
        compact
          ? "mb-2 list-decimal space-y-1 pl-4 text-zinc-400 last:mb-0"
          : "mb-3 list-decimal space-y-1.5 pl-5 text-zinc-300 last:mb-0"
      }
    >
      {children}
    </ol>
  ),
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  blockquote: ({ children }) => (
    <blockquote className="my-3 border-l-2 border-sky-500/40 pl-4 text-zinc-400 italic">
      {children}
    </blockquote>
  ),
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-sky-400 underline decoration-sky-500/30 underline-offset-2 transition hover:text-sky-300"
    >
      {children}
    </a>
  ),
  hr: () => <hr className="my-4 border-white/[0.08]" />,
  strong: ({ children }) => (
    <strong className="font-semibold text-zinc-100">{children}</strong>
  ),
  em: ({ children }) => <em className="text-zinc-200">{children}</em>,
  table: ({ children }) => (
    <div className="my-3 overflow-x-auto rounded-lg border border-white/[0.08]">
      <table className="w-full border-collapse text-left text-xs">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-white/[0.06] text-zinc-300">{children}</thead>,
  tbody: ({ children }) => <tbody className="text-zinc-400">{children}</tbody>,
  tr: ({ children }) => <tr className="border-b border-white/[0.06] last:border-0">{children}</tr>,
  th: ({ children }) => (
    <th className="px-3 py-2 font-medium text-zinc-200">{children}</th>
  ),
  td: ({ children }) => <td className="px-3 py-2 align-top">{children}</td>,
  pre: ({ children }) => (
    <pre className="my-3 overflow-x-auto rounded-xl border border-white/10 bg-black/45 p-4 text-[13px] leading-relaxed shadow-inner">
      {children}
    </pre>
  ),
  code: ({ className, children, ...props }) => {
    const raw = String(children);
    // 带 language-* 或包含换行时，按代码块渲染。
    const isBlock =
      Boolean(className?.trim().startsWith("language-")) || raw.includes("\n");
    if (isBlock) {
      return (
        <code
          className={`block min-w-full whitespace-pre font-mono text-[13px] leading-relaxed text-zinc-200 ${className ?? ""}`.trim()}
          {...props}
        >
          {children}
        </code>
      );
    }
    return (
      <code
        className="rounded-md bg-white/[0.08] px-1.5 py-0.5 font-mono text-[0.9em] text-sky-200/95"
        {...props}
      >
        {children}
      </code>
    );
  },
});

export function MarkdownView({ content, compact = false, className = "" }: Props) {
  return (
    <div className={`markdown-body ${className}`}>
      <ReactMarkdown
        // 支持 GFM 扩展语法，例如表格、任务列表。
        remarkPlugins={[remarkGfm]}
        // 过滤危险 HTML，避免把不可信内容直接注入 DOM。
        rehypePlugins={[rehypeSanitize]}
        components={mdComponents(compact)}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
