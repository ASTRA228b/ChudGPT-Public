import { Bot, Check, Copy, User } from "lucide-react";
import { useState } from "react";
import Prism from "prismjs";
import "prismjs/components/prism-csharp";
import "prismjs/components/prism-css";
import "prismjs/components/prism-javascript";
import "prismjs/components/prism-json";
import "prismjs/components/prism-markup";
import "prismjs/components/prism-python";
import "prismjs/components/prism-sql";
import "prismjs/components/prism-typescript";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "../types";

interface Props {
  message: ChatMessage;
  showTimestamp: boolean;
  syntaxHighlighting: boolean;
  onToast: (message: string) => void;
}

export function MessageView({
  message,
  showTimestamp,
  syntaxHighlighting,
  onToast,
}: Props): JSX.Element {
  const [copied, setCopied] = useState(false);
  const copy = async (text: string) => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    onToast("Copied to clipboard");
    setTimeout(() => setCopied(false), 1200);
  };
  return (
    <article className={`message ${message.role}`}>
      <div className="avatar">
        {message.role === "user" ? <User size={17} /> : <Bot size={18} />}
      </div>
      <div className="message-body">
        <div className="message-meta">
          <strong>
            {message.role === "user"
              ? "You"
              : message.role === "error"
                ? "Connection error"
                : "ChudGPT"}
          </strong>
          {showTimestamp && (
            <time>
              {new Date(message.createdAt).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </time>
          )}
        </div>
        <div className="message-content">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              code({ className, children, ...rest }) {
                const language = /language-(\w+)/.exec(className ?? "")?.[1];
                const value = String(children).replace(/\n$/, "");
                if (!language) return <code {...rest}>{children}</code>;
                const grammar = Prism.languages[language];
                const highlighted =
                  syntaxHighlighting && grammar
                    ? Prism.highlight(value, grammar, language)
                    : undefined;
                return (
                  <div className="code-shell">
                    <div className="code-head">
                      <span>{language}</span>
                      <button onClick={() => void copy(value)}>
                        <Copy size={14} /> Copy code
                      </button>
                    </div>
                    <pre>
                      {highlighted ? (
                        <code
                          className={className}
                          dangerouslySetInnerHTML={{ __html: highlighted }}
                        />
                      ) : (
                        <code className={className}>{value}</code>
                      )}
                    </pre>
                  </div>
                );
              },
              a({ href, children }) {
                return (
                  <span className="safe-link" title={href}>
                    {children}
                  </span>
                );
              },
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>
        <button
          className="copy-message"
          onClick={() => void copy(message.content)}
        >
          {copied ? <Check size={14} /> : <Copy size={14} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
    </article>
  );
}
