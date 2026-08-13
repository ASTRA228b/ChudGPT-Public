import {
  ChevronLeft,
  Menu,
  MessageSquare,
  MoreHorizontal,
  Plus,
  Search,
  Settings,
} from "lucide-react";
import { useMemo, useState } from "react";
import { groupDate, searchConversations } from "../lib/state";
import type { Conversation } from "../types";

interface Props {
  conversations: Conversation[];
  activeId: string | null;
  collapsed: boolean;
  onToggle: () => void;
  onNew: () => void;
  onSelect: (id: string) => void;
  onRename: (chat: Conversation) => void;
  onDelete: (chat: Conversation) => void;
  onSettings: () => void;
}

export function Sidebar(props: Props): JSX.Element {
  const [query, setQuery] = useState("");
  const filtered = useMemo(
    () => searchConversations(props.conversations, query),
    [props.conversations, query],
  );
  const groups = ["Today", "Previous"] as const;
  return (
    <aside className={`sidebar ${props.collapsed ? "collapsed" : ""}`}>
      <div className="sidebar-top">
        <button
          className="icon-button collapse"
          aria-label="Toggle sidebar"
          onClick={props.onToggle}
        >
          {props.collapsed ? <Menu size={20} /> : <ChevronLeft size={20} />}
        </button>
        {!props.collapsed && (
          <span className="sidebar-wordmark">
            CHUD<span>GPT</span>
          </span>
        )}
      </div>
      <button className="new-chat" onClick={props.onNew}>
        <Plus size={18} />
        <span>New Chat</span>
      </button>
      {!props.collapsed && (
        <label className="search-box">
          <Search size={16} />
          <input
            id="chat-search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search chats"
          />
        </label>
      )}
      <nav className="chat-list" aria-label="Saved conversations">
        {groups.map((group) => {
          const chats = filtered.filter(
            (chat) => groupDate(chat.updatedAt) === group,
          );
          if (!chats.length) return null;
          return (
            <section key={group} className="chat-group">
              {!props.collapsed && <h2>{group}</h2>}
              {chats.map((chat) => (
                <div
                  key={chat.id}
                  className={`chat-row ${chat.id === props.activeId ? "active" : ""}`}
                >
                  <button
                    className="chat-select"
                    title={chat.title}
                    onClick={() => props.onSelect(chat.id)}
                  >
                    <MessageSquare size={16} />
                    <span>{chat.title}</span>
                  </button>
                  {!props.collapsed && (
                    <button
                      className="chat-menu"
                      aria-label={`Actions for ${chat.title}`}
                      onClick={(event) => {
                        const rename = window.confirm(
                          `Rename “${chat.title}”?\n\nOK = Rename, Cancel = Delete options`,
                        );
                        if (rename) props.onRename(chat);
                        else if (window.confirm(`Delete “${chat.title}”?`))
                          props.onDelete(chat);
                        event.stopPropagation();
                      }}
                    >
                      <MoreHorizontal size={16} />
                    </button>
                  )}
                </div>
              ))}
            </section>
          );
        })}
      </nav>
      <button className="settings-button" onClick={props.onSettings}>
        <Settings size={18} />
        <span>Settings</span>
      </button>
    </aside>
  );
}
