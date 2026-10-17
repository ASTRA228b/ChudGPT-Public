import {
  Check,
  ChevronLeft,
  Menu,
  MessageSquare,
  MoreHorizontal,
  Pencil,
  Plus,
  Search,
  Settings,
  Trash2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { groupDate, searchConversations } from "../lib/state";
import type { Conversation } from "../types";

interface Props {
  conversations: Conversation[];
  activeId: string | null;
  collapsed: boolean;
  onToggle: () => void;
  onNew: () => void;
  onSelect: (id: string) => void;
  onRename: (chat: Conversation, title: string) => void;
  onDelete: (chat: Conversation) => void;
  onSettings: () => void;
}

export function Sidebar(props: Props): JSX.Element {
  const [query, setQuery] = useState("");
  const [menuId, setMenuId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const openMenu = useRef<HTMLDivElement>(null);
  const menuButton = useRef<HTMLButtonElement>(null);
  const filtered = useMemo(
    () => searchConversations(props.conversations, query),
    [props.conversations, query],
  );
  useEffect(() => {
    if (!menuId) return;
    const outside = (event: PointerEvent) => {
      if (
        event.target instanceof Node &&
        !openMenu.current?.contains(event.target)
      )
        setMenuId(null);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMenuId(null);
        menuButton.current?.focus();
      }
    };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape);
    };
  }, [menuId]);
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
                  className="chat-row-wrap"
                  ref={menuId === chat.id ? openMenu : undefined}
                >
                  <div
                    className={`chat-row ${chat.id === props.activeId ? "active" : ""}`}
                  >
                    {editingId === chat.id ? (
                      <form
                        className="chat-rename"
                        onSubmit={(event) => {
                          event.preventDefault();
                          const title = editTitle.trim();
                          if (!title) return;
                          props.onRename(chat, title);
                          setEditingId(null);
                        }}
                      >
                        <input
                          aria-label={`New name for ${chat.title}`}
                          autoFocus
                          maxLength={80}
                          value={editTitle}
                          onChange={(event) => setEditTitle(event.target.value)}
                          onKeyDown={(event) => {
                            if (event.key === "Escape") setEditingId(null);
                          }}
                        />
                        <button
                          type="submit"
                          aria-label="Save chat name"
                          disabled={!editTitle.trim()}
                        >
                          <Check size={16} />
                        </button>
                        <button
                          type="button"
                          aria-label="Cancel rename"
                          onClick={() => setEditingId(null)}
                        >
                          <X size={16} />
                        </button>
                      </form>
                    ) : (
                      <>
                        <button
                          className="chat-select"
                          title={chat.title}
                          onClick={() => {
                            setMenuId(null);
                            props.onSelect(chat.id);
                          }}
                        >
                          <MessageSquare size={16} />
                          <span>{chat.title}</span>
                        </button>
                        {!props.collapsed && (
                          <button
                            ref={menuId === chat.id ? menuButton : undefined}
                            className="chat-menu"
                            aria-label={`Actions for ${chat.title}`}
                            aria-expanded={menuId === chat.id}
                            onClick={() =>
                              setMenuId((id) =>
                                id === chat.id ? null : chat.id,
                              )
                            }
                          >
                            <MoreHorizontal size={16} />
                          </button>
                        )}
                      </>
                    )}
                  </div>
                  {menuId === chat.id && !props.collapsed && (
                    <div
                      className="chat-actions"
                      role="group"
                      aria-label={`Actions for ${chat.title}`}
                    >
                      <button
                        onClick={() => {
                          setEditTitle(chat.title);
                          setEditingId(chat.id);
                          setMenuId(null);
                        }}
                      >
                        <Pencil size={15} /> Rename
                      </button>
                      <button
                        onClick={() => {
                          setMenuId(null);
                          props.onDelete(chat);
                        }}
                      >
                        <Trash2 size={15} /> Delete
                      </button>
                    </div>
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
