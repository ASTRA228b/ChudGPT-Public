import {
  Bot,
  PanelLeftClose,
  PanelLeftOpen,
  RefreshCw,
  Wifi,
  WifiOff,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cancel, chat, clearSession, makeRequestId, status } from "./lib/api";
import {
  createConversation,
  defaultSettings,
  emptyState,
  normalizeState,
  titleFromMessage,
} from "./lib/state";
import type {
  AppSettings,
  ChatMessage,
  ConnectionState,
  Conversation,
  PersistedState,
} from "./types";
import { Composer } from "./components/Composer";
import { MessageView } from "./components/MessageView";
import { Onboarding } from "./components/Onboarding";
import { SettingsModal } from "./components/SettingsModal";
import { Sidebar } from "./components/Sidebar";
import { TitleBar } from "./components/TitleBar";
import { Toast, ToastStack } from "./components/ToastStack";
import { Welcome } from "./components/Welcome";

const now = () => new Date().toISOString();
const message = (role: ChatMessage["role"], content: string): ChatMessage => ({
  id: crypto.randomUUID(),
  role,
  content,
  createdAt: now(),
});

export default function App(): JSX.Element {
  const [state, setState] = useState<PersistedState>(emptyState);
  const [loaded, setLoaded] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [requestId, setRequestId] = useState<string | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [version, setVersion] = useState("0.1.0");
  const [toasts, setToasts] = useState<Toast[]>([]);
  const messagesEnd = useRef<HTMLDivElement>(null);
  const active = useMemo(
    () =>
      state.conversations.find(
        (chat) => chat.id === state.activeConversationId,
      ) ?? null,
    [state],
  );

  const toast = useCallback((text: string) => {
    const id = crypto.randomUUID();
    setToasts((items) => [...items, { id, message: text }]);
    setTimeout(
      () => setToasts((items) => items.filter((item) => item.id !== id)),
      2800,
    );
  }, []);

  useEffect(() => {
    Promise.all([
      window.chudDesktop.loadData(),
      window.chudDesktop.getAppInfo(),
    ])
      .then(([data, info]) => {
        const restored = normalizeState(data);
        if (
          restored.settings.startOnNewChat ||
          !restored.settings.rememberLastChat
        )
          restored.activeConversationId = null;
        setState(restored);
        setVersion(info.version);
        setLoaded(true);
      })
      .catch(() => {
        setLoaded(true);
        toast("Local data could not be loaded");
      });
  }, [toast]);

  useEffect(() => {
    if (loaded)
      void window.chudDesktop.saveData(
        state as unknown as Record<string, unknown>,
      );
  }, [state, loaded]);
  const checkConnection = useCallback(async () => {
    setConnection("connecting");
    try {
      await status();
      setConnection("online");
    } catch {
      setConnection("offline");
    }
  }, []);
  useEffect(() => {
    if (!loaded) return;
    void checkConnection();
    const id = window.setInterval(() => void checkConnection(), 60_000);
    return () => window.clearInterval(id);
  }, [checkConnection, loaded]);
  useEffect(() => {
    if (state.settings.autoScroll)
      messagesEnd.current?.scrollIntoView({
        behavior: state.settings.reduceAnimations ? "auto" : "smooth",
      });
  }, [
    active?.messages.length,
    busy,
    state.settings.autoScroll,
    state.settings.reduceAnimations,
  ]);
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if (!event.ctrlKey) return;
      if (event.key.toLowerCase() === "n") {
        event.preventDefault();
        newChat();
      }
      if (event.key.toLowerCase() === "k") {
        event.preventDefault();
        document.querySelector<HTMLInputElement>("#chat-search")?.focus();
      }
      if (event.key === ",") {
        event.preventDefault();
        setSettingsOpen(true);
      }
      if (event.key.toLowerCase() === "l") {
        event.preventDefault();
        window.dispatchEvent(new Event("chud:focus-composer"));
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  });

  const updateChat = (
    id: string,
    updater: (chat: Conversation) => Conversation,
  ) =>
    setState((current) => ({
      ...current,
      conversations: current.conversations.map((chat) =>
        chat.id === id ? updater(chat) : chat,
      ),
    }));
  const newChat = () => {
    const chat = createConversation();
    setState((current) => ({
      ...current,
      conversations: [chat, ...current.conversations],
      activeConversationId: chat.id,
    }));
    setDraft("");
  };
  const ensureChat = (): Conversation => {
    if (active) return active;
    const created = createConversation();
    setState((current) => ({
      ...current,
      conversations: [created, ...current.conversations],
      activeConversationId: created.id,
    }));
    return created;
  };
  const send = async () => {
    const clean = draft.trim();
    if (!clean || busy) return;
    const target = ensureChat();
    const userMessage = message("user", clean);
    setDraft("");
    setBusy(true);
    updateChat(target.id, (item) => ({
      ...item,
      title:
        state.settings.autoTitles && !item.messages.length
          ? titleFromMessage(clean)
          : item.title,
      messages: [...item.messages, userMessage],
      updatedAt: now(),
    }));
    const id = makeRequestId();
    setRequestId(id);
    try {
      const result = await chat(clean, target.sessionId, id);
      updateChat(target.id, (item) => ({
        ...item,
        messages: [...item.messages, message("assistant", result.reply)],
        updatedAt: now(),
      }));
      setConnection("online");
    } catch (error) {
      const text =
        error instanceof Error
          ? error.message
          : "Could not connect to ChudGPT-Public.";
      updateChat(target.id, (item) => ({
        ...item,
        messages: [
          ...item.messages,
          message(
            "error",
            `${text}\n\nYour local chat is still saved. Try again when the server is online.`,
          ),
        ],
        updatedAt: now(),
      }));
      setConnection("offline");
      toast("Could not connect to ChudGPT-Public");
    } finally {
      setBusy(false);
      setRequestId(null);
    }
  };
  const stop = () => {
    if (requestId) void cancel(requestId);
    setBusy(false);
    setRequestId(null);
    toast("Generation stopped");
  };
  const removeChat = (chat: Conversation) => {
    if (
      state.settings.confirmDeletes &&
      !window.confirm(`Delete “${chat.title}”? This cannot be undone.`)
    )
      return;
    void clearSession(chat.sessionId).catch(() => undefined);
    setState((current) => ({
      ...current,
      conversations: current.conversations.filter(
        (item) => item.id !== chat.id,
      ),
      activeConversationId:
        current.activeConversationId === chat.id
          ? null
          : current.activeConversationId,
    }));
    toast("Chat deleted");
  };
  const renameChat = (chat: Conversation) => {
    const title = window.prompt("Rename conversation", chat.title)?.trim();
    if (!title) return;
    updateChat(chat.id, (item) => ({
      ...item,
      title: title.slice(0, 80),
      updatedAt: now(),
    }));
    toast("Chat renamed");
  };
  const changeSettings = (settings: AppSettings) => {
    setState((current) => ({ ...current, settings }));
    toast("Settings saved");
  };
  const exportData = async () => {
    if (
      await window.chudDesktop.exportData(
        state as unknown as Record<string, unknown>,
      )
    )
      toast("Chats exported");
  };
  const importData = async () => {
    try {
      const data = await window.chudDesktop.importData();
      if (!data) return;
      const imported = normalizeState(data);
      if (
        !window.confirm(
          `Import ${imported.conversations.length} conversations and replace current local data?`,
        )
      )
        return;
      setState(imported);
      toast("Chats imported");
    } catch {
      toast("That file is not a valid ChudGPT export");
    }
  };

  if (!loaded)
    return (
      <div className="boot-screen">
        <div className="hero-mark">
          <Bot />
        </div>
        <p>Initializing ChudGPT Desktop...</p>
      </div>
    );
  if (!state.onboardingComplete)
    return (
      <>
        <TitleBar />
        <Onboarding
          onFinish={() =>
            setState((current) => ({ ...current, onboardingComplete: true }))
          }
        />
      </>
    );
  const style = {
    "--ui-scale": state.settings.interfaceScale / 100,
    "--glow": state.settings.glowIntensity / 100,
  } as React.CSSProperties;
  return (
    <div
      className={`app theme-${state.settings.theme} density-${state.settings.density} ${state.settings.reduceAnimations ? "reduce-motion" : ""}`}
      style={style}
    >
      <TitleBar />
      <div className="workspace">
        <Sidebar
          conversations={state.conversations}
          activeId={state.activeConversationId}
          collapsed={sidebarCollapsed}
          onToggle={() => setSidebarCollapsed((value) => !value)}
          onNew={newChat}
          onSelect={(id) =>
            setState((current) => ({ ...current, activeConversationId: id }))
          }
          onRename={renameChat}
          onDelete={removeChat}
          onSettings={() => setSettingsOpen(true)}
        />
        <main className="main-panel">
          <header className="model-header">
            <button
              className="mobile-sidebar"
              onClick={() => setSidebarCollapsed((value) => !value)}
            >
              {sidebarCollapsed ? <PanelLeftOpen /> : <PanelLeftClose />}
            </button>
            <div>
              <h1>
                ChudGPT <span>Public</span>
              </h1>
              <p>Independent experimental language model</p>
            </div>
            <button
              className={`status-pill ${connection}`}
              onClick={() => void checkConnection()}
            >
              {connection === "offline" ? (
                <WifiOff />
              ) : connection === "connecting" ? (
                <RefreshCw className="spin" />
              ) : (
                <Wifi />
              )}
              <span>
                Public · {connection[0].toUpperCase() + connection.slice(1)}
              </span>
            </button>
          </header>
          <section className="conversation">
            <div className="message-scroll">
              {!active || !active.messages.length ? (
                <Welcome
                  onStarter={(text) => {
                    setDraft(text);
                    window.dispatchEvent(new Event("chud:focus-composer"));
                  }}
                />
              ) : (
                active.messages.map((item) => (
                  <MessageView
                    key={item.id}
                    message={item}
                    showTimestamp={state.settings.showTimestamps}
                    syntaxHighlighting={state.settings.syntaxHighlighting}
                    onToast={toast}
                  />
                ))
              )}
              {busy && (
                <div className="thinking">
                  <div className="avatar">
                    <Bot />
                  </div>
                  <span>ChudGPT is thinking</span>
                  <i />
                  <i />
                  <i />
                </div>
              )}
              <div ref={messagesEnd} />
            </div>
            <Composer
              value={draft}
              busy={busy}
              sendWithEnter={state.settings.sendWithEnter}
              onChange={setDraft}
              onSend={() => void send()}
              onStop={stop}
            />
          </section>
        </main>
      </div>
      {settingsOpen && (
        <SettingsModal
          settings={state.settings}
          state={state}
          version={version}
          onChange={changeSettings}
          onClose={() => setSettingsOpen(false)}
          onExport={() => void exportData()}
          onImport={() => void importData()}
          onDeleteAll={() => {
            if (
              window.confirm(
                "Delete every locally saved conversation? This cannot be undone.",
              )
            ) {
              setState((current) => ({
                ...current,
                conversations: [],
                activeConversationId: null,
              }));
              toast("All chats deleted");
            }
          }}
          onReset={() => {
            if (window.confirm("Reset all settings to defaults?"))
              changeSettings({ ...defaultSettings });
          }}
          onToast={toast}
        />
      )}
      <ToastStack toasts={toasts} />
    </div>
  );
}
