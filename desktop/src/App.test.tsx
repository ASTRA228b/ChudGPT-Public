import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App";
import { createConversation, emptyState } from "./lib/state";

vi.mock("./components/Composer", () => ({
  Composer: (props: {
    onChange: (value: string) => void;
    onSend: () => void;
    onStop: () => void;
    busy: boolean;
    modelProfile: string;
    onModelChange: (profile: "public" | "plato-mini") => void;
  }) => (
    <div>
      <button onClick={() => props.onChange("hello")}>Draft</button>
      <button onClick={props.onSend}>Send</button>
      <button onClick={props.onStop}>Stop</button>
      <button onClick={() => props.onModelChange("plato-mini")}>
        Choose Plato Mini
      </button>
      <span data-testid="selected-model">{props.modelProfile}</span>
      <span>{props.busy ? "busy" : "idle"}</span>
    </div>
  ),
}));
vi.mock("./components/Sidebar", () => ({
  Sidebar: (props: {
    conversations: { id: string; title: string }[];
    onSelect: (id: string) => void;
  }) => (
    <nav>
      {props.conversations.map((chat) => (
        <button key={chat.id} onClick={() => props.onSelect(chat.id)}>
          {chat.title}
        </button>
      ))}
    </nav>
  ),
}));
vi.mock("./components/TitleBar", () => ({ TitleBar: () => null }));
vi.mock("./components/MessageView", () => ({
  MessageView: ({ message }: { message: { content: string } }) => (
    <p>{message.content}</p>
  ),
}));
vi.mock("./components/Welcome", () => ({ Welcome: () => null }));
vi.mock("./components/ToastStack", () => ({ ToastStack: () => null }));
vi.mock("./components/SettingsModal", () => ({ SettingsModal: () => null }));

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
let root: Root;
let container: HTMLDivElement;

beforeEach(() => {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  window.HTMLElement.prototype.scrollIntoView = vi.fn();
});
afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.restoreAllMocks();
});

it("blocks a double send and ignores a late success after Stop", async () => {
  let complete!: (value: { reply: string }) => void;
  const pending = new Promise<{ reply: string }>((resolve) => {
    complete = resolve;
  });
  const apiRequest = vi.fn((endpoint: string) =>
    endpoint === "chat" ? pending : Promise.resolve({ ready: true }),
  );
  const saved = emptyState();
  const conversation = createConversation("public");
  saved.conversations = [conversation];
  saved.activeConversationId = conversation.id;
  saved.onboardingComplete = true;
  Object.defineProperty(window, "chudDesktop", {
    configurable: true,
    value: {
      loadData: vi.fn().mockResolvedValue(saved),
      getAppInfo: vi.fn().mockResolvedValue({ version: "test" }),
      saveData: vi.fn().mockResolvedValue(true),
      apiRequest,
      cancelRequest: vi.fn().mockResolvedValue(true),
    },
  });
  await act(async () => {
    root.render(<App />);
  });
  const button = (name: string) =>
    [...container.querySelectorAll("button")].find(
      (item) => item.textContent === name,
    )!;
  act(() => button("Draft").click());
  act(() => {
    button("Send").click();
    button("Send").click();
  });
  expect(
    apiRequest.mock.calls.filter(([endpoint]) => endpoint === "chat"),
  ).toHaveLength(1);
  act(() => button("Stop").click());
  await act(async () => complete({ reply: "late reply" }));
  expect(container.textContent).not.toContain("late reply");
  expect(container.textContent).toContain("idle");
});

it("switches to a fresh Plato chat and restores a saved chat's model", async () => {
  const saved = emptyState();
  const oldChat = createConversation("public");
  oldChat.title = "Old Public";
  oldChat.messages.push({
    id: "old-user",
    role: "user",
    content: "Old public history",
    createdAt: new Date().toISOString(),
  });
  saved.conversations = [oldChat];
  saved.activeConversationId = oldChat.id;
  saved.onboardingComplete = true;
  const apiRequest = vi.fn(async (...args: [string, string, unknown]) =>
    args[0] === "main/plato/models/plato-mini/chat"
      ? {
          reply: "Plato reply",
          model: "plato-mini",
          source: "neural",
          checkpoint_step: 1100,
        }
      : { reply: "Public reply", ready: true },
  );
  Object.defineProperty(window, "chudDesktop", {
    configurable: true,
    value: {
      loadData: vi.fn().mockResolvedValue(saved),
      getAppInfo: vi.fn().mockResolvedValue({ version: "test" }),
      saveData: vi.fn().mockResolvedValue(true),
      apiRequest,
      cancelRequest: vi.fn().mockResolvedValue(true),
    },
  });
  await act(async () => root.render(<App />));
  const button = (name: string) =>
    [...container.querySelectorAll("button")].find(
      (item) => item.textContent?.trim() === name,
    )!;
  act(() => button("Choose Plato Mini").click());
  expect(
    container.querySelector('[data-testid="selected-model"]')?.textContent,
  ).toBe("plato-mini");
  expect(container.textContent).not.toContain("Old public history");
  act(() => button("Draft").click());
  await act(async () => button("Send").click());
  const platoCall = apiRequest.mock.calls.find(
    ([endpoint]) => endpoint === "main/plato/models/plato-mini/chat",
  );
  expect(platoCall?.[2]).toEqual({
    messages: [{ role: "user", content: "hello" }],
    max_new_tokens: 128,
  });
  act(() => button("Old Public").click());
  expect(
    container.querySelector('[data-testid="selected-model"]')?.textContent,
  ).toBe("public");
  expect(container.textContent).toContain("Old public history");
  act(() => button("Draft").click());
  await act(async () => button("Send").click());
  const publicCall = apiRequest.mock.calls.find(
    ([endpoint]) => endpoint === "chat",
  );
  expect(publicCall?.[2]).toEqual({
    message: "hello",
    session_id: oldChat.sessionId,
  });
});

it("ignores a stale status result after the selected model changes", async () => {
  let finishOld!: (value: { ready: boolean }) => void;
  const oldStatus = new Promise<{ ready: boolean }>((resolve) => {
    finishOld = resolve;
  });
  const saved = emptyState();
  saved.onboardingComplete = true;
  Object.defineProperty(window, "chudDesktop", {
    configurable: true,
    value: {
      loadData: vi.fn().mockResolvedValue(saved),
      getAppInfo: vi.fn().mockResolvedValue({ version: "test" }),
      saveData: vi.fn().mockResolvedValue(true),
      apiRequest: vi.fn((endpoint: string) =>
        endpoint === "status"
          ? oldStatus
          : Promise.resolve({ status: "online" }),
      ),
      cancelRequest: vi.fn().mockResolvedValue(true),
    },
  });
  await act(async () => root.render(<App />));
  const choose = [...container.querySelectorAll("button")].find(
    (item) => item.textContent?.trim() === "Choose Plato Mini",
  )!;
  await act(async () => choose.click());
  expect(container.textContent).toContain("Plato Mini · Gateway online");
  await act(async () => finishOld({ ready: false }));
  expect(container.textContent).toContain("Plato Mini · Gateway online");
});
