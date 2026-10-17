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
  }) => (
    <div>
      <button onClick={() => props.onChange("hello")}>Draft</button>
      <button onClick={props.onSend}>Send</button>
      <button onClick={props.onStop}>Stop</button>
      <span>{props.busy ? "busy" : "idle"}</span>
    </div>
  ),
}));
vi.mock("./components/Sidebar", () => ({ Sidebar: () => null }));
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
