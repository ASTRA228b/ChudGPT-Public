import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createConversation } from "../lib/state";
import { Sidebar } from "./Sidebar";

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
let root: Root;
let container: HTMLDivElement;

beforeEach(() => {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});
afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.restoreAllMocks();
});

it("offers separate Rename and Delete actions and saves an inline name", () => {
  const chat = createConversation("public");
  const onRename = vi.fn();
  const onDelete = vi.fn();
  act(() =>
    root.render(
      <Sidebar
        conversations={[chat]}
        activeId={chat.id}
        collapsed={false}
        onToggle={vi.fn()}
        onNew={vi.fn()}
        onSelect={vi.fn()}
        onRename={onRename}
        onDelete={onDelete}
        onSettings={vi.fn()}
      />,
    ),
  );
  const button = (label: string) =>
    [...container.querySelectorAll("button")].find(
      (item) =>
        item.textContent?.trim() === label ||
        item.getAttribute("aria-label") === label,
    )!;
  act(() => button(`Actions for ${chat.title}`).click());
  expect(button("Rename")).toBeTruthy();
  expect(button("Delete")).toBeTruthy();
  expect(onDelete).not.toHaveBeenCalled();
  act(() => button("Rename").click());
  const input =
    container.querySelector<HTMLInputElement>(".chat-rename input")!;
  expect(document.activeElement).toBe(input);
  act(() => {
    Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype,
      "value",
    )!.set!.call(input, "New title");
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  act(() => button("Save chat name").click());
  expect(onRename).toHaveBeenCalledWith(chat, "New title");
  expect(onDelete).not.toHaveBeenCalled();
  act(() => button(`Actions for ${chat.title}`).click());
  act(() => button("Delete").click());
  expect(onDelete).toHaveBeenCalledOnce();
});

it("canceling rename leaves the chat untouched", () => {
  const chat = createConversation("public");
  const onRename = vi.fn();
  const onDelete = vi.fn();
  act(() =>
    root.render(
      <Sidebar
        conversations={[chat]}
        activeId={chat.id}
        collapsed={false}
        onToggle={vi.fn()}
        onNew={vi.fn()}
        onSelect={vi.fn()}
        onRename={onRename}
        onDelete={onDelete}
        onSettings={vi.fn()}
      />,
    ),
  );
  const actions = container.querySelector<HTMLButtonElement>(".chat-menu")!;
  act(() => actions.click());
  const rename = [...container.querySelectorAll("button")].find(
    (item) => item.textContent?.trim() === "Rename",
  )!;
  act(() => rename.click());
  const cancel = container.querySelector<HTMLButtonElement>(
    '[aria-label="Cancel rename"]',
  )!;
  act(() => cancel.click());
  expect(container.querySelector(".chat-rename")).toBeNull();
  expect(onRename).not.toHaveBeenCalled();
  expect(onDelete).not.toHaveBeenCalled();
});
