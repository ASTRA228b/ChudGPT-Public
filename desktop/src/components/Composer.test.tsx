import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Composer } from "./Composer";

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });

describe("Composer focus", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.append(container);
    root = createRoot(container);
    vi.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => {
      callback(0);
      return 1;
    });
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  const render = (focusKey: string) =>
    act(() =>
      root.render(
        <Composer
          value=""
          busy={false}
          sendWithEnter
          focusKey={focusKey}
          modelProfile="public"
          onChange={vi.fn()}
          onSend={vi.fn()}
          onStop={vi.fn()}
          onModelChange={vi.fn()}
        />,
      ),
    );

  it("focuses the textarea when a new conversation becomes active", () => {
    render("first-chat");
    const composer =
      container.querySelector<HTMLTextAreaElement>("#message-composer");
    expect(document.activeElement).toBe(composer);

    const outside = document.createElement("button");
    container.append(outside);
    outside.focus();
    expect(document.activeElement).toBe(outside);

    render("second-chat");
    expect(document.activeElement).toBe(composer);
  });

  it("supports the global focus-composer shortcut event", () => {
    render("chat");
    const composer =
      container.querySelector<HTMLTextAreaElement>("#message-composer");
    document.body.focus();
    window.dispatchEvent(new Event("chud:focus-composer"));
    expect(document.activeElement).toBe(composer);
  });
});
