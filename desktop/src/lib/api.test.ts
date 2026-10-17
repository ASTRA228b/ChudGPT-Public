import { beforeEach, describe, expect, it, vi } from "vitest";
import { buildPlatoMessages, chat, clearSession, status } from "./api";
import type { ChatMessage } from "../types";

const apiRequest = vi.fn();

beforeEach(() => {
  apiRequest.mockReset();
  apiRequest.mockImplementation(async (endpoint: string) =>
    endpoint.startsWith("main/plato/models/")
      ? {
          reply: "ok",
          model: endpoint.includes("plato-mini") ? "plato-mini" : "plato",
          source: "neural",
          checkpoint_step: 1200,
        }
      : { reply: "ok", ready: true },
  );
  Object.defineProperty(window, "chudDesktop", {
    configurable: true,
    value: { apiRequest, cancelRequest: vi.fn() },
  });
});

describe("desktop API routing", () => {
  it("uses the stable Public V20 routes", async () => {
    await status("public", "status-id");
    await chat("hello", "session", "chat-id", "public");
    await clearSession("session", "public");

    expect(apiRequest.mock.calls.map(([endpoint]) => endpoint)).toEqual([
      "status",
      "chat",
      "clear",
    ]);
  });

  it("uses the stable Music V1 routes", async () => {
    await status("music", "status-id");
    await chat("song", "session", "chat-id", "music");
    await clearSession("session", "music");

    expect(apiRequest.mock.calls.map(([endpoint]) => endpoint)).toEqual([
      "music/status",
      "music/chat",
      "music/clear",
    ]);
  });

  it("keeps main-family models on their canonical API", async () => {
    await status("buggy", "status-id");
    await chat("hello", "session", "chat-id", "buggy");
    await clearSession("session", "buggy");

    expect(apiRequest.mock.calls.map(([endpoint]) => endpoint)).toEqual([
      "main/models/buggy",
      "main/models/buggy/chat",
      "main/models/buggy/clear",
    ]);
  });

  it("routes both Plato models through the exact stateless namespace", async () => {
    const history = [row("user", "hello")];
    await status("plato", "status-id");
    await chat("hello", "ignored-session", "chat-id", "plato", history);
    await chat("hello", "ignored-session", "mini-id", "plato-mini", history);
    await clearSession("ignored-session", "plato");
    expect(apiRequest.mock.calls.map(([endpoint]) => endpoint)).toEqual([
      "main/plato/status",
      "main/plato/models/plato/chat",
      "main/plato/models/plato-mini/chat",
    ]);
    expect(apiRequest.mock.calls[1][2]).toEqual({
      messages: [{ role: "user", content: "hello" }],
      max_new_tokens: 128,
    });
  });

  it("rejects malformed or mismatched model replies", async () => {
    apiRequest.mockResolvedValueOnce({ reply: "wrong model", model: "plato" });
    await expect(
      chat("hi", "unused", "bad-id", "plato-mini", [row("user", "hi")]),
    ).rejects.toThrow("invalid response");
    apiRequest.mockResolvedValueOnce({ error: "oops" });
    await expect(chat("hi", "session", "bad-public", "public")).rejects.toThrow(
      "invalid response",
    );
  });
});

function row(role: ChatMessage["role"], content: string): ChatMessage {
  return {
    id: crypto.randomUUID(),
    role,
    content,
    createdAt: new Date().toISOString(),
  };
}

describe("Plato history bounds", () => {
  it("omits errors and trims whole oldest turns", () => {
    const history: ChatMessage[] = [];
    for (let i = 0; i < 8; i++) {
      history.push(
        row("user", `${i}`.repeat(350)),
        row("assistant", `answer-${i}`),
      );
      history.push(row("error", "offline"));
    }
    history.push(row("user", "latest"));
    const messages = buildPlatoMessages(history);
    expect(messages.at(-1)).toEqual({ role: "user", content: "latest" });
    expect(messages.length).toBeLessThanOrEqual(12);
    expect(
      messages.reduce((n, item) => n + item.content.length, 0),
    ).toBeLessThanOrEqual(4000);
    expect(messages[0].role).toBe("user");
    expect(messages.every((item) => item.content !== "offline")).toBe(true);
  });
  it("rejects a history without a final user turn", () => {
    expect(() =>
      buildPlatoMessages([row("user", "hi"), row("assistant", "hello")]),
    ).toThrow();
  });
});
