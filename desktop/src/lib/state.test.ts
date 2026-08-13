import { describe, expect, it } from "vitest";
import {
  createConversation,
  normalizeState,
  searchConversations,
  titleFromMessage,
} from "./state";

describe("desktop state", () => {
  it("creates isolated API sessions", () => {
    const first = createConversation();
    const second = createConversation();
    expect(first.id).not.toBe(second.id);
    expect(first.sessionId).not.toBe(second.sessionId);
  });

  it("makes local titles without an API request", () => {
    expect(titleFromMessage("  Unity   movement script  ")).toBe(
      "Unity movement script",
    );
    expect(titleFromMessage("x".repeat(60)).length).toBeLessThanOrEqual(42);
  });

  it("searches title and message content", () => {
    const chat = createConversation();
    chat.title = "Random questions";
    chat.messages.push({
      id: "m",
      role: "user",
      content: "Explain gravity",
      createdAt: new Date().toISOString(),
    });
    expect(searchConversations([chat], "gravity")).toHaveLength(1);
    expect(searchConversations([chat], "random")).toHaveLength(1);
    expect(searchConversations([chat], "missing")).toHaveLength(0);
  });

  it("safely normalizes malformed imports", () => {
    const normalized = normalizeState({
      conversations: [{ nope: true }],
      settings: { theme: "black" },
    });
    expect(normalized.conversations).toEqual([]);
    expect(normalized.settings.theme).toBe("black");
    expect(normalized.version).toBe(1);
  });
});
