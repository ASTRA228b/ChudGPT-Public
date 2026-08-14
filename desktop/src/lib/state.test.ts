import { describe, expect, it } from "vitest";
import {
  createConversation,
  normalizeState,
  searchConversations,
  titleFromMessage,
  visibleMessages,
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

  it("rejects an unknown imported theme", () => {
    const normalized = normalizeState({ settings: { theme: "broken-theme" } });
    expect(normalized.settings.theme).toBe("neon");
  });

  it("repairs unsafe imported performance settings", () => {
    const normalized = normalizeState({
      settings: {
        interfaceScale: 999,
        glowIntensity: -40,
        density: "broken",
        statusPollSeconds: 7,
        renderMessageLimit: 3,
        contentWidth: 999,
        sidebarWidth: 900,
        composerFontSize: 2,
      },
    });
    expect(normalized.settings.interfaceScale).toBe(125);
    expect(normalized.settings.glowIntensity).toBe(0);
    expect(normalized.settings.density).toBe("comfortable");
    expect(normalized.settings.statusPollSeconds).toBe(60);
    expect(normalized.settings.renderMessageLimit).toBe(250);
    expect(normalized.settings.contentWidth).toBe(880);
    expect(normalized.settings.sidebarWidth).toBe(360);
    expect(normalized.settings.composerFontSize).toBe(12);
  });

  it("limits rendering without deleting chat history", () => {
    const messages = Array.from({ length: 600 }, (_, index) => index);
    expect(visibleMessages(messages, 250)).toEqual(messages.slice(-250));
    expect(visibleMessages(messages, 0)).toBe(messages);
  });
});
