import { beforeEach, describe, expect, it, vi } from "vitest";
import { chat, clearSession, status } from "./api";

const apiRequest = vi.fn();

beforeEach(() => {
  apiRequest.mockReset();
  apiRequest.mockResolvedValue({});
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
});
