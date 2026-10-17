import type {
  ApiStatus,
  ChatMessage,
  ChatResponse,
  ModelProfile,
} from "../types";
import { modelProfileInfo } from "./models";

export const makeRequestId = (): string => crypto.randomUUID();

function endpointFor(
  profile: ModelProfile,
  action: "status" | "chat" | "clear",
): string {
  const model = modelProfileInfo(profile);
  if (model.family === "main") {
    return `main/models/${profile}${action === "status" ? "" : `/${action}`}`;
  }
  if (model.family === "plato") {
    if (action === "clear")
      throw new Error("Plato conversations are stateless.");
    return action === "status"
      ? "main/plato/status"
      : `main/plato/models/${profile}/chat`;
  }
  // Keep the desktop compatible with the stable Public Vercel routes. The
  // canonical /api/models/* aliases are exposed by the CUDA server, but older
  // Vercel deployments only publish /api/chat and /api/music/*.
  if (profile === "music") return `music/${action}`;
  return action;
}

export function buildPlatoMessages(
  history: ChatMessage[],
): { role: "user" | "assistant"; content: string }[] {
  const turns: { role: "user" | "assistant"; content: string }[][] = [];
  for (const row of history) {
    if (row.role === "error" || !row.content.trim()) continue;
    if (row.role === "user")
      turns.push([{ role: "user", content: row.content }]);
    else if (turns.length && turns[turns.length - 1].length === 1)
      turns[turns.length - 1].push({ role: "assistant", content: row.content });
  }
  if (!turns.length) throw new Error("A user message is required.");
  // The pending final turn must end with the latest user message.
  const pending = turns[turns.length - 1];
  if (pending.length !== 1)
    throw new Error("The latest message must be from the user.");
  if (pending[0].content.length > 4000)
    throw new Error("Message exceeds 4,000 characters.");
  while (turns.length > 1) {
    const flat = turns.flat();
    if (
      flat.length <= 12 &&
      flat.reduce((n, item) => n + item.content.length, 0) <= 4000
    )
      break;
    turns.shift();
  }
  return turns.flat();
}

export function status(
  profile: ModelProfile,
  requestId = makeRequestId(),
): Promise<ApiStatus> {
  return window.chudDesktop.apiRequest<ApiStatus>(
    endpointFor(profile, "status"),
    "GET",
    null,
    requestId,
  );
}

export async function chat(
  message: string,
  sessionId: string,
  requestId: string,
  profile: ModelProfile,
  history?: ChatMessage[],
): Promise<ChatResponse> {
  if (modelProfileInfo(profile).family === "plato") {
    if (!history) throw new Error("Plato requires local conversation history.");
    const response = await window.chudDesktop.apiRequest<ChatResponse>(
      endpointFor(profile, "chat"),
      "POST",
      { messages: buildPlatoMessages(history), max_new_tokens: 128 },
      requestId,
    );
    if (
      typeof response?.reply !== "string" ||
      response.model !== profile ||
      response.source !== "neural" ||
      typeof response.checkpoint_step !== "number"
    )
      throw new Error("Plato returned an invalid response.");
    return response;
  }
  const response = await window.chudDesktop.apiRequest<ChatResponse>(
    endpointFor(profile, "chat"),
    "POST",
    { message, session_id: sessionId },
    requestId,
  );
  if (typeof response?.reply !== "string")
    throw new Error("The selected model returned an invalid response.");
  return response;
}

export function clearSession(
  sessionId: string,
  profile: ModelProfile,
): Promise<{ cleared: boolean }> {
  if (modelProfileInfo(profile).family === "plato")
    return Promise.resolve({ cleared: true });
  return window.chudDesktop.apiRequest(
    endpointFor(profile, "clear"),
    "POST",
    { session_id: sessionId },
    makeRequestId(),
  );
}

export function cancel(requestId: string): Promise<boolean> {
  return window.chudDesktop.cancelRequest(requestId);
}
