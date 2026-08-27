import type { ApiStatus, ChatResponse, ModelProfile } from "../types";

export const makeRequestId = (): string => crypto.randomUUID();

export function status(
  profile: ModelProfile,
  requestId = makeRequestId(),
): Promise<ApiStatus> {
  return window.chudDesktop.apiRequest<ApiStatus>(
    profile === "music" ? "music/status" : "status",
    "GET",
    null,
    requestId,
  );
}

export function chat(
  message: string,
  sessionId: string,
  requestId: string,
  profile: ModelProfile,
): Promise<ChatResponse> {
  return window.chudDesktop.apiRequest<ChatResponse>(
    profile === "music" ? "music/chat" : "chat",
    "POST",
    { message, session_id: sessionId },
    requestId,
  );
}

export function clearSession(
  sessionId: string,
  profile: ModelProfile,
): Promise<{ cleared: boolean }> {
  return window.chudDesktop.apiRequest(
    profile === "music" ? "music/clear" : "clear",
    "POST",
    { session_id: sessionId },
    makeRequestId(),
  );
}

export function cancel(requestId: string): Promise<boolean> {
  return window.chudDesktop.cancelRequest(requestId);
}
