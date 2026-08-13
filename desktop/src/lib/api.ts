import type { ApiStatus, ChatResponse } from "../types";

export const makeRequestId = (): string => crypto.randomUUID();

export function status(requestId = makeRequestId()): Promise<ApiStatus> {
  return window.chudDesktop.apiRequest<ApiStatus>(
    "status",
    "GET",
    null,
    requestId,
  );
}

export function chat(
  message: string,
  sessionId: string,
  requestId: string,
): Promise<ChatResponse> {
  return window.chudDesktop.apiRequest<ChatResponse>(
    "chat",
    "POST",
    { message, session_id: sessionId },
    requestId,
  );
}

export function clearSession(sessionId: string): Promise<{ cleared: boolean }> {
  return window.chudDesktop.apiRequest(
    "clear",
    "POST",
    { session_id: sessionId },
    makeRequestId(),
  );
}

export function cancel(requestId: string): Promise<boolean> {
  return window.chudDesktop.cancelRequest(requestId);
}
