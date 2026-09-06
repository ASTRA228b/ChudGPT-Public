import type { ApiStatus, ChatResponse, ModelProfile } from "../types";
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
  // Keep the desktop compatible with the stable Public Vercel routes. The
  // canonical /api/models/* aliases are exposed by the CUDA server, but older
  // Vercel deployments only publish /api/chat and /api/music/*.
  if (profile === "music") return `music/${action}`;
  return action;
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

export function chat(
  message: string,
  sessionId: string,
  requestId: string,
  profile: ModelProfile,
): Promise<ChatResponse> {
  return window.chudDesktop.apiRequest<ChatResponse>(
    endpointFor(profile, "chat"),
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
    endpointFor(profile, "clear"),
    "POST",
    { session_id: sessionId },
    makeRequestId(),
  );
}

export function cancel(requestId: string): Promise<boolean> {
  return window.chudDesktop.cancelRequest(requestId);
}
