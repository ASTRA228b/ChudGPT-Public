import type { ApiStatus, ChatResponse, ModelProfile } from "../types";
import { modelProfileInfo } from "./models";

export const makeRequestId = (): string => crypto.randomUUID();

export function status(
  profile: ModelProfile,
  requestId = makeRequestId(),
): Promise<ApiStatus> {
  const model = modelProfileInfo(profile);
  return window.chudDesktop.apiRequest<ApiStatus>(
    `${model.family === "main" ? "main/" : ""}models/${profile}`,
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
  const model = modelProfileInfo(profile);
  return window.chudDesktop.apiRequest<ChatResponse>(
    `${model.family === "main" ? "main/" : ""}models/${profile}/chat`,
    "POST",
    { message, session_id: sessionId },
    requestId,
  );
}

export function clearSession(
  sessionId: string,
  profile: ModelProfile,
): Promise<{ cleared: boolean }> {
  const model = modelProfileInfo(profile);
  return window.chudDesktop.apiRequest(
    `${model.family === "main" ? "main/" : ""}models/${profile}/clear`,
    "POST",
    { session_id: sessionId },
    makeRequestId(),
  );
}

export function cancel(requestId: string): Promise<boolean> {
  return window.chudDesktop.cancelRequest(requestId);
}
