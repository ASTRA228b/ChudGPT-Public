export type Role = "user" | "assistant" | "error";
export type ThemeName = "neon" | "midnight" | "black";
export type Density = "comfortable" | "compact";

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  createdAt: string;
}

export interface Conversation {
  id: string;
  sessionId: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
}

export interface AppSettings {
  theme: ThemeName;
  interfaceScale: number;
  compactSidebar: boolean;
  reduceAnimations: boolean;
  glowIntensity: number;
  density: Density;
  sendWithEnter: boolean;
  showTimestamps: boolean;
  autoScroll: boolean;
  syntaxHighlighting: boolean;
  autoTitles: boolean;
  startOnNewChat: boolean;
  rememberLastChat: boolean;
  confirmDeletes: boolean;
  launchAtLogin: boolean;
}

export interface PersistedState {
  version: 1;
  conversations: Conversation[];
  activeConversationId: string | null;
  settings: AppSettings;
  onboardingComplete: boolean;
}

export interface ApiStatus {
  ready: boolean;
  model: string;
  device: string;
  parameters: number;
  context_length: number;
  step: number;
}

export interface ChatResponse {
  reply: string;
  session_id: string;
  step: number;
}

export type ConnectionState = "connecting" | "online" | "offline";
