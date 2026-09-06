export type Role = "user" | "assistant" | "error";
export type ThemeName =
  | "neon"
  | "midnight"
  | "black"
  | "synthwave"
  | "forest"
  | "ember"
  | "arctic"
  | "sunset"
  | "toxic"
  | "royal"
  | "rose"
  | "terminal"
  | "chud"
  | "deepsea"
  | "sandstorm"
  | "bubblegum"
  | "copper"
  | "galaxy"
  | "lavender"
  | "bloodmoon"
  | "cyberyellow"
  | "slate"
  | "mint";
export type Density = "comfortable" | "compact";
export type StatusPollSeconds = 0 | 30 | 60 | 120 | 300;
export type RenderMessageLimit = 0 | 100 | 250 | 500;
export type ContentWidth = 720 | 880 | 1080 | 1400;
export type ModelProfile =
  | "public"
  | "music"
  | "buggy"
  | "700"
  | "1300"
  | "1500"
  | "1600"
  | "ultimate"
  | "plus"
  | "pro"
  | "code"
  | "mega";

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
  modelProfile: ModelProfile;
  theme: ThemeName;
  interfaceScale: number;
  compactSidebar: boolean;
  reduceAnimations: boolean;
  performanceMode: boolean;
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
  statusPollSeconds: StatusPollSeconds;
  renderMessageLimit: RenderMessageLimit;
  contentWidth: ContentWidth;
  sidebarWidth: number;
  composerFontSize: number;
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
