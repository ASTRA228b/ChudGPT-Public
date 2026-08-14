import type {
  AppSettings,
  Conversation,
  PersistedState,
  ThemeName,
} from "../types";

export const themeNames: readonly ThemeName[] = [
  "neon",
  "midnight",
  "black",
  "synthwave",
  "forest",
  "ember",
  "arctic",
  "sunset",
  "toxic",
  "royal",
  "rose",
  "terminal",
  "chud",
  "deepsea",
  "sandstorm",
  "bubblegum",
  "copper",
  "galaxy",
  "lavender",
  "bloodmoon",
  "cyberyellow",
  "slate",
  "mint",
];

export const defaultSettings: AppSettings = {
  theme: "neon",
  interfaceScale: 100,
  compactSidebar: false,
  reduceAnimations: false,
  performanceMode: false,
  glowIntensity: 65,
  density: "comfortable",
  sendWithEnter: true,
  showTimestamps: false,
  autoScroll: true,
  syntaxHighlighting: true,
  autoTitles: true,
  startOnNewChat: false,
  rememberLastChat: true,
  confirmDeletes: true,
  launchAtLogin: false,
  statusPollSeconds: 60,
  renderMessageLimit: 250,
  contentWidth: 880,
  sidebarWidth: 268,
  composerFontSize: 14,
};

export const emptyState = (): PersistedState => ({
  version: 1,
  conversations: [],
  activeConversationId: null,
  settings: { ...defaultSettings },
  onboardingComplete: false,
});

export function isConversation(value: unknown): value is Conversation {
  if (!value || typeof value !== "object") return false;
  const chat = value as Partial<Conversation>;
  return (
    typeof chat.id === "string" &&
    typeof chat.sessionId === "string" &&
    typeof chat.title === "string" &&
    Array.isArray(chat.messages) &&
    chat.messages.every(
      (message) =>
        message &&
        typeof message.id === "string" &&
        ["user", "assistant", "error"].includes(message.role) &&
        typeof message.content === "string" &&
        typeof message.createdAt === "string",
    )
  );
}

export function normalizeState(value: unknown): PersistedState {
  const fallback = emptyState();
  if (!value || typeof value !== "object") return fallback;
  const candidate = value as Partial<PersistedState>;
  const conversations = Array.isArray(candidate.conversations)
    ? candidate.conversations.filter(isConversation).slice(0, 1_000)
    : [];
  const settings = { ...defaultSettings, ...(candidate.settings ?? {}) };
  if (!themeNames.includes(settings.theme))
    settings.theme = defaultSettings.theme;
  if (!["comfortable", "compact"].includes(settings.density))
    settings.density = defaultSettings.density;
  if (![0, 30, 60, 120, 300].includes(settings.statusPollSeconds))
    settings.statusPollSeconds = defaultSettings.statusPollSeconds;
  if (![0, 100, 250, 500].includes(settings.renderMessageLimit))
    settings.renderMessageLimit = defaultSettings.renderMessageLimit;
  if (![720, 880, 1080, 1400].includes(settings.contentWidth))
    settings.contentWidth = defaultSettings.contentWidth;
  settings.sidebarWidth = Math.min(
    360,
    Math.max(220, Number(settings.sidebarWidth) || 268),
  );
  settings.composerFontSize = Math.min(
    18,
    Math.max(12, Number(settings.composerFontSize) || 14),
  );
  settings.interfaceScale = Math.min(
    125,
    Math.max(85, Number(settings.interfaceScale) || 100),
  );
  settings.glowIntensity = Math.min(
    100,
    Math.max(0, Number(settings.glowIntensity) || 0),
  );
  const active = conversations.some(
    (chat) => chat.id === candidate.activeConversationId,
  )
    ? (candidate.activeConversationId ?? null)
    : null;
  return {
    version: 1,
    conversations,
    activeConversationId: active,
    settings,
    onboardingComplete: Boolean(candidate.onboardingComplete),
  };
}

export function createConversation(): Conversation {
  const now = new Date().toISOString();
  return {
    id: crypto.randomUUID(),
    sessionId: `desktop-${crypto.randomUUID()}`,
    title: "New conversation",
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

export function titleFromMessage(message: string): string {
  const clean = message.replace(/\s+/g, " ").trim();
  if (!clean) return "New conversation";
  return clean.length <= 42 ? clean : `${clean.slice(0, 39).trim()}…`;
}

export function searchConversations(
  conversations: Conversation[],
  query: string,
): Conversation[] {
  const needle = query.trim().toLocaleLowerCase();
  if (!needle) return conversations;
  return conversations.filter(
    (chat) =>
      chat.title.toLocaleLowerCase().includes(needle) ||
      chat.messages.some((message) =>
        message.content.toLocaleLowerCase().includes(needle),
      ),
  );
}

export function groupDate(date: string): "Today" | "Previous" {
  const value = new Date(date);
  const today = new Date();
  return value.toDateString() === today.toDateString() ? "Today" : "Previous";
}

export function visibleMessages<T>(messages: T[], limit: number): T[] {
  if (!limit || messages.length <= limit) return messages;
  return messages.slice(-limit);
}
