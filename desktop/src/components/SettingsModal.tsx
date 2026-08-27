import {
  Download,
  ExternalLink,
  RotateCcw,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import type { AppSettings, PersistedState, ThemeName } from "../types";

interface Props {
  settings: AppSettings;
  state: PersistedState;
  version: string;
  onChange: (settings: AppSettings) => void;
  onClose: () => void;
  onExport: () => void;
  onImport: () => void;
  onDeleteAll: () => void;
  onReset: () => void;
  onToast: (message: string) => void;
}

const Switch = ({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
}) => (
  <button
    className={`switch ${checked ? "on" : ""}`}
    role="switch"
    aria-checked={checked}
    onClick={() => onChange(!checked)}
  >
    <span />
  </button>
);

export function SettingsModal(props: Props): JSX.Element {
  const set = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) =>
    props.onChange({ ...props.settings, [key]: value });
  const open = (url: string) =>
    void window.chudDesktop
      .openExternal(url)
      .catch(() => props.onToast("Could not open that link"));
  return (
    <div
      className="modal-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) props.onClose();
      }}
    >
      <section
        className="settings-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Settings"
      >
        <header>
          <div>
            <span className="eyebrow">CONTROL PANEL</span>
            <h2>Settings</h2>
          </div>
          <button className="icon-button" onClick={props.onClose}>
            <X />
          </button>
        </header>
        <div className="settings-scroll">
          <SettingsSection title="General">
            <Setting label="Model profile">
              <select
                value={props.settings.modelProfile}
                onChange={(event) =>
                  set(
                    "modelProfile",
                    event.target.value as AppSettings["modelProfile"],
                  )
                }
              >
                <option value="public">ChudGPT-Public V20</option>
                <option value="music">ChudGPT-Public-Music V1</option>
              </select>
            </Setting>
            <Setting label="Launch when I sign in">
              <Switch
                checked={props.settings.launchAtLogin}
                onChange={(value) => {
                  set("launchAtLogin", value);
                  void window.chudDesktop.setLaunchAtLogin(value);
                }}
              />
            </Setting>
            <Setting label="Start on New Chat">
              <Switch
                checked={props.settings.startOnNewChat}
                onChange={(value) => set("startOnNewChat", value)}
              />
            </Setting>
            <Setting label="Remember last conversation">
              <Switch
                checked={props.settings.rememberLastChat}
                onChange={(value) => set("rememberLastChat", value)}
              />
            </Setting>
            <Setting label="Confirm before deleting">
              <Switch
                checked={props.settings.confirmDeletes}
                onChange={(value) => set("confirmDeletes", value)}
              />
            </Setting>
          </SettingsSection>
          <SettingsSection title="Appearance">
            <Setting label="Theme">
              <select
                value={props.settings.theme}
                onChange={(event) =>
                  set("theme", event.target.value as ThemeName)
                }
              >
                <option value="neon">Neon Dark</option>
                <option value="midnight">Midnight</option>
                <option value="black">Pure Black</option>
                <option value="synthwave">Synthwave Arcade</option>
                <option value="forest">Deep Forest</option>
                <option value="ember">Ember Core</option>
                <option value="arctic">Arctic Signal</option>
                <option value="sunset">Solar Sunset</option>
                <option value="toxic">Toxic Terminal</option>
                <option value="royal">Royal Void</option>
                <option value="rose">Rose Circuit</option>
                <option value="terminal">Retro Terminal</option>
                <option value="chud">Maximum CHUD</option>
                <option value="deepsea">Deep Sea</option>
                <option value="sandstorm">Sandstorm</option>
                <option value="bubblegum">Bubblegum Glitch</option>
                <option value="copper">Copper Machine</option>
                <option value="galaxy">Galaxy Brain</option>
                <option value="lavender">Lavender Night</option>
                <option value="bloodmoon">Blood Moon</option>
                <option value="cyberyellow">Cyber Yellow</option>
                <option value="slate">Steel Slate</option>
                <option value="mint">Mint Condition</option>
              </select>
            </Setting>
            <Setting
              label={`Interface scale · ${props.settings.interfaceScale}%`}
            >
              <input
                type="range"
                min="85"
                max="125"
                step="5"
                value={props.settings.interfaceScale}
                onChange={(event) =>
                  set("interfaceScale", Number(event.target.value))
                }
              />
            </Setting>
            <Setting
              label={`Glow intensity · ${props.settings.glowIntensity}%`}
            >
              <input
                type="range"
                min="0"
                max="100"
                step="5"
                value={props.settings.glowIntensity}
                onChange={(event) =>
                  set("glowIntensity", Number(event.target.value))
                }
              />
            </Setting>
            <Setting label="Compact sidebar">
              <Switch
                checked={props.settings.compactSidebar}
                onChange={(value) => set("compactSidebar", value)}
              />
            </Setting>
            <Setting label="Reduce animations">
              <Switch
                checked={props.settings.reduceAnimations}
                onChange={(value) => set("reduceAnimations", value)}
              />
            </Setting>
            <Setting label="Message density">
              <select
                value={props.settings.density}
                onChange={(event) =>
                  set("density", event.target.value as AppSettings["density"])
                }
              >
                <option value="comfortable">Comfortable</option>
                <option value="compact">Compact</option>
              </select>
            </Setting>
            <Setting label="Chat content width">
              <select
                value={props.settings.contentWidth}
                onChange={(event) =>
                  set(
                    "contentWidth",
                    Number(event.target.value) as AppSettings["contentWidth"],
                  )
                }
              >
                <option value="720">Narrow · 720px</option>
                <option value="880">Balanced · 880px</option>
                <option value="1080">Wide · 1080px</option>
                <option value="1400">Extra wide · 1400px</option>
              </select>
            </Setting>
            <Setting label={`Sidebar width · ${props.settings.sidebarWidth}px`}>
              <input
                type="range"
                min="220"
                max="360"
                step="4"
                value={props.settings.sidebarWidth}
                onChange={(event) =>
                  set("sidebarWidth", Number(event.target.value))
                }
              />
            </Setting>
            <Setting
              label={`Composer text · ${props.settings.composerFontSize}px`}
            >
              <input
                type="range"
                min="12"
                max="18"
                step="1"
                value={props.settings.composerFontSize}
                onChange={(event) =>
                  set("composerFontSize", Number(event.target.value))
                }
              />
            </Setting>
          </SettingsSection>
          <SettingsSection title="Performance">
            <Setting label="Performance mode">
              <Switch
                checked={props.settings.performanceMode}
                onChange={(value) => set("performanceMode", value)}
              />
            </Setting>
            <Setting label="Server status checks">
              <select
                value={props.settings.statusPollSeconds}
                onChange={(event) =>
                  set(
                    "statusPollSeconds",
                    Number(
                      event.target.value,
                    ) as AppSettings["statusPollSeconds"],
                  )
                }
              >
                <option value="0">Manual only</option>
                <option value="30">Every 30 seconds</option>
                <option value="60">Every minute</option>
                <option value="120">Every 2 minutes</option>
                <option value="300">Every 5 minutes</option>
              </select>
            </Setting>
            <Setting label="Messages rendered per chat">
              <select
                value={props.settings.renderMessageLimit}
                onChange={(event) =>
                  set(
                    "renderMessageLimit",
                    Number(
                      event.target.value,
                    ) as AppSettings["renderMessageLimit"],
                  )
                }
              >
                <option value="100">Latest 100</option>
                <option value="250">Latest 250</option>
                <option value="500">Latest 500</option>
                <option value="0">All messages</option>
              </select>
            </Setting>
            <p className="privacy-note">
              Performance mode reduces glow, blur, and animation work. Render
              limits only affect what is currently drawn; complete chats remain
              saved and exportable.
            </p>
          </SettingsSection>
          <SettingsSection title="Chat">
            <Setting label="Send with Enter">
              <Switch
                checked={props.settings.sendWithEnter}
                onChange={(value) => set("sendWithEnter", value)}
              />
            </Setting>
            <Setting label="Show timestamps">
              <Switch
                checked={props.settings.showTimestamps}
                onChange={(value) => set("showTimestamps", value)}
              />
            </Setting>
            <Setting label="Auto-scroll">
              <Switch
                checked={props.settings.autoScroll}
                onChange={(value) => set("autoScroll", value)}
              />
            </Setting>
            <Setting label="Code syntax highlighting">
              <Switch
                checked={props.settings.syntaxHighlighting}
                onChange={(value) => set("syntaxHighlighting", value)}
              />
            </Setting>
            <Setting label="Automatic chat titles">
              <Switch
                checked={props.settings.autoTitles}
                onChange={(value) => set("autoTitles", value)}
              />
            </Setting>
          </SettingsSection>
          <SettingsSection title="Data">
            <div className="action-grid">
              <button onClick={props.onExport}>
                <Download /> Export chats
              </button>
              <button onClick={props.onImport}>
                <Upload /> Import chats
              </button>
              <button className="danger" onClick={props.onDeleteAll}>
                <Trash2 /> Delete all chats
              </button>
              <button onClick={props.onReset}>
                <RotateCcw /> Reset settings
              </button>
            </div>
            <p className="privacy-note">
              Chats are stored locally. Only prompts required for generation are
              sent to ChudGPT-Public.
            </p>
          </SettingsSection>
          <SettingsSection title="About">
            <div className="about-card">
              <h3>
                ChudGPT Desktop <span>v{props.version}</span>
              </h3>
              <p>
                Powered by ChudGPT-Public, an independently trained experimental
                small language model.
              </p>
              <p className="warning-text">
                Responses may be inaccurate, inconsistent, or incorrect. Do not
                rely on them for important decisions.
              </p>
              <div className="link-row">
                <button
                  onClick={() => open("https://chudgpt-landing.vercel.app/")}
                >
                  Website <ExternalLink />
                </button>
                <button
                  onClick={() => open("https://chudgpt-public.vercel.app/")}
                >
                  Web client <ExternalLink />
                </button>
                <button
                  onClick={() =>
                    open("https://github.com/ASTRA228b/ChudGPT-Public")
                  }
                >
                  GitHub <ExternalLink />
                </button>
              </div>
            </div>
          </SettingsSection>
        </div>
      </section>
    </div>
  );
}

function SettingsSection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="settings-section">
      <h3>{title}</h3>
      {children}
    </section>
  );
}
function Setting({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="setting-row">
      <span>{label}</span>
      {children}
    </div>
  );
}
