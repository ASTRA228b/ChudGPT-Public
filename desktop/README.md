# ChudGPT Desktop

ChudGPT Desktop is the official native client for the independently trained ChudGPT-Public model. It is an Electron, React, and TypeScript application—not a wrapper around the website. The native main process owns API networking, file dialogs, local storage, window controls, safe external links, and startup integration; the isolated renderer implements the desktop interface.

## Features

- Frameless, resizable, high-DPI desktop window with native minimize, maximize, and close controls
- Persistent multi-chat library with one API `session_id` per conversation
- Instant local search across titles and messages
- New, switch, rename, delete, and clear-all chat workflows
- Markdown, multiline answers, selectable text, fenced code blocks, and copy actions
- Stop/cancel requests, duplicate-send prevention, timeouts, status checks, and offline-safe error messages
- Local JSON export/import with validation and a 10 MB import safety limit
- Twenty-three themes (the original three plus twenty new choices), from Synthwave Arcade and Galaxy Brain to Retro Terminal and Maximum CHUD; scale, glow, density, and reduced-animation controls
- Keyboard shortcuts: `Ctrl+N`, `Ctrl+K`, `Ctrl+,`, and `Ctrl+L`
- First-launch explanation and experimental-model warning
- No analytics, tracking, accounts, API keys, or code execution

Chats remain in Electron's local application-data store. Only messages sent for generation leave the computer and go to `https://chudgpt-public.vercel.app/api/chat`.

## Architecture

```text
desktop/
├── electron/       Native main process and context-isolated preload bridge
├── src/
│   ├── components/ Reusable interface components
│   ├── lib/        API and validated persistence helpers
│   └── App.tsx     Application state and workflows
├── assets/         ChudGPT application icons
└── release/        Generated packages (ignored by Git)
```

The renderer has no Node.js access. API requests cross a narrow IPC bridge and are performed by the main process. External navigation is denied unless the exact address is in the allowlist. Model-generated code is rendered strictly as text and is never executed.

## Development

Requirements: Node.js 20 or newer and npm 10 or newer.

```powershell
cd desktop
npm install
npm run dev
```

Verification:

```powershell
npm run format:check
npm run lint
npm run typecheck
npm test
npm run build
```

## Production packages

Windows (locally supported):

```powershell
npm run package:win
```

This produces an NSIS installer and portable executable under `desktop/release/`.

Tagged releases use `.github/workflows/desktop-release.yml` to build independently on native runners:

- Windows: NSIS `.exe` installer and portable `.exe`
- macOS: `.dmg` and `.zip`
- Linux: portable `.AppImage`

macOS artifacts are unsigned unless Apple signing/notarization credentials are configured. Windows packages are unsigned unless a code-signing certificate is configured. Linux packages do not require signing for local installation.

Release tags use the form `desktop-v0.1.0`.

## Availability and privacy

The app is a client, not a bundled model. The owner-hosted inference service must be online. If it is unavailable, existing local chats remain intact and the interface offers a clear error rather than freezing.

**ChudGPT-Public is an experimental small language model. Responses may be inaccurate, inconsistent, or incorrect. Do not rely on it for important decisions.**

Supported release targets are Windows 10/11 x64, recent macOS releases on the architecture produced by the native GitHub runner, and common x64 Linux distributions capable of running AppImage or Debian packages.
