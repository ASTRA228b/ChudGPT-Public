# ChudGPT Desktop changelog

## v0.2.1 - 2026-08-14

### Fixed

- New conversations now reliably focus the message composer after the interface renders, so typing works immediately.
- Switching conversations also restores composer focus without scrolling the chat unexpectedly.
- Cancelling a request no longer inserts a false network error or marks the service offline.
- A stale request can no longer clear the busy state or cancel control for a newer request.
- Desktop verification ignores unrelated local Python cache files.

### Optimization

- Added an optional performance mode that removes costly glow, blur, and transition effects.
- Added configurable background server checks: manual, 30 seconds, 1 minute, 2 minutes, or 5 minutes.
- Added a message-render limit of 100, 250, 500, or the full conversation to keep very long chats responsive.
- Long conversations display a clear notice when older messages are temporarily hidden by the render limit.

### Customization

- Added four chat-content widths: compact, comfortable, wide, and ultra-wide.
- Added adjustable sidebar width from 220 to 360 pixels.
- Added adjustable composer text size from 12 to 18 pixels.
- All new preferences are validated and saved locally with the existing desktop settings.
