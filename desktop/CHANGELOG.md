# ChudGPT Desktop changelog

## v0.5.1 - 2026-10-17

- Replaced the ambiguous native chat-action dialogs with separate Rename and Delete controls.
- Renamed chats through an inline editor that supports Save, Cancel, and keyboard input.
- Deleting an active chat now cancels its pending request, clears the draft, and restores the composer.
- Deleting all chats or importing a replacement also cancels any pending request before changing the chat list.

## v0.5.0 - 2026-10-17

- Added Plato and Plato Mini to the model selector, for 14 published profiles across Public, Music, main, and Plato.
- Kept each conversation pinned to its chosen model, including migrated local chats; switching models opens a fresh chat.
- Routed Plato requests through its stateless API with bounded turns and validated neural responses.
- Fixed stale status checks after model switches, duplicate sends, late cancelled replies, import size checks, and ordered local saves.
- Updated the interface and release notes to describe all model families accurately.

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
