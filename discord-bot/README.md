# ChudGPT-Public Discord Bot

A Flask health service plus a Discord bot powered by the public ChudGPT API.
The bot responds in direct messages, when mentioned, or when a message starts
with `!chud`. Each Discord user and channel gets an isolated ChudGPT session.

## Features

- Uses `https://chudgpt-public.vercel.app/api/chat` by default.
- No ChudGPT API key is required.
- Keeps Discord conversations isolated by server, channel, and user.
- Includes per-user rate limiting, typing indicators, safe message splitting,
  clear API errors, and Flask health endpoints.
- Flask endpoints: `/`, `/health`, and `/status`.
- Google translation with automatic source detection and caching is isolated
  to the Discord bot. It works in keyless compatible mode by default and uses
  the official Cloud Translation v2 API when a key is configured.
- Never stores the Discord token in source control.
- Includes an owner-only Discord voice soundboard with a localhost control panel,
  upload chooser, sound list, stop control, and master-volume slider.

## Local setup on Windows

1. Create a Discord application and bot in the Discord Developer Portal.
2. Enable **Message Content Intent** on the bot's settings page.
3. Invite it with the `bot` scope and these permissions:
   **View Channels**, **Send Messages**, **Read Message History**, and
   **Embed Links**, **Connect**, and **Speak**.
4. Open Command Prompt and run:

```bat
cd /d C:\Users\brian\OneDrive\Documents\ChudGPT\ChudGPT-Public\discord-bot
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
notepad .env
```

5. Put the Discord bot token after `DISCORD_TOKEN=` in `.env`. Never post that
   token or commit `.env`.
6. Start the bot:

```bat
.venv\Scripts\python.exe bot.py
```

7. Test `http://127.0.0.1:8080/health`, then DM the bot or type:

```text
!chud Hello! What can you do?
```

Clear your private conversation session in the current channel with:

```text
!chud clear
```

Other built-in commands:

```text
!chud help       Open the interactive command-page menu
!chud help 2     Discord identity and location commands
!chud help 3     Language and translation commands
!chud help 4     Information and utility commands
!chud status     Show bot, API, and server status
!chud about      Show the fixed Public V20 model profile
!chud whoami     Show the Discord name and roles visible to the bot
!chud userid     Show your Discord user ID
!chud channel    Show the current channel
!chud privacy    Explain the private quality-improvement logs
!chud languages  List built-in basic greeting languages
!chud server     Show the current server or DM location
!chud roles      Show your visible server roles
!chud developer  Show who created ChudGPT
!chud source     Show the public source repository
!chud capabilities  Summarize supported tasks
!chud gtag       Show Gorilla Tag knowledge
!chud invite     Explain who can invite the bot
!chud language Spanish                 Translate this conversation to/from Spanish
!chud language auto                    Auto-detect non-English messages
!chud language off                     Disable translation for this conversation
!chud translate Japanese hello there   Translate one message
!chud translation status               Show whether Google translation is configured
!chud ping       Test whether the bot is responding
!chud soundboard enable       Owner only: connect to your current voice channel
!chud join                    Owner only: direct alias for soundboard enable
!chud leave                   Owner only: leave voice and disable soundboard
!chud soundboard status       Owner only: show state and the local panel URL
!chud soundboard volume 65    Owner only: set volume from 0 to 100
!chud soundboard play file.mp3
!chud soundboard stop
!chud soundboard disable
!chud ADMIN-HELP               Owner-only interactive admin command menu
```

## Server administration

Uppercase `!chud SERVER` displays a separate administration menu. It is not
listed in normal `!chud help`, and lowercase `!chud server` keeps showing the
current server/DM location. Server commands rely on Discord's actual guild
owner ID or **Administrator** permission; names, role names, and the soundboard
allowlist cannot grant access.

```text
!chud SERVER                    Show the restricted server command menu
!chud save channels             Save channel/category names and DM the .txt file
!chud delete all                Save + DM a backup, then request deletion confirmation
!chud rebuild server            Attach the saved .txt, then request rebuild confirmation
!chud purge all                 Request confirmation to purge the current channel
```

Destructive commands issue a random one-time code that expires after 60
seconds. Confirm using the exact command shown by the bot. Delete All is
cancelled if the safety snapshot cannot be DMed. Snapshots are created under
`D:\ChudGPT-Discord-Server-Files`, sent by DM, and then deleted from the host
whether delivery succeeds or fails. To rebuild, attach that DMed `.txt` to the
`!chud rebuild server` message. Set `CHUDGPT_SERVER_BACKUP_DIR` to choose a
different temporary host directory. The bot needs
**Manage Channels** for delete/rebuild and **Manage Messages** plus **Read
Message History** for purge. When first added to a server, it privately DMs the
server owner one short welcome message pointing to `!chud help` and uppercase
`!chud SERVER`; it does not automatically send either command list.

## Local soundboard

The Discord controls are locked to user IDs `1386115817325727854` and
`1324847616810422402` by default. Change the comma-separated
`CHUDGPT_SOUNDBOARD_ADMIN_IDS` variable to manage that allowlist. Server
administrators do not automatically gain soundboard access.

1. Install the updated Python requirements. The `discord.py[voice]` extra
   includes PyNaCl and current Discord DAVE voice-protocol support;
   `imageio-ffmpeg` supplies a local FFmpeg binary automatically.
2. A system [FFmpeg](https://ffmpeg.org/download.html) on PATH is also supported.
3. Restart the bot, join a voice channel, and run `!chud soundboard enable`.
4. On the bot host PC, open <http://127.0.0.1:8080/soundboard>.
5. Upload an MP3, WAV, OGG, M4A, FLAC, WebM, or AAC file, set the volume, and press Play.

The panel binds to localhost by default and cannot be reached from other devices.
Uploads are saved under `discord-bot\soundboard_audio` and are limited to 50 MB.

The help response includes **First**, **Previous**, **Next**, and **Last**
buttons. They edit the original help message instead of posting another copy.
Only the user who opened that help menu can control its buttons.

The clear command affects only that Discord user in that channel; it does not
erase another user's session or switch the model checkpoint.

## Hosting

This is a persistent Discord connection, so it must run on a long-lived host
such as your mini PC, Render, Railway, Fly.io, or another Docker host. A Vercel
serverless function cannot keep the Discord gateway connection alive.

For Docker:

```bat
docker build -t chudgpt-discord .
docker run --restart unless-stopped -p 8080:8080 --env-file .env chudgpt-discord
```

Configure your host with:

- Start command: `python bot.py`
- Health check path: `/health`
- Port: `8080` or the platform-provided `PORT`
- Secret: `DISCORD_TOKEN`
- Optional variable: `CHUDGPT_PUBLIC_API_URL=https://chudgpt-public.vercel.app/api`
- Optional secret: `GOOGLE_TRANSLATE_API_KEY` for Discord-only conversation translation

## Translation setup

No key is required for the compatible keyless mode. Restart the updated bot,
run `!chud translation status`, and then use `!chud language Spanish` or
`!chud language auto`.

For the supported Google Cloud API instead:

1. In Google Cloud, enable **Cloud Translation API** for your project. Billing
   and service quotas may apply.
2. Create an API key and restrict it to the Cloud Translation API. Restrict the
   key to the bot host's IP address when your hosting setup provides a stable IP.
3. Add the key to `discord-bot\.env`:

```text
GOOGLE_TRANSLATE_API_KEY=your-key-here
```

4. Restart the bot, then run `!chud translation status` and try
   `!chud language Spanish`. The key is used only by the Discord process; the
   Public website and Public API do not call Google Translate.

## Security

If the token is ever exposed, reset it immediately in the Discord Developer
Portal. Do not enable Administrator permission; this bot does not need it.
