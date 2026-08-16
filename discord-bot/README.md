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

## Local setup on Windows

1. Create a Discord application and bot in the Discord Developer Portal.
2. Enable **Message Content Intent** on the bot's settings page.
3. Invite it with the `bot` scope and these permissions:
   **View Channels**, **Send Messages**, **Read Message History**, and
   **Embed Links**.
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
!chud help       Show all bot commands
!chud status     Show bot, API, and server status
!chud about      Show the fixed Public V20 model profile
!chud whoami     Show the Discord name and roles visible to the bot
!chud privacy    Explain the private quality-improvement logs
!chud languages  List built-in basic greeting languages
!chud server     Show the current server or DM location
!chud roles      Show your visible server roles
!chud developer  Show who created ChudGPT
!chud language Spanish                 Translate this conversation to/from Spanish
!chud language auto                    Auto-detect non-English messages
!chud language off                     Disable translation for this conversation
!chud translate Japanese hello there   Translate one message
!chud translation status               Show whether Google translation is configured
!chud ping       Test whether the bot is responding
```

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
