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

## Security

If the token is ever exposed, reset it immediately in the Discord Developer
Portal. Do not enable Administrator permission; this bot does not need it.
