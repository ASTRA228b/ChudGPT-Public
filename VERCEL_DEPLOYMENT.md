# Deploy ChudGPT-Public on Vercel

Vercel hosts the website and HTTPS proxy. The 21M PyTorch model runs on your CUDA computer and is reached through a Cloudflare tunnel.

## Vercel project settings

- Import `ASTRA228b/ChudGPT-Public`.
- **Root Directory:** `web/static`
- **Framework Preset:** Other.
- Build Command: leave blank.
- Output Directory: leave blank.
- Install Command: leave blank.
- Add environment variable `CHUDGPT_BACKEND_URL` with the current tunnel origin, such as `https://example.trycloudflare.com` (no trailing slash).
- No API key is required. Do not add secrets to browser code.

Redeploy after changing an environment variable. Public endpoints are `GET /api/status`, `GET /api/info`, `POST /api/chat`, `POST /api/generate`, and `POST /api/clear`.

The repository currently includes the active quick-tunnel address as a temporary fallback. The environment variable takes priority and is still the recommended way to replace the address after a tunnel restart.

## Run the model backend

After training has produced `checkpoints/chat/best.pt`:

```cmd
cd /d C:\Users\brian\OneDrive\Documents\ChudGPT\ChudGPT-Public
C:\tmp\ChudGPT-venv\Scripts\python.exe public_api_server.py --device cuda --port 8010
```

In a second Command Prompt:

```cmd
cd /d C:\Users\brian\OneDrive\Documents\ChudGPT
tools\cloudflared.exe tunnel --url http://127.0.0.1:8010
```

Copy the printed `https://...trycloudflare.com` origin into `CHUDGPT_BACKEND_URL` in Vercel and redeploy.

## API example

```bash
curl -X POST https://YOUR-SITE.vercel.app/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"What is 7 + 5?\",\"session_id\":\"demo-user\"}"
```

In Windows Command Prompt, use one line instead of Linux backslashes:

```cmd
curl.exe -X POST "https://YOUR-SITE.vercel.app/api/chat" -H "Content-Type: application/json" -d "{\"message\":\"What is 7 + 5?\",\"session_id\":\"demo-user\"}"
```

`POST /api/generate` accepts the same JSON but does not retain a conversation. Both chat endpoints optionally accept `max_new_tokens` from 1–400 and `temperature` from 0–1.5.

Quick tunnels change address whenever restarted. For a stable production service, configure a named Cloudflare Tunnel and a hostname you own.
