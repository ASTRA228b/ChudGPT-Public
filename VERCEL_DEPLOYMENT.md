# Deploy ChudGPT-Public on Vercel

Vercel hosts the website and HTTPS proxy. The 21M PyTorch model runs on your CUDA computer and is reached through a Cloudflare tunnel.

## Vercel project settings

- Import `ASTRA228b/ChudGPT-Public`.
- **Root Directory:** leave it as the repository root (`.`). Do not select `public` or `api`.
- **Framework Preset:** Other.
- Build Command: leave blank.
- Output Directory: leave blank.
- Install Command: leave blank.
- Add environment variable `CHUDGPT_BACKEND_URL` with the current tunnel origin, such as `https://example.trycloudflare.com` (no trailing slash).
- Optional: set `CHUDGPT_API_KEY`. If set, API callers must send `Authorization: Bearer YOUR_KEY`; the included browser chat is intended for deployments without this optional key.

Redeploy after changing an environment variable. The public endpoints are `GET /api/status`, `POST /api/chat`, and `POST /api/clear`.

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

Quick tunnels change address whenever restarted. For a stable production service, configure a named Cloudflare Tunnel and a hostname you own.
