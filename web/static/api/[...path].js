export default async function handler(request, response) {
  response.setHeader("Access-Control-Allow-Origin", "*");
  response.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  response.setHeader("Access-Control-Allow-Headers", "Content-Type");
  response.setHeader("Cache-Control", "no-store");
  if (request.method === "OPTIONS") return response.status(204).end();

  // This project currently uses a Cloudflare quick tunnel. Its checked-in URL
  // is the active source of truth; an old Vercel environment variable can
  // otherwise silently route requests to a stale backend after a restart.
  const backend = "https://nav-hotel-featuring-pearl.trycloudflare.com";
  if (!backend) {
    return response.status(503).json({ error: "CHUDGPT_BACKEND_URL is not configured" });
  }
  const requestUrl = new URL(request.url, "https://chudgpt-public.invalid");
  const urlPath = requestUrl.pathname.replace(/^\/api\/?/, "");
  const queryPath = Array.isArray(request.query.path)
    ? request.query.path.join("/")
    : request.query.path;
  const selectedPath = queryPath || urlPath || "status";
  const parts = selectedPath.split("/").filter(Boolean);
  const target = `${backend}/api/${parts.map(encodeURIComponent).join("/")}`;
  try {
    const upstream = await fetch(target, {
      method: request.method,
      headers: { "Content-Type": "application/json" },
      body: request.method === "GET" ? undefined : JSON.stringify(request.body || {}),
      signal: AbortSignal.timeout(60_000),
    });
    const body = await upstream.text();
    response.status(upstream.status);
    response.setHeader("Content-Type", upstream.headers.get("content-type") || "application/json");
    return response.send(body);
  } catch (error) {
    return response.status(502).json({ error: `ChudGPT backend unavailable: ${error.message}` });
  }
}















