export default async function handler(request, response) {
  response.setHeader("Access-Control-Allow-Origin", "*");
  response.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  response.setHeader("Access-Control-Allow-Headers", "Content-Type");
  response.setHeader("Cache-Control", "no-store");
  if (request.method === "OPTIONS") return response.status(204).end();

  // Vercel's environment variable takes priority. The checked-in quick-tunnel
  // fallback keeps a fresh deployment usable until that variable is added.
  const fallbackBackend = "https://flame-publicly-supplied-jon.trycloudflare.com";
  const backend = (process.env.CHUDGPT_BACKEND_URL || fallbackBackend).replace(/\/$/, "");
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
