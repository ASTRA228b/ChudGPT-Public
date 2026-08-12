const ALLOWED_METHODS = "GET, POST, OPTIONS";

export default async function handler(request, response) {
  response.setHeader("Access-Control-Allow-Origin", "*");
  response.setHeader("Access-Control-Allow-Methods", ALLOWED_METHODS);
  response.setHeader("Access-Control-Allow-Headers", "Authorization, Content-Type");
  response.setHeader("Cache-Control", "no-store");

  if (request.method === "OPTIONS") return response.status(204).end();

  const requiredKey = process.env.CHUDGPT_API_KEY;
  if (requiredKey && request.headers.authorization !== `Bearer ${requiredKey}`) {
    return response.status(401).json({ error: "Missing or invalid API key" });
  }

  const backend = (process.env.CHUDGPT_BACKEND_URL || "").replace(/\/$/, "");
  if (!backend) {
    return response.status(503).json({
      error: "CHUDGPT_BACKEND_URL is not configured in Vercel",
    });
  }

  const pathParts = Array.isArray(request.query.path)
    ? request.query.path
    : [request.query.path || "status"];
  const target = `${backend}/api/${pathParts.map(encodeURIComponent).join("/")}`;

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
