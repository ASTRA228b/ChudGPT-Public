import proxyHandler from "../[...path].js";

export default async function handler(request, response) {
  // Nested catch-all parameters are not populated consistently by every
  // Vercel runtime. Read the actual request path so POST /api/music/chat can
  // never be mistaken for POST /api/music/status (which returns HTTP 405).
  const requestUrl = new URL(request.url, "https://chudgpt-public.invalid");
  const requestedPath = requestUrl.pathname
    .replace(/^\/api\/music\/?/, "")
    .split("/")
    .filter(Boolean);
  request.query.path = ["music", ...requestedPath];
  if (requestedPath.length === 0) request.query.path.push("status");
  return proxyHandler(request, response);
}
