import proxyHandler from "../[...path].js";

export default async function handler(request, response) {
  // Preserve every segment after /api/models/ and pass the canonical model
  // route to the shared Cloudflare proxy. This explicit nested function is
  // required because Vercel does not consistently match the root catch-all
  // for multi-segment paths in this static deployment.
  const requestUrl = new URL(request.url, "https://chudgpt-public.invalid");
  const requestedPath = requestUrl.pathname
    .replace(/^\/api\/models\/?/, "")
    .split("/")
    .filter(Boolean);
  request.query.path = ["models", ...requestedPath];
  return proxyHandler(request, response);
}
