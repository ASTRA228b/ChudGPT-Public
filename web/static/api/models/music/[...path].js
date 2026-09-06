import proxyHandler from "../../[...path].js";

export default async function handler(request, response) {
  const requestUrl = new URL(request.url, "https://chudgpt-public.invalid");
  const requestedPath = requestUrl.pathname
    .replace(/^\/api\/models\/music\/?/, "")
    .split("/")
    .filter(Boolean);
  request.query.path = ["models", "music", ...requestedPath];
  return proxyHandler(request, response);
}
