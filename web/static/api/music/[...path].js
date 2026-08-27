import proxyHandler from "../[...path].js";

export default async function handler(request, response) {
  const requestedPath = Array.isArray(request.query.path)
    ? request.query.path
    : [request.query.path || "status"];
  request.query.path = ["music", ...requestedPath];
  return proxyHandler(request, response);
}
