/**
 * CORS middleware for handling cross-origin requests
 * Requirements: 5.8
 */

import { Context, Next } from "hono";
import type { Env } from "../types";

/**
 * CORS configuration headers
 *
 * Only the configured frontend origin (or the request origin when it matches)
 * is ever reflected, so browsers on other sites cannot make cross-origin
 * requests. Credentials are intentionally not enabled: the API authenticates
 * via an `Authorization: Bearer` header rather than cookies.
 */
export const getCorsHeaders = (origin: string | undefined, allowedOrigin: string) => {
  const allowOrigin = origin === allowedOrigin ? origin : allowedOrigin;

  return {
    "Access-Control-Allow-Origin": allowOrigin,
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, PATCH, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, Accept, Origin",
    "Access-Control-Max-Age": "86400",
  };
};

/**
 * CORS middleware that adds CORS headers to all responses
 * and handles preflight requests
 */
export const corsMiddleware = async (c: Context<{ Bindings: Env }>, next: Next) => {
  const allowedOrigin = c.env.FRONTEND_URL;
  const origin = c.req.header("Origin");

  // Handle preflight requests
  if (c.req.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: getCorsHeaders(origin, allowedOrigin),
    });
  }

  // Continue to next middleware/handler
  await next();

  // Rebuild the response with CORS headers rather than mutating c.res.headers,
  // because some responses (e.g. redirects) expose immutable headers.
  const headers = new Headers(c.res.headers);
  Object.entries(getCorsHeaders(origin, allowedOrigin)).forEach(([key, value]) => {
    headers.set(key, value);
  });

  c.res = new Response(c.res.body, {
    status: c.res.status,
    statusText: c.res.statusText,
    headers,
  });
};
