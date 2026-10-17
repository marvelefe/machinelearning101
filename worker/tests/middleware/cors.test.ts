/**
 * Unit tests for CORS middleware
 * Requirements: 5.8
 */

import { describe, it, expect, vi } from "vitest";
import { Hono } from "hono";
import { corsMiddleware, getCorsHeaders } from "../../src/middleware/cors";
import type { Env } from "../../src/types";

describe("CORS Middleware", () => {
  const mockEnv: Env = {
    ml_platform_db: {} as D1Database,
    GOOGLE_CLIENT_ID: "test-client-id",
    GOOGLE_CLIENT_SECRET: "test-client-secret",
    GOOGLE_REDIRECT_URI: "http://localhost/callback",
    JWT_SECRET: "test-secret",
    FRONTEND_URL: "https://example.com",
  };

  describe("getCorsHeaders", () => {
    it("should return CORS headers with allowed origin when origin matches", () => {
      const origin = "https://example.com";
      const allowedOrigin = "https://example.com";

      const headers = getCorsHeaders(origin, allowedOrigin);

      expect(headers).toEqual({
        "Access-Control-Allow-Origin": "https://example.com",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, PATCH, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, Accept, Origin",
        "Access-Control-Max-Age": "86400",
      });
    });

    it("should return CORS headers with allowed origin when origin does not match", () => {
      const origin = "https://malicious.com";
      const allowedOrigin = "https://example.com";

      const headers = getCorsHeaders(origin, allowedOrigin);

      expect(headers).toEqual({
        "Access-Control-Allow-Origin": "https://example.com",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, PATCH, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, Accept, Origin",
        "Access-Control-Max-Age": "86400",
      });
    });

    it("should validate origin against allowed origin", () => {
      const validOrigin = "https://example.com";
      const invalidOrigin = "https://evil.com";
      const allowedOrigin = "https://example.com";

      const validHeaders = getCorsHeaders(validOrigin, allowedOrigin);
      const invalidHeaders = getCorsHeaders(invalidOrigin, allowedOrigin);

      expect(validHeaders["Access-Control-Allow-Origin"]).toBe(validOrigin);
      expect(invalidHeaders["Access-Control-Allow-Origin"]).toBe(allowedOrigin);
    });
  });

  describe("corsMiddleware", () => {
    it("should handle preflight OPTIONS requests", async () => {
      const app = new Hono<{ Bindings: Env }>();
      app.use("*", corsMiddleware);
      app.get("/test", (c) => c.json({ message: "test" }));

      const req = new Request("http://localhost/test", {
        method: "OPTIONS",
        headers: {
          Origin: "https://example.com",
        },
      });

      const res = await app.fetch(req, mockEnv);

      expect(res.status).toBe(204);
      expect(res.headers.get("Access-Control-Allow-Origin")).toBe("https://example.com");
      expect(res.headers.get("Access-Control-Allow-Methods")).toBe("GET, POST, PUT, DELETE, PATCH, OPTIONS");
      expect(res.headers.get("Access-Control-Allow-Headers")).toBe("Content-Type, Authorization, X-Requested-With, Accept, Origin");
      expect(res.headers.get("Access-Control-Max-Age")).toBe("86400");
    });

    it("should add CORS headers to regular requests", async () => {
      const app = new Hono<{ Bindings: Env }>();
      app.use("*", corsMiddleware);
      app.get("/test", (c) => c.json({ message: "test" }));

      const req = new Request("http://localhost/test", {
        method: "GET",
        headers: {
          Origin: "https://example.com",
        },
      });

      const res = await app.fetch(req, mockEnv);

      expect(res.status).toBe(200);
      expect(res.headers.get("Access-Control-Allow-Origin")).toBe("https://example.com");
      expect(res.headers.get("Access-Control-Allow-Methods")).toBe("GET, POST, PUT, DELETE, PATCH, OPTIONS");
      expect(res.headers.get("Access-Control-Allow-Headers")).toBe("Content-Type, Authorization, X-Requested-With, Accept, Origin");
      expect(res.headers.get("Access-Control-Max-Age")).toBe("86400");
    });

    it("should validate origin and use allowed origin for non-matching origins", async () => {
      const app = new Hono<{ Bindings: Env }>();
      app.use("*", corsMiddleware);
      app.get("/test", (c) => c.json({ message: "test" }));

      const req = new Request("http://localhost/test", {
        method: "GET",
        headers: {
          Origin: "https://malicious.com",
        },
      });

      const res = await app.fetch(req, mockEnv);

      expect(res.status).toBe(200);
      expect(res.headers.get("Access-Control-Allow-Origin")).toBe("https://example.com");
    });

    it("should handle requests without Origin header", async () => {
      const app = new Hono<{ Bindings: Env }>();
      app.use("*", corsMiddleware);
      app.get("/test", (c) => c.json({ message: "test" }));

      const req = new Request("http://localhost/test", {
        method: "GET",
      });

      const res = await app.fetch(req, mockEnv);

      expect(res.status).toBe(200);
      expect(res.headers.get("Access-Control-Allow-Origin")).toBe("https://example.com");
    });

    it("should add CORS headers to POST requests", async () => {
      const app = new Hono<{ Bindings: Env }>();
      app.use("*", corsMiddleware);
      app.post("/test", (c) => c.json({ message: "created" }));

      const req = new Request("http://localhost/test", {
        method: "POST",
        headers: {
          Origin: "https://example.com",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ data: "test" }),
      });

      const res = await app.fetch(req, mockEnv);

      expect(res.status).toBe(200);
      expect(res.headers.get("Access-Control-Allow-Origin")).toBe("https://example.com");
      expect(res.headers.get("Access-Control-Allow-Methods")).toBe("GET, POST, PUT, DELETE, PATCH, OPTIONS");
    });

    it("should handle preflight for POST requests with Authorization header", async () => {
      const app = new Hono<{ Bindings: Env }>();
      app.use("*", corsMiddleware);
      app.post("/test", (c) => c.json({ message: "created" }));

      const req = new Request("http://localhost/test", {
        method: "OPTIONS",
        headers: {
          Origin: "https://example.com",
          "Access-Control-Request-Method": "POST",
          "Access-Control-Request-Headers": "Content-Type, Authorization",
        },
      });

      const res = await app.fetch(req, mockEnv);

      expect(res.status).toBe(204);
      expect(res.headers.get("Access-Control-Allow-Headers")).toContain("Authorization");
      expect(res.headers.get("Access-Control-Allow-Headers")).toContain("Content-Type");
    });
  });
});
