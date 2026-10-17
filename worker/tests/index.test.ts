/**
 * Main worker entry point tests
 * Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import app from '../src/index'
import type { Env } from '../src/index'

// Mock environment
const mockEnv: Env = {
  ml_platform_db: {} as D1Database,
  GOOGLE_CLIENT_ID: 'test-client-id',
  GOOGLE_CLIENT_SECRET: 'test-client-secret',
  GOOGLE_REDIRECT_URI: 'https://api.test.com/api/auth/google/callback',
  JWT_SECRET: 'test-jwt-secret-key-min-32-chars-long',
  FRONTEND_URL: 'https://test.com'
}

describe('Worker Entry Point', () => {
  describe('Health Check', () => {
    it('should return health check response', async () => {
      const req = new Request('https://api.test.com/')
      const res = await app.fetch(req, mockEnv)
      
      expect(res.status).toBe(200)
      
      const data = await res.json()
      expect(data).toEqual({
        status: 'ok',
        message: 'ML Platform Auth Worker'
      })
    })
  })

  describe('CORS Middleware', () => {
    it('should apply CORS headers to all responses', async () => {
      const req = new Request('https://api.test.com/')
      const res = await app.fetch(req, mockEnv)
      
      expect(res.headers.get('Access-Control-Allow-Origin')).toBe('https://test.com')
      expect(res.headers.get('Access-Control-Allow-Methods')).toBe('GET, POST, PUT, DELETE, PATCH, OPTIONS')
      expect(res.headers.get('Access-Control-Allow-Headers')).toBe('Content-Type, Authorization, X-Requested-With, Accept, Origin')
    })

    it('should handle preflight OPTIONS requests', async () => {
      const req = new Request('https://api.test.com/api/auth/me', {
        method: 'OPTIONS'
      })
      const res = await app.fetch(req, mockEnv)
      
      expect(res.status).toBe(204)
      expect(res.headers.get('Access-Control-Allow-Origin')).toBe('https://test.com')
    })
  })

  describe('Route Registration', () => {
    it('should have auth routes registered', async () => {
      // Test that routes exist by checking they don't return 404
      const routes = [
        '/api/auth/google',
        '/api/auth/google/callback?code=test&state=test',
        '/api/auth/me',
      ]

      for (const route of routes) {
        const req = new Request(`https://api.test.com${route}`)
        const res = await app.fetch(req, mockEnv)
        
        // Should not be 404 (routes exist, even if they fail for other reasons)
        expect(res.status).not.toBe(404)
      }
    })

    it('should have progress routes registered', async () => {
      const routes = [
        { path: '/api/progress', method: 'GET' },
        { path: '/api/progress/sync', method: 'POST' }
      ]

      for (const { path, method } of routes) {
        const req = new Request(`https://api.test.com${path}`, {
          method,
          headers: method === 'POST' ? { 'Content-Type': 'application/json' } : {}
        })
        const res = await app.fetch(req, mockEnv)
        
        // Should not be 404 (routes exist, even if they fail for other reasons)
        expect(res.status).not.toBe(404)
      }
    })

    it('should return 404 for unknown routes', async () => {
      const req = new Request('https://api.test.com/api/unknown')
      const res = await app.fetch(req, mockEnv)
      
      expect(res.status).toBe(404)
    })
  })

  describe('Error Handling Middleware', () => {
    it('should handle errors and return formatted error responses', async () => {
      // Request to protected endpoint without auth should trigger error handling
      const req = new Request('https://api.test.com/api/auth/me')
      const res = await app.fetch(req, mockEnv)
      
      // Should return error response (401 for missing auth)
      expect(res.status).toBe(401)
      expect(res.headers.get('Content-Type')).toBe('application/json')
      
      const data = await res.json() as any
      expect(data).toHaveProperty('code')
      expect(data).toHaveProperty('message')
    })

    it('should include CORS headers in error responses', async () => {
      const req = new Request('https://api.test.com/api/auth/me')
      const res = await app.fetch(req, mockEnv)
      
      // Error responses should still have CORS headers
      expect(res.headers.get('Access-Control-Allow-Origin')).toBe('https://test.com')
    })
  })

  describe('POST /api/auth/logout', () => {
    it('should handle logout requests', async () => {
      const req = new Request('https://api.test.com/api/auth/logout', {
        method: 'POST'
      })
      const res = await app.fetch(req, mockEnv)
      
      expect(res.status).toBe(200)
      
      const data = await res.json() as any
      expect(data).toEqual({ success: true })
    })
  })
})
