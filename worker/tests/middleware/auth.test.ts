/**
 * Authentication middleware tests
 * Requirements: 3.1, 3.2, 3.3, 3.4
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { extractToken, authenticateRequest } from '../../src/middleware/auth'
import * as jwtService from '../../src/services/jwt'
import type { Env, JWTPayload } from '../../src/types'
import { ErrorCode } from '../../src/types'
import { ApiErrorClass } from '../../src/utils/errors'

describe('Auth Middleware', () => {
  describe('extractToken', () => {
    it('should extract token from valid Authorization header', () => {
      // Requirements: 3.1
      const request = new Request('https://example.com', {
        headers: {
          'Authorization': 'Bearer test-token-123'
        }
      })

      const token = extractToken(request)
      expect(token).toBe('test-token-123')
    })

    it('should return null when Authorization header is missing', () => {
      // Requirements: 3.1
      const request = new Request('https://example.com')

      const token = extractToken(request)
      expect(token).toBeNull()
    })

    it('should return null when Authorization header format is invalid', () => {
      // Requirements: 3.1
      const request = new Request('https://example.com', {
        headers: {
          'Authorization': 'InvalidFormat token'
        }
      })

      const token = extractToken(request)
      expect(token).toBeNull()
    })

    it('should return null when Authorization header has no token', () => {
      // Requirements: 3.1
      const request = new Request('https://example.com', {
        headers: {
          'Authorization': 'Bearer'
        }
      })

      const token = extractToken(request)
      expect(token).toBeNull()
    })

    it('should return null when Authorization header has extra parts', () => {
      // Requirements: 3.1
      const request = new Request('https://example.com', {
        headers: {
          'Authorization': 'Bearer token extra-part'
        }
      })

      const token = extractToken(request)
      expect(token).toBeNull()
    })
  })

  describe('authenticateRequest', () => {
    let mockEnv: Env

    beforeEach(() => {
      mockEnv = {
        ml_platform_db: {} as D1Database,
        GOOGLE_CLIENT_ID: 'test-client-id',
        GOOGLE_CLIENT_SECRET: 'test-client-secret',
        GOOGLE_REDIRECT_URI: 'https://example.com/callback',
        JWT_SECRET: 'test-secret',
        FRONTEND_URL: 'https://frontend.com'
      }

      // Clear all mocks before each test
      vi.clearAllMocks()
    })

    it('should authenticate request with valid token', async () => {
      // Requirements: 3.1, 3.2, 3.4
      const mockPayload: JWTPayload = {
        sub: 'user-123',
        email: 'test@example.com',
        name: 'Test User',
        iat: Math.floor(Date.now() / 1000),
        exp: Math.floor(Date.now() / 1000) + 3600
      }

      vi.spyOn(jwtService, 'verifyToken').mockResolvedValue(mockPayload)

      const request = new Request('https://example.com', {
        headers: {
          'Authorization': 'Bearer valid-token'
        }
      })

      const result = await authenticateRequest(request, mockEnv)

      expect(result).toEqual({
        user: {
          id: 'user-123',
          email: 'test@example.com',
          name: 'Test User'
        }
      })
      expect(jwtService.verifyToken).toHaveBeenCalledWith('valid-token', 'test-secret')
    })

    it('should attach user context from token payload', async () => {
      // Requirements: 3.4
      const mockPayload: JWTPayload = {
        sub: 'user-456',
        email: 'another@example.com',
        name: 'Another User',
        iat: Math.floor(Date.now() / 1000),
        exp: Math.floor(Date.now() / 1000) + 3600
      }

      vi.spyOn(jwtService, 'verifyToken').mockResolvedValue(mockPayload)

      const request = new Request('https://example.com', {
        headers: {
          'Authorization': 'Bearer another-token'
        }
      })

      const result = await authenticateRequest(request, mockEnv)

      expect(result.user.id).toBe('user-456')
      expect(result.user.email).toBe('another@example.com')
      expect(result.user.name).toBe('Another User')
    })

    it('should throw 401 error when Authorization header is missing', async () => {
      // Requirements: 3.1, 3.2
      const request = new Request('https://example.com')

      await expect(authenticateRequest(request, mockEnv)).rejects.toThrow(ApiErrorClass)
      
      try {
        await authenticateRequest(request, mockEnv)
      } catch (error) {
        expect(error).toBeInstanceOf(ApiErrorClass)
        expect((error as ApiErrorClass).code).toBe(ErrorCode.UNAUTHORIZED)
        expect((error as ApiErrorClass).statusCode).toBe(401)
        expect((error as ApiErrorClass).message).toBe('Missing authentication token')
      }
    })

    it('should throw 401 error when token is invalid', async () => {
      // Requirements: 3.2, 3.3
      vi.spyOn(jwtService, 'verifyToken').mockRejectedValue(
        new Error('Token validation failed: invalid signature')
      )

      const request = new Request('https://example.com', {
        headers: {
          'Authorization': 'Bearer invalid-token'
        }
      })

      await expect(authenticateRequest(request, mockEnv)).rejects.toThrow(ApiErrorClass)
      
      try {
        await authenticateRequest(request, mockEnv)
      } catch (error) {
        expect(error).toBeInstanceOf(ApiErrorClass)
        expect((error as ApiErrorClass).code).toBe(ErrorCode.INVALID_TOKEN)
        expect((error as ApiErrorClass).statusCode).toBe(401)
        expect((error as ApiErrorClass).message).toContain('Invalid token signature')
      }
    })

    it('should throw 401 error when token is expired', async () => {
      // Requirements: 3.2, 3.3
      vi.spyOn(jwtService, 'verifyToken').mockRejectedValue(
        new Error('Token has expired')
      )

      const request = new Request('https://example.com', {
        headers: {
          'Authorization': 'Bearer expired-token'
        }
      })

      await expect(authenticateRequest(request, mockEnv)).rejects.toThrow(ApiErrorClass)
      
      try {
        await authenticateRequest(request, mockEnv)
      } catch (error) {
        expect(error).toBeInstanceOf(ApiErrorClass)
        expect((error as ApiErrorClass).code).toBe(ErrorCode.INVALID_TOKEN)
        expect((error as ApiErrorClass).statusCode).toBe(401)
        expect((error as ApiErrorClass).message).toBe('Token has expired')
      }
    })

    it('should throw 401 error when token has invalid signature', async () => {
      // Requirements: 3.2, 3.3
      vi.spyOn(jwtService, 'verifyToken').mockRejectedValue(
        new Error('Invalid token signature')
      )

      const request = new Request('https://example.com', {
        headers: {
          'Authorization': 'Bearer tampered-token'
        }
      })

      await expect(authenticateRequest(request, mockEnv)).rejects.toThrow(ApiErrorClass)
      
      try {
        await authenticateRequest(request, mockEnv)
      } catch (error) {
        expect(error).toBeInstanceOf(ApiErrorClass)
        expect((error as ApiErrorClass).code).toBe(ErrorCode.INVALID_TOKEN)
        expect((error as ApiErrorClass).statusCode).toBe(401)
        expect((error as ApiErrorClass).message).toBe('Invalid token signature')
      }
    })

    it('should throw 401 error for malformed Authorization header', async () => {
      // Requirements: 3.1, 3.2
      const request = new Request('https://example.com', {
        headers: {
          'Authorization': 'InvalidFormat'
        }
      })

      await expect(authenticateRequest(request, mockEnv)).rejects.toThrow(ApiErrorClass)
      
      try {
        await authenticateRequest(request, mockEnv)
      } catch (error) {
        expect(error).toBeInstanceOf(ApiErrorClass)
        expect((error as ApiErrorClass).code).toBe(ErrorCode.UNAUTHORIZED)
        expect((error as ApiErrorClass).statusCode).toBe(401)
      }
    })

    it('should handle generic token validation errors', async () => {
      // Requirements: 3.2, 3.3
      vi.spyOn(jwtService, 'verifyToken').mockRejectedValue(
        new Error('Some other validation error')
      )

      const request = new Request('https://example.com', {
        headers: {
          'Authorization': 'Bearer problematic-token'
        }
      })

      await expect(authenticateRequest(request, mockEnv)).rejects.toThrow(ApiErrorClass)
      
      try {
        await authenticateRequest(request, mockEnv)
      } catch (error) {
        expect(error).toBeInstanceOf(ApiErrorClass)
        expect((error as ApiErrorClass).code).toBe(ErrorCode.INVALID_TOKEN)
        expect((error as ApiErrorClass).statusCode).toBe(401)
        expect((error as ApiErrorClass).message).toBe('Token validation failed')
      }
    })

    it('should call verifyToken with correct parameters', async () => {
      // Requirements: 3.2, 3.4
      const mockPayload: JWTPayload = {
        sub: 'user-789',
        email: 'verify@example.com',
        name: 'Verify User',
        iat: Math.floor(Date.now() / 1000),
        exp: Math.floor(Date.now() / 1000) + 3600
      }

      const verifyTokenSpy = vi.spyOn(jwtService, 'verifyToken').mockResolvedValue(mockPayload)

      const request = new Request('https://example.com', {
        headers: {
          'Authorization': 'Bearer test-token-xyz'
        }
      })

      await authenticateRequest(request, mockEnv)

      expect(verifyTokenSpy).toHaveBeenCalledTimes(1)
      expect(verifyTokenSpy).toHaveBeenCalledWith('test-token-xyz', 'test-secret')
    })
  })
})
