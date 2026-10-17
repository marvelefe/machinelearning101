/**
 * Authentication handlers tests
 * Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.5, 5.1, 5.2, 5.3, 5.4
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import {
  handleGoogleOAuthInitiation,
  handleOAuthCallback,
  handleGetUserProfile,
  handleLogout
} from '../../src/handlers/auth'
import type { Env } from '../../src/types'

import { generateAuthorizationUrl, exchangeCodeForToken, fetchUserProfile, validateState } from '../../src/services/oauth'
import { generateToken } from '../../src/services/jwt'
import { createUser, getUserByGoogleId, updateUser, getUserById } from '../../src/services/database'
import { authenticateRequest } from '../../src/middleware/auth'

// Mock dependencies
vi.mock('../../src/services/oauth')
vi.mock('../../src/services/jwt')
vi.mock('../../src/services/database')
vi.mock('../../src/middleware/auth')

describe('Authentication Handlers', () => {
  let mockEnv: Env
  
  beforeEach(() => {
    vi.clearAllMocks()
    
    mockEnv = {
      ml_platform_db: {} as D1Database,
      GOOGLE_CLIENT_ID: 'test-client-id',
      GOOGLE_CLIENT_SECRET: 'test-client-secret',
      GOOGLE_REDIRECT_URI: 'https://api.test.com/auth/callback',
      JWT_SECRET: 'test-jwt-secret',
      FRONTEND_URL: 'https://test.com'
    }
  })
  
  describe('handleGoogleOAuthInitiation', () => {
    it('should generate state and redirect to Google OAuth', async () => {
      // Requirements: 1.1, 5.1
      const mockState = 'random-state-123'
      const mockUrl = 'https://accounts.google.com/o/oauth2/v2/auth?client_id=test'
      
      // Ensure the mock is set up before the test runs
      vi.mocked(generateAuthorizationUrl).mockReturnValue({
        url: mockUrl,
        state: mockState
      })
      
      const request = new Request('https://api.test.com/auth/google')
      const response = await handleGoogleOAuthInitiation(request, mockEnv)
      
      expect(response.status).toBe(302)
      expect(response.headers.get('Location')).toBe(mockUrl)
      
      // Check that state is stored in cookie
      const setCookie = response.headers.get('Set-Cookie')
      expect(setCookie).toContain(`oauth_state=${mockState}`)
      expect(setCookie).toContain('HttpOnly')
      expect(setCookie).toContain('Secure')
      expect(setCookie).toContain('SameSite=Lax')
      
      expect(generateAuthorizationUrl).toHaveBeenCalledWith(
        mockEnv.GOOGLE_CLIENT_ID,
        mockEnv.GOOGLE_REDIRECT_URI
      )
    })
    
    it('should handle errors during OAuth initiation', async () => {
      // Requirements: 5.1, 7.2
      vi.mocked(generateAuthorizationUrl).mockImplementation(() => {
        throw new Error('OAuth service error')
      })
      
      const request = new Request('https://api.test.com/auth/google')
      
      await expect(handleGoogleOAuthInitiation(request, mockEnv)).rejects.toThrow('Failed to initiate OAuth flow')
    })
  })
  
  describe('handleOAuthCallback', () => {
    it('should process valid OAuth callback and create new user', async () => {
      // Requirements: 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4
      const mockCode = 'auth-code-123'
      const mockState = 'state-123'
      const mockAccessToken = 'access-token-123'
      const mockJwtToken = 'jwt-token-123'
      
      const mockUserProfile = {
        id: 'google-123',
        email: 'test@example.com',
        verified_email: true,
        name: 'Test User',
        given_name: 'Test',
        family_name: 'User',
        picture: 'https://example.com/photo.jpg'
      }
      
      const mockUser = {
        id: 'user-123',
        google_id: 'google-123',
        email: 'test@example.com',
        name: 'Test User',
        picture_url: 'https://example.com/photo.jpg',
        created_at: Date.now(),
        updated_at: Date.now()
      }
      
      vi.mocked(validateState).mockReturnValue(true)
      vi.mocked(exchangeCodeForToken).mockResolvedValue({
        access_token: mockAccessToken,
        expires_in: 3600,
        token_type: 'Bearer',
        scope: 'openid email profile'
      })
      vi.mocked(fetchUserProfile).mockResolvedValue(mockUserProfile)
      vi.mocked(getUserByGoogleId).mockResolvedValue(null) // New user
      vi.mocked(createUser).mockResolvedValue(mockUser)
      vi.mocked(generateToken).mockResolvedValue(mockJwtToken)
      
      const request = new Request(
        `https://api.test.com/auth/callback?code=${mockCode}&state=${mockState}`,
        {
          headers: {
            'Cookie': `oauth_state=${mockState}`
          }
        }
      )
      
      const response = await handleOAuthCallback(request, mockEnv)
      
      expect(response.status).toBe(302)
      expect(response.headers.get('Location')).toContain(mockEnv.FRONTEND_URL)
      expect(response.headers.get('Location')).toContain(`token=${encodeURIComponent(mockJwtToken)}`)
      
      // Verify state cookie is cleared
      const setCookie = response.headers.get('Set-Cookie')
      expect(setCookie).toContain('oauth_state=')
      expect(setCookie).toContain('Max-Age=0')
      
      expect(validateState).toHaveBeenCalledWith(mockState, mockState)
      expect(exchangeCodeForToken).toHaveBeenCalledWith(
        mockCode,
        mockEnv.GOOGLE_CLIENT_ID,
        mockEnv.GOOGLE_CLIENT_SECRET,
        mockEnv.GOOGLE_REDIRECT_URI
      )
      expect(fetchUserProfile).toHaveBeenCalledWith(mockAccessToken)
      expect(getUserByGoogleId).toHaveBeenCalledWith(mockEnv.ml_platform_db, mockUserProfile.id)
      expect(createUser).toHaveBeenCalledWith(mockEnv.ml_platform_db, {
        google_id: mockUserProfile.id,
        email: mockUserProfile.email,
        name: mockUserProfile.name,
        picture_url: mockUserProfile.picture
      })
      expect(generateToken).toHaveBeenCalledWith(
        mockUser.id,
        mockUser.email,
        mockUser.name,
        mockEnv.JWT_SECRET
      )
    })
    
    it('should update existing user on subsequent login', async () => {
      // Requirements: 1.4, 2.3, 2.4
      const mockCode = 'auth-code-123'
      const mockState = 'state-123'
      const mockAccessToken = 'access-token-123'
      const mockJwtToken = 'jwt-token-123'
      
      const mockUserProfile = {
        id: 'google-123',
        email: 'updated@example.com',
        verified_email: true,
        name: 'Updated Name',
        given_name: 'Updated',
        family_name: 'Name',
        picture: 'https://example.com/new-photo.jpg'
      }
      
      const existingUser = {
        id: 'user-123',
        google_id: 'google-123',
        email: 'old@example.com',
        name: 'Old Name',
        picture_url: 'https://example.com/old-photo.jpg',
        created_at: Date.now() - 1000000,
        updated_at: Date.now() - 1000000
      }
      
      const updatedUser = {
        ...existingUser,
        email: mockUserProfile.email,
        name: mockUserProfile.name,
        picture_url: mockUserProfile.picture,
        updated_at: Date.now()
      }
      
      vi.mocked(validateState).mockReturnValue(true)
      vi.mocked(exchangeCodeForToken).mockResolvedValue({
        access_token: mockAccessToken,
        expires_in: 3600,
        token_type: 'Bearer',
        scope: 'openid email profile'
      })
      vi.mocked(fetchUserProfile).mockResolvedValue(mockUserProfile)
      vi.mocked(getUserByGoogleId).mockResolvedValue(existingUser)
      vi.mocked(updateUser).mockResolvedValue(updatedUser)
      vi.mocked(generateToken).mockResolvedValue(mockJwtToken)
      
      const request = new Request(
        `https://api.test.com/auth/callback?code=${mockCode}&state=${mockState}`,
        {
          headers: {
            'Cookie': `oauth_state=${mockState}`
          }
        }
      )
      
      const response = await handleOAuthCallback(request, mockEnv)
      
      expect(response.status).toBe(302)
      expect(updateUser).toHaveBeenCalledWith(mockEnv.ml_platform_db, existingUser.id, {
        email: mockUserProfile.email,
        name: mockUserProfile.name,
        picture_url: mockUserProfile.picture
      })
      expect(createUser).not.toHaveBeenCalled()
    })
    
    it('should handle OAuth error from Google', async () => {
      // Requirements: 2.6
      const request = new Request(
        'https://api.test.com/auth/callback?error=access_denied'
      )
      
      const response = await handleOAuthCallback(request, mockEnv)
      
      expect(response.status).toBe(302)
      expect(response.headers.get('Location')).toContain(mockEnv.FRONTEND_URL)
      expect(response.headers.get('Location')).toContain('auth_error=access_denied')
    })
    
    it('should handle missing code parameter', async () => {
      // Requirements: 2.6
      const mockState = 'state-123'
      const request = new Request(
        `https://api.test.com/auth/callback?state=${mockState}`
      )
      
      const response = await handleOAuthCallback(request, mockEnv)
      
      expect(response.status).toBe(302)
      expect(response.headers.get('Location')).toContain('auth_error=missing_parameters')
    })
    
    it('should handle invalid state parameter', async () => {
      // Requirements: 2.6, 7.2
      const mockCode = 'auth-code-123'
      const receivedState = 'state-123'
      const expectedState = 'different-state-456'
      
      vi.mocked(validateState).mockReturnValue(false)
      
      const request = new Request(
        `https://api.test.com/auth/callback?code=${mockCode}&state=${receivedState}`,
        {
          headers: {
            'Cookie': `oauth_state=${expectedState}`
          }
        }
      )
      
      const response = await handleOAuthCallback(request, mockEnv)
      
      expect(response.status).toBe(302)
      expect(response.headers.get('Location')).toContain('auth_error=invalid_state')
    })
    
    it('should handle token exchange failure', async () => {
      // Requirements: 2.6, 7.3
      const mockCode = 'auth-code-123'
      const mockState = 'state-123'
      
      vi.mocked(validateState).mockReturnValue(true)
      vi.mocked(exchangeCodeForToken).mockRejectedValue(new Error('Token exchange failed'))
      
      const request = new Request(
        `https://api.test.com/auth/callback?code=${mockCode}&state=${mockState}`,
        {
          headers: {
            'Cookie': `oauth_state=${mockState}`
          }
        }
      )
      
      const response = await handleOAuthCallback(request, mockEnv)
      
      expect(response.status).toBe(302)
      expect(response.headers.get('Location')).toContain('auth_error=authentication_failed')
    })
  })
  
  describe('handleGetUserProfile', () => {
    it('should return user profile for authenticated user', async () => {
      // Requirements: 5.3
      const mockUser = {
        id: 'user-123',
        google_id: 'google-123',
        email: 'test@example.com',
        name: 'Test User',
        picture_url: 'https://example.com/photo.jpg',
        created_at: Date.now(),
        updated_at: Date.now()
      }
      
      vi.mocked(authenticateRequest).mockResolvedValue({
        user: {
          id: mockUser.id,
          email: mockUser.email,
          name: mockUser.name
        }
      })
      vi.mocked(getUserById).mockResolvedValue(mockUser)
      
      const request = new Request('https://api.test.com/auth/me', {
        headers: {
          'Authorization': 'Bearer valid-token'
        }
      })
      
      const response = await handleGetUserProfile(request, mockEnv)
      
      expect(response.status).toBe(200)
      
      const data = await response.json()
      expect(data).toEqual({
        id: mockUser.id,
        email: mockUser.email,
        name: mockUser.name,
        picture_url: mockUser.picture_url
      })
      
      expect(authenticateRequest).toHaveBeenCalledWith(request, mockEnv)
      expect(getUserById).toHaveBeenCalledWith(mockEnv.ml_platform_db, mockUser.id)
    })
    
    it('should handle user not found', async () => {
      // Requirements: 5.3
      vi.mocked(authenticateRequest).mockResolvedValue({
        user: {
          id: 'user-123',
          email: 'test@example.com',
          name: 'Test User'
        }
      })
      vi.mocked(getUserById).mockResolvedValue(null)
      
      const request = new Request('https://api.test.com/auth/me', {
        headers: {
          'Authorization': 'Bearer valid-token'
        }
      })
      
      await expect(handleGetUserProfile(request, mockEnv)).rejects.toThrow()
    })
    
    it('should handle authentication failure', async () => {
      // Requirements: 5.3
      vi.mocked(authenticateRequest).mockRejectedValue(new Error('Invalid token'))
      
      const request = new Request('https://api.test.com/auth/me', {
        headers: {
          'Authorization': 'Bearer invalid-token'
        }
      })
      
      await expect(handleGetUserProfile(request, mockEnv)).rejects.toThrow()
    })
  })
  
  describe('handleLogout', () => {
    it('should return success response', async () => {
      // Requirements: 3.5, 5.4
      const request = new Request('https://api.test.com/auth/logout', {
        method: 'POST'
      })
      
      const response = await handleLogout(request, mockEnv)
      
      expect(response.status).toBe(200)
      
      const data = await response.json()
      expect(data).toEqual({
        success: true
      })
    })
  })
})
