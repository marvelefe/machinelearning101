/**
 * Progress handlers tests
 * Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.5, 5.6
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { handleSyncProgress, handleGetProgress } from '../../src/handlers/progress'
import * as authMiddleware from '../../src/middleware/auth'
import * as database from '../../src/services/database'
import * as validation from '../../src/utils/validation'
import type { Env, SyncProgressRequest, Progress, QuizScore, BookmarkWithParams } from '../../src/types'

// Mock dependencies
vi.mock('../../src/middleware/auth')
vi.mock('../../src/services/database')
vi.mock('../../src/utils/validation')

describe('Progress Handlers', () => {
  let mockEnv: Env
  
  beforeEach(() => {
    vi.clearAllMocks()
    
    mockEnv = {
      ml_platform_db: {} as D1Database,
      GOOGLE_CLIENT_ID: 'test-client-id',
      GOOGLE_CLIENT_SECRET: 'test-client-secret',
      GOOGLE_REDIRECT_URI: 'https://api.test.com/callback',
      JWT_SECRET: 'test-secret',
      FRONTEND_URL: 'https://test.com'
    }
    
    // Default mock for isValidModuleId
    vi.mocked(validation.isValidModuleId).mockReturnValue(true)
  })
  
  describe('handleSyncProgress', () => {
    it('should sync progress data successfully', async () => {
      // Mock authenticated user
      vi.mocked(authMiddleware.authenticateRequest).mockResolvedValue({
        user: {
          id: 'user-123',
          email: 'test@example.com',
          name: 'Test User'
        }
      })
      
      // Mock database operations
      vi.mocked(database.saveProgress).mockResolvedValue({
        id: 'progress-1',
        user_id: 'user-123',
        module_id: 'linear-regression',
        completed: true,
        completed_at: 1234567890
      })
      
      vi.mocked(database.saveQuizScore).mockResolvedValue({
        id: 'quiz-1',
        user_id: 'user-123',
        module_id: 'linear-regression',
        score: 8,
        total: 10,
        completed_at: 1234567890
      })
      
      vi.mocked(database.saveBookmark).mockResolvedValue({
        id: 'bookmark-1',
        user_id: 'user-123',
        module_id: 'linear-regression',
        name: 'Test Bookmark',
        parameters: '{"learningRate":0.01}',
        created_at: 1234567890,
        updated_at: 1234567890
      })
      
      // Create request with sync data
      const syncData: SyncProgressRequest = {
        modules: [
          {
            module_id: 'linear-regression',
            completed: true,
            completed_at: 1234567890
          }
        ],
        quizzes: [
          {
            module_id: 'linear-regression',
            score: 8,
            total: 10,
            completed_at: 1234567890
          }
        ],
        bookmarks: [
          {
            id: 'bookmark-1',
            module_id: 'linear-regression',
            name: 'Test Bookmark',
            parameters: { learningRate: 0.01 },
            created_at: 1234567890,
            updated_at: 1234567890
          }
        ]
      }
      
      const request = new Request('https://api.test.com/api/progress/sync', {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer valid-token',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(syncData)
      })
      
      const response = await handleSyncProgress(request, mockEnv)
      const result = await response.json() as any
      
      expect(response.status).toBe(200)
      expect(result.success).toBe(true)
      expect(result.synced_at).toBeTypeOf('number')
      
      // Verify database operations were called
      expect(database.saveProgress).toHaveBeenCalledWith(
        mockEnv.ml_platform_db,
        'user-123',
        'linear-regression',
        true,
        1234567890
      )
      
      expect(database.saveQuizScore).toHaveBeenCalledWith(
        mockEnv.ml_platform_db,
        'user-123',
        'linear-regression',
        8,
        10,
        1234567890
      )
      
      expect(database.saveBookmark).toHaveBeenCalledWith(
        mockEnv.ml_platform_db,
        'user-123',
        'bookmark-1',
        'linear-regression',
        'Test Bookmark',
        { learningRate: 0.01 },
        1234567890,
        1234567890
      )
    })
    
    it('should handle empty arrays in sync data', async () => {
      vi.mocked(authMiddleware.authenticateRequest).mockResolvedValue({
        user: {
          id: 'user-123',
          email: 'test@example.com',
          name: 'Test User'
        }
      })
      
      const syncData: SyncProgressRequest = {
        modules: [],
        quizzes: [],
        bookmarks: []
      }
      
      const request = new Request('https://api.test.com/api/progress/sync', {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer valid-token',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(syncData)
      })
      
      const response = await handleSyncProgress(request, mockEnv)
      const result = await response.json() as any
      
      expect(response.status).toBe(200)
      expect(result.success).toBe(true)
      
      // Verify no database operations were called
      expect(database.saveProgress).not.toHaveBeenCalled()
      expect(database.saveQuizScore).not.toHaveBeenCalled()
      expect(database.saveBookmark).not.toHaveBeenCalled()
    })
    
    it('should reject invalid JSON', async () => {
      vi.mocked(authMiddleware.authenticateRequest).mockResolvedValue({
        user: {
          id: 'user-123',
          email: 'test@example.com',
          name: 'Test User'
        }
      })
      
      const request = new Request('https://api.test.com/api/progress/sync', {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer valid-token',
          'Content-Type': 'application/json'
        },
        body: 'invalid json'
      })
      
      await expect(handleSyncProgress(request, mockEnv)).rejects.toThrow()
    })
    
    it('should validate module_id format', async () => {
      vi.mocked(authMiddleware.authenticateRequest).mockResolvedValue({
        user: {
          id: 'user-123',
          email: 'test@example.com',
          name: 'Test User'
        }
      })
      
      // Mock isValidModuleId to return false for invalid module
      vi.mocked(validation.isValidModuleId).mockReturnValue(false)
      
      const syncData: SyncProgressRequest = {
        modules: [
          {
            module_id: 'invalid<script>',
            completed: true,
            completed_at: 1234567890
          }
        ],
        quizzes: [],
        bookmarks: []
      }
      
      const request = new Request('https://api.test.com/api/progress/sync', {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer valid-token',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(syncData)
      })
      
      await expect(handleSyncProgress(request, mockEnv)).rejects.toThrow()
    })
    
    it('should validate completed field is boolean', async () => {
      vi.mocked(authMiddleware.authenticateRequest).mockResolvedValue({
        user: {
          id: 'user-123',
          email: 'test@example.com',
          name: 'Test User'
        }
      })
      
      const syncData = {
        modules: [
          {
            module_id: 'linear-regression',
            completed: 'true', // Invalid: should be boolean
            completed_at: 1234567890
          }
        ],
        quizzes: [],
        bookmarks: []
      }
      
      const request = new Request('https://api.test.com/api/progress/sync', {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer valid-token',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(syncData)
      })
      
      await expect(handleSyncProgress(request, mockEnv)).rejects.toThrow()
    })
    
    it('should validate quiz score is non-negative', async () => {
      vi.mocked(authMiddleware.authenticateRequest).mockResolvedValue({
        user: {
          id: 'user-123',
          email: 'test@example.com',
          name: 'Test User'
        }
      })
      
      const syncData: SyncProgressRequest = {
        modules: [],
        quizzes: [
          {
            module_id: 'linear-regression',
            score: -5, // Invalid: negative score
            total: 10,
            completed_at: 1234567890
          }
        ],
        bookmarks: []
      }
      
      const request = new Request('https://api.test.com/api/progress/sync', {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer valid-token',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(syncData)
      })
      
      await expect(handleSyncProgress(request, mockEnv)).rejects.toThrow()
    })
    
    it('should validate required fields are present', async () => {
      vi.mocked(authMiddleware.authenticateRequest).mockResolvedValue({
        user: {
          id: 'user-123',
          email: 'test@example.com',
          name: 'Test User'
        }
      })
      
      const syncData = {
        modules: [] // Missing quizzes and bookmarks
      }
      
      const request = new Request('https://api.test.com/api/progress/sync', {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer valid-token',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(syncData)
      })
      
      await expect(handleSyncProgress(request, mockEnv)).rejects.toThrow()
    })
    
    it('should handle conflict resolution with timestamps', async () => {
      vi.mocked(authMiddleware.authenticateRequest).mockResolvedValue({
        user: {
          id: 'user-123',
          email: 'test@example.com',
          name: 'Test User'
        }
      })
      
      // Mock saveProgress to simulate upsert behavior
      vi.mocked(database.saveProgress).mockResolvedValue({
        id: 'progress-1',
        user_id: 'user-123',
        module_id: 'linear-regression',
        completed: true,
        completed_at: 1234567999 // Later timestamp wins
      })
      
      const syncData: SyncProgressRequest = {
        modules: [
          {
            module_id: 'linear-regression',
            completed: true,
            completed_at: 1234567999 // Newer timestamp
          }
        ],
        quizzes: [],
        bookmarks: []
      }
      
      const request = new Request('https://api.test.com/api/progress/sync', {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer valid-token',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(syncData)
      })
      
      const response = await handleSyncProgress(request, mockEnv)
      const result = await response.json() as any
      
      expect(response.status).toBe(200)
      expect(result.success).toBe(true)
      
      // Verify saveProgress was called with the newer timestamp
      expect(database.saveProgress).toHaveBeenCalledWith(
        mockEnv.ml_platform_db,
        'user-123',
        'linear-regression',
        true,
        1234567999
      )
    })
    
    it('should handle authentication errors', async () => {
      // Mock authentication failure
      vi.mocked(authMiddleware.authenticateRequest).mockRejectedValue({
        code: 'UNAUTHORIZED',
        message: 'Invalid token'
      })
      
      const syncData: SyncProgressRequest = {
        modules: [],
        quizzes: [],
        bookmarks: []
      }
      
      const request = new Request('https://api.test.com/api/progress/sync', {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer invalid-token',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(syncData)
      })
      
      await expect(handleSyncProgress(request, mockEnv)).rejects.toMatchObject({
        code: 'UNAUTHORIZED'
      })
    })
  })
  
  describe('handleGetProgress', () => {
    it('should retrieve all progress data successfully', async () => {
      // Mock authenticated user
      vi.mocked(authMiddleware.authenticateRequest).mockResolvedValue({
        user: {
          id: 'user-123',
          email: 'test@example.com',
          name: 'Test User'
        }
      })
      
      // Mock database responses
      const mockModules: Progress[] = [
        {
          id: 'progress-1',
          user_id: 'user-123',
          module_id: 'linear-regression',
          completed: true,
          completed_at: 1234567890
        },
        {
          id: 'progress-2',
          user_id: 'user-123',
          module_id: 'k-means',
          completed: false,
          completed_at: null
        }
      ]
      
      const mockQuizzes: QuizScore[] = [
        {
          id: 'quiz-1',
          user_id: 'user-123',
          module_id: 'linear-regression',
          score: 8,
          total: 10,
          completed_at: 1234567890
        }
      ]
      
      const mockBookmarks: BookmarkWithParams[] = [
        {
          id: 'bookmark-1',
          user_id: 'user-123',
          module_id: 'pca',
          name: 'PCA Bookmark',
          parameters: {},
          created_at: 1234567890,
          updated_at: 1234567890
        }
      ]
      
      vi.mocked(database.getProgressByUserId).mockResolvedValue(mockModules)
      vi.mocked(database.getQuizScoresByUserId).mockResolvedValue(mockQuizzes)
      vi.mocked(database.getBookmarksByUserId).mockResolvedValue(mockBookmarks)
      
      const request = new Request('https://api.test.com/api/progress', {
        method: 'GET',
        headers: {
          'Authorization': 'Bearer valid-token'
        }
      })
      
      const response = await handleGetProgress(request, mockEnv)
      const result = await response.json() as any
      
      expect(response.status).toBe(200)
      expect(result.modules).toEqual(mockModules)
      expect(result.quizzes).toEqual(mockQuizzes)
      expect(result.bookmarks).toEqual(mockBookmarks)
      
      // Verify database operations were called
      expect(database.getProgressByUserId).toHaveBeenCalledWith(mockEnv.ml_platform_db, 'user-123')
      expect(database.getQuizScoresByUserId).toHaveBeenCalledWith(mockEnv.ml_platform_db, 'user-123')
      expect(database.getBookmarksByUserId).toHaveBeenCalledWith(mockEnv.ml_platform_db, 'user-123')
    })
    
    it('should return empty arrays when user has no progress', async () => {
      vi.mocked(authMiddleware.authenticateRequest).mockResolvedValue({
        user: {
          id: 'user-123',
          email: 'test@example.com',
          name: 'Test User'
        }
      })
      
      // Mock empty responses
      vi.mocked(database.getProgressByUserId).mockResolvedValue([])
      vi.mocked(database.getQuizScoresByUserId).mockResolvedValue([])
      vi.mocked(database.getBookmarksByUserId).mockResolvedValue([])
      
      const request = new Request('https://api.test.com/api/progress', {
        method: 'GET',
        headers: {
          'Authorization': 'Bearer valid-token'
        }
      })
      
      const response = await handleGetProgress(request, mockEnv)
      const result = await response.json() as any
      
      expect(response.status).toBe(200)
      expect(result.modules).toEqual([])
      expect(result.quizzes).toEqual([])
      expect(result.bookmarks).toEqual([])
    })
    
    it('should handle authentication errors', async () => {
      // Mock authentication failure
      vi.mocked(authMiddleware.authenticateRequest).mockRejectedValue({
        code: 'UNAUTHORIZED',
        message: 'Invalid token'
      })
      
      const request = new Request('https://api.test.com/api/progress', {
        method: 'GET',
        headers: {
          'Authorization': 'Bearer invalid-token'
        }
      })
      
      await expect(handleGetProgress(request, mockEnv)).rejects.toMatchObject({
        code: 'UNAUTHORIZED'
      })
    })
    
    it('should handle database errors', async () => {
      vi.mocked(authMiddleware.authenticateRequest).mockResolvedValue({
        user: {
          id: 'user-123',
          email: 'test@example.com',
          name: 'Test User'
        }
      })
      
      // Mock database error
      vi.mocked(database.getProgressByUserId).mockRejectedValue({
        code: 'DATABASE_ERROR',
        message: 'Database connection failed'
      })
      
      const request = new Request('https://api.test.com/api/progress', {
        method: 'GET',
        headers: {
          'Authorization': 'Bearer valid-token'
        }
      })
      
      await expect(handleGetProgress(request, mockEnv)).rejects.toMatchObject({
        code: 'DATABASE_ERROR'
      })
    })
  })
})
