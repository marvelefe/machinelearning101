/**
 * Progress sync integration test
 * Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
 * 
 * Tests the complete progress synchronization flow including syncing new data,
 * merging with existing data, conflict resolution, and batch operations.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import app from '../../src/index'
import type { Env } from '../../src/types'
import { generateToken } from '../../src/services/jwt'

describe('Progress Sync Integration', () => {
  let mockEnv: Env
  let mockDB: any
  let testUserId: string
  let testJwtToken: string
  
  beforeEach(async () => {
    testUserId = 'test-user-123'
    
    // Mock D1 database
    mockDB = {
      prepare: vi.fn().mockReturnThis(),
      bind: vi.fn().mockReturnThis(),
      run: vi.fn().mockResolvedValue({ success: true, meta: { changes: 1 } }),
      first: vi.fn().mockResolvedValue(null),
      all: vi.fn().mockResolvedValue({ results: [] })
    }
    
    mockEnv = {
      ml_platform_db: mockDB as D1Database,
      GOOGLE_CLIENT_ID: 'test-client-id',
      GOOGLE_CLIENT_SECRET: 'test-client-secret',
      GOOGLE_REDIRECT_URI: 'https://api.test.com/callback',
      JWT_SECRET: 'test-jwt-secret-key-for-testing-purposes',
      FRONTEND_URL: 'https://test.com'
    }
    
    // Generate a valid JWT token for testing
    testJwtToken = await generateToken(
      testUserId,
      'test@example.com',
      'Test User',
      mockEnv.JWT_SECRET
    )
  })
  
  afterEach(() => {
    vi.clearAllMocks()
  })
  
  it('should sync new progress data successfully', async () => {
    // Requirements: 4.1, 4.2, 4.3, 4.5
    
    const syncData = {
      modules: [
        {
          module_id: 'regression/linear-regression',
          completed: true,
          completed_at: Date.now()
        },
        {
          module_id: 'classification/logistic-regression',
          completed: false,
          completed_at: null
        }
      ],
      quizzes: [
        {
          module_id: 'regression/linear-regression',
          score: 8,
          total: 10,
          completed_at: Date.now()
        }
      ],
      bookmarks: [
        {
          id: 'bookmark-kmeans-1',
          module_id: 'clustering/k-means',
          name: 'K-Means Bookmark',
          parameters: { clusters: 3 },
          created_at: Date.now(),
          updated_at: Date.now()
        }
      ]
    }
    
    const request = new Request('https://api.test.com/api/progress/sync', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${testJwtToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(syncData)
    })
    
    const response = await app.fetch(request, mockEnv)
    
    expect(response.status).toBe(200)
    
    const data = (await response.json()) as Record<string, any>
    expect(data).toHaveProperty('success', true)
    expect(data).toHaveProperty('synced_at')
    expect(typeof data.synced_at).toBe('number')
    
    // Verify progress data was saved
    expect(mockDB.prepare).toHaveBeenCalledWith(
      expect.stringContaining('INSERT INTO progress')
    )
    
    // Verify quiz scores were saved
    expect(mockDB.prepare).toHaveBeenCalledWith(
      expect.stringContaining('INSERT INTO quiz_scores')
    )
    
    // Verify bookmarks were saved
    expect(mockDB.prepare).toHaveBeenCalledWith(
      expect.stringContaining('INSERT INTO bookmarks')
    )
    
    // Verify all items were processed
    const progressCalls = mockDB.prepare.mock.calls.filter((call: any) => 
      call[0].includes('INSERT INTO progress')
    )
    expect(progressCalls.length).toBe(2) // 2 modules
    
    const quizCalls = mockDB.prepare.mock.calls.filter((call: any) => 
      call[0].includes('INSERT INTO quiz_scores')
    )
    expect(quizCalls.length).toBe(1) // 1 quiz
    
    const bookmarkCalls = mockDB.prepare.mock.calls.filter((call: any) => 
      call[0].includes('INSERT INTO bookmarks')
    )
    expect(bookmarkCalls.length).toBe(1) // 1 bookmark
  })
  
  it('should merge with existing data using upsert', async () => {
    // Requirements: 4.4, 4.5, 4.6
    
    // Mock existing progress data
    const existingProgress = [
      {
        id: 'progress-1',
        user_id: testUserId,
        module_id: 'regression/linear-regression',
        completed: false,
        completed_at: null
      }
    ]
    
    mockDB.all.mockResolvedValueOnce({ results: existingProgress })
    
    // Sync updated progress for the same module
    const syncData = {
      modules: [
        {
          module_id: 'regression/linear-regression',
          completed: true,
          completed_at: Date.now()
        }
      ],
      quizzes: [],
      bookmarks: []
    }
    
    const syncRequest = new Request('https://api.test.com/api/progress/sync', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${testJwtToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(syncData)
    })
    
    const syncResponse = await app.fetch(syncRequest, mockEnv)
    
    expect(syncResponse.status).toBe(200)
    
    // Verify upsert was used (INSERT ... ON CONFLICT DO UPDATE)
    expect(mockDB.prepare).toHaveBeenCalledWith(
      expect.stringContaining('ON CONFLICT')
    )
    expect(mockDB.prepare).toHaveBeenCalledWith(
      expect.stringContaining('DO UPDATE')
    )
    
    // Verify the data was bound correctly
    expect(mockDB.bind).toHaveBeenCalledWith(
      expect.any(String), // id
      testUserId,
      'regression/linear-regression',
      1, // completed (true as integer)
      expect.any(Number), // completed_at
      1, // completed for UPDATE clause
      expect.any(Number)  // completed_at for UPDATE clause
    )
  })
  
  it('should handle conflict resolution with timestamps', async () => {
    // Requirements: 4.6
    
    const oldTimestamp = Date.now() - 10000
    const newTimestamp = Date.now()
    
    // Sync data with newer timestamp (should win)
    const syncData = {
      modules: [
        {
          module_id: 'clustering/k-means',
          completed: true,
          completed_at: newTimestamp
        }
      ],
      quizzes: [],
      bookmarks: [
        {
          id: 'bookmark-kmeans-2',
          module_id: 'clustering/k-means',
          name: 'K-Means Bookmark',
          parameters: { clusters: 3 },
          created_at: newTimestamp,
          updated_at: newTimestamp
        }
      ]
    }
    
    const request = new Request('https://api.test.com/api/progress/sync', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${testJwtToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(syncData)
    })
    
    const response = await app.fetch(request, mockEnv)
    
    expect(response.status).toBe(200)
    
    // Verify the newer timestamp was used
    expect(mockDB.bind).toHaveBeenCalledWith(
      expect.any(String),
      testUserId,
      'clustering/k-means',
      1,
      newTimestamp,
      1,
      newTimestamp
    )
    
    // Verify bookmark with newer timestamp
    // saveBookmark binds: id, user_id, module_id, name, parameters,
    // created_at, updated_at, then name, parameters, updated_at for the
    // ON CONFLICT UPDATE clause.
    expect(mockDB.bind).toHaveBeenCalledWith(
      'bookmark-kmeans-2',
      testUserId,
      'clustering/k-means',
      'K-Means Bookmark',
      JSON.stringify({ clusters: 3 }),
      newTimestamp,
      newTimestamp,
      'K-Means Bookmark',
      JSON.stringify({ clusters: 3 }),
      newTimestamp
    )
  })
  
  it('should handle batch operations efficiently', async () => {
    // Requirements: 4.1, 4.2, 4.3, 4.5
    
    // Create a large batch of progress data
    const batchSize = 20
    const syncData = {
      modules: Array.from({ length: batchSize }, (_, i) => ({
        module_id: `domain/module-${i}`,
        completed: i % 2 === 0,
        completed_at: i % 2 === 0 ? Date.now() : null
      })),
      quizzes: Array.from({ length: batchSize }, (_, i) => ({
        module_id: `domain/module-${i}`,
        score: 7 + (i % 4),
        total: 10,
        completed_at: Date.now() - i * 1000
      })),
      bookmarks: Array.from({ length: batchSize }, (_, i) => ({
        id: `bookmark-${i}`,
        module_id: `domain/module-${i}`,
        name: `Bookmark ${i}`,
        parameters: { index: i },
        created_at: Date.now() - i * 2000,
        updated_at: Date.now() - i * 2000
      }))
    }
    
    const request = new Request('https://api.test.com/api/progress/sync', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${testJwtToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(syncData)
    })
    
    const response = await app.fetch(request, mockEnv)
    
    expect(response.status).toBe(200)
    
    const data = (await response.json()) as Record<string, any>
    expect(data.success).toBe(true)
    
    // Verify all items were processed
    const progressCalls = mockDB.prepare.mock.calls.filter((call: any) => 
      call[0].includes('INSERT INTO progress')
    )
    expect(progressCalls.length).toBe(batchSize)
    
    const quizCalls = mockDB.prepare.mock.calls.filter((call: any) => 
      call[0].includes('INSERT INTO quiz_scores')
    )
    expect(quizCalls.length).toBe(batchSize)
    
    const bookmarkCalls = mockDB.prepare.mock.calls.filter((call: any) => 
      call[0].includes('INSERT INTO bookmarks')
    )
    expect(bookmarkCalls.length).toBe(batchSize)
  })
  
  it('should retrieve all progress data for a user', async () => {
    // Requirements: 4.4
    
    const mockProgressData = [
      {
        id: 'progress-1',
        user_id: testUserId,
        module_id: 'regression/linear-regression',
        completed: 1,
        completed_at: Date.now() - 5000
      },
      {
        id: 'progress-2',
        user_id: testUserId,
        module_id: 'classification/logistic-regression',
        completed: 0,
        completed_at: null
      }
    ]
    
    const mockQuizData = [
      {
        id: 'quiz-1',
        user_id: testUserId,
        module_id: 'regression/linear-regression',
        score: 9,
        total: 10,
        completed_at: Date.now() - 3000
      }
    ]
    
    const mockBookmarkData = [
      {
        id: 'bookmark-1',
        user_id: testUserId,
        module_id: 'clustering/k-means',
        created_at: Date.now() - 10000
      }
    ]
    
    // Mock database responses
    mockDB.all
      .mockResolvedValueOnce({ results: mockProgressData })
      .mockResolvedValueOnce({ results: mockQuizData })
      .mockResolvedValueOnce({ results: mockBookmarkData })
    
    const request = new Request('https://api.test.com/api/progress', {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${testJwtToken}`
      }
    })
    
    const response = await app.fetch(request, mockEnv)
    
    expect(response.status).toBe(200)
    
    const data = (await response.json()) as Record<string, any>
    expect(data).toHaveProperty('modules')
    expect(data).toHaveProperty('quizzes')
    expect(data).toHaveProperty('bookmarks')
    
    expect(data.modules).toHaveLength(2)
    expect(data.quizzes).toHaveLength(1)
    expect(data.bookmarks).toHaveLength(1)
    
    // Verify completed is converted to boolean
    expect(data.modules[0].completed).toBe(true)
    expect(data.modules[1].completed).toBe(false)
    
    // Verify database queries were made
    expect(mockDB.prepare).toHaveBeenCalledWith(
      expect.stringContaining('SELECT * FROM progress WHERE user_id = ?')
    )
    expect(mockDB.prepare).toHaveBeenCalledWith(
      expect.stringContaining('SELECT * FROM quiz_scores WHERE user_id = ?')
    )
    expect(mockDB.prepare).toHaveBeenCalledWith(
      expect.stringContaining('SELECT * FROM bookmarks WHERE user_id = ?')
    )
  })
  
  it('should validate module IDs in sync request', async () => {
    // Requirements: 4.5
    
    const syncData = {
      modules: [
        {
          module_id: 'invalid module id with spaces',
          completed: true,
          completed_at: Date.now()
        }
      ],
      quizzes: [],
      bookmarks: []
    }
    
    const request = new Request('https://api.test.com/api/progress/sync', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${testJwtToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(syncData)
    })
    
    const response = await app.fetch(request, mockEnv)
    
    expect(response.status).toBe(400)
    
    const data = (await response.json()) as Record<string, any>
    expect(data).toHaveProperty('code', 'VALIDATION_ERROR')
    expect(data.message).toContain('Invalid module_id')
  })
  
  it('should validate quiz score data types', async () => {
    // Requirements: 4.5
    
    const syncData = {
      modules: [],
      quizzes: [
        {
          module_id: 'regression/linear-regression',
          score: 'not-a-number', // Invalid type
          total: 10,
          completed_at: Date.now()
        }
      ],
      bookmarks: []
    }
    
    const request = new Request('https://api.test.com/api/progress/sync', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${testJwtToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(syncData)
    })
    
    const response = await app.fetch(request, mockEnv)
    
    expect(response.status).toBe(400)
    
    const data = (await response.json()) as Record<string, any>
    expect(data).toHaveProperty('code', 'VALIDATION_ERROR')
    expect(data.message).toContain('score')
  })
  
  it('should require authentication for sync operations', async () => {
    // Requirements: 4.5
    
    const syncData = {
      modules: [],
      quizzes: [],
      bookmarks: []
    }
    
    // Request without Authorization header
    const request = new Request('https://api.test.com/api/progress/sync', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(syncData)
    })
    
    const response = await app.fetch(request, mockEnv)
    
    expect(response.status).toBe(401)
    
    const data = (await response.json()) as Record<string, any>
    expect(data).toHaveProperty('code', 'UNAUTHORIZED')
  })
  
  it('should require authentication for progress retrieval', async () => {
    // Requirements: 4.4
    
    // Request without Authorization header
    const request = new Request('https://api.test.com/api/progress', {
      method: 'GET'
    })
    
    const response = await app.fetch(request, mockEnv)
    
    expect(response.status).toBe(401)
    
    const data = (await response.json()) as Record<string, any>
    expect(data).toHaveProperty('code', 'UNAUTHORIZED')
  })
  
  it('should handle empty sync data gracefully', async () => {
    // Requirements: 4.5
    
    const syncData = {
      modules: [],
      quizzes: [],
      bookmarks: []
    }
    
    const request = new Request('https://api.test.com/api/progress/sync', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${testJwtToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(syncData)
    })
    
    const response = await app.fetch(request, mockEnv)
    
    expect(response.status).toBe(200)
    
    const data = (await response.json()) as Record<string, any>
    expect(data).toHaveProperty('success', true)
    expect(data).toHaveProperty('synced_at')
    
    // Verify no database operations were performed
    const insertCalls = mockDB.prepare.mock.calls.filter((call: any) => 
      call[0].includes('INSERT')
    )
    expect(insertCalls.length).toBe(0)
  })
  
  it('should handle database errors gracefully', async () => {
    // Requirements: 4.5
    
    const syncData = {
      modules: [
        {
          module_id: 'regression/linear-regression',
          completed: true,
          completed_at: Date.now()
        }
      ],
      quizzes: [],
      bookmarks: []
    }
    
    // Create a fresh mock DB that will fail
    const failingDB = {
      prepare: vi.fn().mockReturnThis(),
      bind: vi.fn().mockReturnThis(),
      run: vi.fn().mockRejectedValue(new Error('Database connection failed')),
      first: vi.fn(),
      all: vi.fn()
    }
    
    const failingEnv = {
      ...mockEnv,
      ml_platform_db: failingDB as unknown as D1Database
    }
    
    const request = new Request('https://api.test.com/api/progress/sync', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${testJwtToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(syncData)
    })
    
    const response = await app.fetch(request, failingEnv)
    
    expect(response.status).toBe(500)
    
    const data = (await response.json()) as Record<string, any>
    expect(data).toHaveProperty('code', 'DATABASE_ERROR')
    expect(data).toHaveProperty('message')
  })
})
