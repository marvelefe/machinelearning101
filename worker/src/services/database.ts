/**
 * Database service for D1 operations
 * Requirements: 1.3, 1.4, 2.3, 4.1, 4.2, 4.3, 4.4, 4.5, 6.1, 6.2, 6.3, 6.4, 7.5
 */

import type { User, Progress, QuizScore, Bookmark, BookmarkWithParams, Env } from '../types'
import {
  createDatabaseError,
  createValidationError,
  createNotFoundError,
} from '../utils/errors'

/**
 * Generate a unique ID using crypto
 */
function generateId(): string {
  return crypto.randomUUID()
}

/**
 * User operations
 * Requirements: 1.3, 1.4, 2.3, 6.1, 7.5
 */

/**
 * Create a new user in the database
 * @param db D1 database instance
 * @param userData User data from Google OAuth
 * @returns Created user
 */
export async function createUser(
  db: D1Database,
  userData: {
    google_id: string
    email: string
    name: string
    picture_url: string | null
  }
): Promise<User> {
  const id = generateId()
  const now = Date.now()

  try {
    await db
      .prepare(
        `INSERT INTO users (id, google_id, email, name, picture_url, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)`
      )
      .bind(id, userData.google_id, userData.email, userData.name, userData.picture_url, now, now)
      .run()

    return {
      id,
      google_id: userData.google_id,
      email: userData.email,
      name: userData.name,
      picture_url: userData.picture_url,
      created_at: now,
      updated_at: now,
    }
  } catch (error) {
    throw createDatabaseError('Failed to create user', error)
  }
}

/**
 * Get user by Google ID
 * @param db D1 database instance
 * @param googleId Google user ID
 * @returns User or null if not found
 */
export async function getUserByGoogleId(db: D1Database, googleId: string): Promise<User | null> {
  try {
    const result = await db
      .prepare('SELECT * FROM users WHERE google_id = ?')
      .bind(googleId)
      .first<User>()

    return result
  } catch (error) {
    console.error(error)
    throw createDatabaseError('Failed to get user by Google ID', error)
  }
}

/**
 * Get user by user ID
 * @param db D1 database instance
 * @param userId User ID
 * @returns User or null if not found
 */
export async function getUserById(db: D1Database, userId: string): Promise<User | null> {
  try {
    const result = await db
      .prepare('SELECT * FROM users WHERE id = ?')
      .bind(userId)
      .first<User>()

    return result
  } catch (error) {
    throw createDatabaseError('Failed to get user by ID', error)
  }
}

/**
 * Update user information
 * @param db D1 database instance
 * @param userId User ID
 * @param updates User data to update
 * @returns Updated user
 */
export async function updateUser(
  db: D1Database,
  userId: string,
  updates: {
    email?: string
    name?: string
    picture_url?: string | null
  }
): Promise<User> {
  const now = Date.now()

  try {
    // Build dynamic update query based on provided fields
    const fields: string[] = []
    const values: any[] = []

    if (updates.email !== undefined) {
      fields.push('email = ?')
      values.push(updates.email)
    }
    if (updates.name !== undefined) {
      fields.push('name = ?')
      values.push(updates.name)
    }
    if (updates.picture_url !== undefined) {
      fields.push('picture_url = ?')
      values.push(updates.picture_url)
    }

    fields.push('updated_at = ?')
    values.push(now)
    values.push(userId)

    await db
      .prepare(`UPDATE users SET ${fields.join(', ')} WHERE id = ?`)
      .bind(...values)
      .run()

    const user = await getUserById(db, userId)
    if (!user) {
      throw createNotFoundError('User not found after update')
    }

    return user
  } catch (error) {
    if ((error as any).code) {
      throw error
    }
    throw createDatabaseError('Failed to update user', error)
  }
}

/**
 * Progress operations
 * Requirements: 4.1, 4.4, 4.5, 6.2, 7.5
 */

/**
 * Save or update progress data (upsert)
 * @param db D1 database instance
 * @param userId User ID
 * @param moduleId Module ID
 * @param completed Whether module is completed
 * @param completedAt Completion timestamp
 * @returns Saved progress
 */
export async function saveProgress(
  db: D1Database,
  userId: string,
  moduleId: string,
  completed: boolean,
  completedAt: number | null
): Promise<Progress> {
  const id = generateId()

  try {
    // Use INSERT OR REPLACE for upsert behavior
    await db
      .prepare(
        `INSERT INTO progress (id, user_id, module_id, completed, completed_at)
         VALUES (?, ?, ?, ?, ?)
         ON CONFLICT(user_id, module_id) 
         DO UPDATE SET completed = ?, completed_at = ?`
      )
      .bind(id, userId, moduleId, completed ? 1 : 0, completedAt, completed ? 1 : 0, completedAt)
      .run()

    return {
      id,
      user_id: userId,
      module_id: moduleId,
      completed,
      completed_at: completedAt,
    }
  } catch (error) {
    throw createDatabaseError('Failed to save progress', error)
  }
}

/**
 * Get all progress data for a user
 * @param db D1 database instance
 * @param userId User ID
 * @returns Array of progress records
 */
export async function getProgressByUserId(db: D1Database, userId: string): Promise<Progress[]> {
  try {
    const result = await db
      .prepare('SELECT * FROM progress WHERE user_id = ? ORDER BY completed_at DESC')
      .bind(userId)
      .all<Progress>()

    // Convert completed from integer to boolean
    return (result.results || []).map((p) => ({
      ...p,
      completed: Boolean(p.completed),
    }))
  } catch (error) {
    throw createDatabaseError('Failed to get progress', error)
  }
}

/**
 * Quiz score operations
 * Requirements: 4.2, 4.4, 4.5, 6.3, 7.5
 */

/**
 * Save a quiz score
 * @param db D1 database instance
 * @param userId User ID
 * @param moduleId Module ID
 * @param score Score achieved
 * @param total Total possible score
 * @param completedAt Completion timestamp
 * @returns Saved quiz score
 */
export async function saveQuizScore(
  db: D1Database,
  userId: string,
  moduleId: string,
  score: number,
  total: number,
  completedAt: number
): Promise<QuizScore> {
  const id = generateId()

  try {
    await db
      .prepare(
        `INSERT INTO quiz_scores (id, user_id, module_id, score, total, completed_at)
         VALUES (?, ?, ?, ?, ?, ?)`
      )
      .bind(id, userId, moduleId, score, total, completedAt)
      .run()

    return {
      id,
      user_id: userId,
      module_id: moduleId,
      score,
      total,
      completed_at: completedAt,
    }
  } catch (error) {
    throw createDatabaseError('Failed to save quiz score', error)
  }
}

/**
 * Get all quiz scores for a user
 * @param db D1 database instance
 * @param userId User ID
 * @returns Array of quiz score records
 */
export async function getQuizScoresByUserId(db: D1Database, userId: string): Promise<QuizScore[]> {
  try {
    const result = await db
      .prepare('SELECT * FROM quiz_scores WHERE user_id = ? ORDER BY completed_at DESC')
      .bind(userId)
      .all<QuizScore>()

    return result.results || []
  } catch (error) {
    throw createDatabaseError('Failed to get quiz scores', error)
  }
}

/**
 * Bookmark operations
 * Requirements: 4.3, 4.4, 4.5, 6.4, 7.5
 */

/**
 * Save or update a bookmark (upsert)
 * Requirements: 4.1, 4.3, 6.2
 * @param db D1 database instance
 * @param userId User ID
 * @param bookmarkId Bookmark ID
 * @param moduleId Module ID
 * @param name Bookmark name
 * @param parameters Parameter values object
 * @param createdAt Creation timestamp
 * @param updatedAt Update timestamp
 * @returns Saved bookmark
 */
export async function saveBookmark(
  db: D1Database,
  userId: string,
  bookmarkId: string,
  moduleId: string,
  name: string,
  parameters: Record<string, any>,
  createdAt: number,
  updatedAt: number
): Promise<Bookmark> {
  try {
    // Serialize parameters to JSON string
    const parametersJson = JSON.stringify(parameters)
    
    // Validate JSON size is under 10KB
    const jsonSize = new TextEncoder().encode(parametersJson).length
    if (jsonSize > 10240) {
      throw createValidationError('Parameters exceed 10KB limit', { size: jsonSize, limit: 10240 })
    }

    console.log("Initiating insert")

    // Use INSERT with ON CONFLICT to handle upserts by bookmark ID
    await db
      .prepare(
        `INSERT INTO bookmarks (id, user_id, module_id, name, parameters, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) 
         DO UPDATE SET name = ?, parameters = ?, updated_at = ?`
      )
      .bind(
        bookmarkId,
        userId,
        moduleId,
        name,
        parametersJson,
        createdAt,
        updatedAt,
        name,
        parametersJson,
        updatedAt
      )
      .run()

    return {
      id: bookmarkId,
      user_id: userId,
      module_id: moduleId,
      name,
      parameters: parametersJson,
      created_at: createdAt,
      updated_at: updatedAt,
    }
  } catch (error) {
    console.error(error)
    if ((error as any).code) {
      throw error
    }
    throw createDatabaseError('Failed to save bookmark', error)
  }
}

/**
 * Update an existing bookmark
 * Requirements: 3.1, 3.2
 * @param db D1 database instance
 * @param userId User ID
 * @param bookmarkId Bookmark ID
 * @param updates Object containing fields to update (name, parameters)
 * @returns Updated bookmark
 */
export async function updateBookmark(
  db: D1Database,
  userId: string,
  bookmarkId: string,
  updates: {
    name?: string
    parameters?: Record<string, any>
  }
): Promise<Bookmark> {
  const now = Date.now()

  try {
    // Build dynamic UPDATE query based on provided fields
    const fields: string[] = []
    const values: any[] = []

    if (updates.name !== undefined) {
      fields.push('name = ?')
      values.push(updates.name)
    }
    
    if (updates.parameters !== undefined) {
      const parametersJson = JSON.stringify(updates.parameters)
      
      // Validate JSON size is under 10KB
      const jsonSize = new TextEncoder().encode(parametersJson).length
      if (jsonSize > 10240) {
        throw createValidationError('Parameters exceed 10KB limit', { size: jsonSize, limit: 10240 })
      }
      
      fields.push('parameters = ?')
      values.push(parametersJson)
    }

    // Always update updated_at timestamp
    fields.push('updated_at = ?')
    values.push(now)
    
    // Add WHERE clause parameters
    values.push(bookmarkId)
    values.push(userId)

    await db
      .prepare(`UPDATE bookmarks SET ${fields.join(', ')} WHERE id = ? AND user_id = ?`)
      .bind(...values)
      .run()

    // Fetch and return the updated bookmark
    const result = await db
      .prepare('SELECT * FROM bookmarks WHERE id = ? AND user_id = ?')
      .bind(bookmarkId, userId)
      .first<Bookmark>()

    if (!result) {
      throw createNotFoundError('Bookmark not found')
    }

    return result
  } catch (error) {
    if ((error as any).code) {
      throw error
    }
    throw createDatabaseError('Failed to update bookmark', error)
  }
}

/**
 * Get all bookmarks for a user
 * Requirements: 4.4, 5.1
 * @param db D1 database instance
 * @param userId User ID
 * @returns Array of bookmark records with deserialized parameters
 */
export async function getBookmarksByUserId(db: D1Database, userId: string): Promise<BookmarkWithParams[]> {
  try {
    const result = await db
      .prepare('SELECT * FROM bookmarks WHERE user_id = ? ORDER BY created_at DESC')
      .bind(userId)
      .all<Bookmark>()

    // Deserialize parameters JSON string to object
    const bookmarks = (result.results || []).map((bookmark) => {
      try {
        // Handle JSON parse errors gracefully
        const parameters = JSON.parse(bookmark.parameters)
        return {
          ...bookmark,
          parameters,
        }
      } catch (parseError) {
        console.error(`Failed to parse parameters for bookmark ${bookmark.id}:`, parseError)
        // Return bookmark with empty parameters object if parse fails
        return {
          ...bookmark,
          parameters: {},
        }
      }
    })

    return bookmarks
  } catch (error) {
    throw createDatabaseError('Failed to get bookmarks', error)
  }
}

/**
 * Delete a bookmark
 * Requirements: 3.1, 3.4
 * @param db D1 database instance
 * @param userId User ID
 * @param bookmarkId Bookmark ID
 * @returns True if deleted, false if not found
 */
export async function deleteBookmark(
  db: D1Database,
  userId: string,
  bookmarkId: string
): Promise<boolean> {
  try {
    const result = await db
      .prepare('DELETE FROM bookmarks WHERE id = ? AND user_id = ?')
      .bind(bookmarkId, userId)
      .run()

    return (result.meta?.changes || 0) > 0
  } catch (error) {
    throw createDatabaseError('Failed to delete bookmark', error)
  }
}
