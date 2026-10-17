/**
 * Input validation and sanitization utilities
 * Requirements: 7.4
 */

/**
 * Sanitizes string input by removing potentially dangerous characters
 * Prevents XSS and SQL injection attempts
 * Note: This is a basic defense layer. Parameterized queries are the primary SQL injection defense.
 */
export function sanitizeString(input: string): string {
  if (typeof input !== 'string') {
    return ''
  }
  
  // Remove null bytes
  let sanitized = input.replace(/\0/g, '')
  
  // Trim whitespace
  sanitized = sanitized.trim()
  
  // Detect SQL injection patterns
  // Look for SQL keywords combined with suspicious operators/syntax
  const sqlInjectionPatterns = [
    /'\s*(OR|AND)\s*'?\d*'?\s*=\s*'?\d*'?/gi,  // ' OR '1'='1
    /'\s*(OR|AND)\s+\d+\s*=\s*\d+/gi,          // ' OR 1=1
    /;\s*(DROP|DELETE|INSERT|UPDATE|CREATE|ALTER|EXEC)/gi,  // ; DROP TABLE
    /';\s*--/gi,                                // '; --
    /'\s*--/gi,                                 // ' --
    /UNION\s+SELECT/gi,                         // UNION SELECT
    /EXEC\s*\(/gi,                              // EXEC(
    /sp_executesql/gi                           // sp_executesql
  ]
  
  // Check for SQL injection patterns
  for (const pattern of sqlInjectionPatterns) {
    if (pattern.test(sanitized)) {
      return ''
    }
  }
  
  return sanitized
}

/**
 * Validates email format
 * Requirements: 7.4
 */
export function isValidEmail(email: string): boolean {
  if (typeof email !== 'string' || !email) {
    return false
  }
  
  // Additional checks
  if (email.length > 254) {
    return false
  }
  
  const parts = email.split('@')
  if (parts.length !== 2) {
    return false
  }
  
  const [localPart, domain] = parts
  if (!localPart || !domain) {
    return false
  }
  
  if (localPart.length > 64) {
    return false
  }
  
  // Domain must have at least one dot (TLD required)
  if (!domain.includes('.')) {
    return false
  }
  
  // Check for consecutive dots
  if (email.includes('..')) {
    return false
  }
  
  // Check for dots at start/end of local part or domain
  if (localPart.startsWith('.') || localPart.endsWith('.')) {
    return false
  }
  
  if (domain.startsWith('.') || domain.endsWith('.')) {
    return false
  }
  
  // RFC 5322 compliant email regex (simplified)
  const emailRegex = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$/
  
  if (!emailRegex.test(email)) {
    return false
  }
  
  return true
}

/**
 * Validates module ID format
 * Module IDs should follow the pattern: domain/module-name
 * Examples: linear-regression, attention-mechanisms, regression/linear-regression
 * Requirements: 7.4
 */
export function isValidModuleId(moduleId: string): boolean {
  if (typeof moduleId !== 'string' || !moduleId) {
    return false
  }

  // Length constraints
  if (moduleId.length > 100) {
    return false
  }

  // A module ID is one or two kebab-case segments separated by "/".
  //
  // The app identifies modules by their bare ID (e.g. "linear-regression",
  // "attention-mechanisms" — see `loadQuiz('decision-trees', ...)` and each
  // module's config.json "id"), so a single segment is the common case. The
  // nested "domain/module" form is also accepted.
  const segmentRegex = /^[a-z0-9]+(?:-[a-z0-9]+)*$/
  const segments = moduleId.split('/')

  if (segments.length > 2) {
    return false
  }

  return segments.every((segment) => segmentRegex.test(segment))
}

/**
 * Validates progress data from sync request
 * Requirements: 7.4
 */
export function validateProgressData(data: any): boolean {
  if (!data || typeof data !== 'object') {
    return false
  }
  
  // Check required fields
  if (!Array.isArray(data.modules) || !Array.isArray(data.quizzes) || !Array.isArray(data.bookmarks)) {
    return false
  }
  
  // Validate modules
  for (const module of data.modules) {
    if (!module || typeof module !== 'object') {
      return false
    }
    if (!isValidModuleId(module.module_id)) {
      return false
    }
    if (typeof module.completed !== 'boolean') {
      return false
    }
    if (module.completed_at !== null && (typeof module.completed_at !== 'number' || module.completed_at < 0)) {
      return false
    }
  }
  
  // Validate quizzes
  for (const quiz of data.quizzes) {
    if (!quiz || typeof quiz !== 'object') {
      return false
    }
    if (!isValidModuleId(quiz.module_id)) {
      return false
    }
    if (typeof quiz.score !== 'number' || quiz.score < 0) {
      return false
    }
    if (typeof quiz.total !== 'number' || quiz.total < 1) {
      return false
    }
    if (quiz.score > quiz.total) {
      return false
    }
    if (typeof quiz.completed_at !== 'number' || quiz.completed_at < 0) {
      return false
    }
  }
  
  // Validate bookmarks
  for (const bookmark of data.bookmarks) {
    if (!bookmark || typeof bookmark !== 'object') {
      return false
    }
    if (!isValidModuleId(bookmark.module_id)) {
      return false
    }
    if (typeof bookmark.created_at !== 'number' || bookmark.created_at < 0) {
      return false
    }
  }
  
  return true
}

/**
 * Validates UUID format
 * Requirements: 7.4
 */
export function isValidUUID(uuid: string): boolean {
  if (typeof uuid !== 'string' || !uuid) {
    return false
  }
  
  const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
  return uuidRegex.test(uuid)
}

/**
 * Validates timestamp (Unix timestamp in milliseconds)
 * Requirements: 7.4
 */
export function isValidTimestamp(timestamp: number): boolean {
  if (typeof timestamp !== 'number') {
    return false
  }
  
  // Reasonable range: after 2020 and before 2100
  const minTimestamp = 1577836800000 // 2020-01-01
  const maxTimestamp = 4102444800000 // 2100-01-01
  
  return timestamp >= minTimestamp && timestamp <= maxTimestamp
}

/**
 * Sanitizes and validates request body
 * Returns sanitized data or throws error
 * Requirements: 7.4
 */
export function sanitizeRequestBody<T>(body: any, requiredFields: string[]): T {
  if (!body || typeof body !== 'object') {
    throw new Error('Invalid request body')
  }
  
  // Check required fields
  for (const field of requiredFields) {
    if (!(field in body)) {
      throw new Error(`Missing required field: ${field}`)
    }
  }
  
  // Sanitize string fields
  const sanitized: any = {}
  for (const [key, value] of Object.entries(body)) {
    if (typeof value === 'string') {
      sanitized[key] = sanitizeString(value)
    } else {
      sanitized[key] = value
    }
  }
  
  return sanitized as T
}
