/**
 * Unit tests for validation utilities
 * Requirements: 7.4, 7.5
 */

import { describe, it, expect } from 'vitest'
import {
  sanitizeString,
  isValidEmail,
  isValidModuleId,
  validateProgressData,
  isValidUUID,
  isValidTimestamp,
  sanitizeRequestBody
} from '../../src/utils/validation'

describe('sanitizeString', () => {
  it('should remove null bytes', () => {
    const input = 'hello\0world'
    const result = sanitizeString(input)
    expect(result).toBe('helloworld')
  })

  it('should trim whitespace', () => {
    const input = '  hello world  '
    const result = sanitizeString(input)
    expect(result).toBe('hello world')
  })

  it('should handle empty strings', () => {
    expect(sanitizeString('')).toBe('')
  })

  it('should handle non-string input', () => {
    expect(sanitizeString(123 as any)).toBe('')
    expect(sanitizeString(null as any)).toBe('')
    expect(sanitizeString(undefined as any)).toBe('')
  })

  it('should detect SQL injection patterns', () => {
    const maliciousInputs = [
      "1' OR '1'='1",
      "admin'--",
      "1; DROP TABLE users",
      "' UNION SELECT * FROM users--"
    ]
    
    for (const input of maliciousInputs) {
      const result = sanitizeString(input)
      // Should return empty string for suspicious SQL patterns
      expect(result).toBe('')
    }
  })

  it('should allow normal text with SQL keywords in safe context', () => {
    const safeInputs = [
      'I want to select a module',
      'Please update my profile',
      'Create a new bookmark'
    ]
    
    for (const input of safeInputs) {
      const result = sanitizeString(input)
      expect(result).toBe(input)
    }
  })

  it('should handle special characters safely', () => {
    const input = 'hello@world.com'
    const result = sanitizeString(input)
    expect(result).toBe('hello@world.com')
  })
})

describe('isValidEmail', () => {
  it('should validate correct email addresses', () => {
    const validEmails = [
      'user@example.com',
      'test.user@example.com',
      'user+tag@example.co.uk',
      'user_name@example-domain.com',
      'a@b.co'
    ]
    
    for (const email of validEmails) {
      expect(isValidEmail(email)).toBe(true)
    }
  })

  it('should reject invalid email addresses', () => {
    const invalidEmails = [
      '',
      'notanemail',
      '@example.com',
      'user@',
      'user @example.com',
      'user@example',
      'user..name@example.com',
      'user@.example.com',
      'user@example..com'
    ]
    
    for (const email of invalidEmails) {
      expect(isValidEmail(email), `Expected ${email} to be invalid`).toBe(false)
    }
  })

  it('should reject emails that are too long', () => {
    const longEmail = 'a'.repeat(250) + '@example.com'
    expect(isValidEmail(longEmail)).toBe(false)
  })

  it('should reject emails with local part too long', () => {
    const longLocalPart = 'a'.repeat(65) + '@example.com'
    expect(isValidEmail(longLocalPart)).toBe(false)
  })

  it('should handle non-string input', () => {
    expect(isValidEmail(123 as any)).toBe(false)
    expect(isValidEmail(null as any)).toBe(false)
    expect(isValidEmail(undefined as any)).toBe(false)
  })
})

describe('isValidModuleId', () => {
  it('should validate correct module IDs', () => {
    const validIds = [
      // Bare module IDs — this is what the app actually sends
      // (e.g. `loadQuiz('decision-trees', ...)`, config.json "id")
      'linear-regression',
      'k-means',
      'attention-mechanisms',
      'decision-trees',
      // Nested domain/module form
      'regression/linear-regression',
      'clustering/k-means',
      'preprocessing/feature-scaling',
      'advanced/neural-network',
      'optimization/gradient-descent'
    ]
    
    for (const id of validIds) {
      expect(isValidModuleId(id)).toBe(true)
    }
  })

  it('should reject invalid module IDs', () => {
    const invalidIds = [
      '',
      // Bare IDs are valid, but these bare forms are not kebab-case:
      'Invalid',
      'has_underscore',
      'double--hyphen',
      '-leading-hyphen',
      'trailing-hyphen-',
      'Invalid/Module',
      'module/with spaces',
      'module/with_underscore',
      'module//double-slash',
      'module/--double-hyphen',
      '-module/name',
      'module/-name',
      'module/name-',
      'module/name/',
      '/module/name',
      'module/name/extra'
    ]
    
    for (const id of invalidIds) {
      expect(isValidModuleId(id)).toBe(false)
    }
  })

  it('should reject module IDs that are too long', () => {
    const longId = 'a'.repeat(50) + '/' + 'b'.repeat(50)
    expect(isValidModuleId(longId)).toBe(false)
  })

  it('should handle non-string input', () => {
    expect(isValidModuleId(123 as any)).toBe(false)
    expect(isValidModuleId(null as any)).toBe(false)
    expect(isValidModuleId(undefined as any)).toBe(false)
  })
})

describe('validateProgressData', () => {
  it('should validate correct progress data', () => {
    const validData = {
      modules: [
        {
          // Bare module ID — the shape the app actually sends
          module_id: 'linear-regression',
          completed: true,
          completed_at: Date.now()
        }
      ],
      quizzes: [
        {
          module_id: 'decision-trees',
          score: 8,
          total: 10,
          completed_at: Date.now()
        }
      ],
      bookmarks: [
        {
          module_id: 'clustering/k-means',
          created_at: Date.now()
        }
      ]
    }
    
    expect(validateProgressData(validData)).toBe(true)
  })

  it('should validate empty arrays', () => {
    const validData = {
      modules: [],
      quizzes: [],
      bookmarks: []
    }
    
    expect(validateProgressData(validData)).toBe(true)
  })

  it('should reject missing required fields', () => {
    expect(validateProgressData({})).toBe(false)
    expect(validateProgressData({ modules: [] })).toBe(false)
    expect(validateProgressData({ modules: [], quizzes: [] })).toBe(false)
  })

  it('should reject invalid module data', () => {
    const invalidData = {
      modules: [
        {
          // Spaces are not valid in module IDs
          module_id: 'invalid id',
          completed: true,
          completed_at: Date.now()
        }
      ],
      quizzes: [],
      bookmarks: []
    }
    
    expect(validateProgressData(invalidData)).toBe(false)
  })

  it('should reject invalid quiz data', () => {
    const invalidData = {
      modules: [],
      quizzes: [
        {
          module_id: 'regression/linear-regression',
          score: 11,
          total: 10,
          completed_at: Date.now()
        }
      ],
      bookmarks: []
    }
    
    expect(validateProgressData(invalidData)).toBe(false)
  })

  it('should reject negative scores', () => {
    const invalidData = {
      modules: [],
      quizzes: [
        {
          module_id: 'regression/linear-regression',
          score: -1,
          total: 10,
          completed_at: Date.now()
        }
      ],
      bookmarks: []
    }
    
    expect(validateProgressData(invalidData)).toBe(false)
  })

  it('should reject invalid bookmark data', () => {
    const invalidData = {
      modules: [],
      quizzes: [],
      bookmarks: [
        {
          // Spaces are not valid in module IDs
          module_id: 'invalid id',
          created_at: Date.now()
        }
      ]
    }
    
    expect(validateProgressData(invalidData)).toBe(false)
  })

  it('should reject non-object input', () => {
    expect(validateProgressData(null)).toBe(false)
    expect(validateProgressData(undefined)).toBe(false)
    expect(validateProgressData('string')).toBe(false)
    expect(validateProgressData(123)).toBe(false)
  })

  it('should reject invalid timestamps', () => {
    const invalidData = {
      modules: [
        {
          module_id: 'regression/linear-regression',
          completed: true,
          completed_at: -1
        }
      ],
      quizzes: [],
      bookmarks: []
    }
    
    expect(validateProgressData(invalidData)).toBe(false)
  })
})

describe('isValidUUID', () => {
  it('should validate correct UUIDs', () => {
    const validUUIDs = [
      '123e4567-e89b-12d3-a456-426614174000',
      '550e8400-e29b-41d4-a716-446655440000',
      'f47ac10b-58cc-4372-a567-0e02b2c3d479'
    ]
    
    for (const uuid of validUUIDs) {
      expect(isValidUUID(uuid)).toBe(true)
    }
  })

  it('should reject invalid UUIDs', () => {
    const invalidUUIDs = [
      '',
      'not-a-uuid',
      '123e4567-e89b-12d3-a456',
      '123e4567-e89b-12d3-a456-42661417400g',
      '123e4567e89b12d3a456426614174000',
      '123e4567-e89b-12d3-a456-4266141740000'
    ]
    
    for (const uuid of invalidUUIDs) {
      expect(isValidUUID(uuid)).toBe(false)
    }
  })

  it('should handle non-string input', () => {
    expect(isValidUUID(123 as any)).toBe(false)
    expect(isValidUUID(null as any)).toBe(false)
    expect(isValidUUID(undefined as any)).toBe(false)
  })
})

describe('isValidTimestamp', () => {
  it('should validate current timestamps', () => {
    const now = Date.now()
    expect(isValidTimestamp(now)).toBe(true)
  })

  it('should validate timestamps in valid range', () => {
    const year2021 = new Date('2021-01-01').getTime()
    const year2030 = new Date('2030-01-01').getTime()
    
    expect(isValidTimestamp(year2021)).toBe(true)
    expect(isValidTimestamp(year2030)).toBe(true)
  })

  it('should reject timestamps outside valid range', () => {
    const year2019 = new Date('2019-01-01').getTime()
    const year2101 = new Date('2101-01-01').getTime()
    
    expect(isValidTimestamp(year2019)).toBe(false)
    expect(isValidTimestamp(year2101)).toBe(false)
  })

  it('should reject negative timestamps', () => {
    expect(isValidTimestamp(-1)).toBe(false)
  })

  it('should handle non-number input', () => {
    expect(isValidTimestamp('123' as any)).toBe(false)
    expect(isValidTimestamp(null as any)).toBe(false)
    expect(isValidTimestamp(undefined as any)).toBe(false)
  })
})

describe('sanitizeRequestBody', () => {
  it('should sanitize string fields', () => {
    const body = {
      name: '  John Doe  ',
      email: 'john@example.com'
    }
    
    const result = sanitizeRequestBody<{ name: string; email: string }>(body, ['name', 'email'])
    expect(result.name).toBe('John Doe')
    expect(result.email).toBe('john@example.com')
  })

  it('should preserve non-string fields', () => {
    const body = {
      name: 'John',
      age: 30,
      active: true
    }
    
    const result = sanitizeRequestBody<{ name: string; age: number; active: boolean }>(body, ['name'])
    expect(result.age).toBe(30)
    expect(result.active).toBe(true)
  })

  it('should throw error for missing required fields', () => {
    const body = {
      name: 'John'
    }
    
    expect(() => {
      sanitizeRequestBody(body, ['name', 'email'])
    }).toThrow('Missing required field: email')
  })

  it('should throw error for invalid body', () => {
    expect(() => {
      sanitizeRequestBody(null, [])
    }).toThrow('Invalid request body')
    
    expect(() => {
      sanitizeRequestBody(undefined, [])
    }).toThrow('Invalid request body')
    
    expect(() => {
      sanitizeRequestBody('string', [])
    }).toThrow('Invalid request body')
  })

  it('should handle empty required fields array', () => {
    const body = {
      name: 'John'
    }
    
    const result = sanitizeRequestBody<{ name: string }>(body, [])
    expect(result.name).toBe('John')
  })

  it('should sanitize SQL injection attempts in strings', () => {
    const body = {
      name: "John'; DROP TABLE users--"
    }
    
    const result = sanitizeRequestBody<{ name: string }>(body, ['name'])
    expect(result.name).toBe('')
  })
})

describe('SQL injection prevention', () => {
  it('should detect common SQL injection patterns', () => {
    const sqlInjectionAttempts = [
      "1' OR '1'='1",
      "admin'--",
      "1; DROP TABLE users",
      "' UNION SELECT * FROM users--",
      "1' AND 1=1--",
      "' OR 1=1--",
      "admin' OR '1'='1'--",
      "1' EXEC sp_executesql--"
    ]
    
    for (const attempt of sqlInjectionAttempts) {
      const sanitized = sanitizeString(attempt)
      expect(sanitized).toBe('')
    }
  })

  it('should allow legitimate strings with SQL keywords', () => {
    const legitimateStrings = [
      'I want to select a course',
      'Please update my information',
      'Create a new account',
      'Delete my old bookmarks',
      'Insert a new record'
    ]
    
    for (const str of legitimateStrings) {
      const sanitized = sanitizeString(str)
      expect(sanitized).toBe(str)
    }
  })
})
