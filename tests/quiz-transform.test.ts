import { describe, it, expect, vi } from 'vitest';
import { transformQuizData, safeTransformQuizData, QuizValidationError } from '../utils/quiz-transform';
import type { Quiz } from '../types/quiz';

/**
 * Quiz Transformation Utility Tests
 * 
 * Tests data transformation, validation, and error handling
 * for quiz.json data conversion to Quiz interface format.
 * 
 * Requirements: 2.2, 2.4, 2.5, 6.1, 6.2, 6.4
 */

describe('Quiz Transformation Utility', () => {
  describe('9.1 Data Transformation', () => {
    describe('New Format (text field)', () => {
      it('should transform valid quiz data with text field', () => {
        // Arrange
        const rawData = {
          id: 'test-quiz',
          title: 'Test Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'What is 2 + 2?',
              options: ['2', '3', '4', '5'],
              correctAnswer: 2,
              explanation: 'The answer is 4',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act
        const result = transformQuizData(rawData, moduleId);

        // Assert
        expect(result).toBeDefined();
        expect(result.id).toBe('test-quiz');
        expect(result.moduleId).toBe('test-module');
        expect(result.title).toBe('Test Quiz');
        expect(result.questions).toHaveLength(1);
        expect(result.questions[0].text).toBe('What is 2 + 2?');
        expect(result.questions[0].type).toBe('multiple-choice');
        expect(result.questions[0].options).toEqual(['2', '3', '4', '5']);
        expect(result.questions[0].correctAnswer).toBe(2);
        expect(result.questions[0].explanation).toBe('The answer is 4');
      });

      it('should transform multiple questions correctly', () => {
        // Arrange
        const rawData = {
          title: 'Multi-Question Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Question 1?',
              options: ['A', 'B', 'C'],
              correctAnswer: 0,
              explanation: 'Explanation 1',
            },
            {
              id: 'q2',
              type: 'true-false' as const,
              text: 'Question 2?',
              options: ['True', 'False'],
              correctAnswer: 1,
              explanation: 'Explanation 2',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act
        const result = transformQuizData(rawData, moduleId);

        // Assert
        expect(result.questions).toHaveLength(2);
        expect(result.questions[0].id).toBe('q1');
        expect(result.questions[1].id).toBe('q2');
        expect(result.questions[0].type).toBe('multiple-choice');
        expect(result.questions[1].type).toBe('true-false');
      });
    });

    describe('Old Format (question field)', () => {
      it('should transform quiz data with question field (old format)', () => {
        // Arrange
        const rawData = {
          title: 'Old Format Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              question: 'What is the capital of France?', // Old format
              options: ['London', 'Paris', 'Berlin', 'Madrid'],
              correctAnswer: 1,
              explanation: 'Paris is the capital of France',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act
        const result = transformQuizData(rawData, moduleId);

        // Assert
        expect(result.questions[0].text).toBe('What is the capital of France?');
      });

      it('should prefer text field over question field when both exist', () => {
        // Arrange
        const rawData = {
          title: 'Mixed Format Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'New format text',
              question: 'Old format question',
              options: ['A', 'B'],
              correctAnswer: 0,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act
        const result = transformQuizData(rawData, moduleId);

        // Assert
        expect(result.questions[0].text).toBe('New format text');
      });
    });

    describe('Missing Fields Handling', () => {
      it('should generate quiz ID from moduleId when id is missing', () => {
        // Arrange
        const rawData = {
          title: 'Quiz Without ID',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Question?',
              options: ['A', 'B'],
              correctAnswer: 0,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'linear-regression';

        // Act
        const result = transformQuizData(rawData, moduleId);

        // Assert
        expect(result.id).toBe('linear-regression-quiz');
      });

      it('should ignore optional fields like description and passingScore', () => {
        // Arrange
        const rawData = {
          id: 'test-quiz',
          title: 'Quiz With Extra Fields',
          description: 'This is a description',
          passingScore: 70,
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Question?',
              options: ['A', 'B'],
              correctAnswer: 0,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act
        const result = transformQuizData(rawData, moduleId);

        // Assert
        expect(result).toBeDefined();
        expect(result.id).toBe('test-quiz');
        expect(result.title).toBe('Quiz With Extra Fields');
        // description and passingScore should not be in the result
        expect('description' in result).toBe(false);
        expect('passingScore' in result).toBe(false);
      });
    });

    describe('Quiz Data Structure Validation', () => {
      it('should validate true-false questions have exactly 2 options', () => {
        // Arrange
        const rawData = {
          title: 'True-False Quiz',
          questions: [
            {
              id: 'q1',
              type: 'true-false' as const,
              text: 'Is this true?',
              options: ['True', 'False'],
              correctAnswer: 0,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act
        const result = transformQuizData(rawData, moduleId);

        // Assert
        expect(result.questions[0].options).toHaveLength(2);
      });

      it('should validate multiple-choice questions can have any number of options', () => {
        // Arrange
        const rawData = {
          title: 'Multiple Choice Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Pick one',
              options: ['A', 'B', 'C', 'D', 'E'],
              correctAnswer: 2,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act
        const result = transformQuizData(rawData, moduleId);

        // Assert
        expect(result.questions[0].options).toHaveLength(5);
      });

      it('should validate correctAnswer is within options bounds', () => {
        // Arrange
        const rawData = {
          title: 'Valid Answer Index Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Question?',
              options: ['A', 'B', 'C'],
              correctAnswer: 2, // Last valid index
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act
        const result = transformQuizData(rawData, moduleId);

        // Assert
        expect(result.questions[0].correctAnswer).toBe(2);
        expect(result.questions[0].correctAnswer).toBeLessThan(result.questions[0].options.length);
      });

      it('should validate all options are strings', () => {
        // Arrange
        const rawData = {
          title: 'String Options Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Question?',
              options: ['Option 1', 'Option 2', 'Option 3'],
              correctAnswer: 0,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act
        const result = transformQuizData(rawData, moduleId);

        // Assert
        result.questions[0].options.forEach(option => {
          expect(typeof option).toBe('string');
        });
      });
    });
  });

  describe('9.2 Error Handling', () => {
    describe('Malformed JSON / Invalid Data', () => {
      it('should throw QuizValidationError for null data', () => {
        // Arrange
        const rawData = null;
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow('Quiz data must be an object');
      });

      it('should throw QuizValidationError for undefined data', () => {
        // Arrange
        const rawData = undefined;
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow('Quiz data must be an object');
      });

      it('should throw QuizValidationError for non-object data', () => {
        // Arrange
        const rawData = 'not an object';
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow('Quiz data must be an object');
      });

      it('should throw QuizValidationError for array data', () => {
        // Arrange
        const rawData = ['not', 'an', 'object'];
        const moduleId = 'test-module';

        // Act & Assert
        // Arrays are objects in JavaScript, so this will fail on missing title
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow('Quiz must have a valid title');
      });

      it('should throw QuizValidationError for empty moduleId', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Question?',
              options: ['A', 'B'],
              correctAnswer: 0,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = '';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow('Module ID must be a non-empty string');
      });

      it('should throw QuizValidationError for missing title', () => {
        // Arrange
        const rawData = {
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Question?',
              options: ['A', 'B'],
              correctAnswer: 0,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow('Quiz must have a valid title');
      });

      it('should throw QuizValidationError for non-string title', () => {
        // Arrange
        const rawData = {
          title: 123,
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Question?',
              options: ['A', 'B'],
              correctAnswer: 0,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow('Quiz must have a valid title');
      });

      it('should throw QuizValidationError for missing questions array', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow('Quiz must have a questions array');
      });

      it('should throw QuizValidationError for non-array questions', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: 'not an array',
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow('Quiz must have a questions array');
      });

      it('should throw QuizValidationError for empty questions array', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [],
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow('Quiz must have at least one question');
      });
    });

    describe('Invalid Question Data', () => {
      it('should throw QuizValidationError for question missing id', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [
            {
              type: 'multiple-choice' as const,
              text: 'Question?',
              options: ['A', 'B'],
              correctAnswer: 0,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow("Missing or invalid 'id' field");
      });

      it('should throw QuizValidationError for question with non-string id', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [
            {
              id: 123,
              type: 'multiple-choice' as const,
              text: 'Question?',
              options: ['A', 'B'],
              correctAnswer: 0,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow("Missing or invalid 'id' field");
      });

      it('should throw QuizValidationError for question missing type', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [
            {
              id: 'q1',
              text: 'Question?',
              options: ['A', 'B'],
              correctAnswer: 0,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow("Invalid 'type'");
      });

      it('should throw QuizValidationError for question with invalid type', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [
            {
              id: 'q1',
              type: 'invalid-type',
              text: 'Question?',
              options: ['A', 'B'],
              correctAnswer: 0,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow("Invalid 'type'");
      });

      it('should throw QuizValidationError for question missing both text and question fields', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              options: ['A', 'B'],
              correctAnswer: 0,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow("Missing question text");
      });

      it('should throw QuizValidationError for question missing options', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Question?',
              correctAnswer: 0,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow("Missing or empty 'options' array");
      });

      it('should throw QuizValidationError for question with empty options array', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Question?',
              options: [],
              correctAnswer: 0,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow("Missing or empty 'options' array");
      });

      it('should throw QuizValidationError for question with non-string options', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Question?',
              options: ['A', 123, 'C'],
              correctAnswer: 0,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow('All options must be strings');
      });

      it('should throw QuizValidationError for question missing correctAnswer', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Question?',
              options: ['A', 'B'],
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow("Missing or invalid 'correctAnswer'");
      });

      it('should throw QuizValidationError for question with non-number correctAnswer', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Question?',
              options: ['A', 'B'],
              correctAnswer: 'A',
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow("Missing or invalid 'correctAnswer'");
      });

      it('should throw QuizValidationError for question with negative correctAnswer', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Question?',
              options: ['A', 'B'],
              correctAnswer: -1,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow('out of bounds');
      });

      it('should throw QuizValidationError for question with correctAnswer out of bounds', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Question?',
              options: ['A', 'B'],
              correctAnswer: 5,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow('out of bounds');
      });

      it('should throw QuizValidationError for question missing explanation', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Question?',
              options: ['A', 'B'],
              correctAnswer: 0,
            },
          ],
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow("Missing or invalid 'explanation' field");
      });

      it('should throw QuizValidationError for question with non-string explanation', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Question?',
              options: ['A', 'B'],
              correctAnswer: 0,
              explanation: 123,
            },
          ],
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow("Missing or invalid 'explanation' field");
      });

      it('should throw QuizValidationError for true-false question with wrong number of options', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [
            {
              id: 'q1',
              type: 'true-false' as const,
              text: 'Is this true?',
              options: ['True', 'False', 'Maybe'],
              correctAnswer: 0,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow('True-false questions must have exactly 2 options');
      });

      it('should include question number and id in error messages', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Valid question',
              options: ['A', 'B'],
              correctAnswer: 0,
              explanation: 'Explanation',
            },
            {
              id: 'q2-invalid',
              type: 'multiple-choice' as const,
              text: 'Invalid question',
              options: ['A', 'B'],
              correctAnswer: 5, // Out of bounds
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act & Assert
        expect(() => transformQuizData(rawData, moduleId)).toThrow(QuizValidationError);
        expect(() => transformQuizData(rawData, moduleId)).toThrow('Question 2');
        expect(() => transformQuizData(rawData, moduleId)).toThrow('q2-invalid');
      });
    });

    describe('Safe Transform Function', () => {
      it('should return null for invalid data instead of throwing', () => {
        // Arrange
        const rawData = null;
        const moduleId = 'test-module';

        // Act
        const result = safeTransformQuizData(rawData, moduleId);

        // Assert
        expect(result).toBeNull();
      });

      it('should return null for malformed quiz data', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: 'not an array',
        };
        const moduleId = 'test-module';

        // Act
        const result = safeTransformQuizData(rawData, moduleId);

        // Assert
        expect(result).toBeNull();
      });

      it('should return valid Quiz object for valid data', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [
            {
              id: 'q1',
              type: 'multiple-choice' as const,
              text: 'Question?',
              options: ['A', 'B'],
              correctAnswer: 0,
              explanation: 'Explanation',
            },
          ],
        };
        const moduleId = 'test-module';

        // Act
        const result = safeTransformQuizData(rawData, moduleId);

        // Assert
        expect(result).not.toBeNull();
        expect(result?.title).toBe('Test Quiz');
        expect(result?.questions).toHaveLength(1);
      });

      it('should log error for QuizValidationError', () => {
        // Arrange
        const rawData = {
          title: 'Test Quiz',
          questions: [],
        };
        const moduleId = 'test-module';
        const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

        // Act
        const result = safeTransformQuizData(rawData, moduleId);

        // Assert
        expect(result).toBeNull();
        expect(consoleErrorSpy).toHaveBeenCalled();
        expect(consoleErrorSpy).toHaveBeenCalledWith(
          expect.stringContaining("Quiz validation error for module 'test-module'"),
          expect.any(String)
        );

        // Cleanup
        consoleErrorSpy.mockRestore();
      });

      it('should log error for unexpected errors', () => {
        // Arrange
        const rawData = {
          get title() {
            throw new Error('Unexpected error');
          },
          questions: [],
        };
        const moduleId = 'test-module';
        const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

        // Act
        const result = safeTransformQuizData(rawData, moduleId);

        // Assert
        expect(result).toBeNull();
        expect(consoleErrorSpy).toHaveBeenCalled();
        expect(consoleErrorSpy).toHaveBeenCalledWith(
          expect.stringContaining("Unexpected error transforming quiz data for module 'test-module'"),
          expect.any(Error)
        );

        // Cleanup
        consoleErrorSpy.mockRestore();
      });
    });
  });

  describe('Integration: Real-World Scenarios', () => {
    it('should handle complete quiz from linear regression module', () => {
      // Arrange - Simulating actual quiz.json structure
      const rawData = {
        id: 'linear-regression-quiz',
        title: 'Linear Regression Quiz',
        questions: [
          {
            id: 'lr-q1',
            type: 'multiple-choice' as const,
            text: 'What is the purpose of linear regression?',
            options: [
              'Classification',
              'Clustering',
              'Predicting continuous values',
              'Dimensionality reduction',
            ],
            correctAnswer: 2,
            explanation: 'Linear regression is used to predict continuous numerical values.',
          },
          {
            id: 'lr-q2',
            type: 'true-false' as const,
            text: 'The learning rate should always be set to 1.0',
            options: ['True', 'False'],
            correctAnswer: 1,
            explanation: 'A learning rate of 1.0 is usually too high and can cause divergence.',
          },
        ],
      };
      const moduleId = 'linear-regression';

      // Act
      const result = transformQuizData(rawData, moduleId);

      // Assert
      expect(result).toBeDefined();
      expect(result.id).toBe('linear-regression-quiz');
      expect(result.moduleId).toBe('linear-regression');
      expect(result.questions).toHaveLength(2);
      expect(result.questions[0].type).toBe('multiple-choice');
      expect(result.questions[1].type).toBe('true-false');
    });

    it('should handle quiz with mixed old and new format questions', () => {
      // Arrange
      const rawData = {
        title: 'Mixed Format Quiz',
        questions: [
          {
            id: 'q1',
            type: 'multiple-choice' as const,
            text: 'New format question',
            options: ['A', 'B', 'C'],
            correctAnswer: 0,
            explanation: 'Explanation 1',
          },
          {
            id: 'q2',
            type: 'true-false' as const,
            question: 'Old format question',
            options: ['True', 'False'],
            correctAnswer: 1,
            explanation: 'Explanation 2',
          },
        ],
      };
      const moduleId = 'test-module';

      // Act
      const result = transformQuizData(rawData, moduleId);

      // Assert
      expect(result.questions[0].text).toBe('New format question');
      expect(result.questions[1].text).toBe('Old format question');
    });

    it('should handle quiz with all edge cases combined', () => {
      // Arrange
      const rawData = {
        // No id - should generate from moduleId
        title: 'Edge Case Quiz',
        description: 'Should be ignored',
        passingScore: 70,
        questions: [
          {
            id: 'q1',
            type: 'multiple-choice' as const,
            text: 'Question with new format',
            options: ['A', 'B', 'C', 'D'],
            correctAnswer: 3,
            explanation: 'Last option is correct',
          },
          {
            id: 'q2',
            type: 'true-false' as const,
            question: 'Question with old format',
            options: ['True', 'False'],
            correctAnswer: 0,
            explanation: 'True is correct',
          },
        ],
      };
      const moduleId = 'edge-case-module';

      // Act
      const result = transformQuizData(rawData, moduleId);

      // Assert
      expect(result.id).toBe('edge-case-module-quiz');
      expect(result.moduleId).toBe('edge-case-module');
      expect(result.title).toBe('Edge Case Quiz');
      expect(result.questions).toHaveLength(2);
      expect('description' in result).toBe(false);
      expect('passingScore' in result).toBe(false);
    });
  });
});
