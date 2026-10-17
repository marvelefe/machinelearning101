import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { nextTick } from 'vue';
import { useProgressStore } from '../stores/progress';
import { useAnalyticsStore } from '../stores/analytics';
import { useQuizLoader } from '../composables/useQuizLoader';
import { transformQuizData } from '../utils/quiz-transform';
import { createTestPinia } from './helpers/pinia';
import type { Quiz as QuizType, QuizResult, QuizAnswer } from '../types/quiz';

/**
 * Quiz Integration Tests
 * 
 * Tests the complete quiz flow including loading quiz data,
 * answering questions, score calculation, and progress store updates.
 * 
 * Requirements: 3.1, 3.2, 3.3, 3.4, 5.1, 5.3
 */

describe('Quiz Integration Tests', () => {
  let progressStore: ReturnType<typeof useProgressStore>;
  let analyticsStore: ReturnType<typeof useAnalyticsStore>;
  
  // Sample quiz data for testing
  const mockQuiz: QuizType = {
    id: 'test-quiz',
    moduleId: 'test-module',
    title: 'Test Quiz',
    questions: [
      {
        id: 'q1',
        type: 'multiple-choice',
        text: 'What is 2 + 2?',
        options: ['2', '3', '4', '5'],
        correctAnswer: 2,
        explanation: 'The answer is 4',
      },
      {
        id: 'q2',
        type: 'true-false',
        text: 'The sky is blue',
        options: ['True', 'False'],
        correctAnswer: 0,
        explanation: 'The sky appears blue due to Rayleigh scattering',
      },
      {
        id: 'q3',
        type: 'multiple-choice',
        text: 'What is the capital of France?',
        options: ['London', 'Paris', 'Berlin', 'Madrid'],
        correctAnswer: 1,
        explanation: 'Paris is the capital of France',
      },
      {
        id: 'q4',
        type: 'multiple-choice',
        text: 'Which is a programming language?',
        options: ['HTML', 'CSS', 'JavaScript', 'JSON'],
        correctAnswer: 2,
        explanation: 'JavaScript is a programming language',
      },
      {
        id: 'q5',
        type: 'true-false',
        text: 'Water boils at 100°C at sea level',
        options: ['True', 'False'],
        correctAnswer: 0,
        explanation: 'Water boils at 100°C (212°F) at sea level',
      },
    ],
  };

  /**
   * Helper function to simulate quiz completion
   * Mimics the logic in Quiz.vue's completeQuiz method
   */
  const simulateQuizCompletion = (
    quiz: QuizType,
    userAnswers: Record<string, number>
  ): QuizResult => {
    const answers: QuizAnswer[] = quiz.questions.map((question) => {
      const selectedAnswer = userAnswers[question.id];
      const isCorrect = selectedAnswer === question.correctAnswer;
      
      return {
        questionId: question.id,
        selectedAnswer,
        isCorrect,
      };
    });

    const correctAnswers = answers.filter(a => a.isCorrect).length;
    const score = Math.round((correctAnswers / quiz.questions.length) * 100);

    const result: QuizResult = {
      quizId: quiz.id,
      score,
      totalQuestions: quiz.questions.length,
      correctAnswers,
      answers,
      completedAt: Date.now(),
    };

    // Save score to progress store (mimics Quiz.vue behavior)
    progressStore.saveQuizScore(quiz.moduleId, score);

    // Mark module as completed if score is passing (>= 70%)
    if (score >= 70) {
      progressStore.markModuleComplete(quiz.moduleId);
    }

    // Track quiz completion
    analyticsStore.trackQuizComplete(quiz.moduleId, score, quiz.questions.length);

    return result;
  };

  beforeEach(() => {
    // Create a fresh pinia instance for each test
    createTestPinia();
    progressStore = useProgressStore();
    analyticsStore = useAnalyticsStore();
    
    // Clear localStorage before each test
    localStorage.clear();
    
    // Mock console methods to avoid cluttering test output
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    // Clean up after each test
    localStorage.clear();
    vi.restoreAllMocks();
  });

  describe('10.1 Test Complete Quiz Flow', () => {
    /**
     * Test loading quiz data
     * Requirements: 3.1
     */
    it('should load and transform quiz data correctly', () => {
      // Arrange
      const rawQuizData = {
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

      // Act
      const transformedQuiz = transformQuizData(rawQuizData, 'test-module');

      // Assert - Quiz should be transformed correctly
      expect(transformedQuiz).toBeDefined();
      expect(transformedQuiz.id).toBe('test-quiz');
      expect(transformedQuiz.moduleId).toBe('test-module');
      expect(transformedQuiz.title).toBe('Test Quiz');
      expect(transformedQuiz.questions).toHaveLength(1);
      expect(transformedQuiz.questions[0].text).toBe('What is 2 + 2?');
    });

    /**
     * Test answering questions
     * Requirements: 3.2
     */
    it('should track user answers correctly', () => {
      // Arrange
      const userAnswers: Record<string, number> = {};

      // Act - Simulate answering questions
      mockQuiz.questions.forEach((question, index) => {
        // Select an answer for each question
        // Even indices get correct answer, odd indices get first option (0)
        userAnswers[question.id] = index % 2 === 0 ? question.correctAnswer : 0;
      });

      // Assert - All questions should have answers
      expect(Object.keys(userAnswers).length).toBe(mockQuiz.questions.length);
      expect(userAnswers['q1']).toBe(2); // Index 0 (even) - Correct answer
      expect(userAnswers['q2']).toBe(0); // Index 1 (odd) - First option
      expect(userAnswers['q3']).toBe(1); // Index 2 (even) - Correct answer
      expect(userAnswers['q4']).toBe(0); // Index 3 (odd) - First option
      expect(userAnswers['q5']).toBe(0); // Index 4 (even) - Correct answer
    });

    /**
     * Test score calculation
     * Requirements: 3.3
     */
    it('should calculate score correctly based on answers', () => {
      // Arrange - Answer 3 out of 5 correctly (60%)
      const userAnswers: Record<string, number> = {
        'q1': 2, // Correct
        'q2': 1, // Incorrect
        'q3': 1, // Correct
        'q4': 0, // Incorrect
        'q5': 0, // Correct
      };

      // Act - Simulate quiz completion
      const result = simulateQuizCompletion(mockQuiz, userAnswers);

      // Assert - Score should be 60% (3 out of 5 correct)
      expect(result).toBeDefined();
      expect(result.score).toBe(60);
      expect(result.correctAnswers).toBe(3);
      expect(result.totalQuestions).toBe(5);
      expect(result.quizId).toBe('test-quiz');
      expect(result.completedAt).toBeDefined();
      expect(result.answers).toHaveLength(5);
    });

    /**
     * Test progress store updates
     * Requirements: 3.4, 5.1
     */
    it('should save quiz score to progress store on completion', () => {
      // Arrange - Answer all questions correctly
      const userAnswers: Record<string, number> = {};
      mockQuiz.questions.forEach((question) => {
        userAnswers[question.id] = question.correctAnswer;
      });

      // Act - Simulate quiz completion
      const result = simulateQuizCompletion(mockQuiz, userAnswers);

      // Assert - Score should be saved to progress store
      expect(result.score).toBe(100);
      expect(progressStore.quizScores['test-module']).toBe(100);
      expect(progressStore.getQuizScore('test-module')).toBe(100);
    });

    /**
     * Test module completion on passing score
     * Requirements: 3.4, 5.1
     */
    it('should mark module as complete when score is 70% or higher', () => {
      // Arrange - Answer 4 out of 5 correctly (80% score)
      const userAnswers: Record<string, number> = {
        'q1': 2, // Correct
        'q2': 0, // Correct
        'q3': 1, // Correct
        'q4': 0, // Incorrect
        'q5': 0, // Correct
      };

      // Act - Simulate quiz completion
      const result = simulateQuizCompletion(mockQuiz, userAnswers);

      // Assert - Module should be marked as complete
      expect(result.score).toBe(80);
      expect(progressStore.isModuleCompleted('test-module')).toBe(true);
      expect(progressStore.completedModules).toContain('test-module');
    });

    /**
     * Test module not completed on failing score
     * Requirements: 3.4
     */
    it('should not mark module as complete when score is below 70%', () => {
      // Arrange - Answer 2 out of 5 correctly (40% score)
      const userAnswers: Record<string, number> = {
        'q1': 2, // Correct
        'q2': 1, // Incorrect
        'q3': 0, // Incorrect
        'q4': 0, // Incorrect
        'q5': 0, // Correct
      };

      // Act - Simulate quiz completion
      const result = simulateQuizCompletion(mockQuiz, userAnswers);

      // Assert - Module should NOT be marked as complete
      expect(result.score).toBe(40);
      expect(progressStore.isModuleCompleted('test-module')).toBe(false);
      expect(progressStore.completedModules).not.toContain('test-module');
    });

    /**
     * Test quiz result structure
     * Requirements: 3.3
     */
    it('should return complete quiz result with all required fields', () => {
      // Arrange
      const userAnswers: Record<string, number> = {};
      mockQuiz.questions.forEach((question) => {
        userAnswers[question.id] = question.correctAnswer;
      });

      // Act
      const result = simulateQuizCompletion(mockQuiz, userAnswers);

      // Assert - Result should have all required fields
      expect(result.quizId).toBe('test-quiz');
      expect(result.score).toBe(100);
      expect(result.totalQuestions).toBe(5);
      expect(result.correctAnswers).toBe(5);
      expect(result.completedAt).toBeDefined();
      expect(result.answers).toHaveLength(5);
      
      // Check answer structure
      result.answers.forEach((answer) => {
        expect(answer).toHaveProperty('questionId');
        expect(answer).toHaveProperty('selectedAnswer');
        expect(answer).toHaveProperty('isCorrect');
      });
    });

    /**
     * Test analytics tracking
     * Requirements: 3.3
     */
    it('should track quiz completion in analytics store', () => {
      // Arrange
      const trackQuizCompleteSpy = vi.spyOn(analyticsStore, 'trackQuizComplete');
      const userAnswers: Record<string, number> = {};
      mockQuiz.questions.forEach((question) => {
        userAnswers[question.id] = question.correctAnswer;
      });

      // Act
      simulateQuizCompletion(mockQuiz, userAnswers);

      // Assert - Analytics should be tracked
      expect(trackQuizCompleteSpy).toHaveBeenCalledWith('test-module', 100, 5);
    });
  });

  describe('10.2 Test Quiz Retake Flow', () => {
    /**
     * Test retaking a quiz
     * Requirements: 5.3
     */
    it('should allow retaking a quiz with fresh state', () => {
      // Arrange - First attempt
      const firstAttemptAnswers: Record<string, number> = {};
      mockQuiz.questions.forEach((question) => {
        firstAttemptAnswers[question.id] = question.correctAnswer;
      });

      // Act - First quiz completion
      const firstResult = simulateQuizCompletion(mockQuiz, firstAttemptAnswers);

      // Assert - First attempt should be completed
      expect(firstResult.score).toBe(100);
      expect(progressStore.getQuizScore('test-module')).toBe(100);

      // Act - Simulate retake with different answers
      const retakeAnswers: Record<string, number> = {
        'q1': 0, // Incorrect
        'q2': 1, // Incorrect
        'q3': 1, // Correct
        'q4': 2, // Correct
        'q5': 0, // Correct
      };
      const retakeResult = simulateQuizCompletion(mockQuiz, retakeAnswers);

      // Assert - Retake should have new score
      expect(retakeResult.score).toBe(60);
      expect(retakeResult.quizId).toBe('test-quiz');
      // Timestamps should be different (or at least not less than first)
      expect(retakeResult.completedAt).toBeGreaterThanOrEqual(firstResult.completedAt);
    });

    /**
     * Test score replacement on retake
     * Requirements: 5.3
     */
    it('should replace old score with new score on retake', () => {
      // Arrange - First attempt with 60% score
      const firstAttemptAnswers: Record<string, number> = {
        'q1': 2, // Correct
        'q2': 1, // Incorrect
        'q3': 1, // Correct
        'q4': 0, // Incorrect
        'q5': 0, // Correct
      };

      // Act - First quiz completion
      simulateQuizCompletion(mockQuiz, firstAttemptAnswers);

      // Assert - First score should be 60%
      expect(progressStore.getQuizScore('test-module')).toBe(60);

      // Act - Retake with 100% score
      const retakeAnswers: Record<string, number> = {};
      mockQuiz.questions.forEach((question) => {
        retakeAnswers[question.id] = question.correctAnswer;
      });
      simulateQuizCompletion(mockQuiz, retakeAnswers);

      // Assert - Score should be replaced with 100%
      expect(progressStore.getQuizScore('test-module')).toBe(100);
    });

    /**
     * Test progress updates on retake
     * Requirements: 5.3
     */
    it('should update progress correctly on retake', () => {
      // Arrange - First attempt with failing score (40%)
      const firstAttemptAnswers: Record<string, number> = {
        'q1': 0, // Incorrect
        'q2': 1, // Incorrect
        'q3': 1, // Correct
        'q4': 0, // Incorrect
        'q5': 0, // Correct
      };

      // Act - First quiz completion
      simulateQuizCompletion(mockQuiz, firstAttemptAnswers);

      // Assert - Module should not be complete
      expect(progressStore.getQuizScore('test-module')).toBe(40);
      expect(progressStore.isModuleCompleted('test-module')).toBe(false);

      // Act - Retake with passing score (80%)
      const retakeAnswers: Record<string, number> = {
        'q1': 2, // Correct
        'q2': 0, // Correct
        'q3': 1, // Correct
        'q4': 0, // Incorrect
        'q5': 0, // Correct
      };
      simulateQuizCompletion(mockQuiz, retakeAnswers);

      // Assert - Module should now be complete
      expect(progressStore.getQuizScore('test-module')).toBe(80);
      expect(progressStore.isModuleCompleted('test-module')).toBe(true);
    });

    /**
     * Test persistence after retake
     * Requirements: 5.3
     */
    it('should persist retake score to localStorage', async () => {
      // Arrange - First attempt
      const firstAttemptAnswers: Record<string, number> = {
        'q1': 2,
        'q2': 1,
        'q3': 1,
        'q4': 0,
        'q5': 0,
      };

      // Act - First quiz completion
      simulateQuizCompletion(mockQuiz, firstAttemptAnswers);

      // Act - Retake with perfect score
      const retakeAnswers: Record<string, number> = {};
      mockQuiz.questions.forEach((question) => {
        retakeAnswers[question.id] = question.correctAnswer;
      });
      simulateQuizCompletion(mockQuiz, retakeAnswers);
      await nextTick();

      // Simulate page reload with a fresh Pinia instance (rehydrates from localStorage)
      createTestPinia();
      const newStore = useProgressStore();

      // Assert - New score should be persisted
      expect(newStore.getQuizScore('test-module')).toBe(100);
    });

    /**
     * Test multiple retakes
     * Requirements: 5.3
     */
    it('should handle multiple retakes correctly', () => {
      // Arrange
      const targetScores = [40, 60, 80, 100];

      // Act - Take quiz multiple times
      for (const targetScore of targetScores) {
        // Calculate how many correct answers needed for target score
        const totalQuestions = mockQuiz.questions.length;
        const correctAnswersNeeded = Math.round((targetScore / 100) * totalQuestions);
        const userAnswers: Record<string, number> = {};
        
        // Answer first N questions correctly, rest incorrectly
        mockQuiz.questions.forEach((question, index) => {
          if (index < correctAnswersNeeded) {
            userAnswers[question.id] = question.correctAnswer;
          } else {
            // Use an incorrect answer (not the correct one)
            userAnswers[question.id] = (question.correctAnswer + 1) % question.options.length;
          }
        });

        const result = simulateQuizCompletion(mockQuiz, userAnswers);

        // Assert - Score should match target (allowing for rounding)
        expect(result.score).toBe(targetScore);
        expect(progressStore.getQuizScore('test-module')).toBe(targetScore);
      }

      // Assert - Final score should be the last one
      expect(progressStore.getQuizScore('test-module')).toBe(100);
    });

    /**
     * Test retake improves from failing to passing
     * Requirements: 5.3
     */
    it('should allow improving from failing to passing score', () => {
      // Arrange - First attempt fails (20%)
      const failingAnswers: Record<string, number> = {
        'q1': 2, // Correct
        'q2': 1, // Incorrect
        'q3': 0, // Incorrect
        'q4': 0, // Incorrect
        'q5': 1, // Incorrect
      };

      // Act - First attempt
      simulateQuizCompletion(mockQuiz, failingAnswers);

      // Assert - Should fail and not complete module
      expect(progressStore.getQuizScore('test-module')).toBe(20);
      expect(progressStore.isModuleCompleted('test-module')).toBe(false);

      // Act - Retake with passing score (100%)
      const passingAnswers: Record<string, number> = {};
      mockQuiz.questions.forEach((question) => {
        passingAnswers[question.id] = question.correctAnswer;
      });
      simulateQuizCompletion(mockQuiz, passingAnswers);

      // Assert - Should pass and complete module
      expect(progressStore.getQuizScore('test-module')).toBe(100);
      expect(progressStore.isModuleCompleted('test-module')).toBe(true);
    });
  });
});
