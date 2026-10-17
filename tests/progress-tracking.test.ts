import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { nextTick } from 'vue';
import { useProgressStore } from '../stores/progress';
import { createTestPinia } from './helpers/pinia';

/**
 * Progress Tracking Integration Tests
 * 
 * Tests quiz score persistence, module completion tracking,
 * and quiz retake functionality.
 * 
 * Requirements: 5.1, 5.2, 5.3, 3.4, 5.4
 */

describe('Progress Tracking Integration', () => {
  let progressStore: ReturnType<typeof useProgressStore>;
  const STORAGE_KEY = 'ml-platform-progress';

  beforeEach(() => {
    // Create a fresh pinia instance for each test
    createTestPinia();
    progressStore = useProgressStore();
    
    // Clear localStorage before each test
    localStorage.clear();
  });

  afterEach(() => {
    // Clean up after each test
    localStorage.clear();
  });

  describe('8.1 Quiz Score Persistence', () => {
    it('should save quiz score to progress store', () => {
      // Arrange
      const moduleId = 'linear-regression';
      const score = 85;

      // Act
      progressStore.saveQuizScore(moduleId, score);

      // Assert
      expect(progressStore.quizScores[moduleId]).toBe(score);
      expect(progressStore.getQuizScore(moduleId)).toBe(score);
    });

    it('should persist quiz score to localStorage', async () => {
      // Arrange
      const moduleId = 'linear-regression';
      const score = 85;

      // Act
      progressStore.saveQuizScore(moduleId, score);
      // Persistence is async (pinia-plugin-persistedstate writes via $subscribe)
      await nextTick();

      // Assert
      const stored = localStorage.getItem(STORAGE_KEY);
      expect(stored).toBeTruthy();
      
      const data = JSON.parse(stored!);
      expect(data.quizScores[moduleId]).toBe(score);
    });

    it('should load persisted quiz score after page reload', async () => {
      // Arrange
      const moduleId = 'linear-regression';
      const score = 85;

      // Act - Save score
      progressStore.saveQuizScore(moduleId, score);
      await nextTick();

      // Simulate page reload with a fresh Pinia instance (rehydrates from localStorage)
      createTestPinia();
      const newStore = useProgressStore();

      // Assert
      expect(newStore.quizScores[moduleId]).toBe(score);
      expect(newStore.getQuizScore(moduleId)).toBe(score);
    });

    it('should persist multiple quiz scores', async () => {
      // Arrange
      const scores = {
        'linear-regression': 85,
        'logistic-regression': 90,
        'pca': 75,
      };

      // Act
      Object.entries(scores).forEach(([moduleId, score]) => {
        progressStore.saveQuizScore(moduleId, score);
      });
      await nextTick();

      // Simulate page reload with a fresh Pinia instance (rehydrates from localStorage)
      createTestPinia();
      const newStore = useProgressStore();

      // Assert
      Object.entries(scores).forEach(([moduleId, expectedScore]) => {
        expect(newStore.getQuizScore(moduleId)).toBe(expectedScore);
      });
    });
  });

  describe('8.2 Module Completion Tracking', () => {
    it('should mark module as complete when score is 70% or higher', () => {
      // Arrange
      const moduleId = 'linear-regression';
      const passingScore = 70;

      // Act
      progressStore.saveQuizScore(moduleId, passingScore);
      progressStore.markModuleComplete(moduleId);

      // Assert
      expect(progressStore.isModuleCompleted(moduleId)).toBe(true);
      expect(progressStore.completedModules).toContain(moduleId);
    });

    it('should mark module as complete with score above 70%', () => {
      // Arrange
      const moduleId = 'logistic-regression';
      const highScore = 95;

      // Act
      progressStore.saveQuizScore(moduleId, highScore);
      progressStore.markModuleComplete(moduleId);

      // Assert
      expect(progressStore.isModuleCompleted(moduleId)).toBe(true);
    });

    it('should persist module completion to localStorage', async () => {
      // Arrange
      const moduleId = 'pca';
      const score = 80;

      // Act
      progressStore.saveQuizScore(moduleId, score);
      progressStore.markModuleComplete(moduleId);
      await nextTick();

      // Assert
      const stored = localStorage.getItem(STORAGE_KEY);
      expect(stored).toBeTruthy();
      
      const data = JSON.parse(stored!);
      expect(data.completedModules).toContain(moduleId);
    });

    it('should load persisted module completion after page reload', async () => {
      // Arrange
      const moduleId = 'gradient-descent';
      const score = 85;

      // Act - Save score and mark complete
      progressStore.saveQuizScore(moduleId, score);
      progressStore.markModuleComplete(moduleId);
      await nextTick();

      // Simulate page reload with a fresh Pinia instance (rehydrates from localStorage)
      createTestPinia();
      const newStore = useProgressStore();

      // Assert
      expect(newStore.isModuleCompleted(moduleId)).toBe(true);
      expect(newStore.completedModules).toContain(moduleId);
    });

    it('should track multiple completed modules', async () => {
      // Arrange
      const modules = [
        { id: 'linear-regression', score: 85 },
        { id: 'logistic-regression', score: 90 },
        { id: 'pca', score: 75 },
      ];

      // Act
      modules.forEach(({ id, score }) => {
        progressStore.saveQuizScore(id, score);
        progressStore.markModuleComplete(id);
      });
      await nextTick();

      // Simulate page reload with a fresh Pinia instance (rehydrates from localStorage)
      createTestPinia();
      const newStore = useProgressStore();

      // Assert
      modules.forEach(({ id }) => {
        expect(newStore.isModuleCompleted(id)).toBe(true);
      });
      expect(newStore.completedModules.length).toBe(modules.length);
    });

    it('should not duplicate module completion', () => {
      // Arrange
      const moduleId = 'linear-regression';

      // Act - Mark complete multiple times
      progressStore.markModuleComplete(moduleId);
      progressStore.markModuleComplete(moduleId);
      progressStore.markModuleComplete(moduleId);

      // Assert
      const completionCount = progressStore.completedModules.filter(
        id => id === moduleId
      ).length;
      expect(completionCount).toBe(1);
    });
  });

  describe('8.3 Quiz Retake Functionality', () => {
    it('should replace old score with new score on retake', () => {
      // Arrange
      const moduleId = 'linear-regression';
      const firstScore = 60;
      const retakeScore = 85;

      // Act - First attempt
      progressStore.saveQuizScore(moduleId, firstScore);
      expect(progressStore.getQuizScore(moduleId)).toBe(firstScore);

      // Act - Retake
      progressStore.saveQuizScore(moduleId, retakeScore);

      // Assert
      expect(progressStore.getQuizScore(moduleId)).toBe(retakeScore);
      expect(progressStore.quizScores[moduleId]).toBe(retakeScore);
    });

    it('should persist updated score after retake', async () => {
      // Arrange
      const moduleId = 'logistic-regression';
      const firstScore = 65;
      const retakeScore = 90;

      // Act - First attempt
      progressStore.saveQuizScore(moduleId, firstScore);

      // Act - Retake
      progressStore.saveQuizScore(moduleId, retakeScore);
      await nextTick();

      // Simulate page reload with a fresh Pinia instance (rehydrates from localStorage)
      createTestPinia();
      const newStore = useProgressStore();

      // Assert
      expect(newStore.getQuizScore(moduleId)).toBe(retakeScore);
    });

    it('should mark module complete after passing retake', () => {
      // Arrange
      const moduleId = 'pca';
      const failingScore = 50;
      const passingScore = 80;

      // Act - First attempt (failing)
      progressStore.saveQuizScore(moduleId, failingScore);
      expect(progressStore.isModuleCompleted(moduleId)).toBe(false);

      // Act - Retake (passing)
      progressStore.saveQuizScore(moduleId, passingScore);
      progressStore.markModuleComplete(moduleId);

      // Assert
      expect(progressStore.isModuleCompleted(moduleId)).toBe(true);
      expect(progressStore.getQuizScore(moduleId)).toBe(passingScore);
    });

    it('should maintain module completion even if retake score is lower', () => {
      // Arrange
      const moduleId = 'gradient-descent';
      const firstScore = 85;
      const lowerRetakeScore = 70;

      // Act - First attempt (passing)
      progressStore.saveQuizScore(moduleId, firstScore);
      progressStore.markModuleComplete(moduleId);
      expect(progressStore.isModuleCompleted(moduleId)).toBe(true);

      // Act - Retake with lower score (still passing)
      progressStore.saveQuizScore(moduleId, lowerRetakeScore);

      // Assert - Module should still be marked complete
      expect(progressStore.isModuleCompleted(moduleId)).toBe(true);
      expect(progressStore.getQuizScore(moduleId)).toBe(lowerRetakeScore);
    });

    it('should handle multiple retakes correctly', () => {
      // Arrange
      const moduleId = 'missing-value-imputation';
      const scores = [55, 65, 75, 90, 85];

      // Act - Multiple attempts
      scores.forEach(score => {
        progressStore.saveQuizScore(moduleId, score);
      });

      // Assert - Should have the last score
      expect(progressStore.getQuizScore(moduleId)).toBe(scores[scores.length - 1]);
    });

    it('should persist retake updates correctly', async () => {
      // Arrange
      const moduleId = 'feature-scaling';
      const scores = [60, 75, 90];

      // Act - Multiple attempts with persistence checks
      for (const score of scores) {
        progressStore.saveQuizScore(moduleId, score);
        await nextTick();

        // Simulate page reload after each attempt with a fresh Pinia instance
        createTestPinia();
        progressStore = useProgressStore();

        // Assert - Should have the current score
        expect(progressStore.getQuizScore(moduleId)).toBe(score);
      }
    });
  });

  describe('Integration: Complete Quiz Flow', () => {
    it('should handle complete quiz flow with persistence', async () => {
      // Arrange
      const moduleId = 'linear-regression';
      const score = 85;

      // Act - Complete quiz
      progressStore.saveQuizScore(moduleId, score);
      progressStore.markModuleComplete(moduleId);
      await nextTick();

      // Simulate page reload with a fresh Pinia instance (rehydrates from localStorage)
      createTestPinia();
      const newStore = useProgressStore();

      // Assert - All data persisted
      expect(newStore.getQuizScore(moduleId)).toBe(score);
      expect(newStore.isModuleCompleted(moduleId)).toBe(true);
    });

    it('should calculate average quiz score correctly', () => {
      // Arrange
      const quizzes = [
        { moduleId: 'linear-regression', score: 80 },
        { moduleId: 'logistic-regression', score: 90 },
        { moduleId: 'pca', score: 70 },
      ];

      // Act
      quizzes.forEach(({ moduleId, score }) => {
        progressStore.saveQuizScore(moduleId, score);
      });

      // Assert
      const expectedAverage = Math.round((80 + 90 + 70) / 3);
      expect(progressStore.averageQuizScore).toBe(expectedAverage);
    });

    it('should handle edge case: no quizzes taken', () => {
      // Assert
      expect(progressStore.averageQuizScore).toBe(0);
      expect(Object.keys(progressStore.quizScores).length).toBe(0);
    });

    it('should handle edge case: localStorage quota exceeded', () => {
      // Arrange
      const moduleId = 'test-module';
      const score = 85;

      // Mock localStorage.setItem to throw QuotaExceededError
      const originalSetItem = Storage.prototype.setItem;
      const quotaError = new Error('QuotaExceededError');
      quotaError.name = 'QuotaExceededError';
      
      Storage.prototype.setItem = vi.fn(() => {
        throw quotaError;
      });

      // Act & Assert - Should not throw
      expect(() => {
        progressStore.saveQuizScore(moduleId, score);
      }).not.toThrow();

      // Cleanup
      Storage.prototype.setItem = originalSetItem;
    });
  });
});
