import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { useAuth } from '~/composables/useAuth';
import { useAuthStore } from '~/stores/auth';
import { useProgressStore } from '~/stores/progress';
import { useToastStore } from '~/stores/toast';
import { createTestPinia } from './helpers/pinia';

// Mock fetch
global.fetch = vi.fn();

// Mock window.location
delete (window as any).location;
window.location = { href: '', search: '' } as any;

// Mock Nuxt composables
const mockNavigateTo = vi.fn();
const mockUseRuntimeConfig = vi.fn(() => ({
  public: {
    apiUrl: 'http://localhost:8787',
  },
}));

vi.mock('#app', () => ({
  navigateTo: mockNavigateTo,
  useRuntimeConfig: mockUseRuntimeConfig,
}));

// Make useRuntimeConfig available globally
(globalThis as any).useRuntimeConfig = mockUseRuntimeConfig;
(globalThis as any).navigateTo = mockNavigateTo;

describe('useAuth Composable', () => {
  beforeEach(() => {
    createTestPinia();
    vi.clearAllMocks();
    localStorage.clear();
    (fetch as any).mockClear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('signInWithGoogle', () => {
    it('should redirect to Google OAuth URL', () => {
      const { signInWithGoogle } = useAuth();
      
      signInWithGoogle();
      
      expect(window.location.href).toBe('http://localhost:8787/api/auth/google');
    });

    it('should handle errors gracefully', () => {
      const { signInWithGoogle } = useAuth();
      const toastStore = useToastStore();
      const errorSpy = vi.spyOn(toastStore, 'error');
      
      // Mock window.location to throw error
      Object.defineProperty(window, 'location', {
        value: {
          get href() {
            throw new Error('Navigation error');
          },
          set href(value) {
            throw new Error('Navigation error');
          },
        },
        writable: true,
      });
      
      signInWithGoogle();
      
      expect(errorSpy).toHaveBeenCalledWith('Failed to start sign-in process');
    });
  });

  describe('handleOAuthCallback', () => {
    it('should handle successful OAuth callback with token', async () => {
      const mockUser = {
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
        picture_url: 'https://example.com/pic.jpg',
      };

      // Mock URL with token
      Object.defineProperty(window, 'location', {
        value: {
          search: '?token=valid-jwt-token',
        },
        writable: true,
      });

      // Mock fetch for user profile
      (fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockUser,
      });

      // Mock fetch for progress sync
      (fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, synced_at: Date.now() }),
      });

      const { handleOAuthCallback } = useAuth();
      const authStore = useAuthStore();
      const toastStore = useToastStore();
      const successSpy = vi.spyOn(toastStore, 'success');

      const result = await handleOAuthCallback();

      expect(result).toBe(true);
      expect(authStore.token).toBe('valid-jwt-token');
      expect(authStore.user).toEqual(mockUser);
      expect(successSpy).toHaveBeenCalledWith('Welcome back, Test User!');
    });

    it('should handle OAuth error in callback', async () => {
      // Mock URL with error
      Object.defineProperty(window, 'location', {
        value: {
          search: '?error=access_denied',
        },
        writable: true,
      });

      const { handleOAuthCallback } = useAuth();
      const toastStore = useToastStore();
      const errorSpy = vi.spyOn(toastStore, 'error');

      const result = await handleOAuthCallback();

      expect(result).toBe(false);
      expect(errorSpy).toHaveBeenCalledWith('Authentication failed. Please try again.');
    });

    it('should return false when no token or error in URL', async () => {
      // Mock URL without token or error
      Object.defineProperty(window, 'location', {
        value: {
          search: '',
        },
        writable: true,
      });

      const { handleOAuthCallback } = useAuth();
      const result = await handleOAuthCallback();

      expect(result).toBe(false);
    });
  });

  describe('logout', () => {
    it('should clear auth state and call logout endpoint', async () => {
      const authStore = useAuthStore();
      authStore.setToken('valid-token');
      authStore.setUser({
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
        picture_url: null,
      });

      // Mock logout endpoint
      (fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      });

      const { logout } = useAuth();
      const toastStore = useToastStore();
      const infoSpy = vi.spyOn(toastStore, 'info');

      await logout();

      expect(authStore.user).toBeNull();
      expect(authStore.token).toBeNull();
      expect(authStore.isAuthenticated).toBe(false);
      expect(infoSpy).toHaveBeenCalledWith('You have been logged out');
    });

    it('should handle logout even if endpoint fails', async () => {
      const authStore = useAuthStore();
      authStore.setToken('valid-token');

      // Mock logout endpoint to fail
      (fetch as any).mockRejectedValueOnce(new Error('Network error'));

      const { logout } = useAuth();

      await logout();

      expect(authStore.user).toBeNull();
      expect(authStore.token).toBeNull();
    });
  });

  describe('syncProgress', () => {
    it('should sync progress data when authenticated', async () => {
      const authStore = useAuthStore();
      const progressStore = useProgressStore();

      authStore.setToken('valid-token');
      authStore.setUser({
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
        picture_url: null,
      });

      progressStore.markModuleComplete('module-1');
      progressStore.saveQuizScore('module-1', 85);

      // Mock sync endpoint
      (fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, synced_at: Date.now() }),
      });

      const { syncProgress } = useAuth();
      const result = await syncProgress();

      expect(result).toBe(true);
      expect(fetch).toHaveBeenCalledWith(
        'http://localhost:8787/api/progress/sync',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Authorization': 'Bearer valid-token',
          }),
        })
      );
    });

    it('should not sync when not authenticated', async () => {
      const { syncProgress } = useAuth();
      const result = await syncProgress();

      expect(result).toBe(false);
      expect(fetch).not.toHaveBeenCalled();
    });

    it('should handle 401 and logout user', async () => {
      const authStore = useAuthStore();
      authStore.setToken('expired-token');
      authStore.setUser({
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
        picture_url: null,
      });

      // Mock sync endpoint to return 401
      (fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 401,
      });

      const { syncProgress } = useAuth();
      const toastStore = useToastStore();
      const warningSpy = vi.spyOn(toastStore, 'warning');

      const result = await syncProgress();

      expect(result).toBe(false);
      expect(authStore.isAuthenticated).toBe(false);
      expect(warningSpy).toHaveBeenCalledWith('Session expired. Please sign in again.');
    });
  });

  describe('fetchProgress', () => {
    it('should fetch and merge progress data', async () => {
      const authStore = useAuthStore();
      const progressStore = useProgressStore();

      authStore.setToken('valid-token');
      authStore.setUser({
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
        picture_url: null,
      });

      const mockProgressData = {
        modules: [
          {
            id: '1',
            user_id: '123',
            module_id: 'module-1',
            completed: true,
            completed_at: Date.now(),
          },
        ],
        quizzes: [
          {
            id: '1',
            user_id: '123',
            module_id: 'module-1',
            score: 90,
            total: 100,
            completed_at: Date.now(),
          },
        ],
        bookmarks: [
          {
            id: '1',
            user_id: '123',
            module_id: 'module-2',
            created_at: Date.now(),
          },
        ],
      };

      // Mock fetch endpoint
      (fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockProgressData,
      });

      const { fetchProgress } = useAuth();
      const result = await fetchProgress();

      expect(result).toBe(true);
      expect(progressStore.completedModules).toContain('module-1');
      expect(progressStore.quizScores['module-1']).toBe(90);
      expect(progressStore.bookmarks.length).toBeGreaterThan(0);
    });

    it('should not fetch when not authenticated', async () => {
      const { fetchProgress } = useAuth();
      const result = await fetchProgress();

      expect(result).toBe(false);
      expect(fetch).not.toHaveBeenCalled();
    });
  });

  describe('initializeAuth', () => {
    it('should load auth state from localStorage and verify token', async () => {
      const mockUser = {
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
        picture_url: null,
      };

      localStorage.setItem('ml-platform-auth', JSON.stringify({
        user: mockUser,
        token: 'valid-token',
        isAuthenticated: true,
      }));

      // Mock user profile fetch
      (fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockUser,
      });

      // Mock progress fetch
      (fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          modules: [],
          quizzes: [],
          bookmarks: [],
        }),
      });

      const { initializeAuth } = useAuth();
      const authStore = useAuthStore();

      await initializeAuth();

      expect(authStore.token).toBe('valid-token');
      expect(authStore.user).toEqual(mockUser);
    });

    it('should clear auth if token is invalid', async () => {
      localStorage.setItem('ml-platform-auth', JSON.stringify({
        user: { id: '123', email: 'test@example.com', name: 'Test User', picture_url: null },
        token: 'invalid-token',
        isAuthenticated: true,
      }));

      // Mock user profile fetch to fail
      (fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 401,
      });

      const { initializeAuth } = useAuth();
      const authStore = useAuthStore();

      await initializeAuth();

      expect(authStore.isAuthenticated).toBe(false);
      expect(authStore.token).toBeNull();
    });
  });

  describe('Computed Properties', () => {
    it('should expose reactive state', () => {
      const { user, isAuthenticated, isLoading } = useAuth();
      const authStore = useAuthStore();

      expect(user.value).toBeNull();
      expect(isAuthenticated.value).toBe(false);
      expect(isLoading.value).toBe(false);

      authStore.setUser({
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
        picture_url: null,
      });

      expect(user.value).toBeTruthy();
      expect(isAuthenticated.value).toBe(true);
    });
  });
});
