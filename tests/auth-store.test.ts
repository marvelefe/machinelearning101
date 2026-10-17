import { describe, it, expect, beforeEach, vi } from 'vitest';
import { nextTick } from 'vue';
import { useAuthStore } from '~/stores/auth';
import type { User } from '~/stores/auth';
import { createTestPinia } from './helpers/pinia';

describe('Auth Store', () => {
  beforeEach(() => {
    createTestPinia();
    // Clear localStorage before each test
    localStorage.clear();
  });

  describe('Initial State', () => {
    it('should have correct initial state', () => {
      const store = useAuthStore();
      
      expect(store.user).toBeNull();
      expect(store.token).toBeNull();
      expect(store.isAuthenticated).toBe(false);
      expect(store.isLoading).toBe(false);
    });
  });

  describe('Getters', () => {
    it('should return "Guest" as displayName when no user', () => {
      const store = useAuthStore();
      expect(store.displayName).toBe('Guest');
    });

    it('should return user name as displayName when user exists', () => {
      const store = useAuthStore();
      const user: User = {
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
        picture_url: 'https://example.com/pic.jpg',
      };
      store.setUser(user);
      
      expect(store.displayName).toBe('Test User');
    });

    it('should return user email', () => {
      const store = useAuthStore();
      const user: User = {
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
        picture_url: null,
      };
      store.setUser(user);
      
      expect(store.userEmail).toBe('test@example.com');
    });

    it('should return profile picture URL', () => {
      const store = useAuthStore();
      const user: User = {
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
        picture_url: 'https://example.com/pic.jpg',
      };
      store.setUser(user);
      
      expect(store.profilePicture).toBe('https://example.com/pic.jpg');
    });

    it('should check if token is valid', () => {
      const store = useAuthStore();
      expect(store.hasValidToken).toBe(false);
      
      store.setToken('valid-token');
      expect(store.hasValidToken).toBe(true);
    });
  });

  describe('Actions', () => {
    it('should set user and mark as authenticated', () => {
      const store = useAuthStore();
      const user: User = {
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
        picture_url: null,
      };
      
      store.setUser(user);
      
      expect(store.user).toEqual(user);
      expect(store.isAuthenticated).toBe(true);
    });

    it('should set token and mark as authenticated', () => {
      const store = useAuthStore();
      const token = 'jwt-token-123';
      
      store.setToken(token);
      
      expect(store.token).toBe(token);
      expect(store.isAuthenticated).toBe(true);
    });

    it('should set loading state', () => {
      const store = useAuthStore();
      
      store.setLoading(true);
      expect(store.isLoading).toBe(true);
      
      store.setLoading(false);
      expect(store.isLoading).toBe(false);
    });

    it('should clear auth state', () => {
      const store = useAuthStore();
      const user: User = {
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
        picture_url: null,
      };
      
      store.setUser(user);
      store.setToken('token');
      store.setLoading(true);
      
      store.clearAuth();
      
      expect(store.user).toBeNull();
      expect(store.token).toBeNull();
      expect(store.isAuthenticated).toBe(false);
      expect(store.isLoading).toBe(false);
    });

    it('should update user profile', () => {
      const store = useAuthStore();
      const user: User = {
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
        picture_url: null,
      };
      
      store.setUser(user);
      store.updateUser({ name: 'Updated Name' });
      
      expect(store.user?.name).toBe('Updated Name');
      expect(store.user?.email).toBe('test@example.com');
    });
  });

  describe('Persistence', () => {
    it('should save to localStorage when setting user', async () => {
      const store = useAuthStore();
      const user: User = {
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
        picture_url: null,
      };
      
      store.setUser(user);
      // Persistence is async (pinia-plugin-persistedstate writes via $subscribe)
      await nextTick();
      
      const stored = localStorage.getItem('ml-platform-auth');
      expect(stored).toBeTruthy();
      
      const data = JSON.parse(stored!);
      expect(data.user).toEqual(user);
      expect(data.isAuthenticated).toBe(true);
    });

    it('should save to localStorage when setting token', async () => {
      const store = useAuthStore();
      const token = 'jwt-token-123';
      
      store.setToken(token);
      // Persistence is async (pinia-plugin-persistedstate writes via $subscribe)
      await nextTick();
      
      const stored = localStorage.getItem('ml-platform-auth');
      expect(stored).toBeTruthy();
      
      const data = JSON.parse(stored!);
      expect(data.token).toBe(token);
    });

    it('should load from localStorage on initialize', () => {
      const user: User = {
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
        picture_url: null,
      };
      const token = 'jwt-token-123';
      
      localStorage.setItem('ml-platform-auth', JSON.stringify({
        user,
        token,
        isAuthenticated: true,
      }));
      
      // Create a new store instance which will automatically load from localStorage
      createTestPinia();
      const store = useAuthStore();
      
      expect(store.user).toEqual(user);
      expect(store.token).toBe(token);
      expect(store.isAuthenticated).toBe(true);
    });

    it('should clear localStorage on clearAuth', async () => {
      const store = useAuthStore();
      const user: User = {
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
        picture_url: null,
      };
      
      store.setUser(user);
      // Wait for persistence to complete
      await nextTick();
      expect(localStorage.getItem('ml-platform-auth')).toBeTruthy();
      
      store.clearAuth();
      // After clearing, the persisted state should reflect the cleared state
      await nextTick();
      const stored = localStorage.getItem('ml-platform-auth');
      expect(stored).toBeTruthy();
      const data = JSON.parse(stored!);
      expect(data.user).toBeNull();
      expect(data.token).toBeNull();
    });

    it('should handle corrupted localStorage data', () => {
      localStorage.setItem('ml-platform-auth', 'invalid-json');
      
      // Create a new store instance which will try to load from localStorage
      createTestPinia();
      const store = useAuthStore();
      
      // The store should have default values when localStorage is corrupted
      expect(store.user).toBeNull();
      expect(store.token).toBeNull();
    });
  });
});
