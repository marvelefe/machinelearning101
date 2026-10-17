import { createApp } from 'vue';
import { createPinia, setActivePinia, type Pinia } from 'pinia';
import piniaPluginPersistedstate from 'pinia-plugin-persistedstate';

/**
 * Create a Pinia instance wired up the same way as the running app.
 *
 * Pinia defers plugins registered with `pinia.use()` until the instance is
 * installed into a Vue app (`app.use(pinia)`). If that install step is skipped
 * (as it was in these tests), the persistence plugin never runs and stores with
 * `persist: true` silently skip localStorage.
 *
 * This helper performs the install and registers the plugin so persistence
 * behaves exactly as it does in the Nuxt app.
 */
export function createTestPinia(): Pinia {
  const app = createApp({});
  const pinia = createPinia();

  app.use(pinia);
  pinia.use(piniaPluginPersistedstate);
  setActivePinia(pinia);

  return pinia;
}
