/**
 * Analytics Plugin
 * 
 * Initializes analytics tracking and integrates Plausible Analytics.
 * Only runs on client-side.
 * 
 * Requirements: 11.1, 11.3
 */

export default defineNuxtPlugin((nuxtApp) => {
  const analyticsStore = useAnalyticsStore();
  const config = useRuntimeConfig();

  // Initialize analytics store
  analyticsStore.initialize();

  // Only load Plausible if user has opted in
  if (analyticsStore.optedIn) {
    loadPlausibleScript();
  }

  // Watch for opt-in changes
  watch(() => analyticsStore.optedIn, (optedIn) => {
    if (optedIn) {
      loadPlausibleScript();
    }
  });

  /**
   * Load Plausible Analytics script
   */
  function loadPlausibleScript() {
    // Check if already loaded
    if (typeof window !== 'undefined' && (window as any).plausible) {
      return;
    }

    // Get Plausible configuration from runtime config or use defaults
    const plausibleDomain = (config.public.plausibleDomain as string | undefined) || '';
    const plausibleHost = config.public.plausibleHost || 'https://plausible.io';

    // Only load if domain is configured
    if (!plausibleDomain) {
      console.warn('[Analytics] Plausible domain not configured. Set NUXT_PUBLIC_PLAUSIBLE_DOMAIN environment variable.');
      return;
    }

    // Use Nuxt Scripts to load Plausible
    useScript({
      src: `${plausibleHost}/js/script.js`,
      defer: true,
      'data-domain': plausibleDomain,
    });
  }

  // Track page views on route changes
  nuxtApp.hook('page:finish', () => {
    if (analyticsStore.optedIn) {
      const route = useRoute();
      analyticsStore.trackPageView(route.path, route.meta.title as string);
    }
  });
});
