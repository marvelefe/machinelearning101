// https://nuxt.com/docs/api/configuration/nuxt-config
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

const rootDir = fileURLToPath(new URL(".", import.meta.url));

export default defineNuxtConfig({
  compatibilityDate: "2025-07-15",
  devtools: { enabled: true },

  // experimental: {
  //   defaults: {
  //     nuxtLink: {
  //       trailingSlash: "append",
  //     },
  //   },
  // },

  modules: [ 
    "@nuxtjs/seo",
    "@nuxt/fonts",
    "@nuxt/icon",
    "@nuxt/image",
    "@nuxt/scripts",
    "@pinia/nuxt",
    "@nuxtjs/tailwindcss", 
    'pinia-plugin-persistedstate/nuxt',
     "@nuxt/content",
     "nuxt-gtag",
  ],

  site: {
    url: "https://machinelearning101.com",
    name: "Machine Learning 101",
    trailingSlash: false,
  },


  sitemap: {
    zeroRuntime: true
  },

  // Component auto-import configuration
  components: [
    {
      path: "~/components",
      pathPrefix: false,
    },
  ],

  // Plugins configuration
  plugins: [
    { src: '~/plugins/plotly.client', mode: 'client' },
  ],

  // Global CSS
  css: [
    "~/assets/css/accessibility.css",
    "~/assets/css/animations.css",
    "~/assets/css/utilities.css",
    "~/assets/css/mobile.css",
  ],

  // TypeScript configuration
  typescript: {
    strict: false,
    typeCheck: false,
    tsConfig: {
      // `worker/` is a separate Cloudflare Worker sub-project with its own
      // tsconfig.json, runtime types (@cloudflare/workers-types) and deps
      // (hono, jose). The Nuxt app's type-check must not pull it in.
      exclude: ["../worker"],
    },
  },

  // Path aliases.
  //
  // Nuxt provides `~`, `@`, `~~` and `@@` by default, but they are declared
  // explicitly here so the intent is clear. Values MUST be absolute: Nuxt writes
  // these verbatim into the generated `.nuxt/tsconfig.*.json` files, which live
  // inside `.nuxt/`, so relative values (e.g. "." or "./algorithms") resolve to
  // the wrong directory and break `npm run type-check`.
  alias: {
    "@": rootDir,
    "~": rootDir,
    algorithms: resolve(rootDir, "algorithms"),
    components: resolve(rootDir, "components"),
    composables: resolve(rootDir, "composables"),
    stores: resolve(rootDir, "stores"),
    types: resolve(rootDir, "types"),
    utils: resolve(rootDir, "utils"),
  },

    content: {
    build: {
      markdown: {
        highlight: {
          // Theme used in all color schemes.
          theme: {
            default: "github-light",
            dark: "github-dark",
            sepia: "github-dark",
          },
          langs: [
            "c",
            "cpp",
            "java",
            "json",
            "js",
            "ts",
            "html",
            "css",
            "vue",
            "shell",
            "mdc",
            "md",
            "yaml",
            "python",
            "go",
            "bash",
            "ruby",
            "php",
            "rust",
          ],
        },
        toc: {
          depth: 3, // include h3 headings
        },
      },
    },
  },

  // Static site generation configuration
  ssr: true,
  nitro: {
    prerender: {
      routes: [
        "/",
        "/about",
        "/why-learn-here",
        "/contact",
        "/privacy",
        "/faq",
        "/getting-started",
        "/modules", 
        // "/sitemap.xml", // Removed as handled by nuxt-seo
        // "/modules/*",
      ],
      ignore: [ 
      ],
       autoSubfolderIndex: false, 
    },
  },

  // Google Analytics configuration
  
  
  gtag: {
    // Set via environment variable: NUXT_PUBLIC_GTAG_ID=G-XXXXXXXXXX
    id: process.env.NUXT_PUBLIC_GTAG_ID || "",
    initMode: "manual", // Manual initialization for GDPR compliance
    initCommands: [
      // Setup consent mode v2 - default to denied
      // Wait for up to 2 seconds for consent update before proceeding
      [
        "consent",
        "default",
        {
          ad_user_data: "denied",
          ad_personalization: "denied",
          ad_storage: "denied",
          analytics_storage: "denied",
          wait_for_update: 5000, // Increased from 500ms to allow time for user choice
        },
      ],
    ],
  },

  // Runtime configuration
  runtimeConfig: {
    public: {
      // API URL for authentication backend
      // Set via environment variable:
      // NUXT_PUBLIC_API_URL=https://api.yourdomain.com
      apiUrl: "",
    },
  },
 

    app: {
    head: {
      link: [
        { rel: "apple-touch-icon", sizes: "57x57", href: "/apple-icon-57x57.png" },
        { rel: "apple-touch-icon", sizes: "60x60", href: "/apple-icon-60x60.png" },
        { rel: "apple-touch-icon", sizes: "72x72", href: "/apple-icon-72x72.png" },
        { rel: "apple-touch-icon", sizes: "76x76", href: "/apple-icon-76x76.png" },
        { rel: "apple-touch-icon", sizes: "114x114", href: "/apple-icon-114x114.png" },
        { rel: "apple-touch-icon", sizes: "120x120", href: "/apple-icon-120x120.png" },
        { rel: "apple-touch-icon", sizes: "144x144", href: "/apple-icon-144x144.png" },
        { rel: "apple-touch-icon", sizes: "152x152", href: "/apple-icon-152x152.png" },
        { rel: "apple-touch-icon", sizes: "180x180", href: "/apple-icon-180x180.png" },
        { rel: "icon", type: "image/png", sizes: "192x192", href: "/android-icon-192x192.png" },
        { rel: "icon", type: "image/png", sizes: "32x32", href: "/favicon-32x32.png" },
        { rel: "icon", type: "image/png", sizes: "96x96", href: "/favicon-96x96.png" },
        { rel: "icon", type: "image/png", sizes: "16x16", href: "/favicon-16x16.png" },
        { rel: "manifest", href: "/manifest.json" }, 
      ],
      meta: [
        { name: "msapplication-TileColor", content: "#ffffff" },
        { name: "msapplication-TileImage", content: "/ms-icon-144x144.png" },
        { name: "theme-color", content: "#ffffff" }
      ]
    }
  }
});
