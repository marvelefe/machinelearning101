<template>
    <div
        class="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 px-4">
        <div class="max-w-2xl w-full text-center">
            <!-- Error Code -->
            <div class="mb-8">
                <h1 class="text-9xl font-bold text-indigo-600 dark:text-indigo-400 mb-4">
                    {{ error.statusCode }}
                </h1>
                <div class="text-6xl mb-4">{{ errorEmoji }}</div>
            </div>

            <!-- Error Message -->
            <h2 class="text-3xl font-bold text-gray-900 dark:text-white mb-4">
                {{ errorTitle }}
            </h2>
            <p class="text-lg text-gray-600 dark:text-gray-300 mb-8">
                {{ errorMessage }}
            </p>

            <!-- Action Buttons -->
            <div class="flex flex-col sm:flex-row gap-4 justify-center items-center">
                <NuxtLink to="/"
                    class="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-lg transition-colors duration-200 inline-flex items-center gap-2">
                    <Icon name="heroicons:home" class="w-5 h-5" />
                    Go Home
                </NuxtLink>

                <NuxtLink to="/modules"
                    class="px-6 py-3 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-900 dark:text-white font-medium rounded-lg border border-gray-300 dark:border-gray-600 transition-colors duration-200 inline-flex items-center gap-2">
                    <Icon name="heroicons:academic-cap" class="w-5 h-5" />
                    Browse Modules
                </NuxtLink>
            </div>

            <!-- Helpful Links (only for 404) -->
            <div v-if="error.statusCode === 404" class="mt-12 pt-8 border-t border-gray-300 dark:border-gray-700">
                <p class="text-sm text-gray-500 dark:text-gray-400 mb-4">
                    Popular learning paths:
                </p>
                <div class="flex flex-wrap gap-3 justify-center">
                    <NuxtLink to="/modules/regression/linear-regression"
                        class="text-sm text-indigo-600 dark:text-indigo-400 hover:underline">
                        Linear Regression
                    </NuxtLink>
                    <span class="text-gray-300 dark:text-gray-600">•</span>
                    <NuxtLink to="/modules/classification/logistic-regression"
                        class="text-sm text-indigo-600 dark:text-indigo-400 hover:underline">
                        Classification
                    </NuxtLink>
                    <span class="text-gray-300 dark:text-gray-600">•</span>
                    <NuxtLink to="/modules/clustering/k-means"
                        class="text-sm text-indigo-600 dark:text-indigo-400 hover:underline">
                        K-Means Clustering
                    </NuxtLink>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import type { NuxtError } from '#app';

const props = defineProps({
    error: {
        type: Object as () => NuxtError,
        required: true
    }
})

// Determine error details based on status code
const errorEmoji = computed(() => {
    switch (props.error.statusCode) {
        case 404:
            return '🤖'
        case 500:
            return '⚠️'
        default:
            return '❌'
    }
})

const errorTitle = computed(() => {
    switch (props.error.statusCode) {
        case 404:
            return 'Page Not Found'
        case 500:
            return 'Server Error'
        default:
            return 'An Error Occurred'
    }
})

const errorMessage = computed(() => {
    switch (props.error.statusCode) {
        case 404:
            return "Looks like this algorithm hasn't been trained yet. The page you're looking for doesn't exist or has been moved."
        case 500:
            return 'Something went wrong on our end. Our algorithms are working to fix the issue.'
        default:
            return props.error.message || 'An unexpected error occurred. Please try again.'
    }
})

// Set page metadata
useSEO({
    title: `${props.error.statusCode} - ${errorTitle.value}`,
    description: errorMessage.value,
    url: '/error',
    type: 'website'
})
</script>
