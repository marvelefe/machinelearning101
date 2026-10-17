<template>
  <ModuleLayout title="Adam Optimizer"
    description="Learn how Adam combines momentum and adaptive learning rates for efficient optimization"
    difficulty="intermediate" :estimated-time="35" :table-of-contents="tableOfContents"
    module-id="adam-optimizer"
    :parameters="parameters"
    @load-bookmark="(val) => { parameters = val as typeof parameters; handleLoadBookmark(val); }">
    <!-- Content Section -->
    <template #content>
      <ContentRenderer v-if="content" :value="content" class="prose prose-slate max-w-none" />
    </template>

    <!-- Interactive Section -->
    <template #interactive>
      <InteractiveSection :loading="isOptimizing" :progress="optimizationProgress" loading-message="Optimizing..."
        :error="error" :on-retry="handleRetry">
        <!-- Controls -->
        <template #controls>
          <ControlPanel :controls="controls" :values="parameters" :loading="isOptimizing"
            submit-label="Start Optimization" @change="handleParameterChange" @reset="handleReset"
            @submit="handleStart" />

          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Optimization Results</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Iterations</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.iterations }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Final Loss</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.finalLoss?.toFixed(6) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Loss Reduction</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.lossReduction?.toFixed(6) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Convergence Rate</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.convergenceRate?.toFixed(6) }}</div>
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <!-- Loss History Chart -->
            <LineChart v-if="lossHistory && lossHistory.length > 0" :series="lossHistory" title="Loss Over Iterations"
              x-label="Iteration" y-label="Loss" :show-legend="false" height="400px" />

            <!-- Parameter Evolution Chart -->
            <LineChart v-if="paramHistory && paramHistory.length > 0" :series="paramHistory" 
              title="Parameter Evolution" x-label="Iteration" y-label="Parameter Value" height="400px" />

            <!-- Moment Evolution Chart (Adam-specific) -->
            <LineChart v-if="momentHistory && momentHistory.length > 0" :series="momentHistory" 
              title="Moment Evolution" x-label="Iteration" y-label="Moment Value" height="400px" />
          </div>
        </template>
      </InteractiveSection>
    </template>

    <!-- Quiz Section -->
    <template #quiz>
      <QuizLoadingState v-if="quizLoading" />
      <QuizErrorDisplay v-else-if="quizError" :error="quizError" @retry="retryQuizLoad" />
      <QuizErrorDisplay v-else-if="!quizData" :coming-soon="true" />
      <Quiz v-else :quiz="quizData" :passing-score="quizData.passingScore || 70" @quiz-completed="handleQuizComplete" />
    </template>
  </ModuleLayout>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { useModuleStore } from '~/stores/module';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { Adam } from '~/algorithms/optimization/Adam';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { LineSeries } from '~/components/visualizations/LineChart.vue';
import type { QuizResult } from '~/types/quiz';
import Quiz from '~/components/modules/Quiz.vue';
import QuizLoadingState from '~/components/modules/QuizLoadingState.vue';
import QuizErrorDisplay from '~/components/modules/QuizErrorDisplay.vue';
import { useQuizLoader } from '~/composables/useQuizLoader';
import { useBookmarkRestore } from '~/composables/useBookmarkRestore';

import { useAsyncData } from '#app'; 

// Load content from Nuxt Content
const { data: content } = await useAsyncData(() =>
  queryCollection('content').path('/modules/optimization/adam-optimizer/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores
const moduleStore = useModuleStore();

// Parameters
const parameters = ref({
  lossFunction: 'quadratic-bowl',
  learningRate: 0.01,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,
  maxIterations: 100,
  tolerance: 1e-6,
  startX: -3,
  startY: 2,
});

// Controls configuration
const controls = computed<ControlConfig[]>(() => [
  {
    id: 'lossFunction',
    type: 'select',
    label: 'Loss Function',
    options: [
      { label: 'Quadratic Bowl', value: 'quadratic-bowl' },
      { label: 'Elongated Valley', value: 'elongated-valley' },
      { label: 'Rosenbrock', value: 'rosenbrock' },
      { label: 'Saddle Point', value: 'saddle-point' },
      { label: 'Beale', value: 'beale' },
    ],
    tooltip: 'Choose a loss landscape to optimize',
    group: 'Function',
  },
  {
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.001,
    max: 0.1,
    step: 0.001,
    tooltip: 'Base learning rate for Adam updates',
    group: 'Optimizer',
  },
  {
    id: 'beta1',
    type: 'slider',
    label: 'Beta1 (Momentum)',
    min: 0.1,
    max: 0.999,
    step: 0.001,
    tooltip: 'Exponential decay rate for first moment estimates',
    group: 'Optimizer',
  },
  {
    id: 'beta2',
    type: 'slider',
    label: 'Beta2 (RMSProp)',
    min: 0.9,
    max: 0.9999,
    step: 0.0001,
    tooltip: 'Exponential decay rate for second moment estimates',
    group: 'Optimizer',
  },
  {
    id: 'epsilon',
    type: 'slider',
    label: 'Epsilon',
    min: 1e-10,
    max: 1e-6,
    step: 1e-10,
    tooltip: 'Small constant to prevent division by zero',
    group: 'Optimizer',
  },
  {
    id: 'maxIterations',
    type: 'slider',
    label: 'Max Iterations',
    min: 10,
    max: 500,
    step: 10,
    tooltip: 'Maximum number of optimization steps',
    group: 'Optimizer',
  },
  {
    id: 'startX',
    type: 'slider',
    label: 'Start X',
    min: -5,
    max: 5,
    step: 0.1,
    tooltip: 'Initial x parameter value',
    group: 'Initial Parameters',
  },
  {
    id: 'startY',
    type: 'slider',
    label: 'Start Y',
    min: -5,
    max: 5,
    step: 0.1,
    tooltip: 'Initial y parameter value',
    group: 'Initial Parameters',
  },
]);

// Optimization state
const isOptimizing = ref(false);
const optimizationProgress = ref(0);
const error = ref<string | null>(null);
const metrics = ref<any>(null);

// Algorithm instance
let algorithm = ref<Adam | null>(null);

// Computed visualization data
const lossHistory = computed<LineSeries[]>(() => {
  if (!metrics.value || !algorithm.value) return [];

  const vizData = algorithm.value.getVisualizationData();
  const lossViz = vizData.find(v => v.type === 'line' && v.layout?.title?.includes('Loss Over'));
  
  if (!lossViz || !lossViz.data) return [];

  return [{
    x: lossViz.data.x as number[],
    y: lossViz.data.y as number[],
    name: 'Loss',
    color: '#10b981',
    width: 2,
  }];
});

const paramHistory = computed<LineSeries[]>(() => {
  if (!metrics.value || !algorithm.value) return [];

  const vizData = algorithm.value.getVisualizationData();
  const paramViz = vizData.find(v => v.type === 'line' && v.layout?.title?.includes('Parameter Evolution'));
  
  if (!paramViz || !Array.isArray(paramViz.data)) return [];

  return paramViz.data.map((series: any, idx: number) => ({
    x: series.x as number[],
    y: series.y as number[],
    name: series.name || `Parameter ${idx}`,
    color: idx === 0 ? '#3b82f6' : '#f59e0b',
    width: 2,
  }));
});

const momentHistory = computed<LineSeries[]>(() => {
  if (!metrics.value || !algorithm.value) return [];

  const vizData = algorithm.value.getVisualizationData();
  const momentViz = vizData.find(v => v.type === 'line' && v.layout?.title?.includes('Moment Evolution'));
  
  if (!momentViz || !Array.isArray(momentViz.data)) return [];

  return momentViz.data.map((series: any, idx: number) => ({
    x: series.x as number[],
    y: series.y as number[],
    name: series.name || `Moment ${idx}`,
    color: ['#3b82f6', '#f59e0b', '#10b981', '#ef4444'][idx % 4],
    width: 2,
  }));
});

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('adam-optimizer');
  moduleStore.setParameters(parameters.value);

  // Initialize algorithm
  initializeAlgorithm();

  // Load quiz data
  await loadQuiz('adam-optimizer', '/content/modules/optimization/adam-optimizer/quiz.json');
});

/**
 * Initialize Adam algorithm
 */
const initializeAlgorithm = () => {
  algorithm.value = new Adam(parameters.value);
};

/**
 * Handle parameter change
 */
const handleParameterChange = (id: string, value: any) => {
  (parameters.value as any)[id] = value;
  moduleStore.setParameter(id, value);

  // Reinitialize algorithm with new parameters
  if (algorithm.value) {
    algorithm.value.updateConfig({ [id]: value });
  }
};

/**
 * Handle reset
 */
const handleReset = () => {
  parameters.value = {
    lossFunction: 'quadratic-bowl',
    learningRate: 0.01,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    maxIterations: 100,
    tolerance: 1e-6,
    startX: -3,
    startY: 2,
  };
  moduleStore.setParameters(parameters.value);

  initializeAlgorithm();
  metrics.value = null;
  error.value = null;
};

/**
 * Handle start optimization
 */
const handleStart = async () => {
  if (!algorithm.value) return;

  isOptimizing.value = true;
  error.value = null;
  optimizationProgress.value = 0;

  try {
    // Run optimization
    await algorithm.value.fit([]);
    
    // Get metrics
    metrics.value = algorithm.value.getMetrics();
    
    optimizationProgress.value = 100;
  } catch (err) {
    console.error('Optimization error:', err);
    error.value = err instanceof Error ? err.message : 'Optimization failed';
  } finally {
    isOptimizing.value = false;
  }
};

/**
 * Handle retry
 */
const handleRetry = () => {
  error.value = null;
  handleStart();
};

/**
 * Handle quiz completion
 */
const handleQuizComplete = (result: QuizResult) => {
  console.log('Quiz completed:', result);
};

// Bookmark restoration
// Requirements: 2.2, 2.3
const { handleLoadBookmark } = useBookmarkRestore(parameters, initializeAlgorithm);
</script>