<template>
  <ModuleLayout title="Momentum & Nesterov"
    description="Learn how momentum accelerates optimization by accumulating gradients over time"
    difficulty="intermediate" :estimated-time="35" :table-of-contents="tableOfContents"
    module-id="momentum-nesterov"
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
                <div class="text-xs text-gray-500">Momentum Type</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.momentumType }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Avg Velocity</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.averageVelocityMagnitude?.toFixed(6) }}</div>
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

            <!-- Velocity Evolution Chart (Momentum-specific) -->
            <LineChart v-if="velocityHistory && velocityHistory.length > 0" :series="velocityHistory" 
              title="Velocity Evolution" x-label="Iteration" y-label="Velocity" height="400px" />

            <!-- Gradient vs Velocity Magnitude Chart -->
            <LineChart v-if="magnitudeHistory && magnitudeHistory.length > 0" :series="magnitudeHistory" 
              title="Gradient vs Velocity Magnitude" x-label="Iteration" y-label="Magnitude" height="400px" />
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
import { Momentum } from '~/algorithms/optimization/Momentum';
import type { MomentumType } from '~/algorithms/optimization/Momentum';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { LineSeries } from '~/components/visualizations/LineChart.vue';
import type { QuizResult } from '~/types/quiz';
import Quiz from '~/components/modules/Quiz.vue';
import QuizLoadingState from '~/components/modules/QuizLoadingState.vue';
import QuizErrorDisplay from '~/components/modules/QuizErrorDisplay.vue';
import { useQuizLoader } from '~/composables/useQuizLoader';
import { useBookmarkRestore } from '~/composables/useBookmarkRestore';

// Load content from Nuxt Content
const { data: content } = await useAsyncData(() =>
  queryCollection('content').path('/modules/optimization/momentum-nesterov/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores
const moduleStore = useModuleStore();

// Parameters
const parameters = ref({
  momentumType: 'standard' as MomentumType,
  lossFunction: 'quadratic-bowl',
  learningRate: 0.01,
  momentum: 0.9,
  maxIterations: 200,
  tolerance: 1e-6,
  startX: -3,
  startY: 2,
});

// Controls configuration
const controls = computed<ControlConfig[]>(() => [
  {
    id: 'momentumType',
    type: 'select',
    label: 'Momentum Type',
    options: [
      { label: 'No Momentum (SGD)', value: 'none' },
      { label: 'Standard Momentum', value: 'standard' },
      { label: 'Nesterov Momentum', value: 'nesterov' },
    ],
    tooltip: 'Choose between no momentum, standard momentum, or Nesterov momentum',
    group: 'Optimizer',
  },
  {
    id: 'lossFunction',
    type: 'select',
    label: 'Loss Function',
    options: [
      { label: 'Quadratic Bowl', value: 'quadratic-bowl' },
      { label: 'Elongated Valley', value: 'elongated-valley' },
      { label: 'Rosenbrock', value: 'rosenbrock' },
      { label: 'Saddle Point', value: 'saddle-point' },
      { label: 'Zigzag', value: 'zigzag' },
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
    tooltip: 'Base learning rate for parameter updates',
    group: 'Optimizer',
  },
  {
    id: 'momentum',
    type: 'slider',
    label: 'Momentum',
    min: 0.0,
    max: 0.99,
    step: 0.01,
    tooltip: 'Momentum coefficient (0 = no momentum, 0.9 = high momentum)',
    group: 'Optimizer',
  },
  {
    id: 'maxIterations',
    type: 'slider',
    label: 'Max Iterations',
    min: 50,
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
let algorithm = ref<Momentum | null>(null);

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

const velocityHistory = computed<LineSeries[]>(() => {
  if (!metrics.value || !algorithm.value || parameters.value.momentumType === 'none') return [];

  const vizData = algorithm.value.getVisualizationData();
  const velocityViz = vizData.find(v => v.type === 'line' && v.layout?.title?.includes('Velocity Evolution'));
  
  if (!velocityViz || !Array.isArray(velocityViz.data)) return [];

  return velocityViz.data.map((series: any, idx: number) => ({
    x: series.x as number[],
    y: series.y as number[],
    name: series.name || `Velocity ${idx}`,
    color: idx === 0 ? '#8b5cf6' : '#ec4899',
    width: 2,
  }));
});

const magnitudeHistory = computed<LineSeries[]>(() => {
  if (!metrics.value || !algorithm.value || parameters.value.momentumType === 'none') return [];

  const vizData = algorithm.value.getVisualizationData();
  const magViz = vizData.find(v => v.type === 'line' && v.layout?.title?.includes('Gradient vs Velocity'));
  
  if (!magViz || !Array.isArray(magViz.data)) return [];

  return magViz.data.map((series: any, idx: number) => ({
    x: series.x as number[],
    y: series.y as number[],
    name: series.name || `Magnitude ${idx}`,
    color: idx === 0 ? '#ef4444' : '#10b981',
    width: 2,
  }));
});

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('momentum-nesterov');
  moduleStore.setParameters(parameters.value);

  // Initialize algorithm
  initializeAlgorithm();

  // Load quiz data
  await loadQuiz('momentum-nesterov', '/content/modules/optimization/momentum-nesterov/quiz.json');
});

/**
 * Initialize Momentum algorithm
 */
const initializeAlgorithm = () => {
  algorithm.value = new Momentum(parameters.value);
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
    momentumType: 'standard' as MomentumType,
    lossFunction: 'quadratic-bowl',
    learningRate: 0.01,
    momentum: 0.9,
    maxIterations: 200,
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