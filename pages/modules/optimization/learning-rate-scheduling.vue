<template>
  <ModuleLayout title="Learning Rate Scheduling"
    description="Learn how learning rate scheduling improves optimization convergence and stability"
    difficulty="intermediate" :estimated-time="40" :table-of-contents="tableOfContents"
    module-id="learning-rate-scheduling"
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
                <div class="text-xs text-gray-500">Schedule Type</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.scheduleType }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Final LR</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.finalLearningRate?.toFixed(6) }}</div>
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <!-- Learning Rate Schedule Chart -->
            <LineChart v-if="lrScheduleHistory && lrScheduleHistory.length > 0" :series="lrScheduleHistory" 
              title="Learning Rate Schedule" x-label="Iteration" y-label="Learning Rate" :show-legend="false" height="400px" />

            <!-- Loss History Chart -->
            <LineChart v-if="lossHistory && lossHistory.length > 0" :series="lossHistory" title="Loss Over Iterations"
              x-label="Iteration" y-label="Loss" :show-legend="false" height="400px" />

            <!-- Parameter Evolution Chart -->
            <LineChart v-if="paramHistory && paramHistory.length > 0" :series="paramHistory" 
              title="Parameter Evolution" x-label="Iteration" y-label="Parameter Value" height="400px" />

            <!-- Gradient Magnitude Chart -->
            <LineChart v-if="gradientHistory && gradientHistory.length > 0" :series="gradientHistory" 
              title="Gradient Magnitude" x-label="Iteration" y-label="Magnitude" :show-legend="false" height="400px" />
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
import { LearningRateScheduler } from '~/algorithms/optimization/LearningRateScheduler';
import type { ScheduleType } from '~/algorithms/optimization/LearningRateScheduler';
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
  queryCollection('content').path('/modules/optimization/learning-rate-scheduling/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores
const moduleStore = useModuleStore();

// Parameters
const parameters = ref({
  scheduleType: 'constant' as ScheduleType,
  lossFunction: 'quadratic-bowl',
  initialLearningRate: 0.1,
  maxIterations: 200,
  tolerance: 1e-6,
  startX: -3,
  startY: 2,
  stepSize: 50,
  gamma: 0.5,
  decayRate: 0.95,
  minLearningRate: 0.001,
  power: 1.0,
  patience: 10,
  factor: 0.5,
  threshold: 1e-4,
});

// Controls configuration - dynamically adjust based on schedule type
const controls = computed<ControlConfig[]>(() => {
  const baseControls: ControlConfig[] = [
    {
      id: 'scheduleType',
      type: 'select',
      label: 'Schedule Type',
      options: [
        { label: 'Constant', value: 'constant' },
        { label: 'Step Decay', value: 'step' },
        { label: 'Exponential Decay', value: 'exponential' },
        { label: 'Cosine Annealing', value: 'cosine' },
        { label: 'Polynomial Decay', value: 'polynomial' },
        { label: 'Reduce on Plateau', value: 'plateau' },
      ],
      tooltip: 'Choose a learning rate scheduling strategy',
      group: 'Schedule',
    },
    {
      id: 'lossFunction',
      type: 'select',
      label: 'Loss Function',
      options: [
        { label: 'Quadratic Bowl', value: 'quadratic-bowl' },
        { label: 'Elongated Valley', value: 'elongated-valley' },
        { label: 'Rosenbrock', value: 'rosenbrock' },
        { label: 'Noisy Quadratic', value: 'noisy-quadratic' },
      ],
      tooltip: 'Choose a loss landscape to optimize',
      group: 'Function',
    },
    {
      id: 'initialLearningRate',
      type: 'slider',
      label: 'Initial Learning Rate',
      min: 0.001,
      max: 1.0,
      step: 0.001,
      tooltip: 'Starting learning rate value',
      group: 'Schedule',
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
  ];

  // Add schedule-specific controls
  const scheduleType = parameters.value.scheduleType;
  
  if (scheduleType === 'step') {
    baseControls.push(
      {
        id: 'stepSize',
        type: 'slider',
        label: 'Step Size',
        min: 10,
        max: 100,
        step: 5,
        tooltip: 'Number of iterations between learning rate reductions',
        group: 'Step Schedule',
      },
      {
        id: 'gamma',
        type: 'slider',
        label: 'Gamma',
        min: 0.1,
        max: 0.9,
        step: 0.1,
        tooltip: 'Factor by which learning rate is reduced',
        group: 'Step Schedule',
      }
    );
  } else if (scheduleType === 'exponential') {
    baseControls.push({
      id: 'decayRate',
      type: 'slider',
      label: 'Decay Rate',
      min: 0.9,
      max: 0.999,
      step: 0.001,
      tooltip: 'Exponential decay factor per iteration',
      group: 'Exponential Schedule',
    });
  } else if (scheduleType === 'cosine') {
    baseControls.push({
      id: 'minLearningRate',
      type: 'slider',
      label: 'Min Learning Rate',
      min: 0.0001,
      max: 0.01,
      step: 0.0001,
      tooltip: 'Minimum learning rate for cosine annealing',
      group: 'Cosine Schedule',
    });
  } else if (scheduleType === 'polynomial') {
    baseControls.push({
      id: 'power',
      type: 'slider',
      label: 'Power',
      min: 0.5,
      max: 2.0,
      step: 0.1,
      tooltip: 'Power for polynomial decay',
      group: 'Polynomial Schedule',
    });
  } else if (scheduleType === 'plateau') {
    baseControls.push(
      {
        id: 'patience',
        type: 'slider',
        label: 'Patience',
        min: 5,
        max: 50,
        step: 1,
        tooltip: 'Number of iterations to wait before reducing LR',
        group: 'Plateau Schedule',
      },
      {
        id: 'factor',
        type: 'slider',
        label: 'Factor',
        min: 0.1,
        max: 0.9,
        step: 0.1,
        tooltip: 'Factor by which to reduce LR on plateau',
        group: 'Plateau Schedule',
      }
    );
  }

  // Add initial parameters controls
  baseControls.push(
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
    }
  );

  return baseControls;
});

// Optimization state
const isOptimizing = ref(false);
const optimizationProgress = ref(0);
const error = ref<string | null>(null);
const metrics = ref<any>(null);

// Algorithm instance
let algorithm = ref<LearningRateScheduler | null>(null);

// Computed visualization data
const lrScheduleHistory = computed<LineSeries[]>(() => {
  if (!metrics.value || !algorithm.value) return [];

  const vizData = algorithm.value.getVisualizationData();
  const lrViz = vizData.find(v => v.type === 'line' && v.layout?.title?.includes('Learning Rate Schedule'));
  
  if (!lrViz || !lrViz.data) return [];

  return [{
    x: lrViz.data.x as number[],
    y: lrViz.data.y as number[],
    name: 'Learning Rate',
    color: '#8b5cf6',
    width: 2,
  }];
});

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

const gradientHistory = computed<LineSeries[]>(() => {
  if (!metrics.value || !algorithm.value) return [];

  const vizData = algorithm.value.getVisualizationData();
  const gradViz = vizData.find(v => v.type === 'line' && v.layout?.title?.includes('Gradient Magnitude'));
  
  if (!gradViz || !gradViz.data) return [];

  return [{
    x: gradViz.data.x as number[],
    y: gradViz.data.y as number[],
    name: 'Gradient Magnitude',
    color: '#ef4444',
    width: 2,
  }];
});

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('learning-rate-scheduling');
  moduleStore.setParameters(parameters.value);

  // Initialize algorithm
  initializeAlgorithm();

  // Load quiz data
  await loadQuiz('learning-rate-scheduling', '/content/modules/optimization/learning-rate-scheduling/quiz.json');
});

/**
 * Initialize Learning Rate Scheduler algorithm
 */
const initializeAlgorithm = () => {
  algorithm.value = new LearningRateScheduler(parameters.value);
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
    scheduleType: 'constant' as ScheduleType,
    lossFunction: 'quadratic-bowl',
    initialLearningRate: 0.1,
    maxIterations: 200,
    tolerance: 1e-6,
    startX: -3,
    startY: 2,
    stepSize: 50,
    gamma: 0.5,
    decayRate: 0.95,
    minLearningRate: 0.001,
    power: 1.0,
    patience: 10,
    factor: 0.5,
    threshold: 1e-4,
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