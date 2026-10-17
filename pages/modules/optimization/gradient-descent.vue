<template>
  <ModuleLayout title="Gradient Descent Optimization"
    description="Visualize how different optimization algorithms navigate loss landscapes to find optimal parameters"
    difficulty="intermediate" :estimated-time="45" :table-of-contents="tableOfContents"
    module-id="gradient-descent"
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

          <!-- Step Controls -->
          <div class="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Step Controls</h3>
            <div class="flex gap-2">
              <Button variant="secondary" size="sm" :disabled="isOptimizing || state === 'completed'"
                @click="handleStep">
                Step
              </Button>
              <Button variant="secondary" size="sm" :disabled="isOptimizing || state === 'completed'"
                @click="handlePlay">
                {{ state === 'running' ? 'Pause' : 'Play' }}
              </Button>
              <Button variant="secondary" size="sm" :disabled="state === 'idle'" @click="handleResetOptimization">
                Reset
              </Button>
            </div>
          </div>

          <!-- Metrics Display -->
          <div v-if="currentLoss !== null" class="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Current State</h3>
            <div class="space-y-2">
              <div>
                <div class="text-xs text-gray-500">Iteration</div>
                <div class="text-lg font-semibold text-gray-900">{{ currentIteration }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Loss</div>
                <div class="text-lg font-semibold text-gray-900">{{ currentLoss.toFixed(6) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Parameters</div>
                <div class="text-sm font-mono text-gray-900">
                  [{{ currentParams[0]?.toFixed(3) }}, {{ currentParams[1]?.toFixed(3) }}]
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Gradient Norm</div>
                <div class="text-sm font-mono text-gray-900">{{ gradientNorm.toFixed(6) }}</div>
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <!-- Visualization Type Selector -->
            <div class="flex gap-2 mb-4">
              <Button :variant="visualizationType === 'contour' ? 'primary' : 'secondary'" size="sm"
                @click="visualizationType = 'contour'">
                Contour Plot
              </Button>
              <Button :variant="visualizationType === '3d' ? 'primary' : 'secondary'" size="sm"
                @click="visualizationType = '3d'">
                3D Surface
              </Button>
            </div>

            <!-- Plotly Visualization -->
            <Plot3D v-if="plotData && plotLayout" :data="plotData" :layout="plotLayout" :config="plotConfig"
              :height="visualizationType === '3d' ? '600px' : '500px'" />

            <!-- Loss History Chart -->
            <LineChart v-if="lossHistory && lossHistory.length > 0" :series="lossHistory" title="Loss Over Iterations"
              x-label="Iteration" y-label="Loss" :show-legend="false" height="300px" />
          </div>
        </template>
      </InteractiveSection>
    </template>

    <!-- Quiz Section -->
    <!-- Requirements: 3.1, 3.2, 3.3, 3.4 -->
    <template #quiz>
      <!-- Loading state -->
      <QuizLoadingState v-if="quizLoading" />

      <!-- Error state with retry -->
      <QuizErrorDisplay v-else-if="quizError" :error="quizError" @retry="retryQuizLoad" />

      <!-- Coming soon state -->
      <QuizErrorDisplay v-else-if="!quizData" :coming-soon="true" />

      <!-- Quiz component -->
      <Quiz v-else :quiz="quizData" :passing-score="quizData.passingScore || 70" @quiz-completed="handleQuizComplete" />
    </template>
  </ModuleLayout>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue';
import { useModuleStore } from '~/stores/module';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { GradientDescentVisualization } from '~/algorithms/optimization/GradientDescentVisualization';
import { getLossFunction, getAvailableLossFunctions } from '~/algorithms/optimization/LossFunctions';
import {
  generateLossLandscape,
  generateContourPlotData,
  generate3DSurfacePlotData,
  getContourPlotLayout,
  get3DSurfacePlotLayout,
  getPlotConfig as getPlotlyConfig
} from '~/algorithms/optimization/LossLandscape';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { LineSeries } from '~/components/visualizations/LineChart.vue';
import type { OptimizationState } from '~/algorithms/optimization/GradientDescentVisualization';
import type { QuizResult } from '~/types/quiz';
import Quiz from '~/components/modules/Quiz.vue';
import QuizLoadingState from '~/components/modules/QuizLoadingState.vue';
import QuizErrorDisplay from '~/components/modules/QuizErrorDisplay.vue';
import { useQuizLoader } from '~/composables/useQuizLoader';
import { useBookmarkRestore } from '~/composables/useBookmarkRestore';


// Load content from Nuxt Content
const { data: content } = await useAsyncData(() =>
  queryCollection('content').path('/modules/optimization/gradient-descent/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores
const moduleStore = useModuleStore();

// Parameters
const parameters = ref({
  lossFunction: 'quadratic-bowl',
  optimizer: 'sgd',
  learningRate: 0.1,
  momentum: 0.9,
  beta: 0.9,
  beta1: 0.9,
  beta2: 0.999,
  startX: -3,
  startY: 2,
});

// Get available loss functions
const lossFunctionOptions = getAvailableLossFunctions().map(fn => ({
  label: fn.name,
  value: fn.id
}));

// Controls configuration
const controls = computed<ControlConfig[]>(() => [
  {
    id: 'lossFunction',
    type: 'select',
    label: 'Loss Function',
    options: lossFunctionOptions,
    tooltip: 'Choose a loss landscape to optimize',
    group: 'Function',
  },
  {
    id: 'optimizer',
    type: 'select',
    label: 'Optimizer',
    options: [
      { label: 'SGD', value: 'sgd' },
      { label: 'SGD with Momentum', value: 'momentum' },
      { label: 'RMSProp', value: 'rmsprop' },
      { label: 'Adam', value: 'adam' },
    ],
    tooltip: 'Choose an optimization algorithm',
    group: 'Optimizer',
  },
  {
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.001,
    max: 1.0,
    step: 0.001,
    tooltip: 'Controls step size in parameter space',
    group: 'Optimizer',
  },
  ...(parameters.value.optimizer === 'momentum' ? [{
    id: 'momentum',
    type: 'slider',
    label: 'Momentum',
    min: 0,
    max: 0.99,
    step: 0.01,
    tooltip: 'Momentum coefficient (β)',
    group: 'Optimizer',
  }] as ControlConfig[] : []),
  ...(parameters.value.optimizer === 'rmsprop' ? [{
    id: 'beta',
    type: 'slider',
    label: 'Beta',
    min: 0.8,
    max: 0.999,
    step: 0.001,
    tooltip: 'Decay rate for squared gradient average',
    group: 'Optimizer',
  }] as ControlConfig[] : []),
  ...(parameters.value.optimizer === 'adam' ? [
    {
      id: 'beta1',
      type: 'slider',
      label: 'Beta 1',
      min: 0.8,
      max: 0.999,
      step: 0.001,
      tooltip: 'Decay rate for first moment',
      group: 'Optimizer',
    },
    {
      id: 'beta2',
      type: 'slider',
      label: 'Beta 2',
      min: 0.8,
      max: 0.9999,
      step: 0.0001,
      tooltip: 'Decay rate for second moment',
      group: 'Optimizer',
    }
  ] as ControlConfig[] : []),
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
const state = ref<OptimizationState>('idle');

// Visualization
const visualizationType = ref<'contour' | '3d'>('contour');
const plotData = ref<any>(null);
const plotLayout = ref<any>(null);
const plotConfig = ref<any>(getPlotlyConfig());

// Current state
const currentIteration = ref(0);
const currentLoss = ref<number | null>(null);
const currentParams = ref<number[]>([0, 0]);
const currentGradient = ref<number[]>([0, 0]);

// Computed
const gradientNorm = computed(() => {
  const g = currentGradient.value;
  if (!g || g.length < 2) return 0;
  return Math.sqrt(g[0] * g[0] + g[1] * g[1]);
});

const lossHistory = computed<LineSeries[]>(() => {
  if (!visualization.value) return [];

  const path = visualization.value.getOptimizationPath();
  if (path.length === 0) return [];

  return [{
    x: path.map((step: any) => step.iteration),
    y: path.map((step: any) => step.loss),
    name: 'Loss',
    color: '#10b981',
    width: 2,
  }];
});

// Visualization instance
let visualization = ref<GradientDescentVisualization | null>(null);
let animationFrameId: number | null = null;

// Quiz data with loading state
// Requirements: 1.1, 2.1, 2.4
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('gradient-descent');
  moduleStore.setParameters(parameters.value);

  // Initialize visualization
  initializeVisualization();

  // Load quiz data
  await loadQuiz('gradient-descent', '/content/modules/optimization/gradient-descent/quiz.json');
});

/**
 * Initialize gradient descent visualization
 */
const initializeVisualization = () => {
  const lossFunc = getLossFunction(parameters.value.lossFunction);
  const initialParams = [parameters.value.startX, parameters.value.startY];

  visualization.value = new GradientDescentVisualization(
    lossFunc.loss,
    lossFunc.gradient,
    initialParams,
    {
      optimizerType: parameters.value.optimizer as any,
      learningRate: parameters.value.learningRate,
      maxIterations: 1000,
      tolerance: 1e-6,
      momentum: parameters.value.momentum,
      beta: parameters.value.beta,
      beta1: parameters.value.beta1,
      beta2: parameters.value.beta2,
    }
  );

  // Update current state
  updateCurrentState();

  // Generate landscape and update visualization
  updateVisualization();
};

/**
 * Update current state from visualization
 */
const updateCurrentState = () => {
  if (!visualization.value) return;

  currentIteration.value = visualization.value.getCurrentIteration();
  currentLoss.value = visualization.value.getCurrentLoss();
  currentParams.value = visualization.value.getCurrentParams();
  currentGradient.value = visualization.value.getCurrentGradient();
  state.value = visualization.value.getState();
};

/**
 * Update visualization
 */
const updateVisualization = () => {
  if (!visualization.value) return;

  const lossFunc = getLossFunction(parameters.value.lossFunction);
  const domain = lossFunc.domain || { min: [-5, -5], max: [5, 5] };

  // Generate loss landscape
  const landscape = generateLossLandscape(lossFunc.loss, {
    xRange: [domain.min[0], domain.max[0]],
    yRange: [domain.min[1], domain.max[1]],
    resolution: 50
  });

  // Get optimization path
  const path = visualization.value.getOptimizationPath();

  // Generate plot data based on visualization type
  if (visualizationType.value === 'contour') {
    plotData.value = generateContourPlotData(landscape, path);
    plotLayout.value = getContourPlotLayout(`${lossFunc.name} - ${visualization.value.getOptimizerName()}`);
  } else {
    plotData.value = generate3DSurfacePlotData(landscape, path);
    plotLayout.value = get3DSurfacePlotLayout(`${lossFunc.name} - ${visualization.value.getOptimizerName()}`);
  }
};

/**
 * Watch for visualization type changes
 */
watch(visualizationType, () => {
  updateVisualization();
});

/**
 * Handle parameter change
 */
const handleParameterChange = (id: string, value: any) => {
  (parameters.value as any)[id] = value;
  moduleStore.setParameter(id, value);

  // Reinitialize if loss function or initial params changed
  if (['lossFunction', 'startX', 'startY', 'optimizer', 'learningRate', 'momentum', 'beta', 'beta1', 'beta2'].includes(id)) {
    initializeVisualization();
  }
};

/**
 * Handle reset
 */
const handleReset = () => {
  parameters.value = {
    lossFunction: 'quadratic-bowl',
    optimizer: 'sgd',
    learningRate: 0.1,
    momentum: 0.9,
    beta: 0.9,
    beta1: 0.9,
    beta2: 0.999,
    startX: -3,
    startY: 2,
  };
  moduleStore.setParameters(parameters.value);

  initializeVisualization();
  error.value = null;
};

/**
 * Handle start optimization
 */
const handleStart = async () => {
  if (!visualization.value) return;

  isOptimizing.value = true;
  error.value = null;

  try {
    await visualization.value.runToCompletion();
    updateCurrentState();
    updateVisualization();
  } catch (err) {
    console.error('Optimization error:', err);
    error.value = err instanceof Error ? err.message : 'Optimization failed';
  } finally {
    isOptimizing.value = false;
  }
};

/**
 * Handle single step
 */
const handleStep = () => {
  if (!visualization.value) return;

  visualization.value.step();
  updateCurrentState();
  updateVisualization();
};

/**
 * Handle play/pause
 */
const handlePlay = () => {
  if (!visualization.value) return;

  if (state.value === 'running') {
    visualization.value.pause();
    if (animationFrameId !== null) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }
  } else {
    visualization.value.resume();
    animateOptimization();
  }

  updateCurrentState();
};

/**
 * Animate optimization
 */
const animateOptimization = () => {
  if (!visualization.value || state.value !== 'running') return;

  // Perform step
  visualization.value.step();
  updateCurrentState();
  updateVisualization();

  // Continue animation if not completed
  if (!visualization.value.isComplete()) {
    animationFrameId = requestAnimationFrame(() => {
      setTimeout(animateOptimization, 100); // 10 steps per second
    });
  }
};

/**
 * Handle reset optimization
 */
const handleResetOptimization = () => {
  if (animationFrameId !== null) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }

  initializeVisualization();
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
 * Processes quiz results and verifies progress store integration
 * 
 * Requirements: 3.3, 3.4
 */
const handleQuizComplete = (result: QuizResult) => {
  console.log('Quiz completed:', result);

  // The Quiz component already handles:
  // - Saving score to progress store
  // - Marking module as complete if score >= 70%
  // - Tracking analytics
};

// Bookmark restoration
// Requirements: 2.2, 2.3
const { handleLoadBookmark } = useBookmarkRestore(parameters, initializeVisualization);
</script>
