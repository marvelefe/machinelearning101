<template>
  <ModuleLayout
    title="Hyperparameter Tuning"
    description="Learn how to optimize model hyperparameters using grid search, random search, and Bayesian optimization"
    difficulty="advanced"
    :estimated-time="45"
    module-id="hyperparameter-tuning"
    :table-of-contents="tableOfContents"
    :parameters="parameters"
    @load-bookmark="(val) => { parameters = val as typeof parameters; handleLoadBookmark(val); }"
  >
    <!-- Content Section -->
    <template #content>
      <ContentRenderer v-if="content" :value="content" class="prose prose-slate max-w-none" />
    </template>

    <!-- Interactive Section -->
    <template #interactive>
      <InteractiveSection
        :loading="isTraining"
        :progress="trainingProgress"
        loading-message="Tuning hyperparameters..."
        :error="error"
        :on-retry="handleRetry"
      >
        <!-- Controls -->
        <template #controls>
          <ControlPanel
            :controls="controls"
            :values="parameters"
            :loading="isTraining"
            submit-label="Start Tuning"
            @change="handleParameterChange"
            @reset="handleReset"
            @submit="handleTrain"
          />

          <!-- Tuning Configuration Display -->
          <div class="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
            <h3 class="text-sm font-semibold text-blue-700 mb-3">Tuning Configuration</h3>
            <div class="space-y-2">
              <div class="flex items-center gap-2">
                <span class="text-xs text-blue-600 w-40">Search Strategy:</span>
                <span class="text-sm font-semibold text-blue-900 capitalize">
                  {{ parameters.searchStrategy }}
                </span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-blue-600 w-40">Max Iterations:</span>
                <span class="text-sm font-semibold text-blue-900">{{ parameters.maxIterations }}</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-blue-600 w-40">CV Folds:</span>
                <span class="text-sm font-semibold text-blue-900">{{ parameters.crossValidationFolds }}</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-blue-600 w-40">Scoring Metric:</span>
                <span class="text-sm font-semibold text-blue-900 uppercase">{{ parameters.scoringMetric }}</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-blue-600 w-40">Early Stopping:</span>
                <span class="text-sm font-semibold text-blue-900">{{ parameters.earlyStoppingPatience }} iterations</span>
              </div>
            </div>
          </div>

          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Tuning Results</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Best Score</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.bestScore?.toFixed(4) }}
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Total Iterations</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.totalIterations }}
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Best Iteration</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.bestIteration }}
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Convergence Rate</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ (metrics.convergenceRate * 100).toFixed(1) }}%
                </div>
              </div>
            </div>
          </div>

          <!-- Best Parameters Display -->
          <div v-if="bestParameters" class="mt-6 p-4 bg-green-50 rounded-lg border border-green-200">
            <h3 class="text-sm font-semibold text-green-700 mb-3">Best Parameters Found</h3>
            <div class="space-y-1">
              <div v-for="(value, key) in bestParameters" :key="key" class="flex items-center gap-2">
                <span class="text-xs text-green-600 w-32">{{ key }}:</span>
                <span class="text-sm font-semibold text-green-900">{{ value }}</span>
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <div class="grid grid-cols-2 gap-4">
              <LineChart
                v-if="optimizationProgressSeries && optimizationProgressSeries.length > 0"
                :series="optimizationProgressSeries"
                title="Optimization Progress"
                x-label="Iteration"
                y-label="Validation Score"
                :show-legend="false"
                height="300px"
              />

              <LineChart
                v-if="bestScoreSeries && bestScoreSeries.length > 0"
                :series="bestScoreSeries"
                title="Best Score Evolution"
                x-label="Iteration"
                y-label="Best Score So Far"
                :show-legend="false"
                height="300px"
              />
            </div>

            <Heatmap
              v-if="parameterSpaceViz"
              :data="parameterSpaceViz"
              title="Parameter Space Exploration"
              x-label="Parameter 1"
              y-label="Parameter 2"
              height="400px"
            />

            <div v-if="parameterImportanceViz" class="p-4 bg-white rounded-lg border border-gray-200">
              <h3 class="text-sm font-semibold text-gray-700 mb-3">Parameter Importance</h3>
              <div class="space-y-2">
                <div v-for="(importance, param) in parameterImportanceViz" :key="param" class="flex items-center gap-2">
                  <span class="text-xs text-gray-600 w-32">{{ param }}:</span>
                  <div class="flex-1 bg-gray-200 rounded-full h-4">
                    <div 
                      class="bg-blue-500 h-4 rounded-full"
                      :style="{ width: `${importance * 100}%` }"
                    ></div>
                  </div>
                  <span class="text-xs text-gray-600 w-12 text-right">{{ (importance * 100).toFixed(1) }}%</span>
                </div>
              </div>
            </div>
          </div>
        </template>
      </InteractiveSection>
    </template>

    <!-- Quiz Section -->
    <template #quiz>
      <QuizLoadingState v-if="quizLoading" />
      <QuizErrorDisplay v-else-if="quizError" :error="quizError" @retry="retryQuizLoad" />
      <QuizErrorDisplay v-else-if="!quizData" :coming-soon="true" />
      <Quiz
        v-else
        :quiz="quizData"
        :passing-score="quizData.passingScore || 70"
        @quiz-completed="handleQuizComplete"
      />
    </template>
  </ModuleLayout>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { useModuleStore } from '~/stores/module';
import { useProgressStore } from '~/stores/progress';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { HyperparameterTuning } from '~/algorithms/advanced/HyperparameterTuning';
import LineChart from '~/components/visualizations/LineChart.vue';
import Heatmap from '~/components/visualizations/Heatmap.vue';
import Quiz from '~/components/modules/Quiz.vue';
import QuizLoadingState from '~/components/modules/QuizLoadingState.vue';
import QuizErrorDisplay from '~/components/modules/QuizErrorDisplay.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { LineSeries } from '~/components/visualizations/LineChart.vue';
import type { QuizResult } from '~/types/quiz';
import { useQuizLoader } from '~/composables/useQuizLoader';

// Load content from Nuxt Content
const { data: content } = await useAsyncData(() =>
  queryCollection('content').path('/modules/advanced/hyperparameter-tuning/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const progressStore = useProgressStore();

// Parameters
const parameters = ref({
  searchStrategy: 'random',
  maxIterations: 50,
  crossValidationFolds: 5,
  scoringMetric: 'accuracy',
  earlyStoppingPatience: 10,
  dataset: 'classification',
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'dataset',
    type: 'select',
    label: 'Dataset',
    options: [
      { label: 'Binary Classification', value: 'classification' },
      { label: 'Regression', value: 'regression' },
      { label: 'Multi-class Classification', value: 'multiclass' },
    ],
    tooltip: 'Choose a dataset for hyperparameter tuning',
    group: 'Data',
  },
  {
    id: 'searchStrategy',
    type: 'select',
    label: 'Search Strategy',
    options: [
      { label: 'Grid Search', value: 'grid' },
      { label: 'Random Search', value: 'random' },
      { label: 'Bayesian Optimization', value: 'bayesian' },
    ],
    tooltip: 'Strategy for exploring hyperparameter space',
    group: 'Strategy',
  },
  {
    id: 'maxIterations',
    type: 'slider',
    label: 'Max Iterations',
    min: 10,
    max: 100,
    step: 5,
    tooltip: 'Maximum number of configurations to try',
    group: 'Strategy',
  },
  {
    id: 'crossValidationFolds',
    type: 'slider',
    label: 'CV Folds',
    min: 3,
    max: 10,
    step: 1,
    tooltip: 'Number of cross-validation folds',
    group: 'Evaluation',
  },
  {
    id: 'scoringMetric',
    type: 'select',
    label: 'Scoring Metric',
    options: [
      { label: 'Accuracy', value: 'accuracy' },
      { label: 'F1 Score', value: 'f1' },
      { label: 'MSE', value: 'mse' },
      { label: 'MAE', value: 'mae' },
    ],
    tooltip: 'Metric to optimize during tuning',
    group: 'Evaluation',
  },
  {
    id: 'earlyStoppingPatience',
    type: 'slider',
    label: 'Early Stopping Patience',
    min: 5,
    max: 20,
    step: 1,
    tooltip: 'Stop if no improvement for this many iterations',
    group: 'Strategy',
  },
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const bestParameters = ref<Record<string, any> | null>(null);
const optimizationProgressSeries = ref<LineSeries[]>([]);
const bestScoreSeries = ref<LineSeries[]>([]);
const parameterSpaceViz = ref<any>(null);
const parameterImportanceViz = ref<Record<string, number> | null>(null);

// Quiz data
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

// Algorithm instance
let algorithm: HyperparameterTuning | null = null;

/**
 * Handle quiz completion
 */
const handleQuizComplete = (result: QuizResult) => {
  console.log('Quiz completed with score:', result.score);
};

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('hyperparameter-tuning');
  moduleStore.setParameters(parameters.value);

  // Load default dataset
  await loadInitialDataset();

  // Load quiz data
  await loadQuiz('hyperparameter-tuning', '/content/modules/advanced/hyperparameter-tuning/quiz.json');
});

/**
 * Generate synthetic datasets
 */
const generateSyntheticData = (datasetType: string) => {
  const n = 200;
  const data = [];

  for (let i = 0; i < n; i++) {
    let features: number[];
    let label: number;

    if (datasetType === 'classification') {
      const cluster = Math.random() > 0.5 ? 1 : 0;
      features = [
        cluster * 2 + (Math.random() - 0.5) * 1.5,
        cluster * 2 + (Math.random() - 0.5) * 1.5,
        cluster + (Math.random() - 0.5) * 0.5,
        Math.sin(cluster * Math.PI) + (Math.random() - 0.5) * 0.3,
      ];
      label = cluster;
    } else if (datasetType === 'regression') {
      const x = Math.random() * 10;
      features = [x, x ** 2, Math.sin(x), Math.cos(x)];
      label = 2 * x + 3 + (Math.random() - 0.5) * 2;
    } else {
      // Multiclass
      const cluster = Math.floor(Math.random() * 3);
      features = [
        cluster * 2 + (Math.random() - 0.5),
        cluster * 2 + (Math.random() - 0.5),
        cluster + (Math.random() - 0.5) * 0.3,
        Math.sin(cluster * Math.PI / 2) + (Math.random() - 0.5) * 0.2,
      ];
      label = cluster;
    }

    data.push({
      features,
      label,
      id: `point-${i}`,
    });
  }

  return data;
};

/**
 * Load initial dataset
 */
const loadInitialDataset = async () => {
  const syntheticData = generateSyntheticData(parameters.value.dataset);
  moduleStore.loadDataset(syntheticData);
};

/**
 * Handle parameter change
 */
const handleParameterChange = async (id: string, value: any) => {
  parameters.value[id] = value;
  moduleStore.setParameter(id, value);

  // Reload dataset if changed
  if (id === 'dataset') {
    await loadInitialDataset();
    // Clear previous results
    metrics.value = null;
    bestParameters.value = null;
    optimizationProgressSeries.value = [];
    bestScoreSeries.value = [];
    parameterSpaceViz.value = null;
    parameterImportanceViz.value = null;
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    searchStrategy: 'random',
    maxIterations: 50,
    crossValidationFolds: 5,
    scoringMetric: 'accuracy',
    earlyStoppingPatience: 10,
    dataset: 'classification',
  };
  moduleStore.setParameters(parameters.value);

  await loadInitialDataset();

  metrics.value = null;
  bestParameters.value = null;
  optimizationProgressSeries.value = [];
  bestScoreSeries.value = [];
  parameterSpaceViz.value = null;
  parameterImportanceViz.value = null;
  error.value = null;
};

/**
 * Handle train
 */
const handleTrain = async () => {
  isTraining.value = true;
  trainingProgress.value = 0;
  error.value = null;

  try {
    // Get dataset from store
    const dataset = moduleStore.dataset;
    if (!dataset || dataset.length === 0) {
      throw new Error('No dataset loaded');
    }

    // Prepare data
    const X = dataset.map(d => d.features);
    const y = dataset.map(d => d.label as number);

    // Create and run hyperparameter tuning
    algorithm = new HyperparameterTuning({
      searchStrategy: parameters.value.searchStrategy as 'grid' | 'random' | 'bayesian',
      maxIterations: parameters.value.maxIterations,
      crossValidationFolds: parameters.value.crossValidationFolds,
      scoringMetric: parameters.value.scoringMetric as 'accuracy' | 'f1' | 'mse' | 'mae',
      earlyStoppingPatience: parameters.value.earlyStoppingPatience,
    });

    // Run tuning
    await algorithm.fit(X, y);

    console.log('Hyperparameter tuning completed');

    // Get results
    metrics.value = algorithm.getMetrics();
    bestParameters.value = algorithm.getBestParameters();
    const visualizations = algorithm.getVisualizationData();

    // Process visualizations
    if (Array.isArray(visualizations)) {
      for (const viz of visualizations) {
        if (viz.type === 'line') {
          if (viz.data?.progress) {
            optimizationProgressSeries.value = [{
              x: viz.data.progress.x,
              y: viz.data.progress.y,
              name: 'Score',
              color: '#3b82f6',
              width: 2,
            }];
          }
          if (viz.data?.bestScore) {
            bestScoreSeries.value = [{
              x: viz.data.bestScore.x,
              y: viz.data.bestScore.y,
              name: 'Best Score',
              color: '#10b981',
              width: 2,
            }];
          }
        } else if (viz.type === 'heatmap' && viz.data) {
          parameterSpaceViz.value = viz.data;
        } else if (viz.type === 'bar' && viz.data?.importance) {
          parameterImportanceViz.value = viz.data.importance;
        }
      }
    }

    trainingProgress.value = 1;
  } catch (err) {
    console.error('Hyperparameter tuning error:', err);
    error.value = err instanceof Error ? err.message : 'Tuning failed. Please try again.';
  } finally {
    isTraining.value = false;
  }
};

/**
 * Handle retry
 */
const handleRetry = () => {
  error.value = null;
  handleTrain();
};

/**
 * Handle loading bookmark
 * Requirements: 2.2, 2.3
 */
const handleLoadBookmark = async (bookmarkParameters: Record<string, any>) => {
  try {
    // Update parameters with bookmark values
    Object.assign(parameters.value, bookmarkParameters);
    moduleStore.setParameters(parameters.value);
    
    // Reload dataset if dataset parameter changed
    if (bookmarkParameters.dataset) {
      await loadInitialDataset();
    }
    
    // Clear previous results to indicate fresh state
    metrics.value = null;
    bestParameters.value = null;
    optimizationProgressSeries.value = [];
    bestScoreSeries.value = [];
    parameterSpaceViz.value = null;
    parameterImportanceViz.value = null;
  } catch (err) {
    console.error('Error loading bookmark:', err);
    error.value = 'Failed to load bookmark configuration';
  }
};
</script>
