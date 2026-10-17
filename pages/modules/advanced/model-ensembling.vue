<template>
  <ModuleLayout
    title="Model Ensembling"
    description="Learn how combining multiple models improves prediction accuracy and reduces overfitting"
    difficulty="advanced"
    :estimated-time="50"
    module-id="model-ensembling"
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
        loading-message="Training ensemble..."
        :error="error"
        :on-retry="handleRetry"
      >
        <!-- Controls -->
        <template #controls>
          <ControlPanel
            :controls="controls"
            :values="parameters"
            :loading="isTraining"
            submit-label="Train Ensemble"
            @change="handleParameterChange"
            @reset="handleReset"
            @submit="handleTrain"
          />

          <!-- Ensemble Configuration Display -->
          <div class="mt-6 p-4 bg-emerald-50 rounded-lg border border-emerald-200">
            <h3 class="text-sm font-semibold text-emerald-700 mb-3">Ensemble Configuration</h3>
            <div class="space-y-2">
              <div class="flex items-center gap-2">
                <span class="text-xs text-emerald-600 w-32">Ensemble Type:</span>
                <span class="text-sm font-semibold text-emerald-900 capitalize">
                  {{ parameters.ensembleType }}
                </span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-emerald-600 w-32">Number of Models:</span>
                <span class="text-sm font-semibold text-emerald-900">{{ parameters.numModels }}</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-emerald-600 w-32">Base Model:</span>
                <span class="text-sm font-semibold text-emerald-900 capitalize">
                  {{ parameters.baseModelType }}
                </span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-emerald-600 w-32">Voting Type:</span>
                <span class="text-sm font-semibold text-emerald-900 capitalize">
                  {{ parameters.votingType }}
                </span>
              </div>
            </div>
          </div>

          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Ensemble Results</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Ensemble Accuracy</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ (metrics.ensembleAccuracy * 100).toFixed(2) }}%
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Avg Individual Accuracy</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ (metrics.avgIndividualAccuracy * 100).toFixed(2) }}%
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Improvement</div>
                <div class="text-lg font-semibold text-emerald-600">
                  +{{ (metrics.improvementOverBest * 100).toFixed(2) }}%
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Model Diversity</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.modelDiversity?.toFixed(4) }}
                </div>
              </div>
            </div>
          </div>

          <!-- Model Weights Display -->
          <div v-if="modelWeights && modelWeights.length > 0" class="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
            <h3 class="text-sm font-semibold text-blue-700 mb-3">Model Contributions</h3>
            <div class="space-y-2">
              <div v-for="(weight, idx) in modelWeights" :key="idx" class="flex items-center gap-2">
                <span class="text-xs text-blue-600 w-20">Model {{ idx + 1 }}:</span>
                <div class="flex-1 bg-blue-200 rounded-full h-4">
                  <div 
                    class="bg-blue-500 h-4 rounded-full"
                    :style="{ width: `${weight * 100}%` }"
                  ></div>
                </div>
                <span class="text-xs text-blue-600 w-12 text-right">{{ (weight * 100).toFixed(1) }}%</span>
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <div class="grid grid-cols-2 gap-4">
              <LineChart
                v-if="ensembleErrorSeries && ensembleErrorSeries.length > 0"
                :series="ensembleErrorSeries"
                title="Error Reduction with Ensemble Size"
                x-label="Number of Models"
                y-label="Error Rate"
                :show-legend="true"
                height="300px"
              />

              <LineChart
                v-if="diversitySeries && diversitySeries.length > 0"
                :series="diversitySeries"
                title="Model Diversity"
                x-label="Model Pair"
                y-label="Diversity Score"
                :show-legend="false"
                height="300px"
              />
            </div>

            <div class="grid grid-cols-2 gap-4">
              <ConfusionMatrix
                v-if="ensembleConfusionMatrix"
                :confusion-matrix="ensembleConfusionMatrix"
                :class-labels="['Class 0', 'Class 1']"
                title="Ensemble Confusion Matrix"
              />

              <Heatmap
                v-if="modelCorrelationsViz"
                :data="modelCorrelationsViz"
                title="Model Prediction Correlations"
                x-label="Model"
                y-label="Model"
                height="300px"
              />
            </div>

            <Scatter2D
              v-if="predictionComparisonViz"
              :series="predictionComparisonViz"
              title="Individual vs Ensemble Predictions"
              x-label="Feature 1"
              y-label="Feature 2"
              :show-legend="true"
              height="400px"
            />
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
import { Ensemble } from '~/algorithms/advanced/Ensemble';
import LineChart from '~/components/visualizations/LineChart.vue';
import Scatter2D from '~/components/visualizations/Scatter2D.vue';
import Heatmap from '~/components/visualizations/Heatmap.vue';
import ConfusionMatrix from '~/components/visualizations/ConfusionMatrix.vue';
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
  queryCollection('content').path('/modules/advanced/model-ensembling/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const progressStore = useProgressStore();

// Parameters
const parameters = ref({
  ensembleType: 'bagging',
  numModels: 5,
  baseModelType: 'linear',
  votingType: 'soft',
  learningRate: 0.1,
  maxDepth: 3,
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
      { label: 'Noisy Classification', value: 'noisy-classification' },
      { label: 'Multi-class Classification', value: 'multiclass' },
    ],
    tooltip: 'Choose a dataset for ensemble learning',
    group: 'Data',
  },
  {
    id: 'ensembleType',
    type: 'select',
    label: 'Ensemble Type',
    options: [
      { label: 'Bagging', value: 'bagging' },
      { label: 'Boosting', value: 'boosting' },
      { label: 'Voting', value: 'voting' },
    ],
    tooltip: 'Type of ensemble method',
    group: 'Ensemble',
  },
  {
    id: 'numModels',
    type: 'slider',
    label: 'Number of Models',
    min: 3,
    max: 10,
    step: 1,
    tooltip: 'Number of base models in ensemble',
    group: 'Ensemble',
  },
  {
    id: 'baseModelType',
    type: 'select',
    label: 'Base Model Type',
    options: [
      { label: 'Linear', value: 'linear' },
      { label: 'Tree', value: 'tree' },
      { label: 'Neural', value: 'neural' },
    ],
    tooltip: 'Type of base models',
    group: 'Ensemble',
  },
  {
    id: 'votingType',
    type: 'select',
    label: 'Voting Type',
    options: [
      { label: 'Hard Voting', value: 'hard' },
      { label: 'Soft Voting', value: 'soft' },
    ],
    tooltip: 'Hard (majority) or soft (average probabilities)',
    group: 'Ensemble',
  },
  {
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.01,
    max: 0.5,
    step: 0.01,
    tooltip: 'Learning rate for boosting',
    group: 'Training',
  },
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const modelWeights = ref<number[]>([]);
const ensembleErrorSeries = ref<LineSeries[]>([]);
const diversitySeries = ref<LineSeries[]>([]);
const ensembleConfusionMatrix = ref<number[][] | null>(null);
const modelCorrelationsViz = ref<any>(null);
const predictionComparisonViz = ref<any>(null);

// Quiz data
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

// Algorithm instance
let algorithm: Ensemble | null = null;

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
  moduleStore.setCurrentModule('model-ensembling');
  moduleStore.setParameters(parameters.value);

  // Load default dataset
  await loadInitialDataset();

  // Load quiz data
  await loadQuiz('model-ensembling', '/content/modules/advanced/model-ensembling/quiz.json');
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
      ];
      label = cluster;
    } else if (datasetType === 'noisy-classification') {
      const cluster = Math.random() > 0.5 ? 1 : 0;
      features = [
        cluster * 2 + (Math.random() - 0.5) * 2.5,
        cluster * 2 + (Math.random() - 0.5) * 2.5,
        cluster + (Math.random() - 0.5) * 1.0,
      ];
      // Add label noise
      label = Math.random() < 0.1 ? 1 - cluster : cluster;
    } else {
      // Multiclass
      const cluster = Math.floor(Math.random() * 3);
      features = [
        cluster * 2 + (Math.random() - 0.5),
        cluster * 2 + (Math.random() - 0.5),
        cluster + (Math.random() - 0.5) * 0.3,
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
    modelWeights.value = [];
    ensembleErrorSeries.value = [];
    diversitySeries.value = [];
    ensembleConfusionMatrix.value = null;
    modelCorrelationsViz.value = null;
    predictionComparisonViz.value = null;
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    ensembleType: 'bagging',
    numModels: 5,
    baseModelType: 'linear',
    votingType: 'soft',
    learningRate: 0.1,
    maxDepth: 3,
    dataset: 'classification',
  };
  moduleStore.setParameters(parameters.value);

  await loadInitialDataset();

  metrics.value = null;
  modelWeights.value = [];
  ensembleErrorSeries.value = [];
  diversitySeries.value = [];
  ensembleConfusionMatrix.value = null;
  modelCorrelationsViz.value = null;
  predictionComparisonViz.value = null;
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

    // Create and train ensemble
    algorithm = new Ensemble({
      ensembleType: parameters.value.ensembleType as 'bagging' | 'boosting' | 'voting',
      numModels: parameters.value.numModels,
      baseModelType: parameters.value.baseModelType as 'linear' | 'tree' | 'neural',
      votingType: parameters.value.votingType as 'hard' | 'soft',
      learningRate: parameters.value.learningRate,
      maxDepth: parameters.value.maxDepth,
    });

    // Train
    await algorithm.fit(X, y);

    console.log('Ensemble training completed');

    // Get results
    metrics.value = algorithm.getMetrics();
    modelWeights.value = algorithm.getModelWeights();
    const visualizations = algorithm.getVisualizationData();

    // Process visualizations
    if (Array.isArray(visualizations)) {
      for (const viz of visualizations) {
        if (viz.type === 'line') {
          if (viz.data?.ensembleError) {
            ensembleErrorSeries.value = [{
              x: viz.data.ensembleError.x,
              y: viz.data.ensembleError.y,
              name: 'Ensemble Error',
              color: '#10b981',
              width: 2,
            }];
            if (viz.data.individualError) {
              ensembleErrorSeries.value.push({
                x: viz.data.individualError.x,
                y: viz.data.individualError.y,
                name: 'Avg Individual Error',
                color: '#ef4444',
                width: 2,
              });
            }
          }
          if (viz.data?.diversity) {
            diversitySeries.value = [{
              x: viz.data.diversity.x,
              y: viz.data.diversity.y,
              name: 'Diversity',
              color: '#8b5cf6',
              width: 2,
            }];
          }
        } else if (viz.type === 'confusion' && viz.data) {
          ensembleConfusionMatrix.value = viz.data;
        } else if (viz.type === 'heatmap' && viz.data) {
          // Extract the correlation matrix from the nested structure
          modelCorrelationsViz.value = viz.data.correlations?.z || viz.data;
        } else if (viz.type === 'scatter' && viz.data) {
          predictionComparisonViz.value = viz.data;
        }
      }
    }

    trainingProgress.value = 1;
  } catch (err) {
    console.error('Ensemble training error:', err);
    error.value = err instanceof Error ? err.message : 'Training failed. Please try again.';
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
    modelWeights.value = [];
    ensembleErrorSeries.value = [];
    diversitySeries.value = [];
    ensembleConfusionMatrix.value = null;
    modelCorrelationsViz.value = null;
    predictionComparisonViz.value = null;
  } catch (err) {
    console.error('Error loading bookmark:', err);
    error.value = 'Failed to load bookmark configuration';
  }
};
</script>
