<template>
  <ModuleLayout title="Decision Trees"
    description="Learn how decision trees make predictions using if-then-else rules and splitting criteria"
    difficulty="intermediate" :estimated-time="30" :table-of-contents="tableOfContents"
    module-id="decision-trees"
    :parameters="parameters"
    @load-bookmark="(val) => { parameters = val as typeof parameters; handleLoadBookmark(val); }">
    <!-- Content Section -->
    <template #content>
      <ContentRenderer v-if="content" :value="content" class="prose prose-slate max-w-none" />
    </template>

    <!-- Interactive Section -->
    <template #interactive>
      <InteractiveSection :loading="isTraining" :progress="trainingProgress" loading-message="Training model..."
        :error="error" :on-retry="handleRetry">
        <!-- Controls -->
        <template #controls>
          <ControlPanel :controls="controls" :values="parameters" :loading="isTraining" submit-label="Train Model"
            @change="handleParameterChange" @reset="handleReset" @submit="handleTrain" />

          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Model Performance</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Accuracy</div>
                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.accuracy || 0) * 100).toFixed(2) }}%</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Precision</div>
                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.precision || 0) * 100).toFixed(2) }}%
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Recall</div>
                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.recall || 0) * 100).toFixed(2) }}%</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">F1 Score</div>
                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.f1Score || 0) * 100).toFixed(2) }}%</div>
              </div>
            </div>

            <!-- Confusion Matrix -->
            <div v-if="metrics.confusionMatrix" class="mt-4">
              <h4 class="text-xs font-semibold text-gray-700 mb-2">Confusion Matrix</h4>
              <div class="grid grid-cols-2 gap-2 text-center text-sm">
                <div class="bg-blue-100 p-2 rounded">
                  <div class="text-xs text-gray-600">True Neg</div>
                  <div class="font-semibold">{{ metrics.confusionMatrix[0][0] }}</div>
                </div>
                <div class="bg-red-100 p-2 rounded">
                  <div class="text-xs text-gray-600">False Pos</div>
                  <div class="font-semibold">{{ metrics.confusionMatrix[0][1] }}</div>
                </div>
                <div class="bg-red-100 p-2 rounded">
                  <div class="text-xs text-gray-600">False Neg</div>
                  <div class="font-semibold">{{ metrics.confusionMatrix[1][0] }}</div>
                </div>
                <div class="bg-green-100 p-2 rounded">
                  <div class="text-xs text-gray-600">True Pos</div>
                  <div class="font-semibold">{{ metrics.confusionMatrix[1][1] }}</div>
                </div>
              </div>
            </div>

            <!-- Feature Importance -->
            <div v-if="featureImportances && featureImportances.length > 0" class="mt-4">
              <h4 class="text-xs font-semibold text-gray-700 mb-2">Feature Importance</h4>
              <div class="space-y-2">
                <div v-for="(importance, idx) in featureImportances" :key="idx" class="flex items-center gap-2">
                  <span class="text-xs text-gray-600 w-20">Feature {{ idx + 1 }}</span>
                  <div class="flex-1 bg-gray-200 rounded-full h-2">
                    <div class="bg-blue-500 h-2 rounded-full" :style="{ width: `${importance * 100}%` }"></div>
                  </div>
                  <span class="text-xs text-gray-600 w-12 text-right">{{ (importance * 100).toFixed(1) }}%</span>
                </div>
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <div class="relative">
              <Scatter2D v-if="scatterSeries && scatterSeries.length > 0" :series="scatterSeries"
                title="Classification Data & Decision Boundary" x-label="Feature 1" y-label="Feature 2"
                :show-legend="true" height="500px" />

              <!-- Overlay heatmap for decision boundary if available -->
              <Heatmap v-if="heatmapData" :data="heatmapData.z" :x-labels="heatmapData.x.map(v => v.toFixed(1))"
                :y-labels="heatmapData.y.map(v => v.toFixed(1))" title="Decision Boundary (Decision Regions)"
                colorscale="RdBu" :show-annotations="false" :reverse-y="false" height="500px" class="mt-4" />
            </div>

            <Heatmap v-if="confusionMatrixData" :data="confusionMatrixData" :x-labels="['Predicted 0', 'Predicted 1']"
              :y-labels="['Actual 0', 'Actual 1']" title="Confusion Matrix" colorscale="Greens" :show-annotations="true"
              height="350px" />
          </div>
        </template>
      </InteractiveSection>
    </template>

    <!-- Quiz Section -->
    <!-- Requirements: 1.1, 3.1, 3.2, 3.5 -->
    <template #quiz>
      <!-- Loading state -->
      <QuizLoadingState v-if="quizLoading" />

      <!-- Error state with retry -->
      <QuizErrorDisplay v-else-if="quizError" :error="quizError" @retry="retryQuizLoad" />

      <!-- Coming soon state -->
      <QuizErrorDisplay v-else-if="!quizData" :coming-soon="true" />

      <!-- Quiz component -->
      <Quiz v-else :quiz="quizData" :passing-score="quizData.passingScore || 70"
        @quiz-completed="handleQuizComplete" />
    </template>
  </ModuleLayout>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { useModuleStore } from '~/stores/module';
import { useDataset } from '~/composables/useDataset';
import { useModules } from '~/composables/useModules';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { DecisionTree } from '~/algorithms/classification/DecisionTree';
import Scatter2D from '~/components/visualizations/Scatter2D.vue';
import Heatmap from '~/components/visualizations/Heatmap.vue';
import Quiz from '~/components/modules/Quiz.vue';
import QuizErrorDisplay from '~/components/modules/QuizErrorDisplay.vue';
import QuizLoadingState from '~/components/modules/QuizLoadingState.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { ScatterSeries } from '~/components/visualizations/Scatter2D.vue';
import type { ModuleNavItem } from '~/types/module';
import type { QuizResult } from '~/types/quiz';
import { useQuizLoader } from '~/composables/useQuizLoader';
import { useBookmarkRestore } from '~/composables/useBookmarkRestore';

import { useAsyncData } from '#app';


const { data: content } = await useAsyncData(() => queryCollection('content').path('/modules/classification/decision-trees/content').first())

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const { loadDataset, getDatasetsByType } = useDataset();
const { getNextModule, getPreviousModule } = useModules();

// Navigation
const previousModule = ref<ModuleNavItem | null>(null);
const nextModule = ref<ModuleNavItem | null>(null);
const suggestedModule = ref<ModuleNavItem | null>(null);

// Parameters
const parameters = ref({
  maxDepth: 5,
  minSamplesSplit: 2,
  criterion: 'gini',
  dataset: 'binary-classification',
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'dataset',
    type: 'select',
    label: 'Dataset',
    options: getDatasetsByType('classification').map(d => ({
      label: d.name,
      value: d.id
    })),
    tooltip: 'Choose a classification dataset to train on',
    group: 'Data',
  },
  {
    id: 'maxDepth',
    type: 'slider',
    label: 'Max Tree Depth',
    min: 1,
    max: 15,
    step: 1,
    tooltip: 'Maximum depth of the tree. Deeper trees can model more complex patterns but may overfit.',
    group: 'Model Parameters',
  },
  {
    id: 'minSamplesSplit',
    type: 'slider',
    label: 'Min Samples to Split',
    min: 2,
    max: 20,
    step: 1,
    tooltip: 'Minimum number of samples required to split a node. Higher values prevent overfitting.',
    group: 'Model Parameters',
  },
  {
    id: 'criterion',
    type: 'select',
    label: 'Split Criterion',
    options: [
      { label: 'Gini Impurity', value: 'gini' },
      { label: 'Entropy (Information Gain)', value: 'entropy' }
    ],
    tooltip: 'Method to measure split quality',
    group: 'Model Parameters',
  },
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const scatterSeries = ref<ScatterSeries[]>([]);
const heatmapData = ref<{ x: number[], y: number[], z: number[][] } | null>(null);
const confusionMatrixData = ref<number[][] | null>(null);
const featureImportances = ref<number[]>([]);

// Algorithm instance
let algorithm: DecisionTree | null = null;

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('decision-trees');
  moduleStore.setParameters(parameters.value);

  // Load default dataset
  await loadInitialDataset();

  // Load navigation
  previousModule.value = await getPreviousModule('decision-trees');
  nextModule.value = await getNextModule('decision-trees');

  // For now, suggested module is the same as next module
  suggestedModule.value = nextModule.value;

  // Load quiz data
  // Requirements: 1.1, 2.1, 2.4, 6.1, 6.2
  await loadQuiz('decision-trees', '/content/modules/classification/decision-trees/quiz.json');
});

/**
 * Load initial dataset
 */
const loadInitialDataset = async () => {
  const dataset = await loadDataset(parameters.value.dataset);
  if (dataset) {
    moduleStore.loadDataset(dataset.data);
  }
};

/**
 * Handle parameter change
 */
const handleParameterChange = async (id: string, value: any) => {
  parameters.value[id] = value;
  moduleStore.setParameter(id, value);

  // Reload dataset if changed
  if (id === 'dataset') {
    const dataset = await loadDataset(value);
    if (dataset) {
      moduleStore.loadDataset(dataset.data);
      // Clear previous results
      metrics.value = null;
      scatterSeries.value = [];
      heatmapData.value = null;
      confusionMatrixData.value = null;
      featureImportances.value = [];
    }
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    maxDepth: 5,
    minSamplesSplit: 2,
    criterion: 'gini',
    dataset: 'binary-classification',
  };
  moduleStore.setParameters(parameters.value);

  await loadInitialDataset();

  metrics.value = null;
  scatterSeries.value = [];
  heatmapData.value = null;
  confusionMatrixData.value = null;
  featureImportances.value = [];
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

    // Create and train algorithm
    algorithm = new DecisionTree({
      maxDepth: parameters.value.maxDepth,
      minSamplesSplit: parameters.value.minSamplesSplit,
      criterion: parameters.value.criterion as 'gini' | 'entropy'
    });

    // Train
    const startTime = Date.now();
    await algorithm.fit(X, y);
    const endTime = Date.now();

    console.log(`Training completed in ${endTime - startTime}ms`);

    // Get results
    metrics.value = algorithm.getMetrics();
    const visualizations = algorithm.getVisualizationData();

    // Get feature importances
    const params = algorithm.getParameters();
    featureImportances.value = params.featureImportances || [];

    // Set confusion matrix data
    if (metrics.value.confusionMatrix) {
      confusionMatrixData.value = metrics.value.confusionMatrix;
    }

    // Process visualizations
    if (visualizations.length > 0) {
      // Scatter plot with classes
      const scatterViz = visualizations[0];
      if (scatterViz.type === 'scatter2d' && scatterViz.data) {
        const series: ScatterSeries[] = [];

        // Add all class series
        for (const [key, value] of Object.entries(scatterViz.data)) {
          if (key.startsWith('class')) {
            const classData = value as any;
            series.push({
              x: classData.x,
              y: classData.y,
              name: classData.name || key,
              color: classData.marker?.color || '#3b82f6',
              mode: 'markers',
              marker: {
                size: classData.marker?.size || 8,
                symbol: classData.marker?.symbol || 'circle',
              }
            });
          }
        }

        scatterSeries.value = series;
      }

      // Heatmap for decision boundary
      const heatmapViz = visualizations.find(v => v.type === 'heatmap');
      if (heatmapViz && heatmapViz.data) {
        heatmapData.value = {
          x: heatmapViz.data.x,
          y: heatmapViz.data.y,
          z: heatmapViz.data.z
        };
      }
    }

    trainingProgress.value = 1;
  } catch (err) {
    console.error('Training error:', err);
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
 * Handle quiz completion
 * Processes quiz results and verifies progress store integration
 * 
 * Requirements: 3.3, 3.4, 5.1, 5.2, 5.4
 */
const handleQuizComplete = (result: QuizResult) => {
  console.log('Quiz completed:', result);

  // The Quiz component already handles:
  // - Saving score to progress store (Requirement 5.1, 5.2)
  // - Marking module as complete if score >= 70% (Requirement 3.4, 5.4)
  // - Tracking analytics (Requirement 3.3)

  // Optional: Show success message or navigate to next module
  if (result.score >= 70) {
    console.log('Quiz passed! Module marked as complete.');
  }
};

// Bookmark restoration
// Requirements: 2.2, 2.3
const { handleLoadBookmark } = useBookmarkRestore(parameters, handleTrain);
</script>
