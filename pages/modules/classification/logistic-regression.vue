<template>
  <ModuleLayout title="Logistic Regression"
    description="Learn how logistic regression classifies data using the sigmoid function and gradient descent"
    difficulty="beginner" :estimated-time="35" :table-of-contents="tableOfContents"
    module-id="logistic-regression"
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
                :y-labels="heatmapData.y.map(v => v.toFixed(1))" title="Decision Boundary (Probability Heatmap)"
                colorscale="RdBu" :show-annotations="false" :reverse-y="false" height="500px" class="mt-4" />
            </div>

            <Heatmap v-if="confusionMatrixData" :data="confusionMatrixData" :x-labels="['Predicted 0', 'Predicted 1']"
              :y-labels="['Actual 0', 'Actual 1']" title="Confusion Matrix" colorscale="Greens" :show-annotations="true"
              height="350px" />

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <LineChart v-if="lossSeries && lossSeries.length > 0" :series="lossSeries" title="Training Loss"
                x-label="Epoch" y-label="Loss" :show-legend="false" height="300px" />

              <LineChart v-if="accuracySeries && accuracySeries.length > 0" :series="accuracySeries"
                title="Training Accuracy" x-label="Epoch" y-label="Accuracy" :show-legend="false" height="300px" />
            </div>
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
import { LogisticRegression } from '~/algorithms/classification/LogisticRegression';
import Scatter2D from '~/components/visualizations/Scatter2D.vue';
import LineChart from '~/components/visualizations/LineChart.vue';
import Heatmap from '~/components/visualizations/Heatmap.vue';
import Quiz from '~/components/modules/Quiz.vue';
import QuizErrorDisplay from '~/components/modules/QuizErrorDisplay.vue';
import QuizLoadingState from '~/components/modules/QuizLoadingState.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { ScatterSeries } from '~/components/visualizations/Scatter2D.vue';
import type { LineSeries } from '~/components/visualizations/LineChart.vue';
import type { ModuleNavItem } from '~/types/module';
import type { Quiz as QuizType, QuizResult } from '~/types/quiz';
import { useQuizLoader } from '~/composables/useQuizLoader';
import { useBookmarkRestore } from '~/composables/useBookmarkRestore';

import { useAsyncData } from '#app';


const { data: content } = await useAsyncData(() => queryCollection('content').path('/modules/classification/logistic-regression/content').first())

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
  learningRate: 0.1,
  epochs: 100,
  regularization: 0,
  dataset: 'binary-classification',
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'dataset',
    type: 'select',
    label: 'Dataset',
    options: getDatasetsByType('classification')
      .filter(d => d.id === 'binary-classification') // Only binary classification for logistic regression
      .map(d => ({
        label: d.name,
        value: d.id
      })),
    tooltip: 'Choose a binary classification dataset to train on',
    group: 'Data',
  },
  {
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.001,
    max: 1.0,
    step: 0.001,
    tooltip: 'Controls how quickly the model learns. Higher values converge faster but may be unstable.',
    group: 'Training Parameters',
  },
  {
    id: 'epochs',
    type: 'slider',
    label: 'Epochs',
    min: 10,
    max: 500,
    step: 10,
    tooltip: 'Number of training iterations. More epochs allow better learning but take longer.',
    group: 'Training Parameters',
  },
  {
    id: 'regularization',
    type: 'slider',
    label: 'Regularization (L2)',
    min: 0,
    max: 1,
    step: 0.01,
    tooltip: 'Prevents overfitting by penalizing large weights. Higher values create simpler models.',
    group: 'Training Parameters',
  },
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const scatterSeries = ref<ScatterSeries[]>([]);
const lossSeries = ref<LineSeries[]>([]);
const accuracySeries = ref<LineSeries[]>([]);
const heatmapData = ref<{ x: number[], y: number[], z: number[][] } | null>(null);
const confusionMatrixData = ref<number[][] | null>(null);

// Algorithm instance
let algorithm: LogisticRegression | null = null;

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('logistic-regression');
  moduleStore.setParameters(parameters.value);

  // Load default dataset
  await loadInitialDataset();

  // Load navigation
  previousModule.value = await getPreviousModule('logistic-regression');
  nextModule.value = await getNextModule('logistic-regression');

  // For now, suggested module is the same as next module
  suggestedModule.value = nextModule.value;

  // Load quiz data
  // Requirements: 1.1, 2.1, 2.4, 6.1, 6.2
  await loadQuiz('logistic-regression', '/content/modules/classification/logistic-regression/quiz.json');
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
      lossSeries.value = [];
      accuracySeries.value = [];
      heatmapData.value = null;
      confusionMatrixData.value = null;
    }
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    learningRate: 0.1,
    epochs: 100,
    regularization: 0,
    dataset: 'binary-classification',
  };
  moduleStore.setParameters(parameters.value);

  await loadInitialDataset();

  metrics.value = null;
  scatterSeries.value = [];
  lossSeries.value = [];
  accuracySeries.value = [];
  heatmapData.value = null;
  confusionMatrixData.value = null;
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

    // Validate binary labels
    const uniqueLabels = [...new Set(y)];
    if (uniqueLabels.length !== 2 || !uniqueLabels.includes(0) || !uniqueLabels.includes(1)) {
      throw new Error('Dataset must have exactly 2 classes labeled as 0 and 1');
    }

    // Create and train algorithm
    algorithm = new LogisticRegression({
      learningRate: parameters.value.learningRate,
      epochs: parameters.value.epochs,
      regularization: parameters.value.regularization,
      normalize: false
    });

    // Train with progress updates
    const startTime = Date.now();
    await algorithm.fit(X, y);
    const endTime = Date.now();

    console.log(`Training completed in ${endTime - startTime}ms`);

    // Get results
    metrics.value = algorithm.getMetrics();
    const visualizations = algorithm.getVisualizationData();

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

        // Add class 0 points
        if (scatterViz.data.class0) {
          series.push({
            x: scatterViz.data.class0.x,
            y: scatterViz.data.class0.y,
            name: scatterViz.data.class0.name || 'Class 0',
            color: scatterViz.data.class0.marker?.color || '#3b82f6',
            mode: 'markers',
            marker: {
              size: scatterViz.data.class0.marker?.size || 8,
              symbol: scatterViz.data.class0.marker?.symbol || 'circle',
            }
          });
        }

        // Add class 1 points
        if (scatterViz.data.class1) {
          series.push({
            x: scatterViz.data.class1.x,
            y: scatterViz.data.class1.y,
            name: scatterViz.data.class1.name || 'Class 1',
            color: scatterViz.data.class1.marker?.color || '#ef4444',
            mode: 'markers',
            marker: {
              size: scatterViz.data.class1.marker?.size || 8,
              symbol: scatterViz.data.class1.marker?.symbol || 'x',
            }
          });
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

      // Loss curve - find all line visualizations
      const lineVisualizations = visualizations.filter(v => v.type === 'line');
      
      const lossViz = lineVisualizations.find(v => v.data?.name === 'Training Loss');
      if (lossViz && lossViz.data && Array.isArray(lossViz.data.x) && Array.isArray(lossViz.data.y) && lossViz.data.x.length > 0) {
        lossSeries.value = [{
          x: lossViz.data.x,
          y: lossViz.data.y,
          name: lossViz.data.name || 'Training Loss',
          color: lossViz.data.line?.color || '#10b981',
          width: lossViz.data.line?.width || 2,
        }];
      } else {
        lossSeries.value = [];
      }

      // Accuracy curve
      const accuracyViz = lineVisualizations.find(v => v.data?.name === 'Training Accuracy');
      if (accuracyViz && accuracyViz.data && Array.isArray(accuracyViz.data.x) && Array.isArray(accuracyViz.data.y) && accuracyViz.data.x.length > 0) {
        accuracySeries.value = [{
          x: accuracyViz.data.x,
          y: accuracyViz.data.y,
          name: accuracyViz.data.name || 'Training Accuracy',
          color: accuracyViz.data.line?.color || '#8b5cf6',
          width: accuracyViz.data.line?.width || 2,
        }];
      } else {
        accuracySeries.value = [];
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
