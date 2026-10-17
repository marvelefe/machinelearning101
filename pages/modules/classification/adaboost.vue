<template>
  <ModuleLayout title="AdaBoost"
    description="Learn AdaBoost classification using adaptive sample weighting and weak learners"
    difficulty="intermediate" :estimated-time="45" :table-of-contents="tableOfContents"
    module-id="adaboost"
    :parameters="parameters"
    @load-bookmark="(val) => { parameters = val as typeof parameters; handleLoadBookmark(val); }">
    <!-- Content Section -->
    <template #content>
      <ContentRenderer v-if="content" :value="content" class="prose prose-slate max-w-none" />
    </template>

    <!-- Interactive Section -->
    <template #interactive>
      <InteractiveSection :loading="isTraining" :progress="trainingProgress" loading-message="Training AdaBoost..."
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
              <div v-if="metrics.precision !== undefined">
                <div class="text-xs text-gray-500">Precision</div>
                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.precision || 0) * 100).toFixed(2) }}%</div>
              </div>
              <div v-if="metrics.recall !== undefined">
                <div class="text-xs text-gray-500">Recall</div>
                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.recall || 0) * 100).toFixed(2) }}%</div>
              </div>
              <div v-if="metrics.f1Score !== undefined">
                <div class="text-xs text-gray-500">F1 Score</div>
                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.f1Score || 0) * 100).toFixed(2) }}%</div>
              </div>
            </div>

            <!-- Confusion Matrix -->
            <div v-if="metrics.confusionMatrix" class="mt-4">
              <h4 class="text-xs font-semibold text-gray-700 mb-2">Confusion Matrix</h4>
              <div v-if="metrics.confusionMatrix.length === 2" class="grid grid-cols-2 gap-2 text-center text-sm">
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
                :y-labels="heatmapData.y.map(v => v.toFixed(1))" title="Decision Boundary (Class Predictions)"
                colorscale="RdBu" :show-annotations="false" :reverse-y="false" height="500px" class="mt-4" />
            </div>

            <Heatmap v-if="confusionMatrixData" :data="confusionMatrixData" 
              :x-labels="confusionMatrixLabels" :y-labels="confusionMatrixLabels"
              title="Confusion Matrix" colorscale="Greens" :show-annotations="true" height="350px" />

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <LineChart v-if="lossSeriesData && lossSeriesData.length > 0" :series="lossSeriesData"
                title="Exponential Loss" x-label="Iteration" y-label="Loss" :show-legend="false" height="300px" />

              <LineChart v-if="accuracySeries && accuracySeries.length > 0" :series="accuracySeries"
                title="Training Accuracy" x-label="Iteration" y-label="Accuracy" :show-legend="false" height="300px" />

              <LineChart v-if="weightedErrorSeries && weightedErrorSeries.length > 0" :series="weightedErrorSeries"
                title="Weighted Error per Weak Learner" x-label="Iteration" y-label="Weighted Error" 
                :show-legend="false" height="300px" />
            </div>

            <!-- Bar charts for alpha values and feature importance -->
            <div class="grid grid-cols-1 gap-4">
              <div v-if="alphaBarData" class="bg-white p-4 rounded-lg border border-gray-200">
                <h4 class="text-sm font-semibold text-gray-700 mb-2">Weak Learner Weights (α)</h4>
                <div class="h-64">
                  <canvas ref="alphaChartCanvas"></canvas>
                </div>
              </div>

              <div v-if="featureImportanceData" class="bg-white p-4 rounded-lg border border-gray-200">
                <h4 class="text-sm font-semibold text-gray-700 mb-2">Feature Importance</h4>
                <div class="h-64">
                  <canvas ref="featureChartCanvas"></canvas>
                </div>
              </div>
            </div>
          </div>
        </template>
      </InteractiveSection>
    </template>

    <!-- Quiz Section -->
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
import { ref, onMounted, nextTick, watch } from 'vue';
import { useModuleStore } from '~/stores/module';
import { useDataset } from '~/composables/useDataset';
import { useModules } from '~/composables/useModules';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { AdaBoost } from '~/algorithms/classification/AdaBoost';
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

const { data: content } = await useAsyncData(() => queryCollection('content').path('/modules/classification/adaboost/content').first())

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
  nEstimators: 50,
  learningRate: 1.0,
  algorithm: 'SAMME',
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
    id: 'nEstimators',
    type: 'slider',
    label: 'Number of Estimators',
    min: 10,
    max: 200,
    step: 10,
    tooltip: 'Number of weak learners (decision stumps) in the ensemble. More estimators can improve performance but may cause overfitting.',
    group: 'Ensemble Parameters',
  },
  {
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.1,
    max: 2.0,
    step: 0.1,
    tooltip: 'Shrinks the contribution of each weak learner. Lower values can improve generalization but may require more estimators.',
    group: 'Ensemble Parameters',
  },
  {
    id: 'algorithm',
    type: 'select',
    label: 'Algorithm Variant',
    options: [
      { label: 'SAMME', value: 'SAMME' },
      { label: 'SAMME.R', value: 'SAMME.R' }
    ],
    tooltip: 'SAMME is the discrete AdaBoost algorithm, SAMME.R uses class probabilities for faster convergence.',
    group: 'Algorithm',
  },
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const scatterSeries = ref<ScatterSeries[]>([]);
const lossSeriesData = ref<LineSeries[]>([]);
const accuracySeries = ref<LineSeries[]>([]);
const weightedErrorSeries = ref<LineSeries[]>([]);
const heatmapData = ref<{ x: number[], y: number[], z: number[][] } | null>(null);
const confusionMatrixData = ref<number[][] | null>(null);
const confusionMatrixLabels = ref<string[]>([]);
const alphaBarData = ref<any>(null);
const featureImportanceData = ref<any>(null);

// Chart canvas refs
const alphaChartCanvas = ref<HTMLCanvasElement | null>(null);
const featureChartCanvas = ref<HTMLCanvasElement | null>(null);

// Chart instances
let alphaChart: any = null;
let featureChart: any = null;

// Algorithm instance
let algorithm: AdaBoost | null = null;

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('adaboost');
  moduleStore.setParameters(parameters.value);

  // Load default dataset
  await loadInitialDataset();

  // Load navigation
  previousModule.value = await getPreviousModule('adaboost');
  nextModule.value = await getNextModule('adaboost');

  // For now, suggested module is the same as next module
  suggestedModule.value = nextModule.value;

  // Load quiz data
  await loadQuiz('adaboost', '/content/modules/classification/adaboost/quiz.json');
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
      clearResults();
    }
  }
};

/**
 * Clear previous results
 */
const clearResults = () => {
  metrics.value = null;
  scatterSeries.value = [];
  lossSeriesData.value = [];
  accuracySeries.value = [];
  weightedErrorSeries.value = [];
  heatmapData.value = null;
  confusionMatrixData.value = null;
  confusionMatrixLabels.value = [];
  alphaBarData.value = null;
  featureImportanceData.value = null;
  
  // Destroy existing charts
  if (alphaChart) {
    alphaChart.destroy();
    alphaChart = null;
  }
  if (featureChart) {
    featureChart.destroy();
    featureChart = null;
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    nEstimators: 50,
    learningRate: 1.0,
    algorithm: 'SAMME',
    dataset: 'binary-classification',
  };
  moduleStore.setParameters(parameters.value);

  await loadInitialDataset();
  clearResults();
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
    algorithm = new AdaBoost({
      nEstimators: parameters.value.nEstimators,
      learningRate: parameters.value.learningRate,
      algorithm: parameters.value.algorithm as 'SAMME' | 'SAMME.R'
    });

    // Train
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
      const uniqueLabels = [...new Set(y)].sort((a, b) => a - b);
      confusionMatrixLabels.value = uniqueLabels.map(label => `Class ${label}`);
    }

    // Process visualizations
    processVisualizations(visualizations);

    trainingProgress.value = 1;
  } catch (err) {
    console.error('Training error:', err);
    error.value = err instanceof Error ? err.message : 'Training failed. Please try again.';
  } finally {
    isTraining.value = false;
  }
};

/**
 * Process visualizations from algorithm
 */
const processVisualizations = (visualizations: any[]) => {
  clearResults();

  if (!visualizations || visualizations.length === 0) return;

  // Process scatter plots (data points by class)
  const scatterVizs = visualizations.filter(v => v.type === 'scatter2d');
  if (scatterVizs.length > 0) {
    const series: ScatterSeries[] = [];
    
    for (const viz of scatterVizs) {
      if (viz.data && viz.data.x && viz.data.y) {
        series.push({
          x: viz.data.x,
          y: viz.data.y,
          name: viz.data.name || 'Data',
          color: viz.data.marker?.color || '#3b82f6',
          mode: 'markers',
          marker: {
            size: viz.data.marker?.size || 8,
            symbol: viz.data.marker?.symbol || 'circle',
          }
        });
      }
    }
    
    scatterSeries.value = series;
  }

  // Process heatmaps
  const heatmapVizs = visualizations.filter(v => v.type === 'heatmap');
  for (const viz of heatmapVizs) {
    if (viz.data && viz.data.x && viz.data.y && viz.data.z) {
      if (viz.data.name === 'Decision Boundary') {
        heatmapData.value = {
          x: viz.data.x,
          y: viz.data.y,
          z: viz.data.z
        };
      }
    }
  }

  // Process line charts
  const lineVizs = visualizations.filter(v => v.type === 'line');
  
  for (const viz of lineVizs) {
    if (viz.data && viz.data.x && viz.data.y) {
      const series: LineSeries = {
        x: viz.data.x,
        y: viz.data.y,
        name: viz.data.name || 'Line',
        color: viz.data.line?.color || '#3b82f6',
        width: viz.data.line?.width || 2,
      };

      if (viz.data.name && viz.data.name.includes('Loss')) {
        lossSeriesData.value = [series];
      } else if (viz.data.name && viz.data.name.includes('Accuracy')) {
        accuracySeries.value = [series];
      } else if (viz.data.name && viz.data.name.includes('Weighted Error')) {
        weightedErrorSeries.value = [series];
      }
    }
  }

  // Process bar charts
  const barVizs = visualizations.filter(v => v.type === 'bar');
  
  for (const viz of barVizs) {
    if (viz.data && viz.data.x && viz.data.y) {
      if (viz.data.name && viz.data.name.includes('Alpha')) {
        alphaBarData.value = viz.data;
        nextTick(() => renderAlphaChart());
      } else if (viz.data.name && viz.data.name.includes('Feature Importance')) {
        featureImportanceData.value = viz.data;
        nextTick(() => renderFeatureChart());
      }
    }
  }
};

/**
 * Render alpha chart using Chart.js
 */
const renderAlphaChart = async () => {
  if (!alphaChartCanvas.value || !alphaBarData.value) return;

  // Dynamically import Chart.js
  const { Chart, registerables } = await import('chart.js');
  Chart.register(...registerables);

  const ctx = alphaChartCanvas.value.getContext('2d');
  if (!ctx) return;

  if (alphaChart) {
    alphaChart.destroy();
  }

  alphaChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: alphaBarData.value.x,
      datasets: [{
        label: alphaBarData.value.name,
        data: alphaBarData.value.y,
        backgroundColor: alphaBarData.value.marker?.color || '#8b5cf6',
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Alpha Value'
          }
        }
      }
    }
  });
};

/**
 * Render feature importance chart using Chart.js
 */
const renderFeatureChart = async () => {
  if (!featureChartCanvas.value || !featureImportanceData.value) return;

  // Dynamically import Chart.js
  const { Chart, registerables } = await import('chart.js');
  Chart.register(...registerables);

  const ctx = featureChartCanvas.value.getContext('2d');
  if (!ctx) return;

  if (featureChart) {
    featureChart.destroy();
  }

  featureChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: featureImportanceData.value.x,
      datasets: [{
        label: featureImportanceData.value.name,
        data: featureImportanceData.value.y,
        backgroundColor: featureImportanceData.value.marker?.color || '#06b6d4',
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Importance'
          }
        }
      }
    }
  });
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
 */
const handleQuizComplete = (result: QuizResult) => {
  console.log('Quiz completed:', result);

  if (result.score >= 70) {
    console.log('Quiz passed! Module marked as complete.');
  }
};

// Bookmark restoration
// Requirements: 2.2, 2.3
const { handleLoadBookmark } = useBookmarkRestore(parameters, handleTrain);
</script>
