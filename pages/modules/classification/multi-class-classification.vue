<template>
  <ModuleLayout title="Multi-class Classification"
    description="Learn multi-class classification using One-vs-Rest strategy with logistic regression"
    difficulty="intermediate" :estimated-time="45" :table-of-contents="tableOfContents"
    module-id="multi-class-classification"
    :parameters="parameters"
    @load-bookmark="(val) => { parameters = val as typeof parameters; handleLoadBookmark(val); }">
    <!-- Content Section -->
    <template #content>
      <ContentRenderer v-if="content" :value="content" class="prose prose-slate max-w-none" />
    </template>

    <!-- Interactive Section -->
    <template #interactive>
      <InteractiveSection :loading="isTraining" :progress="trainingProgress" loading-message="Training classifiers..."
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
                <div class="text-xs text-gray-500">Overall Accuracy</div>
                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.accuracy || 0) * 100).toFixed(2) }}%</div>
              </div>
              <div v-if="metrics.precision !== undefined">
                <div class="text-xs text-gray-500">Macro Precision</div>
                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.precision || 0) * 100).toFixed(2) }}%</div>
              </div>
              <div v-if="metrics.recall !== undefined">
                <div class="text-xs text-gray-500">Macro Recall</div>
                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.recall || 0) * 100).toFixed(2) }}%</div>
              </div>
              <div v-if="metrics.f1Score !== undefined">
                <div class="text-xs text-gray-500">Macro F1 Score</div>
                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.f1Score || 0) * 100).toFixed(2) }}%</div>
              </div>
            </div>

            <!-- Per-class metrics -->
            <div v-if="perClassMetrics && perClassMetrics.length > 0" class="mt-4">
              <h4 class="text-xs font-semibold text-gray-700 mb-2">Per-Class Performance</h4>
              <div class="space-y-2">
                <div v-for="classMetric in perClassMetrics" :key="classMetric.class" 
                  class="grid grid-cols-4 gap-2 text-xs">
                  <div class="font-medium text-gray-700">{{ classMetric.class }}</div>
                  <div class="text-gray-600">P: {{ (classMetric.precision * 100).toFixed(1) }}%</div>
                  <div class="text-gray-600">R: {{ (classMetric.recall * 100).toFixed(1) }}%</div>
                  <div class="text-gray-600">F1: {{ (classMetric.f1 * 100).toFixed(1) }}%</div>
                </div>
              </div>
            </div>

            <!-- Confusion Matrix -->
            <div v-if="metrics.confusionMatrix" class="mt-4">
              <h4 class="text-xs font-semibold text-gray-700 mb-2">Confusion Matrix</h4>
              <div class="text-sm text-gray-600">
                {{ metrics.confusionMatrix.length }} classes
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <div class="relative">
              <Scatter2D v-if="scatterSeries && scatterSeries.length > 0" :series="scatterSeries"
                title="Multi-class Data & Decision Boundaries" x-label="Feature 1" y-label="Feature 2"
                :show-legend="true" height="500px" />

              <!-- Overlay heatmap for decision boundary if available -->
              <Heatmap v-if="heatmapData" :data="heatmapData.z" :x-labels="heatmapData.x.map(v => v.toFixed(1))"
                :y-labels="heatmapData.y.map(v => v.toFixed(1))" title="Decision Boundary (Predicted Classes)"
                colorscale="Viridis" :show-annotations="false" :reverse-y="false" height="500px" class="mt-4" />
            </div>

            <Heatmap v-if="confusionMatrixData" :data="confusionMatrixData" 
              :x-labels="confusionMatrixLabels.map(l => `Pred ${l}`)" 
              :y-labels="confusionMatrixLabels.map(l => `Act ${l}`)"
              title="Confusion Matrix" colorscale="Greens" :show-annotations="true" height="400px" />

            <div class="grid grid-cols-1 gap-4">
              <!-- Classifier performance comparison -->
              <div v-if="classifierPerformanceData" class="bg-white p-4 rounded-lg border border-gray-200">
                <h4 class="text-sm font-semibold text-gray-700 mb-2">Per-Classifier Loss</h4>
                <div class="h-64">
                  <canvas ref="classifierChartCanvas"></canvas>
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
import { ref, onMounted, nextTick } from 'vue';
import { useModuleStore } from '~/stores/module';
import { useDataset } from '~/composables/useDataset';
import { useModules } from '~/composables/useModules';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { MultiClassClassification } from '~/algorithms/classification/MultiClassClassification';
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

const { data: content } = await useAsyncData(() => queryCollection('content').path('/modules/classification/multi-class-classification/content').first())

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
  strategy: 'ovr',
  maxIterations: 1000,
  learningRate: 0.01,
  regularization: 0.01,
  tolerance: 1e-6,
  dataset: 'multi-classification',
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
    tooltip: 'Choose a multi-class dataset to train on',
    group: 'Data',
  },
  {
    id: 'strategy',
    type: 'select',
    label: 'Multi-class Strategy',
    options: [
      { label: 'One-vs-Rest (OvR)', value: 'ovr' },
      { label: 'One-vs-One (OvO)', value: 'ovo' }
    ],
    tooltip: 'Strategy for extending binary classifiers to multi-class problems',
    group: 'Model Parameters',
  },
  {
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.001,
    max: 0.1,
    step: 0.001,
    tooltip: 'Step size for gradient descent optimization',
    group: 'Model Parameters',
  },
  {
    id: 'regularization',
    type: 'slider',
    label: 'Regularization Strength',
    min: 0.0,
    max: 0.1,
    step: 0.001,
    tooltip: 'L2 regularization to prevent overfitting. Higher values = more regularization.',
    group: 'Model Parameters',
  },
  {
    id: 'maxIterations',
    type: 'slider',
    label: 'Max Iterations',
    min: 100,
    max: 2000,
    step: 100,
    tooltip: 'Maximum number of iterations for training each binary classifier',
    group: 'Training',
  },
  {
    id: 'tolerance',
    type: 'slider',
    label: 'Convergence Tolerance',
    min: 1e-8,
    max: 1e-3,
    step: 1e-8,
    tooltip: 'Tolerance for convergence. Training stops when loss change is below this value.',
    group: 'Training',
  },
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const perClassMetrics = ref<Array<{ class: string, precision: number, recall: number, f1: number }>>([]);
const scatterSeries = ref<ScatterSeries[]>([]);
const heatmapData = ref<{ x: number[], y: number[], z: number[][] } | null>(null);
const confusionMatrixData = ref<number[][] | null>(null);
const confusionMatrixLabels = ref<string[]>([]);
const classifierPerformanceData = ref<any>(null);

// Chart canvas refs
const classifierChartCanvas = ref<HTMLCanvasElement | null>(null);

// Chart instances
let classifierChart: any = null;

// Algorithm instance
let algorithm: MultiClassClassification | null = null;

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('multi-class-classification');
  moduleStore.setParameters(parameters.value);

  // Load default dataset
  await loadInitialDataset();

  // Load navigation
  previousModule.value = await getPreviousModule('multi-class-classification');
  nextModule.value = await getNextModule('multi-class-classification');

  // For now, suggested module is the same as next module
  suggestedModule.value = nextModule.value;

  // Load quiz data
  await loadQuiz('multi-class-classification', '/content/modules/classification/multi-class-classification/quiz.json');
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
  perClassMetrics.value = [];
  scatterSeries.value = [];
  heatmapData.value = null;
  confusionMatrixData.value = null;
  confusionMatrixLabels.value = [];
  classifierPerformanceData.value = null;
  
  // Destroy existing chart
  if (classifierChart) {
    classifierChart.destroy();
    classifierChart = null;
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    strategy: 'ovr',
    maxIterations: 1000,
    learningRate: 0.01,
    regularization: 0.01,
    tolerance: 1e-6,
    dataset: 'multi-classification',
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
    algorithm = new MultiClassClassification({
      strategy: parameters.value.strategy as 'ovr' | 'ovo',
      maxIterations: parameters.value.maxIterations,
      learningRate: parameters.value.learningRate,
      regularization: parameters.value.regularization,
      tolerance: parameters.value.tolerance
    });

    // Train
    const startTime = Date.now();
    await algorithm.fit(X, y);
    const endTime = Date.now();

    console.log(`Training completed in ${endTime - startTime}ms`);

    // Get results
    metrics.value = algorithm.getMetrics();
    const visualizations = algorithm.getVisualizationData();

    // Extract per-class metrics
    if (metrics.value) {
      const uniqueLabels = [...new Set(y)].sort((a, b) => a - b);
      perClassMetrics.value = uniqueLabels.map(cls => ({
        class: `Class ${cls}`,
        precision: metrics.value![`class_${cls}_precision`] || 0,
        recall: metrics.value![`class_${cls}_recall`] || 0,
        f1: metrics.value![`class_${cls}_f1`] || 0
      }));
    }

    // Set confusion matrix data
    if (metrics.value.confusionMatrix) {
      confusionMatrixData.value = metrics.value.confusionMatrix;
      const uniqueLabels = [...new Set(y)].sort((a, b) => a - b);
      confusionMatrixLabels.value = uniqueLabels.map(label => `${label}`);
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

  // Process line charts for classifier performance
  const lineVizs = visualizations.filter(v => v.type === 'line');
  
  for (const viz of lineVizs) {
    if (viz.data && viz.data.x && viz.data.y) {
      if (viz.data.name && viz.data.name.includes('Classifier Losses')) {
        classifierPerformanceData.value = viz.data;
        nextTick(() => renderClassifierChart());
      }
    }
  }
};

/**
 * Render classifier performance chart using Chart.js
 */
const renderClassifierChart = async () => {
  if (!classifierChartCanvas.value || !classifierPerformanceData.value) return;

  // Dynamically import Chart.js
  const { Chart, registerables } = await import('chart.js');
  Chart.register(...registerables);

  const ctx = classifierChartCanvas.value.getContext('2d');
  if (!ctx) return;

  if (classifierChart) {
    classifierChart.destroy();
  }

  classifierChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: classifierPerformanceData.value.x,
      datasets: [{
        label: 'Loss',
        data: classifierPerformanceData.value.y,
        backgroundColor: classifierPerformanceData.value.marker?.color || '#ef4444',
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
            text: 'Loss'
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
