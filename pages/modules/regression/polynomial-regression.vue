<template>
  <ModuleLayout
    title="Polynomial Regression"
    description="Learn how polynomial regression models non-linear relationships by transforming features"
    difficulty="intermediate"
    :estimated-time="35"
    module-id="polynomial-regression"
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
        loading-message="Training model..."
        :error="error"
        :on-retry="handleRetry"
      >
        <!-- Controls -->
        <template #controls>
          <ControlPanel
            :controls="controls"
            :values="parameters"
            :loading="isTraining"
            submit-label="Train Model"
            @change="handleParameterChange"
            @reset="handleReset"
            @submit="handleTrain"
          />
          
          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Model Performance</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">MSE</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.mse?.toFixed(4) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">RMSE</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.rmse?.toFixed(4) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">R² Score</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.r2Score?.toFixed(4) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">MAE</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.mae?.toFixed(4) }}</div>
              </div>
            </div>
            <div v-if="metrics.degree" class="mt-3 pt-3 border-t border-gray-200">
              <div class="text-xs text-gray-500">Polynomial Degree</div>
              <div class="text-lg font-semibold text-gray-900">{{ metrics.degree }}</div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <Scatter2D
              v-if="scatterSeries && scatterSeries.length > 0"
              :series="scatterSeries"
              title="Data & Polynomial Curve"
              x-label="X"
              y-label="Y"
              :show-legend="true"
              height="500px"
            />
            
            <Scatter2D
              v-if="residualSeries && residualSeries.length > 0"
              :series="residualSeries"
              title="Residuals"
              x-label="X"
              y-label="Residual"
              :show-legend="true"
              height="350px"
            />
            
            <LineChart
              v-if="lossSeries && lossSeries.length > 0"
              :series="lossSeries"
              title="Training Loss"
              x-label="Epoch"
              y-label="Loss"
              :show-legend="false"
              height="350px"
            />
          </div>
        </template>
      </InteractiveSection>
    </template>

    <!-- Quiz Section -->
    <template #quiz>
      <QuizLoadingState v-if="quizLoading" />
      <QuizErrorDisplay
        v-else-if="quizError"
        :error="quizError"
        :on-retry="retryQuizLoad"
      />
      <QuizErrorDisplay
        v-else-if="!quizData"
        :coming-soon="true"
      />
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
import { ref, onMounted } from 'vue';
import { useModuleStore } from '~/stores/module';
import { useAnalyticsStore } from '~/stores/analytics';
import { useDataset } from '~/composables/useDataset';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { PolynomialRegression } from '~/algorithms/regression/PolynomialRegression';
import Scatter2D from '~/components/visualizations/Scatter2D.vue';
import LineChart from '~/components/visualizations/LineChart.vue';
import Quiz from '~/components/modules/Quiz.vue';
import QuizErrorDisplay from '~/components/modules/QuizErrorDisplay.vue';
import QuizLoadingState from '~/components/modules/QuizLoadingState.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { ScatterSeries } from '~/components/visualizations/Scatter2D.vue';
import type { LineSeries } from '~/components/visualizations/LineChart.vue';
import type { QuizResult } from '~/types/quiz';
import { useQuizLoader } from '~/composables/useQuizLoader';
import { useBookmarkRestore } from '~/composables/useBookmarkRestore';

// Load content from Nuxt Content
const { data: content } = await useAsyncData(() => 
  queryCollection('content').path('/modules/regression/polynomial-regression/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const analyticsStore = useAnalyticsStore();
const { loadDataset, getDatasetsByType } = useDataset();

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

// Parameters
const parameters = ref({
  degree: 2,
  learningRate: 0.01,
  epochs: 100,
  dataset: 'polynomial',
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'dataset',
    type: 'select',
    label: 'Dataset',
    options: getDatasetsByType('regression').map(d => ({
      label: d.name,
      value: d.id
    })),
    tooltip: 'Choose a dataset to train on',
    group: 'Data',
  },
  {
    id: 'degree',
    type: 'slider',
    label: 'Polynomial Degree',
    min: 1,
    max: 10,
    step: 1,
    tooltip: 'Degree of the polynomial. Higher degrees can fit more complex curves but may overfit.',
    group: 'Model Parameters',
  },
  {
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.001,
    max: 0.1,
    step: 0.001,
    tooltip: 'Controls how quickly the model learns.',
    group: 'Training Parameters',
  },
  {
    id: 'epochs',
    type: 'slider',
    label: 'Epochs',
    min: 10,
    max: 500,
    step: 10,
    tooltip: 'Number of training iterations.',
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
const residualSeries = ref<ScatterSeries[]>([]);
const lossSeries = ref<LineSeries[]>([]);

// Algorithm instance
let algorithm: PolynomialRegression | null = null;

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('polynomial-regression');
  moduleStore.setParameters(parameters.value);
  
  // Track module visit
  analyticsStore.trackModuleVisit('polynomial-regression', 'Polynomial Regression');
  
  // Load default dataset
  await loadInitialDataset();

  // Load quiz data
  await loadQuiz('polynomial-regression', '/content/modules/regression/polynomial-regression/quiz.json');
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
  
  // Track parameter change
  analyticsStore.trackParameterChange('polynomial-regression', id, value);
  
  // Reload dataset if changed
  if (id === 'dataset') {
    const dataset = await loadDataset(value);
    if (dataset) {
      moduleStore.loadDataset(dataset.data);
      // Clear previous results
      metrics.value = null;
      scatterSeries.value = [];
      residualSeries.value = [];
      lossSeries.value = [];
    }
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    degree: 2,
    learningRate: 0.01,
    epochs: 100,
    dataset: 'polynomial',
  };
  moduleStore.setParameters(parameters.value);
  
  await loadInitialDataset();
  
  metrics.value = null;
  scatterSeries.value = [];
  residualSeries.value = [];
  lossSeries.value = [];
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
    algorithm = new PolynomialRegression({
      degree: parameters.value.degree,
      learningRate: parameters.value.learningRate,
      epochs: parameters.value.epochs,
      normalize: true
    });
    
    // Train with progress updates
    const startTime = Date.now();
    await algorithm.fit(X, y);
    const endTime = Date.now();
    const duration = endTime - startTime;
    
    console.log(`Training completed in ${duration}ms`);
    
    // Track algorithm run
    analyticsStore.trackAlgorithmRun('polynomial-regression', parameters.value, duration);
    
    // Get results
    metrics.value = algorithm.getMetrics();
    const visualizations = algorithm.getVisualizationData();
    
    // Process visualizations
    if (visualizations.length > 0) {
      // Scatter plot with polynomial curve
      const scatterViz = visualizations[0];
      if (scatterViz.type === 'scatter2d' && scatterViz.data) {
        const series: ScatterSeries[] = [];
        
        // Add data points
        if (scatterViz.data.scatter) {
          series.push({
            x: scatterViz.data.scatter.x,
            y: scatterViz.data.scatter.y,
            name: scatterViz.data.scatter.name || 'Data Points',
            color: scatterViz.data.scatter.marker?.color || '#3b82f6',
            mode: 'markers',
            marker: {
              size: scatterViz.data.scatter.marker?.size || 8,
            }
          });
        }
        
        // Add polynomial curve
        if (scatterViz.data.curve) {
          series.push({
            x: scatterViz.data.curve.x,
            y: scatterViz.data.curve.y,
            name: scatterViz.data.curve.name || 'Polynomial Curve',
            color: scatterViz.data.curve.line?.color || '#ef4444',
            mode: 'lines',
          });
        }
        
        scatterSeries.value = series;
      }
      
      // Residuals plot
      if (visualizations.length > 1) {
        const residualViz = visualizations[1];
        if (residualViz.type === 'scatter2d' && residualViz.data) {
          const series: ScatterSeries[] = [];
          
          if (residualViz.data.residuals) {
            series.push({
              x: residualViz.data.residuals.x,
              y: residualViz.data.residuals.y,
              name: residualViz.data.residuals.name || 'Residuals',
              color: residualViz.data.residuals.marker?.color || '#8b5cf6',
              mode: 'markers',
              marker: {
                size: residualViz.data.residuals.marker?.size || 6,
              }
            });
          }
          
          if (residualViz.data.zero) {
            series.push({
              x: residualViz.data.zero.x,
              y: residualViz.data.zero.y,
              name: residualViz.data.zero.name || 'Zero Line',
              color: residualViz.data.zero.line?.color || '#6b7280',
              mode: 'lines',
              line: {
                dash: 'dash',
              }
            });
          }
          
          residualSeries.value = series;
        }
      }
      
      // Loss curve
      if (visualizations.length > 2) {
        const lossViz = visualizations[2];
        if (lossViz.type === 'line' && lossViz.data) {
          lossSeries.value = [{
            x: lossViz.data.x,
            y: lossViz.data.y,
            name: lossViz.data.name || 'Training Loss',
            color: lossViz.data.line?.color || '#10b981',
            width: lossViz.data.line?.width || 2,
          }];
        }
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
 */
const handleQuizComplete = (result: QuizResult) => {
  console.log('Quiz completed:', result);
};

// Bookmark restoration
// Requirements: 2.2, 2.3
const { handleLoadBookmark } = useBookmarkRestore(parameters, handleTrain);

// SEO
useSEO({
  title: 'Polynomial Regression - Interactive ML Tutorial',
  description: 'Learn polynomial regression through interactive visualization. Model non-linear relationships by transforming features into polynomial terms. Understand overfitting and underfitting.',
  keywords: 'polynomial regression, non-linear regression, feature engineering, overfitting, underfitting, curve fitting',
  url: '/modules/regression/polynomial-regression',
  type: 'article',
  section: 'Regression',
})

// Course structured data
useCourseStructuredData({
  name: 'Polynomial Regression',
  description: 'Learn how polynomial regression models non-linear relationships by transforming features',
  provider: 'Machine Learning 101',
  url: '/modules/regression/polynomial-regression',
  difficulty: 'intermediate',
  timeRequired: 'PT35M',
  keywords: ['polynomial regression', 'non-linear regression', 'machine learning', 'feature engineering'],
})

// Breadcrumb structured data
useBreadcrumbStructuredData([
  { name: 'Home', url: '/' },
  { name: 'Learning Modules', url: '/modules' },
  { name: 'Regression', url: '/modules/regression' },
  { name: 'Polynomial Regression', url: '/modules/regression/polynomial-regression' },
])
</script>
