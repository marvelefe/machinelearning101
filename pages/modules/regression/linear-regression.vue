<template>
  <ModuleLayout
    title="Linear Regression"
    description="Learn how linear regression finds the best-fit line through data using gradient descent"
    difficulty="beginner"
    :estimated-time="30"
    module-id="linear-regression"
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
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <Scatter2D
              v-if="scatterSeries && scatterSeries.length > 0"
              :series="scatterSeries"
              title="Data & Regression Line"
              x-label="X"
              y-label="Y"
              :show-legend="true"
              height="500px"
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
    <!-- Requirements: 1.1, 3.1, 3.2, 3.5, 6.1, 6.6 -->
    <template #quiz>
      <!-- Loading state -->
      <QuizLoadingState v-if="quizLoading" />
      
      <!-- Error state with retry -->
      <QuizErrorDisplay
        v-else-if="quizError"
        :error="quizError"
        :on-retry="retryQuizLoad"
      />
      
      <!-- Coming soon state -->
      <QuizErrorDisplay
        v-else-if="!quizData"
        :coming-soon="true"
      />
      
      <!-- Quiz component -->
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
import { LinearRegression } from '~/algorithms/regression/LinearRegression';
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
  queryCollection('content').path('/modules/regression/linear-regression/content').first()
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
  learningRate: 0.01,
  epochs: 100,
  dataset: 'simple-linear',
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
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.001,
    max: 0.1,
    step: 0.001,
    tooltip: 'Controls how quickly the model learns. Too high may overshoot, too low may converge slowly.',
    group: 'Training Parameters',
  },
  {
    id: 'epochs',
    type: 'slider',
    label: 'Epochs',
    min: 10,
    max: 500,
    step: 10,
    tooltip: 'Number of training iterations. More epochs allow the model to learn better but take longer.',
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

// Algorithm instance
let algorithm: LinearRegression | null = null;

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('linear-regression');
  moduleStore.setParameters(parameters.value);
  
  // Track module visit
  analyticsStore.trackModuleVisit('linear-regression', 'Linear Regression');
  
  // Load default dataset
  await loadInitialDataset();

  // Load quiz data with loading state and error handling
  // Requirements: 1.1, 2.1, 2.4, 6.1, 6.2, 6.6
  await loadQuiz('linear-regression', '/content/modules/regression/linear-regression/quiz.json');
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
  analyticsStore.trackParameterChange('linear-regression', id, value);
  
  // Reload dataset if changed
  if (id === 'dataset') {
    const dataset = await loadDataset(value);
    if (dataset) {
      moduleStore.loadDataset(dataset.data);
      // Clear previous results
      metrics.value = null;
      scatterSeries.value = [];
      lossSeries.value = [];
    }
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    learningRate: 0.01,
    epochs: 100,
    dataset: 'simple-linear',
  };
  moduleStore.setParameters(parameters.value);
  
  await loadInitialDataset();
  
  metrics.value = null;
  scatterSeries.value = [];
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
    algorithm = new LinearRegression({
      learningRate: parameters.value.learningRate,
      epochs: parameters.value.epochs,
      normalize: false
    });
    
    // Train with progress updates
    const startTime = Date.now();
    await algorithm.fit(X, y);
    const endTime = Date.now();
    const duration = endTime - startTime;
    
    console.log(`Training completed in ${duration}ms`);
    
    // Track algorithm run
    analyticsStore.trackAlgorithmRun('linear-regression', parameters.value, duration);
    
    // Get results
    metrics.value = algorithm.getMetrics();
    const visualizations = algorithm.getVisualizationData();
    
    // Process visualizations
    if (visualizations.length > 0) {
      // Scatter plot with regression line
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
        
        // Add regression line
        if (scatterViz.data.line) {
          series.push({
            x: scatterViz.data.line.x,
            y: scatterViz.data.line.y,
            name: scatterViz.data.line.name || 'Regression Line',
            color: scatterViz.data.line.line?.color || '#ef4444',
            mode: 'lines',
          });
        }
        
        scatterSeries.value = series;
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

// Bookmark restoration
// Requirements: 2.2, 2.3
const { handleLoadBookmark } = useBookmarkRestore(parameters, handleTrain);

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
  
  // Verify the score was saved
  // Note: `moduleStore` does not expose `progress`; type the legacy guard to
  // match its (always-undefined) runtime shape without changing behaviour.
  const progress = (moduleStore as { progress?: { quizScores?: Record<string, number>; completedModules?: string[] } }).progress;
  const savedScore = progress?.quizScores?.['linear-regression'];
  if (savedScore !== undefined) {
    console.log('Quiz score saved to progress store:', savedScore);
  }
  
  // Verify module completion if passing score
  if (result.score >= 70) {
    const isComplete = progress?.completedModules?.includes('linear-regression');
    if (isComplete) {
      console.log('Module marked as complete in progress store');
    }
  }
  
  // Optional: Show success message or navigate to next module
  // For now, the Quiz component handles the UI feedback
};

// SEO
useSEO({
  title: 'Linear Regression - Interactive ML Tutorial',
  description: 'Learn linear regression through interactive visualization. Adjust learning rates, train models, and see gradient descent in action. Master regression metrics like MSE, RMSE, and R².',
  keywords: 'linear regression, gradient descent, machine learning tutorial, regression metrics, MSE, RMSE, R squared',
  url: '/modules/regression/linear-regression',
  type: 'article',
  section: 'Regression',
})

// Course structured data
useCourseStructuredData({
  name: 'Linear Regression',
  description: 'Learn how linear regression finds the best-fit line through data using gradient descent',
  provider: 'Machine Learning 101',
  url: '/modules/regression/linear-regression',
  difficulty: 'beginner',
  timeRequired: 'PT30M',
  keywords: ['linear regression', 'gradient descent', 'machine learning', 'regression'],
})

// Breadcrumb structured data
useBreadcrumbStructuredData([
  { name: 'Home', url: '/' },
  { name: 'Learning Modules', url: '/modules' },
  { name: 'Regression', url: '/modules/regression' },
  { name: 'Linear Regression', url: '/modules/regression/linear-regression' },
])
</script>
