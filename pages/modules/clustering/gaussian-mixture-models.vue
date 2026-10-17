<template>
  <ModuleLayout
    title="Gaussian Mixture Models"
    description="Learn probabilistic clustering with Gaussian distributions using the EM algorithm"
    difficulty="advanced"
    :estimated-time="45"
    module-id="gaussian-mixture-models"
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
        loading-message="Training GMM..."
        :error="error"
        :on-retry="handleRetry"
      >
        <!-- Controls -->
        <template #controls>
          <ControlPanel
            :controls="controls"
            :values="parameters"
            :loading="isTraining"
            submit-label="Train GMM"
            @change="handleParameterChange"
            @reset="handleReset"
            @submit="handleTrain"
          />
          
          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Model Results</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Log-Likelihood</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.logLikelihood?.toFixed(2) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Silhouette Score</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.silhouetteScore?.toFixed(4) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">AIC</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.aic?.toFixed(2) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">BIC</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.bic?.toFixed(2) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Iterations</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.iterations }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Converged</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.converged ? 'Yes' : 'No' }}
                </div>
              </div>
            </div>
            
            <!-- Component Sizes -->
            <div v-if="metrics.clusterSizes && metrics.clusterSizes.length > 0" class="mt-4">
              <h4 class="text-xs font-semibold text-gray-700 mb-2">Component Sizes</h4>
              <div class="grid grid-cols-3 gap-2 text-center text-sm">
                <div
                  v-for="(size, idx) in metrics.clusterSizes"
                  :key="idx"
                  class="bg-blue-100 p-2 rounded"
                >
                  <div class="text-xs text-gray-600">Component {{ idx }}</div>
                  <div class="font-semibold">{{ size }} points</div>
                </div>
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
              title="Gaussian Mixture Model"
              x-label="Feature 1"
              y-label="Feature 2"
              :show-legend="true"
              height="500px"
            />
            
            <LineChart
              v-if="likelihoodSeries && likelihoodSeries.length > 0"
              :series="likelihoodSeries"
              title="Log-Likelihood Over Iterations"
              x-label="Iteration"
              y-label="Log-Likelihood"
              :show-legend="false"
              height="350px"
            />
            
            <div v-if="(!scatterSeries || scatterSeries.length === 0) && (!likelihoodSeries || likelihoodSeries.length === 0)" class="text-center py-8 text-gray-500">
              Train GMM to see clustering results
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
      <QuizErrorDisplay
        v-else-if="quizError"
        :error="quizError"
        @retry="retryQuizLoad"
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
import { useQuizLoader } from '~/composables/useQuizLoader';
import type { QuizResult } from '~/types/quiz';
import { useModuleStore } from '~/stores/module';
import { useDataset } from '~/composables/useDataset';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { GMM } from '~/algorithms/clustering/GMM';
import Scatter2D from '~/components/visualizations/Scatter2D.vue';
import LineChart from '~/components/visualizations/LineChart.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { ScatterSeries } from '~/components/visualizations/Scatter2D.vue';
import type { LineSeries } from '~/components/visualizations/LineChart.vue';

import { useAsyncData } from '#app'; 

// Load content from Nuxt Content
const { data: content } = await useAsyncData(() => 
  queryCollection('content').path('/modules/clustering/gaussian-mixture-models/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const { loadDataset, getDatasetsByType } = useDataset();

// Parameters
const parameters = ref({
  k: 3,
  maxIterations: 100,
  tolerance: 0.000001,
  initialization: 'kmeans++',
  covarianceType: 'full',
  dataset: 'blobs',
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'dataset',
    type: 'select',
    label: 'Dataset',
    options: getDatasetsByType('clustering').map(d => ({
      label: d.name,
      value: d.id
    })),
    tooltip: 'Choose a clustering dataset',
    group: 'Data',
  },
  {
    id: 'k',
    type: 'slider',
    label: 'Number of Components (k)',
    min: 2,
    max: 8,
    step: 1,
    tooltip: 'Number of Gaussian components in the mixture model.',
    group: 'Algorithm Parameters',
  },
  {
    id: 'initialization',
    type: 'select',
    label: 'Initialization Method',
    options: [
      { label: 'K-Means++', value: 'kmeans++' },
      { label: 'Random', value: 'random' }
    ],
    tooltip: 'Method for initializing component parameters.',
    group: 'Algorithm Parameters',
  },
  {
    id: 'covarianceType',
    type: 'select',
    label: 'Covariance Type',
    options: [
      { label: 'Full', value: 'full' },
      { label: 'Diagonal', value: 'diagonal' }
    ],
    tooltip: 'Type of covariance matrix for each component.',
    group: 'Algorithm Parameters',
  },
  {
    id: 'maxIterations',
    type: 'slider',
    label: 'Max Iterations',
    min: 10,
    max: 300,
    step: 10,
    tooltip: 'Maximum number of EM iterations.',
    group: 'Algorithm Parameters',
  },
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const scatterSeries = ref<ScatterSeries[]>([]);
const likelihoodSeries = ref<LineSeries[]>([]);

// Algorithm instance
let algorithm: GMM | null = null;

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

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
  moduleStore.setCurrentModule('gaussian-mixture-models');
  moduleStore.setParameters(parameters.value);
  
  // Load quiz data
  await loadQuiz('gaussian-mixture-models', '/content/modules/clustering/gaussian-mixture-models/quiz.json');
  
  // Load default dataset
  await loadInitialDataset();
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
      likelihoodSeries.value = [];
    }
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    k: 3,
    maxIterations: 100,
    tolerance: 0.000001,
    initialization: 'kmeans++',
    covarianceType: 'full',
    dataset: 'blobs',
  };
  moduleStore.setParameters(parameters.value);
  
  await loadInitialDataset();
  
  metrics.value = null;
  scatterSeries.value = [];
  likelihoodSeries.value = [];
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
    
    // Prepare data (clustering doesn't need labels)
    const X = dataset.map(d => d.features);
    
    // Create and train algorithm
    algorithm = new GMM({
      k: parameters.value.k,
      maxIterations: parameters.value.maxIterations,
      tolerance: parameters.value.tolerance,
      initialization: parameters.value.initialization as 'random' | 'kmeans++',
      covarianceType: parameters.value.covarianceType as 'full' | 'diagonal',
      randomSeed: 42 // For reproducibility
    });
    
    // Train
    const startTime = Date.now();
    await algorithm.fit(X);
    const endTime = Date.now();
    
    console.log(`GMM completed in ${endTime - startTime}ms`);
    
    // Get results
    metrics.value = algorithm.getMetrics();
    
    // Update visualizations
    updateVisualization();
    
    trainingProgress.value = 1;
  } catch (err) {
    console.error('GMM error:', err);
    error.value = err instanceof Error ? err.message : 'GMM training failed. Please try again.';
  } finally {
    isTraining.value = false;
  }
};

/**
 * Update visualization
 */
const updateVisualization = () => {
  if (!algorithm) return;
  
  const visualizations = algorithm.getVisualizationData();
  
  // Process visualizations
  if (visualizations.length > 0) {
    // Scatter plot with clusters
    const scatterViz = visualizations[0];
    if ((scatterViz.type === 'scatter2d' || scatterViz.type === 'scatter3d') && scatterViz.data) {
      const series: ScatterSeries[] = [];
      
      // Add cluster points
      const clusterKeys = Object.keys(scatterViz.data).filter(key => key.startsWith('cluster'));
      for (const clusterKey of clusterKeys) {
        const clusterData = scatterViz.data[clusterKey];
        if (clusterData && clusterData.x && clusterData.y) {
          series.push({
            x: clusterData.x,
            y: clusterData.y,
            name: clusterData.name || clusterKey,
            color: clusterData.marker?.color || '#3b82f6',
            mode: 'markers',
            marker: {
              size: clusterData.marker?.size || 8,
              opacity: clusterData.marker?.opacity || 0.7
            }
          });
        }
      }
      
      // Add component centers
      if (scatterViz.data.centers) {
        const centersData = scatterViz.data.centers;
        series.push({
          x: centersData.x,
          y: centersData.y,
          name: centersData.name || 'Component Centers',
          color: centersData.marker?.color || '#000000',
          mode: 'markers',
          marker: {
            size: centersData.marker?.size || 15,
            symbol: centersData.marker?.symbol || 'x'
          }
        });
      }
      
      scatterSeries.value = series;
    }
    
    // Log-likelihood curve
    if (visualizations.length > 1) {
      const likelihoodViz = visualizations[1];
      if (likelihoodViz.type === 'line' && likelihoodViz.data) {
        likelihoodSeries.value = [{
          x: likelihoodViz.data.x,
          y: likelihoodViz.data.y,
          name: likelihoodViz.data.name || 'Log-Likelihood',
          color: likelihoodViz.data.line?.color || '#10b981',
          width: likelihoodViz.data.line?.width || 2,
        }];
      }
    }
  }
};

/**
 * Handle loading bookmark
 * Requirements: 2.2, 2.3
 */
const handleLoadBookmark = async (bookmarkParameters: Record<string, any>) => {
  try {
    // Reload dataset if dataset parameter changed
    if (bookmarkParameters.dataset && bookmarkParameters.dataset !== parameters.value.dataset) {
      const dataset = await loadDataset(bookmarkParameters.dataset);
      if (dataset) {
        moduleStore.loadDataset(dataset.data);
      }
    }
    
    // Trigger re-training with new parameters
    await handleTrain();
  } catch (err) {
    console.error('Error loading bookmark:', err);
    error.value = 'Failed to load bookmark configuration';
  }
};

/**
 * Handle retry
 */
const handleRetry = () => {
  error.value = null;
  handleTrain();
};
</script>