<template>
  <ModuleLayout
    title="Mean Shift Clustering"
    description="Learn mode-seeking clustering that automatically finds density peaks and cluster numbers"
    difficulty="intermediate"
    :estimated-time="40"
    module-id="mean-shift"
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
        loading-message="Running Mean Shift..."
        :error="error"
        :on-retry="handleRetry"
      >
        <!-- Controls -->
        <template #controls>
          <ControlPanel
            :controls="controls"
            :values="parameters"
            :loading="isTraining"
            submit-label="Run Mean Shift"
            @change="handleParameterChange"
            @reset="handleReset"
            @submit="handleTrain"
          />
          
          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Clustering Results</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Clusters Found</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.numClusters }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Silhouette Score</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.silhouetteScore?.toFixed(4) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Convergence Rate</div>
                <div class="text-lg font-semibold text-gray-900">{{ (metrics.convergenceRate * 100).toFixed(1) }}%</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Avg Iterations</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.avgIterations?.toFixed(1) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Bandwidth</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.bandwidth?.toFixed(2) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Total Points</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.totalPoints }}</div>
              </div>
            </div>
            
            <!-- Cluster Sizes -->
            <div v-if="metrics.clusterSizes && metrics.clusterSizes.length > 0" class="mt-4">
              <h4 class="text-xs font-semibold text-gray-700 mb-2">Cluster Sizes</h4>
              <div class="grid grid-cols-3 gap-2 text-center text-sm">
                <div
                  v-for="(size, idx) in metrics.clusterSizes"
                  :key="idx"
                  class="bg-blue-100 p-2 rounded"
                >
                  <div class="text-xs text-gray-600">Cluster {{ idx }}</div>
                  <div class="font-semibold">{{ size }} points</div>
                </div>
              </div>
            </div>
            
            <!-- Mode Densities -->
            <div v-if="metrics.modeDensities && metrics.modeDensities.length > 0" class="mt-4">
              <h4 class="text-xs font-semibold text-gray-700 mb-2">Mode Densities</h4>
              <div class="grid grid-cols-3 gap-2 text-center text-sm">
                <div
                  v-for="(density, idx) in metrics.modeDensities"
                  :key="idx"
                  class="bg-green-100 p-2 rounded"
                >
                  <div class="text-xs text-gray-600">Mode {{ idx }}</div>
                  <div class="font-semibold">{{ density.toFixed(2) }}</div>
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
              title="Mean Shift Clustering"
              x-label="Feature 1"
              y-label="Feature 2"
              :show-legend="true"
              height="500px"
            />
            
            <div v-if="!scatterSeries || scatterSeries.length === 0" class="text-center py-8 text-gray-500">
              Run Mean Shift to see clustering results
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
import { MeanShift } from '~/algorithms/clustering/MeanShift';
import Scatter2D from '~/components/visualizations/Scatter2D.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { ScatterSeries } from '~/components/visualizations/Scatter2D.vue';

import { useAsyncData } from '#app';


// Load content from Nuxt Content
const { data: content } = await useAsyncData(() => 
  queryCollection('content').path('/modules/clustering/mean-shift/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const { loadDataset, getDatasetsByType } = useDataset();

// Parameters
const parameters = ref({
  bandwidth: 1.0,
  maxIterations: 300,
  tolerance: 0.001,
  kernel: 'gaussian',
  binSeeding: false,
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
    id: 'bandwidth',
    type: 'slider',
    label: 'Bandwidth',
    min: 0.1,
    max: 3.0,
    step: 0.1,
    tooltip: 'Controls the size of the neighborhood for density estimation. Larger values create fewer, larger clusters.',
    group: 'Algorithm Parameters',
  },
  {
    id: 'kernel',
    type: 'select',
    label: 'Kernel Type',
    options: [
      { label: 'Gaussian', value: 'gaussian' },
      { label: 'Flat (Uniform)', value: 'flat' }
    ],
    tooltip: 'Type of kernel function for density estimation.',
    group: 'Algorithm Parameters',
  },
  {
    id: 'binSeeding',
    type: 'toggle',
    label: 'Bin Seeding',
    tooltip: 'Use grid-based seeding instead of using all data points as seeds. Can be faster for large datasets.',
    group: 'Algorithm Parameters',
  },
  {
    id: 'maxIterations',
    type: 'slider',
    label: 'Max Iterations',
    min: 50,
    max: 500,
    step: 50,
    tooltip: 'Maximum number of iterations for each point to converge to a mode.',
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

// Algorithm instance
let algorithm: MeanShift | null = null;

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
  moduleStore.setCurrentModule('mean-shift');
  moduleStore.setParameters(parameters.value);
  
  // Load quiz data
  await loadQuiz('mean-shift', '/content/modules/clustering/mean-shift/quiz.json');
  
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
    }
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    bandwidth: 1.0,
    maxIterations: 300,
    tolerance: 0.001,
    kernel: 'gaussian',
    binSeeding: false,
    dataset: 'blobs',
  };
  moduleStore.setParameters(parameters.value);
  
  await loadInitialDataset();
  
  metrics.value = null;
  scatterSeries.value = [];
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
    algorithm = new MeanShift({
      bandwidth: parameters.value.bandwidth,
      maxIterations: parameters.value.maxIterations,
      tolerance: parameters.value.tolerance,
      kernel: parameters.value.kernel as 'gaussian' | 'flat',
      binSeeding: parameters.value.binSeeding
    });
    
    // Train
    const startTime = Date.now();
    await algorithm.fit(X);
    const endTime = Date.now();
    
    console.log(`Mean Shift completed in ${endTime - startTime}ms`);
    
    // Get results
    metrics.value = algorithm.getMetrics();
    
    // Update visualizations
    updateVisualization();
    
    trainingProgress.value = 1;
  } catch (err) {
    console.error('Mean Shift error:', err);
    error.value = err instanceof Error ? err.message : 'Mean Shift failed. Please try again.';
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
              opacity: clusterData.marker?.opacity || 0.8
            }
          });
        }
      }
      
      // Add modes
      if (scatterViz.data.modes) {
        const modesData = scatterViz.data.modes;
        series.push({
          x: modesData.x,
          y: modesData.y,
          name: modesData.name || 'Modes',
          color: modesData.marker?.color || '#000000',
          mode: 'markers',
          marker: {
            size: modesData.marker?.size || 15,
            symbol: modesData.marker?.symbol || 'star'
          }
        });
      }
      
      scatterSeries.value = series;
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