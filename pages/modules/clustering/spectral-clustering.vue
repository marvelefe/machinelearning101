<template>
  <ModuleLayout
    title="Spectral Clustering"
    description="Learn graph-based clustering using eigenvalue decomposition to handle non-convex cluster shapes"
    difficulty="advanced"
    :estimated-time="50"
    module-id="spectral-clustering"
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
        loading-message="Running Spectral Clustering..."
        :error="error"
        :on-retry="handleRetry"
      >
        <!-- Controls -->
        <template #controls>
          <ControlPanel
            :controls="controls"
            :values="parameters"
            :loading="isTraining"
            submit-label="Run Spectral Clustering"
            @change="handleParameterChange"
            @reset="handleReset"
            @submit="handleTrain"
          />
          
          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Clustering Results</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Silhouette Score</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.silhouetteScore?.toFixed(4) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Spectral Gap</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.spectralGap?.toFixed(4) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Affinity Type</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.affinity }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Laplacian Type</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.laplacian }}</div>
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
            
            <!-- Eigenvalues -->
            <div v-if="metrics.eigenvalues && metrics.eigenvalues.length > 0" class="mt-4">
              <h4 class="text-xs font-semibold text-gray-700 mb-2">Eigenvalues</h4>
              <div class="grid grid-cols-4 gap-2 text-center text-sm">
                <div
                  v-for="(eigenval, idx) in metrics.eigenvalues.slice(0, 8)"
                  :key="idx"
                  class="bg-green-100 p-2 rounded"
                >
                  <div class="text-xs text-gray-600">λ{{ idx + 1 }}</div>
                  <div class="font-semibold">{{ eigenval.toFixed(3) }}</div>
                </div>
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <Scatter2D
              v-if="originalSeries && originalSeries.length > 0"
              :series="originalSeries"
              title="Spectral Clustering (Original Space)"
              x-label="Feature 1"
              y-label="Feature 2"
              :show-legend="true"
              height="400px"
            />
            
            <Scatter2D
              v-if="embeddingSeries && embeddingSeries.length > 0"
              :series="embeddingSeries"
              title="Spectral Embedding"
              x-label="1st Eigenvector"
              y-label="2nd Eigenvector"
              :show-legend="true"
              height="400px"
            />
            
            <div v-if="(!originalSeries || originalSeries.length === 0) && (!embeddingSeries || embeddingSeries.length === 0)" class="text-center py-8 text-gray-500">
              Run Spectral Clustering to see results
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
import { SpectralClustering } from '~/algorithms/clustering/SpectralClustering';
import Scatter2D from '~/components/visualizations/Scatter2D.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { ScatterSeries } from '~/components/visualizations/Scatter2D.vue';

import { useAsyncData } from '#app'; 

// Load content from Nuxt Content
const { data: content } = await useAsyncData(() => 
  queryCollection('content').path('/modules/clustering/spectral-clustering/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const { loadDataset, getDatasetsByType } = useDataset();

// Parameters
const parameters = ref({
  k: 3,
  sigma: 1.0,
  affinity: 'rbf',
  nNeighbors: 10,
  gamma: 1.0,
  laplacian: 'rw',
  dataset: 'moons',
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
    label: 'Number of Clusters (k)',
    min: 2,
    max: 8,
    step: 1,
    tooltip: 'Number of clusters to find using spectral clustering.',
    group: 'Algorithm Parameters',
  },
  {
    id: 'affinity',
    type: 'select',
    label: 'Affinity Type',
    options: [
      { label: 'RBF (Gaussian)', value: 'rbf' },
      { label: 'Nearest Neighbors', value: 'nearest_neighbors' },
      { label: 'Polynomial', value: 'polynomial' }
    ],
    tooltip: 'Method for constructing the affinity matrix.',
    group: 'Algorithm Parameters',
  },
  {
    id: 'gamma',
    type: 'slider',
    label: 'Gamma',
    min: 0.1,
    max: 5.0,
    step: 0.1,
    tooltip: 'Kernel coefficient for RBF and polynomial kernels. Higher values create more localized similarities.',
    group: 'Algorithm Parameters',
  },
  {
    id: 'nNeighbors',
    type: 'slider',
    label: 'Number of Neighbors',
    min: 3,
    max: 50,
    step: 1,
    tooltip: 'Number of neighbors for nearest neighbors affinity (only used when affinity is \'nearest_neighbors\').',
    group: 'Algorithm Parameters',
  },
  {
    id: 'laplacian',
    type: 'select',
    label: 'Laplacian Type',
    options: [
      { label: 'Random Walk (RW)', value: 'rw' },
      { label: 'Symmetric', value: 'symmetric' },
      { label: 'Unnormalized', value: 'unnormalized' }
    ],
    tooltip: 'Type of Laplacian matrix normalization.',
    group: 'Algorithm Parameters',
  },
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const originalSeries = ref<ScatterSeries[]>([]);
const embeddingSeries = ref<ScatterSeries[]>([]);

// Algorithm instance
let algorithm: SpectralClustering | null = null;

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
  moduleStore.setCurrentModule('spectral-clustering');
  moduleStore.setParameters(parameters.value);
  
  // Load quiz data
  await loadQuiz('spectral-clustering', '/content/modules/clustering/spectral-clustering/quiz.json');
  
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
      originalSeries.value = [];
      embeddingSeries.value = [];
    }
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    k: 3,
    sigma: 1.0,
    affinity: 'rbf',
    nNeighbors: 10,
    gamma: 1.0,
    laplacian: 'rw',
    dataset: 'moons',
  };
  moduleStore.setParameters(parameters.value);
  
  await loadInitialDataset();
  
  metrics.value = null;
  originalSeries.value = [];
  embeddingSeries.value = [];
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
    algorithm = new SpectralClustering({
      k: parameters.value.k,
      sigma: parameters.value.sigma,
      affinity: parameters.value.affinity as 'rbf' | 'nearest_neighbors' | 'polynomial',
      nNeighbors: parameters.value.nNeighbors,
      degree: 3,
      gamma: parameters.value.gamma,
      laplacian: parameters.value.laplacian as 'unnormalized' | 'rw' | 'symmetric',
      eigenSolver: 'arpack',
      randomSeed: 42 // For reproducibility
    });
    
    // Train
    const startTime = Date.now();
    await algorithm.fit(X);
    const endTime = Date.now();
    
    console.log(`Spectral Clustering completed in ${endTime - startTime}ms`);
    
    // Get results
    metrics.value = algorithm.getMetrics();
    
    // Update visualizations
    updateVisualization();
    
    trainingProgress.value = 1;
  } catch (err) {
    console.error('Spectral Clustering error:', err);
    error.value = err instanceof Error ? err.message : 'Spectral Clustering failed. Please try again.';
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
    // Original space visualization
    const originalViz = visualizations[0];
    if ((originalViz.type === 'scatter2d' || originalViz.type === 'scatter3d') && originalViz.data) {
      const series: ScatterSeries[] = [];
      
      // Add cluster points
      const clusterKeys = Object.keys(originalViz.data).filter(key => key.startsWith('cluster'));
      for (const clusterKey of clusterKeys) {
        const clusterData = originalViz.data[clusterKey];
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
      
      originalSeries.value = series;
    }
    
    // Spectral embedding visualization
    if (visualizations.length > 1) {
      const embeddingViz = visualizations[1];
      if ((embeddingViz.type === 'scatter2d' || embeddingViz.type === 'scatter3d') && embeddingViz.data) {
        const series: ScatterSeries[] = [];
        
        // Add cluster points in embedding space
        const clusterKeys = Object.keys(embeddingViz.data).filter(key => key.startsWith('cluster'));
        for (const clusterKey of clusterKeys) {
          const clusterData = embeddingViz.data[clusterKey];
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
        
        embeddingSeries.value = series;
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