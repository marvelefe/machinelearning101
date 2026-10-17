<template>
  <ModuleLayout
    title="Hierarchical Clustering"
    description="Learn how hierarchical clustering builds a tree of clusters using agglomerative or divisive approaches"
    difficulty="intermediate"
    :estimated-time="30"
    module-id="hierarchical-clustering"
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
        loading-message="Building cluster hierarchy..."
        :error="error"
        :on-retry="handleRetry"
      >
        <!-- Controls -->
        <template #controls>
          <ControlPanel
            :controls="controls"
            :values="parameters"
            :loading="isTraining"
            submit-label="Run Clustering"
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
                <div class="text-xs text-gray-500 mt-1">(-1 to 1, higher is better)</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Davies-Bouldin Score</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.daviesBouldinScore?.toFixed(4) }}</div>
                <div class="text-xs text-gray-500 mt-1">(Lower is better)</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Calinski-Harabasz Score</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.calinskiHarabaszScore?.toFixed(2) }}</div>
                <div class="text-xs text-gray-500 mt-1">(Higher is better)</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Number of Merges</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.nMerges }}</div>
              </div>
            </div>
            
            <!-- Cluster Sizes -->
            <div v-if="metrics.clusterSizes" class="mt-4">
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
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <Scatter2D
              v-if="scatterSeries && scatterSeries.length > 0"
              :series="scatterSeries"
              title="Hierarchical Clustering Results"
              x-label="Feature 1"
              y-label="Feature 2"
              :show-legend="true"
              height="500px"
            />
            
            <div v-if="metrics" class="p-4 bg-blue-50 rounded-lg border border-blue-200">
              <h3 class="text-sm font-semibold text-blue-700 mb-2">About Linkage Methods</h3>
              <div class="text-sm text-blue-900">
                <p v-if="parameters.linkage === 'ward'">
                  <strong>Ward's Method:</strong> Minimizes within-cluster variance. Creates compact, evenly-sized clusters. Best for well-separated, spherical clusters.
                </p>
                <p v-else-if="parameters.linkage === 'complete'">
                  <strong>Complete Linkage:</strong> Uses maximum distance between clusters. Creates compact, spherical clusters and avoids chaining.
                </p>
                <p v-else-if="parameters.linkage === 'average'">
                  <strong>Average Linkage:</strong> Uses mean distance between all pairs. Balances between single and complete linkage.
                </p>
                <p v-else-if="parameters.linkage === 'single'">
                  <strong>Single Linkage:</strong> Uses minimum distance between clusters. Can create elongated clusters (chaining effect).
                </p>
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
import { HierarchicalClustering } from '~/algorithms/clustering/HierarchicalClustering';
import Scatter2D from '~/components/visualizations/Scatter2D.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { ScatterSeries } from '~/components/visualizations/Scatter2D.vue';
import { useAsyncData } from '#app';


// Load content from Nuxt Content
const { data: content } = await useAsyncData(() => 
  queryCollection('content').path('/modules/clustering/hierarchical-clustering/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const { loadDataset, getDatasetsByType } = useDataset();

// Parameters
const parameters = ref({
  nClusters: 3,
  linkage: 'ward',
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
    id: 'nClusters',
    type: 'slider',
    label: 'Number of Clusters',
    min: 2,
    max: 8,
    step: 1,
    tooltip: 'Number of clusters to form by cutting the dendrogram',
    group: 'Model Parameters',
  },
  {
    id: 'linkage',
    type: 'select',
    label: 'Linkage Method',
    options: [
      { label: 'Ward (Minimize variance)', value: 'ward' },
      { label: 'Complete (Maximum distance)', value: 'complete' },
      { label: 'Average (Mean distance)', value: 'average' },
      { label: 'Single (Minimum distance)', value: 'single' }
    ],
    tooltip: 'Method to calculate distance between clusters',
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

// Algorithm instance
let algorithm: HierarchicalClustering | null = null;

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
  moduleStore.setCurrentModule('hierarchical-clustering');
  moduleStore.setParameters(parameters.value);
  
  // Load quiz data
  await loadQuiz('hierarchical-clustering', '/content/modules/clustering/hierarchical-clustering/quiz.json');
  
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
    nClusters: 3,
    linkage: 'ward',
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
    algorithm = new HierarchicalClustering({
      nClusters: parameters.value.nClusters,
      linkage: parameters.value.linkage as 'single' | 'complete' | 'average' | 'ward'
    });
    
    // Train
    const startTime = Date.now();
    await algorithm.fit(X);
    const endTime = Date.now();
    
    console.log(`Clustering completed in ${endTime - startTime}ms`);
    
    // Get results
    metrics.value = algorithm.getMetrics();
    
    // Update visualizations
    updateVisualization();
    
    trainingProgress.value = 1;
  } catch (err) {
    console.error('Clustering error:', err);
    error.value = err instanceof Error ? err.message : 'Clustering failed. Please try again.';
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
      for (let c = 0; c < parameters.value.nClusters; c++) {
        const clusterKey = `cluster${c}`;
        if (scatterViz.data[clusterKey]) {
          series.push({
            x: scatterViz.data[clusterKey].x,
            y: scatterViz.data[clusterKey].y,
            name: scatterViz.data[clusterKey].name || `Cluster ${c}`,
            color: scatterViz.data[clusterKey].marker?.color || '#3b82f6',
            mode: 'markers',
            marker: {
              size: scatterViz.data[clusterKey].marker?.size || 8,
              opacity: scatterViz.data[clusterKey].marker?.opacity || 0.7
            }
          });
        }
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
