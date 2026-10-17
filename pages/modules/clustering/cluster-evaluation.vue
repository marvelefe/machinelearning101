<template>
  <ModuleLayout
    title="Cluster Evaluation Metrics"
    description="Learn to evaluate clustering quality using internal and external metrics"
    difficulty="intermediate"
    :estimated-time="35"
    module-id="cluster-evaluation"
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
        :loading="isEvaluating"
        :progress="evaluationProgress"
        loading-message="Evaluating clustering..."
        :error="error"
        :on-retry="handleRetry"
      >
        <!-- Controls -->
        <template #controls>
          <ControlPanel
            :controls="controls"
            :values="parameters"
            :loading="isEvaluating"
            submit-label="Evaluate Clustering"
            @change="handleParameterChange"
            @reset="handleReset"
            @submit="handleEvaluate"
          />
          
          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Evaluation Results</h3>
            
            <!-- Summary -->
            <div v-if="metrics.summary" class="mb-4 p-3 bg-blue-50 rounded border border-blue-200">
              <h4 class="text-xs font-semibold text-blue-700 mb-2">Summary</h4>
              <ul class="text-sm text-blue-800 space-y-1">
                <li v-for="(item, idx) in metrics.summary" :key="idx" class="flex items-start">
                  <span class="w-2 h-2 bg-blue-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                  {{ item }}
                </li>
              </ul>
            </div>
            
            <!-- Internal Metrics -->
            <div class="mb-4">
              <h4 class="text-xs font-semibold text-gray-700 mb-2">Internal Metrics</h4>
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <div class="text-xs text-gray-500">Silhouette Score</div>
                  <div class="text-lg font-semibold text-gray-900">{{ metrics.silhouetteScore?.toFixed(4) }}</div>
                </div>
                <div>
                  <div class="text-xs text-gray-500">Calinski-Harabasz</div>
                  <div class="text-lg font-semibold text-gray-900">{{ metrics.calinskiHarabaszIndex?.toFixed(2) }}</div>
                </div>
                <div>
                  <div class="text-xs text-gray-500">Davies-Bouldin</div>
                  <div class="text-lg font-semibold text-gray-900">{{ metrics.daviesBouldinIndex?.toFixed(4) }}</div>
                </div>
                <div>
                  <div class="text-xs text-gray-500">Dunn Index</div>
                  <div class="text-lg font-semibold text-gray-900">{{ metrics.dunn?.toFixed(4) }}</div>
                </div>
              </div>
            </div>
            
            <!-- External Metrics (if available) -->
            <div v-if="parameters.withGroundTruth && metrics.adjustedRandIndex !== undefined" class="mb-4">
              <h4 class="text-xs font-semibold text-gray-700 mb-2">External Metrics</h4>
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <div class="text-xs text-gray-500">Adjusted Rand Index</div>
                  <div class="text-lg font-semibold text-gray-900">{{ metrics.adjustedRandIndex?.toFixed(4) }}</div>
                </div>
                <div>
                  <div class="text-xs text-gray-500">Normalized MI</div>
                  <div class="text-lg font-semibold text-gray-900">{{ metrics.normalizedMutualInfo?.toFixed(4) }}</div>
                </div>
                <div>
                  <div class="text-xs text-gray-500">V-Measure</div>
                  <div class="text-lg font-semibold text-gray-900">{{ metrics.vMeasure?.toFixed(4) }}</div>
                </div>
                <div>
                  <div class="text-xs text-gray-500">Fowlkes-Mallows</div>
                  <div class="text-lg font-semibold text-gray-900">{{ metrics.fowlkesMallowsIndex?.toFixed(4) }}</div>
                </div>
              </div>
            </div>
            
            <!-- Basic Info -->
            <div class="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div class="text-xs text-gray-500">Number of Clusters</div>
                <div class="font-semibold text-gray-900">{{ metrics.numClusters }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Number of Points</div>
                <div class="font-semibold text-gray-900">{{ metrics.numPoints }}</div>
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <Scatter2D
              v-if="clusteringSeries && clusteringSeries.length > 0"
              :series="clusteringSeries"
              title="Clustering Results"
              x-label="Feature 1"
              y-label="Feature 2"
              :show-legend="true"
              height="400px"
            />
            
            <div v-if="!clusteringSeries || clusteringSeries.length === 0" class="text-center py-8 text-gray-500">
              Run clustering evaluation to see results
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
import { ClusterEvaluation } from '~/algorithms/clustering/ClusterEvaluation';
import { KMeans } from '~/algorithms/clustering/KMeans';
import { DBSCAN } from '~/algorithms/clustering/DBSCAN';
import { GMM } from '~/algorithms/clustering/GMM';
import { MeanShift } from '~/algorithms/clustering/MeanShift';
import type { KMeansParams } from '~/algorithms/clustering/KMeans';
import type { DBSCANParams } from '~/algorithms/clustering/DBSCAN';
import type { GMMParams } from '~/algorithms/clustering/GMM';
import type { MeanShiftParams } from '~/algorithms/clustering/MeanShift';
import Scatter2D from '~/components/visualizations/Scatter2D.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { ScatterSeries } from '~/components/visualizations/Scatter2D.vue';

import { useAsyncData } from '#app'; 

// Load content from Nuxt Content
const { data: content } = await useAsyncData(() => 
  queryCollection('content').path('/modules/clustering/cluster-evaluation/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const { loadDataset, getDatasetsByType } = useDataset();

// Parameters
const parameters = ref({
  dataset: 'blobs',
  algorithm: 'kmeans',
  k: 3,
  withGroundTruth: true,
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
    id: 'algorithm',
    type: 'select',
    label: 'Clustering Algorithm',
    options: [
      { label: 'K-Means', value: 'kmeans' },
      { label: 'DBSCAN', value: 'dbscan' },
      { label: 'GMM', value: 'gmm' },
      { label: 'Mean Shift', value: 'meanshift' }
    ],
    tooltip: 'Choose which clustering algorithm to evaluate',
    group: 'Algorithm',
  },
  {
    id: 'k',
    type: 'slider',
    label: 'Number of Clusters (k)',
    min: 2,
    max: 8,
    step: 1,
    tooltip: 'Number of clusters for algorithms that require it (K-Means, GMM).',
    group: 'Algorithm Parameters',
  },
  {
    id: 'withGroundTruth',
    type: 'toggle',
    label: 'Use Ground Truth',
    tooltip: 'Include external metrics that compare against ground truth labels.',
    group: 'Evaluation',
  },
];

// Evaluation state
const isEvaluating = ref(false);
const evaluationProgress = ref(0);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const clusteringSeries = ref<ScatterSeries[]>([]);

// Algorithm instances
let evaluator: ClusterEvaluation | null = null;

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
  moduleStore.setCurrentModule('cluster-evaluation');
  moduleStore.setParameters(parameters.value);
  
  // Load quiz data
  await loadQuiz('cluster-evaluation', '/content/modules/clustering/cluster-evaluation/quiz.json');
  
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
      clusteringSeries.value = [];
    }
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    dataset: 'blobs',
    algorithm: 'kmeans',
    k: 3,
    withGroundTruth: true,
  };
  moduleStore.setParameters(parameters.value);
  
  await loadInitialDataset();
  
  metrics.value = null;
  clusteringSeries.value = [];
  error.value = null;
};

/**
 * Run clustering algorithm
 */
const runClusteringAlgorithm = async (X: number[][]): Promise<{ labels: number[]; algorithm: any }> => {
  const { algorithm, k } = parameters.value;
  
  let clusteringAlgorithm: any;
  
  switch (algorithm) {
    case 'kmeans':
      clusteringAlgorithm = new KMeans({ k, randomSeed: 42 } as KMeansParams);
      break;
    case 'dbscan':
      clusteringAlgorithm = new DBSCAN({ eps: 0.5, minPts: 5 } as DBSCANParams);
      break;
    case 'gmm':
      clusteringAlgorithm = new GMM({ k, randomSeed: 42 } as GMMParams);
      break;
    case 'meanshift':
      clusteringAlgorithm = new MeanShift({ bandwidth: 1.0 } as MeanShiftParams);
      break;
    default:
      throw new Error(`Unknown algorithm: ${algorithm}`);
  }
  
  await clusteringAlgorithm.fit(X);
  const labels = clusteringAlgorithm.predict(X);
  
  return { labels, algorithm: clusteringAlgorithm };
};

/**
 * Handle evaluate
 */
const handleEvaluate = async () => {
  isEvaluating.value = true;
  evaluationProgress.value = 0;
  error.value = null;

  try {
    // Get dataset from store
    const dataset = moduleStore.dataset;
    if (!dataset || dataset.length === 0) {
      throw new Error('No dataset loaded');
    }
    
    // Prepare data
    const X = dataset.map(d => d.features);
    const trueLabels = parameters.value.withGroundTruth ? dataset.map(d => (d.label as number) || 0) : undefined;
    
    evaluationProgress.value = 0.3;
    
    // Run clustering algorithm
    const { labels } = await runClusteringAlgorithm(X);
    
    evaluationProgress.value = 0.6;
    
    // Create and run evaluation
    evaluator = new ClusterEvaluation();
    
    const startTime = Date.now();
    const results = await evaluator.evaluateClustering(X, labels, trueLabels);
    const endTime = Date.now();
    
    console.log(`Cluster evaluation completed in ${endTime - startTime}ms`);
    
    // Get results
    metrics.value = evaluator.getMetrics();
    
    // Update visualizations
    updateVisualization(X, labels);
    
    evaluationProgress.value = 1;
  } catch (err) {
    console.error('Cluster evaluation error:', err);
    error.value = err instanceof Error ? err.message : 'Cluster evaluation failed. Please try again.';
  } finally {
    isEvaluating.value = false;
  }
};

/**
 * Update visualization
 */
const updateVisualization = (data: number[][], labels: number[]) => {
  if (data[0].length < 2) return;
  
  const series: ScatterSeries[] = [];
  const uniqueLabels = [...new Set(labels)];
  const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];
  
  // Add cluster points
  for (let c = 0; c < uniqueLabels.length; c++) {
    const clusterPoints = data.filter((_, i) => labels[i] === uniqueLabels[c]);
    
    if (clusterPoints.length > 0) {
      const x = clusterPoints.map(p => p[0]);
      const y = clusterPoints.map(p => p[1]);
      
      series.push({
        x,
        y,
        name: `Cluster ${uniqueLabels[c]}`,
        color: colors[c % colors.length],
        mode: 'markers',
        marker: {
          size: 8,
          opacity: 0.8
        }
      });
    }
  }
  
  clusteringSeries.value = series;
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
    
    // Trigger evaluation with new parameters
    await handleEvaluate();
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
  handleEvaluate();
};
</script>