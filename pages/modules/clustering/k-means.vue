<template>
  <ModuleLayout
    title="K-Means Clustering"
    description="Learn how K-Means groups similar data points into clusters using iterative centroid updates"
    difficulty="beginner"
    :estimated-time="35"
    module-id="k-means"
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
        loading-message="Clustering data..."
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
                <div class="text-xs text-gray-500">Inertia</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.inertia?.toFixed(2) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Silhouette Score</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.silhouetteScore?.toFixed(4) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Iterations</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.iterations }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Converged</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.iterations < parameters.maxIterations ? 'Yes' : 'No' }}
                </div>
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
          
          <!-- Step-by-step controls -->
          <div v-if="stepByStepMode && clusterHistory.length > 0" class="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
            <h3 class="text-sm font-semibold text-blue-700 mb-3">Step-by-Step Mode</h3>
            <div class="flex items-center gap-2">
              <Button
                size="sm"
                variant="secondary"
                :disabled="currentStep === 0"
                @click="previousStep"
              >
                Previous
              </Button>
              <div class="flex-1 text-center text-sm font-medium text-blue-900">
                Iteration {{ currentStep }} / {{ clusterHistory.length - 1 }}
              </div>
              <Button
                size="sm"
                variant="secondary"
                :disabled="currentStep === clusterHistory.length - 1"
                @click="nextStep"
              >
                Next
              </Button>
            </div>
            <div class="mt-2">
              <input
                type="range"
                :min="0"
                :max="clusterHistory.length - 1"
                v-model.number="currentStep"
                class="w-full"
              />
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <Scatter2D
              v-if="scatterSeries && scatterSeries.length > 0"
              :series="scatterSeries"
              title="K-Means Clustering"
              x-label="Feature 1"
              y-label="Feature 2"
              :show-legend="true"
              height="500px"
            />
            
            <LineChart
              v-if="inertiaSeries && inertiaSeries.length > 0"
              :series="inertiaSeries"
              title="Inertia Over Iterations"
              x-label="Iteration"
              y-label="Inertia (Within-Cluster Sum of Squares)"
              :show-legend="false"
              height="350px"
            />
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
import { ref, onMounted, watch } from 'vue';
import { useQuizLoader } from '~/composables/useQuizLoader';
import type { QuizResult } from '~/types/quiz';
import { useModuleStore } from '~/stores/module';
import { useDataset } from '~/composables/useDataset';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { KMeans } from '~/algorithms/clustering/KMeans';
import Scatter2D from '~/components/visualizations/Scatter2D.vue';
import LineChart from '~/components/visualizations/LineChart.vue';
import Button from '~/components/shared/Button.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { ScatterSeries } from '~/components/visualizations/Scatter2D.vue';
import type { LineSeries } from '~/components/visualizations/LineChart.vue';
import { useAsyncData } from '#app';


// Load content from Nuxt Content
const { data: content } = await useAsyncData(() => 
  queryCollection('content').path('/modules/clustering/k-means/content').first()
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
  tolerance: 0.0001,
  initialization: 'kmeans++',
  stepByStep: false,
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
    label: 'Number of Clusters (k)',
    min: 2,
    max: 8,
    step: 1,
    tooltip: 'Number of clusters to find. Try different values to see how it affects the results.',
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
    tooltip: 'K-Means++ provides better initial centroids than random initialization.',
    group: 'Algorithm Parameters',
  },
  {
    id: 'maxIterations',
    type: 'slider',
    label: 'Max Iterations',
    min: 10,
    max: 300,
    step: 10,
    tooltip: 'Maximum number of iterations before stopping.',
    group: 'Algorithm Parameters',
  },
  {
    id: 'stepByStep',
    type: 'toggle',
    label: 'Step-by-Step Mode',
    tooltip: 'Enable to see how clusters evolve iteration by iteration.',
    group: 'Visualization',
  },
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const scatterSeries = ref<ScatterSeries[]>([]);
const inertiaSeries = ref<LineSeries[]>([]);

// Step-by-step mode
const stepByStepMode = ref(false);
const currentStep = ref(0);
const clusterHistory = ref<any[]>([]);

// Algorithm instance
let algorithm: KMeans | null = null;

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
  moduleStore.setCurrentModule('k-means');
  moduleStore.setParameters(parameters.value);
  
  // Load quiz data
  await loadQuiz('k-means', '/content/modules/clustering/k-means/quiz.json');
  
  // Load default dataset
  await loadInitialDataset();
});

/**
 * Watch for step changes in step-by-step mode
 */
watch(currentStep, (newStep) => {
  if (stepByStepMode.value && clusterHistory.value.length > 0) {
    updateVisualizationForStep(newStep);
  }
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
  
  // Handle step-by-step mode toggle
  if (id === 'stepByStep') {
    stepByStepMode.value = value;
    if (!value && algorithm) {
      // Show final results when disabling step-by-step
      updateVisualizationForFinal();
    }
  }
  
  // Reload dataset if changed
  if (id === 'dataset') {
    const dataset = await loadDataset(value);
    if (dataset) {
      moduleStore.loadDataset(dataset.data);
      // Clear previous results
      metrics.value = null;
      scatterSeries.value = [];
      inertiaSeries.value = [];
      clusterHistory.value = [];
      currentStep.value = 0;
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
    tolerance: 0.0001,
    initialization: 'kmeans++',
    stepByStep: false,
    dataset: 'blobs',
  };
  moduleStore.setParameters(parameters.value);
  
  await loadInitialDataset();
  
  metrics.value = null;
  scatterSeries.value = [];
  inertiaSeries.value = [];
  clusterHistory.value = [];
  currentStep.value = 0;
  stepByStepMode.value = false;
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
    algorithm = new KMeans({
      k: parameters.value.k,
      maxIterations: parameters.value.maxIterations,
      tolerance: parameters.value.tolerance,
      initialization: parameters.value.initialization as 'random' | 'kmeans++',
      randomSeed: 42 // For reproducibility
    });
    
    // Train
    const startTime = Date.now();
    await algorithm.fit(X);
    const endTime = Date.now();
    
    console.log(`Clustering completed in ${endTime - startTime}ms`);
    
    // Get results
    metrics.value = algorithm.getMetrics();
    
    // Get cluster history for step-by-step mode
    clusterHistory.value = algorithm.getClusterHistory();
    currentStep.value = clusterHistory.value.length - 1;
    
    // Update visualizations
    if (stepByStepMode.value) {
      updateVisualizationForStep(currentStep.value);
    } else {
      updateVisualizationForFinal();
    }
    
    trainingProgress.value = 1;
  } catch (err) {
    console.error('Clustering error:', err);
    error.value = err instanceof Error ? err.message : 'Clustering failed. Please try again.';
  } finally {
    isTraining.value = false;
  }
};

/**
 * Update visualization for a specific step
 */
const updateVisualizationForStep = (step: number) => {
  if (!algorithm || clusterHistory.value.length === 0) return;
  
  const historyItem = clusterHistory.value[step];
  const dataset = moduleStore.dataset;
  
  if (!dataset) return;
  
  // Create scatter series for this step
  const series: ScatterSeries[] = [];
  const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];
  
  // Group points by cluster
  for (let c = 0; c < parameters.value.k; c++) {
    const clusterPoints = dataset.filter((_, i) => historyItem.assignments[i] === c);
    const x = clusterPoints.map(p => p.features[0]);
    const y = clusterPoints.map(p => p.features[1]);
    
    series.push({
      x,
      y,
      name: `Cluster ${c}`,
      color: colors[c % colors.length],
      mode: 'markers',
      marker: {
        size: 8,
        opacity: 0.7
      }
    });
  }
  
  // Add centroids
  series.push({
    x: historyItem.centroids.map((c: number[]) => c[0]),
    y: historyItem.centroids.map((c: number[]) => c[1]),
    name: 'Centroids',
    color: '#000000',
    mode: 'markers',
    marker: {
      size: 15,
      symbol: 'x',
      line: { width: 2, color: '#ffffff' }
    }
  });
  
  scatterSeries.value = series;
  
  // Update inertia curve up to current step
  const epochs = clusterHistory.value.slice(0, step + 1).map(h => h.iteration);
  const inertias = clusterHistory.value.slice(0, step + 1).map(h => h.inertia);
  
  inertiaSeries.value = [{
    x: epochs,
    y: inertias,
    name: 'Inertia',
    color: '#10b981',
    width: 2,
  }];
};

/**
 * Update visualization for final results
 */
const updateVisualizationForFinal = () => {
  if (!algorithm) return;
  
  const visualizations = algorithm.getVisualizationData();
  
  // Process visualizations
  if (visualizations.length > 0) {
    // Scatter plot with clusters
    const scatterViz = visualizations[0];
    if ((scatterViz.type === 'scatter2d' || scatterViz.type === 'scatter3d') && scatterViz.data) {
      const series: ScatterSeries[] = [];
      
      // Add cluster points
      for (let c = 0; c < parameters.value.k; c++) {
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
      
      // Add centroids
      if (scatterViz.data.centroids) {
        series.push({
          x: scatterViz.data.centroids.x,
          y: scatterViz.data.centroids.y,
          name: scatterViz.data.centroids.name || 'Centroids',
          color: scatterViz.data.centroids.marker?.color || '#000000',
          mode: 'markers',
          marker: {
            size: scatterViz.data.centroids.marker?.size || 15,
            symbol: scatterViz.data.centroids.marker?.symbol || 'x',
          }
        });
      }
      
      scatterSeries.value = series;
    }
    
    // Inertia curve
    if (visualizations.length > 1) {
      const inertiaViz = visualizations[1];
      if (inertiaViz.type === 'line' && inertiaViz.data) {
        inertiaSeries.value = [{
          x: inertiaViz.data.x,
          y: inertiaViz.data.y,
          name: inertiaViz.data.name || 'Inertia',
          color: inertiaViz.data.line?.color || '#10b981',
          width: inertiaViz.data.line?.width || 2,
        }];
      }
    }
  }
};

/**
 * Previous step
 */
const previousStep = () => {
  if (currentStep.value > 0) {
    currentStep.value--;
  }
};

/**
 * Next step
 */
const nextStep = () => {
  if (currentStep.value < clusterHistory.value.length - 1) {
    currentStep.value++;
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
    
    // Update step-by-step mode if changed
    if (bookmarkParameters.stepByStep !== undefined) {
      stepByStepMode.value = bookmarkParameters.stepByStep;
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
