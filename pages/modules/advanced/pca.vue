<template>
  <ModuleLayout
    title="Principal Component Analysis (PCA)"
    description="Learn how PCA reduces dimensionality while preserving variance in data"
    difficulty="intermediate"
    :estimated-time="35"
    module-id="pca"
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
        loading-message="Computing principal components..."
        :error="error"
        :on-retry="handleRetry"
      >
        <!-- Controls -->
        <template #controls>
          <ControlPanel
            :controls="controls"
            :values="parameters"
            :loading="isTraining"
            submit-label="Run PCA"
            @change="handleParameterChange"
            @reset="handleReset"
            @submit="handleTrain"
          />
          
          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">PCA Results</h3>
            <div class="space-y-3">
              <div>
                <div class="text-xs text-gray-500">Components</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.nComponents }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Original Features</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.nFeatures }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Cumulative Variance Explained</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ (metrics.cumulativeVariance * 100).toFixed(2) }}%
                </div>
              </div>
              <div v-if="metrics.explainedVarianceRatio">
                <div class="text-xs text-gray-500 mb-1">Variance by Component</div>
                <div class="space-y-1">
                  <div 
                    v-for="(ratio, i) in metrics.explainedVarianceRatio" 
                    :key="i"
                    class="flex items-center gap-2"
                  >
                    <span class="text-xs text-gray-600 w-16">PC{{ i + 1 }}:</span>
                    <div class="flex-1 bg-gray-200 rounded-full h-2">
                      <div 
                        class="bg-blue-500 h-2 rounded-full"
                        :style="{ width: `${ratio * 100}%` }"
                      ></div>
                    </div>
                    <span class="text-xs text-gray-600 w-12 text-right">
                      {{ (ratio * 100).toFixed(1) }}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <LineChart
              v-if="screeSeries && screeSeries.length > 0"
              :series="screeSeries"
              title="Scree Plot - Explained Variance"
              x-label="Principal Component"
              y-label="Variance Ratio"
              :show-legend="true"
              height="400px"
            />
            
            <Scatter2D
              v-if="projectionSeries && projectionSeries.length > 0"
              :series="projectionSeries"
              title="2D Projection (PC1 vs PC2)"
              x-label="First Principal Component"
              y-label="Second Principal Component"
              :show-legend="false"
              height="500px"
            />
            
            <Plot3D
              v-if="projection3D"
              :data="projection3D"
              title="3D Projection (PC1, PC2, PC3)"
              x-label="PC1"
              y-label="PC2"
              z-label="PC3"
              height="600px"
            />
          </div>
        </template>
      </InteractiveSection>
    </template>

    <!-- Quiz Section -->
    <!-- Requirements: 1.1, 3.1, 3.2, 3.5 -->
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
import { useModuleStore } from '~/stores/module';
import { useProgressStore } from '~/stores/progress';
import { useDataset } from '~/composables/useDataset';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { PCA } from '~/algorithms/advanced/PCA';
import Scatter2D from '~/components/visualizations/Scatter2D.vue';
import LineChart from '~/components/visualizations/LineChart.vue';
import Plot3D from '~/components/visualizations/Plot3D.vue';
import Quiz from '~/components/modules/Quiz.vue';
import QuizLoadingState from '~/components/modules/QuizLoadingState.vue';
import QuizErrorDisplay from '~/components/modules/QuizErrorDisplay.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { ScatterSeries } from '~/components/visualizations/Scatter2D.vue';
import type { LineSeries } from '~/components/visualizations/LineChart.vue';
import type { QuizResult } from '~/types/quiz';
import { useQuizLoader } from '~/composables/useQuizLoader';

// Load content from Nuxt Content
const { data: content } = await useAsyncData(() => 
  queryCollection('content').path('/modules/advanced/pca/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const progressStore = useProgressStore();
const { loadDataset, getDatasetsByType } = useDataset();

// Parameters
const parameters = ref({
  nComponents: 2,
  standardize: true,
  dataset: 'iris',
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'dataset',
    type: 'select',
    label: 'Dataset',
    options: [
      { label: 'Iris (4D)', value: 'iris' },
      { label: 'Wine (13D)', value: 'wine' },
      { label: 'Digits (64D)', value: 'digits' },
    ],
    tooltip: 'Choose a high-dimensional dataset',
    group: 'Data',
  },
  {
    id: 'nComponents',
    type: 'slider',
    label: 'Number of Components',
    min: 2,
    max: 10,
    step: 1,
    tooltip: 'Number of principal components to compute',
    group: 'PCA Parameters',
  },
  {
    id: 'standardize',
    type: 'toggle',
    label: 'Standardize Features',
    tooltip: 'Standardize features to zero mean and unit variance before PCA',
    group: 'PCA Parameters',
  },
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const screeSeries = ref<LineSeries[]>([]);
const projectionSeries = ref<ScatterSeries[]>([]);
const projection3D = ref<any>(null);

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

// Algorithm instance
let algorithm: PCA | null = null;

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('pca');
  moduleStore.setParameters(parameters.value);
  
  // Load default dataset
  await loadInitialDataset();
  
  // Load quiz data
  // Requirements: 1.1, 2.1, 2.4, 6.1, 6.2
  await loadQuiz('pca', '/content/modules/advanced/pca/quiz.json');
});

/**
 * Load initial dataset
 */
const loadInitialDataset = async () => {
  // For now, generate synthetic high-dimensional data
  // In a real implementation, you'd load actual datasets
  const syntheticData = generateSyntheticData(parameters.value.dataset);
  moduleStore.loadDataset(syntheticData);
};

/**
 * Generate synthetic high-dimensional data
 */
const generateSyntheticData = (datasetType: string) => {
  const n = 100; // number of samples
  let d = 4; // number of dimensions
  
  if (datasetType === 'wine') d = 13;
  if (datasetType === 'digits') d = 64;
  
  const data = [];
  
  for (let i = 0; i < n; i++) {
    const features = [];
    const cluster = Math.floor(Math.random() * 3);
    
    for (let j = 0; j < d; j++) {
      // Create some structure in the data
      const base = cluster * 2 + (j % 3);
      const noise = (Math.random() - 0.5) * 2;
      features.push(base + noise);
    }
    
    data.push({
      features,
      label: cluster,
      id: `point-${i}`
    });
  }
  
  return data;
};

/**
 * Handle parameter change
 */
const handleParameterChange = async (id: string, value: any) => {
  parameters.value[id] = value;
  moduleStore.setParameter(id, value);
  
  // Reload dataset if changed
  if (id === 'dataset') {
    await loadInitialDataset();
    // Clear previous results
    metrics.value = null;
    screeSeries.value = [];
    projectionSeries.value = [];
    projection3D.value = null;
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    nComponents: 2,
    standardize: true,
    dataset: 'iris',
  };
  moduleStore.setParameters(parameters.value);
  
  await loadInitialDataset();
  
  metrics.value = null;
  screeSeries.value = [];
  projectionSeries.value = [];
  projection3D.value = null;
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
    
    // Validate number of components
    const maxComponents = Math.min(X[0].length, X.length);
    const nComponents = Math.min(parameters.value.nComponents, maxComponents);
    
    if (nComponents !== parameters.value.nComponents) {
      console.warn(`Reduced components from ${parameters.value.nComponents} to ${nComponents}`);
    }
    
    // Create and train algorithm
    algorithm = new PCA({
      nComponents,
      standardize: parameters.value.standardize
    });
    
    // Train
    const startTime = Date.now();
    await algorithm.fit(X);
    const endTime = Date.now();
    
    console.log(`PCA completed in ${endTime - startTime}ms`);
    
    // Get results
    metrics.value = algorithm.getMetrics();
    const visualizations = algorithm.getVisualizationData();
    
    // Process visualizations
    if (visualizations.length > 0) {
      // Scree plot
      const screeViz = visualizations[0];
      if (screeViz.type === 'line' && screeViz.data) {
        const series: LineSeries[] = [];
        
        if (screeViz.data.scree) {
          series.push({
            x: screeViz.data.scree.x,
            y: screeViz.data.scree.y,
            name: screeViz.data.scree.name || 'Explained Variance',
            color: screeViz.data.scree.line?.color || '#3b82f6',
            width: screeViz.data.scree.line?.width || 2,
          });
        }
        
        if (screeViz.data.cumulative) {
          series.push({
            x: screeViz.data.cumulative.x,
            y: screeViz.data.cumulative.y,
            name: screeViz.data.cumulative.name || 'Cumulative',
            color: screeViz.data.cumulative.line?.color || '#10b981',
            width: screeViz.data.cumulative.line?.width || 2,
            dash: 'dash',
          });
        }
        
        screeSeries.value = series;
      }
      
      // 2D projection
      if (visualizations.length > 1) {
        const projViz = visualizations[1];
        if (projViz.type === 'scatter2d' && projViz.data?.projection) {
          projectionSeries.value = [{
            x: projViz.data.projection.x,
            y: projViz.data.projection.y,
            name: projViz.data.projection.name || 'Projected Data',
            color: projViz.data.projection.marker?.color || '#8b5cf6',
            mode: 'markers',
            marker: {
              size: projViz.data.projection.marker?.size || 8,
              opacity: projViz.data.projection.marker?.opacity || 0.7
            }
          }];
        }
      }
      
      // 3D projection
      if (visualizations.length > 2 && nComponents >= 3) {
        const proj3DViz = visualizations[2];
        if (proj3DViz.type === 'scatter3d' && proj3DViz.data) {
          projection3D.value = [proj3DViz.data];
        }
      }
    }
    
    trainingProgress.value = 1;
  } catch (err) {
    console.error('PCA error:', err);
    error.value = err instanceof Error ? err.message : 'PCA failed. Please try again.';
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
 * Handle loading bookmark
 * Requirements: 2.2, 2.3
 */
const handleLoadBookmark = async (bookmarkParameters: Record<string, any>) => {
  try {
    // Update parameters with bookmark values
    Object.assign(parameters.value, bookmarkParameters);
    moduleStore.setParameters(parameters.value);
    
    // Reload dataset if dataset parameter changed
    if (bookmarkParameters.dataset) {
      await loadInitialDataset();
    }
    
    // Clear previous results to indicate fresh state
    metrics.value = null;
    screeSeries.value = [];
    projectionSeries.value = [];
    projection3D.value = null;
  } catch (err) {
    console.error('Error loading bookmark:', err);
    error.value = 'Failed to load bookmark configuration';
  }
};

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
  
  // Optional: Show success message or navigate to next module
  if (result.score >= 70) {
    console.log('Quiz passed! Module marked as complete.');
  }
};
</script>
