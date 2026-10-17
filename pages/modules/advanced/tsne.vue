<template>
  <ModuleLayout
    title="t-SNE Visualization"
    description="Learn how t-SNE reveals structure in high-dimensional data through non-linear dimensionality reduction"
    difficulty="advanced"
    :estimated-time="40"
    module-id="tsne"
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
        loading-message="Running t-SNE optimization..."
        :error="error"
        :on-retry="handleRetry"
      >
        <!-- Controls -->
        <template #controls>
          <ControlPanel
            :controls="controls"
            :values="parameters"
            :loading="isTraining"
            submit-label="Run t-SNE"
            @change="handleParameterChange"
            @reset="handleReset"
            @submit="handleTrain"
          />
          
          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">t-SNE Results</h3>
            <div class="space-y-3">
              <div>
                <div class="text-xs text-gray-500">Final Cost (KL Divergence)</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.finalCost?.toFixed(4) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Iterations</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.iterations }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Perplexity</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.perplexity }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Dimensions</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.nFeatures }}D → {{ metrics.nComponents }}D
                </div>
              </div>
            </div>
          </div>
          
          <!-- Animation Controls -->
          <div v-if="animationFrames.length > 0" class="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
            <h3 class="text-sm font-semibold text-blue-700 mb-3">Animation</h3>
            <div class="space-y-3">
              <div class="flex items-center gap-2">
                <Button
                  :disabled="isAnimating"
                  @click="playAnimation"
                  size="sm"
                >
                  {{ isAnimating ? 'Playing...' : 'Play Animation' }}
                </Button>
                <Button
                  :disabled="!isAnimating"
                  @click="stopAnimation"
                  size="sm"
                  variant="secondary"
                >
                  Stop
                </Button>
              </div>
              <div v-if="isAnimating" class="text-xs text-blue-600">
                Frame {{ currentFrame + 1 }} / {{ animationFrames.length }}
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <LineChart
              v-if="lossSeries && lossSeries.length > 0"
              :series="lossSeries"
              title="Optimization Progress (KL Divergence)"
              x-label="Iteration"
              y-label="Cost"
              :show-legend="false"
              height="350px"
            />
            
            <Scatter2D
              v-if="embeddingSeries && embeddingSeries.length > 0"
              :series="embeddingSeries"
              title="t-SNE 2D Embedding"
              x-label="t-SNE Dimension 1"
              y-label="t-SNE Dimension 2"
              :show-legend="false"
              height="600px"
            />
            
            <Plot3D
              v-if="embedding3D"
              :data="embedding3D"
              title="t-SNE 3D Embedding"
              x-label="Dimension 1"
              y-label="Dimension 2"
              z-label="Dimension 3"
              height="600px"
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
import { ref, onMounted } from 'vue';
import { useModuleStore } from '~/stores/module';
import { useProgressStore } from '~/stores/progress';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { TSNE } from '~/algorithms/advanced/TSNE';
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
  queryCollection('content').path('/modules/advanced/tsne/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const progressStore = useProgressStore();

// Parameters
const parameters = ref({
  nComponents: 2,
  perplexity: 30,
  learningRate: 200,
  maxIterations: 500,
  dataset: 'mnist-sample',
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'dataset',
    type: 'select',
    label: 'Dataset',
    options: [
      { label: 'MNIST Sample (10D)', value: 'mnist-sample' },
      { label: 'Iris (4D)', value: 'iris' },
      { label: 'Swiss Roll (3D)', value: 'swiss-roll' },
    ],
    tooltip: 'Choose a dataset to visualize',
    group: 'Data',
  },
  {
    id: 'perplexity',
    type: 'slider',
    label: 'Perplexity',
    min: 5,
    max: 50,
    step: 5,
    tooltip: 'Balance between local and global structure. Higher values consider more neighbors.',
    group: 't-SNE Parameters',
  },
  {
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 10,
    max: 1000,
    step: 10,
    tooltip: 'Step size for gradient descent. Typical range: 10-1000.',
    group: 't-SNE Parameters',
  },
  {
    id: 'maxIterations',
    type: 'slider',
    label: 'Max Iterations',
    min: 250,
    max: 2000,
    step: 250,
    tooltip: 'Number of optimization iterations. More iterations = better convergence but slower.',
    group: 't-SNE Parameters',
  },
  {
    id: 'nComponents',
    type: 'select',
    label: 'Output Dimensions',
    options: [
      { label: '2D', value: 2 },
      { label: '3D', value: 3 },
    ],
    tooltip: 'Number of dimensions in the output embedding',
    group: 't-SNE Parameters',
  },
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const lossSeries = ref<LineSeries[]>([]);
const embeddingSeries = ref<ScatterSeries[]>([]);
const embedding3D = ref<any>(null);

// Animation
const animationFrames = ref<number[][][]>([]);
const currentFrame = ref(0);
const isAnimating = ref(false);
let animationInterval: any = null;

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

// Algorithm instance
let algorithm: TSNE | null = null;

/**
 * Initialize module
 */
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
  moduleStore.setCurrentModule('tsne');
  moduleStore.setParameters(parameters.value);
  
  // Load default dataset
  await loadInitialDataset();
  
  // Load quiz data
  await loadQuiz('tsne', '/content/modules/advanced/tsne/quiz.json');
});

/**
 * Generate synthetic datasets
 */
const generateSyntheticData = (datasetType: string) => {
  let n = 150;
  let d = 4;
  
  if (datasetType === 'mnist-sample') {
    n = 100;
    d = 10;
  } else if (datasetType === 'swiss-roll') {
    n = 200;
    d = 3;
  }
  
  const data = [];
  const nClusters = 3;
  
  for (let i = 0; i < n; i++) {
    const features = [];
    const cluster = Math.floor(i / (n / nClusters));
    
    if (datasetType === 'swiss-roll') {
      // Swiss roll manifold
      const t = (i / n) * 4 * Math.PI;
      const x = t * Math.cos(t);
      const y = (Math.random() - 0.5) * 20;
      const z = t * Math.sin(t);
      features.push(x, y, z);
    } else {
      // Random high-dimensional data with cluster structure
      for (let j = 0; j < d; j++) {
        const base = cluster * 5 + (j % 2) * 2;
        const noise = (Math.random() - 0.5) * 3;
        features.push(base + noise);
      }
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
 * Load initial dataset
 */
const loadInitialDataset = async () => {
  const syntheticData = generateSyntheticData(parameters.value.dataset);
  moduleStore.loadDataset(syntheticData);
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
    lossSeries.value = [];
    embeddingSeries.value = [];
    embedding3D.value = null;
    animationFrames.value = [];
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  stopAnimation();
  
  parameters.value = {
    nComponents: 2,
    perplexity: 30,
    learningRate: 200,
    maxIterations: 500,
    dataset: 'mnist-sample',
  };
  moduleStore.setParameters(parameters.value);
  
  await loadInitialDataset();
  
  metrics.value = null;
  lossSeries.value = [];
  embeddingSeries.value = [];
  embedding3D.value = null;
  animationFrames.value = [];
  error.value = null;
};

/**
 * Handle train
 */
const handleTrain = async () => {
  stopAnimation();
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
    
    // Validate perplexity
    if (parameters.value.perplexity >= X.length) {
      throw new Error(`Perplexity (${parameters.value.perplexity}) must be less than number of samples (${X.length})`);
    }
    
    // Create and train algorithm
    algorithm = new TSNE({
      nComponents: parameters.value.nComponents,
      perplexity: parameters.value.perplexity,
      learningRate: parameters.value.learningRate,
      maxIterations: parameters.value.maxIterations
    });
    
    // Train
    const startTime = Date.now();
    await algorithm.fit(X);
    const endTime = Date.now();
    
    console.log(`t-SNE completed in ${endTime - startTime}ms`);
    
    // Get results
    metrics.value = algorithm.getMetrics();
    const visualizations = algorithm.getVisualizationData();
    animationFrames.value = algorithm.getAnimationFrames();
    
    // Process visualizations
    if (visualizations.length > 0) {
      // Loss curve
      const lossViz = visualizations[0];
      if (lossViz.type === 'line' && lossViz.data?.loss) {
        lossSeries.value = [{
          x: lossViz.data.loss.x,
          y: lossViz.data.loss.y,
          name: lossViz.data.loss.name || 'KL Divergence',
          color: lossViz.data.loss.line?.color || '#ef4444',
          width: lossViz.data.loss.line?.width || 2,
        }];
      }
      
      // 2D embedding
      if (visualizations.length > 1) {
        const embViz = visualizations[1];
        if (embViz.type === 'scatter2d' && embViz.data?.embedding) {
          embeddingSeries.value = [{
            x: embViz.data.embedding.x,
            y: embViz.data.embedding.y,
            name: embViz.data.embedding.name || 't-SNE Embedding',
            color: embViz.data.embedding.marker?.color || '#8b5cf6',
            mode: 'markers',
            marker: {
              size: embViz.data.embedding.marker?.size || 8,
              opacity: embViz.data.embedding.marker?.opacity || 0.7
            }
          }];
        }
      }
      
      // 3D embedding
      if (visualizations.length > 2 && parameters.value.nComponents >= 3) {
        const emb3DViz = visualizations[2];
        if (emb3DViz.type === 'scatter3d' && emb3DViz.data) {
          embedding3D.value = [emb3DViz.data];
        }
      }
    }
    
    trainingProgress.value = 1;
  } catch (err) {
    console.error('t-SNE error:', err);
    error.value = err instanceof Error ? err.message : 't-SNE failed. Please try again.';
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
    // Stop any running animation
    stopAnimation();
    
    // Update parameters with bookmark values
    Object.assign(parameters.value, bookmarkParameters);
    moduleStore.setParameters(parameters.value);
    
    // Reload dataset if dataset parameter changed
    if (bookmarkParameters.dataset) {
      await loadInitialDataset();
    }
    
    // Clear previous results to indicate fresh state
    metrics.value = null;
    lossSeries.value = [];
    embeddingSeries.value = [];
    embedding3D.value = null;
    animationFrames.value = [];
  } catch (err) {
    console.error('Error loading bookmark:', err);
    error.value = 'Failed to load bookmark configuration';
  }
};

/**
 * Play animation
 */
const playAnimation = () => {
  if (animationFrames.value.length === 0) return;
  
  isAnimating.value = true;
  currentFrame.value = 0;
  
  animationInterval = setInterval(() => {
    currentFrame.value++;
    
    if (currentFrame.value >= animationFrames.value.length) {
      stopAnimation();
      return;
    }
    
    // Update visualization with current frame
    const frame = animationFrames.value[currentFrame.value];
    if (frame && frame.length > 0) {
      const x = frame.map(point => point[0]);
      const y = frame.map(point => point[1]);
      
      embeddingSeries.value = [{
        x,
        y,
        name: `Iteration ${currentFrame.value}`,
        color: '#8b5cf6',
        mode: 'markers',
        marker: {
          size: 8,
          opacity: 0.7
        }
      }];
    }
  }, 500); // 500ms per frame
};

/**
 * Stop animation
 */
const stopAnimation = () => {
  if (animationInterval) {
    clearInterval(animationInterval);
    animationInterval = null;
  }
  isAnimating.value = false;
  
  // Reset to final frame
  if (algorithm && animationFrames.value.length > 0) {
    const finalFrame = animationFrames.value[animationFrames.value.length - 1];
    if (finalFrame && finalFrame.length > 0) {
      const x = finalFrame.map(point => point[0]);
      const y = finalFrame.map(point => point[1]);
      
      embeddingSeries.value = [{
        x,
        y,
        name: 't-SNE Embedding',
        color: '#8b5cf6',
        mode: 'markers',
        marker: {
          size: 8,
          opacity: 0.7
        }
      }];
    }
  }
};

</script>
