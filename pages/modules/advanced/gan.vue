<template>
  <ModuleLayout
    title="Generative Adversarial Networks"
    description="Learn how generator and discriminator networks compete to create realistic synthetic data"
    difficulty="advanced"
    :estimated-time="60"
    module-id="gan"
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
        loading-message="Training GAN..."
        :error="error"
        :on-retry="handleRetry"
      >
        <!-- Controls -->
        <template #controls>
          <ControlPanel
            :controls="controls"
            :values="parameters"
            :loading="isTraining"
            submit-label="Train GAN"
            @change="handleParameterChange"
            @reset="handleReset"
            @submit="handleTrain"
          />

          <!-- Architecture Display -->
          <div class="mt-6 p-4 bg-pink-50 rounded-lg border border-pink-200">
            <h3 class="text-sm font-semibold text-pink-700 mb-3">GAN Architecture</h3>
            <div class="space-y-3">
              <div>
                <div class="text-xs text-pink-600 font-medium mb-1">Generator</div>
                <div class="text-sm text-pink-900">
                  Input: {{ parameters.latentDim }}D noise → 
                  {{ parameters.generatorLayers.join(' → ') }} → 
                  Output: {{ outputDimension }}D
                </div>
              </div>
              <div>
                <div class="text-xs text-pink-600 font-medium mb-1">Discriminator</div>
                <div class="text-sm text-pink-900">
                  Input: {{ outputDimension }}D → 
                  {{ parameters.discriminatorLayers.join(' → ') }} → 
                  Output: 1 (real/fake)
                </div>
              </div>
            </div>
          </div>

          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Training Results</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Generator Loss</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.generatorLoss?.toFixed(4) }}
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Discriminator Loss</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.discriminatorLoss?.toFixed(4) }}
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Discriminator Accuracy</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ (metrics.discriminatorAccuracy * 100).toFixed(2) }}%
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Training Time</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ trainingTime }}ms
                </div>
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <div class="grid grid-cols-2 gap-4">
              <LineChart
                v-if="generatorLossSeries && generatorLossSeries.length > 0"
                :series="generatorLossSeries"
                title="Generator Loss"
                x-label="Epoch"
                y-label="Loss"
                :show-legend="false"
                height="300px"
              />

              <LineChart
                v-if="discriminatorLossSeries && discriminatorLossSeries.length > 0"
                :series="discriminatorLossSeries"
                title="Discriminator Loss"
                x-label="Epoch"
                y-label="Loss"
                :show-legend="false"
                height="300px"
              />
            </div>

            <div class="grid grid-cols-2 gap-4">
              <Scatter2D
                v-if="realDataViz"
                :series="realDataViz"
                title="Real Data Distribution"
                x-label="Feature 1"
                y-label="Feature 2"
                :show-legend="false"
                height="350px"
              />

              <Scatter2D
                v-if="generatedDataViz"
                :series="generatedDataViz"
                title="Generated Data Distribution"
                x-label="Feature 1"
                y-label="Feature 2"
                :show-legend="false"
                height="350px"
              />
            </div>

            <Heatmap
              v-if="discriminatorScoresViz"
              :data="discriminatorScoresViz"
              title="Discriminator Decision Boundary"
              x-label="Feature 1"
              y-label="Feature 2"
              height="400px"
            />
          </div>
        </template>
      </InteractiveSection>
    </template>

    <!-- Quiz Section -->
    <template #quiz>
      <QuizLoadingState v-if="quizLoading" />
      <QuizErrorDisplay v-else-if="quizError" :error="quizError" @retry="retryQuizLoad" />
      <QuizErrorDisplay v-else-if="!quizData" :coming-soon="true" />
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
import { ref, computed, onMounted } from 'vue';
import { useModuleStore } from '~/stores/module';
import { useProgressStore } from '~/stores/progress';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { GAN } from '~/algorithms/advanced/GAN';
import LineChart from '~/components/visualizations/LineChart.vue';
import Scatter2D from '~/components/visualizations/Scatter2D.vue';
import Heatmap from '~/components/visualizations/Heatmap.vue';
import Quiz from '~/components/modules/Quiz.vue';
import QuizLoadingState from '~/components/modules/QuizLoadingState.vue';
import QuizErrorDisplay from '~/components/modules/QuizErrorDisplay.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { LineSeries } from '~/components/visualizations/LineChart.vue';
import type { QuizResult } from '~/types/quiz';
import { useQuizLoader } from '~/composables/useQuizLoader';


// Load content from Nuxt Content
const { data: content } = await useAsyncData(() =>
  queryCollection('content').path('/modules/advanced/gan/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const progressStore = useProgressStore();

// Parameters
const parameters = ref({
  generatorLayers: [64, 32],
  discriminatorLayers: [32, 16],
  latentDim: 10,
  learningRate: 0.0002,
  epochs: 200,
  batchSize: 32,
  activation: 'leaky_relu',
  dataset: '2d-gaussian',
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'dataset',
    type: 'select',
    label: 'Dataset',
    options: [
      { label: '2D Gaussian Mixture', value: '2d-gaussian' },
      { label: 'Two Moons', value: '2d-moons' },
      { label: 'Concentric Circles', value: '2d-circles' },
      { label: 'High-Dimensional Data', value: 'high-dimensional' },
    ],
    tooltip: 'Choose a dataset for GAN training',
    group: 'Data',
  },
  {
    id: 'latentDim',
    type: 'slider',
    label: 'Latent Dimension',
    min: 2,
    max: 50,
    step: 1,
    tooltip: 'Dimension of noise input to generator',
    group: 'Architecture',
  },
  {
    id: 'activation',
    type: 'select',
    label: 'Activation Function',
    options: [
      { label: 'Leaky ReLU', value: 'leaky_relu' },
      { label: 'ReLU', value: 'relu' },
      { label: 'Tanh', value: 'tanh' },
    ],
    tooltip: 'Activation function for hidden layers',
    group: 'Architecture',
  },
  {
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.0001,
    max: 0.01,
    step: 0.0001,
    tooltip: 'Learning rate for both networks',
    group: 'Training',
  },
  {
    id: 'epochs',
    type: 'slider',
    label: 'Epochs',
    min: 50,
    max: 500,
    step: 10,
    tooltip: 'Number of training iterations',
    group: 'Training',
  },
  {
    id: 'batchSize',
    type: 'slider',
    label: 'Batch Size',
    min: 8,
    max: 128,
    step: 8,
    tooltip: 'Number of samples per training batch',
    group: 'Training',
  },
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);
const trainingTime = ref(0);

// Results
const metrics = ref<Metrics | null>(null);
const generatorLossSeries = ref<LineSeries[]>([]);
const discriminatorLossSeries = ref<LineSeries[]>([]);
const realDataViz = ref<any>(null);
const generatedDataViz = ref<any>(null);
const discriminatorScoresViz = ref<any>(null);

// Quiz data
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

// Algorithm instance
let algorithm: GAN | null = null;

// Computed
const outputDimension = computed(() => {
  const dataset = moduleStore.dataset;
  return dataset && dataset.length > 0 ? dataset[0].features.length : 2;
});

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
  moduleStore.setCurrentModule('gan');
  moduleStore.setParameters(parameters.value);

  // Load default dataset
  await loadInitialDataset();

  // Load quiz data
  await loadQuiz('gan', '/content/modules/advanced/gan/quiz.json');
});

/**
 * Generate synthetic datasets
 */
const generateSyntheticData = (datasetType: string) => {
  const n = 200;
  const data = [];

  for (let i = 0; i < n; i++) {
    let features: number[];
    let label = 0;

    if (datasetType === '2d-gaussian') {
      // Gaussian mixture
      const cluster = Math.random() > 0.5 ? 1 : 0;
      const centerX = cluster === 0 ? -2 : 2;
      const centerY = cluster === 0 ? -2 : 2;
      features = [
        centerX + (Math.random() - 0.5) * 2,
        centerY + (Math.random() - 0.5) * 2,
      ];
      label = cluster;
    } else if (datasetType === '2d-moons') {
      // Two moons
      const moon = Math.random() > 0.5 ? 1 : 0;
      const angle = Math.random() * Math.PI;
      const radius = 3;
      features = [
        radius * Math.cos(angle) + (Math.random() - 0.5) * 0.5,
        radius * Math.sin(angle) + (Math.random() - 0.5) * 0.5 + (moon * 2),
      ];
      if (moon === 1) features[0] = -features[0];
      label = moon;
    } else if (datasetType === '2d-circles') {
      // Concentric circles
      const circle = Math.random() > 0.5 ? 1 : 0;
      const angle = Math.random() * 2 * Math.PI;
      const radius = circle === 0 ? 2 : 4;
      features = [
        radius * Math.cos(angle) + (Math.random() - 0.5) * 0.3,
        radius * Math.sin(angle) + (Math.random() - 0.5) * 0.3,
      ];
      label = circle;
    } else {
      // High-dimensional
      const cluster = Math.random() > 0.5 ? 1 : 0;
      features = Array(10).fill(0).map(() => 
        cluster * 2 + (Math.random() - 0.5) * 1.5
      );
      label = cluster;
    }

    data.push({
      features,
      label,
      id: `point-${i}`,
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
    generatorLossSeries.value = [];
    discriminatorLossSeries.value = [];
    realDataViz.value = null;
    generatedDataViz.value = null;
    discriminatorScoresViz.value = null;
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    generatorLayers: [64, 32],
    discriminatorLayers: [32, 16],
    latentDim: 10,
    learningRate: 0.0002,
    epochs: 200,
    batchSize: 32,
    activation: 'leaky_relu',
    dataset: '2d-gaussian',
  };
  moduleStore.setParameters(parameters.value);

  await loadInitialDataset();

  metrics.value = null;
  generatorLossSeries.value = [];
  discriminatorLossSeries.value = [];
  realDataViz.value = null;
  generatedDataViz.value = null;
  discriminatorScoresViz.value = null;
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

    // Create and train algorithm
    algorithm = new GAN({
      generatorLayers: parameters.value.generatorLayers,
      discriminatorLayers: parameters.value.discriminatorLayers,
      latentDim: parameters.value.latentDim,
      learningRate: parameters.value.learningRate,
      epochs: parameters.value.epochs,
      batchSize: parameters.value.batchSize,
      activation: parameters.value.activation as 'leaky_relu' | 'relu' | 'tanh',
    });

    // Train
    const startTime = Date.now();
    await algorithm.fit(X);
    const endTime = Date.now();
    trainingTime.value = endTime - startTime;

    console.log(`GAN training completed in ${trainingTime.value}ms`);

    // Get results
    metrics.value = algorithm.getMetrics();
    const visualizations = algorithm.getVisualizationData();

    // Process visualizations
    if (Array.isArray(visualizations)) {
      for (const viz of visualizations) {
        if (viz.type === 'line') {
          if (viz.data?.generatorLoss) {
            generatorLossSeries.value = [{
              x: viz.data.generatorLoss.x,
              y: viz.data.generatorLoss.y,
              name: 'Generator Loss',
              color: '#ec4899',
              width: 2,
            }];
          }
          if (viz.data?.discriminatorLoss) {
            discriminatorLossSeries.value = [{
              x: viz.data.discriminatorLoss.x,
              y: viz.data.discriminatorLoss.y,
              name: 'Discriminator Loss',
              color: '#8b5cf6',
              width: 2,
            }];
          }
        } else if (viz.type === 'scatter') {
          if (viz.data?.realData) {
            realDataViz.value = [{
              x: viz.data.realData.x,
              y: viz.data.realData.y,
              name: viz.data.realData.name || 'Real Data',
              color: '#10b981',
              mode: 'markers',
              marker: {
                size: 6,
                opacity: 0.7,
              },
            }];
          }
          if (viz.data?.generatedData) {
            generatedDataViz.value = [{
              x: viz.data.generatedData.x,
              y: viz.data.generatedData.y,
              name: viz.data.generatedData.name || 'Generated Data',
              color: '#f59e0b',
              mode: 'markers',
              marker: {
                size: 6,
                opacity: 0.7,
              },
            }];
          }
        } else if (viz.type === 'heatmap' && viz.data) {
          discriminatorScoresViz.value = viz.data;
        }
      }
    }

    trainingProgress.value = 1;
  } catch (err) {
    console.error('GAN training error:', err);
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
    generatorLossSeries.value = [];
    discriminatorLossSeries.value = [];
    realDataViz.value = null;
    generatedDataViz.value = null;
    discriminatorScoresViz.value = null;
  } catch (err) {
    console.error('Error loading bookmark:', err);
    error.value = 'Failed to load bookmark configuration';
  }
};
</script>
