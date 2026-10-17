<template>
  <ModuleLayout
    title="Autoencoders"
    description="Learn how autoencoders compress and reconstruct data through encoder-decoder architectures"
    difficulty="advanced"
    :estimated-time="50"
    module-id="autoencoders"
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
        loading-message="Training autoencoder..."
        :error="error"
        :on-retry="handleRetry"
      >
        <!-- Controls -->
        <template #controls>
          <ControlPanel
            :controls="controls"
            :values="parameters"
            :loading="isTraining"
            submit-label="Train Autoencoder"
            @change="handleParameterChange"
            @reset="handleReset"
            @submit="handleTrain"
          />

          <!-- Architecture Display -->
          <div class="mt-6 p-4 bg-purple-50 rounded-lg border border-purple-200">
            <h3 class="text-sm font-semibold text-purple-700 mb-3">Autoencoder Architecture</h3>
            <div class="space-y-2">
              <div class="flex items-center gap-2">
                <span class="text-xs text-purple-600 w-32">Input Dimension:</span>
                <span class="text-sm font-semibold text-purple-900">{{ inputDimension }}</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-purple-600 w-32">Encoder Layers:</span>
                <span class="text-sm font-semibold text-purple-900">
                  {{ parameters.encoderLayers.join(' → ') }}
                </span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-purple-600 w-32">Latent Dimension:</span>
                <span class="text-sm font-semibold text-purple-900">{{ latentDimension }}</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-purple-600 w-32">Decoder Layers:</span>
                <span class="text-sm font-semibold text-purple-900">
                  {{ decoderLayers.join(' → ') }}
                </span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-purple-600 w-32">Compression Ratio:</span>
                <span class="text-sm font-semibold text-purple-900">{{ compressionRatio }}x</span>
              </div>
            </div>
          </div>

          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Training Results</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Reconstruction Loss</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.finalLoss?.toFixed(4) }}
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Compression Ratio</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.compressionRatio?.toFixed(2) }}x
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Latent Dimension</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.latentDimension }}
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
            <LineChart
              v-if="lossSeries && lossSeries.length > 0"
              :series="lossSeries"
              title="Reconstruction Loss"
              x-label="Epoch"
              y-label="Loss"
              :show-legend="false"
              height="300px"
            />

            <div class="grid grid-cols-2 gap-4">
              <Scatter2D
                v-if="latentSpaceData"
                :series="latentSpaceData"
                title="Latent Space Representation"
                x-label="Latent Dimension 1"
                y-label="Latent Dimension 2"
                :show-legend="true"
                height="350px"
              />

              <Heatmap
                v-if="reconstructionComparison"
                :data="reconstructionComparison"
                title="Reconstruction Quality"
                x-label="Feature"
                y-label="Sample"
                height="350px"
              />
            </div>

            <NetworkDiagram
              v-if="networkArchitecture.length > 0"
              :architecture="networkArchitecture"
              :weights="networkWeights"
              :activations="networkActivations"
              :width="800"
              :height="300"
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
import { Autoencoder } from '~/algorithms/advanced/Autoencoder';
import LineChart from '~/components/visualizations/LineChart.vue';
import Scatter2D from '~/components/visualizations/Scatter2D.vue';
import Heatmap from '~/components/visualizations/Heatmap.vue';
import NetworkDiagram from '~/components/visualizations/NetworkDiagram.vue';
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
  queryCollection('content').path('/modules/advanced/autoencoders/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const progressStore = useProgressStore();

// Parameters
const parameters = ref({
  encoderLayers: [8, 4, 2],
  learningRate: 0.01,
  epochs: 200,
  activation: 'relu',
  dataset: 'high-dimensional',
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'dataset',
    type: 'select',
    label: 'Dataset',
    options: [
      { label: 'High-Dimensional Data', value: 'high-dimensional' },
      { label: 'MNIST Digits', value: 'mnist-digits' },
      { label: 'Synthetic Manifold', value: 'synthetic-manifold' },
    ],
    tooltip: 'Choose a dataset for autoencoder training',
    group: 'Data',
  },
  {
    id: 'activation',
    type: 'select',
    label: 'Activation Function',
    options: [
      { label: 'ReLU', value: 'relu' },
      { label: 'Sigmoid', value: 'sigmoid' },
      { label: 'Tanh', value: 'tanh' },
    ],
    tooltip: 'Activation function for hidden layers',
    group: 'Architecture',
  },
  {
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.001,
    max: 0.1,
    step: 0.001,
    tooltip: 'Step size for gradient descent',
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
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);
const trainingTime = ref(0);

// Results
const metrics = ref<Metrics | null>(null);
const lossSeries = ref<LineSeries[]>([]);
const latentSpaceData = ref<any>(null);
const reconstructionComparison = ref<any>(null);
const networkArchitecture = ref<number[]>([]);
const networkWeights = ref<number[][][]>([]);
const networkActivations = ref<number[][]>([]);

// Quiz data
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

// Algorithm instance
let algorithm: Autoencoder | null = null;

// Computed
const inputDimension = computed(() => {
  const dataset = moduleStore.dataset;
  return dataset && dataset.length > 0 ? dataset[0].features.length : 10;
});

const latentDimension = computed(() => {
  return parameters.value.encoderLayers[parameters.value.encoderLayers.length - 1];
});

const decoderLayers = computed(() => {
  return [...parameters.value.encoderLayers].reverse().slice(1);
});

const compressionRatio = computed(() => {
  return (inputDimension.value / latentDimension.value).toFixed(2);
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
  moduleStore.setCurrentModule('autoencoders');
  moduleStore.setParameters(parameters.value);

  // Load default dataset
  await loadInitialDataset();

  // Load quiz data
  await loadQuiz('autoencoders', '/content/modules/advanced/autoencoders/quiz.json');
});

/**
 * Generate synthetic datasets
 */
const generateSyntheticData = (datasetType: string) => {
  const n = 100;
  const data = [];

  for (let i = 0; i < n; i++) {
    let features: number[];
    let label = 0;

    if (datasetType === 'high-dimensional') {
      // High-dimensional data with underlying structure
      const latent1 = Math.random() * 2 - 1;
      const latent2 = Math.random() * 2 - 1;
      features = [
        latent1 + (Math.random() - 0.5) * 0.1,
        latent2 + (Math.random() - 0.5) * 0.1,
        latent1 * latent2 + (Math.random() - 0.5) * 0.1,
        Math.sin(latent1) + (Math.random() - 0.5) * 0.1,
        Math.cos(latent2) + (Math.random() - 0.5) * 0.1,
        latent1 ** 2 + (Math.random() - 0.5) * 0.1,
        latent2 ** 2 + (Math.random() - 0.5) * 0.1,
        latent1 + latent2 + (Math.random() - 0.5) * 0.1,
        latent1 - latent2 + (Math.random() - 0.5) * 0.1,
        (latent1 + latent2) / 2 + (Math.random() - 0.5) * 0.1,
      ];
      label = latent1 > 0 ? 1 : 0;
    } else if (datasetType === 'mnist-digits') {
      // Simplified MNIST-like data (8x8 = 64 features)
      const digit = Math.floor(Math.random() * 10);
      features = Array(64).fill(0).map(() => Math.random() * (digit / 10));
      label = digit;
    } else {
      // Synthetic manifold
      const angle = Math.random() * 2 * Math.PI;
      const radius = Math.random() * 2;
      features = [
        radius * Math.cos(angle),
        radius * Math.sin(angle),
        radius * Math.cos(2 * angle) + (Math.random() - 0.5) * 0.2,
        radius * Math.sin(2 * angle) + (Math.random() - 0.5) * 0.2,
        radius + (Math.random() - 0.5) * 0.2,
        angle / (2 * Math.PI) + (Math.random() - 0.5) * 0.1,
        Math.sin(angle) * radius + (Math.random() - 0.5) * 0.2,
        Math.cos(angle) * radius + (Math.random() - 0.5) * 0.2,
      ];
      label = radius < 1 ? 0 : 1;
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
    lossSeries.value = [];
    latentSpaceData.value = null;
    reconstructionComparison.value = null;
    networkArchitecture.value = [];
    networkWeights.value = [];
    networkActivations.value = [];
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    encoderLayers: [8, 4, 2],
    learningRate: 0.01,
    epochs: 200,
    activation: 'relu',
    dataset: 'high-dimensional',
  };
  moduleStore.setParameters(parameters.value);

  await loadInitialDataset();

  metrics.value = null;
  lossSeries.value = [];
  latentSpaceData.value = null;
  reconstructionComparison.value = null;
  networkArchitecture.value = [];
  networkWeights.value = [];
  networkActivations.value = [];
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
    algorithm = new Autoencoder({
      encoderLayers: parameters.value.encoderLayers,
      learningRate: parameters.value.learningRate,
      epochs: parameters.value.epochs,
      activation: parameters.value.activation as 'relu' | 'sigmoid' | 'tanh',
    });

    // Train
    const startTime = Date.now();
    await algorithm.fit(X);
    const endTime = Date.now();
    trainingTime.value = endTime - startTime;

    console.log(`Autoencoder training completed in ${trainingTime.value}ms`);

    // Get results
    metrics.value = algorithm.getMetrics();
    const visualizations = algorithm.getVisualizationData();

    // Process visualizations
    if (Array.isArray(visualizations)) {
      for (const viz of visualizations) {
        if (viz.type === 'line' && viz.data?.loss) {
          lossSeries.value = [{
            x: viz.data.loss.x,
            y: viz.data.loss.y,
            name: viz.data.loss.name || 'Loss',
            color: viz.data.loss.line?.color || '#8b5cf6',
            width: viz.data.loss.line?.width || 2,
          }];
        } else if (viz.type === 'scatter' && viz.data?.latent) {
          // Convert Plotly trace format to Scatter2D series format
          const latent = viz.data.latent;
          latentSpaceData.value = [{
            x: latent.x || [],
            y: latent.y || [],
            name: latent.name || 'Latent Space',
            color: Array.isArray(latent.marker?.color) ? undefined : latent.marker?.color,
            mode: 'markers' as const,
            marker: {
              size: latent.marker?.size || 8,
              opacity: 0.8,
            },
          }];
        } else if (viz.type === 'heatmap' && viz.data) {
          // Extract the reconstructed data array for the heatmap
          const reconstructedData = viz.data.reconstructed?.z || viz.data.original?.z;
          if (reconstructedData && Array.isArray(reconstructedData)) {
            reconstructionComparison.value = reconstructedData;
          }
        }
      }
    }

    // Get network architecture
    const inputSize = X[0].length;
    const encoderArch = [inputSize, ...parameters.value.encoderLayers];
    const decoderArch = [...decoderLayers.value, inputSize];
    networkArchitecture.value = [...encoderArch, ...decoderArch];

    trainingProgress.value = 1;
  } catch (err) {
    console.error('Autoencoder training error:', err);
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
    lossSeries.value = [];
    latentSpaceData.value = null;
    reconstructionComparison.value = null;
    networkArchitecture.value = [];
    networkWeights.value = [];
    networkActivations.value = [];
  } catch (err) {
    console.error('Error loading bookmark:', err);
    error.value = 'Failed to load bookmark configuration';
  }
};
</script>
