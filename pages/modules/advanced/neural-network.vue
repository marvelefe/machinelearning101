<template>
  <ModuleLayout
    title="Neural Network Visualization"
    description="Learn how neural networks learn through forward propagation and backpropagation"
    difficulty="advanced"
    :estimated-time="45"
    module-id="neural-network"
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
      <InteractiveSection :loading="isTraining" :progress="trainingProgress"
        loading-message="Training neural network..." :error="error" :on-retry="handleRetry">
        <!-- Controls -->
        <template #controls>
          <ControlPanel :controls="controls" :values="parameters" :loading="isTraining" submit-label="Train Network"
            @change="handleParameterChange" @reset="handleReset" @submit="handleTrain" />

          <!-- Architecture Builder -->
          <div class="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
            <h3 class="text-sm font-semibold text-blue-700 mb-3">Network Architecture</h3>
            <div class="space-y-2">
              <div class="flex items-center gap-2">
                <span class="text-xs text-blue-600 w-24">Input Layer:</span>
                <span class="text-sm font-semibold text-blue-900">{{ inputSize }} neurons</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-blue-600 w-24">Hidden Layers:</span>
                <span class="text-sm font-semibold text-blue-900">
                  {{ parameters.hiddenLayers.join(' → ') }}
                </span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-blue-600 w-24">Output Layer:</span>
                <span class="text-sm font-semibold text-blue-900">1 neuron</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-blue-600 w-24">Total Params:</span>
                <span class="text-sm font-semibold text-blue-900">{{ totalParameters }}</span>
              </div>
            </div>
          </div>

          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Training Results</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Accuracy</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ (metrics.accuracy * 100).toFixed(2) }}%
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Final Loss</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.finalLoss?.toFixed(4) }}
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Precision</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ (metrics.precision * 100).toFixed(2) }}%
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Recall</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ (metrics.recall * 100).toFixed(2) }}%
                </div>
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <NetworkDiagram v-if="networkArchitecture.length > 0" :architecture="networkArchitecture"
              :weights="networkWeights" :activations="networkActivations" :width="800" :height="400" />

            <div class="grid grid-cols-2 gap-4">
              <LineChart v-if="lossSeries && lossSeries.length > 0" :series="lossSeries" title="Training Loss"
                x-label="Epoch" y-label="Loss" :show-legend="false" height="300px" />

              <LineChart v-if="accuracySeries && accuracySeries.length > 0" :series="accuracySeries"
                title="Training Accuracy" x-label="Epoch" y-label="Accuracy" :show-legend="false" height="300px" />
            </div>

            <ConfusionMatrix v-if="metrics && metrics.confusionMatrix" :confusionMatrix="metrics.confusionMatrix"
              :classLabels="['Class 0', 'Class 1']" title="Confusion Matrix" />
          </div>
        </template>
      </InteractiveSection>
    </template>

    <!-- Quiz Section -->
    <template #quiz>
      <!-- Loading state -->
      <QuizLoadingState v-if="quizLoading" />

      <!-- Error state with retry -->
      <QuizErrorDisplay v-else-if="quizError" :error="quizError" @retry="retryQuizLoad" />

      <!-- Coming soon state -->
      <QuizErrorDisplay v-else-if="!quizData" :coming-soon="true" />

      <!-- Quiz component -->
      <Quiz v-else :quiz="quizData" :passing-score="quizData.passingScore || 70"
        @quiz-completed="handleQuizComplete" />
    </template>
  </ModuleLayout>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { useModuleStore } from '~/stores/module';
import { useProgressStore } from '~/stores/progress';
import { useDataset } from '~/composables/useDataset';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { NeuralNetwork } from '~/algorithms/advanced/NeuralNetwork';
import NetworkDiagram from '~/components/visualizations/NetworkDiagram.vue';
import LineChart from '~/components/visualizations/LineChart.vue';
import ConfusionMatrix from '~/components/visualizations/ConfusionMatrix.vue';
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
  queryCollection('content').path('/modules/advanced/neural-network/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const progressStore = useProgressStore();
const { loadDataset, getDatasetsByType } = useDataset();

// Parameters
const parameters = ref({
  hiddenLayers: [4, 3],
  learningRate: 0.1,
  epochs: 100,
  activation: 'sigmoid',
  dataset: 'binary-classification',
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'dataset',
    type: 'select',
    label: 'Dataset',
    options: [
      { label: 'Binary Classification', value: 'binary-classification' },
      { label: 'XOR Problem', value: 'xor' },
      { label: 'Circles', value: 'circles' },
    ],
    tooltip: 'Choose a classification dataset',
    group: 'Data',
  },
  {
    id: 'activation',
    type: 'select',
    label: 'Activation Function',
    options: [
      { label: 'Sigmoid', value: 'sigmoid' },
      { label: 'ReLU', value: 'relu' },
      { label: 'Tanh', value: 'tanh' },
    ],
    tooltip: 'Activation function for hidden layers',
    group: 'Network Architecture',
  },
  {
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.01,
    max: 1.0,
    step: 0.01,
    tooltip: 'Step size for weight updates',
    group: 'Training Parameters',
  },
  {
    id: 'epochs',
    type: 'slider',
    label: 'Epochs',
    min: 50,
    max: 500,
    step: 50,
    tooltip: 'Number of training iterations',
    group: 'Training Parameters',
  },
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const lossSeries = ref<LineSeries[]>([]);
const accuracySeries = ref<LineSeries[]>([]);
const networkArchitecture = ref<number[]>([]);
const networkWeights = ref<number[][][]>([]);
const networkActivations = ref<number[][]>([]);

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

// Algorithm instance
let algorithm: NeuralNetwork | null = null;

// Computed
const inputSize = computed(() => {
  const dataset = moduleStore.dataset;
  return dataset && dataset.length > 0 ? dataset[0].features.length : 2;
});

const totalParameters = computed(() => {
  if (networkArchitecture.value.length === 0) return 0;

  let total = 0;
  for (let i = 1; i < networkArchitecture.value.length; i++) {
    const prevSize = networkArchitecture.value[i - 1];
    const currSize = networkArchitecture.value[i];
    total += prevSize * currSize + currSize; // weights + biases
  }
  return total;
});

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
  moduleStore.setCurrentModule('neural-network');
  moduleStore.setParameters(parameters.value);

  // Load default dataset
  await loadInitialDataset();

  // Load quiz data
  await loadQuiz('neural-network', '/content/modules/advanced/neural-network/quiz.json');
});

/**
 * Generate synthetic datasets
 */
const generateSyntheticData = (datasetType: string) => {
  const n = 100;
  const data = [];

  for (let i = 0; i < n; i++) {
    let x1, x2, label;

    if (datasetType === 'xor') {
      // XOR problem
      x1 = Math.random() > 0.5 ? 1 : 0;
      x2 = Math.random() > 0.5 ? 1 : 0;
      label = (x1 + x2) % 2; // XOR
      // Add noise
      x1 += (Math.random() - 0.5) * 0.2;
      x2 += (Math.random() - 0.5) * 0.2;
    } else if (datasetType === 'circles') {
      // Concentric circles
      const angle = Math.random() * 2 * Math.PI;
      const radius = Math.random() * 2;
      x1 = radius * Math.cos(angle);
      x2 = radius * Math.sin(angle);
      label = radius < 1 ? 0 : 1;
    } else {
      // Simple binary classification
      const cluster = Math.random() > 0.5 ? 1 : 0;
      x1 = cluster * 2 + (Math.random() - 0.5) * 1.5;
      x2 = cluster * 2 + (Math.random() - 0.5) * 1.5;
      label = cluster;
    }

    data.push({
      features: [x1, x2],
      label,
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
    accuracySeries.value = [];
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
    hiddenLayers: [4, 3],
    learningRate: 0.1,
    epochs: 100,
    activation: 'sigmoid',
    dataset: 'binary-classification',
  };
  moduleStore.setParameters(parameters.value);

  await loadInitialDataset();

  metrics.value = null;
  lossSeries.value = [];
  accuracySeries.value = [];
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
    const y = dataset.map(d => d.label as number);

    // Create and train algorithm
    algorithm = new NeuralNetwork({
      hiddenLayers: parameters.value.hiddenLayers,
      learningRate: parameters.value.learningRate,
      epochs: parameters.value.epochs,
      activation: parameters.value.activation as 'sigmoid' | 'relu' | 'tanh'
    });

    // Train
    const startTime = Date.now();
    await algorithm.fit(X, y);
    const endTime = Date.now();

    console.log(`Neural network training completed in ${endTime - startTime}ms`);

    // Get results
    metrics.value = algorithm.getMetrics();
    const visualizations = algorithm.getVisualizationData();

    // Get network state
    const state = algorithm.getNetworkState();
    networkArchitecture.value = [X[0].length, ...parameters.value.hiddenLayers, 1];
    networkWeights.value = state.weights;
    networkActivations.value = state.activations.map(la => la.activations);

    // Process visualizations
    if (visualizations.length > 0) {
      // Loss curve
      const lossViz = visualizations[0];
      if (lossViz.type === 'line' && lossViz.data?.loss) {
        lossSeries.value = [{
          x: lossViz.data.loss.x,
          y: lossViz.data.loss.y,
          name: lossViz.data.loss.name || 'Loss',
          color: lossViz.data.loss.line?.color || '#ef4444',
          width: lossViz.data.loss.line?.width || 2,
        }];
      }

      // Accuracy curve
      if (visualizations.length > 1) {
        const accViz = visualizations[1];
        if (accViz.type === 'line' && accViz.data?.accuracy) {
          accuracySeries.value = [{
            x: accViz.data.accuracy.x,
            y: accViz.data.accuracy.y,
            name: accViz.data.accuracy.name || 'Accuracy',
            color: accViz.data.accuracy.line?.color || '#10b981',
            width: accViz.data.accuracy.line?.width || 2,
          }];
        }
      }
    }

    trainingProgress.value = 1;
  } catch (err) {
    console.error('Neural network training error:', err);
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
    accuracySeries.value = [];
    networkArchitecture.value = [];
    networkWeights.value = [];
    networkActivations.value = [];
  } catch (err) {
    console.error('Error loading bookmark:', err);
    error.value = 'Failed to load bookmark configuration';
  }
};

</script>
