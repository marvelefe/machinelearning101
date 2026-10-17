<template>
  <ModuleLayout title="Convolutional Neural Networks"
    description="Learn how CNNs use convolutional layers, pooling, and feature maps to process images"
    difficulty="advanced" :estimated-time="45" module-id="cnn"
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
      <InteractiveSection :loading="isTraining" :progress="trainingProgress" loading-message="Training CNN..."
        :error="error" :on-retry="handleRetry">
        <!-- Controls -->
        <template #controls>
          <ControlPanel :controls="controls" :values="parameters" :loading="isTraining" submit-label="Train CNN"
            @change="handleParameterChange" @reset="handleReset" @submit="handleTrain" />

          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Model Performance</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Accuracy</div>
                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.accuracy || 0) * 100).toFixed(2) }}%</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Precision</div>
                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.precision || 0) * 100).toFixed(2) }}%
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Recall</div>
                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.recall || 0) * 100).toFixed(2) }}%</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">F1 Score</div>
                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.f1Score || 0) * 100).toFixed(2) }}%</div>
              </div>
            </div>
            <div v-if="metrics.architecture" class="mt-3 pt-3 border-t border-gray-200">
              <div class="text-xs text-gray-500">Architecture</div>
              <div class="text-sm font-mono text-gray-700">{{ metrics.architecture }}</div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-6">
            <!-- CNN Visualizations -->
            <ConvolutionVisualizer 
              v-if="filters.length > 0 || featureMaps.length > 0 || pooledMaps.length > 0"
              :filters="filters"
              :feature-maps="featureMaps"
              :pooled-maps="pooledMaps"
              height="800px"
            />

            <!-- Training curves -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <LineChart v-if="lossSeries && lossSeries.length > 0" :series="lossSeries" title="Training Loss"
                x-label="Epoch" y-label="Loss" :show-legend="false" height="300px" />

              <LineChart v-if="accuracySeries && accuracySeries.length > 0" :series="accuracySeries"
                title="Training Accuracy" x-label="Epoch" y-label="Accuracy" :show-legend="false" height="300px" />
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
import { ref, onMounted } from 'vue';
import { useModuleStore } from '~/stores/module';
import { useDataset } from '~/composables/useDataset';
import { useModules } from '~/composables/useModules';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { CNN } from '~/algorithms/computer-vision/CNN';
import LineChart from '~/components/visualizations/LineChart.vue';
import ConvolutionVisualizer from '~/components/visualizations/ConvolutionVisualizer.vue';
import Quiz from '~/components/modules/Quiz.vue';
import QuizErrorDisplay from '~/components/modules/QuizErrorDisplay.vue';
import QuizLoadingState from '~/components/modules/QuizLoadingState.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { LineSeries } from '~/components/visualizations/LineChart.vue';
import type { ModuleNavItem } from '~/types/module';
import type { Quiz as QuizType, QuizResult } from '~/types/quiz';
import { useQuizLoader } from '~/composables/useQuizLoader';
import { useBookmarkRestore } from '~/composables/useBookmarkRestore';

import { useAsyncData } from '#app';

const { data: content } = await useAsyncData(() => queryCollection('content').path('/modules/computer-vision/cnn/content').first())

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const { loadDataset, getDatasetsByType } = useDataset();
const { getNextModule, getPreviousModule } = useModules();

// Navigation
const previousModule = ref<ModuleNavItem | null>(null);
const nextModule = ref<ModuleNavItem | null>(null);
const suggestedModule = ref<ModuleNavItem | null>(null);

// Parameters
const parameters = ref({
  numFilters: 4,
  filterSize: 3,
  poolSize: 2,
  learningRate: 0.01,
  epochs: 50,
  hiddenSize: 16,
  dataset: 'simple-shapes',
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'dataset',
    type: 'select',
    label: 'Dataset',
    options: [
      { label: 'Simple Shapes (8x8)', value: 'simple-shapes' },
      { label: 'Digit Patterns (8x8)', value: 'digit-patterns' },
    ],
    tooltip: 'Choose a dataset with small images for CNN training',
    group: 'Data',
  },
  {
    id: 'numFilters',
    type: 'slider',
    label: 'Number of Filters',
    min: 2,
    max: 8,
    step: 1,
    tooltip: 'Number of convolutional filters to learn. More filters can detect more diverse features.',
    group: 'CNN Architecture',
  },
  {
    id: 'filterSize',
    type: 'slider',
    label: 'Filter Size',
    min: 3,
    max: 5,
    step: 2,
    tooltip: 'Size of convolutional filters (e.g., 3x3, 5x5). Larger filters capture broader patterns.',
    group: 'CNN Architecture',
  },
  {
    id: 'poolSize',
    type: 'slider',
    label: 'Pooling Size',
    min: 2,
    max: 3,
    step: 1,
    tooltip: 'Size of max pooling window. Reduces spatial dimensions while retaining important features.',
    group: 'CNN Architecture',
  },
  {
    id: 'hiddenSize',
    type: 'slider',
    label: 'Hidden Layer Size',
    min: 8,
    max: 32,
    step: 8,
    tooltip: 'Number of neurons in the dense hidden layer after flattening.',
    group: 'CNN Architecture',
  },
  {
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.001,
    max: 0.1,
    step: 0.001,
    tooltip: 'Controls how quickly the network learns. Higher values train faster but may be unstable.',
    group: 'Training Parameters',
  },
  {
    id: 'epochs',
    type: 'slider',
    label: 'Epochs',
    min: 10,
    max: 100,
    step: 10,
    tooltip: 'Number of training iterations. More epochs allow better learning but take longer.',
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
const filters = ref<number[][][]>([]);
const featureMaps = ref<number[][][]>([]);
const pooledMaps = ref<number[][][]>([]);

// Algorithm instance
let algorithm: CNN | null = null;

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('cnn');
  moduleStore.setParameters(parameters.value);

  // Load default dataset
  await loadInitialDataset();

  // Load navigation
  previousModule.value = await getPreviousModule('cnn');
  nextModule.value = await getNextModule('cnn');

  // For now, suggested module is the same as next module
  suggestedModule.value = nextModule.value;

  // Load quiz data
  await loadQuiz('cnn', '/content/modules/computer-vision/cnn/quiz.json');
});

/**
 * Load initial dataset
 */
const loadInitialDataset = async () => {
  // For CNN, we'll use a simple synthetic dataset
  // In a real implementation, this would load actual image data
  const dataset = generateSimpleShapesDataset();
  moduleStore.loadDataset(dataset);
};

/**
 * Generate simple shapes dataset for CNN demo
 */
const generateSimpleShapesDataset = () => {
  const data = [];
  const size = 8; // 8x8 images
  const numSamples = 100;
  
  for (let i = 0; i < numSamples; i++) {
    const classLabel = i % 3; // 3 classes
    const features = new Array(size * size).fill(0);
    
    // Generate simple patterns based on class
    if (classLabel === 0) {
      // Horizontal line
      for (let j = 3 * size; j < 4 * size; j++) {
        features[j] = 1;
      }
    } else if (classLabel === 1) {
      // Vertical line
      for (let j = 0; j < size; j++) {
        features[j * size + 3] = 1;
      }
    } else {
      // Diagonal line
      for (let j = 0; j < size; j++) {
        features[j * size + j] = 1;
      }
    }
    
    // Add some noise
    for (let j = 0; j < features.length; j++) {
      if (Math.random() < 0.1) {
        features[j] = Math.random();
      }
    }
    
    data.push({
      features,
      label: classLabel,
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
    lossSeries.value = [];
    accuracySeries.value = [];
    filters.value = [];
    featureMaps.value = [];
    pooledMaps.value = [];
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    numFilters: 4,
    filterSize: 3,
    poolSize: 2,
    learningRate: 0.01,
    epochs: 50,
    hiddenSize: 16,
    dataset: 'simple-shapes',
  };
  moduleStore.setParameters(parameters.value);

  await loadInitialDataset();

  metrics.value = null;
  lossSeries.value = [];
  accuracySeries.value = [];
  filters.value = [];
  featureMaps.value = [];
  pooledMaps.value = [];
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

    // Validate multi-class labels
    const uniqueLabels = [...new Set(y)];
    const numClasses = Math.max(...y) + 1;
    
    if (numClasses < 2) {
      throw new Error('Dataset must have at least 2 classes');
    }

    // Create and train algorithm
    algorithm = new CNN({
      numFilters: parameters.value.numFilters,
      filterSize: parameters.value.filterSize,
      poolSize: parameters.value.poolSize,
      learningRate: parameters.value.learningRate,
      epochs: parameters.value.epochs,
      hiddenSize: parameters.value.hiddenSize,
    });

    // Train with progress updates
    const startTime = Date.now();
    await algorithm.fit(X, y);
    const endTime = Date.now();

    console.log(`Training completed in ${endTime - startTime}ms`);

    // Get results
    metrics.value = algorithm.getMetrics();
    const visualizations = algorithm.getVisualizationData();

    // Get filters
    filters.value = algorithm.getFilters();

    // Get feature maps for first sample
    if (X.length > 0) {
      const maps = algorithm.getFeatureMaps(X[0]);
      featureMaps.value = maps.map(m => m.data);
    }

    // Process visualizations
    if (visualizations.length > 0) {
      // Find pooled maps visualization
      const pooledViz = visualizations.find(v => v.data.pooledMaps);
      if (pooledViz && pooledViz.data.pooledMaps) {
        pooledMaps.value = pooledViz.data.pooledMaps.map((pm: any) => pm.z);
      }

      // Loss curve
      const lossViz = visualizations.find(v => v.type === 'line' && v.data.name === 'Training Loss');
      if (lossViz && lossViz.data) {
        lossSeries.value = [{
          x: lossViz.data.x,
          y: lossViz.data.y,
          name: lossViz.data.name || 'Training Loss',
          color: lossViz.data.line?.color || '#10b981',
          width: lossViz.data.line?.width || 2,
        }];
      }

      // Accuracy curve
      const accuracyViz = visualizations.find(v => v.type === 'line' && v.data.name === 'Training Accuracy');
      if (accuracyViz && accuracyViz.data) {
        accuracySeries.value = [{
          x: accuracyViz.data.x,
          y: accuracyViz.data.y,
          name: accuracyViz.data.name || 'Training Accuracy',
          color: accuracyViz.data.line?.color || '#8b5cf6',
          width: accuracyViz.data.line?.width || 2,
        }];
      }
    }

    trainingProgress.value = 1;
  } catch (err) {
    console.error('Training error:', err);
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
 * Handle quiz completion
 */
const handleQuizComplete = (result: QuizResult) => {
  console.log('Quiz completed:', result);

  if (result.score >= 70) {
    console.log('Quiz passed! Module marked as complete.');
  }
};

// Bookmark restoration
// Requirements: 2.2, 2.3
const { handleLoadBookmark } = useBookmarkRestore(parameters, async () => {
  // Clear previous results
  metrics.value = null;
  lossSeries.value = [];
  accuracySeries.value = [];
  filters.value = [];
  featureMaps.value = [];
  pooledMaps.value = [];
  
  // Optionally re-train with new parameters
  // await handleTrain();
});
</script>
