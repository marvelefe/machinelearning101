<template>
  <ModuleLayout title="Transfer Learning"
    description="Learn how to leverage pre-trained models for new tasks with feature extraction and fine-tuning"
    difficulty="advanced" :estimated-time="45" module-id="transfer-learning"
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
      <InteractiveSection :loading="isTraining" :progress="trainingProgress" loading-message="Training model..."
        :error="error" :on-retry="handleRetry">
        <!-- Controls -->
        <template #controls>
          <ControlPanel :controls="controls" :values="parameters" :loading="isTraining" submit-label="Train Model"
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
            <div v-if="metrics.transferLearningMode" class="mt-3 pt-3 border-t border-gray-200">
              <div class="text-xs text-gray-500">Transfer Learning Mode</div>
              <div class="text-sm font-semibold text-gray-700">{{ metrics.transferLearningMode }}</div>
            </div>
            <div v-if="metrics.featureSize" class="mt-2">
              <div class="text-xs text-gray-500">Feature Size</div>
              <div class="text-sm font-mono text-gray-700">{{ metrics.featureSize }} dimensions</div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-6">
            <!-- Feature space visualization -->
            <Scatter2D v-if="featureSpaceSeries && featureSpaceSeries.length > 0" :series="featureSpaceSeries"
              title="Extracted Feature Space" x-label="Feature Dimension 1" y-label="Feature Dimension 2"
              height="400px" />

            <!-- Training curves -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <LineChart v-if="lossSeries && lossSeries.length > 0" :series="lossSeries" title="Training Loss"
                x-label="Epoch" y-label="Loss" :show-legend="false" height="300px" />

              <LineChart v-if="accuracySeries && accuracySeries.length > 0" :series="accuracySeries"
                title="Training Accuracy" x-label="Epoch" y-label="Accuracy" :show-legend="false" height="300px" />
            </div>

            <!-- Confusion Matrix -->
            <ConfusionMatrix v-if="confusionMatrix && confusionMatrix.length > 0" :confusionMatrix="confusionMatrix"
              title="Confusion Matrix" height="400px" />
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
      <Quiz v-else :quiz="quizData" :passing-score="quizData.passingScore || 70" @quiz-completed="handleQuizComplete" />
    </template>
  </ModuleLayout>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { useModuleStore } from '~/stores/module';
import { useDataset } from '~/composables/useDataset';
import { useModules } from '~/composables/useModules';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { TransferLearning } from '~/algorithms/computer-vision/TransferLearning';
import LineChart from '~/components/visualizations/LineChart.vue';
import Scatter2D from '~/components/visualizations/Scatter2D.vue';
import ConfusionMatrix from '~/components/visualizations/ConfusionMatrix.vue';
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

const { data: content } = await useAsyncData(() => 
  queryCollection('content').path('/modules/computer-vision/transfer-learning/content').first()
)

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
  featureExtractorLayers: 2,
  fineTuneLayers: 1,
  learningRate: 0.001,
  epochs: 50,
  freezeFeatures: true,
  dataset: 'image-features',
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'dataset',
    type: 'select',
    label: 'Dataset',
    options: [
      { label: 'Image Features (3 classes)', value: 'image-features' },
      { label: 'Multi-Class Classification', value: 'multi-classification' },
    ],
    tooltip: 'Choose a dataset for the new classification task',
    group: 'Data',
  },
  {
    id: 'freezeFeatures',
    type: 'toggle',
    label: 'Freeze Feature Extractor',
    tooltip: 'When enabled, only trains the classifier. When disabled, fine-tunes the feature extractor too.',
    group: 'Transfer Learning Strategy',
  },
  {
    id: 'featureExtractorLayers',
    type: 'slider',
    label: 'Feature Extractor Layers',
    min: 1,
    max: 4,
    step: 1,
    tooltip: 'Number of layers in the pre-trained feature extractor',
    group: 'Model Architecture',
  },
  {
    id: 'fineTuneLayers',
    type: 'slider',
    label: 'Fine-Tune Layers',
    min: 0,
    max: 3,
    step: 1,
    tooltip: 'Number of feature extractor layers to fine-tune (when not frozen)',
    group: 'Transfer Learning Strategy',
  },
  {
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.0001,
    max: 0.01,
    step: 0.0001,
    tooltip: 'Learning rate for training. Transfer learning typically uses smaller rates.',
    group: 'Training Parameters',
  },
  {
    id: 'epochs',
    type: 'slider',
    label: 'Epochs',
    min: 10,
    max: 200,
    step: 10,
    tooltip: 'Number of training iterations. Transfer learning often needs fewer epochs.',
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
const featureSpaceSeries = ref<any[]>([]);
const confusionMatrix = ref<number[][]>([]);

// Algorithm instance
let algorithm: TransferLearning | null = null;

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('transfer-learning');
  moduleStore.setParameters(parameters.value);

  // Load default dataset
  await loadInitialDataset();

  // Load navigation
  previousModule.value = await getPreviousModule('transfer-learning');
  nextModule.value = await getNextModule('transfer-learning');

  // For now, suggested module is the same as next module
  suggestedModule.value = nextModule.value;

  // Load quiz data
  await loadQuiz('transfer-learning', '/content/modules/computer-vision/transfer-learning/quiz.json');
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
    await loadInitialDataset();
    // Clear previous results
    metrics.value = null;
    lossSeries.value = [];
    accuracySeries.value = [];
    featureSpaceSeries.value = [];
    confusionMatrix.value = [];
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    featureExtractorLayers: 2,
    fineTuneLayers: 1,
    learningRate: 0.001,
    epochs: 50,
    freezeFeatures: true,
    dataset: 'image-features',
  };
  moduleStore.setParameters(parameters.value);

  await loadInitialDataset();

  metrics.value = null;
  lossSeries.value = [];
  accuracySeries.value = [];
  featureSpaceSeries.value = [];
  confusionMatrix.value = [];
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
    algorithm = new TransferLearning({
      featureExtractorLayers: parameters.value.featureExtractorLayers,
      fineTuneLayers: parameters.value.fineTuneLayers,
      learningRate: parameters.value.learningRate,
      epochs: parameters.value.epochs,
      freezeFeatures: parameters.value.freezeFeatures,
    });

    // Train with progress updates
    const startTime = Date.now();
    await algorithm.fit(X, y);
    const endTime = Date.now();

    console.log(`Training completed in ${endTime - startTime}ms`);

    // Get results
    metrics.value = algorithm.getMetrics();
    const visualizations = algorithm.getVisualizationData();

    // Process visualizations
    if (visualizations.length > 0) {
      // Feature space visualization - convert object format to series array
      const featureViz = visualizations.find(v => v.type === 'scatter2d');
      if (featureViz && featureViz.data) {
        // Convert object with class keys to array of series
        const classData = featureViz.data;
        featureSpaceSeries.value = Object.values(classData).map((classInfo: any) => ({
          x: classInfo.x,
          y: classInfo.y,
          name: classInfo.name,
          color: classInfo.marker?.color,
          mode: classInfo.mode || 'markers',
          marker: {
            size: classInfo.marker?.size || 8,
            symbol: classInfo.marker?.symbol || 'circle',
          }
        }));
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

      // Confusion matrix
      const confusionViz = visualizations.find(v => v.type === 'confusion-matrix');
      if (confusionViz && confusionViz.data) {
        confusionMatrix.value = confusionViz.data;
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
  featureSpaceSeries.value = [];
  confusionMatrix.value = [];
  
  // Optionally re-train with new parameters
  // await handleTrain();
});
</script>
