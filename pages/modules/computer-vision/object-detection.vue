<template>
  <ModuleLayout title="Object Detection" description="Learn how to detect and localize objects in images using bounding boxes"
    difficulty="advanced" :estimated-time="50" module-id="object-detection"
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
      <InteractiveSection :loading="isTraining" :progress="trainingProgress" loading-message="Training detector..."
        :error="error" :on-retry="handleRetry">
        <!-- Controls -->
        <template #controls>
          <ControlPanel :controls="controls" :values="parameters" :loading="isTraining" submit-label="Train Detector"
            @change="handleParameterChange" @reset="handleReset" @submit="handleTrain" />

          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Detection Performance</h3>
            <div class="grid grid-cols-2 gap-4">
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
              <div>
                <div class="text-xs text-gray-500">Avg Detections</div>
                <div class="text-lg font-semibold text-gray-900">{{ (metrics.averageDetectionsPerImage ||
                  0).toFixed(2) }}</div>
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <!-- Bounding Box Overlay -->
            <BoundingBoxOverlay v-if="detectionData" :image="detectionData.image" :detections="detectionData.detections"
              :ground-truth="detectionData.groundTruth" :image-width="detectionData.imageWidth"
              :image-height="detectionData.imageHeight" title="Object Detection Results" :show-ground-truth="true"
              :show-detections="true" height="600px" />

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
import { ObjectDetection } from '~/algorithms/computer-vision/ObjectDetection';
import BoundingBoxOverlay from '~/components/visualizations/BoundingBoxOverlay.vue';
import LineChart from '~/components/visualizations/LineChart.vue';
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

const { data: content } = await useAsyncData(() => queryCollection('content').path('/modules/computer-vision/object-detection/content').first())

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
  windowSize: 8,
  stride: 4,
  confidenceThreshold: 0.5,
  iouThreshold: 0.3,
  learningRate: 0.01,
  epochs: 50,
  dataset: 'simple-objects',
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'dataset',
    type: 'select',
    label: 'Dataset',
    options: getDatasetsByType('detection').map(d => ({
      label: d.name,
      value: d.id
    })),
    tooltip: 'Choose a dataset for object detection',
    group: 'Data',
  },
  {
    id: 'windowSize',
    type: 'slider',
    label: 'Window Size',
    min: 4,
    max: 12,
    step: 2,
    tooltip: 'Size of the sliding window for object detection',
    group: 'Detection Parameters',
  },
  {
    id: 'stride',
    type: 'slider',
    label: 'Stride',
    min: 2,
    max: 8,
    step: 2,
    tooltip: 'Step size for sliding the detection window',
    group: 'Detection Parameters',
  },
  {
    id: 'confidenceThreshold',
    type: 'slider',
    label: 'Confidence Threshold',
    min: 0.1,
    max: 0.9,
    step: 0.05,
    tooltip: 'Minimum confidence score to consider a detection valid',
    group: 'Detection Parameters',
  },
  {
    id: 'iouThreshold',
    type: 'slider',
    label: 'IoU Threshold (NMS)',
    min: 0.1,
    max: 0.7,
    step: 0.05,
    tooltip: 'IoU threshold for Non-Maximum Suppression',
    group: 'Detection Parameters',
  },
  {
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.001,
    max: 0.1,
    step: 0.001,
    tooltip: 'Step size for gradient descent updates',
    group: 'Training Parameters',
  },
  {
    id: 'epochs',
    type: 'slider',
    label: 'Training Epochs',
    min: 10,
    max: 100,
    step: 10,
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
const detectionData = ref<any>(null);

// Algorithm instance
let algorithm: ObjectDetection | null = null;

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('object-detection');
  moduleStore.setParameters(parameters.value);

  // Load default dataset
  await loadInitialDataset();

  // Load navigation
  previousModule.value = await getPreviousModule('object-detection');
  nextModule.value = await getNextModule('object-detection');

  // For now, suggested module is the same as next module
  suggestedModule.value = nextModule.value;

  // Load quiz data
  await loadQuiz('object-detection', '/content/modules/computer-vision/object-detection/quiz.json');
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
      lossSeries.value = [];
      accuracySeries.value = [];
      detectionData.value = null;
    }
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    windowSize: 8,
    stride: 4,
    confidenceThreshold: 0.5,
    iouThreshold: 0.3,
    learningRate: 0.01,
    epochs: 50,
    dataset: 'simple-objects',
  };
  moduleStore.setParameters(parameters.value);

  await loadInitialDataset();

  metrics.value = null;
  lossSeries.value = [];
  accuracySeries.value = [];
  detectionData.value = null;
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
    algorithm = new ObjectDetection({
      windowSize: parameters.value.windowSize,
      stride: parameters.value.stride,
      confidenceThreshold: parameters.value.confidenceThreshold,
      iouThreshold: parameters.value.iouThreshold,
      learningRate: parameters.value.learningRate,
      epochs: parameters.value.epochs
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
      // Bounding box visualization
      const bboxViz = visualizations.find(v => v.type === 'heatmap' && v.data.image);
      if (bboxViz && bboxViz.data) {
        detectionData.value = {
          image: bboxViz.data.image,
          detections: bboxViz.data.detections || [],
          groundTruth: bboxViz.data.groundTruth || [],
          imageWidth: bboxViz.data.imageWidth,
          imageHeight: bboxViz.data.imageHeight,
        };
      }

      // Loss curve
      const lossViz = visualizations.find(v => v.data.name === 'Training Loss');
      if (lossViz && lossViz.type === 'line' && lossViz.data) {
        lossSeries.value = [{
          x: lossViz.data.x,
          y: lossViz.data.y,
          name: lossViz.data.name || 'Training Loss',
          color: lossViz.data.line?.color || '#10b981',
          width: lossViz.data.line?.width || 2,
        }];
      }

      // Accuracy curve
      const accuracyViz = visualizations.find(v => v.data.name === 'Training Accuracy');
      if (accuracyViz && accuracyViz.type === 'line' && accuracyViz.data) {
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
  detectionData.value = null;
  
  // Optionally re-train with new parameters
  // await handleTrain();
});
</script>
