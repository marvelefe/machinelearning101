<template>
  <ModuleLayout title="Image Segmentation"
    description="Learn pixel-wise classification for semantic segmentation using clustering algorithms"
    difficulty="intermediate" :estimated-time="45" module-id="image-segmentation"
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
      <InteractiveSection :loading="isTraining" :progress="trainingProgress" loading-message="Segmenting image..."
        :error="error" :on-retry="handleRetry">
        <!-- Controls -->
        <template #controls>
          <ControlPanel :controls="controls" :values="parameters" :loading="isTraining" submit-label="Segment Image"
            @change="handleParameterChange" @reset="handleReset" @submit="handleTrain" />

          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Segmentation Statistics</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Inertia</div>
                <div class="text-lg font-semibold text-gray-900">{{ (metrics.inertia || 0).toFixed(2) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Iterations</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.iterations || 0 }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Segments</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.numSegments || 0 }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Avg Distance</div>
                <div class="text-lg font-semibold text-gray-900">{{ (metrics.avgDistanceToCenter || 0).toFixed(3) }}
                </div>
              </div>
            </div>

            <!-- Segment sizes -->
            <div v-if="metrics.segmentPercentages" class="mt-4">
              <div class="text-xs text-gray-500 mb-2">Segment Coverage</div>
              <div class="space-y-1">
                <div v-for="(percentage, idx) in metrics.segmentPercentages" :key="idx"
                  class="flex items-center justify-between text-sm">
                  <span class="text-gray-600">Segment {{ idx }}</span>
                  <span class="font-medium text-gray-900">{{ percentage.toFixed(1) }}%</span>
                </div>
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <!-- Original and segmented images side by side -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <Heatmap v-if="originalImage" :data="originalImage" title="Original Image" colorscale="Greys"
                :show-scale="false" :show-annotations="false" height="400px" />

              <SegmentationMask v-if="segmentationMask" :mask="segmentationMask.segments" :colors="segmentationMask.colors"
                :width="segmentationMask.width" :height="segmentationMask.height" :num-segments="parameters.numSegments"
                title="Segmentation Mask" :show-legend="true" chart-height="400px" />
            </div>

            <!-- Inertia curve -->
            <LineChart v-if="inertiaSeries && inertiaSeries.length > 0" :series="inertiaSeries"
              title="Inertia Over Iterations" x-label="Iteration" y-label="Inertia" :show-legend="false"
              height="300px" />
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
import { ImageSegmentation } from '~/algorithms/computer-vision/ImageSegmentation';
import type { SegmentationMask as SegmentationMaskType } from '~/algorithms/computer-vision/ImageSegmentation';
import LineChart from '~/components/visualizations/LineChart.vue';
import Heatmap from '~/components/visualizations/Heatmap.vue';
import SegmentationMask from '~/components/visualizations/SegmentationMask.vue';
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

const { data: content } = await useAsyncData(() => queryCollection('content').path('/modules/computer-vision/image-segmentation/content').first())

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
  numSegments: 4,
  maxIterations: 50,
  useColorFeatures: true,
  usePositionFeatures: true,
  positionWeight: 0.3,
  dataset: 'image-features',
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'dataset',
    type: 'select',
    label: 'Dataset',
    options: getDatasetsByType('classification').map(d => ({
      label: d.name,
      value: d.id
    })),
    tooltip: 'Choose an image dataset to segment',
    group: 'Data',
  },
  {
    id: 'numSegments',
    type: 'slider',
    label: 'Number of Segments',
    min: 2,
    max: 8,
    step: 1,
    tooltip: 'Number of distinct regions to segment the image into. More segments capture finer details.',
    group: 'Segmentation Parameters',
  },
  {
    id: 'maxIterations',
    type: 'slider',
    label: 'Max Iterations',
    min: 10,
    max: 100,
    step: 5,
    tooltip: 'Maximum number of clustering iterations. More iterations allow better convergence.',
    group: 'Training Parameters',
  },
  {
    id: 'useColorFeatures',
    type: 'toggle',
    label: 'Use Color Features',
    tooltip: 'Include pixel intensity/color values as features for segmentation.',
    group: 'Feature Selection',
  },
  {
    id: 'usePositionFeatures',
    type: 'toggle',
    label: 'Use Position Features',
    tooltip: 'Include pixel position (x, y coordinates) as features. Helps create spatially coherent segments.',
    group: 'Feature Selection',
  },
  {
    id: 'positionWeight',
    type: 'slider',
    label: 'Position Weight',
    min: 0,
    max: 1,
    step: 0.05,
    tooltip: 'Weight given to position features relative to color. Higher values create more spatially compact segments.',
    group: 'Feature Selection',
  },
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const originalImage = ref<number[][] | null>(null);
const segmentationMask = ref<SegmentationMaskType | null>(null);
const inertiaSeries = ref<LineSeries[]>([]);

// Algorithm instance
let algorithm: ImageSegmentation | null = null;

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('image-segmentation');
  moduleStore.setParameters(parameters.value);

  // Load default dataset
  await loadInitialDataset();

  // Load navigation
  previousModule.value = await getPreviousModule('image-segmentation');
  nextModule.value = await getNextModule('image-segmentation');

  // For now, suggested module is the same as next module
  suggestedModule.value = nextModule.value;

  // Load quiz data
  await loadQuiz('image-segmentation', '/content/modules/computer-vision/image-segmentation/quiz.json');
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
      originalImage.value = null;
      segmentationMask.value = null;
      inertiaSeries.value = [];
    }
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    numSegments: 4,
    maxIterations: 50,
    useColorFeatures: true,
    usePositionFeatures: true,
    positionWeight: 0.3,
    dataset: 'image-features',
  };
  moduleStore.setParameters(parameters.value);

  await loadInitialDataset();

  metrics.value = null;
  originalImage.value = null;
  segmentationMask.value = null;
  inertiaSeries.value = [];
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

    // Prepare data - use first sample as the image to segment
    const X = dataset.map(d => d.features);

    // Validate that features form a square image
    const numFeatures = X[0].length;
    const imageSize = Math.sqrt(numFeatures);
    
    if (!Number.isInteger(imageSize)) {
      throw new Error('Dataset features must form a square image');
    }

    // Create and train algorithm
    algorithm = new ImageSegmentation({
      numSegments: parameters.value.numSegments,
      maxIterations: parameters.value.maxIterations,
      useColorFeatures: parameters.value.useColorFeatures,
      usePositionFeatures: parameters.value.usePositionFeatures,
      positionWeight: parameters.value.positionWeight
    });

    // Train with progress updates
    const startTime = Date.now();
    await algorithm.fit(X);
    const endTime = Date.now();

    console.log(`Segmentation completed in ${endTime - startTime}ms`);

    // Get results
    metrics.value = algorithm.getMetrics();
    const visualizations = algorithm.getVisualizationData();

    // Get segmentation mask
    segmentationMask.value = algorithm.getSegmentationMask(X[0]);

    // Process visualizations
    if (visualizations.length > 0) {
      // Original image
      const originalViz = visualizations.find(v => v.data.name === 'Original Image');
      if (originalViz && originalViz.type === 'heatmap' && originalViz.data.z) {
        originalImage.value = originalViz.data.z;
      }

      // Inertia curve
      const inertiaViz = visualizations.find(v => v.data.name === 'Inertia');
      if (inertiaViz && inertiaViz.type === 'line' && inertiaViz.data) {
        inertiaSeries.value = [{
          x: inertiaViz.data.x,
          y: inertiaViz.data.y,
          name: inertiaViz.data.name || 'Inertia',
          color: inertiaViz.data.line?.color || '#10b981',
          width: inertiaViz.data.line?.width || 2,
        }];
      }
    }

    trainingProgress.value = 1;
  } catch (err) {
    console.error('Segmentation error:', err);
    error.value = err instanceof Error ? err.message : 'Segmentation failed. Please try again.';
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
  originalImage.value = null;
  segmentationMask.value = null;
  inertiaSeries.value = [];
  
  // Optionally re-train with new parameters
  // await handleTrain();
});
</script>
