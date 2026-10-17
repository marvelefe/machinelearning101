<template>
  <ModuleLayout title="Image Classification Basics"
    description="Learn how neural networks classify images using extracted features and multi-class classification"
    difficulty="beginner" :estimated-time="40" module-id="image-classification"
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
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <!-- Feature space scatter plot -->
            <Scatter2D v-if="scatterSeries && scatterSeries.length > 0" :series="scatterSeries"
              title="Feature Space Visualization" x-label="Feature 1" y-label="Feature 2"
              :show-legend="true" height="500px" />

            <!-- Confusion Matrix -->
            <Heatmap v-if="confusionMatrixData" :data="confusionMatrixData" 
              :x-labels="confusionMatrixLabels"
              :y-labels="confusionMatrixLabels" 
              title="Confusion Matrix" 
              colorscale="Blues" 
              :show-annotations="true"
              height="400px" />

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
import { ImageClassification } from '~/algorithms/computer-vision/ImageClassification';
import Scatter2D from '~/components/visualizations/Scatter2D.vue';
import LineChart from '~/components/visualizations/LineChart.vue';
import Heatmap from '~/components/visualizations/Heatmap.vue';
import Quiz from '~/components/modules/Quiz.vue';
import QuizErrorDisplay from '~/components/modules/QuizErrorDisplay.vue';
import QuizLoadingState from '~/components/modules/QuizLoadingState.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { ScatterSeries } from '~/components/visualizations/Scatter2D.vue';
import type { LineSeries } from '~/components/visualizations/LineChart.vue';
import type { ModuleNavItem } from '~/types/module';
import type { Quiz as QuizType, QuizResult } from '~/types/quiz';
import { useQuizLoader } from '~/composables/useQuizLoader';
import { useBookmarkRestore } from '~/composables/useBookmarkRestore';
import { useAsyncData } from '#app';


const { data: content } = await useAsyncData(() => 
  queryCollection('content').path('/modules/computer-vision/image-classification/content').first()
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
  learningRate: 0.01,
  epochs: 100,
  hiddenSize: 10,
  regularization: 0.01,
  dataset: 'multi-classification',
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
    tooltip: 'Choose a dataset with extracted image features',
    group: 'Data',
  },
  {
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.001,
    max: 0.5,
    step: 0.001,
    tooltip: 'Controls how quickly the network learns. Higher values train faster but may be unstable.',
    group: 'Training Parameters',
  },
  {
    id: 'epochs',
    type: 'slider',
    label: 'Epochs',
    min: 10,
    max: 500,
    step: 10,
    tooltip: 'Number of training iterations. More epochs allow better learning but take longer.',
    group: 'Training Parameters',
  },
  {
    id: 'hiddenSize',
    type: 'slider',
    label: 'Hidden Layer Size',
    min: 5,
    max: 50,
    step: 5,
    tooltip: 'Number of neurons in the hidden layer. More neurons increase model capacity.',
    group: 'Model Architecture',
  },
  {
    id: 'regularization',
    type: 'slider',
    label: 'Regularization (L2)',
    min: 0,
    max: 0.1,
    step: 0.001,
    tooltip: 'Prevents overfitting by penalizing large weights. Higher values create simpler models.',
    group: 'Training Parameters',
  },
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const scatterSeries = ref<ScatterSeries[]>([]);
const lossSeries = ref<LineSeries[]>([]);
const accuracySeries = ref<LineSeries[]>([]);
const confusionMatrixData = ref<number[][] | null>(null);
const confusionMatrixLabels = ref<string[]>([]);

// Algorithm instance
let algorithm: ImageClassification | null = null;

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('image-classification');
  moduleStore.setParameters(parameters.value);

  // Load default dataset
  await loadInitialDataset();

  // Load navigation
  previousModule.value = await getPreviousModule('image-classification');
  nextModule.value = await getNextModule('image-classification');

  // For now, suggested module is the same as next module
  suggestedModule.value = nextModule.value;

  // Load quiz data
  await loadQuiz('image-classification', '/content/modules/computer-vision/image-classification/quiz.json');
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
      scatterSeries.value = [];
      lossSeries.value = [];
      accuracySeries.value = [];
      confusionMatrixData.value = null;
    }
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    learningRate: 0.01,
    epochs: 100,
    hiddenSize: 10,
    regularization: 0.01,
    dataset: 'multi-classification',
  };
  moduleStore.setParameters(parameters.value);

  await loadInitialDataset();

  metrics.value = null;
  scatterSeries.value = [];
  lossSeries.value = [];
  accuracySeries.value = [];
  confusionMatrixData.value = null;
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
    algorithm = new ImageClassification({
      learningRate: parameters.value.learningRate,
      epochs: parameters.value.epochs,
      hiddenSize: parameters.value.hiddenSize,
      regularization: parameters.value.regularization
    });

    // Train with progress updates
    const startTime = Date.now();
    await algorithm.fit(X, y);
    const endTime = Date.now();

    console.log(`Training completed in ${endTime - startTime}ms`);

    // Get results
    metrics.value = algorithm.getMetrics();
    const visualizations = algorithm.getVisualizationData();

    // Set confusion matrix data and labels
    if (metrics.value.confusionMatrix) {
      confusionMatrixData.value = metrics.value.confusionMatrix;
      confusionMatrixLabels.value = Array.from({ length: numClasses }, (_, i) => `Class ${i}`);
    }

    // Process visualizations
    if (visualizations.length > 0) {
      // Scatter plot with classes
      const scatterViz = visualizations.find(v => v.type === 'scatter2d');
      if (scatterViz && scatterViz.data) {
        const series: ScatterSeries[] = [];
        
        // Process each class
        Object.keys(scatterViz.data).forEach(key => {
          const classData = scatterViz.data[key];
          if (classData && classData.x && classData.y) {
            series.push({
              x: classData.x,
              y: classData.y,
              name: classData.name || key,
              color: classData.marker?.color || '#3b82f6',
              mode: 'markers',
              marker: {
                size: classData.marker?.size || 8,
                symbol: classData.marker?.symbol || 'circle',
              }
            });
          }
        });

        scatterSeries.value = series;
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
  scatterSeries.value = [];
  lossSeries.value = [];
  accuracySeries.value = [];
  confusionMatrixData.value = null;
  
  // Optionally re-train with new parameters
  // await handleTrain();
});
</script>
