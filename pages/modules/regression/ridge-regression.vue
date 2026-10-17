<template>
  <ModuleLayout
    title="Ridge Regression"
    description="Learn how Ridge regression prevents overfitting using L2 regularization"
    difficulty="intermediate"
    :estimated-time="35"
    module-id="ridge-regression"
    :table-of-contents="tableOfContents"
    :parameters="parameters"
    @load-bookmark="(val) => { parameters = val as typeof parameters; handleLoadBookmark(val); }"
  >
    <template #content>
      <ContentRenderer v-if="content" :value="content" class="prose prose-slate max-w-none" />
    </template>

    <template #interactive>
      <InteractiveSection
        :loading="isTraining"
        :progress="trainingProgress"
        loading-message="Training model..."
        :error="error"
        :on-retry="handleRetry"
      >
        <template #controls>
          <ControlPanel
            :controls="controls"
            :values="parameters"
            :loading="isTraining"
            submit-label="Train Model"
            @change="handleParameterChange"
            @reset="handleReset"
            @submit="handleTrain"
          />
          
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Model Performance</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">MSE</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.mse?.toFixed(4) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">RMSE</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.rmse?.toFixed(4) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">R² Score</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.r2Score?.toFixed(4) }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">MAE</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.mae?.toFixed(4) }}</div>
              </div>
            </div>
            <div v-if="metrics.l2Norm" class="mt-3 pt-3 border-t border-gray-200">
              <div class="text-xs text-gray-500">L2 Norm of Weights</div>
              <div class="text-lg font-semibold text-gray-900">{{ metrics.l2Norm?.toFixed(4) }}</div>
            </div>
          </div>
        </template>

        <template #visualization>
          <div class="space-y-4">
            <Scatter2D
              v-if="scatterSeries && scatterSeries.length > 0"
              :series="scatterSeries"
              title="Data & Ridge Regression Line"
              x-label="X"
              y-label="Y"
              :show-legend="true"
              height="500px"
            />
            
            <LineChart
              v-if="lossSeries && lossSeries.length > 0"
              :series="lossSeries"
              title="Training Loss"
              x-label="Epoch"
              y-label="Loss"
              :show-legend="false"
              height="350px"
            />
          </div>
        </template>
      </InteractiveSection>
    </template>

    <template #quiz>
      <QuizLoadingState v-if="quizLoading" />
      <QuizErrorDisplay v-else-if="quizError" :error="quizError" :on-retry="retryQuizLoad" />
      <QuizErrorDisplay v-else-if="!quizData" :coming-soon="true" />
      <Quiz v-else :quiz="quizData" :passing-score="quizData.passingScore || 70" @quiz-completed="handleQuizComplete" />
    </template>
  </ModuleLayout>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { useModuleStore } from '~/stores/module';
import { useAnalyticsStore } from '~/stores/analytics';
import { useDataset } from '~/composables/useDataset';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { RidgeRegression } from '~/algorithms/regression/RidgeRegression';
import Scatter2D from '~/components/visualizations/Scatter2D.vue';
import LineChart from '~/components/visualizations/LineChart.vue';
import Quiz from '~/components/modules/Quiz.vue';
import QuizErrorDisplay from '~/components/modules/QuizErrorDisplay.vue';
import QuizLoadingState from '~/components/modules/QuizLoadingState.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { ScatterSeries } from '~/components/visualizations/Scatter2D.vue';
import type { LineSeries } from '~/components/visualizations/LineChart.vue';
import type { QuizResult } from '~/types/quiz';
import { useQuizLoader } from '~/composables/useQuizLoader';
import { useBookmarkRestore } from '~/composables/useBookmarkRestore';

const { data: content } = await useAsyncData(() => 
  queryCollection('content').path('/modules/regression/ridge-regression/content').first()
);

const tableOfContents = useTableOfContents(content);
const moduleStore = useModuleStore();
const analyticsStore = useAnalyticsStore();
const { loadDataset, getDatasetsByType } = useDataset();
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

const parameters = ref({
  alpha: 1.0,
  learningRate: 0.01,
  epochs: 100,
  dataset: 'noisy-linear',
});

const controls: ControlConfig[] = [
  {
    id: 'dataset',
    type: 'select',
    label: 'Dataset',
    options: getDatasetsByType('regression').map(d => ({ label: d.name, value: d.id })),
    tooltip: 'Choose a dataset to train on',
    group: 'Data',
  },
  {
    id: 'alpha',
    type: 'slider',
    label: 'Alpha (Regularization Strength)',
    min: 0,
    max: 10,
    step: 0.1,
    tooltip: 'Controls regularization strength. Higher values shrink coefficients more.',
    group: 'Model Parameters',
  },
  {
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.001,
    max: 0.1,
    step: 0.001,
    tooltip: 'Controls how quickly the model learns.',
    group: 'Training Parameters',
  },
  {
    id: 'epochs',
    type: 'slider',
    label: 'Epochs',
    min: 10,
    max: 500,
    step: 10,
    tooltip: 'Number of training iterations.',
    group: 'Training Parameters',
  },
];

const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);
const metrics = ref<Metrics | null>(null);
const scatterSeries = ref<ScatterSeries[]>([]);
const lossSeries = ref<LineSeries[]>([]);

let algorithm: RidgeRegression | null = null;

onMounted(async () => {
  moduleStore.setCurrentModule('ridge-regression');
  moduleStore.setParameters(parameters.value);
  analyticsStore.trackModuleVisit('ridge-regression', 'Ridge Regression');
  await loadInitialDataset();
  await loadQuiz('ridge-regression', '/content/modules/regression/ridge-regression/quiz.json');
});

const loadInitialDataset = async () => {
  const dataset = await loadDataset(parameters.value.dataset);
  if (dataset) moduleStore.loadDataset(dataset.data);
};

const handleParameterChange = async (id: string, value: any) => {
  parameters.value[id] = value;
  moduleStore.setParameter(id, value);
  analyticsStore.trackParameterChange('ridge-regression', id, value);
  
  if (id === 'dataset') {
    const dataset = await loadDataset(value);
    if (dataset) {
      moduleStore.loadDataset(dataset.data);
      metrics.value = null;
      scatterSeries.value = [];
      lossSeries.value = [];
    }
  }
};

const handleReset = async () => {
  parameters.value = { alpha: 1.0, learningRate: 0.01, epochs: 100, dataset: 'noisy-linear' };
  moduleStore.setParameters(parameters.value);
  await loadInitialDataset();
  metrics.value = null;
  scatterSeries.value = [];
  lossSeries.value = [];
  error.value = null;
};

const handleTrain = async () => {
  isTraining.value = true;
  trainingProgress.value = 0;
  error.value = null;

  try {
    const dataset = moduleStore.dataset;
    if (!dataset || dataset.length === 0) throw new Error('No dataset loaded');
    
    const X = dataset.map(d => d.features);
    const y = dataset.map(d => d.label as number);
    
    algorithm = new RidgeRegression({
      alpha: parameters.value.alpha,
      learningRate: parameters.value.learningRate,
      epochs: parameters.value.epochs,
      normalize: true
    });
    
    const startTime = Date.now();
    await algorithm.fit(X, y);
    const duration = Date.now() - startTime;
    
    analyticsStore.trackAlgorithmRun('ridge-regression', parameters.value, duration);
    
    metrics.value = algorithm.getMetrics();
    const visualizations = algorithm.getVisualizationData();
    
    if (visualizations.length > 0) {
      const scatterViz = visualizations[0];
      if (scatterViz.type === 'scatter2d' && scatterViz.data) {
        const series: ScatterSeries[] = [];
        if (scatterViz.data.scatter) {
          series.push({
            x: scatterViz.data.scatter.x,
            y: scatterViz.data.scatter.y,
            name: scatterViz.data.scatter.name || 'Data Points',
            color: scatterViz.data.scatter.marker?.color || '#3b82f6',
            mode: 'markers',
            marker: { size: scatterViz.data.scatter.marker?.size || 8 }
          });
        }
        if (scatterViz.data.line) {
          series.push({
            x: scatterViz.data.line.x,
            y: scatterViz.data.line.y,
            name: scatterViz.data.line.name || 'Ridge Line',
            color: scatterViz.data.line.line?.color || '#ef4444',
            mode: 'lines'
          });
        }
        scatterSeries.value = series;
      }
      
      if (visualizations.length > 2) {
        const lossViz = visualizations[2];
        if (lossViz.type === 'line' && lossViz.data) {
          lossSeries.value = [{
            x: lossViz.data.x,
            y: lossViz.data.y,
            name: lossViz.data.name || 'Training Loss',
            color: lossViz.data.line?.color || '#10b981',
            width: lossViz.data.line?.width || 2
          }];
        }
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

const handleRetry = () => {
  error.value = null;
  handleTrain();
};

const handleQuizComplete = (result: QuizResult) => {
  console.log('Quiz completed:', result);
};

// Bookmark restoration
// Requirements: 2.2, 2.3
const { handleLoadBookmark } = useBookmarkRestore(parameters, handleTrain);

useSEO({
  title: 'Ridge Regression - Interactive ML Tutorial',
  description: 'Learn Ridge regression and L2 regularization through interactive visualization. Understand how regularization prevents overfitting and improves model generalization.',
  keywords: 'ridge regression, L2 regularization, regularization, overfitting prevention, machine learning',
  url: '/modules/regression/ridge-regression',
  type: 'article',
  section: 'Regression',
})

useCourseStructuredData({
  name: 'Ridge Regression',
  description: 'Learn how Ridge regression prevents overfitting using L2 regularization',
  provider: 'Machine Learning 101',
  url: '/modules/regression/ridge-regression',
  difficulty: 'intermediate',
  timeRequired: 'PT35M',
  keywords: ['ridge regression', 'L2 regularization', 'machine learning', 'regularization'],
})

useBreadcrumbStructuredData([
  { name: 'Home', url: '/' },
  { name: 'Learning Modules', url: '/modules' },
  { name: 'Regression', url: '/modules/regression' },
  { name: 'Ridge Regression', url: '/modules/regression/ridge-regression' },
])
</script>
