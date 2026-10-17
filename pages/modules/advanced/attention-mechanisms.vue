<template>
  <ModuleLayout title="Attention Mechanisms"
    description="Learn how attention allows models to focus on relevant parts of input sequences" difficulty="advanced"
    :estimated-time="55" module-id="attention-mechanisms" :table-of-contents="tableOfContents" :parameters="parameters"
    @load-bookmark="(val) => { parameters = val as typeof parameters; handleLoadBookmark(val); }">
    <!-- Content Section -->
    <template #content>
      <ContentRenderer v-if="content" :value="content" class="prose prose-slate max-w-none" />
    </template>

    <!-- Interactive Section -->
    <template #interactive>
      <InteractiveSection :loading="isTraining" :progress="trainingProgress" loading-message="Computing attention..."
        :error="error" :on-retry="handleRetry">
        <!-- Controls -->
        <template #controls>
          <ControlPanel :controls="controls" :values="parameters" :loading="isTraining" submit-label="Compute Attention"
            @change="handleParameterChange" @reset="handleReset" @submit="handleTrain" />

          <!-- Architecture Display -->
          <div class="mt-6 p-4 bg-indigo-50 rounded-lg border border-indigo-200">
            <h3 class="text-sm font-semibold text-indigo-700 mb-3">Attention Configuration</h3>
            <div class="space-y-2">
              <div class="flex items-center gap-2">
                <span class="text-xs text-indigo-600 w-32">Sequence Length:</span>
                <span class="text-sm font-semibold text-indigo-900">{{ parameters.sequenceLength }}</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-indigo-600 w-32">Embedding Dim:</span>
                <span class="text-sm font-semibold text-indigo-900">{{ parameters.embeddingDim }}</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-indigo-600 w-32">Attention Type:</span>
                <span class="text-sm font-semibold text-indigo-900">
                  {{ parameters.attentionType === 'self' ? 'Self-Attention' : 'Multi-Head Attention' }}
                </span>
              </div>
              <div v-if="parameters.attentionType === 'multi-head'" class="flex items-center gap-2">
                <span class="text-xs text-indigo-600 w-32">Number of Heads:</span>
                <span class="text-sm font-semibold text-indigo-900">{{ parameters.numHeads }}</span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-xs text-indigo-600 w-32">Temperature:</span>
                <span class="text-sm font-semibold text-indigo-900">{{ parameters.temperature }}</span>
              </div>
            </div>
          </div>

          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Attention Statistics</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Attention Entropy</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.attentionEntropy?.toFixed(4) }}
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Max Attention Weight</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.maxAttentionWeight?.toFixed(4) }}
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Total Parameters</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.totalParameters }}
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Computation Time</div>
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
            <Heatmap v-if="attentionWeightsViz" :data="attentionWeightsViz" title="Attention Weights Heatmap"
              x-label="Key Position" y-label="Query Position" height="400px" />

            <div v-if="parameters.attentionType === 'multi-head' && multiHeadViz" class="grid grid-cols-2 gap-4">
              <Heatmap v-for="(headViz, idx) in multiHeadViz.slice(0, 4)" :key="idx" :data="headViz"
                :title="`Head ${idx + 1} Attention`" x-label="Key" y-label="Query" height="250px" />
            </div>

            <div class="grid grid-cols-2 gap-4">
              <LineChart v-if="attentionDistributionSeries && attentionDistributionSeries.length > 0"
                :series="attentionDistributionSeries" title="Attention Distribution" x-label="Position"
                y-label="Average Attention Weight" :show-legend="false" height="300px" />

              <LineChart v-if="entropySeries && entropySeries.length > 0" :series="entropySeries"
                title="Attention Entropy by Position" x-label="Query Position" y-label="Entropy" :show-legend="false"
                height="300px" />
            </div>
          </div>
        </template>
      </InteractiveSection>
    </template>

    <!-- Quiz Section -->
    <template #quiz>
      <QuizLoadingState v-if="quizLoading" />
      <QuizErrorDisplay v-else-if="quizError" :error="quizError" @retry="retryQuizLoad" />
      <QuizErrorDisplay v-else-if="!quizData" :coming-soon="true" />
      <Quiz v-else :quiz="quizData" :passing-score="quizData.passingScore || 70" @quiz-completed="handleQuizComplete" />
    </template>
  </ModuleLayout>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { useModuleStore } from '~/stores/module';
import { useProgressStore } from '~/stores/progress';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { Attention } from '~/algorithms/advanced/Attention';
import LineChart from '~/components/visualizations/LineChart.vue';
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
  queryCollection('content').path('/modules/advanced/attention-mechanisms/content').first()
);


// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const progressStore = useProgressStore();

// Parameters
const parameters = ref({
  sequenceLength: 8,
  embeddingDim: 16,
  numHeads: 4,
  attentionType: 'self',
  temperature: 1.0,
  dataset: 'synthetic-sequences',
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'dataset',
    type: 'select',
    label: 'Dataset',
    options: [
      { label: 'Synthetic Sequences', value: 'synthetic-sequences' },
      { label: 'Text Sequences', value: 'text-sequences' },
      { label: 'Pattern Sequences', value: 'pattern-sequences' },
    ],
    tooltip: 'Choose a sequence dataset',
    group: 'Data',
  },
  {
    id: 'attentionType',
    type: 'select',
    label: 'Attention Type',
    options: [
      { label: 'Self-Attention', value: 'self' },
      { label: 'Multi-Head Attention', value: 'multi-head' },
    ],
    tooltip: 'Type of attention mechanism',
    group: 'Architecture',
  },
  {
    id: 'sequenceLength',
    type: 'slider',
    label: 'Sequence Length',
    min: 4,
    max: 16,
    step: 1,
    tooltip: 'Length of input sequences',
    group: 'Architecture',
  },
  {
    id: 'embeddingDim',
    type: 'slider',
    label: 'Embedding Dimension',
    min: 8,
    max: 64,
    step: 4,
    tooltip: 'Dimension of token embeddings',
    group: 'Architecture',
  },
  {
    id: 'numHeads',
    type: 'slider',
    label: 'Number of Heads',
    min: 1,
    max: 8,
    step: 1,
    tooltip: 'Number of attention heads (for multi-head)',
    group: 'Architecture',
  },
  {
    id: 'temperature',
    type: 'slider',
    label: 'Temperature',
    min: 0.1,
    max: 2.0,
    step: 0.1,
    tooltip: 'Temperature for softmax (higher = more uniform)',
    group: 'Parameters',
  },
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);
const trainingTime = ref(0);

// Results
const metrics = ref<Metrics | null>(null);
const attentionWeightsViz = ref<any>(null);
const multiHeadViz = ref<any[]>([]);
const attentionDistributionSeries = ref<LineSeries[]>([]);
const entropySeries = ref<LineSeries[]>([]);

// Quiz data
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

// Algorithm instance
let algorithm: Attention | null = null;

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
  moduleStore.setCurrentModule('attention-mechanisms');
  moduleStore.setParameters(parameters.value);

  // Load default dataset
  await loadInitialDataset();

  // Load quiz data
  await loadQuiz('attention-mechanisms', '/content/modules/advanced/attention-mechanisms/quiz.json');
});

/**
 * Generate synthetic sequence data
 */
const generateSyntheticData = (datasetType: string) => {
  const seqLength = parameters.value.sequenceLength;
  const embeddingDim = parameters.value.embeddingDim;
  const n = 50;
  const data = [];

  for (let i = 0; i < n; i++) {
    const sequence = [];

    for (let j = 0; j < seqLength; j++) {
      const embedding = Array(embeddingDim).fill(0).map(() => Math.random() * 2 - 1);
      sequence.push(embedding);
    }

    data.push({
      features: sequence.flat(),
      label: 0,
      id: `seq-${i}`,
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

  // Reload dataset if sequence length or embedding dim changed
  if (id === 'dataset' || id === 'sequenceLength' || id === 'embeddingDim') {
    await loadInitialDataset();
    // Clear previous results
    metrics.value = null;
    attentionWeightsViz.value = null;
    multiHeadViz.value = [];
    attentionDistributionSeries.value = [];
    entropySeries.value = [];
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    sequenceLength: 8,
    embeddingDim: 16,
    numHeads: 4,
    attentionType: 'self',
    temperature: 1.0,
    dataset: 'synthetic-sequences',
  };
  moduleStore.setParameters(parameters.value);

  await loadInitialDataset();

  metrics.value = null;
  attentionWeightsViz.value = null;
  multiHeadViz.value = [];
  attentionDistributionSeries.value = [];
  entropySeries.value = [];
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

    // Prepare data - reshape to sequences
    const sequences = dataset.map(d => {
      const flat = d.features;
      const seq = [];
      for (let i = 0; i < parameters.value.sequenceLength; i++) {
        seq.push(flat.slice(i * parameters.value.embeddingDim, (i + 1) * parameters.value.embeddingDim));
      }
      return seq;
    });

    // Create and compute attention
    algorithm = new Attention({
      sequenceLength: parameters.value.sequenceLength,
      embeddingDim: parameters.value.embeddingDim,
      numHeads: parameters.value.numHeads,
      attentionType: parameters.value.attentionType as 'self' | 'multi-head',
      temperature: parameters.value.temperature,
    });

    // Compute attention
    const startTime = Date.now();
    await algorithm.fit(sequences);
    const endTime = Date.now();
    trainingTime.value = endTime - startTime;

    console.log(`Attention computation completed in ${trainingTime.value}ms`);

    // Get results
    metrics.value = algorithm.getMetrics();
    const visualizations = algorithm.getVisualizationData();

    // Process visualizations
    if (Array.isArray(visualizations)) {
      for (const viz of visualizations) {
        if (viz.type === 'heatmap') {
          if (viz.data?.attentionWeights) {
            attentionWeightsViz.value = viz.data.attentionWeights;
          }
          if (viz.data?.multiHead) {
            multiHeadViz.value = viz.data.multiHead;
          }
        } else if (viz.type === 'line') {
          if (viz.data?.distribution) {
            attentionDistributionSeries.value = [{
              x: viz.data.distribution.x,
              y: viz.data.distribution.y,
              name: 'Attention',
              color: '#6366f1',
              width: 2,
            }];
          }
          if (viz.data?.entropy) {
            entropySeries.value = [{
              x: viz.data.entropy.x,
              y: viz.data.entropy.y,
              name: 'Entropy',
              color: '#8b5cf6',
              width: 2,
            }];
          }
        }
      }
    }

    trainingProgress.value = 1;
  } catch (err) {
    console.error('Attention computation error:', err);
    error.value = err instanceof Error ? err.message : 'Computation failed. Please try again.';
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
    attentionWeightsViz.value = null;
    multiHeadViz.value = [];
    attentionDistributionSeries.value = [];
    entropySeries.value = [];

    // auto-train with loaded parameters
    await handleTrain();
  } catch (err) {
    console.error('Error loading bookmark:', err);
    error.value = 'Failed to load bookmark configuration';
  }
};
</script>