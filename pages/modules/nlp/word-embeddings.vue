<template>
  <ModuleLayout title="Word Embeddings"
    description="Learn how word embeddings capture semantic relationships in vector space"
    difficulty="intermediate"
    :estimated-time="45"
    module-id="word-embeddings"
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
      <InteractiveSection :loading="isTraining" loading-message="Training embeddings..." :error="error" :on-retry="handleRetry">
        <!-- Controls -->
        <template #controls>
          <!-- Text Input Area -->
          <div class="mb-6">
            <label for="text-input" class="block text-sm font-medium text-gray-700 mb-2">
              Training Corpus
            </label>
            <textarea id="text-input" v-model="inputText" rows="8" class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500" placeholder="Enter text corpus to train embeddings..." :disabled="isTraining"></textarea>
          </div>

          <ControlPanel :controls="controls" :values="parameters" :loading="isTraining" submit-label="Train Embeddings" @change="handleParameterChange" @reset="handleReset" @submit="handleTrain" />

          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-indigo-50 rounded-lg border border-indigo-200">
            <h3 class="text-sm font-semibold text-indigo-700 mb-3">Embedding Statistics</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-indigo-500">Vocabulary Size</div>
                <div class="text-lg font-semibold text-indigo-900">{{ metrics.vocabularySize || 0 }}</div>
              </div>
              <div>
                <div class="text-xs text-indigo-500">Embedding Dimension</div>
                <div class="text-lg font-semibold text-indigo-900">{{ metrics.embeddingDimension || 0 }}</div>
              </div>
              <div>
                <div class="text-xs text-indigo-500">Corpus Size</div>
                <div class="text-lg font-semibold text-indigo-900">{{ metrics.corpusSize || 0 }} tokens</div>
              </div>
              <div>
                <div class="text-xs text-indigo-500">Training Epochs</div>
                <div class="text-lg font-semibold text-indigo-900">{{ metrics.epochs || 0 }}</div>
              </div>
              <div>
                <div class="text-xs text-indigo-500">Final Loss</div>
                <div class="text-lg font-semibold text-indigo-900">{{ (metrics.finalLoss || 0).toFixed(4) }}</div>
              </div>
              <div>
                <div class="text-xs text-indigo-500">Avg Word Frequency</div>
                <div class="text-lg font-semibold text-indigo-900">{{ (metrics.averageWordFrequency || 0).toFixed(2) }}</div>
              </div>
            </div>
          </div>

          <!-- Word Search -->
          <div v-if="algorithm && algorithm.trained" class="mt-6">
            <label for="word-search" class="block text-sm font-medium text-gray-700 mb-2">
              Find Similar Words
            </label>
            <div class="flex gap-2">
              <input id="word-search" v-model="searchWord" type="text" class="flex-1 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500" placeholder="Enter a word..." @keyup.enter="handleWordSearch" />
              <button @click="handleWordSearch" class="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2" :disabled="!searchWord.trim()">
                Search
              </button>
            </div>

            <!-- Similar Words Display -->
            <div v-if="similarWords.length > 0" class="mt-4 p-4 bg-white rounded-lg border border-gray-200">
              <h4 class="text-sm font-semibold text-gray-700 mb-2">
                Words similar to "{{ searchWord }}":
              </h4>
              <div class="flex flex-wrap gap-2">
                <span v-for="sim in similarWords" :key="sim.word" class="px-3 py-1 bg-indigo-100 text-indigo-800 rounded-full text-sm font-medium">
                  {{ sim.word }} ({{ (sim.similarity * 100).toFixed(1) }}%)
                </span>
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-6">
            <!-- Embedding Projection -->
            <EmbeddingProjection v-if="embeddingData" :words="embeddingData.words" :coordinates2d="embeddingData.coordinates2d" :coordinates3d="embeddingData.coordinates3d" :frequencies="embeddingData.frequencies" title="Word Embedding Space" :on-word-select="handleWordSelect" @word-click="handleWordClick" />

            <!-- Training Loss Chart -->
            <div v-if="lossChartData">
              <LineChart :series="lossChartData" title="Training Loss Over Epochs" x-label="Epoch" y-label="Loss" :show-legend="false" height="400px" />
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
import { useModules } from '~/composables/useModules';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { WordEmbeddings } from '~/algorithms/nlp/WordEmbeddings';
import EmbeddingProjection from '~/components/visualizations/EmbeddingProjection.vue';
import LineChart from '~/components/visualizations/LineChart.vue';
import Quiz from '~/components/modules/Quiz.vue';
import QuizErrorDisplay from '~/components/modules/QuizErrorDisplay.vue';
import QuizLoadingState from '~/components/modules/QuizLoadingState.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { LineSeries } from '~/components/visualizations/LineChart.vue';
import type { ModuleNavItem } from '~/types/module';
import type { QuizResult } from '~/types/quiz';
import { useQuizLoader } from '~/composables/useQuizLoader';

const { data: content } = await useAsyncData(() => 
  queryCollection('content').path('/modules/nlp/word-embeddings/content').first()
)

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const { getNextModule, getPreviousModule } = useModules();

// Navigation
const previousModule = ref<ModuleNavItem | null>(null);
const nextModule = ref<ModuleNavItem | null>(null);

// Sample texts
const sampleTexts = {
  sample1: "Machine learning and artificial intelligence are transforming technology. Deep learning models use neural networks to learn patterns. Computer vision enables machines to see and understand images. Natural language processing helps computers understand human language. Data science combines statistics and programming to extract insights from data.",
  sample2: "The cat sat on the mat. The dog played in the park. Birds fly in the sky. Fish swim in the ocean. Elephants walk on land. Lions roar in the jungle. Dolphins jump in the water. Butterflies flutter among flowers.",
  sample3: "Cooking delicious food requires fresh ingredients. Baking bread needs flour and yeast. Grilling meat creates smoky flavors. Steaming vegetables preserves nutrients. Frying eggs makes breakfast easy. Boiling pasta is simple and quick. Roasting chicken produces crispy skin. Sautéing onions adds flavor to dishes."
};

// Input text
const inputText = ref(sampleTexts.sample1);

// Parameters
const parameters = ref({
  sampleText: 'sample1',
  embeddingDim: 50,
  windowSize: 2,
  minCount: 2,
  learningRate: 0.025,
  epochs: 5,
  negSamples: 5,
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'sampleText',
    type: 'select',
    label: 'Sample Corpus',
    options: [
      { label: 'Technology Text', value: 'sample1' },
      { label: 'Nature & Animals', value: 'sample2' },
      { label: 'Food & Cooking', value: 'sample3' },
    ],
    tooltip: 'Choose a sample text corpus to train embeddings',
    group: 'Input',
  },
  {
    id: 'embeddingDim',
    type: 'slider',
    label: 'Embedding Dimension',
    min: 10,
    max: 100,
    step: 10,
    tooltip: 'Dimensionality of the word vectors (higher = more expressive but slower)',
    group: 'Model Architecture',
  },
  {
    id: 'windowSize',
    type: 'slider',
    label: 'Context Window Size',
    min: 1,
    max: 5,
    step: 1,
    tooltip: 'Number of words to consider on each side of the target word',
    group: 'Training',
  },
  {
    id: 'minCount',
    type: 'slider',
    label: 'Minimum Word Count',
    min: 1,
    max: 5,
    step: 1,
    tooltip: 'Minimum frequency for a word to be included in vocabulary',
    group: 'Preprocessing',
  },
  {
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.001,
    max: 0.1,
    step: 0.001,
    tooltip: 'Step size for gradient updates during training',
    group: 'Training',
  },
  {
    id: 'epochs',
    type: 'slider',
    label: 'Training Epochs',
    min: 1,
    max: 20,
    step: 1,
    tooltip: 'Number of times to iterate through the entire corpus',
    group: 'Training',
  },
  {
    id: 'negSamples',
    type: 'slider',
    label: 'Negative Samples',
    min: 1,
    max: 10,
    step: 1,
    tooltip: 'Number of negative samples per positive sample (negative sampling)',
    group: 'Training',
  },
];

// Training state
const isTraining = ref(false);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const embeddingData = ref<{
  words: string[];
  coordinates2d: number[][];
  coordinates3d: number[][];
  frequencies: number[];
} | null>(null);
const lossChartData = ref<LineSeries[] | null>(null);

// Word search
const searchWord = ref('');
const similarWords = ref<Array<{ word: string; similarity: number }>>([]);

// Algorithm instance
let algorithm: WordEmbeddings | null = null;

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('word-embeddings');
  moduleStore.setParameters(parameters.value);

  // Load navigation
  previousModule.value = await getPreviousModule('word-embeddings');
  nextModule.value = await getNextModule('word-embeddings');

  // Load quiz data
  await loadQuiz('word-embeddings', '/content/modules/nlp/word-embeddings/quiz.json');

  // Train initial embeddings
  await handleTrain();
});

/**
 * Handle parameter change
 */
const handleParameterChange = async (id: string, value: any) => {
  parameters.value[id] = value;
  moduleStore.setParameter(id, value);

  // Update input text if sample changed
  if (id === 'sampleText') {
    inputText.value = sampleTexts[value as keyof typeof sampleTexts];
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    sampleText: 'sample1',
    embeddingDim: 50,
    windowSize: 2,
    minCount: 2,
    learningRate: 0.025,
    epochs: 5,
    negSamples: 5,
  };
  moduleStore.setParameters(parameters.value);

  inputText.value = sampleTexts.sample1;

  metrics.value = null;
  embeddingData.value = null;
  lossChartData.value = null;
  similarWords.value = [];
  searchWord.value = '';
  error.value = null;
};

/**
 * Handle train
 */
const handleTrain = async () => {
  isTraining.value = true;
  error.value = null;

  try {
    // Validate input
    if (!inputText.value || inputText.value.trim().length === 0) {
      throw new Error('Please enter some text to train embeddings');
    }

    // Create and configure algorithm
    algorithm = new WordEmbeddings({
      embeddingDim: parameters.value.embeddingDim,
      windowSize: parameters.value.windowSize,
      minCount: parameters.value.minCount,
      learningRate: parameters.value.learningRate,
      epochs: parameters.value.epochs,
      negSamples: parameters.value.negSamples,
    });

    // Train embeddings
    await algorithm.trainOnText(inputText.value);

    // Get results
    metrics.value = algorithm.getMetrics();

    // Get visualization data
    const vizData = algorithm.getVisualizationData();
    const vizArray = Array.isArray(vizData) ? vizData : [vizData];

    // Extract embedding projection data
    const projection2D = vizArray.find(v => v.type === 'scatter2d');
    const projection3D = vizArray.find(v => v.type === 'scatter3d');

    if (projection2D && projection3D) {
      embeddingData.value = {
        words: projection2D.data.words,
        coordinates2d: projection2D.data.x.map((x: number, i: number) => [x, projection2D.data.y[i]]),
        coordinates3d: projection3D.data.x.map((x: number, i: number) => [x, projection3D.data.y[i], projection3D.data.z[i]]),
        frequencies: projection2D.data.frequencies,
      };
    }

    // Extract loss chart data
    const lossViz = vizArray.find(v => v.type === 'line');
    if (lossViz && lossViz.data.datasets) {
      lossChartData.value = [{
        x: lossViz.data.labels,
        y: lossViz.data.datasets[0].data,
        name: 'Training Loss',
        color: '#6366f1',
        width: 2,
      }];
    }

  } catch (err) {
    console.error('Training error:', err);
    error.value = err instanceof Error ? err.message : 'Training failed. Please try again.';
  } finally {
    isTraining.value = false;
  }
};

/**
 * Handle word search
 */
const handleWordSearch = () => {
  if (!algorithm || !searchWord.value.trim()) return;

  try {
    const similar = algorithm.findSimilar(searchWord.value.trim(), 5);
    similarWords.value = similar;
  } catch (err) {
    console.error('Search error:', err);
    similarWords.value = [];
  }
};

/**
 * Handle word selection from visualization
 */
const handleWordSelect = (word: string) => {
  if (!algorithm) return [];

  try {
    return algorithm.findSimilar(word, 5);
  } catch (err) {
    console.error('Word select error:', err);
    return [];
  }
};

/**
 * Handle word click from visualization
 */
const handleWordClick = (word: string) => {
  searchWord.value = word;
  handleWordSearch();
};

/**
 * Handle retry
 */
const handleRetry = () => {
  error.value = null;
  handleTrain();
};

/**
 * Handle load bookmark
 * Requirements: 2.2, 2.3
 */
const handleLoadBookmark = (bookmarkParams: Record<string, any>) => {
  // Update parameters with bookmark values
  Object.assign(parameters.value, bookmarkParams);
  moduleStore.setParameters(parameters.value);

  // Update input text if sample changed
  if (bookmarkParams.sampleText) {
    inputText.value = sampleTexts[bookmarkParams.sampleText as keyof typeof sampleTexts];
  }

  // Retrain embeddings with loaded parameters
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
</script>
