<template>
  <ModuleLayout title="Text Preprocessing & Tokenization"
    description="Learn how to tokenize and preprocess text for natural language processing tasks"
    difficulty="beginner"
    :estimated-time="35"
    module-id="text-preprocessing"
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
      <InteractiveSection :loading="isProcessing" loading-message="Processing text..." :error="error"
        :on-retry="handleRetry">
        <!-- Controls -->
        <template #controls>
          <!-- Text Input Area -->
          <div class="mb-6">
            <label for="text-input" class="block text-sm font-medium text-gray-700 mb-2">
              Input Text
            </label>
            <textarea id="text-input" v-model="inputText" rows="6"
              class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              placeholder="Enter text to tokenize..." :disabled="isProcessing"></textarea>
          </div>

          <ControlPanel :controls="controls" :values="parameters" :loading="isProcessing" submit-label="Tokenize Text"
            @change="handleParameterChange" @reset="handleReset" @submit="handleProcess" />

          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Tokenization Statistics</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Total Tokens</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.totalTokens || 0 }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Word Tokens</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.wordTokens || 0 }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Vocabulary Size</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.vocabularySize || 0 }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Unique Token Ratio</div>
                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.uniqueTokenRatio || 0) *
                  100).toFixed(1) }}%</div>
              </div>
              <div v-if="metrics.sentenceCount && metrics.sentenceCount > 0">
                <div class="text-xs text-gray-500">Sentences</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.sentenceCount }}</div>
              </div>
              <div v-if="metrics.stopwordTokens">
                <div class="text-xs text-gray-500">Stopwords</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.stopwordTokens }}</div>
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-4">
            <!-- Text Highlighter -->
            <TextHighlighter v-if="tokens.length > 0" :tokens="tokens" :sentences="sentences" :text="inputText"
              :show-legend="true" :show-stats="false" />

            <!-- Token Frequency Chart -->
            <div v-if="frequencyChartData">
              <LineChart :series="frequencyChartData" title="Top 20 Token Frequencies" x-label="Token"
                y-label="Frequency" :show-legend="false" height="400px" />
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
import { useModules } from '~/composables/useModules';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { Tokenizer, type Token, type Sentence } from '~/algorithms/nlp/Tokenizer';
import TextHighlighter from '~/components/visualizations/TextHighlighter.vue';
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

import { useAsyncData } from '#app';

const { data: content } = await useAsyncData(() => queryCollection('content').path('/modules/nlp/text-preprocessing/content').first())

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
  sample1: "Machine learning is amazing! It helps computers learn from data.",
  sample2: "Natural language processing enables computers to understand human language. It's a fascinating field. Many applications use NLP today.",
  sample3: "The transformer architecture revolutionized NLP in 2017. Self-attention mechanisms allow models to process sequences efficiently. BERT and GPT are examples of transformer-based models."
};

// Input text
const inputText = ref(sampleTexts.sample1);

// Parameters
const parameters = ref({
  lowercase: true,
  removePunctuation: false,
  removeStopwords: false,
  stemming: false,
  tokenizeBy: 'word',
  sampleText: 'sample1',
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'sampleText',
    type: 'select',
    label: 'Sample Text',
    options: [
      { label: 'Simple Sentence', value: 'sample1' },
      { label: 'Multiple Sentences', value: 'sample2' },
      { label: 'Technical Text', value: 'sample3' },
    ],
    tooltip: 'Choose a sample text to tokenize',
    group: 'Input',
  },
  {
    id: 'tokenizeBy',
    type: 'select',
    label: 'Tokenization Mode',
    options: [
      { label: 'Word', value: 'word' },
      { label: 'Sentence', value: 'sentence' },
      { label: 'Character', value: 'character' },
    ],
    tooltip: 'Choose how to split the text into tokens',
    group: 'Tokenization',
  },
  {
    id: 'lowercase',
    type: 'toggle',
    label: 'Convert to Lowercase',
    tooltip: 'Convert all text to lowercase for normalization',
    group: 'Normalization',
  },
  {
    id: 'removePunctuation',
    type: 'toggle',
    label: 'Remove Punctuation',
    tooltip: 'Remove punctuation marks from the tokenized output',
    group: 'Filtering',
  },
  {
    id: 'removeStopwords',
    type: 'toggle',
    label: 'Remove Stopwords',
    tooltip: 'Filter out common words like \'the\', \'is\', \'and\'',
    group: 'Filtering',
  },
  {
    id: 'stemming',
    type: 'toggle',
    label: 'Apply Stemming',
    tooltip: 'Reduce words to their root form (e.g., \'running\' → \'run\')',
    group: 'Normalization',
  },
];

// Processing state
const isProcessing = ref(false);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const tokens = ref<Token[]>([]);
const sentences = ref<Sentence[]>([]);
const frequencyChartData = ref<LineSeries[] | null>(null);

// Algorithm instance
let algorithm: Tokenizer | null = null;

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('text-preprocessing');
  moduleStore.setParameters(parameters.value);

  // Load navigation
  previousModule.value = await getPreviousModule('text-preprocessing');
  nextModule.value = await getNextModule('text-preprocessing');

  // Load quiz data
  await loadQuiz('text-preprocessing', '/content/modules/nlp/text-preprocessing/quiz.json');

  // Process initial text
  await handleProcess();
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
    lowercase: true,
    removePunctuation: false,
    removeStopwords: false,
    stemming: false,
    tokenizeBy: 'word',
    sampleText: 'sample1',
  };
  moduleStore.setParameters(parameters.value);

  inputText.value = sampleTexts.sample1;

  metrics.value = null;
  tokens.value = [];
  sentences.value = [];
  frequencyChartData.value = null;
  error.value = null;
};

/**
 * Handle process
 */
const handleProcess = async () => {
  isProcessing.value = true;
  error.value = null;

  try {
    // Validate input
    if (!inputText.value || inputText.value.trim().length === 0) {
      throw new Error('Please enter some text to tokenize');
    }

    // Create and configure tokenizer
    algorithm = new Tokenizer({
      lowercase: parameters.value.lowercase,
      removePunctuation: parameters.value.removePunctuation,
      removeStopwords: parameters.value.removeStopwords,
      stemming: parameters.value.stemming,
      tokenizeBy: parameters.value.tokenizeBy as 'word' | 'sentence' | 'character',
    });

    // Process text
    await algorithm.processText(inputText.value);

    // Get results
    tokens.value = algorithm.getTokens();
    sentences.value = algorithm.getSentences();
    metrics.value = algorithm.getMetrics();

    // Get token frequency data
    const tokenCounts = algorithm.getTokenCounts();
    const sortedTokens = Array.from(tokenCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20);

    if (sortedTokens.length > 0) {
      frequencyChartData.value = [{
        x: sortedTokens.map((_, i) => i),
        y: sortedTokens.map(([, count]) => count),
        name: 'Frequency',
        color: '#6366f1',
        width: 2,
      }];
    }

  } catch (err) {
    console.error('Processing error:', err);
    error.value = err instanceof Error ? err.message : 'Processing failed. Please try again.';
  } finally {
    isProcessing.value = false;
  }
};

/**
 * Handle retry
 */
const handleRetry = () => {
  error.value = null;
  handleProcess();
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

  // Reprocess text with loaded parameters
  handleProcess();
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
