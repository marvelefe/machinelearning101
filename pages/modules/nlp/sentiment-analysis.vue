<template>
  <ModuleLayout title="Sentiment Analysis"
    description="Learn how to classify text by sentiment using machine learning"
    difficulty="intermediate"
    :estimated-time="40"
    module-id="sentiment-analysis"
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
      <InteractiveSection :loading="isProcessing" loading-message="Training sentiment classifier..." :error="error"
        :on-retry="handleRetry">
        <!-- Controls -->
        <template #controls>
          <ControlPanel :controls="controls" :values="parameters" :loading="isProcessing" submit-label="Train Model"
            @change="handleParameterChange" @reset="handleReset" @submit="handleTrain" />

          <!-- Test Text Input -->
          <div v-if="isTrained" class="mt-6">
            <label for="test-text" class="block text-sm font-medium text-gray-700 mb-2">
              Test Text (one per line)
            </label>
            <textarea id="test-text" v-model="testText" rows="4"
              class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              placeholder="Enter text to analyze sentiment...&#10;Example: This product is amazing!&#10;Example: Terrible experience, very disappointed."
              :disabled="isProcessing"></textarea>
            <button @click="handlePredict"
              class="mt-2 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
              :disabled="isProcessing || !testText.trim()">
              Analyze Sentiment
            </button>
          </div>

          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Model Statistics</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Vocabulary Size</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.vocabularySize || 0 }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Training Size</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.trainingSize || 0 }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Sentiment Classes</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.sentimentClasses || 0 }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Final Loss</div>
                <div class="text-lg font-semibold text-gray-900">{{ (metrics.finalLoss || 0).toFixed(4) }}</div>
              </div>
              <div v-if="metrics.accuracy !== undefined">
                <div class="text-xs text-gray-500">Training Accuracy</div>
                <div class="text-lg font-semibold text-gray-900">{{ ((metrics.accuracy || 0) * 100).toFixed(1) }}%
                </div>
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <VisualizationPanel :visualizations="visualizations" :loading="isProcessing" />
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
import { ref, onMounted, computed } from 'vue';
import { useModuleStore } from '~/stores/module';
import { useModules } from '~/composables/useModules';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { SentimentAnalysis, type SentimentPrediction } from '~/algorithms/nlp/SentimentAnalysis';
import VisualizationPanel from '~/components/modules/VisualizationPanel.vue';
import Quiz from '~/components/modules/Quiz.vue';
import QuizErrorDisplay from '~/components/modules/QuizErrorDisplay.vue';
import QuizLoadingState from '~/components/modules/QuizLoadingState.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics, VisualizationData } from '~/types/algorithm';
import type { ModuleNavItem } from '~/types/module';
import type { QuizResult } from '~/types/quiz';
import { useQuizLoader } from '~/composables/useQuizLoader';

const { data: content } = await useAsyncData(() => 
  queryCollection('content').path('/modules/nlp/sentiment-analysis/content').first()
)

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const { getNextModule, getPreviousModule } = useModules();

// Navigation
const previousModule = ref<ModuleNavItem | null>(null);
const nextModule = ref<ModuleNavItem | null>(null);

// Training datasets
const trainingDatasets = {
  'movie-reviews': {
    texts: [
      "This movie was absolutely amazing! Best film I've seen all year.",
      "Terrible movie, complete waste of time and money.",
      "The movie was okay, nothing special but not bad either.",
      "Loved every minute of it! Brilliant acting and great story.",
      "Awful experience. The plot made no sense at all.",
      "It was fine. Some good parts, some boring parts.",
      "Fantastic! A masterpiece of cinema.",
      "Disappointing and boring. Expected much better.",
      "Pretty good overall, enjoyed most of it.",
      "Horrible. One of the worst movies ever made.",
      "Excellent film with great performances.",
      "Not great, not terrible. Just average.",
      "Wonderful movie! Highly recommend it.",
      "Very poor quality. Don't waste your time.",
      "Decent movie, worth watching once.",
      "Outstanding! A true work of art.",
      "Bad acting and weak storyline.",
      "It's alright, nothing to write home about.",
      "Superb! Exceeded all my expectations.",
      "Dreadful. Couldn't even finish watching it."
    ],
    labels: [2, 0, 1, 2, 0, 1, 2, 0, 2, 0, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0]
  },
  'product-reviews': {
    texts: [
      "Great product! Works exactly as described.",
      "Poor quality, broke after one week.",
      "It's okay for the price, nothing special.",
      "Love it! Best purchase I've made this year.",
      "Terrible quality, complete waste of money.",
      "Decent product, does what it's supposed to.",
      "Excellent! Highly recommend to everyone.",
      "Disappointing, not worth the price.",
      "Good value for money, satisfied with purchase.",
      "Awful product, returned it immediately.",
      "Perfect! Exactly what I needed.",
      "Average quality, nothing impressive.",
      "Amazing product! Exceeded expectations.",
      "Very poor, don't buy this.",
      "Fair product, gets the job done.",
      "Outstanding quality and performance!",
      "Bad design, difficult to use.",
      "It's fine, meets basic requirements.",
      "Fantastic! Would buy again.",
      "Horrible experience, very disappointed."
    ],
    labels: [2, 0, 1, 2, 0, 1, 2, 0, 2, 0, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0]
  },
  'social-media': {
    texts: [
      "Having an amazing day! Everything is going great!",
      "This is so frustrating. Nothing is working.",
      "Just another day, nothing exciting happening.",
      "Best news ever! So happy right now!",
      "Worst day of my life. Everything went wrong.",
      "It's okay, could be better could be worse.",
      "Feeling fantastic! Life is wonderful!",
      "Really annoyed and disappointed today.",
      "Pretty good day overall, can't complain.",
      "Absolutely terrible. So angry right now.",
      "Loving life! Everything is perfect!",
      "Meh, just a regular boring day.",
      "Incredible! This is the best!",
      "Hate this. So done with everything.",
      "Not bad, just a normal day.",
      "Awesome! Things are going so well!",
      "Ugh, this is awful and annoying.",
      "It's fine, nothing special today.",
      "Brilliant! Couldn't be happier!",
      "Terrible experience. Very upset."
    ],
    labels: [2, 0, 1, 2, 0, 1, 2, 0, 2, 0, 2, 1, 2, 0, 1, 2, 0, 1, 2, 0]
  }
};

// Parameters
const parameters = ref({
  dataset: 'movie-reviews',
  sentimentClasses: 3,
  learningRate: 0.1,
  epochs: 50,
  regularization: 0.01,
  minWordFrequency: 2,
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'dataset',
    type: 'select',
    label: 'Training Dataset',
    options: [
      { label: 'Movie Reviews', value: 'movie-reviews' },
      { label: 'Product Reviews', value: 'product-reviews' },
      { label: 'Social Media', value: 'social-media' },
    ],
    tooltip: 'Choose a dataset to train the sentiment classifier',
    group: 'Data',
  },
  {
    id: 'sentimentClasses',
    type: 'select',
    label: 'Sentiment Classes',
    options: [
      { label: '2 Classes (Positive/Negative)', value: 2 },
      { label: '3 Classes (Positive/Neutral/Negative)', value: 3 },
    ],
    tooltip: 'Number of sentiment categories to classify',
    group: 'Model',
  },
  {
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.01,
    max: 0.5,
    step: 0.01,
    tooltip: 'Controls how quickly the model learns from data',
    group: 'Training',
  },
  {
    id: 'epochs',
    type: 'slider',
    label: 'Training Epochs',
    min: 10,
    max: 100,
    step: 10,
    tooltip: 'Number of times to iterate through the training data',
    group: 'Training',
  },
  {
    id: 'regularization',
    type: 'slider',
    label: 'Regularization',
    min: 0,
    max: 0.1,
    step: 0.01,
    tooltip: 'Prevents overfitting by penalizing large weights',
    group: 'Training',
  },
  {
    id: 'minWordFrequency',
    type: 'slider',
    label: 'Min Word Frequency',
    min: 1,
    max: 10,
    step: 1,
    tooltip: 'Minimum times a word must appear to be included in vocabulary',
    group: 'Preprocessing',
  },
];

// Processing state
const isProcessing = ref(false);
const isTrained = ref(false);
const error = ref<string | null>(null);

// Test text
const testText = ref("This is amazing! I love it!\nTerrible experience, very disappointed.\nIt's okay, nothing special.");

// Results
const metrics = ref<Metrics | null>(null);
const predictions = ref<SentimentPrediction[]>([]);
const visualizations = ref<VisualizationData[]>([]);

// Algorithm instance
let algorithm: SentimentAnalysis | null = null;

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('sentiment-analysis');
  moduleStore.setParameters(parameters.value);

  // Load navigation
  previousModule.value = await getPreviousModule('sentiment-analysis');
  nextModule.value = await getNextModule('sentiment-analysis');

  // Load quiz data
  await loadQuiz('sentiment-analysis', '/content/modules/nlp/sentiment-analysis/quiz.json');

  // Train initial model
  await handleTrain();
});

/**
 * Handle parameter change
 */
const handleParameterChange = async (id: string, value: any) => {
  parameters.value[id] = value;
  moduleStore.setParameter(id, value);
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    dataset: 'movie-reviews',
    sentimentClasses: 3,
    learningRate: 0.1,
    epochs: 50,
    regularization: 0.01,
    minWordFrequency: 2,
  };
  moduleStore.setParameters(parameters.value);

  isTrained.value = false;
  metrics.value = null;
  predictions.value = [];
  visualizations.value = [];
  error.value = null;
};

/**
 * Handle train
 */
const handleTrain = async () => {
  isProcessing.value = true;
  error.value = null;

  try {
    // Get training data
    const dataset = trainingDatasets[parameters.value.dataset as keyof typeof trainingDatasets];

    // Adjust labels for 2-class classification
    let labels = dataset.labels;
    if (parameters.value.sentimentClasses === 2) {
      // Map: 0 (negative) -> 0, 1 (neutral) -> 1, 2 (positive) -> 1
      // Or better: remove neutral samples
      const filtered = dataset.texts
        .map((text, i) => ({ text, label: dataset.labels[i] }))
        .filter(item => item.label !== 1); // Remove neutral

      const texts = filtered.map(item => item.text);
      labels = filtered.map(item => item.label === 0 ? 0 : 1); // 0: negative, 1: positive

      // Create and configure algorithm
      algorithm = new SentimentAnalysis({
        sentimentClasses: parameters.value.sentimentClasses,
        learningRate: parameters.value.learningRate,
        epochs: parameters.value.epochs,
        regularization: parameters.value.regularization,
        minWordFrequency: parameters.value.minWordFrequency,
      });

      // Train model
      await algorithm.trainOnText(texts, labels);
    } else {
      // Create and configure algorithm
      algorithm = new SentimentAnalysis({
        sentimentClasses: parameters.value.sentimentClasses,
        learningRate: parameters.value.learningRate,
        epochs: parameters.value.epochs,
        regularization: parameters.value.regularization,
        minWordFrequency: parameters.value.minWordFrequency,
      });

      // Train model
      await algorithm.trainOnText(dataset.texts, labels);
    }

    // Get metrics
    metrics.value = algorithm.getMetrics();

    // Get visualizations (training loss, top words)
    const vizData = algorithm.getVisualizationData();
    visualizations.value = Array.isArray(vizData) ? vizData : [vizData];

    isTrained.value = true;

    // Automatically predict on test text if available
    if (testText.value.trim()) {
      await handlePredict();
    }

  } catch (err) {
    console.error('Training error:', err);
    error.value = err instanceof Error ? err.message : 'Training failed. Please try again.';
    isTrained.value = false;
  } finally {
    isProcessing.value = false;
  }
};

/**
 * Handle predict
 */
const handlePredict = async () => {
  if (!algorithm || !isTrained.value) {
    error.value = 'Please train the model first';
    return;
  }

  isProcessing.value = true;
  error.value = null;

  try {
    // Split test text by lines
    const texts = testText.value
      .split('\n')
      .map(line => line.trim())
      .filter(line => line.length > 0);

    if (texts.length === 0) {
      throw new Error('Please enter some text to analyze');
    }

    // Predict sentiment
    predictions.value = algorithm.predictSentiment(texts);

    // Update visualizations with predictions
    const vizData = algorithm.getVisualizationData();
    visualizations.value = Array.isArray(vizData) ? vizData : [vizData];

  } catch (err) {
    console.error('Prediction error:', err);
    error.value = err instanceof Error ? err.message : 'Prediction failed. Please try again.';
  } finally {
    isProcessing.value = false;
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
 * Handle load bookmark
 * Requirements: 2.2, 2.3
 */
const handleLoadBookmark = (bookmarkParams: Record<string, any>) => {
  // Update parameters with bookmark values
  Object.assign(parameters.value, bookmarkParams);
  moduleStore.setParameters(parameters.value);

  // Retrain model with loaded parameters
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
