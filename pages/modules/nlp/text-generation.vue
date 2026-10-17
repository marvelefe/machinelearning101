<template>
  <ModuleLayout title="Text Generation & Language Models"
    description="Learn how machines generate text using recurrent neural networks"
    difficulty="advanced"
    :estimated-time="45"
    module-id="text-generation"
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
      <InteractiveSection :loading="isProcessing" loading-message="Training language model..." :error="error"
        :on-retry="handleRetry">
        <!-- Controls -->
        <template #controls>
          <ControlPanel :controls="controls" :values="parameters" :loading="isProcessing" submit-label="Train Model"
            @change="handleParameterChange" @reset="handleReset" @submit="handleTrain" />

          <!-- Seed Text Input -->
          <div v-if="isTrained" class="mt-6">
            <label for="seed-text" class="block text-sm font-medium text-gray-700 mb-2">
              Seed Text
            </label>
            <input id="seed-text" v-model="seedText" type="text"
              class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              placeholder="Enter seed text to start generation..."
              :disabled="isProcessing" />
            <button @click="handleGenerate"
              class="mt-2 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
              :disabled="isProcessing || !seedText.trim()">
              Generate Text
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
                <div class="text-xs text-gray-500">Hidden Size</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.hiddenSize || 0 }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Sequence Length</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.sequenceLength || 0 }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Final Loss</div>
                <div class="text-lg font-semibold text-gray-900">{{ (metrics.finalLoss || 0).toFixed(4) }}</div>
              </div>
              <div v-if="metrics.generatedTextsCount !== undefined">
                <div class="text-xs text-gray-500">Generated Texts</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.generatedTextsCount }}</div>
              </div>
              <div v-if="metrics.averageGeneratedLength !== undefined">
                <div class="text-xs text-gray-500">Avg Length</div>
                <div class="text-lg font-semibold text-gray-900">{{ Math.round(metrics.averageGeneratedLength) }}</div>
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <!-- Text Generation Steps Display -->
          <SequenceDisplay
            v-if="generatedText"
            :generated-text="generatedText"
            :loading="isProcessing"
            :error="error"
          />
          
          <!-- Other Visualizations -->
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
import { ref, onMounted } from 'vue';
import { useModuleStore } from '~/stores/module';
import { useModules } from '~/composables/useModules';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { TextGeneration, type GeneratedText } from '~/algorithms/nlp/TextGeneration';
import VisualizationPanel from '~/components/modules/VisualizationPanel.vue';
import SequenceDisplay from '~/components/visualizations/SequenceDisplay.vue';
import Quiz from '~/components/modules/Quiz.vue';
import QuizErrorDisplay from '~/components/modules/QuizErrorDisplay.vue';
import QuizLoadingState from '~/components/modules/QuizLoadingState.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics, VisualizationData } from '~/types/algorithm';
import type { ModuleNavItem } from '~/types/module';
import type { QuizResult } from '~/types/quiz';
import { useQuizLoader } from '~/composables/useQuizLoader';

import { useAsyncData } from '#app';

const { data: content } = await useAsyncData(() => queryCollection('content').path('/modules/nlp/text-generation/content').first())

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const { getNextModule, getPreviousModule } = useModules();

// Navigation
const previousModule = ref<ModuleNavItem | null>(null);
const nextModule = ref<ModuleNavItem | null>(null);

// Training corpora
const trainingCorpora = {
  'shakespeare': `To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.`,
  'poetry': `I wandered lonely as a cloud
That floats on high o'er vales and hills,
When all at once I saw a crowd,
A host, of golden daffodils;
Beside the lake, beneath the trees,
Fluttering and dancing in the breeze.
Continuous as the stars that shine
And twinkle on the milky way,
They stretched in never-ending line
Along the margin of a bay:
Ten thousand saw I at a glance,
Tossing their heads in sprightly dance.`,
  'code': `function fibonacci(n) {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}

function factorial(n) {
  if (n === 0) return 1;
  return n * factorial(n - 1);
}

const sum = (a, b) => a + b;
const multiply = (a, b) => a * b;

for (let i = 0; i < 10; i++) {
  console.log(i);
}`
};

// Parameters
const parameters = ref({
  dataset: 'shakespeare',
  sequenceLength: 25,
  temperature: 0.8,
  maxLength: 200,
  learningRate: 0.01,
  epochs: 100,
  hiddenSize: 100,
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'dataset',
    type: 'select',
    label: 'Training Corpus',
    options: [
      { label: 'Shakespeare', value: 'shakespeare' },
      { label: 'Poetry', value: 'poetry' },
      { label: 'Code Snippets', value: 'code' },
    ],
    tooltip: 'Choose a text corpus to train the language model',
    group: 'Data',
  },
  {
    id: 'sequenceLength',
    type: 'slider',
    label: 'Sequence Length',
    min: 10,
    max: 50,
    step: 5,
    tooltip: 'Length of input sequences for training (longer = more context)',
    group: 'Model',
  },
  {
    id: 'hiddenSize',
    type: 'slider',
    label: 'Hidden Layer Size',
    min: 50,
    max: 200,
    step: 25,
    tooltip: 'Size of the RNN hidden layer (larger = more capacity)',
    group: 'Model',
  },
  {
    id: 'temperature',
    type: 'slider',
    label: 'Sampling Temperature',
    min: 0.2,
    max: 2.0,
    step: 0.1,
    tooltip: 'Controls randomness: low = conservative, high = creative',
    group: 'Generation',
  },
  {
    id: 'maxLength',
    type: 'slider',
    label: 'Max Generation Length',
    min: 50,
    max: 500,
    step: 50,
    tooltip: 'Maximum number of characters to generate',
    group: 'Generation',
  },
  {
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.001,
    max: 0.1,
    step: 0.001,
    tooltip: 'Controls how quickly the model learns',
    group: 'Training',
  },
  {
    id: 'epochs',
    type: 'slider',
    label: 'Training Epochs',
    min: 20,
    max: 200,
    step: 20,
    tooltip: 'Number of times to iterate through the training data',
    group: 'Training',
  },
];

// Processing state
const isProcessing = ref(false);
const isTrained = ref(false);
const error = ref<string | null>(null);

// Seed text
const seedText = ref("To be or not");

// Results
const metrics = ref<Metrics | null>(null);
const generatedText = ref<GeneratedText | null>(null);
const visualizations = ref<VisualizationData[]>([]);

// Algorithm instance
let algorithm: TextGeneration | null = null;

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('text-generation');
  moduleStore.setParameters(parameters.value);

  // Load navigation
  previousModule.value = await getPreviousModule('text-generation');
  nextModule.value = await getNextModule('text-generation');

  // Load quiz data
  await loadQuiz('text-generation', '/content/modules/nlp/text-generation/quiz.json');

  // Train initial model
  await handleTrain();
});

/**
 * Handle parameter change
 */
const handleParameterChange = async (id: string, value: any) => {
  parameters.value[id] = value;
  moduleStore.setParameter(id, value);

  // Update algorithm temperature if already trained
  if (id === 'temperature' && algorithm && isTrained.value) {
    algorithm['options'].temperature = value;
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    dataset: 'shakespeare',
    sequenceLength: 25,
    temperature: 0.8,
    maxLength: 200,
    learningRate: 0.01,
    epochs: 100,
    hiddenSize: 100,
  };
  moduleStore.setParameters(parameters.value);

  isTrained.value = false;
  metrics.value = null;
  generatedText.value = null;
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
    // Get training corpus
    const corpus = trainingCorpora[parameters.value.dataset as keyof typeof trainingCorpora];

    // Create and configure algorithm
    algorithm = new TextGeneration({
      sequenceLength: parameters.value.sequenceLength,
      temperature: parameters.value.temperature,
      maxLength: parameters.value.maxLength,
      learningRate: parameters.value.learningRate,
      epochs: parameters.value.epochs,
      hiddenSize: parameters.value.hiddenSize,
    });

    // Train model
    await algorithm.trainOnText(corpus);

    // Get metrics
    metrics.value = algorithm.getMetrics();

    // Get visualizations (training loss)
    const vizData = algorithm.getVisualizationData();
    visualizations.value = Array.isArray(vizData) ? vizData : [vizData];

    isTrained.value = true;

    // Set default seed based on dataset
    if (parameters.value.dataset === 'shakespeare') {
      seedText.value = "To be or not";
    } else if (parameters.value.dataset === 'poetry') {
      seedText.value = "I wandered";
    } else {
      seedText.value = "function ";
    }

    // Automatically generate text
    await handleGenerate();

  } catch (err) {
    console.error('Training error:', err);
    error.value = err instanceof Error ? err.message : 'Training failed. Please try again.';
    isTrained.value = false;
  } finally {
    isProcessing.value = false;
  }
};

/**
 * Handle generate
 */
const handleGenerate = async () => {
  if (!algorithm || !isTrained.value) {
    error.value = 'Please train the model first';
    return;
  }

  isProcessing.value = true;
  error.value = null;

  try {
    if (!seedText.value.trim()) {
      throw new Error('Please enter seed text');
    }

    // Generate text
    generatedText.value = algorithm.generateText(seedText.value, parameters.value.maxLength);

    // Update visualizations with generation steps
    const vizData = algorithm.getVisualizationData();
    visualizations.value = Array.isArray(vizData) ? vizData : [vizData];

    // Update metrics
    metrics.value = algorithm.getMetrics();

  } catch (err) {
    console.error('Generation error:', err);
    error.value = err instanceof Error ? err.message : 'Generation failed. Please try again.';
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
