<template>
  <ModuleLayout title="Named Entity Recognition" description="Learn how to identify and classify named entities in text"
    difficulty="intermediate" :estimated-time="45" module-id="named-entity-recognition"
    :table-of-contents="tableOfContents" :parameters="parameters"
    @load-bookmark="(val) => { parameters = val as typeof parameters; handleLoadBookmark(val); }">
    <!-- Content Section -->
    <template #content>
      <ContentRenderer v-if="content" :value="content" class="prose prose-slate max-w-none" />
    </template>

    <!-- Interactive Section -->
    <template #interactive>
      <InteractiveSection :loading="isProcessing" loading-message="Extracting entities..." :error="error"
        :on-retry="handleRetry">
        <!-- Controls -->
        <template #controls>
          <ControlPanel :controls="controls" :values="parameters" :loading="isProcessing"
            submit-label="Extract Entities" @change="handleParameterChange" @reset="handleReset"
            @submit="handleExtract" />

          <!-- Test Text Input -->
          <div class="mt-6">
            <label for="test-text" class="block text-sm font-medium text-gray-700 mb-2">
              Text to Analyze
            </label>
            <textarea id="test-text" v-model="testText" rows="6"
              class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-mono text-sm"
              placeholder="Enter text to extract entities from...&#10;Example: Apple Inc. announced that Tim Cook will visit Paris on January 15, 2024."
              :disabled="isProcessing"></textarea>
            <button @click="handleExtract"
              class="mt-2 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
              :disabled="isProcessing || !testText.trim()">
              Extract Entities
            </button>
          </div>

          <!-- Entity Results -->
          <div v-if="predictions.length > 0" class="mt-6">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Extracted Entities</h3>
            <div v-for="(pred, idx) in predictions" :key="idx"
              class="mb-4 p-4 bg-white rounded-lg border border-gray-200">
              <div class="mb-2 text-sm text-gray-600">Text {{ idx + 1 }}:</div>
              <div class="mb-3 p-3 bg-gray-50 rounded text-sm font-mono" v-html="pred.highlightedText"></div>
              <div v-if="pred.entities.length > 0" class="space-y-2">
                <div v-for="(entity, eIdx) in pred.entities" :key="eIdx"
                  class="flex items-center justify-between text-sm p-2 bg-gray-50 rounded">
                  <div class="flex items-center space-x-3">
                    <span class="font-semibold text-gray-900">{{ entity.text }}</span>
                    <span :class="getEntityTypeClass(entity.type)" class="px-2 py-1 rounded text-xs font-medium">
                      {{ entity.type }}
                    </span>
                  </div>
                  <div class="text-xs text-gray-500">
                    {{ (entity.confidence * 100).toFixed(0) }}% confidence
                  </div>
                </div>
              </div>
              <div v-else class="text-sm text-gray-500 italic">No entities found</div>
            </div>
          </div>

          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Extraction Statistics</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Entity Types</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.entityTypes || 0 }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Gazetteer Size</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.gazetteerSize || 0 }}</div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Patterns Count</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.patternsCount || 0 }}</div>
              </div>
              <div v-if="metrics.totalEntitiesExtracted !== undefined">
                <div class="text-xs text-gray-500">Total Entities</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.totalEntitiesExtracted }}</div>
              </div>
              <div v-if="metrics.averageConfidence !== undefined">
                <div class="text-xs text-gray-500">Avg Confidence</div>
                <div class="text-lg font-semibold text-gray-900">{{ (metrics.averageConfidence * 100).toFixed(1) }}%
                </div>
              </div>
              <div v-if="metrics.averageEntitiesPerText !== undefined">
                <div class="text-xs text-gray-500">Avg Entities/Text</div>
                <div class="text-lg font-semibold text-gray-900">{{ metrics.averageEntitiesPerText.toFixed(1) }}</div>
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
      <Quiz v-else :quiz="quizData" :passing-score="quizData.passingScore || 70" @quiz-completed="handleQuizComplete" />
    </template>
  </ModuleLayout>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { useModuleStore } from '~/stores/module';
import { useModules } from '~/composables/useModules';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { NamedEntityRecognition, type Entity, type NERPrediction } from '~/algorithms/nlp/NamedEntityRecognition';
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
  queryCollection('content').path('/modules/nlp/named-entity-recognition/content').first()
)

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores and composables
const moduleStore = useModuleStore();
const { getNextModule, getPreviousModule } = useModules();

// Navigation
const previousModule = ref<ModuleNavItem | null>(null);
const nextModule = ref<ModuleNavItem | null>(null);

// Sample training data
const trainingData = {
  texts: [
    "Apple Inc. announced that Tim Cook will visit Paris on January 15, 2024.",
    "Microsoft Corporation CEO Satya Nadella spoke at the conference in Seattle.",
    "The meeting with Dr. Jane Smith is scheduled for 3:00 PM tomorrow.",
    "Amazon opened a new office in New York City last month.",
    "President Biden will address the nation from Washington D.C.",
  ],
  entities: [
    [
      { text: "Apple Inc.", type: "ORGANIZATION", start: 0, end: 10, confidence: 0.95 },
      { text: "Tim Cook", type: "PERSON", start: 26, end: 34, confidence: 0.92 },
      { text: "Paris", type: "LOCATION", start: 46, end: 51, confidence: 0.88 },
      { text: "January 15, 2024", type: "DATE", start: 55, end: 71, confidence: 0.98 }
    ],
    [
      { text: "Microsoft Corporation", type: "ORGANIZATION", start: 0, end: 21, confidence: 0.96 },
      { text: "Satya Nadella", type: "PERSON", start: 26, end: 39, confidence: 0.91 },
      { text: "Seattle", type: "LOCATION", start: 68, end: 75, confidence: 0.87 }
    ],
    [
      { text: "Dr. Jane Smith", type: "PERSON", start: 17, end: 31, confidence: 0.93 },
      { text: "3:00 PM", type: "TIME", start: 49, end: 56, confidence: 0.97 }
    ],
    [
      { text: "Amazon", type: "ORGANIZATION", start: 0, end: 6, confidence: 0.94 },
      { text: "New York City", type: "LOCATION", start: 28, end: 41, confidence: 0.89 }
    ],
    [
      { text: "President Biden", type: "PERSON", start: 0, end: 15, confidence: 0.95 },
      { text: "Washington D.C.", type: "LOCATION", start: 47, end: 62, confidence: 0.90 }
    ]
  ]
};

// Parameters
const parameters = ref({
  dataset: 'news-articles',
  entityTypes: ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE'],
  caseSensitive: true,
  useContextWindow: true,
  contextWindowSize: 5,
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'dataset',
    type: 'select',
    label: 'Text Dataset',
    options: [
      { label: 'News Articles', value: 'news-articles' },
      { label: 'Business Documents', value: 'business-docs' },
      { label: 'Social Media Posts', value: 'social-media' },
    ],
    tooltip: 'Choose a dataset to extract entities from',
    group: 'Data',
  },
  {
    id: 'caseSensitive',
    type: 'toggle',
    label: 'Case Sensitive',
    tooltip: 'Whether entity matching should be case-sensitive',
    group: 'Model',
  },
  {
    id: 'useContextWindow',
    type: 'toggle',
    label: 'Use Context Window',
    tooltip: 'Use surrounding words to improve entity classification confidence',
    group: 'Model',
  },
  {
    id: 'contextWindowSize',
    type: 'slider',
    label: 'Context Window Size',
    min: 1,
    max: 10,
    step: 1,
    tooltip: 'Number of words before and after entity to consider for context',
    group: 'Model',
  },
];

// Processing state
const isProcessing = ref(false);
const error = ref<string | null>(null);

// Test text
const testText = ref("Apple Inc. announced that Tim Cook will visit Paris on January 15, 2024. The company's headquarters in Cupertino will host a meeting with Microsoft CEO Satya Nadella next week. Dr. Jane Smith from Stanford University will present her research. Contact us at info@example.com or call +1-555-123-4567.");

// Results
const metrics = ref<Metrics | null>(null);
const predictions = ref<NERPrediction[]>([]);
const visualizations = ref<VisualizationData[]>([]);

// Algorithm instance
let algorithm: NamedEntityRecognition | null = null;

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('named-entity-recognition');
  moduleStore.setParameters(parameters.value);

  // Load navigation
  previousModule.value = await getPreviousModule('named-entity-recognition');
  nextModule.value = await getNextModule('named-entity-recognition');

  // Load quiz data
  await loadQuiz('named-entity-recognition', '/content/modules/nlp/named-entity-recognition/quiz.json');

  // Initialize algorithm
  initializeAlgorithm();

  // Extract entities from initial text
  await handleExtract();
});

/**
 * Initialize algorithm
 */
const initializeAlgorithm = () => {
  algorithm = new NamedEntityRecognition({
    entityTypes: parameters.value.entityTypes,
    caseSensitive: parameters.value.caseSensitive,
    useContextWindow: parameters.value.useContextWindow,
    contextWindowSize: parameters.value.contextWindowSize,
  });

  // Train on sample data to build gazetteer
  algorithm.trainOnAnnotatedText(trainingData.texts, trainingData.entities);
};

/**
 * Handle parameter change
 */
const handleParameterChange = async (id: string, value: any) => {
  parameters.value[id] = value;
  moduleStore.setParameter(id, value);

  // Reinitialize algorithm with new parameters
  if (id !== 'dataset') {
    initializeAlgorithm();
  }
};

/**
 * Handle reset
 */
const handleReset = async () => {
  parameters.value = {
    dataset: 'news-articles',
    entityTypes: ['PERSON', 'ORGANIZATION', 'LOCATION', 'DATE'],
    caseSensitive: true,
    useContextWindow: true,
    contextWindowSize: 5,
  };
  moduleStore.setParameters(parameters.value);

  metrics.value = null;
  predictions.value = [];
  visualizations.value = [];
  error.value = null;

  // Reinitialize algorithm
  initializeAlgorithm();
};

/**
 * Handle extract entities
 */
const handleExtract = async () => {
  if (!algorithm) {
    initializeAlgorithm();
  }

  isProcessing.value = true;
  error.value = null;

  try {
    // Split test text by sentences (simple split)
    const texts = testText.value
      .split(/[.!?]+/)
      .map(line => line.trim())
      .filter(line => line.length > 0);

    if (texts.length === 0) {
      throw new Error('Please enter some text to analyze');
    }

    // Extract entities
    predictions.value = algorithm!.extractEntities(texts);

    // Get metrics
    metrics.value = algorithm!.getMetrics();

    // Get visualizations
    const vizData = algorithm!.getVisualizationData();
    visualizations.value = Array.isArray(vizData) ? vizData : [vizData];

  } catch (err) {
    console.error('Extraction error:', err);
    error.value = err instanceof Error ? err.message : 'Entity extraction failed. Please try again.';
  } finally {
    isProcessing.value = false;
  }
};

/**
 * Handle retry
 */
const handleRetry = () => {
  error.value = null;
  handleExtract();
};

/**
 * Get entity type CSS class
 */
const getEntityTypeClass = (type: string): string => {
  const classes: { [key: string]: string } = {
    PERSON: 'bg-blue-100 text-blue-800',
    ORGANIZATION: 'bg-green-100 text-green-800',
    LOCATION: 'bg-amber-100 text-amber-800',
    DATE: 'bg-red-100 text-red-800',
    TIME: 'bg-purple-100 text-purple-800',
    MONEY: 'bg-pink-100 text-pink-800',
    EMAIL: 'bg-teal-100 text-teal-800',
    PHONE: 'bg-orange-100 text-orange-800',
  };
  return classes[type] || 'bg-gray-100 text-gray-800';
};

/**
 * Handle load bookmark
 * Requirements: 2.2, 2.3
 */
const handleLoadBookmark = (bookmarkParams: Record<string, any>) => {
  // Update parameters with bookmark values
  Object.assign(parameters.value, bookmarkParams);
  moduleStore.setParameters(parameters.value);

  // Reinitialize algorithm with loaded parameters
  initializeAlgorithm();

  // Re-extract entities with loaded parameters
  handleExtract();
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

<style scoped>
/* Entity highlighting styles */
:deep(.entity-person) {
  background-color: rgba(59, 130, 246, 0.2);
  border-bottom: 2px solid rgb(59, 130, 246);
  padding: 2px 4px;
  border-radius: 3px;
}

:deep(.entity-organization) {
  background-color: rgba(16, 185, 129, 0.2);
  border-bottom: 2px solid rgb(16, 185, 129);
  padding: 2px 4px;
  border-radius: 3px;
}

:deep(.entity-location) {
  background-color: rgba(245, 158, 11, 0.2);
  border-bottom: 2px solid rgb(245, 158, 11);
  padding: 2px 4px;
  border-radius: 3px;
}

:deep(.entity-date) {
  background-color: rgba(239, 68, 68, 0.2);
  border-bottom: 2px solid rgb(239, 68, 68);
  padding: 2px 4px;
  border-radius: 3px;
}

:deep(.entity-time) {
  background-color: rgba(139, 92, 246, 0.2);
  border-bottom: 2px solid rgb(139, 92, 246);
  padding: 2px 4px;
  border-radius: 3px;
}

:deep(.entity-money) {
  background-color: rgba(236, 72, 153, 0.2);
  border-bottom: 2px solid rgb(236, 72, 153);
  padding: 2px 4px;
  border-radius: 3px;
}

:deep(.entity-email) {
  background-color: rgba(20, 184, 166, 0.2);
  border-bottom: 2px solid rgb(20, 184, 166);
  padding: 2px 4px;
  border-radius: 3px;
}

:deep(.entity-phone) {
  background-color: rgba(251, 146, 60, 0.2);
  border-bottom: 2px solid rgb(251, 146, 60);
  padding: 2px 4px;
  border-radius: 3px;
}
</style>
