<template>
  <ModuleLayout module-id="feature-selection" title="Feature Selection" domain="preprocessing" difficulty="intermediate"
    :table-of-contents="tableOfContents"
    :parameters="params"
    @load-bookmark="(val) => { params = val as typeof params; handleLoadBookmark(val); }">
    <template #content>
      <ContentRenderer v-if="content" :value="content" class="prose prose-slate max-w-none" />
    </template>

    <template #interactive>
      <InteractiveSection>
        <template #controls>
          <ControlPanel :controls="controls" :values="params" @change="handleParamChange" @reset="resetModule">
            <template #actions>
              <Button @click="applySelection" :disabled="isProcessing">
                {{ isProcessing ? 'Processing...' : 'Apply Selection' }}
              </Button>
            </template>
          </ControlPanel>
        </template>

        <template #visualization>
          <div class="space-y-4">
            <!-- Feature Scores Chart -->
            <div v-if="result">
              <h3 class="text-lg font-semibold mb-2">Feature Importance Scores</h3>
              <ClientOnly> 
              <BaseChart :data="featureScoresData" :layout="featureScoresLayout" />
              </ClientOnly>
            </div>

            <!-- Statistics Comparison -->
            <div v-if="result" class="bg-gray-50 p-4 rounded-lg">
              <h3 class="text-lg font-semibold mb-3">Selection Statistics</h3>
              <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <div class="text-sm text-gray-600">Original Features</div>
                  <div class="text-2xl font-bold">{{ result.statistics.originalFeatureCount }}</div>
                </div>
                <div>
                  <div class="text-sm text-gray-600">Selected Features</div>
                  <div class="text-2xl font-bold text-green-600">{{ result.statistics.selectedFeatureCount }}</div>
                </div>
                <div>
                  <div class="text-sm text-gray-600">Selection Ratio</div>
                  <div class="text-2xl font-bold">{{ (result.statistics.selectionRatio * 100).toFixed(1) }}%</div>
                </div>
                <div>
                  <div class="text-sm text-gray-600">Avg Score</div>
                  <div class="text-2xl font-bold">{{ result.statistics.averageScore.toFixed(3) }}</div>
                </div>
              </div>
            </div>

            <!-- Selected Features Heatmap -->
            <div v-if="result && result.selectedFeatures.length > 0">
              <h3 class="text-lg font-semibold mb-2">Selected Features Heatmap</h3>
              <ClientOnly>
              <BaseChart :data="heatmapData" :layout="heatmapLayout" />
              </ClientOnly>
            </div>

            <!-- Feature Comparison -->
            <div v-if="result" class="bg-gray-50 p-4 rounded-lg">
              <h3 class="text-lg font-semibold mb-3">Selected Feature Indices</h3>
              <div class="flex flex-wrap gap-2">
                <span v-for="idx in result.selectedIndices" :key="idx"
                  class="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
                  Feature {{ idx }}
                </span>
              </div>
            </div>
          </div>
        </template>
      </InteractiveSection>
    </template>

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
import { ref, computed, onMounted } from 'vue';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { FeatureSelection, type SelectionMethod, type FeatureSelectionResult } from '~/algorithms/preprocessing/FeatureSelection';
import { useQuizLoader } from '~/composables/useQuizLoader';
import type { QuizResult } from '~/types/quiz';
import { useAsyncData } from '#app';
import { useBookmarkRestore } from '~/composables/useBookmarkRestore';

// Load content from Nuxt Content
const { data: content } = await useAsyncData(() =>
  queryCollection('content').path('/modules/preprocessing/feature-selection/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Component state
const params = ref({
  method: 'correlation' as SelectionMethod,
  numSamples: 100,
  relevantFeatures: 5,
  irrelevantFeatures: 10,
  threshold: 0.3,
  k: 5,
});

const isProcessing = ref(false);
const selector = ref<FeatureSelection | null>(null);
const result = ref<FeatureSelectionResult | null>(null);

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

// Controls configuration
const controls = computed(() => [
  {
    id: 'method',
    type: 'select' as const,
    label: 'Selection Method',
    options: [
      { label: 'Variance Threshold', value: 'variance' },
      { label: 'Correlation', value: 'correlation' },
      { label: 'Mutual Information', value: 'mutual-info' },
      { label: 'F-Score (ANOVA)', value: 'f-score' },
      { label: 'Recursive Feature Elimination', value: 'rfe' },
    ],
    tooltip: 'Choose the feature selection method to apply',
  },
  {
    id: 'numSamples',
    type: 'slider' as const,
    label: 'Number of Samples',
    min: 50,
    max: 200,
    step: 10,
    tooltip: 'Number of data points to generate',
  },
  {
    id: 'relevantFeatures',
    type: 'slider' as const,
    label: 'Relevant Features',
    min: 3,
    max: 10,
    step: 1,
    tooltip: 'Number of features that correlate with the target',
  },
  {
    id: 'irrelevantFeatures',
    type: 'slider' as const,
    label: 'Irrelevant Features',
    min: 5,
    max: 20,
    step: 1,
    tooltip: 'Number of random noise features',
  },
  {
    id: 'threshold',
    type: 'slider' as const,
    label: 'Selection Threshold',
    min: 0.1,
    max: 0.9,
    step: 0.1,
    tooltip: 'Threshold for variance/correlation methods',
    visible: computed(() => ['variance', 'correlation'].includes(params.value.method)),
  },
  {
    id: 'k',
    type: 'slider' as const,
    label: 'Number of Features (k)',
    min: 1,
    max: 15,
    step: 1,
    tooltip: 'Number of top features to select',
    visible: computed(() => ['mutual-info', 'f-score', 'rfe'].includes(params.value.method)),
  },
]);

// Generate sample data with relevant and irrelevant features
const generateData = () => {
  return FeatureSelection.generateDataWithIrrelevantFeatures(
    params.value.numSamples,
    params.value.relevantFeatures,
    params.value.irrelevantFeatures,
    0.1
  );
};

// Apply feature selection
const applySelection = () => {
  isProcessing.value = true;

  try {
    const { data, target } = generateData();
    selector.value = new FeatureSelection();
    selector.value.setData(data, target, params.value.threshold);

    // Apply the selected method
    switch (params.value.method) {
      case 'variance':
        result.value = selector.value.selectByVarianceThreshold(params.value.threshold);
        break;
      case 'correlation':
        result.value = selector.value.selectByCorrelation(params.value.threshold);
        break;
      case 'mutual-info':
        result.value = selector.value.selectByMutualInformation(params.value.k);
        break;
      case 'f-score':
        result.value = selector.value.selectByFScore(params.value.k);
        break;
      case 'rfe':
        result.value = selector.value.selectByRFE(params.value.k);
        break;
      default:
        result.value = selector.value.selectByCorrelation(params.value.threshold);
    }
  } catch (error) {
    console.error('Error applying feature selection:', error);
  } finally {
    isProcessing.value = false;
  }
};

// Computed properties for visualizations
const featureScoresData = computed(() => {
  if (!result.value) return [];

  const totalFeatures = result.value.featureScores.length;
  const selectedSet = new Set(result.value.selectedIndices);

  return [
    {
      x: Array.from({ length: totalFeatures }, (_, i) => i),
      y: result.value.featureScores,
      type: 'bar',
      marker: {
        color: Array.from({ length: totalFeatures }, (_, i) =>
          selectedSet.has(i) ? '#10b981' : '#94a3b8'
        ),
      },
      name: 'Feature Scores',
    },
  ];
});

const featureScoresLayout = computed(() => ({
  title: `Feature Scores (${params.value.method})`,
  xaxis: { title: 'Feature Index' },
  yaxis: { title: 'Score' },
  height: 350,
  margin: { t: 40, r: 20, b: 40, l: 50 },
  showlegend: false,
}));

const heatmapData = computed(() => {
  if (!result.value || result.value.selectedFeatures.length === 0) return [];

  // Show first 20 samples for visualization
  const samplesToShow = Math.min(20, result.value.selectedFeatures.length);
  const dataSlice = result.value.selectedFeatures.slice(0, samplesToShow);

  return [
    {
      z: dataSlice,
      x: result.value.selectedIndices.map(i => `F${i}`),
      y: Array.from({ length: samplesToShow }, (_, i) => `S${i + 1}`),
      type: 'heatmap',
      colorscale: 'Viridis',
      showscale: true,
    },
  ];
});

const heatmapLayout = computed(() => ({
  title: 'Selected Features Heatmap',
  xaxis: { title: 'Selected Features' },
  yaxis: { title: 'Samples' },
  height: 400,
  margin: { t: 40, r: 20, b: 60, l: 50 },
}));

// Event handlers
const handleParamChange = (key: string, value: any) => {
  (params.value as Record<string, any>)[key] = value;
};

const resetModule = () => {
  params.value = {
    method: 'correlation',
    numSamples: 100,
    relevantFeatures: 5,
    irrelevantFeatures: 10,
    threshold: 0.3,
    k: 5,
  };
  result.value = null;
  selector.value = null;
  applySelection();
};

const handleQuizComplete = (quizResult: QuizResult) => {
  console.log('Quiz completed with score:', quizResult.score);
};

// Load content on mount
onMounted(async () => {
  // Load quiz data
  await loadQuiz('feature-selection', '/content/modules/preprocessing/feature-selection/quiz.json');

  // Apply initial selection
  applySelection();
});

// Bookmark restoration
// Requirements: 2.2, 2.3
const { handleLoadBookmark } = useBookmarkRestore(params, applySelection);
</script>