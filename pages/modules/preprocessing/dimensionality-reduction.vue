<template>
  <ModuleLayout 
    module-id="dimensionality-reduction" 
    title="Dimensionality Reduction" 
    domain="preprocessing" 
    difficulty="advanced"
    :table-of-contents="tableOfContents"
    :parameters="params"
    @load-bookmark="(val) => { params = val as typeof params; handleLoadBookmark(val); }"
  >
    <template #content>
      <ContentRenderer v-if="content" :value="content" class="prose prose-slate max-w-none" />
    </template>

    <template #interactive>
      <InteractiveSection>
        <template #controls>
          <ControlPanel :controls="controls" :values="params" @change="handleParamChange" @reset="resetModule">
            <template #actions>
              <Button @click="applyReduction" :disabled="isProcessing">
                {{ isProcessing ? 'Processing...' : 'Apply Reduction' }}
              </Button>
            </template>
          </ControlPanel>
        </template>

        <template #visualization>
          <div class="space-y-4">
            <!-- Method Info -->
            <div v-if="result" class="bg-blue-50 p-4 rounded-lg">
              <h3 class="text-lg font-semibold mb-2">Reduction Statistics</h3>
              <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span class="font-medium">Original Dimensions:</span>
                  <div class="text-2xl font-bold text-blue-600">{{ result.statistics.originalDimensions }}</div>
                </div>
                <div>
                  <span class="font-medium">Reduced Dimensions:</span>
                  <div class="text-2xl font-bold text-green-600">{{ result.statistics.reducedDimensions }}</div>
                </div>
                <div>
                  <span class="font-medium">Variance Retained:</span>
                  <div class="text-2xl font-bold text-purple-600">{{ (result.statistics.varianceRetained * 100).toFixed(1) }}%</div>
                </div>
                <div>
                  <span class="font-medium">Compression Ratio:</span>
                  <div class="text-2xl font-bold text-orange-600">{{ (result.statistics.compressionRatio * 100).toFixed(1) }}%</div>
                </div>
              </div>
            </div>

            <!-- Scree Plot -->
            <div v-if="screePlotData">
              <h3 class="text-lg font-semibold mb-2">Scree Plot - Explained Variance</h3>
              <ClientOnly> 
              <BaseChart :data="screePlotData" :layout="screePlotLayout" />
              </ClientOnly>
            </div>

            <!-- Cumulative Variance -->
            <div v-if="cumulativeVariancePlotData">
              <h3 class="text-lg font-semibold mb-2">Cumulative Explained Variance</h3>
              <ClientOnly> 
              <BaseChart :data="cumulativeVariancePlotData" :layout="cumulativeVarianceLayout" />
              </ClientOnly>
            </div>

            <!-- 2D Projection -->
            <div v-if="projectionData">
              <h3 class="text-lg font-semibold mb-2">2D Projection of Reduced Data</h3>
              <ClientOnly>
              <BaseChart :data="projectionData" :layout="projectionLayout" />
              </ClientOnly>
            </div>

            <!-- Component Loadings (for first 2 components) -->
            <div v-if="result && result.components.length >= 2" class="bg-gray-50 p-4 rounded-lg">
              <h3 class="text-lg font-semibold mb-3">Component Loadings</h3>
              <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 class="font-medium mb-2">Component 1</h4>
                  <div class="space-y-1 text-sm max-h-40 overflow-y-auto">
                    <div v-for="(loading, idx) in result.components[0].slice(0, 10)" :key="idx" class="flex justify-between">
                      <span>Feature {{ idx + 1 }}:</span>
                      <span class="font-mono">{{ loading.toFixed(3) }}</span>
                    </div>
                  </div>
                </div>
                <div>
                  <h4 class="font-medium mb-2">Component 2</h4>
                  <div class="space-y-1 text-sm max-h-40 overflow-y-auto">
                    <div v-for="(loading, idx) in result.components[1].slice(0, 10)" :key="idx" class="flex justify-between">
                      <span>Feature {{ idx + 1 }}:</span>
                      <span class="font-mono">{{ loading.toFixed(3) }}</span>
                    </div>
                  </div>
                </div>
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
      <Quiz 
        v-else 
        :quiz="quizData" 
        :passing-score="quizData.passingScore || 70"
        @quiz-completed="handleQuizComplete" 
      />
    </template>
  </ModuleLayout>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { DimensionalityReduction, type ReductionMethod, type DimensionalityReductionResult } from '~/algorithms/preprocessing/DimensionalityReduction';
import { useQuizLoader } from '~/composables/useQuizLoader';
import type { QuizResult } from '~/types/quiz';
import { useAsyncData } from '#app';
import { useBookmarkRestore } from '~/composables/useBookmarkRestore';

// Load content from Nuxt Content
const { data: content } = await useAsyncData(() =>
  queryCollection('content').path('/modules/preprocessing/dimensionality-reduction/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Component state
const params = ref({
  method: 'pca' as ReductionMethod,
  numSamples: 200,
  originalDimensions: 10,
  intrinsicDimensions: 2,
  nComponents: 2,
  noiseLevel: 0.1,
});

const isProcessing = ref(false);
const reducer = ref<DimensionalityReduction | null>(null);
const result = ref<DimensionalityReductionResult | null>(null);
const generatedData = ref<{ data: number[][], target?: number[] } | null>(null);

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

// Controls configuration
const controls = computed(() => [
  {
    id: 'method',
    type: 'select' as const,
    label: 'Reduction Method',
    options: [
      { label: 'PCA (Principal Component Analysis)', value: 'pca' },
      { label: 'LDA (Linear Discriminant Analysis)', value: 'lda' },
      { label: 'ICA (Independent Component Analysis)', value: 'ica' },
      { label: 'Truncated SVD', value: 'truncated-svd' },
    ],
    tooltip: 'Choose the dimensionality reduction technique to apply',
  },
  {
    id: 'numSamples',
    type: 'slider' as const,
    label: 'Number of Samples',
    min: 100,
    max: 500,
    step: 50,
    tooltip: 'Number of data points to generate',
  },
  {
    id: 'originalDimensions',
    type: 'slider' as const,
    label: 'Original Dimensions',
    min: 5,
    max: 20,
    step: 1,
    tooltip: 'Number of features in the high-dimensional data',
  },
  {
    id: 'intrinsicDimensions',
    type: 'slider' as const,
    label: 'Intrinsic Dimensions',
    min: 2,
    max: 5,
    step: 1,
    tooltip: 'True underlying dimensionality of the data',
  },
  {
    id: 'nComponents',
    type: 'slider' as const,
    label: 'Target Components',
    min: 2,
    max: 10,
    step: 1,
    tooltip: 'Number of components to reduce to',
  },
  {
    id: 'noiseLevel',
    type: 'slider' as const,
    label: 'Noise Level',
    min: 0,
    max: 1,
    step: 0.1,
    tooltip: 'Amount of noise to add to the data',
  },
]);

// Generate high-dimensional data
const generateData = () => {
  return DimensionalityReduction.generateHighDimensionalData(
    params.value.numSamples,
    params.value.originalDimensions,
    params.value.intrinsicDimensions,
    params.value.noiseLevel
  );
};

// Apply dimensionality reduction
const applyReduction = async () => {
  isProcessing.value = true;

  try {
    // Generate data
    generatedData.value = generateData();
    
    // Create reducer instance
    reducer.value = new DimensionalityReduction();
    
    // Apply the selected method
    const method = params.value.method;
    const nComponents = Math.min(params.value.nComponents, params.value.originalDimensions);
    
    if (method === 'pca') {
      reducer.value.setData(generatedData.value.data, generatedData.value.target, nComponents);
      result.value = reducer.value.applyPCA(nComponents);
    } else if (method === 'lda') {
      if (!generatedData.value.target || generatedData.value.target.length === 0) {
        throw new Error('LDA requires labeled data');
      }
      reducer.value.setData(generatedData.value.data, generatedData.value.target, nComponents);
      result.value = reducer.value.applyLDA(nComponents);
    } else if (method === 'ica') {
      reducer.value.setData(generatedData.value.data, generatedData.value.target, nComponents);
      result.value = reducer.value.applyICA(nComponents);
    } else if (method === 'truncated-svd') {
      reducer.value.setData(generatedData.value.data, generatedData.value.target, nComponents);
      result.value = reducer.value.applyTruncatedSVD(nComponents);
    }
  } catch (error) {
    console.error('Error applying dimensionality reduction:', error);
    result.value = null;
  } finally {
    isProcessing.value = false;
  }
};

// Computed properties for visualizations
const screePlotData = computed(() => {
  if (!result.value || result.value.explainedVariance.length === 0) return null;

  return [
    {
      x: Array.from({ length: result.value.explainedVariance.length }, (_, i) => i + 1),
      y: result.value.explainedVariance,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Explained Variance',
      line: { color: '#3b82f6', width: 2 },
      marker: { size: 8, color: '#3b82f6' },
    },
  ];
});

const screePlotLayout = {
  title: 'Scree Plot',
  xaxis: { title: 'Component Number' },
  yaxis: { title: 'Explained Variance' },
  height: 350,
  margin: { t: 40, r: 20, b: 50, l: 60 },
};

const cumulativeVariancePlotData = computed(() => {
  if (!result.value || result.value.cumulativeVariance.length === 0) return null;

  return [
    {
      x: Array.from({ length: result.value.cumulativeVariance.length }, (_, i) => i + 1),
      y: result.value.cumulativeVariance.map(v => v * 100),
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Cumulative Variance',
      line: { color: '#ef4444', width: 2 },
      marker: { size: 8, color: '#ef4444' },
      fill: 'tozeroy',
      fillcolor: 'rgba(239, 68, 68, 0.1)',
    },
  ];
});

const cumulativeVarianceLayout = {
  title: 'Cumulative Explained Variance',
  xaxis: { title: 'Number of Components' },
  yaxis: { title: 'Cumulative Variance (%)', range: [0, 105] },
  height: 350,
  margin: { t: 40, r: 20, b: 50, l: 60 },
};

const projectionData = computed(() => {
  if (!result.value || result.value.reducedData.length === 0) return null;
  if (result.value.reducedData[0].length < 2) return null;

  const colors = generatedData.value?.target && generatedData.value.target.length > 0
    ? generatedData.value.target.map(val => val === 1 ? '#10b981' : '#3b82f6')
    : Array(result.value.reducedData.length).fill('#3b82f6');

  return [
    {
      x: result.value.reducedData.map(row => row[0]),
      y: result.value.reducedData.map(row => row[1]),
      type: 'scatter',
      mode: 'markers',
      marker: {
        color: colors,
        size: 6,
        opacity: 0.7,
        line: { color: 'white', width: 0.5 },
      },
      name: 'Reduced Data',
    },
  ];
});

const projectionLayout = computed(() => ({
  title: `${params.value.method.toUpperCase()} - 2D Projection`,
  xaxis: { title: 'First Component', zeroline: true },
  yaxis: { title: 'Second Component', zeroline: true },
  height: 450,
  margin: { t: 40, r: 20, b: 50, l: 60 },
}));

// Event handlers
const handleParamChange = (key: string, value: any) => {
  (params.value as Record<string, any>)[key] = value;
  
  // Ensure nComponents doesn't exceed originalDimensions
  if (key === 'originalDimensions' && params.value.nComponents > value) {
    params.value.nComponents = Math.min(value, 10);
  }
};

const resetModule = () => {
  params.value = {
    method: 'pca',
    numSamples: 200,
    originalDimensions: 10,
    intrinsicDimensions: 2,
    nComponents: 2,
    noiseLevel: 0.1,
  };
  result.value = null;
  reducer.value = null;
  generatedData.value = null;
  applyReduction();
};

const handleQuizComplete = (quizResult: QuizResult) => {
  console.log('Quiz completed with score:', quizResult.score);
};

// Load content on mount
onMounted(async () => {
  // Load quiz data
  await loadQuiz('dimensionality-reduction', '/content/modules/preprocessing/dimensionality-reduction/quiz.json');

  // Apply initial reduction
  await applyReduction();
});

// Bookmark restoration
// Requirements: 2.2, 2.3
const { handleLoadBookmark } = useBookmarkRestore(params, applyReduction);
</script>
