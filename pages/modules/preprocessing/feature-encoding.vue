<template>
  <ModuleLayout module-id="feature-encoding" title="Feature Encoding" domain="preprocessing" difficulty="intermediate"
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
              <Button @click="applyEncoding" :disabled="isProcessing">
                {{ isProcessing ? 'Processing...' : 'Apply Encoding' }}
              </Button>
            </template>
          </ControlPanel>
        </template>

        <template #visualization>
          <div class="space-y-4">
            <!-- Encoding Method Info -->
            <div v-if="result" class="bg-blue-50 p-4 rounded-lg">
              <h3 class="text-lg font-semibold mb-2">Encoding Summary</h3>
              <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span class="font-medium">Method:</span>
                  <span class="ml-2">{{ result.method }}</span>
                </div>
                <div>
                  <span class="font-medium">Original Features:</span>
                  <span class="ml-2">{{ result.statistics.originalFeatures }}</span>
                </div>
                <div>
                  <span class="font-medium">Encoded Features:</span>
                  <span class="ml-2">{{ result.statistics.encodedFeatures }}</span>
                </div>
                <div>
                  <span class="font-medium">Efficiency:</span>
                  <span class="ml-2">{{ result.statistics.encodingEfficiency.toFixed(2) }}x</span>
                </div>
              </div>
            </div>

            <!-- Original Data Table -->
            <div v-if="result">
              <h3 class="text-lg font-semibold mb-2">Original Categorical Data (First 10 Rows)</h3>
              <div class="overflow-x-auto">
                <table class="min-w-full border border-gray-300 text-sm">
                  <thead class="bg-gray-100">
                    <tr>
                      <th class="border border-gray-300 px-3 py-2">Sample</th>
                      <th v-for="(_, idx) in result.original[0]" :key="idx" class="border border-gray-300 px-3 py-2">
                        Feature {{ idx + 1 }}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(row, idx) in result.original.slice(0, 10)" :key="idx">
                      <td class="border border-gray-300 px-3 py-2 font-medium">{{ idx + 1 }}</td>
                      <td v-for="(val, colIdx) in row" :key="colIdx" class="border border-gray-300 px-3 py-2">
                        {{ val }}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Encoded Data Heatmap -->
            <div v-if="encodedHeatmap">
              <h3 class="text-lg font-semibold mb-2">Encoded Data Visualization</h3>
              <ClientOnly>
              <BaseChart :data="encodedHeatmap" :layout="heatmapLayout" />
              </ClientOnly>
            </div>

            <!-- Feature Names -->
            <div v-if="result && result.featureNames.length > 0" class="bg-gray-50 p-4 rounded-lg">
              <h3 class="text-lg font-semibold mb-2">Encoded Feature Names</h3>
              <div class="flex flex-wrap gap-2">
                <span v-for="(name, idx) in result.featureNames" :key="idx"
                  class="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                  {{ name }}
                </span>
              </div>
            </div>

            <!-- Statistics -->
            <div v-if="result" class="bg-gray-50 p-4 rounded-lg">
              <h3 class="text-lg font-semibold mb-3">Encoding Statistics</h3>
              <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <h4 class="font-medium mb-2">Unique Values per Feature</h4>
                  <div class="space-y-1 text-sm">
                    <div v-for="(count, idx) in result.statistics.uniqueValues" :key="idx">
                      <span class="font-medium">Feature {{ idx + 1 }}:</span> {{ count }} categories
                    </div>
                  </div>
                </div>
                <div>
                  <h4 class="font-medium mb-2">Dimensionality</h4>
                  <div class="space-y-1 text-sm">
                    <div>
                      <span class="font-medium">Before:</span> {{ result.statistics.originalFeatures }} features
                    </div>
                    <div>
                      <span class="font-medium">After:</span> {{ result.statistics.encodedFeatures }} features
                    </div>
                    <div>
                      <span class="font-medium">Change:</span>
                      <span :class="result.statistics.encodedFeatures > result.statistics.originalFeatures ? 'text-red-600' : 'text-green-600'">
                        {{ result.statistics.encodedFeatures > result.statistics.originalFeatures ? '+' : '' }}{{ result.statistics.encodedFeatures - result.statistics.originalFeatures }}
                      </span>
                    </div>
                  </div>
                </div>
                <div>
                  <h4 class="font-medium mb-2">Encoding Info</h4>
                  <div class="space-y-1 text-sm">
                    <div>
                      <span class="font-medium">Method:</span> {{ result.method }}
                    </div>
                    <div>
                      <span class="font-medium">Samples:</span> {{ result.encoded.length }}
                    </div>
                    <div>
                      <span class="font-medium">Sparsity:</span> {{ calculateSparsity().toFixed(1) }}%
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
      <Quiz v-else :quiz="quizData" :passing-score="quizData.passingScore || 70" @quiz-completed="handleQuizComplete" />
    </template>
  </ModuleLayout>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { FeatureEncoding, type EncodingMethod } from '~/algorithms/preprocessing/FeatureEncoding';
import { useQuizLoader } from '~/composables/useQuizLoader';
import type { QuizResult } from '~/types/quiz';
import { useAsyncData } from '#app';
import { useBookmarkRestore } from '~/composables/useBookmarkRestore';

// Load content from Nuxt Content
const { data: content } = await useAsyncData(() =>
  queryCollection('content').path('/modules/preprocessing/feature-encoding/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Component state
const params = ref({
  method: 'one-hot' as EncodingMethod,
  numSamples: 50,
  numFeatures: 3,
  categoriesPerFeature: [3, 4, 2],
});

const isProcessing = ref(false);
const encoder = ref<FeatureEncoding | null>(null);
const result = ref<any>(null);

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

// Controls configuration
const controls = computed(() => [
  {
    id: 'method',
    type: 'select' as const,
    label: 'Encoding Method',
    options: [
      { label: 'One-Hot Encoding', value: 'one-hot' },
      { label: 'Label Encoding', value: 'label' },
      { label: 'Ordinal Encoding', value: 'ordinal' },
      { label: 'Binary Encoding', value: 'binary' },
      { label: 'Target Encoding', value: 'target' },
    ],
    tooltip: 'Choose the encoding method to apply to categorical features',
  },
  {
    id: 'numSamples',
    type: 'slider' as const,
    label: 'Number of Samples',
    min: 20,
    max: 100,
    step: 10,
    tooltip: 'Number of data samples to generate',
  },
  {
    id: 'numFeatures',
    type: 'slider' as const,
    label: 'Number of Features',
    min: 2,
    max: 5,
    step: 1,
    tooltip: 'Number of categorical features',
  },
]);

// Generate sample categorical data
const generateData = () => {
  const categoriesPerFeature = params.value.categoriesPerFeature.slice(0, params.value.numFeatures);
  
  // Pad with default values if needed
  while (categoriesPerFeature.length < params.value.numFeatures) {
    categoriesPerFeature.push(3);
  }
  
  return FeatureEncoding.generateCategoricalData(
    params.value.numSamples,
    params.value.numFeatures,
    categoriesPerFeature
  );
};

// Generate target values for target encoding
const generateTargetValues = (numSamples: number): number[] => {
  return Array.from({ length: numSamples }, () => Math.random() > 0.5 ? 1 : 0);
};

// Apply encoding
const applyEncoding = () => {
  isProcessing.value = true;

  try {
    const data = generateData();
    encoder.value = new FeatureEncoding();
    
    const targetValues = generateTargetValues(params.value.numSamples);
    encoder.value.setData(data, targetValues);

    switch (params.value.method) {
      case 'one-hot':
        result.value = encoder.value.oneHotEncode();
        break;
      case 'label':
        result.value = encoder.value.labelEncode();
        break;
      case 'ordinal':
        result.value = encoder.value.ordinalEncode();
        break;
      case 'binary':
        result.value = encoder.value.binaryEncode();
        break;
      case 'target':
        result.value = encoder.value.targetEncode();
        break;
      default:
        result.value = encoder.value.oneHotEncode();
    }
  } catch (error) {
    console.error('Error applying encoding:', error);
  } finally {
    isProcessing.value = false;
  }
};

// Computed properties for visualizations
const encodedHeatmap = computed(() => {
  if (!result.value) return null;

  const displayRows = Math.min(20, result.value.encoded.length);
  const z = result.value.encoded.slice(0, displayRows);
  const y = Array.from({ length: displayRows }, (_, i) => `Sample ${i + 1}`);

  return [
    {
      z,
      x: result.value.featureNames,
      y,
      type: 'heatmap',
      colorscale: 'Viridis',
      showscale: true,
      hoverongaps: false,
    },
  ];
});

const heatmapLayout = computed(() => ({
  title: `${params.value.method.charAt(0).toUpperCase() + params.value.method.slice(1)} Encoding Result`,
  xaxis: {
    title: 'Encoded Features',
    tickangle: -45,
  },
  yaxis: {
    title: 'Samples',
  },
  height: 500,
  margin: { t: 60, r: 20, b: 100, l: 80 },
}));

// Calculate sparsity (percentage of zeros)
const calculateSparsity = () => {
  if (!result.value) return 0;
  
  let totalValues = 0;
  let zeroValues = 0;
  
  result.value.encoded.forEach((row: number[]) => {
    row.forEach((val: number) => {
      totalValues++;
      if (val === 0) zeroValues++;
    });
  });
  
  return totalValues > 0 ? (zeroValues / totalValues) * 100 : 0;
};

// Event handlers
const handleParamChange = (key: string, value: any) => {
  (params.value as Record<string, any>)[key] = value;
};

const resetModule = () => {
  params.value = {
    method: 'one-hot',
    numSamples: 50,
    numFeatures: 3,
    categoriesPerFeature: [3, 4, 2],
  };
  result.value = null;
  encoder.value = null;
  applyEncoding();
};

const handleQuizComplete = (quizResult: QuizResult) => {
  console.log('Quiz completed with score:', quizResult.score);
};

// Load content on mount
onMounted(async () => {
  // Load quiz data
  await loadQuiz('feature-encoding', '/content/modules/preprocessing/feature-encoding/quiz.json');

  // Apply initial encoding
  applyEncoding();
});

// Bookmark restoration
// Requirements: 2.2, 2.3
const { handleLoadBookmark } = useBookmarkRestore(params, applyEncoding);
</script>