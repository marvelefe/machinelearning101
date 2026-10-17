<template>
  <ModuleLayout
    module-id="missing-value-imputation"
    title="Missing Value Imputation"
    domain="preprocessing"
    difficulty="beginner"
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
          <ControlPanel
            :controls="controls"
            :values="params"
            @change="handleParamChange"
            @reset="resetModule"
          >
            <template #actions>
              <Button @click="generateNewData" class="mb-2">
                Generate New Data
              </Button>
              <Button @click="applyImputation" :disabled="isProcessing">
                {{ isProcessing ? 'Processing...' : 'Apply Imputation' }}
              </Button>
            </template>
          </ControlPanel>
        </template>

        <template #visualization> 
            <div class="space-y-4">
              <!-- Missing Data Pattern -->
              <div>
                <h3 class="text-lg font-semibold mb-2">Missing Data Pattern</h3>
                <div class="bg-gray-50 p-4 rounded-lg">
                  <div class="grid grid-cols-2 gap-4 mb-4">
                    <div>
                      <p class="text-sm font-medium">Total Values</p>
                      <p class="text-2xl font-bold">{{ totalValues }}</p>
                    </div>
                    <div>
                      <p class="text-sm font-medium">Missing Values</p>
                      <p class="text-2xl font-bold text-red-600">
                        {{ result?.missingCount || 0 }} ({{ (result?.missingPercentage || 0).toFixed(1) }}%)
                      </p>
                    </div>
                  </div>
                  <div v-if="missingPatternData" class="mt-4">
                    <ClientOnly> 
                    <BaseChart
                      :data="missingPatternData"
                      :layout="missingPatternLayout"
                    />
                    </ClientOnly>
                  </div>
                </div>
              </div>

              <!-- Before/After Comparison -->
              <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h3 class="text-lg font-semibold mb-2">Original Data (with missing)</h3>
                  <div class="bg-white border rounded-lg p-4 max-h-64 overflow-auto">
                    <table class="w-full text-sm">
                      <thead>
                        <tr>
                          <th class="text-left p-1">Row</th>
                          <th v-for="j in numFeatures" :key="j" class="text-left p-1">F{{ j }}</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr v-for="(row, i) in displayOriginalData" :key="i">
                          <td class="p-1 font-medium">{{ i + 1 }}</td>
                          <td 
                            v-for="(val, j) in row" 
                            :key="j" 
                            class="p-1"
                            :class="{ 'text-red-600 font-bold': val === null }"
                          >
                            {{ val === null ? 'NULL' : val.toFixed(2) }}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
                <div>
                  <h3 class="text-lg font-semibold mb-2">Imputed Data</h3>
                  <div class="bg-white border rounded-lg p-4 max-h-64 overflow-auto">
                    <table class="w-full text-sm">
                      <thead>
                        <tr>
                          <th class="text-left p-1">Row</th>
                          <th v-for="j in numFeatures" :key="j" class="text-left p-1">F{{ j }}</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr v-for="(row, i) in displayImputedData" :key="i">
                          <td class="p-1 font-medium">{{ i + 1 }}</td>
                          <td 
                            v-for="(val, j) in row" 
                            :key="j" 
                            class="p-1"
                            :class="{ 'text-green-600 font-bold': wasImputed(i, j) }"
                          >
                            {{ val === null ? 'NULL' : val.toFixed(2) }}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              <!-- Statistics -->
              <div v-if="result?.statistics" class="bg-gray-50 p-4 rounded-lg">
                <h3 class="text-lg font-semibold mb-3">Imputation Statistics</h3>
                <div class="grid grid-cols-3 gap-4 text-sm">
                  <div v-if="result.statistics.mean">
                    <p class="font-medium">Mean Values</p>
                    <p v-for="(val, idx) in result.statistics.mean" :key="idx">
                      Feature {{ idx + 1 }}: {{ val.toFixed(2) }}
                    </p>
                  </div>
                  <div v-if="result.statistics.median">
                    <p class="font-medium">Median Values</p>
                    <p v-for="(val, idx) in result.statistics.median" :key="idx">
                      Feature {{ idx + 1 }}: {{ val.toFixed(2) }}
                    </p>
                  </div>
                  <div v-if="result.statistics.mode">
                    <p class="font-medium">Mode Values</p>
                    <p v-for="(val, idx) in result.statistics.mode" :key="idx">
                      Feature {{ idx + 1 }}: {{ val.toFixed(2) }}
                    </p>
                  </div>
                </div>
              </div>

              <!-- Distribution Comparison -->
              <div v-if="distributionData">
                <h3 class="text-lg font-semibold mb-2">Distribution Comparison</h3>
                <ClientOnly>
                <BaseChart
                  :data="distributionData"
                  :layout="distributionLayout"
                />
                </ClientOnly>
              </div>
            </div> 
        </template>
      </InteractiveSection>
    </template>

    <template #quiz>
      <!-- Loading state -->
      <QuizLoadingState v-if="quizLoading" />
      
      <!-- Error state with retry -->
      <QuizErrorDisplay
        v-else-if="quizError"
        :error="quizError"
        @retry="retryQuizLoad"
      />
      
      <!-- Coming soon state -->
      <QuizErrorDisplay
        v-else-if="!quizData"
        :coming-soon="true"
      />
      
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
import { useAsyncData } from '#app';
import { ref, computed, onMounted } from 'vue';
import { MissingValueImputation, type ImputationMethod } from '~/algorithms/preprocessing/MissingValueImputation';
import { useTableOfContents } from '~/composables/useTableOfContents';
import type { QuizResult } from '~/types/quiz';
import { useQuizLoader } from '~/composables/useQuizLoader';
import { useBookmarkRestore } from '~/composables/useBookmarkRestore';

// Load content from Nuxt Content
const { data: content } = await useAsyncData(() => 
  queryCollection('content').path('/modules/preprocessing/missing-value-imputation/content').first()
);

// Extract table of contents
const tableOfContents = useTableOfContents(content);

// Component state
const params = ref({
  method: 'mean' as ImputationMethod,
  numSamples: 20,
  numFeatures: 3,
  missingPercentage: 20,
});

const isProcessing = ref(false);
const imputer = ref<MissingValueImputation | null>(null);
const result = ref<any>(null);

// Quiz data with loading state
// Requirements: 1.1, 2.1, 2.4, 6.1, 6.2
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

// Controls configuration
const controls = computed(() => [
  {
    id: 'method',
    type: 'select' as const,
    label: 'Imputation Method',
    options: [
      { label: 'Mean', value: 'mean' },
      { label: 'Median', value: 'median' },
      { label: 'Mode', value: 'mode' },
      { label: 'Forward Fill', value: 'forward-fill' },
    ],
    tooltip: 'Choose the strategy for filling missing values',
  },
  {
    id: 'numSamples',
    type: 'slider' as const,
    label: 'Number of Samples',
    min: 10,
    max: 50,
    step: 5,
    tooltip: 'Number of data points to generate',
  },
  {
    id: 'numFeatures',
    type: 'slider' as const,
    label: 'Number of Features',
    min: 2,
    max: 5,
    step: 1,
    tooltip: 'Number of features in the dataset',
  },
  {
    id: 'missingPercentage',
    type: 'slider' as const,
    label: 'Missing Percentage',
    min: 5,
    max: 50,
    step: 5,
    tooltip: 'Percentage of values to make missing',
  },
]);

// Generate data with missing values
const generateNewData = () => {
  const data = MissingValueImputation.generateDataWithMissing(
    params.value.numSamples,
    params.value.numFeatures,
    params.value.missingPercentage
  );
  
  imputer.value = new MissingValueImputation(data);
  result.value = imputer.value.getResult();
};

// Apply imputation
const applyImputation = () => {
  if (!imputer.value) {
    generateNewData();
  }
  
  isProcessing.value = true;
  
  try {
    switch (params.value.method) {
      case 'mean':
        result.value = imputer.value!.imputeMean();
        break;
      case 'median':
        result.value = imputer.value!.imputeMedian();
        break;
      case 'mode':
        result.value = imputer.value!.imputeMode();
        break;
      case 'forward-fill':
        result.value = imputer.value!.imputeForwardFill();
        break;
    }
  } catch (error) {
    console.error('Error applying imputation:', error);
  } finally {
    isProcessing.value = false;
  }
};

// Computed properties
const totalValues = computed(() => 
  params.value.numSamples * params.value.numFeatures
);

const numFeatures = computed(() => params.value.numFeatures);

const displayOriginalData = computed(() => 
  result.value?.original.slice(0, 10) || []
);

const displayImputedData = computed(() => 
  result.value?.imputed.slice(0, 10) || []
);

const wasImputed = (i: number, j: number): boolean => {
  return result.value?.imputedValues.has(`${i},${j}`) || false;
};

const missingPatternData = computed(() => {
  if (!result.value) return null;
  
  const pattern = result.value.original.map((row: any[]) => 
    row.map((val: any) => val === null ? 1 : 0)
  );
  
  return [{
    z: pattern,
    type: 'heatmap',
    colorscale: [[0, '#10b981'], [1, '#ef4444']],
    showscale: false,
  }];
});

const missingPatternLayout = {
  title: 'Missing Values (Red = Missing, Green = Present)',
  xaxis: { title: 'Features' },
  yaxis: { title: 'Samples' },
  height: 300,
  margin: { t: 40, r: 20, b: 40, l: 50 },
};

const distributionData = computed(() => {
  if (!result.value) return null;
  
  // Get first feature for comparison
  const originalValues = result.value.original
    .map((row: any[]) => row[0])
    .filter((val: any) => val !== null);
  
  const imputedValues = result.value.imputed
    .map((row: any[]) => row[0])
    .filter((val: any) => val !== null);
  
  return [
    {
      x: originalValues,
      type: 'histogram',
      name: 'Original (non-null)',
      opacity: 0.7,
      marker: { color: '#3b82f6' },
    },
    {
      x: imputedValues,
      type: 'histogram',
      name: 'After Imputation',
      opacity: 0.7,
      marker: { color: '#10b981' },
    },
  ];
});

const distributionLayout = {
  title: 'Feature 1 Distribution',
  xaxis: { title: 'Value' },
  yaxis: { title: 'Frequency' },
  height: 300,
  barmode: 'overlay',
  margin: { t: 40, r: 20, b: 40, l: 50 },
};

// Event handlers
const handleParamChange = (key: string, value: any) => {
  (params.value as Record<string, any>)[key] = value;
};

const resetModule = () => {
  params.value = {
    method: 'mean',
    numSamples: 20,
    numFeatures: 3,
    missingPercentage: 20,
  };
  result.value = null;
  imputer.value = null;
  generateNewData();
};

/**
 * Handle quiz completion
 * Requirements: 3.3, 3.4, 5.1, 5.2, 5.4
 */
const handleQuizComplete = (result: QuizResult) => {
  console.log('Quiz completed with score:', result.score);
  // Quiz component handles progress store updates internally
};

// Load content on mount
onMounted(async () => {
  // Load quiz data
  // Requirements: 1.1, 2.1, 2.4, 6.1, 6.2
  await loadQuiz('missing-value-imputation', '/content/modules/preprocessing/missing-value-imputation/quiz.json');
  
  // Generate initial data
  generateNewData();
  applyImputation();
});

// Bookmark restoration
// Requirements: 2.2, 2.3
const { handleLoadBookmark } = useBookmarkRestore(params, () => {
  generateNewData();
  applyImputation();
});
</script>
