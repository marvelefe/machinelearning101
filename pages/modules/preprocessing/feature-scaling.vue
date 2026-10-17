<template>
  <ModuleLayout module-id="feature-scaling" title="Feature Scaling" domain="preprocessing" difficulty="beginner"
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
              <Button @click="applyScaling" :disabled="isProcessing">
                {{ isProcessing ? 'Processing...' : 'Apply Scaling' }}
              </Button>
            </template>
          </ControlPanel>
        </template>

        <template #visualization> 
            <div class="space-y-4">
              <!-- Before/After Comparison -->
              <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h3 class="text-lg font-semibold mb-2">Original Data</h3>
                  <ClientOnly> 
                  <BaseChart v-if="originalDistribution" :data="originalDistribution"
                    :layout="chartLayout('Original Distribution')" />
                    </ClientOnly>
                </div>
                <div>
                  <h3 class="text-lg font-semibold mb-2">Scaled Data</h3>
                  <ClientOnly> 
                  <BaseChart v-if="scaledDistribution" :data="scaledDistribution"
                    :layout="chartLayout('Scaled Distribution')" />
                    </ClientOnly>
                </div>
              </div>

              <!-- Statistics Comparison -->
              <div v-if="statistics" class="bg-gray-50 p-4 rounded-lg">
                <h3 class="text-lg font-semibold mb-3">Statistics Comparison</h3>
                <div class="grid grid-cols-2 gap-4">
                  <div>
                    <h4 class="font-medium mb-2">Original Data</h4>
                    <div class="space-y-1 text-sm">
                      <div v-for="(stat, idx) in originalStats" :key="idx">
                        <span class="font-medium">{{ stat.label }}:</span> {{ stat.value }}
                      </div>
                    </div>
                  </div>
                  <div>
                    <h4 class="font-medium mb-2">Scaled Data</h4>
                    <div class="space-y-1 text-sm">
                      <div v-for="(stat, idx) in scaledStats" :key="idx">
                        <span class="font-medium">{{ stat.label }}:</span> {{ stat.value }}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <!-- Scatter Plot Comparison -->
              <div v-if="scatterData">
                <h3 class="text-lg font-semibold mb-2">Data Visualization</h3>
                <ClientOnly>
                <BaseChart :data="scatterData" :layout="scatterLayout" />
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
import { FeatureScaling, type ScalingMethod } from '~/algorithms/preprocessing/FeatureScaling';
import { useQuizLoader } from '~/composables/useQuizLoader';
import type { QuizResult } from '~/types/quiz';
import { useAsyncData } from '#app';
import { useBookmarkRestore } from '~/composables/useBookmarkRestore';



// Load content from Nuxt Content
const { data: content } = await useAsyncData(() =>
  queryCollection('content').path('/modules/preprocessing/feature-scaling/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Component state
const params = ref({
  method: 'normalize' as ScalingMethod,
  numSamples: 100,
  featureRange: 50,
});

const isProcessing = ref(false);
const scaler = ref<FeatureScaling | null>(null);
const result = ref<any>(null);

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

// Controls configuration
const controls = computed(() => [
  {
    id: 'method',
    type: 'select' as const,
    label: 'Scaling Method',
    options: [
      { label: 'Normalization (Min-Max)', value: 'normalize' },
      { label: 'Standardization (Z-Score)', value: 'standardize' },
    ],
    tooltip: 'Choose between normalization (scales to [0,1]) and standardization (mean=0, std=1)',
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
    id: 'featureRange',
    type: 'slider' as const,
    label: 'Feature Range',
    min: 10,
    max: 100,
    step: 10,
    tooltip: 'Range of values for generated features',
  },
]);

// Generate sample data
const generateData = () => {
  const data: number[][] = [];
  const n = params.value.numSamples;
  const range = params.value.featureRange;

  for (let i = 0; i < n; i++) {
    // Generate two features with different scales
    const feature1 = Math.random() * range;
    const feature2 = Math.random() * range * 2 + range; // Different scale
    data.push([feature1, feature2]);
  }

  return data;
};

// Apply scaling
const applyScaling = () => {
  isProcessing.value = true;

  try {
    const data = generateData();
    scaler.value = new FeatureScaling(data);

    if (params.value.method === 'normalize') {
      result.value = scaler.value.normalize();
    } else if (params.value.method === 'standardize') {
      result.value = scaler.value.standardize();
    }
  } catch (error) {
    console.error('Error applying scaling:', error);
  } finally {
    isProcessing.value = false;
  }
};

// Computed properties for visualizations
const statistics = computed(() => result.value?.statistics);

const originalStats = computed(() => {
  if (!statistics.value) return [];
  return [
    { label: 'Min', value: statistics.value.originalMin?.map((v: number) => v.toFixed(2)).join(', ') },
    { label: 'Max', value: statistics.value.originalMax?.map((v: number) => v.toFixed(2)).join(', ') },
    { label: 'Mean', value: statistics.value.originalMean?.map((v: number) => v.toFixed(2)).join(', ') },
    { label: 'Std Dev', value: statistics.value.originalStd?.map((v: number) => v.toFixed(2)).join(', ') },
  ];
});

const scaledStats = computed(() => {
  if (!statistics.value) return [];
  return [
    { label: 'Min', value: statistics.value.scaledMin?.map((v: number) => v.toFixed(2)).join(', ') },
    { label: 'Max', value: statistics.value.scaledMax?.map((v: number) => v.toFixed(2)).join(', ') },
    { label: 'Mean', value: statistics.value.scaledMean?.map((v: number) => v.toFixed(2)).join(', ') },
    { label: 'Std Dev', value: statistics.value.scaledStd?.map((v: number) => v.toFixed(2)).join(', ') },
  ];
});

const originalDistribution = computed(() => {
  if (!result.value) return null;

  const feature1 = result.value.original.map((row: number[]) => row[0]);
  const feature2 = result.value.original.map((row: number[]) => row[1]);

  return [
    {
      x: feature1,
      type: 'histogram',
      name: 'Feature 1',
      opacity: 0.7,
      marker: { color: '#3b82f6' },
    },
    {
      x: feature2,
      type: 'histogram',
      name: 'Feature 2',
      opacity: 0.7,
      marker: { color: '#ef4444' },
    },
  ];
});

const scaledDistribution = computed(() => {
  if (!result.value) return null;

  const feature1 = result.value.scaled.map((row: number[]) => row[0]);
  const feature2 = result.value.scaled.map((row: number[]) => row[1]);

  return [
    {
      x: feature1,
      type: 'histogram',
      name: 'Feature 1',
      opacity: 0.7,
      marker: { color: '#3b82f6' },
    },
    {
      x: feature2,
      type: 'histogram',
      name: 'Feature 2',
      opacity: 0.7,
      marker: { color: '#ef4444' },
    },
  ];
});

const scatterData = computed(() => {
  if (!result.value) return null;

  return [
    {
      x: result.value.original.map((row: number[]) => row[0]),
      y: result.value.original.map((row: number[]) => row[1]),
      mode: 'markers',
      type: 'scatter',
      name: 'Original',
      marker: { color: '#3b82f6', size: 6, opacity: 0.6 },
    },
    {
      x: result.value.scaled.map((row: number[]) => row[0]),
      y: result.value.scaled.map((row: number[]) => row[1]),
      mode: 'markers',
      type: 'scatter',
      name: 'Scaled',
      marker: { color: '#10b981', size: 6, opacity: 0.6 },
    },
  ];
});

const chartLayout = (title: string) => ({
  title,
  xaxis: { title: 'Value' },
  yaxis: { title: 'Frequency' },
  height: 300,
  margin: { t: 40, r: 20, b: 40, l: 50 },
});

const scatterLayout = {
  title: 'Original vs Scaled Data',
  xaxis: { title: 'Feature 1' },
  yaxis: { title: 'Feature 2' },
  height: 400,
  margin: { t: 40, r: 20, b: 40, l: 50 },
};


// Event handlers
const handleParamChange = (key: string, value: any) => {
  (params.value as Record<string, any>)[key] = value;
};

const resetModule = () => {
  params.value = {
    method: 'normalize',
    numSamples: 100,
    featureRange: 50,
  };
  result.value = null;
  scaler.value = null;
  applyScaling();
};

const handleQuizComplete = (result: QuizResult) => {
  console.log('Quiz completed with score:', result.score);
};

// Load content on mount
onMounted(async () => {
  // Load quiz data
  await loadQuiz('feature-scaling', '/content/modules/preprocessing/feature-scaling/quiz.json');

  // Apply initial scaling
  applyScaling();
});

// Bookmark restoration
// Requirements: 2.2, 2.3
const { handleLoadBookmark } = useBookmarkRestore(params, applyScaling);
</script>
