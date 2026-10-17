<template>
  <ModuleLayout module-id="data-augmentation" title="Data Augmentation" domain="preprocessing" difficulty="intermediate"
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
              <Button @click="applyAugmentation" :disabled="isProcessing">
                {{ isProcessing ? 'Processing...' : 'Apply Augmentation' }}
              </Button>
            </template>
          </ControlPanel>
        </template>

        <template #visualization>
          <div class="space-y-4">
            <!-- Original vs Augmented Data Scatter Plot -->
            <div v-if="scatterData">
              <h3 class="text-lg font-semibold mb-2">Original vs Augmented Data</h3>
              <ClientOnly>
              <BaseChart :data="scatterData" :layout="scatterLayout" />
              </ClientOnly>
            </div>

            <!-- Statistics Comparison -->
            <div v-if="statistics" class="bg-gray-50 p-4 rounded-lg">
              <h3 class="text-lg font-semibold mb-3">Dataset Statistics</h3>
              <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="text-center">
                  <div class="text-2xl font-bold text-blue-600">{{ statistics.originalSamples }}</div>
                  <div class="text-sm text-gray-600">Original Samples</div>
                </div>
                <div class="text-center">
                  <div class="text-2xl font-bold text-red-600">{{ statistics.augmentedSamples - statistics.originalSamples }}</div>
                  <div class="text-sm text-gray-600">Generated Samples</div>
                </div>
                <div class="text-center">
                  <div class="text-2xl font-bold text-green-600">{{ statistics.totalSamples }}</div>
                  <div class="text-sm text-gray-600">Total Samples</div>
                </div>
                <div class="text-center">
                  <div class="text-2xl font-bold text-purple-600">{{ statistics.augmentationRatio.toFixed(2) }}x</div>
                  <div class="text-sm text-gray-600">Augmentation Ratio</div>
                </div>
              </div>
            </div>

            <!-- Sample Count Comparison Bar Chart -->
            <div v-if="barChartData">
              <h3 class="text-lg font-semibold mb-2">Dataset Size Comparison</h3>
              <ClientOnly>
              <BaseChart :data="barChartData" :layout="barChartLayout" />
              </ClientOnly>
            </div>

            <!-- Class Distribution (if labels available) -->
            <div v-if="classDistributionData">
              <h3 class="text-lg font-semibold mb-2">Class Distribution</h3>
              <ClientOnly> 
              <BaseChart :data="classDistributionData" :layout="classDistributionLayout" />
              </ClientOnly>
            </div>

            <!-- Metrics Display -->
            <div v-if="metrics" class="bg-blue-50 p-4 rounded-lg">
              <h3 class="text-lg font-semibold mb-3">Augmentation Metrics</h3>
              <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                <div>
                  <span class="font-medium">Method:</span>
                  <span class="ml-2 capitalize">{{ params.method }}</span>
                </div>
                <div>
                  <span class="font-medium">Dataset Increase:</span>
                  <span class="ml-2">{{ metrics.datasetIncrease?.toFixed(1) }}%</span>
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
import { DataAugmentation, type AugmentationMethod } from '~/algorithms/preprocessing/DataAugmentation';
import { useQuizLoader } from '~/composables/useQuizLoader';
import type { QuizResult } from '~/types/quiz';
import { useAsyncData } from '#app';
import { useBookmarkRestore } from '~/composables/useBookmarkRestore';

// Load content from Nuxt Content
const { data: content } = await useAsyncData(() =>
  queryCollection('content').path('/modules/preprocessing/data-augmentation/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Component state
const params = ref({
  method: 'noise' as AugmentationMethod,
  numSamples: 100,
  numFeatures: 2,
  numClasses: 2,
  augmentationFactor: 3,
  noiseLevel: 0.1,
  maxAngle: 30,
  scaleRange: [0.8, 1.2] as [number, number],
  translationRange: 0.1,
});

const isProcessing = ref(false);
const augmenter = ref<DataAugmentation | null>(null);
const result = ref<any>(null);

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

// Controls configuration
const controls = computed(() => [
  {
    id: 'method',
    type: 'select' as const,
    label: 'Augmentation Method',
    options: [
      { label: 'Gaussian Noise', value: 'noise' },
      { label: 'Rotation', value: 'rotation' },
      { label: 'Scaling', value: 'scaling' },
      { label: 'Translation', value: 'translation' },
      { label: 'SMOTE', value: 'smote' },
      { label: 'Mixup', value: 'mixup' },
    ],
    tooltip: 'Choose the augmentation technique to apply',
  },
  {
    id: 'numSamples',
    type: 'slider' as const,
    label: 'Number of Samples',
    min: 50,
    max: 200,
    step: 10,
    tooltip: 'Number of original data points to generate',
  },
  {
    id: 'augmentationFactor',
    type: 'slider' as const,
    label: 'Augmentation Factor',
    min: 2,
    max: 5,
    step: 1,
    tooltip: 'Multiplier for dataset size (e.g., 3x means triple the data)',
  },
  {
    id: 'numClasses',
    type: 'slider' as const,
    label: 'Number of Classes',
    min: 2,
    max: 4,
    step: 1,
    tooltip: 'Number of classes for classification data',
    visible: () => ['smote', 'mixup'].includes(params.value.method),
  },
  {
    id: 'noiseLevel',
    type: 'slider' as const,
    label: 'Noise Level',
    min: 0.05,
    max: 0.5,
    step: 0.05,
    tooltip: 'Standard deviation of Gaussian noise',
    visible: () => params.value.method === 'noise',
  },
  {
    id: 'maxAngle',
    type: 'slider' as const,
    label: 'Max Rotation Angle',
    min: 10,
    max: 90,
    step: 10,
    tooltip: 'Maximum rotation angle in degrees',
    visible: () => params.value.method === 'rotation',
  },
  {
    id: 'translationRange',
    type: 'slider' as const,
    label: 'Translation Range',
    min: 0.05,
    max: 0.5,
    step: 0.05,
    tooltip: 'Maximum translation as fraction of feature range',
    visible: () => params.value.method === 'translation',
  },
]);

// Generate sample data based on method
const generateData = () => {
  const method = params.value.method;
  
  if (method === 'smote') {
    // Generate imbalanced data for SMOTE demonstration
    const majoritySize = Math.floor(params.value.numSamples * 0.7);
    const minoritySize = params.value.numSamples - majoritySize;
    return DataAugmentation.generateImbalancedData(majoritySize, minoritySize, params.value.numFeatures);
  } else {
    // Generate balanced data for other methods
    return DataAugmentation.generateSampleData(
      params.value.numSamples,
      params.value.numFeatures,
      params.value.numClasses,
      1.5
    );
  }
};

// Apply augmentation
const applyAugmentation = () => {
  isProcessing.value = true;

  try {
    const { data, labels } = generateData();
    augmenter.value = new DataAugmentation();
    augmenter.value.setData(data, labels, params.value.augmentationFactor);

    // Apply the selected augmentation method
    switch (params.value.method) {
      case 'noise':
        result.value = augmenter.value.augmentWithNoise(params.value.noiseLevel);
        break;
      case 'rotation':
        result.value = augmenter.value.augmentWithRotation(params.value.maxAngle);
        break;
      case 'scaling':
        result.value = augmenter.value.augmentWithScaling(params.value.scaleRange);
        break;
      case 'translation':
        result.value = augmenter.value.augmentWithTranslation(params.value.translationRange);
        break;
      case 'smote':
        result.value = augmenter.value.augmentWithSMOTE(5);
        break;
      case 'mixup':
        result.value = augmenter.value.augmentWithMixup(0.2);
        break;
      default:
        result.value = augmenter.value.getResult();
    }
  } catch (error) {
    console.error('Error applying augmentation:', error);
  } finally {
    isProcessing.value = false;
  }
};

// Computed properties for visualizations
const statistics = computed(() => result.value?.statistics);

const metrics = computed(() => {
  if (!augmenter.value || !result.value) return null;
  return augmenter.value.getMetrics();
});

const scatterData = computed(() => {
  if (!result.value || result.value.originalData[0]?.length < 2) return null;

  const originalX = result.value.originalData.map((row: number[]) => row[0]);
  const originalY = result.value.originalData.map((row: number[]) => row[1]);
  
  const augmentedOnly = result.value.augmentedData.slice(result.value.originalData.length);
  const augmentedX = augmentedOnly.map((row: number[]) => row[0]);
  const augmentedY = augmentedOnly.map((row: number[]) => row[1]);

  const traces: any[] = [];

  // Original data
  if (result.value.originalLabels && result.value.originalLabels.length > 0) {
    // Color by class
    const classes = Array.from(new Set(result.value.originalLabels));
    const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b'];
    
    classes.forEach((cls: number, idx: number) => {
      const classIndices = result.value.originalLabels
        .map((label: number, i: number) => label === cls ? i : -1)
        .filter((i: number) => i !== -1);
      
      traces.push({
        x: classIndices.map((i: number) => originalX[i]),
        y: classIndices.map((i: number) => originalY[i]),
        mode: 'markers',
        type: 'scatter',
        name: `Original Class ${cls}`,
        marker: { 
          color: colors[idx % colors.length], 
          size: 8, 
          opacity: 0.8,
          symbol: 'circle'
        },
      });
    });
  } else {
    traces.push({
      x: originalX,
      y: originalY,
      mode: 'markers',
      type: 'scatter',
      name: 'Original Data',
      marker: { color: '#3b82f6', size: 8, opacity: 0.8 },
    });
  }

  // Augmented data
  if (augmentedX.length > 0) {
    if (result.value.augmentedLabels && result.value.augmentedLabels.length > result.value.originalData.length) {
      // Color by class
      const classes = Array.from(new Set(result.value.augmentedLabels));
      const colors = ['#93c5fd', '#fca5a5', '#6ee7b7', '#fcd34d'];
      const augmentedLabelsOnly = result.value.augmentedLabels.slice(result.value.originalData.length);
      
      classes.forEach((cls: number, idx: number) => {
        const classIndices = augmentedLabelsOnly
          .map((label: number, i: number) => label === cls ? i : -1)
          .filter((i: number) => i !== -1);
        
        if (classIndices.length > 0) {
          traces.push({
            x: classIndices.map((i: number) => augmentedX[i]),
            y: classIndices.map((i: number) => augmentedY[i]),
            mode: 'markers',
            type: 'scatter',
            name: `Augmented Class ${cls}`,
            marker: { 
              color: colors[idx % colors.length], 
              size: 5, 
              opacity: 0.5,
              symbol: 'x'
            },
          });
        }
      });
    } else {
      traces.push({
        x: augmentedX,
        y: augmentedY,
        mode: 'markers',
        type: 'scatter',
        name: 'Augmented Data',
        marker: { color: '#ef4444', size: 5, opacity: 0.5, symbol: 'x' },
      });
    }
  }

  return traces;
});

const scatterLayout = computed(() => ({
  title: `Data Augmentation - ${params.value.method.toUpperCase()} Method`,
  xaxis: { title: 'Feature 1' },
  yaxis: { title: 'Feature 2' },
  height: 500,
  margin: { t: 40, r: 20, b: 40, l: 50 },
  showlegend: true,
  legend: { x: 1.05, y: 1 },
}));

const barChartData = computed(() => {
  if (!statistics.value) return null;

  return [
    {
      x: ['Original', 'Generated', 'Total'],
      y: [
        statistics.value.originalSamples,
        statistics.value.augmentedSamples - statistics.value.originalSamples,
        statistics.value.totalSamples,
      ],
      type: 'bar',
      marker: {
        color: ['#3b82f6', '#ef4444', '#10b981'],
      },
      text: [
        statistics.value.originalSamples.toString(),
        (statistics.value.augmentedSamples - statistics.value.originalSamples).toString(),
        statistics.value.totalSamples.toString(),
      ],
      textposition: 'auto',
    },
  ];
});

const barChartLayout = {
  title: 'Dataset Size Comparison',
  xaxis: { title: 'Dataset Type' },
  yaxis: { title: 'Number of Samples' },
  height: 300,
  margin: { t: 40, r: 20, b: 40, l: 50 },
};

const classDistributionData = computed(() => {
  if (!result.value?.originalLabels || result.value.originalLabels.length === 0) return null;

  const originalCounts: { [key: number]: number } = {};
  const augmentedCounts: { [key: number]: number } = {};

  // Count original class distribution
  result.value.originalLabels.forEach((label: number) => {
    originalCounts[label] = (originalCounts[label] || 0) + 1;
  });

  // Count augmented class distribution
  if (result.value.augmentedLabels && result.value.augmentedLabels.length > 0) {
    result.value.augmentedLabels.forEach((label: number) => {
      augmentedCounts[label] = (augmentedCounts[label] || 0) + 1;
    });
  }

  const classes = Array.from(new Set([...Object.keys(originalCounts), ...Object.keys(augmentedCounts)])).map(Number).sort();

  return [
    {
      x: classes.map(c => `Class ${c}`),
      y: classes.map(c => originalCounts[c] || 0),
      type: 'bar',
      name: 'Original',
      marker: { color: '#3b82f6' },
    },
    {
      x: classes.map(c => `Class ${c}`),
      y: classes.map(c => augmentedCounts[c] || 0),
      type: 'bar',
      name: 'Augmented',
      marker: { color: '#10b981' },
    },
  ];
});

const classDistributionLayout = {
  title: 'Class Distribution Comparison',
  xaxis: { title: 'Class' },
  yaxis: { title: 'Number of Samples' },
  height: 300,
  margin: { t: 40, r: 20, b: 40, l: 50 },
  barmode: 'group',
};

// Event handlers
const handleParamChange = (key: string, value: any) => {
  (params.value as Record<string, any>)[key] = value;
};

const resetModule = () => {
  params.value = {
    method: 'noise',
    numSamples: 100,
    numFeatures: 2,
    numClasses: 2,
    augmentationFactor: 3,
    noiseLevel: 0.1,
    maxAngle: 30,
    scaleRange: [0.8, 1.2],
    translationRange: 0.1,
  };
  result.value = null;
  augmenter.value = null;
  applyAugmentation();
};

const handleQuizComplete = (result: QuizResult) => {
  console.log('Quiz completed with score:', result.score);
};

// Load content on mount
onMounted(async () => {
  // Load quiz data
  await loadQuiz('data-augmentation', '/content/modules/preprocessing/data-augmentation/quiz.json');

  // Apply initial augmentation
  applyAugmentation();
});

// Bookmark restoration
// Requirements: 2.2, 2.3
const { handleLoadBookmark } = useBookmarkRestore(params, applyAugmentation);
</script>
