<template>
  <ModuleLayout 
    module-id="outlier-detection" 
    title="Outlier Detection & Handling" 
    domain="preprocessing" 
    difficulty="intermediate"
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
              <Button @click="detectOutliers" :disabled="isProcessing">
                {{ isProcessing ? 'Processing...' : 'Detect Outliers' }}
              </Button>
            </template>
          </ControlPanel>
        </template>

        <template #visualization>
          <div class="space-y-4">
            <!-- Statistics Summary -->
            <div v-if="result" class="bg-gray-50 p-4 rounded-lg">
              <h3 class="text-lg font-semibold mb-3">Detection Summary</h3>
              <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span class="font-medium">Total Samples:</span>
                  <div class="text-2xl font-bold text-blue-600">{{ result.statistics.totalSamples }}</div>
                </div>
                <div>
                  <span class="font-medium">Outliers Detected:</span>
                  <div class="text-2xl font-bold text-red-600">{{ result.statistics.outliersDetected }}</div>
                </div>
                <div>
                  <span class="font-medium">Outlier %:</span>
                  <div class="text-2xl font-bold text-orange-600">{{ result.statistics.outlierPercentage.toFixed(1) }}%</div>
                </div>
                <div>
                  <span class="font-medium">Cleaned Samples:</span>
                  <div class="text-2xl font-bold text-green-600">{{ result.cleanedData.length }}</div>
                </div>
              </div>
            </div>

            <!-- Data Visualization -->
            <div v-if="scatterData" class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h3 class="text-lg font-semibold mb-2">Original Data with Outliers</h3>
                <ClientOnly> 
                <BaseChart :data="scatterData.original" :layout="scatterLayout('Original Data')" />
                </ClientOnly>
              </div>
              <div>
                <h3 class="text-lg font-semibold mb-2">After Handling</h3>
                 <ClientOnly> 
                <BaseChart :data="scatterData.cleaned" :layout="scatterLayout('Cleaned Data')" />
                </ClientOnly>
              </div>
            </div>

            <!-- Outlier Scores -->
            <div v-if="outlierScoresData">
              <h3 class="text-lg font-semibold mb-2">Outlier Scores</h3>
               <ClientOnly> 
              <BaseChart :data="outlierScoresData" :layout="scoresLayout" />
              </ClientOnly>
            </div>

            <!-- Distribution Comparison -->
            <div v-if="distributionData" class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h3 class="text-lg font-semibold mb-2">Feature 1 Distribution</h3>
                 <ClientOnly> 
                <BaseChart :data="distributionData.feature1" :layout="histogramLayout('Feature 1')" />
                </ClientOnly>
              </div>
              <div>
                <h3 class="text-lg font-semibold mb-2">Feature 2 Distribution</h3>
                 <ClientOnly>
                <BaseChart :data="distributionData.feature2" :layout="histogramLayout('Feature 2')" />
                </ClientOnly>
              </div>
            </div>

            <!-- Method Information -->
            <div v-if="result" class="bg-blue-50 p-4 rounded-lg">
              <h3 class="text-lg font-semibold mb-2">Method: {{ methodName }}</h3>
              <p class="text-sm text-gray-700">{{ methodDescription }}</p>
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
import { OutlierDetection, type OutlierMethod, type OutlierHandling } from '~/algorithms/preprocessing/OutlierDetection';
import { useQuizLoader } from '~/composables/useQuizLoader';
import type { QuizResult } from '~/types/quiz';
import { useAsyncData } from '#app';
import { useBookmarkRestore } from '~/composables/useBookmarkRestore';

// Load content from Nuxt Content
const { data: content } = await useAsyncData(() =>
  queryCollection('content').path('/modules/preprocessing/outlier-detection/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Component state
const params = ref({
  method: 'z-score' as OutlierMethod,
  handling: 'keep' as OutlierHandling,
  numSamples: 200,
  numFeatures: 2,
  outlierPercentage: 0.1,
  threshold: 3.0,
  outlierMagnitude: 3,
});

const isProcessing = ref(false);
const detector = ref<OutlierDetection | null>(null);
const result = ref<any>(null);

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

// Controls configuration
const controls = computed(() => [
  {
    id: 'method',
    type: 'select' as const,
    label: 'Detection Method',
    options: [
      { label: 'Z-Score', value: 'z-score' },
      { label: 'IQR (Interquartile Range)', value: 'iqr' },
      { label: 'Isolation Forest', value: 'isolation-forest' },
      { label: 'Local Outlier Factor', value: 'local-outlier-factor' },
      { label: 'DBSCAN', value: 'dbscan' },
    ],
    tooltip: 'Choose the outlier detection method',
  },
  {
    id: 'handling',
    type: 'select' as const,
    label: 'Handling Strategy',
    options: [
      { label: 'Keep (Highlight Only)', value: 'keep' },
      { label: 'Remove', value: 'remove' },
      { label: 'Cap (Winsorize)', value: 'cap' },
      { label: 'Transform', value: 'transform' },
    ],
    tooltip: 'Choose how to handle detected outliers',
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
    id: 'outlierPercentage',
    type: 'slider' as const,
    label: 'Outlier Percentage',
    min: 0.05,
    max: 0.3,
    step: 0.05,
    tooltip: 'Percentage of outliers in generated data',
  },
  {
    id: 'threshold',
    type: 'slider' as const,
    label: 'Detection Threshold',
    min: 1.5,
    max: 5.0,
    step: 0.5,
    tooltip: 'Threshold for outlier detection (meaning varies by method)',
    visible: computed(() => ['z-score', 'iqr'].includes(params.value.method)),
  },
  {
    id: 'outlierMagnitude',
    type: 'slider' as const,
    label: 'Outlier Magnitude',
    min: 1,
    max: 5,
    step: 0.5,
    tooltip: 'How extreme the outliers are',
  },
]);

// Method descriptions
const methodDescriptions: Record<OutlierMethod, { name: string; description: string }> = {
  'z-score': {
    name: 'Z-Score Method',
    description: 'Detects outliers based on how many standard deviations away from the mean. Assumes normal distribution. Points with |Z| > threshold are considered outliers.',
  },
  'iqr': {
    name: 'IQR Method',
    description: 'Uses the Interquartile Range to detect outliers. More robust to extreme values than Z-score. Points outside Q1 - 1.5×IQR and Q3 + 1.5×IQR are outliers.',
  },
  'isolation-forest': {
    name: 'Isolation Forest',
    description: 'Isolates outliers by randomly selecting features and split values. Outliers require fewer splits to be isolated. Works well for high-dimensional data.',
  },
  'local-outlier-factor': {
    name: 'Local Outlier Factor (LOF)',
    description: 'Measures local deviation of density compared to neighbors. Points with significantly lower local density are outliers. Good for varying density data.',
  },
  'dbscan': {
    name: 'DBSCAN Clustering',
    description: 'Identifies outliers as noise points that don\'t belong to any cluster. Works well when data has natural clusters.',
  },
  'none': {
    name: 'No Detection',
    description: 'No outlier detection applied.',
  },
};

const methodName = computed(() => methodDescriptions[params.value.method]?.name || 'Unknown Method');
const methodDescription = computed(() => methodDescriptions[params.value.method]?.description || '');

// Generate sample data with outliers
const generateData = () => {
  return OutlierDetection.generateDataWithOutliers(
    params.value.numSamples,
    params.value.numFeatures,
    params.value.outlierPercentage,
    params.value.outlierMagnitude
  );
};

// Detect outliers
const detectOutliers = () => {
  isProcessing.value = true;

  try {
    const data = generateData();
    detector.value = new OutlierDetection();
    detector.value.setData(data, params.value.threshold, params.value.handling);

    // Apply the selected detection method
    switch (params.value.method) {
      case 'z-score':
        result.value = detector.value.detectByZScore(params.value.threshold);
        break;
      case 'iqr':
        result.value = detector.value.detectByIQR(params.value.threshold);
        break;
      case 'isolation-forest':
        result.value = detector.value.detectByIsolationForest(params.value.outlierPercentage);
        break;
      case 'local-outlier-factor':
        result.value = detector.value.detectByLOF(5, params.value.outlierPercentage);
        break;
      case 'dbscan':
        result.value = detector.value.detectByDBSCAN(0.5, 5);
        break;
    }
  } catch (error) {
    console.error('Error detecting outliers:', error);
  } finally {
    isProcessing.value = false;
  }
};

// Computed properties for visualizations
const scatterData = computed(() => {
  if (!result.value || result.value.originalData[0]?.length < 2) return null;

  const colors = result.value.originalData.map((_: any, i: number) =>
    result.value.outlierIndices.includes(i) ? 'red' : 'blue'
  );

  const original = [
    {
      x: result.value.originalData.map((row: number[]) => row[0]),
      y: result.value.originalData.map((row: number[]) => row[1]),
      mode: 'markers',
      type: 'scatter',
      marker: {
        color: colors,
        size: 8,
        opacity: 0.7,
        line: {
          color: colors.map((c: string) => c === 'red' ? 'darkred' : 'darkblue'),
          width: 1,
        },
      },
      name: 'Data Points',
      text: result.value.originalData.map((_: any, i: number) =>
        result.value.outlierIndices.includes(i) ? 'Outlier' : 'Normal'
      ),
      hovertemplate: '<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
    },
  ];

  const cleanedColors = result.value.cleanedData.map(() => 'green');
  const cleaned = [
    {
      x: result.value.cleanedData.map((row: number[]) => row[0]),
      y: result.value.cleanedData.map((row: number[]) => row[1]),
      mode: 'markers',
      type: 'scatter',
      marker: {
        color: cleanedColors,
        size: 8,
        opacity: 0.7,
      },
      name: 'Cleaned Data',
      hovertemplate: 'X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
    },
  ];

  return { original, cleaned };
});

const outlierScoresData = computed(() => {
  if (!result.value) return null;

  const colors = result.value.outlierScores.map((_: number, i: number) =>
    result.value.outlierIndices.includes(i) ? 'red' : 'blue'
  );

  return [
    {
      x: Array.from({ length: result.value.outlierScores.length }, (_, i) => i),
      y: result.value.outlierScores,
      mode: 'markers',
      type: 'scatter',
      marker: {
        color: colors,
        size: 6,
        opacity: 0.7,
      },
      name: 'Outlier Scores',
      hovertemplate: 'Sample: %{x}<br>Score: %{y:.3f}<extra></extra>',
    },
  ];
});

const distributionData = computed(() => {
  if (!result.value || result.value.originalData[0]?.length < 2) return null;

  const feature1Original = result.value.originalData.map((row: number[]) => row[0]);
  const feature2Original = result.value.originalData.map((row: number[]) => row[1]);
  const feature1Cleaned = result.value.cleanedData.map((row: number[]) => row[0]);
  const feature2Cleaned = result.value.cleanedData.map((row: number[]) => row[1]);

  return {
    feature1: [
      {
        x: feature1Original,
        type: 'histogram',
        name: 'Original',
        opacity: 0.6,
        marker: { color: '#3b82f6' },
      },
      {
        x: feature1Cleaned,
        type: 'histogram',
        name: 'Cleaned',
        opacity: 0.6,
        marker: { color: '#10b981' },
      },
    ],
    feature2: [
      {
        x: feature2Original,
        type: 'histogram',
        name: 'Original',
        opacity: 0.6,
        marker: { color: '#3b82f6' },
      },
      {
        x: feature2Cleaned,
        type: 'histogram',
        name: 'Cleaned',
        opacity: 0.6,
        marker: { color: '#10b981' },
      },
    ],
  };
});

const scatterLayout = (title: string) => ({
  title,
  xaxis: { title: 'Feature 1' },
  yaxis: { title: 'Feature 2' },
  height: 400,
  margin: { t: 40, r: 20, b: 40, l: 50 },
  showlegend: true,
});

const scoresLayout = {
  title: 'Outlier Scores by Sample',
  xaxis: { title: 'Sample Index' },
  yaxis: { title: 'Outlier Score' },
  height: 300,
  margin: { t: 40, r: 20, b: 40, l: 50 },
};

const histogramLayout = (title: string) => ({
  title,
  xaxis: { title: 'Value' },
  yaxis: { title: 'Frequency' },
  height: 300,
  margin: { t: 40, r: 20, b: 40, l: 50 },
  barmode: 'overlay',
});

// Event handlers
const handleParamChange = (key: string, value: any) => {
  (params.value as Record<string, any>)[key] = value;
};

const resetModule = () => {
  params.value = {
    method: 'z-score',
    handling: 'keep',
    numSamples: 200,
    numFeatures: 2,
    outlierPercentage: 0.1,
    threshold: 3.0,
    outlierMagnitude: 3,
  };
  result.value = null;
  detector.value = null;
  detectOutliers();
};

const handleQuizComplete = (quizResult: QuizResult) => {
  console.log('Quiz completed with score:', quizResult.score);
};

// Load content on mount
onMounted(async () => {
  // Load quiz data
  await loadQuiz('outlier-detection', '/content/modules/preprocessing/outlier-detection/quiz.json');

  // Detect outliers initially
  detectOutliers();
});

// Bookmark restoration
// Requirements: 2.2, 2.3
const { handleLoadBookmark } = useBookmarkRestore(params, detectOutliers);
</script>