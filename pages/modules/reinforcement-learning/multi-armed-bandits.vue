<template>
  <ModuleLayout
    title="Multi-Armed Bandits"
    description="Learn about the exploration-exploitation tradeoff through epsilon-greedy, UCB, and Thompson Sampling strategies"
    difficulty="beginner"
    :estimated-time="35"
    module-id="multi-armed-bandits"
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
      <InteractiveSection
        :loading="isTraining"
        :progress="trainingProgress"
        loading-message="Running bandit algorithm..."
        :error="error"
        :on-retry="handleRetry"
      >
        <!-- Controls -->
        <template #controls>
          <ControlPanel
            :controls="controls"
            :values="parameters"
            :loading="isTraining"
            submit-label="Run Algorithm"
            @change="handleParameterChange"
            @reset="handleReset"
            @submit="handleTrain"
          />
          
          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Results</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Total Reward</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.totalReward?.toFixed(0) }}
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Total Regret</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.totalRegret?.toFixed(2) }}
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Avg Reward/Trial</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.averageReward?.toFixed(3) }}
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Optimal Pull Rate</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ (metrics.optimalPullRate * 100).toFixed(1) }}%
                </div>
              </div>
            </div>
            
            <div class="mt-4 grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Best Arm Found</div>
                <div class="text-lg font-semibold" :class="metrics.foundOptimal ? 'text-green-600' : 'text-amber-600'">
                  Arm {{ metrics.bestArmFound }}
                  <span v-if="metrics.foundOptimal" class="text-xs">✓</span>
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Optimal Arm</div>
                <div class="text-lg font-semibold text-gray-900">
                  Arm {{ metrics.optimalArm }}
                </div>
              </div>
            </div>
            
            <div class="mt-3 text-xs text-gray-600">
              Strategy: {{ metrics.strategy }}
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-6">
            <!-- Cumulative Reward -->
            <RewardCurve
              v-if="rewardData && rewardData.length > 0"
              :rewards="rewardData"
              title="Cumulative Reward Over Time"
              x-label="Trial"
              y-label="Cumulative Reward"
              height="400px"
            />
            
            <!-- Cumulative Regret -->
            <RewardCurve
              v-if="regretData && regretData.length > 0"
              :rewards="regretData"
              title="Cumulative Regret Over Time"
              x-label="Trial"
              y-label="Cumulative Regret"
              height="400px"
            />
            
            <!-- Moving Average Reward -->
            <LineChart
              v-if="movingAvgSeries && movingAvgSeries.length > 0"
              :series="movingAvgSeries"
              title="Moving Average Reward"
              x-label="Trial"
              y-label="Average Reward"
              :show-legend="false"
              height="350px"
            />
            
            <!-- Arm Pull Distribution -->
            <div v-if="armPullData" class="bg-white p-4 rounded-lg border border-gray-200">
              <h3 class="text-lg font-semibold mb-4">Arm Pull Distribution</h3>
              <div class="space-y-2">
                <div v-for="(pulls, index) in armPullData" :key="index" class="flex items-center">
                  <div class="w-20 text-sm text-gray-600">Arm {{ index }}</div>
                  <div class="flex-1 bg-gray-200 rounded-full h-6 overflow-hidden">
                    <div 
                      class="bg-blue-500 h-full flex items-center justify-end pr-2 text-xs text-white font-medium"
                      :style="{ width: `${(pulls / Math.max(...armPullData)) * 100}%` }"
                    >
                      {{ pulls }}
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <!-- True vs Estimated Rewards -->
            <div v-if="trueRewards && estimatedRewards" class="bg-white p-4 rounded-lg border border-gray-200">
              <h3 class="text-lg font-semibold mb-4">True vs Estimated Rewards</h3>
              <div class="space-y-3">
                <div v-for="(trueReward, index) in trueRewards" :key="index" class="space-y-1">
                  <div class="flex justify-between text-sm">
                    <span class="text-gray-600">Arm {{ index }}</span>
                    <span class="text-gray-900 font-medium">
                      True: {{ trueReward.toFixed(3) }} | Est: {{ estimatedRewards[index].toFixed(3) }}
                    </span>
                  </div>
                  <div class="flex gap-2">
                    <div class="flex-1 bg-green-200 rounded-full h-4 overflow-hidden">
                      <div 
                        class="bg-green-500 h-full"
                        :style="{ width: `${trueReward * 100}%` }"
                      />
                    </div>
                    <div class="flex-1 bg-blue-200 rounded-full h-4 overflow-hidden">
                      <div 
                        class="bg-blue-500 h-full"
                        :style="{ width: `${estimatedRewards[index] * 100}%` }"
                      />
                    </div>
                  </div>
                </div>
                <div class="flex gap-4 text-xs text-gray-600 mt-2">
                  <div class="flex items-center gap-1">
                    <div class="w-3 h-3 bg-green-500 rounded"></div>
                    <span>True Reward</span>
                  </div>
                  <div class="flex items-center gap-1">
                    <div class="w-3 h-3 bg-blue-500 rounded"></div>
                    <span>Estimated Reward</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </template>
      </InteractiveSection>
    </template>

    <!-- Quiz Section -->
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
import { ref, onMounted } from 'vue';
import { useQuizLoader } from '~/composables/useQuizLoader';
import type { QuizResult } from '~/types/quiz';
import { useModuleStore } from '~/stores/module';
import { useTableOfContents } from '~/composables/useTableOfContents';
import { useBookmarkRestore } from '~/composables/useBookmarkRestore';
import { MultiArmedBandits } from '~/algorithms/reinforcement-learning/MultiArmedBandits';
import type { BanditStrategy } from '~/algorithms/reinforcement-learning/MultiArmedBandits';
import LineChart from '~/components/visualizations/LineChart.vue';
import RewardCurve from '~/components/visualizations/RewardCurve.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { LineSeries } from '~/components/visualizations/LineChart.vue';
import type { RewardData } from '~/components/visualizations/RewardCurve.vue';

import { useAsyncData } from '#app';

// Load content from Nuxt Content
const { data: content } = await useAsyncData(() => 
  queryCollection('content').path('/modules/reinforcement-learning/multi-armed-bandits/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores
const moduleStore = useModuleStore();

// Parameters
const parameters = ref({
  numArms: 5,
  numTrials: 1000,
  strategy: 'epsilon-greedy' as BanditStrategy,
  epsilon: 0.1,
  c: 2.0,
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'strategy',
    type: 'select',
    label: 'Strategy',
    options: [
      { label: 'Epsilon-Greedy', value: 'epsilon-greedy' },
      { label: 'UCB (Upper Confidence Bound)', value: 'ucb' },
      { label: 'Thompson Sampling', value: 'thompson-sampling' }
    ],
    tooltip: 'Choose the bandit algorithm strategy to use',
    group: 'Algorithm',
  },
  {
    id: 'numArms',
    type: 'slider',
    label: 'Number of Arms',
    min: 2,
    max: 10,
    step: 1,
    tooltip: 'Number of slot machines (arms) to choose from',
    group: 'Environment',
  },
  {
    id: 'numTrials',
    type: 'slider',
    label: 'Number of Trials',
    min: 100,
    max: 5000,
    step: 100,
    tooltip: 'Number of times to pull an arm (more trials = better learning)',
    group: 'Environment',
  },
  {
    id: 'epsilon',
    type: 'slider',
    label: 'Epsilon (ε)',
    min: 0.0,
    max: 1.0,
    step: 0.05,
    tooltip: 'Exploration rate for epsilon-greedy strategy. Higher values explore more.',
    group: 'Epsilon-Greedy',
  },
  {
    id: 'c',
    type: 'slider',
    label: 'Exploration Parameter (c)',
    min: 0.1,
    max: 5.0,
    step: 0.1,
    tooltip: 'Controls exploration in UCB. Higher values encourage more exploration.',
    group: 'UCB',
  },
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const rewardData = ref<RewardData[]>([]);
const regretData = ref<RewardData[]>([]);
const movingAvgSeries = ref<LineSeries[]>([]);
const armPullData = ref<number[]>([]);
const trueRewards = ref<number[]>([]);
const estimatedRewards = ref<number[]>([]);

// Algorithm instance
let algorithm: MultiArmedBandits | null = null;

// Quiz data with loading state
const { quizData, loading: quizLoading, error: quizError, loadQuiz, retry: retryQuizLoad } = useQuizLoader();

/**
 * Handle quiz completion
 */
const handleQuizComplete = (result: QuizResult) => {
  console.log('Quiz completed with score:', result.score);
};

/**
 * Initialize module
 */
onMounted(async () => {
  moduleStore.setCurrentModule('multi-armed-bandits');
  moduleStore.setParameters(parameters.value);
  
  // Load quiz data
  await loadQuiz('multi-armed-bandits', '/content/modules/reinforcement-learning/multi-armed-bandits/quiz.json');
});

/**
 * Handle parameter change
 */
const handleParameterChange = (id: string, value: any) => {
  parameters.value[id] = value;
  moduleStore.setParameter(id, value);
};

/**
 * Handle reset
 */
const handleReset = () => {
  parameters.value = {
    numArms: 5,
    numTrials: 1000,
    strategy: 'epsilon-greedy',
    epsilon: 0.1,
    c: 2.0,
  };
  moduleStore.setParameters(parameters.value);
  
  metrics.value = null;
  rewardData.value = [];
  regretData.value = [];
  movingAvgSeries.value = [];
  armPullData.value = [];
  trueRewards.value = [];
  estimatedRewards.value = [];
  error.value = null;
};

/**
 * Handle train
 */
const handleTrain = async () => {
  isTraining.value = true;
  trainingProgress.value = 0;
  error.value = null;

  try {
    // Create and run algorithm
    algorithm = new MultiArmedBandits({
      numArms: parameters.value.numArms,
      numTrials: parameters.value.numTrials,
      strategy: parameters.value.strategy,
      epsilon: parameters.value.epsilon,
      c: parameters.value.c,
    });
    
    // Run algorithm
    const startTime = Date.now();
    await algorithm.fit([], []);
    const endTime = Date.now();
    
    console.log(`Algorithm completed in ${endTime - startTime}ms`);
    
    // Get results
    metrics.value = algorithm.getMetrics();
    trueRewards.value = algorithm.getTrueRewards();
    
    // Get arm statistics
    const armStats = algorithm.getArmStats();
    estimatedRewards.value = armStats.map(s => s.meanReward);
    armPullData.value = armStats.map(s => s.pulls);
    
    // Update visualizations
    updateVisualizations();
    
    trainingProgress.value = 1;
  } catch (err) {
    console.error('Algorithm error:', err);
    error.value = err instanceof Error ? err.message : 'Algorithm failed. Please try again.';
  } finally {
    isTraining.value = false;
  }
};

/**
 * Update visualizations
 */
const updateVisualizations = () => {
  if (!algorithm) return;
  
  const visualizations = algorithm.getVisualizationData();
  
  // Process visualizations
  if (visualizations.length > 0) {
    // Cumulative reward
    const rewardViz = visualizations[0];
    if (rewardViz.type === 'line' && rewardViz.data?.cumulativeReward) {
      const rewardTrace = rewardViz.data.cumulativeReward;
      if (Array.isArray(rewardTrace.x) && Array.isArray(rewardTrace.y) && rewardTrace.x.length > 0 && rewardTrace.y.length > 0) {
        rewardData.value = [{
          x: rewardTrace.x,
          y: rewardTrace.y,
          name: rewardTrace.name || 'Cumulative Reward',
          color: rewardTrace.line?.color || '#10b981',
        }];
      }
    }
    
    // Cumulative regret
    if (visualizations.length > 1) {
      const regretViz = visualizations[1];
      if (regretViz.type === 'line' && regretViz.data?.cumulativeRegret) {
        const regretTrace = regretViz.data.cumulativeRegret;
        if (Array.isArray(regretTrace.x) && Array.isArray(regretTrace.y) && regretTrace.x.length > 0 && regretTrace.y.length > 0) {
          regretData.value = [{
            x: regretTrace.x,
            y: regretTrace.y,
            name: regretTrace.name || 'Cumulative Regret',
            color: regretTrace.line?.color || '#ef4444',
          }];
        }
      }
    }
    
    // Moving average reward
    if (visualizations.length > 4) {
      const movingAvgViz = visualizations[4];
      if (movingAvgViz.type === 'line' && movingAvgViz.data?.movingAvg) {
        const movingAvgTrace = movingAvgViz.data.movingAvg;
        if (Array.isArray(movingAvgTrace.x) && Array.isArray(movingAvgTrace.y) && movingAvgTrace.x.length > 0 && movingAvgTrace.y.length > 0) {
          movingAvgSeries.value = [{
            x: movingAvgTrace.x,
            y: movingAvgTrace.y,
            name: movingAvgTrace.name || 'Moving Average',
            color: movingAvgTrace.line?.color || '#8b5cf6',
            width: movingAvgTrace.line?.width || 2,
          }];
        }
      }
    }
  }
};

/**
 * Handle retry
 */
const handleRetry = () => {
  error.value = null;
  handleTrain();
};

// Bookmark restoration
// Requirements: 2.2, 2.3
const { handleLoadBookmark } = useBookmarkRestore(parameters, handleTrain);
</script>
