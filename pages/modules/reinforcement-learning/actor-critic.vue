<template>
  <ModuleLayout
    title="Actor-Critic Methods"
    description="Learn how Actor-Critic algorithms combine policy and value learning for stable and efficient reinforcement learning"
    difficulty="advanced"
    :estimated-time="50"
    module-id="actor-critic"
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
        loading-message="Training Actor-Critic agent..."
        :error="error"
        :on-retry="handleRetry"
      >
        <!-- Controls -->
        <template #controls>
          <ControlPanel
            :controls="controls"
            :values="parameters"
            :loading="isTraining"
            submit-label="Train Agent"
            @change="handleParameterChange"
            @reset="handleReset"
            @submit="handleTrain"
          />
          
          <!-- Metrics Display -->
          <div v-if="metrics" class="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h3 class="text-sm font-semibold text-gray-700 mb-3">Training Results</h3>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Avg Reward (Last 10)</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.averageReward?.toFixed(2) }}
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Avg Steps (Last 10)</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.averageSteps?.toFixed(1) }}
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Optimal Path Length</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.optimalPathLength }}
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Success Rate</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ (metrics.successRate * 100).toFixed(0) }}%
                </div>
              </div>
            </div>
            
            <div class="mt-4 grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Total Episodes</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.totalEpisodes }}
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Entropy Coefficient</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.entropyCoefficient?.toFixed(3) }}
                </div>
              </div>
            </div>
            
            <div class="mt-4 grid grid-cols-2 gap-4">
              <div>
                <div class="text-xs text-gray-500">Avg Actor Loss</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.averageActorLoss?.toFixed(4) }}
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Avg Critic Loss</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.averageCriticLoss?.toFixed(4) }}
                </div>
              </div>
            </div>
          </div>
        </template>

        <!-- Visualization -->
        <template #visualization>
          <div class="space-y-6">
            <!-- Environment Renderer -->
            <div v-if="environment" class="bg-white p-4 rounded-lg border border-gray-200">
              <h3 class="text-lg font-semibold mb-4">Environment & Optimal Path</h3>
              <EnvironmentRenderer
                :grid-size="environment.gridSize"
                :start-state="environment.startState"
                :goal-state="environment.goalState"
                :obstacles="environment.obstacles"
                :optimal-path="optimalPath"
                :show-actions="false"
                height="500px"
              />
            </div>
            
            <!-- State Values Heatmap (from Critic) -->
            <div v-if="stateValues && environment" class="bg-white p-4 rounded-lg border border-gray-200">
              <Heatmap
                :data="stateValues"
                title="State Values (Critic Network)"
                x-label="Column"
                y-label="Row"
                height="400px"
                :color-scale="'Viridis'"
              />
            </div>
            
            <!-- Policy Visualization -->
            <div v-if="policyData && environment" class="bg-white p-4 rounded-lg border border-gray-200">
              <PolicyVisualization
                :grid-size="environment.gridSize"
                :policy-data="policyData"
                :start-state="environment.startState"
                :goal-state="environment.goalState"
                :obstacles="environment.obstacles"
                :actions="['Up', 'Right', 'Down', 'Left']"
                title="Learned Policy (Actor Network)"
                height="500px"
              />
            </div>
            
            <!-- Reward Curve -->
            <LineChart
              v-if="rewardSeries && rewardSeries.length > 0"
              :series="rewardSeries"
              title="Total Reward per Episode"
              x-label="Episode"
              y-label="Total Reward"
              :show-legend="false"
              height="350px"
            />
            
            <!-- Steps Curve -->
            <LineChart
              v-if="stepsSeries && stepsSeries.length > 0"
              :series="stepsSeries"
              title="Steps per Episode"
              x-label="Episode"
              y-label="Steps to Goal"
              :show-legend="false"
              height="350px"
            />
            
            <!-- Loss Curves -->
            <LineChart
              v-if="lossSeries && lossSeries.length > 0"
              :series="lossSeries"
              title="Actor and Critic Loss"
              x-label="Episode"
              y-label="Loss"
              :show-legend="true"
              height="350px"
            />
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
import { ActorCritic } from '~/algorithms/reinforcement-learning/ActorCritic';
import type { GridEnvironment } from '~/algorithms/reinforcement-learning/ActorCritic';
import LineChart from '~/components/visualizations/LineChart.vue';
import EnvironmentRenderer from '~/components/visualizations/EnvironmentRenderer.vue';
import PolicyVisualization from '~/components/visualizations/PolicyVisualization.vue';
import Heatmap from '~/components/visualizations/Heatmap.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { LineSeries } from '~/components/visualizations/LineChart.vue';

import { useAsyncData } from '#app';



// Load content from Nuxt Content
const { data: content } = await useAsyncData(() => 
  queryCollection('content').path('/modules/reinforcement-learning/actor-critic/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores
const moduleStore = useModuleStore();

// Parameters
const parameters = ref({
  actorLearningRate: 0.001,
  criticLearningRate: 0.005,
  discountFactor: 0.99,
  episodes: 200,
  maxStepsPerEpisode: 100,
  actorHiddenLayers: [64, 64],
  criticHiddenLayers: [64, 64],
  entropy: 0.01,
  environmentType: 'simple',
});

// Controls configuration
const controls: ControlConfig[] = [
  {
    id: 'environmentType',
    type: 'select',
    label: 'Environment',
    options: [
      { label: 'Simple Grid', value: 'simple' },
      { label: 'Maze', value: 'maze' },
      { label: 'Cliff Walking', value: 'cliff' }
    ],
    tooltip: 'Choose a grid-world environment with different obstacles and rewards',
    group: 'Environment',
  },
  {
    id: 'actorLearningRate',
    type: 'slider',
    label: 'Actor Learning Rate',
    min: 0.0001,
    max: 0.01,
    step: 0.0001,
    tooltip: 'Learning rate for the policy network (actor). Lower values provide more stable policy updates.',
    group: 'Network Parameters',
  },
  {
    id: 'criticLearningRate',
    type: 'slider',
    label: 'Critic Learning Rate',
    min: 0.001,
    max: 0.02,
    step: 0.001,
    tooltip: 'Learning rate for the value network (critic). Often higher than actor learning rate for faster value learning.',
    group: 'Network Parameters',
  },
  {
    id: 'discountFactor',
    type: 'slider',
    label: 'Discount Factor (γ)',
    min: 0.9,
    max: 0.999,
    step: 0.001,
    tooltip: 'How much the agent values future rewards. Higher values make the agent more far-sighted.',
    group: 'Algorithm Parameters',
  },
  {
    id: 'entropy',
    type: 'slider',
    label: 'Entropy Coefficient',
    min: 0.0,
    max: 0.1,
    step: 0.001,
    tooltip: 'Entropy bonus for exploration. Higher values encourage more exploration by penalizing deterministic policies.',
    group: 'Algorithm Parameters',
  },
  {
    id: 'episodes',
    type: 'slider',
    label: 'Training Episodes',
    min: 100,
    max: 500,
    step: 50,
    tooltip: 'Number of episodes to train the agent. More episodes allow better learning but take longer.',
    group: 'Training',
  },
  {
    id: 'maxStepsPerEpisode',
    type: 'slider',
    label: 'Max Steps per Episode',
    min: 50,
    max: 200,
    step: 10,
    tooltip: 'Maximum number of steps allowed in each episode before timeout.',
    group: 'Training',
  },
];

// Training state
const isTraining = ref(false);
const trainingProgress = ref(0);
const error = ref<string | null>(null);

// Results
const metrics = ref<Metrics | null>(null);
const rewardSeries = ref<LineSeries[]>([]);
const stepsSeries = ref<LineSeries[]>([]);
const lossSeries = ref<LineSeries[]>([]);
const environment = ref<GridEnvironment | null>(null);
const stateValues = ref<number[][] | null>(null);
const policyData = ref<{ [key: string]: number[][] } | null>(null);
const optimalPath = ref<[number, number][]>([]);

// Algorithm instance
let algorithm: ActorCritic | null = null;

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
  moduleStore.setCurrentModule('actor-critic');
  moduleStore.setParameters(parameters.value);
  
  // Load quiz data
  await loadQuiz('actor-critic', '/content/modules/reinforcement-learning/actor-critic/quiz.json');
  
  // Initialize environment
  initializeEnvironment();
});

/**
 * Create environment based on type
 */
const createEnvironment = (type: string): GridEnvironment => {
  const gridSize = 5;
  
  if (type === 'simple') {
    return {
      gridSize,
      startState: [0, 0],
      goalState: [4, 4],
      obstacles: [[1, 1], [2, 2], [3, 1]],
      rewards: new Map([
        ['goal', 100],
        ['default', -1]
      ])
    };
  } else if (type === 'maze') {
    return {
      gridSize,
      startState: [0, 0],
      goalState: [4, 4],
      obstacles: [
        [0, 2], [1, 2], [2, 2],
        [2, 0], [3, 2], [3, 3]
      ],
      rewards: new Map([
        ['goal', 100],
        ['default', -1]
      ])
    };
  } else { // cliff
    return {
      gridSize,
      startState: [4, 0],
      goalState: [4, 4],
      obstacles: [[4, 1], [4, 2], [4, 3]], // Cliff
      rewards: new Map([
        ['goal', 100],
        ['4,1', -100], // Cliff penalty
        ['4,2', -100],
        ['4,3', -100],
        ['default', -1]
      ])
    };
  }
};

/**
 * Initialize environment
 */
const initializeEnvironment = () => {
  environment.value = createEnvironment(parameters.value.environmentType);
};

/**
 * Handle parameter change
 */
const handleParameterChange = (id: string, value: any) => {
  parameters.value[id] = value;
  moduleStore.setParameter(id, value);
  
  // Reinitialize environment if type changed
  if (id === 'environmentType') {
    initializeEnvironment();
    // Clear previous results
    clearResults();
  }
};

/**
 * Clear previous results
 */
const clearResults = () => {
  metrics.value = null;
  rewardSeries.value = [];
  stepsSeries.value = [];
  lossSeries.value = [];
  stateValues.value = null;
  policyData.value = null;
  optimalPath.value = [];
};

/**
 * Handle reset
 */
const handleReset = () => {
  parameters.value = {
    actorLearningRate: 0.001,
    criticLearningRate: 0.005,
    discountFactor: 0.99,
    episodes: 200,
    maxStepsPerEpisode: 100,
    actorHiddenLayers: [64, 64],
    criticHiddenLayers: [64, 64],
    entropy: 0.01,
    environmentType: 'simple',
  };
  moduleStore.setParameters(parameters.value);
  
  initializeEnvironment();
  clearResults();
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
    if (!environment.value) {
      throw new Error('No environment initialized');
    }
    
    // Create and train algorithm
    algorithm = new ActorCritic(environment.value, {
      actorLearningRate: parameters.value.actorLearningRate,
      criticLearningRate: parameters.value.criticLearningRate,
      discountFactor: parameters.value.discountFactor,
      episodes: parameters.value.episodes,
      maxStepsPerEpisode: parameters.value.maxStepsPerEpisode,
      actorHiddenLayers: parameters.value.actorHiddenLayers,
      criticHiddenLayers: parameters.value.criticHiddenLayers,
      entropy: parameters.value.entropy
    });
    
    // Train (Actor-Critic doesn't use X, y format)
    const startTime = Date.now();
    await algorithm.fit([], []);
    const endTime = Date.now();
    
    console.log(`Training completed in ${endTime - startTime}ms`);
    
    // Get results
    metrics.value = algorithm.getMetrics();
    
    // Update visualizations
    updateVisualizations();
    
    trainingProgress.value = 1;
  } catch (err) {
    console.error('Training error:', err);
    error.value = err instanceof Error ? err.message : 'Training failed. Please try again.';
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
  for (const viz of visualizations) {
    if (viz.type === 'line') {
      if (viz.data?.rewards) {
        rewardSeries.value = [{
          x: viz.data.rewards.x,
          y: viz.data.rewards.y,
          name: viz.data.rewards.name || 'Total Reward',
          color: viz.data.rewards.line?.color || '#10b981',
          width: viz.data.rewards.line?.width || 2,
        }];
      }
      
      if (viz.data?.steps) {
        stepsSeries.value = [{
          x: viz.data.steps.x,
          y: viz.data.steps.y,
          name: viz.data.steps.name || 'Steps per Episode',
          color: viz.data.steps.line?.color || '#3b82f6',
          width: viz.data.steps.line?.width || 2,
        }];
      }
      
      if (viz.data?.actorLoss && viz.data?.criticLoss) {
        lossSeries.value = [
          {
            x: viz.data.actorLoss.x,
            y: viz.data.actorLoss.y,
            name: viz.data.actorLoss.name || 'Actor Loss',
            color: viz.data.actorLoss.line?.color || '#ef4444',
            width: viz.data.actorLoss.line?.width || 2,
          },
          {
            x: viz.data.criticLoss.x,
            y: viz.data.criticLoss.y,
            name: viz.data.criticLoss.name || 'Critic Loss',
            color: viz.data.criticLoss.line?.color || '#f59e0b',
            width: viz.data.criticLoss.line?.width || 2,
          }
        ];
      }
    } else if (viz.type === 'heatmap') {
      if (viz.data?.z) {
        stateValues.value = viz.data.z;
      }
    } else if (viz.type === 'policy') {
      if (viz.data?.policyData) {
        policyData.value = viz.data.policyData;
      }
    } else if (viz.type === 'scatter2d') {
      if (viz.data?.environment?.optimalPath) {
        optimalPath.value = viz.data.environment.optimalPath;
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
const { handleLoadBookmark } = useBookmarkRestore(parameters, async () => {
  // Reinitialize environment if environmentType changed
  initializeEnvironment();
  // Trigger training with restored parameters
  await handleTrain();
});
</script>