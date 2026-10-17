<template>
  <ModuleLayout
    title="Deep Q-Networks (DQN)"
    description="Learn how Deep Q-Networks combine neural networks with Q-learning to handle large state spaces using experience replay and target networks"
    difficulty="advanced"
    :estimated-time="50"
    module-id="deep-q-networks"
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
        loading-message="Training DQN agent..."
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
                <div class="text-xs text-gray-500">Replay Buffer Size</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.replayBufferSize }}
                </div>
              </div>
              <div>
                <div class="text-xs text-gray-500">Avg Training Loss</div>
                <div class="text-lg font-semibold text-gray-900">
                  {{ metrics.averageLoss?.toFixed(4) }}
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
            
            <!-- Q-Value Heatmap -->
            <div v-if="qValueGrid && environment" class="bg-white p-4 rounded-lg border border-gray-200">
              <h3 class="text-lg font-semibold mb-4">Learned Q-Value Heatmap</h3>
              <Heatmap
                :data="qValueGrid"
                title="Maximum Q-Values per State"
                x-label="Column"
                y-label="Row"
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
            
            <!-- Loss Curve -->
            <LineChart
              v-if="lossSeries && lossSeries.length > 0"
              :series="lossSeries"
              title="Training Loss (Smoothed)"
              x-label="Training Step"
              y-label="Loss"
              :show-legend="false"
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
import { DQN } from '~/algorithms/reinforcement-learning/DQN';
import type { GridEnvironment } from '~/algorithms/reinforcement-learning/DQN';
import LineChart from '~/components/visualizations/LineChart.vue';
import EnvironmentRenderer from '~/components/visualizations/EnvironmentRenderer.vue';
import Heatmap from '~/components/visualizations/Heatmap.vue';
import type { ControlConfig } from '~/components/modules/ControlPanel.vue';
import type { Metrics } from '~/types/algorithm';
import type { LineSeries } from '~/components/visualizations/LineChart.vue';

import { useAsyncData } from '#app';


// Load content from Nuxt Content
const { data: content } = await useAsyncData(() => 
  queryCollection('content').path('/modules/reinforcement-learning/deep-q-networks/content').first()
);

// Extract table of contents from content
const tableOfContents = useTableOfContents(content);

// Stores
const moduleStore = useModuleStore();

// Parameters
const parameters = ref({
  learningRate: 0.001,
  discountFactor: 0.95,
  epsilon: 1.0,
  epsilonDecay: 0.995,
  minEpsilon: 0.01,
  episodes: 200,
  maxStepsPerEpisode: 100,
  batchSize: 32,
  replayBufferSize: 10000,
  targetUpdateFrequency: 10,
  environmentType: 'simple',
  hiddenLayer1: 64,
  hiddenLayer2: 64,
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
    id: 'learningRate',
    type: 'slider',
    label: 'Learning Rate',
    min: 0.0001,
    max: 0.01,
    step: 0.0001,
    tooltip: 'Learning rate for neural network gradient descent. Lower values train more slowly but more stably.',
    group: 'Neural Network',
  },
  {
    id: 'hiddenLayer1',
    type: 'slider',
    label: 'Hidden Layer 1 Size',
    min: 16,
    max: 128,
    step: 16,
    tooltip: 'Number of neurons in the first hidden layer. More neurons increase capacity but slow training.',
    group: 'Neural Network',
  },
  {
    id: 'hiddenLayer2',
    type: 'slider',
    label: 'Hidden Layer 2 Size',
    min: 16,
    max: 128,
    step: 16,
    tooltip: 'Number of neurons in the second hidden layer. More neurons increase capacity but slow training.',
    group: 'Neural Network',
  },
  {
    id: 'discountFactor',
    type: 'slider',
    label: 'Discount Factor (γ)',
    min: 0.0,
    max: 1.0,
    step: 0.05,
    tooltip: 'How much the agent values future rewards. Higher values make the agent more far-sighted.',
    group: 'RL Parameters',
  },
  {
    id: 'epsilon',
    type: 'slider',
    label: 'Initial Exploration Rate (ε)',
    min: 0.5,
    max: 1.0,
    step: 0.1,
    tooltip: 'Initial probability of taking random actions. DQN typically starts with high exploration.',
    group: 'RL Parameters',
  },
  {
    id: 'epsilonDecay',
    type: 'slider',
    label: 'Epsilon Decay',
    min: 0.95,
    max: 0.999,
    step: 0.001,
    tooltip: 'Rate at which exploration decreases. Higher values decay more slowly.',
    group: 'RL Parameters',
  },
  {
    id: 'batchSize',
    type: 'slider',
    label: 'Batch Size',
    min: 16,
    max: 128,
    step: 16,
    tooltip: 'Number of experiences sampled from replay buffer for each training step.',
    group: 'Experience Replay',
  },
  {
    id: 'replayBufferSize',
    type: 'slider',
    label: 'Replay Buffer Size',
    min: 1000,
    max: 20000,
    step: 1000,
    tooltip: 'Maximum number of experiences stored in the replay buffer.',
    group: 'Experience Replay',
  },
  {
    id: 'targetUpdateFrequency',
    type: 'slider',
    label: 'Target Update Frequency',
    min: 5,
    max: 50,
    step: 5,
    tooltip: 'Number of episodes between target network updates. Higher values provide more stability.',
    group: 'Experience Replay',
  },
  {
    id: 'episodes',
    type: 'slider',
    label: 'Training Episodes',
    min: 100,
    max: 500,
    step: 50,
    tooltip: 'Number of episodes to train the agent. DQN typically needs more episodes than tabular Q-learning.',
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
const qValueGrid = ref<number[][] | null>(null);
const optimalPath = ref<[number, number][]>([]);

// Algorithm instance
let algorithm: DQN | null = null;

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
  moduleStore.setCurrentModule('deep-q-networks');
  moduleStore.setParameters(parameters.value);
  
  // Load quiz data
  await loadQuiz('deep-q-networks', '/content/modules/reinforcement-learning/deep-q-networks/quiz.json');
  
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
    metrics.value = null;
    rewardSeries.value = [];
    stepsSeries.value = [];
    lossSeries.value = [];
    qValueGrid.value = null;
    optimalPath.value = [];
  }
};

/**
 * Handle reset
 */
const handleReset = () => {
  parameters.value = {
    learningRate: 0.001,
    discountFactor: 0.95,
    epsilon: 1.0,
    epsilonDecay: 0.995,
    minEpsilon: 0.01,
    episodes: 200,
    maxStepsPerEpisode: 100,
    batchSize: 32,
    replayBufferSize: 10000,
    targetUpdateFrequency: 10,
    environmentType: 'simple',
    hiddenLayer1: 64,
    hiddenLayer2: 64,
  };
  moduleStore.setParameters(parameters.value);
  
  initializeEnvironment();
  
  metrics.value = null;
  rewardSeries.value = [];
  stepsSeries.value = [];
  lossSeries.value = [];
  qValueGrid.value = null;
  optimalPath.value = [];
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
    algorithm = new DQN(environment.value, {
      learningRate: parameters.value.learningRate,
      discountFactor: parameters.value.discountFactor,
      epsilon: parameters.value.epsilon,
      epsilonDecay: parameters.value.epsilonDecay,
      minEpsilon: parameters.value.minEpsilon,
      episodes: parameters.value.episodes,
      maxStepsPerEpisode: parameters.value.maxStepsPerEpisode,
      batchSize: parameters.value.batchSize,
      replayBufferSize: parameters.value.replayBufferSize,
      targetUpdateFrequency: parameters.value.targetUpdateFrequency,
      hiddenLayers: [parameters.value.hiddenLayer1, parameters.value.hiddenLayer2]
    });
    
    // Train
    const startTime = Date.now();
    await algorithm.fit([], []);
    const endTime = Date.now();
    
    console.log(`Training completed in ${endTime - startTime}ms`);
    
    // Get results
    metrics.value = algorithm.getMetrics();
    
    // Get optimal path
    const env = environment.value;
    let state = [...env.startState] as [number, number];
    optimalPath.value = [state];
    let steps = 0;
    const maxSteps = 100;
    
    while (steps < maxSteps) {
      const actions = algorithm.predict([[...state]]);
      if (actions.length === 0) break;
      
      const action = actions[0];
      const actionVectors = [[-1, 0], [0, 1], [1, 0], [0, -1]];
      const [dRow, dCol] = actionVectors[action];
      const nextState: [number, number] = [state[0] + dRow, state[1] + dCol];
      
      // Check if valid move
      if (nextState[0] < 0 || nextState[0] >= env.gridSize || 
          nextState[1] < 0 || nextState[1] >= env.gridSize) {
        break;
      }
      
      state = nextState;
      optimalPath.value.push(state);
      
      // Check if reached goal
      if (state[0] === env.goalState[0] && state[1] === env.goalState[1]) {
        break;
      }
      
      steps++;
    }
    
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
  if (visualizations.length > 0) {
    // Reward curve
    const rewardViz = visualizations[0];
    if (rewardViz.type === 'line' && rewardViz.data?.rewards) {
      rewardSeries.value = [{
        x: rewardViz.data.rewards.x,
        y: rewardViz.data.rewards.y,
        name: rewardViz.data.rewards.name || 'Total Reward',
        color: rewardViz.data.rewards.line?.color || '#10b981',
        width: rewardViz.data.rewards.line?.width || 2,
      }];
    }
    
    // Steps curve
    if (visualizations.length > 1) {
      const stepsViz = visualizations[1];
      if (stepsViz.type === 'line' && stepsViz.data?.steps) {
        stepsSeries.value = [{
          x: stepsViz.data.steps.x,
          y: stepsViz.data.steps.y,
          name: stepsViz.data.steps.name || 'Steps per Episode',
          color: stepsViz.data.steps.line?.color || '#3b82f6',
          width: stepsViz.data.steps.line?.width || 2,
        }];
      }
    }
    
    // Loss curve
    if (visualizations.length > 2) {
      const lossViz = visualizations[2];
      if (lossViz.type === 'line' && lossViz.data?.loss) {
        lossSeries.value = [{
          x: lossViz.data.loss.x,
          y: lossViz.data.loss.y,
          name: lossViz.data.loss.name || 'Training Loss',
          color: lossViz.data.loss.line?.color || '#ef4444',
          width: lossViz.data.loss.line?.width || 2,
        }];
      }
    }
    
    // Q-value heatmap
    if (visualizations.length > 3) {
      const heatmapViz = visualizations[3];
      if (heatmapViz.type === 'heatmap' && heatmapViz.data?.z) {
        qValueGrid.value = heatmapViz.data.z;
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
