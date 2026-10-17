/**
 * Deep Q-Network (DQN) Algorithm
 * 
 * Implements Deep Q-Learning with neural network Q-function approximation,
 * experience replay buffer, and target network for stable training.
 * 
 * Requirements: 4.2, 4.4, 4.5
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics } from '~/types/algorithm';

export interface DQNParams {
  learningRate: number;
  discountFactor: number;
  epsilon: number;
  epsilonDecay: number;
  minEpsilon: number;
  episodes: number;
  maxStepsPerEpisode: number;
  batchSize: number;
  replayBufferSize: number;
  targetUpdateFrequency: number;
  hiddenLayers: number[];
}

export interface GridEnvironment {
  gridSize: number;
  startState: [number, number];
  goalState: [number, number];
  obstacles: [number, number][];
  rewards: Map<string, number>;
}

export interface Experience {
  state: number[];
  action: number;
  reward: number;
  nextState: number[];
  done: boolean;
}

export interface Episode {
  episode: number;
  totalReward: number;
  steps: number;
  path: [number, number][];
  epsilon: number;
}

/**
 * Simple neural network for Q-value approximation
 */
class NeuralNetwork {
  private weights: number[][][];
  private biases: number[][];
  private layerSizes: number[];
  
  constructor(inputSize: number, hiddenLayers: number[], outputSize: number) {
    this.layerSizes = [inputSize, ...hiddenLayers, outputSize];
    this.weights = [];
    this.biases = [];
    
    // Initialize weights and biases with Xavier initialization
    for (let i = 0; i < this.layerSizes.length - 1; i++) {
      const inputDim = this.layerSizes[i];
      const outputDim = this.layerSizes[i + 1];
      
      // Xavier initialization
      const scale = Math.sqrt(2.0 / (inputDim + outputDim));
      
      const layerWeights: number[][] = [];
      for (let j = 0; j < outputDim; j++) {
        const neuronWeights: number[] = [];
        for (let k = 0; k < inputDim; k++) {
          neuronWeights.push((Math.random() * 2 - 1) * scale);
        }
        layerWeights.push(neuronWeights);
      }
      this.weights.push(layerWeights);
      
      const layerBiases: number[] = new Array(outputDim).fill(0);
      this.biases.push(layerBiases);
    }
  }
  
  /**
   * ReLU activation function
   */
  private relu(x: number): number {
    return Math.max(0, x);
  }
  
  /**
   * ReLU derivative
   */
  private reluDerivative(x: number): number {
    return x > 0 ? 1 : 0;
  }
  
  /**
   * Forward pass through the network
   */
  forward(input: number[]): { output: number[]; activations: number[][]; preActivations: number[][] } {
    const activations: number[][] = [input];
    const preActivations: number[][] = [];
    let current = input;
    
    for (let i = 0; i < this.weights.length; i++) {
      const layerWeights = this.weights[i];
      const layerBiases = this.biases[i];
      const next: number[] = [];
      const preAct: number[] = [];
      
      for (let j = 0; j < layerWeights.length; j++) {
        let sum = layerBiases[j];
        for (let k = 0; k < current.length; k++) {
          sum += current[k] * layerWeights[j][k];
        }
        
        preAct.push(sum);
        
        // Apply ReLU to hidden layers, linear for output layer
        if (i < this.weights.length - 1) {
          next.push(this.relu(sum));
        } else {
          next.push(sum);
        }
      }
      
      preActivations.push(preAct);
      activations.push(next);
      current = next;
    }
    
    return { output: current, activations, preActivations };
  }
  
  /**
   * Predict Q-values for a state
   */
  predict(state: number[]): number[] {
    return this.forward(state).output;
  }
  
  /**
   * Update network weights using gradient descent
   */
  update(
    state: number[],
    action: number,
    target: number,
    learningRate: number
  ): number {
    const { output, activations } = this.forward(state);
    
    // Calculate loss (MSE)
    const prediction = output[action];
    const error = target - prediction;
    const loss = error * error;
    
    // Simple gradient descent update for the selected action only
    // Compute gradients layer by layer
    const deltas: number[][] = [];
    
    // Output layer delta
    const outputDelta = new Array(output.length).fill(0);
    outputDelta[action] = -2 * error; // Derivative of MSE
    deltas.push(outputDelta);
    
    // Backpropagate through hidden layers
    for (let i = this.weights.length - 1; i >= 0; i--) {
      const currentDelta = deltas[deltas.length - 1];
      const prevDelta: number[] = new Array(this.layerSizes[i]).fill(0);
      
      // Compute delta for previous layer
      for (let j = 0; j < this.layerSizes[i]; j++) {
        let sum = 0;
        for (let k = 0; k < currentDelta.length; k++) {
          sum += currentDelta[k] * this.weights[i][k][j];
        }
        // Apply ReLU derivative (only for hidden layers, not input)
        if (i > 0) {
          prevDelta[j] = sum * this.reluDerivative(activations[i][j]);
        } else {
          prevDelta[j] = sum;
        }
      }
      
      deltas.push(prevDelta);
    }
    
    deltas.reverse();
    
    // Update weights and biases
    for (let i = 0; i < this.weights.length; i++) {
      const delta = deltas[i + 1];
      const prevActivation = activations[i];
      
      for (let j = 0; j < this.weights[i].length; j++) {
        // Update bias
        this.biases[i][j] -= learningRate * delta[j];
        
        // Update weights
        for (let k = 0; k < this.weights[i][j].length; k++) {
          this.weights[i][j][k] -= learningRate * delta[j] * prevActivation[k];
        }
      }
    }
    
    return loss;
  }
  
  /**
   * Copy weights from another network
   */
  copyFrom(other: NeuralNetwork): void {
    for (let i = 0; i < this.weights.length; i++) {
      for (let j = 0; j < this.weights[i].length; j++) {
        for (let k = 0; k < this.weights[i][j].length; k++) {
          this.weights[i][j][k] = other.weights[i][j][k];
        }
      }
      for (let j = 0; j < this.biases[i].length; j++) {
        this.biases[i][j] = other.biases[i][j];
      }
    }
  }
}

export class DQN extends BaseAlgorithm {
  name = 'Deep Q-Network';
  
  // Neural networks
  private qNetwork: NeuralNetwork;
  private targetNetwork: NeuralNetwork;
  
  // Environment
  private environment: GridEnvironment;
  
  // Hyperparameters
  private learningRate: number;
  private discountFactor: number;
  private epsilon: number;
  private epsilonDecay: number;
  private minEpsilon: number;
  private episodes: number;
  private maxStepsPerEpisode: number;
  private batchSize: number;
  private targetUpdateFrequency: number;
  
  // Experience replay buffer
  private replayBuffer: Experience[];
  private replayBufferSize: number;
  
  // Training history
  private episodeHistory: Episode[] = [];
  private rewardHistory: number[] = [];
  private stepsHistory: number[] = [];
  private lossHistory: number[] = [];
  
  // Actions: up, right, down, left
  private readonly actions = [
    [-1, 0], // up
    [0, 1],  // right
    [1, 0],  // down
    [0, -1]  // left
  ];
  
  constructor(
    environment: GridEnvironment,
    params: DQNParams = {
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
      hiddenLayers: [64, 64]
    }
  ) {
    super();
    this.environment = environment;
    this.learningRate = params.learningRate;
    this.discountFactor = params.discountFactor;
    this.epsilon = params.epsilon;
    this.epsilonDecay = params.epsilonDecay;
    this.minEpsilon = params.minEpsilon;
    this.episodes = params.episodes;
    this.maxStepsPerEpisode = params.maxStepsPerEpisode;
    this.batchSize = params.batchSize;
    this.replayBufferSize = params.replayBufferSize;
    this.targetUpdateFrequency = params.targetUpdateFrequency;
    
    // Initialize networks
    const stateSize = 2; // [row, col]
    const actionSize = this.actions.length;
    this.qNetwork = new NeuralNetwork(stateSize, params.hiddenLayers, actionSize);
    this.targetNetwork = new NeuralNetwork(stateSize, params.hiddenLayers, actionSize);
    this.targetNetwork.copyFrom(this.qNetwork);
    
    // Initialize replay buffer
    this.replayBuffer = [];
  }
  
  /**
   * Convert state coordinates to normalized array
   */
  private stateToArray(state: [number, number]): number[] {
    // Normalize to [0, 1]
    return [
      state[0] / this.environment.gridSize,
      state[1] / this.environment.gridSize
    ];
  }
  
  /**
   * Check if state is valid
   */
  private isValidState(state: [number, number]): boolean {
    const [row, col] = state;
    const { gridSize, obstacles } = this.environment;
    
    if (row < 0 || row >= gridSize || col < 0 || col >= gridSize) {
      return false;
    }
    
    for (const obstacle of obstacles) {
      if (obstacle[0] === row && obstacle[1] === col) {
        return false;
      }
    }
    
    return true;
  }
  
  /**
   * Check if state is goal state
   */
  private isGoalState(state: [number, number]): boolean {
    const [row, col] = state;
    const [goalRow, goalCol] = this.environment.goalState;
    return row === goalRow && col === goalCol;
  }
  
  /**
   * Get reward for a state
   */
  private getReward(state: [number, number]): number {
    const key = `${state[0]},${state[1]}`;
    
    if (this.isGoalState(state)) {
      return this.environment.rewards.get('goal') ?? 100;
    }
    
    if (this.environment.rewards.has(key)) {
      return this.environment.rewards.get(key)!;
    }
    
    return this.environment.rewards.get('default') ?? -1;
  }
  
  /**
   * Take action and get next state
   */
  private takeAction(state: [number, number], action: number): [number, number] {
    const [row, col] = state;
    const [dRow, dCol] = this.actions[action];
    const nextState: [number, number] = [row + dRow, col + dCol];
    
    if (!this.isValidState(nextState)) {
      return state;
    }
    
    return nextState;
  }
  
  /**
   * Select action using epsilon-greedy policy
   */
  private selectAction(state: [number, number], currentEpsilon: number): number {
    if (Math.random() < currentEpsilon) {
      return Math.floor(Math.random() * this.actions.length);
    }
    
    const stateArray = this.stateToArray(state);
    const qValues = this.qNetwork.predict(stateArray);
    
    let maxQ = -Infinity;
    let bestAction = 0;
    for (let a = 0; a < qValues.length; a++) {
      if (qValues[a] > maxQ) {
        maxQ = qValues[a];
        bestAction = a;
      }
    }
    
    return bestAction;
  }
  
  /**
   * Add experience to replay buffer
   */
  private addExperience(experience: Experience): void {
    this.replayBuffer.push(experience);
    if (this.replayBuffer.length > this.replayBufferSize) {
      this.replayBuffer.shift();
    }
  }
  
  /**
   * Sample random batch from replay buffer
   */
  private sampleBatch(): Experience[] {
    const batch: Experience[] = [];
    const batchSize = Math.min(this.batchSize, this.replayBuffer.length);
    
    for (let i = 0; i < batchSize; i++) {
      const idx = Math.floor(Math.random() * this.replayBuffer.length);
      batch.push(this.replayBuffer[idx]);
    }
    
    return batch;
  }
  
  /**
   * Train on a batch of experiences
   */
  private trainOnBatch(): number {
    if (this.replayBuffer.length < this.batchSize) {
      return 0;
    }
    
    const batch = this.sampleBatch();
    let totalLoss = 0;
    
    for (const experience of batch) {
      const { state, action, reward, nextState, done } = experience;
      
      // Calculate target Q-value
      let target = reward;
      if (!done) {
        const nextQValues = this.targetNetwork.predict(nextState);
        const maxNextQ = Math.max(...nextQValues);
        target = reward + this.discountFactor * maxNextQ;
      }
      
      // Update Q-network
      const loss = this.qNetwork.update(state, action, target, this.learningRate);
      totalLoss += loss;
    }
    
    return totalLoss / batch.length;
  }
  
  /**
   * Run one episode
   */
  private runEpisode(currentEpsilon: number): Episode {
    let state = [...this.environment.startState] as [number, number];
    let totalReward = 0;
    let steps = 0;
    const path: [number, number][] = [state];
    
    while (steps < this.maxStepsPerEpisode && !this.isGoalState(state)) {
      const action = this.selectAction(state, currentEpsilon);
      const nextState = this.takeAction(state, action);
      const reward = this.getReward(nextState);
      const done = this.isGoalState(nextState);
      
      // Store experience
      this.addExperience({
        state: this.stateToArray(state),
        action,
        reward,
        nextState: this.stateToArray(nextState),
        done
      });
      
      // Train on batch
      const loss = this.trainOnBatch();
      if (loss > 0) {
        this.lossHistory.push(loss);
      }
      
      state = nextState;
      totalReward += reward;
      steps++;
      path.push(state);
    }
    
    return {
      episode: this.episodeHistory.length,
      totalReward,
      steps,
      path,
      epsilon: currentEpsilon
    };
  }
  
  /**
   * Train the DQN agent
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    this.replayBuffer = [];
    this.episodeHistory = [];
    this.rewardHistory = [];
    this.stepsHistory = [];
    this.lossHistory = [];
    
    this.trainingHistory = {
      epochs: [],
      loss: []
    };
    
    let currentEpsilon = this.epsilon;
    
    for (let episode = 0; episode < this.episodes; episode++) {
      const episodeData = this.runEpisode(currentEpsilon);
      
      this.episodeHistory.push(episodeData);
      this.rewardHistory.push(episodeData.totalReward);
      this.stepsHistory.push(episodeData.steps);
      
      this.trainingHistory.epochs.push(episode);
      this.trainingHistory.loss.push(-episodeData.totalReward);
      
      // Update target network
      if (episode % this.targetUpdateFrequency === 0) {
        this.targetNetwork.copyFrom(this.qNetwork);
      }
      
      // Decay epsilon
      currentEpsilon = Math.max(this.minEpsilon, currentEpsilon * this.epsilonDecay);
      
      // Allow UI to update
      if (episode % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    this.trained = true;
  }
  
  /**
   * Get optimal path using learned policy
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    
    const path: number[] = [];
    let state = [...this.environment.startState] as [number, number];
    let steps = 0;
    
    while (steps < this.maxStepsPerEpisode && !this.isGoalState(state)) {
      const stateArray = this.stateToArray(state);
      const qValues = this.qNetwork.predict(stateArray);
      
      let maxQ = -Infinity;
      let bestAction = 0;
      for (let a = 0; a < qValues.length; a++) {
        if (qValues[a] > maxQ) {
          maxQ = qValues[a];
          bestAction = a;
        }
      }
      
      path.push(bestAction);
      state = this.takeAction(state, bestAction);
      steps++;
    }
    
    return path;
  }
  
  /**
   * Get visualization data
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained) {
      return [];
    }
    
    const visualizations: VisualizationData[] = [];
    
    // Reward curve
    visualizations.push({
      type: 'line',
      data: {
        rewards: {
          x: Array.from({ length: this.rewardHistory.length }, (_, i) => i),
          y: this.rewardHistory,
          mode: 'lines',
          name: 'Total Reward',
          line: { color: '#10b981', width: 2 }
        }
      }
    });
    
    // Steps per episode
    visualizations.push({
      type: 'line',
      data: {
        steps: {
          x: Array.from({ length: this.stepsHistory.length }, (_, i) => i),
          y: this.stepsHistory,
          mode: 'lines',
          name: 'Steps per Episode',
          line: { color: '#3b82f6', width: 2 }
        }
      }
    });
    
    // Loss curve
    if (this.lossHistory.length > 0) {
      // Smooth loss with moving average
      const windowSize = 10;
      const smoothedLoss: number[] = [];
      for (let i = 0; i < this.lossHistory.length; i++) {
        const start = Math.max(0, i - windowSize + 1);
        const window = this.lossHistory.slice(start, i + 1);
        const avg = window.reduce((a, b) => a + b, 0) / window.length;
        smoothedLoss.push(avg);
      }
      
      visualizations.push({
        type: 'line',
        data: {
          loss: {
            x: Array.from({ length: smoothedLoss.length }, (_, i) => i),
            y: smoothedLoss,
            mode: 'lines',
            name: 'Training Loss (Smoothed)',
            line: { color: '#ef4444', width: 2 }
          }
        }
      });
    }
    
    // Q-value heatmap
    const { gridSize } = this.environment;
    const qValueGrid: number[][] = [];
    
    for (let row = 0; row < gridSize; row++) {
      const rowValues: number[] = [];
      for (let col = 0; col < gridSize; col++) {
        const state: [number, number] = [row, col];
        if (!this.isValidState(state)) {
          rowValues.push(NaN);
        } else {
          const stateArray = this.stateToArray(state);
          const qValues = this.qNetwork.predict(stateArray);
          rowValues.push(Math.max(...qValues));
        }
      }
      qValueGrid.push(rowValues);
    }
    
    visualizations.push({
      type: 'heatmap',
      data: {
        z: qValueGrid,
        colorscale: 'Viridis',
        showscale: true
      }
    });
    
    // Environment with optimal path
    const optimalPath = this.getOptimalPath();
    
    visualizations.push({
      type: 'scatter2d',
      data: {
        environment: {
          gridSize,
          startState: this.environment.startState,
          goalState: this.environment.goalState,
          obstacles: this.environment.obstacles,
          optimalPath
        }
      }
    });
    
    return visualizations;
  }
  
  /**
   * Get optimal path
   */
  private getOptimalPath(): [number, number][] {
    const path: [number, number][] = [];
    let state = [...this.environment.startState] as [number, number];
    let steps = 0;
    
    path.push(state);
    
    while (steps < this.maxStepsPerEpisode && !this.isGoalState(state)) {
      const stateArray = this.stateToArray(state);
      const qValues = this.qNetwork.predict(stateArray);
      
      let maxQ = -Infinity;
      let bestAction = 0;
      for (let a = 0; a < qValues.length; a++) {
        if (qValues[a] > maxQ) {
          maxQ = qValues[a];
          bestAction = a;
        }
      }
      
      state = this.takeAction(state, bestAction);
      path.push(state);
      steps++;
    }
    
    return path;
  }
  
  /**
   * Get performance metrics
   */
  getMetrics(): Metrics {
    if (!this.trained) {
      return {};
    }
    
    const last10Episodes = this.rewardHistory.slice(-10);
    const avgReward = last10Episodes.reduce((a, b) => a + b, 0) / last10Episodes.length;
    
    const last10Steps = this.stepsHistory.slice(-10);
    const avgSteps = last10Steps.reduce((a, b) => a + b, 0) / last10Steps.length;
    
    const optimalPath = this.getOptimalPath();
    const optimalPathLength = optimalPath.length - 1;
    
    const successCount = this.episodeHistory.slice(-10).filter(ep =>
      this.isGoalState(ep.path[ep.path.length - 1])
    ).length;
    const successRate = successCount / Math.min(10, this.episodeHistory.length);
    
    const avgLoss = this.lossHistory.length > 0
      ? this.lossHistory.slice(-100).reduce((a, b) => a + b, 0) / Math.min(100, this.lossHistory.length)
      : 0;
    
    return {
      averageReward: avgReward,
      averageSteps: avgSteps,
      optimalPathLength,
      successRate,
      totalEpisodes: this.episodes,
      replayBufferSize: this.replayBuffer.length,
      averageLoss: avgLoss
    };
  }
  
  /**
   * Reset the algorithm
   */
  override reset(): void {
    super.reset();
    this.replayBuffer = [];
    this.episodeHistory = [];
    this.rewardHistory = [];
    this.stepsHistory = [];
    this.lossHistory = [];
  }
}
