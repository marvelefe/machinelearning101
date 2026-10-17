/**
 * Actor-Critic Algorithm
 * 
 * Implements Actor-Critic methods for reinforcement learning using separate
 * neural networks for policy (actor) and value function (critic). The actor
 * learns the policy while the critic evaluates state values to reduce variance.
 * 
 * Requirements: 4.2, 4.4, 4.5
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics } from '~/types/algorithm';

export interface ActorCriticParams {
  actorLearningRate: number;
  criticLearningRate: number;
  discountFactor: number;
  episodes: number;
  maxStepsPerEpisode: number;
  actorHiddenLayers: number[];
  criticHiddenLayers: number[];
  entropy?: number; // Entropy coefficient for exploration
}

export interface GridEnvironment {
  gridSize: number;
  startState: [number, number];
  goalState: [number, number];
  obstacles: [number, number][];
  rewards: Map<string, number>;
}

export interface Episode {
  episode: number;
  totalReward: number;
  steps: number;
  path: [number, number][];
  actorLoss: number;
  criticLoss: number;
}

export interface StateAction {
  state: number[];
  action: number;
  reward: number;
  nextState: number[];
  done: boolean;
}

/**
 * Simple neural network for function approximation
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
   * Softmax activation for output layer
   */
  private softmax(logits: number[]): number[] {
    const maxLogit = Math.max(...logits);
    const exps = logits.map(x => Math.exp(x - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sumExps);
  }
  
  /**
   * Forward pass through the network
   */
  forward(input: number[], useSoftmax: boolean = false): { output: number[]; activations: number[][] } {
    const activations: number[][] = [input];
    let current = input;
    
    for (let i = 0; i < this.weights.length; i++) {
      const layerWeights = this.weights[i];
      const layerBiases = this.biases[i];
      const next: number[] = [];
      
      for (let j = 0; j < layerWeights.length; j++) {
        let sum = layerBiases[j];
        for (let k = 0; k < current.length; k++) {
          sum += current[k] * layerWeights[j][k];
        }
        next.push(sum);
      }
      
      // Apply activation
      if (i < this.weights.length - 1) {
        // ReLU for hidden layers
        current = next.map(x => this.relu(x));
      } else if (useSoftmax) {
        // Softmax for actor output
        current = this.softmax(next);
      } else {
        // Linear for critic output
        current = next;
      }
      
      activations.push(current);
    }
    
    return { output: current, activations };
  }
  
  /**
   * Predict output for a state
   */
  predict(state: number[], useSoftmax: boolean = false): number[] {
    return this.forward(state, useSoftmax).output;
  }
  
  /**
   * Update network weights using gradient descent
   */
  updateActor(
    state: number[],
    action: number,
    advantage: number,
    learningRate: number,
    entropyCoef: number = 0
  ): number {
    const { output: probabilities, activations } = this.forward(state, true);
    
    // Calculate policy loss: -log(π(a|s)) * A(s,a)
    const actionProb = probabilities[action];
    const policyLoss = -Math.log(Math.max(actionProb, 1e-10)) * advantage;
    
    // Calculate entropy bonus for exploration
    let entropy = 0;
    if (entropyCoef > 0) {
      for (const p of probabilities) {
        if (p > 0) {
          entropy -= p * Math.log(p);
        }
      }
    }
    
    const totalLoss = policyLoss - entropyCoef * entropy;
    
    // Compute gradients for output layer
    const outputGrad = [...probabilities];
    outputGrad[action] -= 1; // Gradient of log probability
    
    // Scale by advantage
    for (let i = 0; i < outputGrad.length; i++) {
      outputGrad[i] *= advantage;
    }
    
    // Add entropy gradient
    if (entropyCoef > 0) {
      for (let i = 0; i < outputGrad.length; i++) {
        outputGrad[i] -= entropyCoef * (Math.log(probabilities[i]) + 1);
      }
    }
    
    // Backpropagate
    this.backpropagate(activations, outputGrad, learningRate, true);
    
    return totalLoss;
  }
  
  /**
   * Update critic network
   */
  updateCritic(
    state: number[],
    target: number,
    learningRate: number
  ): number {
    const { output, activations } = this.forward(state, false);
    
    // Calculate TD error
    const prediction = output[0];
    const error = target - prediction;
    const loss = error * error;
    
    // Gradient for output layer
    const outputGrad = [-2 * error]; // Derivative of MSE
    
    // Backpropagate
    this.backpropagate(activations, outputGrad, learningRate, false);
    
    return loss;
  }
  
  /**
   * Backpropagation through the network
   */
  private backpropagate(
    activations: number[][],
    outputGrad: number[],
    learningRate: number,
    isSoftmax: boolean
  ): void {
    const gradients: number[][] = [outputGrad];
    
    // Backpropagate through hidden layers (from output to input)
    for (let i = this.weights.length - 1; i >= 0; i--) {
      const layerGrad: number[] = new Array(this.layerSizes[i]).fill(0);
      const nextGrad = gradients[gradients.length - 1];
      
      for (let j = 0; j < this.layerSizes[i]; j++) {
        for (let k = 0; k < nextGrad.length; k++) {
          layerGrad[j] += nextGrad[k] * this.weights[i][k][j];
        }
        // Apply ReLU derivative for hidden layers (not for input layer)
        if (i > 0 && activations[i][j] <= 0) {
          layerGrad[j] = 0;
        }
      }
      
      gradients.push(layerGrad);
    }
    
    gradients.reverse();
    
    // Update weights and biases
    for (let i = 0; i < this.weights.length; i++) {
      const grad = gradients[i + 1];
      const prevActivation = activations[i];
      
      for (let j = 0; j < this.weights[i].length; j++) {
        // Update bias
        this.biases[i][j] -= learningRate * grad[j];
        
        // Update weights
        for (let k = 0; k < this.weights[i][j].length; k++) {
          this.weights[i][j][k] -= learningRate * grad[j] * prevActivation[k];
        }
      }
    }
  }
}

export class ActorCritic extends BaseAlgorithm {
  name = 'Actor-Critic';
  
  // Neural networks
  private actorNetwork: NeuralNetwork;
  private criticNetwork: NeuralNetwork;
  
  // Environment
  private environment: GridEnvironment;
  
  // Hyperparameters
  private actorLearningRate: number;
  private criticLearningRate: number;
  private discountFactor: number;
  private episodes: number;
  private maxStepsPerEpisode: number;
  private entropyCoef: number;
  
  // Training history
  private episodeHistory: Episode[] = [];
  private rewardHistory: number[] = [];
  private stepsHistory: number[] = [];
  private actorLossHistory: number[] = [];
  private criticLossHistory: number[] = [];
  
  // Actions: up, right, down, left
  private readonly actions = [
    [-1, 0], // up
    [0, 1],  // right
    [1, 0],  // down
    [0, -1]  // left
  ];
  
  constructor(
    environment: GridEnvironment,
    params: ActorCriticParams = {
      actorLearningRate: 0.001,
      criticLearningRate: 0.005,
      discountFactor: 0.99,
      episodes: 200,
      maxStepsPerEpisode: 100,
      actorHiddenLayers: [64, 64],
      criticHiddenLayers: [64, 64],
      entropy: 0.01
    }
  ) {
    super();
    this.environment = environment;
    this.actorLearningRate = params.actorLearningRate;
    this.criticLearningRate = params.criticLearningRate;
    this.discountFactor = params.discountFactor;
    this.episodes = params.episodes;
    this.maxStepsPerEpisode = params.maxStepsPerEpisode;
    this.entropyCoef = params.entropy ?? 0.01;
    
    // Initialize networks
    const stateSize = 2; // [row, col]
    const actionSize = this.actions.length;
    this.actorNetwork = new NeuralNetwork(stateSize, params.actorHiddenLayers, actionSize);
    this.criticNetwork = new NeuralNetwork(stateSize, params.criticHiddenLayers, 1);
  }
  
  /**
   * Convert state coordinates to normalized array
   */
  private stateToArray(state: [number, number]): number[] {
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
   * Sample action from actor policy
   */
  private sampleAction(state: [number, number]): number {
    const stateArray = this.stateToArray(state);
    const probabilities = this.actorNetwork.predict(stateArray, true);
    
    // Sample from categorical distribution
    const rand = Math.random();
    let cumulative = 0;
    
    for (let i = 0; i < probabilities.length; i++) {
      cumulative += probabilities[i];
      if (rand < cumulative) {
        return i;
      }
    }
    
    return probabilities.length - 1;
  }
  
  /**
   * Get state value from critic
   */
  private getStateValue(state: [number, number]): number {
    const stateArray = this.stateToArray(state);
    return this.criticNetwork.predict(stateArray, false)[0];
  }
  
  /**
   * Run one episode with online learning
   */
  private runEpisode(): Episode {
    let state = [...this.environment.startState] as [number, number];
    let totalReward = 0;
    let steps = 0;
    const path: [number, number][] = [state];
    let totalActorLoss = 0;
    let totalCriticLoss = 0;
    
    while (steps < this.maxStepsPerEpisode && !this.isGoalState(state)) {
      // Sample action from actor
      const action = this.sampleAction(state);
      
      // Take action
      const nextState = this.takeAction(state, action);
      const reward = this.getReward(nextState);
      const done = this.isGoalState(nextState);
      
      // Get state values
      const stateValue = this.getStateValue(state);
      const nextStateValue = done ? 0 : this.getStateValue(nextState);
      
      // Calculate TD error (advantage)
      const tdTarget = reward + this.discountFactor * nextStateValue;
      const advantage = tdTarget - stateValue;
      
      // Update critic
      const stateArray = this.stateToArray(state);
      const criticLoss = this.criticNetwork.updateCritic(
        stateArray,
        tdTarget,
        this.criticLearningRate
      );
      totalCriticLoss += criticLoss;
      
      // Update actor
      const actorLoss = this.actorNetwork.updateActor(
        stateArray,
        action,
        advantage,
        this.actorLearningRate,
        this.entropyCoef
      );
      totalActorLoss += actorLoss;
      
      // Update state
      state = nextState;
      totalReward += reward;
      steps++;
      path.push(state);
    }
    
    const avgActorLoss = steps > 0 ? totalActorLoss / steps : 0;
    const avgCriticLoss = steps > 0 ? totalCriticLoss / steps : 0;
    
    return {
      episode: this.episodeHistory.length,
      totalReward,
      steps,
      path,
      actorLoss: avgActorLoss,
      criticLoss: avgCriticLoss
    };
  }
  
  /**
   * Train the Actor-Critic agent
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    this.episodeHistory = [];
    this.rewardHistory = [];
    this.stepsHistory = [];
    this.actorLossHistory = [];
    this.criticLossHistory = [];
    
    this.trainingHistory = {
      epochs: [],
      loss: []
    };
    
    for (let episode = 0; episode < this.episodes; episode++) {
      // Run episode with online learning
      const episodeData = this.runEpisode();
      
      // Record history
      this.episodeHistory.push(episodeData);
      this.rewardHistory.push(episodeData.totalReward);
      this.stepsHistory.push(episodeData.steps);
      this.actorLossHistory.push(episodeData.actorLoss);
      this.criticLossHistory.push(episodeData.criticLoss);
      
      this.trainingHistory.epochs.push(episode);
      this.trainingHistory.loss.push(episodeData.actorLoss + episodeData.criticLoss);
      
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
      
      // Use greedy action selection (most probable action)
      const probabilities = this.actorNetwork.predict(stateArray, true);
      let maxProb = -Infinity;
      let bestAction = 0;
      
      for (let a = 0; a < probabilities.length; a++) {
        if (probabilities[a] > maxProb) {
          maxProb = probabilities[a];
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
    
    // Actor and Critic loss curves
    if (this.actorLossHistory.length > 0) {
      const windowSize = 10;
      const smoothedActorLoss: number[] = [];
      const smoothedCriticLoss: number[] = [];
      
      for (let i = 0; i < this.actorLossHistory.length; i++) {
        const start = Math.max(0, i - windowSize + 1);
        const actorWindow = this.actorLossHistory.slice(start, i + 1);
        const criticWindow = this.criticLossHistory.slice(start, i + 1);
        
        const avgActor = actorWindow.reduce((a, b) => a + b, 0) / actorWindow.length;
        const avgCritic = criticWindow.reduce((a, b) => a + b, 0) / criticWindow.length;
        
        smoothedActorLoss.push(avgActor);
        smoothedCriticLoss.push(avgCritic);
      }
      
      visualizations.push({
        type: 'line',
        data: {
          actorLoss: {
            x: Array.from({ length: smoothedActorLoss.length }, (_, i) => i),
            y: smoothedActorLoss,
            mode: 'lines',
            name: 'Actor Loss (Smoothed)',
            line: { color: '#ef4444', width: 2 }
          },
          criticLoss: {
            x: Array.from({ length: smoothedCriticLoss.length }, (_, i) => i),
            y: smoothedCriticLoss,
            mode: 'lines',
            name: 'Critic Loss (Smoothed)',
            line: { color: '#f59e0b', width: 2 }
          }
        }
      });
    }
    
    // Value function heatmap (from critic)
    const { gridSize } = this.environment;
    const valueGrid: number[][] = [];
    
    for (let row = 0; row < gridSize; row++) {
      const rowValues: number[] = [];
      for (let col = 0; col < gridSize; col++) {
        const state: [number, number] = [row, col];
        if (!this.isValidState(state)) {
          rowValues.push(NaN);
        } else {
          const value = this.getStateValue(state);
          rowValues.push(value);
        }
      }
      valueGrid.push(rowValues);
    }
    
    visualizations.push({
      type: 'heatmap',
      data: {
        z: valueGrid,
        colorscale: 'Viridis',
        showscale: true,
        title: 'State Values (Critic)'
      }
    });
    
    // Policy visualization (action probabilities from actor)
    const policyData: { [key: string]: number[][] } = {};
    
    for (let actionIdx = 0; actionIdx < this.actions.length; actionIdx++) {
      const actionGrid: number[][] = [];
      
      for (let row = 0; row < gridSize; row++) {
        const rowValues: number[] = [];
        for (let col = 0; col < gridSize; col++) {
          const state: [number, number] = [row, col];
          if (!this.isValidState(state)) {
            rowValues.push(NaN);
          } else {
            const stateArray = this.stateToArray(state);
            const probabilities = this.actorNetwork.predict(stateArray, true);
            rowValues.push(probabilities[actionIdx]);
          }
        }
        actionGrid.push(rowValues);
      }
      
      policyData[`action_${actionIdx}`] = actionGrid;
    }
    
    visualizations.push({
      type: 'policy',
      data: {
        policyData,
        gridSize,
        actions: ['Up', 'Right', 'Down', 'Left']
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
      const probabilities = this.actorNetwork.predict(stateArray, true);
      
      let maxProb = -Infinity;
      let bestAction = 0;
      for (let a = 0; a < probabilities.length; a++) {
        if (probabilities[a] > maxProb) {
          maxProb = probabilities[a];
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
    
    const avgActorLoss = this.actorLossHistory.length > 0
      ? this.actorLossHistory.slice(-10).reduce((a, b) => a + b, 0) / Math.min(10, this.actorLossHistory.length)
      : 0;
    
    const avgCriticLoss = this.criticLossHistory.length > 0
      ? this.criticLossHistory.slice(-10).reduce((a, b) => a + b, 0) / Math.min(10, this.criticLossHistory.length)
      : 0;
    
    return {
      averageReward: avgReward,
      averageSteps: avgSteps,
      optimalPathLength,
      successRate,
      totalEpisodes: this.episodes,
      averageActorLoss: avgActorLoss,
      averageCriticLoss: avgCriticLoss,
      entropyCoefficient: this.entropyCoef
    };
  }
  
  /**
   * Get actor network for visualization
   */
  getActorNetwork(): NeuralNetwork {
    return this.actorNetwork;
  }
  
  /**
   * Get critic network for visualization
   */
  getCriticNetwork(): NeuralNetwork {
    return this.criticNetwork;
  }
  
  /**
   * Get episode history
   */
  getEpisodeHistory(): Episode[] {
    return [...this.episodeHistory];
  }
  
  /**
   * Get environment configuration
   */
  getEnvironment(): GridEnvironment {
    return {
      ...this.environment,
      obstacles: [...this.environment.obstacles],
      rewards: new Map(this.environment.rewards)
    };
  }
  
  /**
   * Reset the algorithm
   */
  override reset(): void {
    super.reset();
    this.episodeHistory = [];
    this.rewardHistory = [];
    this.stepsHistory = [];
    this.actorLossHistory = [];
    this.criticLossHistory = [];
  }
}
