/**
 * Policy Gradient Algorithm (REINFORCE)
 *
 * Implements policy gradient methods for reinforcement learning using
 * a neural network to directly learn a stochastic policy. Uses Monte Carlo
 * sampling and gradient ascent to optimize expected returns.
 *
 * Requirements: 4.2, 4.4, 4.5
 */

import { BaseAlgorithm } from "~/types/algorithm";
import type { VisualizationData, Metrics } from "~/types/algorithm";

export interface PolicyGradientParams {
  learningRate: number;
  discountFactor: number;
  episodes: number;
  maxStepsPerEpisode: number;
  hiddenLayers: number[];
  baseline?: boolean; // Use baseline (value function) to reduce variance
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
  returns: number[];
}

export interface StateAction {
  state: number[];
  action: number;
  reward: number;
}

/**
 * Simple neural network for policy approximation
 */
class PolicyNetwork {
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
    const exps = logits.map((x) => Math.exp(x - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map((x) => x / sumExps);
  }

  /**
   * Forward pass through the network
   */
  forward(input: number[]): {
    probabilities: number[];
    activations: number[][];
  } {
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

      // Apply ReLU to hidden layers
      if (i < this.weights.length - 1) {
        current = next.map((x) => this.relu(x));
      } else {
        // Apply softmax to output layer
        current = this.softmax(next);
      }

      activations.push(current);
    }

    return { probabilities: current, activations };
  }

  /**
   * Get action probabilities for a state
   */
  getActionProbabilities(state: number[]): number[] {
    return this.forward(state).probabilities;
  }

  /**
   * Sample action from policy
   */
  sampleAction(state: number[]): number {
    const probabilities = this.getActionProbabilities(state);

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
   * Update network using policy gradient
   */
  update(
    states: number[][],
    actions: number[],
    returns: number[],
    learningRate: number
  ): number {
    let totalLoss = 0;

    for (let t = 0; t < states.length; t++) {
      const state = states[t];
      const action = actions[t];
      const G = returns[t];

      const { probabilities, activations } = this.forward(state);

      // Calculate loss: -log(π(a|s)) * G
      const actionProb = probabilities[action];
      const loss = -Math.log(Math.max(actionProb, 1e-10)) * G;
      totalLoss += loss;

      // Compute gradients
      const outputGrad = [...probabilities];
      outputGrad[action] -= 1; // Gradient of log probability

      // Scale by return
      for (let i = 0; i < outputGrad.length; i++) {
        outputGrad[i] *= G;
      }

      // Backpropagate
      this.backpropagate(activations, outputGrad, learningRate);
    }

    return totalLoss / states.length;
  }

  /**
   * Backpropagation through the network
   */
  private backpropagate(
    activations: number[][],
    outputGrad: number[],
    learningRate: number
  ): void {
    // Store gradients for each layer (output to input)
    const layerGradients: number[][] = [];
    layerGradients.push(outputGrad);

    // Backpropagate through hidden layers (from output to input)
    for (let i = this.weights.length - 1; i >= 0; i--) {
      const currentGrad = layerGradients[layerGradients.length - 1];
      const prevLayerSize = this.layerSizes[i];
      const prevLayerGrad: number[] = new Array(prevLayerSize).fill(0);

      // Compute gradient for previous layer
      for (let j = 0; j < prevLayerSize; j++) {
        let grad = 0;
        for (let k = 0; k < currentGrad.length; k++) {
          // weights[i][k][j] means: layer i, neuron k in next layer, weight from neuron j in current layer
          grad += currentGrad[k] * this.weights[i][k][j];
        }

        // Apply ReLU derivative for hidden layers (not for input layer)
        if (i > 0 && activations[i][j] <= 0) {
          grad = 0;
        }

        prevLayerGrad[j] = grad;
      }

      layerGradients.push(prevLayerGrad);
    }

    // Reverse to get gradients from input to output
    layerGradients.reverse();

    // Update weights and biases for each layer
    for (let i = 0; i < this.weights.length; i++) {
      const currentLayerGrad = layerGradients[i + 1]; // Gradient for current layer output
      const prevActivation = activations[i]; // Activation from previous layer

      // Update each neuron in the current layer
      for (let j = 0; j < this.weights[i].length; j++) {
        // Update bias
        this.biases[i][j] -= learningRate * currentLayerGrad[j];

        // Update weights
        for (let k = 0; k < this.weights[i][j].length; k++) {
          this.weights[i][j][k] -=
            learningRate * currentLayerGrad[j] * prevActivation[k];
        }
      }
    }
  }
}

export class PolicyGradient extends BaseAlgorithm {
  name = "Policy Gradient";

  // Policy network
  private policyNetwork: PolicyNetwork;

  // Environment
  private environment: GridEnvironment;

  // Hyperparameters
  private learningRate: number;
  private discountFactor: number;
  private episodes: number;
  private maxStepsPerEpisode: number;
  private useBaseline: boolean;

  // Training history
  private episodeHistory: Episode[] = [];
  private rewardHistory: number[] = [];
  private stepsHistory: number[] = [];
  private lossHistory: number[] = [];

  // Actions: up, right, down, left
  private readonly actions = [
    [-1, 0], // up
    [0, 1], // right
    [1, 0], // down
    [0, -1], // left
  ];

  constructor(
    environment: GridEnvironment,
    params: PolicyGradientParams = {
      learningRate: 0.001,
      discountFactor: 0.99,
      episodes: 200,
      maxStepsPerEpisode: 100,
      hiddenLayers: [64, 64],
      baseline: true,
    }
  ) {
    super();
    this.environment = environment;
    this.learningRate = params.learningRate;
    this.discountFactor = params.discountFactor;
    this.episodes = params.episodes;
    this.maxStepsPerEpisode = params.maxStepsPerEpisode;
    this.useBaseline = params.baseline ?? true;

    // Initialize policy network
    const stateSize = 2; // [row, col]
    const actionSize = this.actions.length;
    this.policyNetwork = new PolicyNetwork(
      stateSize,
      params.hiddenLayers,
      actionSize
    );
  }

  /**
   * Convert state coordinates to normalized array
   */
  private stateToArray(state: [number, number]): number[] {
    return [
      state[0] / this.environment.gridSize,
      state[1] / this.environment.gridSize,
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
      return this.environment.rewards.get("goal") ?? 100;
    }

    if (this.environment.rewards.has(key)) {
      return this.environment.rewards.get(key)!;
    }

    return this.environment.rewards.get("default") ?? -1;
  }

  /**
   * Take action and get next state
   */
  private takeAction(
    state: [number, number],
    action: number
  ): [number, number] {
    const [row, col] = state;
    const [dRow, dCol] = this.actions[action];
    const nextState: [number, number] = [row + dRow, col + dCol];

    if (!this.isValidState(nextState)) {
      return state;
    }

    return nextState;
  }

  /**
   * Calculate discounted returns for an episode
   */
  private calculateReturns(rewards: number[]): number[] {
    const returns: number[] = new Array(rewards.length);
    let G = 0;

    // Calculate returns backwards
    for (let t = rewards.length - 1; t >= 0; t--) {
      G = rewards[t] + this.discountFactor * G;
      returns[t] = G;
    }

    // Apply baseline (subtract mean) to reduce variance
    if (this.useBaseline && returns.length > 1) {
      const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
      for (let t = 0; t < returns.length; t++) {
        returns[t] -= mean;
      }
    }

    return returns;
  }

  /**
   * Run one episode and collect trajectory
   */
  private runEpisode(): {
    trajectory: StateAction[];
    totalReward: number;
    path: [number, number][];
  } {
    const trajectory: StateAction[] = [];
    let state = [...this.environment.startState] as [number, number];
    let totalReward = 0;
    let steps = 0;
    const path: [number, number][] = [state];

    while (steps < this.maxStepsPerEpisode && !this.isGoalState(state)) {
      // Sample action from policy
      const stateArray = this.stateToArray(state);
      const action = this.policyNetwork.sampleAction(stateArray);

      // Take action
      const nextState = this.takeAction(state, action);
      const reward = this.getReward(nextState);

      // Store state-action-reward
      trajectory.push({
        state: stateArray,
        action,
        reward,
      });

      state = nextState;
      totalReward += reward;
      steps++;
      path.push(state);
    }

    return { trajectory, totalReward, path };
  }

  /**
   * Train the policy gradient agent
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    this.episodeHistory = [];
    this.rewardHistory = [];
    this.stepsHistory = [];
    this.lossHistory = [];

    this.trainingHistory = {
      epochs: [],
      loss: [],
    };

    for (let episode = 0; episode < this.episodes; episode++) {
      // Run episode
      const { trajectory, totalReward, path } = this.runEpisode();

      // Calculate returns
      const rewards = trajectory.map((sa) => sa.reward);
      const returns = this.calculateReturns(rewards);

      // Update policy
      const states = trajectory.map((sa) => sa.state);
      const actions = trajectory.map((sa) => sa.action);
      const loss = this.policyNetwork.update(
        states,
        actions,
        returns,
        this.learningRate
      );

      // Record history
      this.episodeHistory.push({
        episode,
        totalReward,
        steps: trajectory.length,
        path,
        returns,
      });
      this.rewardHistory.push(totalReward);
      this.stepsHistory.push(trajectory.length);
      this.lossHistory.push(loss);

      this.trainingHistory.epochs.push(episode);
      this.trainingHistory.loss.push(loss);

      // Allow UI to update
      if (episode % 10 === 0) {
        await new Promise((resolve) => setTimeout(resolve, 0));
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
      const probabilities =
        this.policyNetwork.getActionProbabilities(stateArray);
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
      type: "line",
      data: {
        rewards: {
          x: Array.from({ length: this.rewardHistory.length }, (_, i) => i),
          y: this.rewardHistory,
          mode: "lines",
          name: "Total Reward",
          line: { color: "#10b981", width: 2 },
        },
      },
    });

    // Steps per episode
    visualizations.push({
      type: "line",
      data: {
        steps: {
          x: Array.from({ length: this.stepsHistory.length }, (_, i) => i),
          y: this.stepsHistory,
          mode: "lines",
          name: "Steps per Episode",
          line: { color: "#3b82f6", width: 2 },
        },
      },
    });

    // Loss curve (smoothed)
    if (this.lossHistory.length > 0) {
      const windowSize = 10;
      const smoothedLoss: number[] = [];

      for (let i = 0; i < this.lossHistory.length; i++) {
        const start = Math.max(0, i - windowSize + 1);
        const window = this.lossHistory.slice(start, i + 1);
        const avg = window.reduce((a, b) => a + b, 0) / window.length;
        smoothedLoss.push(avg);
      }

      visualizations.push({
        type: "line",
        data: {
          loss: {
            x: Array.from({ length: smoothedLoss.length }, (_, i) => i),
            y: smoothedLoss,
            mode: "lines",
            name: "Policy Loss (Smoothed)",
            line: { color: "#ef4444", width: 2 },
          },
        },
      });
    }

    // Policy visualization (action probabilities heatmap for each action)
    const { gridSize } = this.environment;
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
            const probabilities =
              this.policyNetwork.getActionProbabilities(stateArray);
            rowValues.push(probabilities[actionIdx]);
          }
        }
        actionGrid.push(rowValues);
      }

      policyData[`action_${actionIdx}`] = actionGrid;
    }

    visualizations.push({
      type: "policy",
      data: {
        policyData,
        gridSize,
        actions: ["Up", "Right", "Down", "Left"],
      },
    });

    // Environment with optimal path
    const optimalPath = this.getOptimalPath();

    visualizations.push({
      type: "scatter2d",
      data: {
        environment: {
          gridSize,
          startState: this.environment.startState,
          goalState: this.environment.goalState,
          obstacles: this.environment.obstacles,
          optimalPath,
        },
      },
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
      const probabilities =
        this.policyNetwork.getActionProbabilities(stateArray);

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
    const avgReward =
      last10Episodes.reduce((a, b) => a + b, 0) / last10Episodes.length;

    const last10Steps = this.stepsHistory.slice(-10);
    const avgSteps =
      last10Steps.reduce((a, b) => a + b, 0) / last10Steps.length;

    const optimalPath = this.getOptimalPath();
    const optimalPathLength = optimalPath.length - 1;

    const successCount = this.episodeHistory
      .slice(-10)
      .filter((ep) => this.isGoalState(ep.path[ep.path.length - 1])).length;
    const successRate = successCount / Math.min(10, this.episodeHistory.length);

    const avgLoss =
      this.lossHistory.length > 0
        ? this.lossHistory.slice(-10).reduce((a, b) => a + b, 0) /
          Math.min(10, this.lossHistory.length)
        : 0;

    return {
      averageReward: avgReward,
      averageSteps: avgSteps,
      optimalPathLength,
      successRate,
      totalEpisodes: this.episodes,
      averageLoss: avgLoss,
      useBaseline: this.useBaseline,
    };
  }

  /**
   * Get policy network for visualization
   */
  getPolicyNetwork(): PolicyNetwork {
    return this.policyNetwork;
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
      rewards: new Map(this.environment.rewards),
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
    this.lossHistory = [];
  }
}
