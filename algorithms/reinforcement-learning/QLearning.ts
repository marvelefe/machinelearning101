/**
 * Q-Learning Algorithm
 * 
 * Implements basic Q-learning for reinforcement learning in grid-world environments.
 * Uses epsilon-greedy exploration and Q-table updates via temporal difference learning.
 * 
 * Requirements: 4.2, 4.4, 4.5
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics } from '~/types/algorithm';

export interface QLearningParams {
  learningRate: number;
  discountFactor: number;
  epsilon: number;
  episodes: number;
  maxStepsPerEpisode: number;
  epsilonDecay?: number;
  minEpsilon?: number;
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
}

export class QLearning extends BaseAlgorithm {
  name = 'Q-Learning';
  
  // Q-table: state -> action -> value
  private qTable: Map<string, number[]> = new Map();
  
  // Environment
  private environment: GridEnvironment;
  
  // Hyperparameters
  private learningRate: number;
  private discountFactor: number;
  private epsilon: number;
  private episodes: number;
  private maxStepsPerEpisode: number;
  private epsilonDecay: number;
  private minEpsilon: number;
  
  // Training history
  private episodeHistory: Episode[] = [];
  private rewardHistory: number[] = [];
  private stepsHistory: number[] = [];
  
  // Actions: up, right, down, left
  private readonly actions = [
    [-1, 0], // up
    [0, 1],  // right
    [1, 0],  // down
    [0, -1]  // left
  ];
  
  constructor(
    environment: GridEnvironment,
    params: QLearningParams = {
      learningRate: 0.1,
      discountFactor: 0.95,
      epsilon: 0.1,
      episodes: 100,
      maxStepsPerEpisode: 100,
      epsilonDecay: 0.995,
      minEpsilon: 0.01
    }
  ) {
    super();
    this.environment = environment;
    this.learningRate = params.learningRate;
    this.discountFactor = params.discountFactor;
    this.epsilon = params.epsilon;
    this.episodes = params.episodes;
    this.maxStepsPerEpisode = params.maxStepsPerEpisode;
    this.epsilonDecay = params.epsilonDecay ?? 0.995;
    this.minEpsilon = params.minEpsilon ?? 0.01;
  }
  
  /**
   * Convert state coordinates to string key
   */
  private stateToKey(state: [number, number]): string {
    return `${state[0]},${state[1]}`;
  }
  
  /**
   * Initialize Q-values for a state
   */
  private initializeState(state: [number, number]): void {
    const key = this.stateToKey(state);
    if (!this.qTable.has(key)) {
      this.qTable.set(key, new Array(this.actions.length).fill(0));
    }
  }
  
  /**
   * Get Q-values for a state
   */
  private getQValues(state: [number, number]): number[] {
    this.initializeState(state);
    return this.qTable.get(this.stateToKey(state))!;
  }
  
  /**
   * Set Q-value for a state-action pair
   */
  private setQValue(state: [number, number], action: number, value: number): void {
    this.initializeState(state);
    const qValues = this.qTable.get(this.stateToKey(state))!;
    qValues[action] = value;
  }
  
  /**
   * Check if state is valid (within bounds and not an obstacle)
   */
  private isValidState(state: [number, number]): boolean {
    const [row, col] = state;
    const { gridSize, obstacles } = this.environment;
    
    // Check bounds
    if (row < 0 || row >= gridSize || col < 0 || col >= gridSize) {
      return false;
    }
    
    // Check obstacles
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
    const key = this.stateToKey(state);
    
    // Check if goal state
    if (this.isGoalState(state)) {
      return this.environment.rewards.get('goal') ?? 100;
    }
    
    // Check custom rewards
    if (this.environment.rewards.has(key)) {
      return this.environment.rewards.get(key)!;
    }
    
    // Default step penalty
    return this.environment.rewards.get('default') ?? -1;
  }
  
  /**
   * Take action and get next state
   */
  private takeAction(state: [number, number], action: number): [number, number] {
    const [row, col] = state;
    const [dRow, dCol] = this.actions[action];
    const nextState: [number, number] = [row + dRow, col + dCol];
    
    // If next state is invalid, stay in current state
    if (!this.isValidState(nextState)) {
      return state;
    }
    
    return nextState;
  }
  
  /**
   * Select action using epsilon-greedy policy
   */
  private selectAction(state: [number, number], currentEpsilon: number): number {
    // Exploration: random action
    if (Math.random() < currentEpsilon) {
      return Math.floor(Math.random() * this.actions.length);
    }
    
    // Exploitation: best action based on Q-values
    const qValues = this.getQValues(state);
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
   * Get best action for a state (greedy)
   */
  private getBestAction(state: [number, number]): number {
    const qValues = this.getQValues(state);
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
   * Run one episode of Q-learning
   */
  private runEpisode(currentEpsilon: number): Episode {
    let state = [...this.environment.startState] as [number, number];
    let totalReward = 0;
    let steps = 0;
    const path: [number, number][] = [state];
    
    while (steps < this.maxStepsPerEpisode && !this.isGoalState(state)) {
      // Select action
      const action = this.selectAction(state, currentEpsilon);
      
      // Take action
      const nextState = this.takeAction(state, action);
      const reward = this.getReward(nextState);
      
      // Get current Q-value
      const currentQ = this.getQValues(state)[action];
      
      // Get max Q-value for next state
      const nextQValues = this.getQValues(nextState);
      const maxNextQ = Math.max(...nextQValues);
      
      // Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
      const newQ = currentQ + this.learningRate * (
        reward + this.discountFactor * maxNextQ - currentQ
      );
      
      this.setQValue(state, action, newQ);
      
      // Update state
      state = nextState;
      totalReward += reward;
      steps++;
      path.push(state);
    }
    
    return {
      episode: this.episodeHistory.length,
      totalReward,
      steps,
      path
    };
  }
  
  /**
   * Train the Q-learning agent
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    // Reset Q-table and history
    this.qTable.clear();
    this.episodeHistory = [];
    this.rewardHistory = [];
    this.stepsHistory = [];
    
    this.trainingHistory = {
      epochs: [],
      loss: []
    };
    
    let currentEpsilon = this.epsilon;
    
    // Run episodes
    for (let episode = 0; episode < this.episodes; episode++) {
      const episodeData = this.runEpisode(currentEpsilon);
      
      this.episodeHistory.push(episodeData);
      this.rewardHistory.push(episodeData.totalReward);
      this.stepsHistory.push(episodeData.steps);
      
      this.trainingHistory.epochs.push(episode);
      this.trainingHistory.loss.push(-episodeData.totalReward); // Use negative reward as "loss"
      
      // Decay epsilon
      currentEpsilon = Math.max(this.minEpsilon, currentEpsilon * this.epsilonDecay);
      
      // Allow UI to update periodically
      if (episode % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    this.trained = true;
  }
  
  /**
   * Get optimal path from start to goal using learned policy
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    
    const path: number[] = [];
    let state = [...this.environment.startState] as [number, number];
    let steps = 0;
    
    while (steps < this.maxStepsPerEpisode && !this.isGoalState(state)) {
      const action = this.getBestAction(state);
      path.push(action);
      state = this.takeAction(state, action);
      steps++;
      
      // Prevent infinite loops
      if (steps >= this.maxStepsPerEpisode) {
        break;
      }
    }
    
    return path;
  }
  
  /**
   * Get visualization data for Q-values and environment
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained) {
      return [];
    }
    
    const visualizations: VisualizationData[] = [];
    
    // Reward curve over episodes
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
    
    // Q-value heatmap (max Q-value for each state)
    const { gridSize } = this.environment;
    const qValueGrid: number[][] = [];
    
    for (let row = 0; row < gridSize; row++) {
      const rowValues: number[] = [];
      for (let col = 0; col < gridSize; col++) {
        const state: [number, number] = [row, col];
        if (!this.isValidState(state)) {
          rowValues.push(NaN); // Obstacles
        } else {
          const qValues = this.getQValues(state);
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
    
    // Environment state with optimal path
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
   * Get optimal path from start to goal
   */
  private getOptimalPath(): [number, number][] {
    const path: [number, number][] = [];
    let state = [...this.environment.startState] as [number, number];
    let steps = 0;
    
    path.push(state);
    
    while (steps < this.maxStepsPerEpisode && !this.isGoalState(state)) {
      const action = this.getBestAction(state);
      state = this.takeAction(state, action);
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
    
    // Calculate average reward over last 10 episodes
    const last10Episodes = this.rewardHistory.slice(-10);
    const avgReward = last10Episodes.reduce((a, b) => a + b, 0) / last10Episodes.length;
    
    // Calculate average steps over last 10 episodes
    const last10Steps = this.stepsHistory.slice(-10);
    const avgSteps = last10Steps.reduce((a, b) => a + b, 0) / last10Steps.length;
    
    // Get optimal path length
    const optimalPath = this.getOptimalPath();
    const optimalPathLength = optimalPath.length - 1; // Subtract 1 for start state
    
    // Calculate success rate (reached goal in last 10 episodes)
    const successCount = this.episodeHistory.slice(-10).filter(ep => 
      this.isGoalState(ep.path[ep.path.length - 1])
    ).length;
    const successRate = successCount / Math.min(10, this.episodeHistory.length);
    
    return {
      averageReward: avgReward,
      averageSteps: avgSteps,
      optimalPathLength,
      successRate,
      totalEpisodes: this.episodes,
      qTableSize: this.qTable.size
    };
  }
  
  /**
   * Get Q-table for visualization
   */
  getQTable(): Map<string, number[]> {
    return new Map(this.qTable);
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
    this.qTable.clear();
    this.episodeHistory = [];
    this.rewardHistory = [];
    this.stepsHistory = [];
  }
}
