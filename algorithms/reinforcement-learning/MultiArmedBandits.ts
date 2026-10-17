/**
 * Multi-Armed Bandits Algorithm
 * 
 * Implements various bandit algorithms including epsilon-greedy, UCB (Upper Confidence Bound),
 * and Thompson Sampling for the multi-armed bandit problem.
 * 
 * Requirements: 4.2, 4.4, 4.5
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics } from '~/types/algorithm';

export type BanditStrategy = 'epsilon-greedy' | 'ucb' | 'thompson-sampling';

export interface MultiArmedBanditsParams {
  numArms: number;
  numTrials: number;
  strategy: BanditStrategy;
  epsilon?: number; // For epsilon-greedy
  c?: number; // For UCB exploration parameter
  trueRewards?: number[]; // True mean rewards for each arm (for simulation)
}

export interface Trial {
  trial: number;
  armPulled: number;
  reward: number;
  cumulativeReward: number;
  regret: number;
  cumulativeRegret: number;
}

export interface ArmStats {
  armIndex: number;
  pulls: number;
  totalReward: number;
  meanReward: number;
  // For Thompson Sampling
  alpha: number; // Success count + 1
  beta: number;  // Failure count + 1
}

export class MultiArmedBandits extends BaseAlgorithm {
  name = 'Multi-Armed Bandits';
  
  // Configuration
  private numArms: number;
  private numTrials: number;
  private strategy: BanditStrategy;
  private epsilon: number;
  private c: number; // UCB exploration parameter
  private trueRewards: number[];
  
  // Arm statistics
  private armStats: ArmStats[];
  
  // Trial history
  private trialHistory: Trial[] = [];
  private rewardHistory: number[] = [];
  private cumulativeRewardHistory: number[] = [];
  private regretHistory: number[] = [];
  private cumulativeRegretHistory: number[] = [];
  private armPullCounts: number[] = [];
  
  // Optimal arm (for regret calculation)
  private optimalArm: number;
  private optimalReward: number;
  
  constructor(params: MultiArmedBanditsParams) {
    super();
    
    this.numArms = params.numArms;
    this.numTrials = params.numTrials;
    this.strategy = params.strategy;
    this.epsilon = params.epsilon ?? 0.1;
    this.c = params.c ?? 2.0;
    
    // Initialize true rewards (if not provided, generate random)
    if (params.trueRewards && params.trueRewards.length === this.numArms) {
      this.trueRewards = [...params.trueRewards];
    } else {
      // Generate random true rewards between 0 and 1
      this.trueRewards = Array.from({ length: this.numArms }, () => Math.random());
    }
    
    // Find optimal arm
    this.optimalReward = Math.max(...this.trueRewards);
    this.optimalArm = this.trueRewards.indexOf(this.optimalReward);
    
    // Initialize arm statistics
    this.armStats = Array.from({ length: this.numArms }, (_, i) => ({
      armIndex: i,
      pulls: 0,
      totalReward: 0,
      meanReward: 0,
      alpha: 1, // Prior for Thompson Sampling
      beta: 1   // Prior for Thompson Sampling
    }));
    
    this.armPullCounts = new Array(this.numArms).fill(0);
  }
  
  /**
   * Sample reward from an arm (Bernoulli distribution)
   */
  private sampleReward(armIndex: number): number {
    // Bernoulli reward: 1 with probability trueRewards[armIndex], 0 otherwise
    return Math.random() < this.trueRewards[armIndex] ? 1 : 0;
  }
  
  /**
   * Epsilon-greedy arm selection
   */
  private selectArmEpsilonGreedy(): number {
    // Exploration: random arm
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.numArms);
    }
    
    // Exploitation: arm with highest mean reward
    let bestArm = 0;
    let bestMean = -Infinity;
    
    for (let i = 0; i < this.numArms; i++) {
      const stats = this.armStats[i];
      // If arm hasn't been pulled, prioritize it
      if (stats.pulls === 0) {
        return i;
      }
      if (stats.meanReward > bestMean) {
        bestMean = stats.meanReward;
        bestArm = i;
      }
    }
    
    return bestArm;
  }
  
  /**
   * UCB (Upper Confidence Bound) arm selection
   */
  private selectArmUCB(totalPulls: number): number {
    let bestArm = 0;
    let bestUCB = -Infinity;
    
    for (let i = 0; i < this.numArms; i++) {
      const stats = this.armStats[i];
      
      // If arm hasn't been pulled, prioritize it
      if (stats.pulls === 0) {
        return i;
      }
      
      // UCB formula: mean + c * sqrt(ln(total_pulls) / arm_pulls)
      const exploration = this.c * Math.sqrt(Math.log(totalPulls) / stats.pulls);
      const ucb = stats.meanReward + exploration;
      
      if (ucb > bestUCB) {
        bestUCB = ucb;
        bestArm = i;
      }
    }
    
    return bestArm;
  }
  
  /**
   * Sample from Beta distribution (for Thompson Sampling)
   * Using simple approximation
   */
  private sampleBeta(alpha: number, beta: number): number {
    // Simple approximation using gamma distribution
    // For better accuracy, could use more sophisticated methods
    
    // Generate two gamma samples
    const gammaAlpha = this.sampleGamma(alpha);
    const gammaBeta = this.sampleGamma(beta);
    
    // Beta sample = gamma_alpha / (gamma_alpha + gamma_beta)
    return gammaAlpha / (gammaAlpha + gammaBeta);
  }
  
  /**
   * Sample from Gamma distribution (simple approximation)
   */
  private sampleGamma(shape: number): number {
    // For shape >= 1, use Marsaglia and Tsang method
    if (shape >= 1) {
      const d = shape - 1/3;
      const c = 1 / Math.sqrt(9 * d);
      
      while (true) {
        let x, v;
        do {
          x = this.sampleNormal(0, 1);
          v = 1 + c * x;
        } while (v <= 0);
        
        v = v * v * v;
        const u = Math.random();
        
        if (u < 1 - 0.0331 * x * x * x * x) {
          return d * v;
        }
        
        if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
          return d * v;
        }
      }
    } else {
      // For shape < 1, use shape + 1 and scale
      return this.sampleGamma(shape + 1) * Math.pow(Math.random(), 1 / shape);
    }
  }
  
  /**
   * Sample from Normal distribution (Box-Muller transform)
   */
  private sampleNormal(mean: number, std: number): number {
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + std * z;
  }
  
  /**
   * Thompson Sampling arm selection
   */
  private selectArmThompsonSampling(): number {
    let bestArm = 0;
    let bestSample = -Infinity;
    
    for (let i = 0; i < this.numArms; i++) {
      const stats = this.armStats[i];
      
      // Sample from Beta(alpha, beta) distribution
      const sample = this.sampleBeta(stats.alpha, stats.beta);
      
      if (sample > bestSample) {
        bestSample = sample;
        bestArm = i;
      }
    }
    
    return bestArm;
  }
  
  /**
   * Select arm based on strategy
   */
  private selectArm(totalPulls: number): number {
    switch (this.strategy) {
      case 'epsilon-greedy':
        return this.selectArmEpsilonGreedy();
      case 'ucb':
        return this.selectArmUCB(totalPulls);
      case 'thompson-sampling':
        return this.selectArmThompsonSampling();
      default:
        throw new Error(`Unknown strategy: ${this.strategy}`);
    }
  }
  
  /**
   * Update arm statistics after pulling
   */
  private updateArmStats(armIndex: number, reward: number): void {
    const stats = this.armStats[armIndex];
    
    stats.pulls++;
    stats.totalReward += reward;
    stats.meanReward = stats.totalReward / stats.pulls;
    
    // Update Thompson Sampling parameters
    if (reward === 1) {
      stats.alpha++;
    } else {
      stats.beta++;
    }
    
    this.armPullCounts[armIndex]++;
  }
  
  /**
   * Run the bandit algorithm
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    // Reset state
    this.armStats = Array.from({ length: this.numArms }, (_, i) => ({
      armIndex: i,
      pulls: 0,
      totalReward: 0,
      meanReward: 0,
      alpha: 1,
      beta: 1
    }));
    
    this.trialHistory = [];
    this.rewardHistory = [];
    this.cumulativeRewardHistory = [];
    this.regretHistory = [];
    this.cumulativeRegretHistory = [];
    this.armPullCounts = new Array(this.numArms).fill(0);
    
    this.trainingHistory = {
      epochs: [],
      loss: []
    };
    
    let cumulativeReward = 0;
    let cumulativeRegret = 0;
    
    // Run trials
    for (let trial = 0; trial < this.numTrials; trial++) {
      // Select arm
      const armPulled = this.selectArm(trial);
      
      // Pull arm and get reward
      const reward = this.sampleReward(armPulled);
      
      // Calculate regret (difference from optimal)
      const regret = this.optimalReward - this.trueRewards[armPulled];
      
      // Update cumulative values
      cumulativeReward += reward;
      cumulativeRegret += regret;
      
      // Update arm statistics
      this.updateArmStats(armPulled, reward);
      
      // Record trial
      const trialData: Trial = {
        trial,
        armPulled,
        reward,
        cumulativeReward,
        regret,
        cumulativeRegret
      };
      
      this.trialHistory.push(trialData);
      this.rewardHistory.push(reward);
      this.cumulativeRewardHistory.push(cumulativeReward);
      this.regretHistory.push(regret);
      this.cumulativeRegretHistory.push(cumulativeRegret);
      
      this.trainingHistory.epochs.push(trial);
      this.trainingHistory.loss.push(cumulativeRegret); // Use cumulative regret as "loss"
      
      // Allow UI to update periodically
      if (trial % 50 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    this.trained = true;
  }
  
  /**
   * Predict best arm based on learned statistics
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    
    // Return the arm with highest mean reward
    let bestArm = 0;
    let bestMean = -Infinity;
    
    for (let i = 0; i < this.numArms; i++) {
      if (this.armStats[i].meanReward > bestMean) {
        bestMean = this.armStats[i].meanReward;
        bestArm = i;
      }
    }
    
    return [bestArm];
  }
  
  /**
   * Get visualization data for reward curves
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained) {
      return [];
    }
    
    const visualizations: VisualizationData[] = [];
    
    // Cumulative reward over trials
    visualizations.push({
      type: 'line',
      data: {
        cumulativeReward: {
          x: Array.from({ length: this.cumulativeRewardHistory.length }, (_, i) => i),
          y: this.cumulativeRewardHistory,
          mode: 'lines',
          name: 'Cumulative Reward',
          line: { color: '#10b981', width: 2 }
        }
      }
    });
    
    // Cumulative regret over trials
    visualizations.push({
      type: 'line',
      data: {
        cumulativeRegret: {
          x: Array.from({ length: this.cumulativeRegretHistory.length }, (_, i) => i),
          y: this.cumulativeRegretHistory,
          mode: 'lines',
          name: 'Cumulative Regret',
          line: { color: '#ef4444', width: 2 }
        }
      }
    });
    
    // Arm pull distribution (bar chart)
    visualizations.push({
      type: 'line',
      data: {
        armPulls: {
          x: Array.from({ length: this.numArms }, (_, i) => `Arm ${i}`),
          y: this.armPullCounts,
          type: 'bar',
          name: 'Arm Pulls',
          marker: { color: '#3b82f6' }
        }
      }
    });
    
    // Mean reward estimates vs true rewards
    const meanRewards = this.armStats.map(s => s.meanReward);
    visualizations.push({
      type: 'line',
      data: {
        trueRewards: {
          x: Array.from({ length: this.numArms }, (_, i) => `Arm ${i}`),
          y: this.trueRewards,
          type: 'bar',
          name: 'True Rewards',
          marker: { color: '#10b981' }
        },
        estimatedRewards: {
          x: Array.from({ length: this.numArms }, (_, i) => `Arm ${i}`),
          y: meanRewards,
          type: 'bar',
          name: 'Estimated Rewards',
          marker: { color: '#3b82f6' }
        }
      }
    });
    
    // Moving average reward (window of 100 trials)
    const windowSize = Math.min(100, Math.floor(this.numTrials / 10));
    const movingAvgReward: number[] = [];
    
    for (let i = 0; i < this.rewardHistory.length; i++) {
      const start = Math.max(0, i - windowSize + 1);
      const window = this.rewardHistory.slice(start, i + 1);
      const avg = window.reduce((a, b) => a + b, 0) / window.length;
      movingAvgReward.push(avg);
    }
    
    visualizations.push({
      type: 'line',
      data: {
        movingAvg: {
          x: Array.from({ length: movingAvgReward.length }, (_, i) => i),
          y: movingAvgReward,
          mode: 'lines',
          name: `Moving Avg Reward (window=${windowSize})`,
          line: { color: '#8b5cf6', width: 2 }
        }
      }
    });
    
    return visualizations;
  }
  
  /**
   * Get performance metrics
   */
  getMetrics(): Metrics {
    if (!this.trained) {
      return {};
    }
    
    // Total reward
    const totalReward = this.cumulativeRewardHistory[this.cumulativeRewardHistory.length - 1];
    
    // Total regret
    const totalRegret = this.cumulativeRegretHistory[this.cumulativeRegretHistory.length - 1];
    
    // Average reward per trial
    const avgReward = totalReward / this.numTrials;
    
    // Average regret per trial
    const avgRegret = totalRegret / this.numTrials;
    
    // Best arm found
    let bestArm = 0;
    let bestMean = -Infinity;
    for (let i = 0; i < this.numArms; i++) {
      if (this.armStats[i].meanReward > bestMean) {
        bestMean = this.armStats[i].meanReward;
        bestArm = i;
      }
    }
    
    // Accuracy: did we find the optimal arm?
    const foundOptimal = bestArm === this.optimalArm;
    
    // Percentage of optimal arm pulls
    const optimalPulls = this.armPullCounts[this.optimalArm];
    const optimalPullRate = optimalPulls / this.numTrials;
    
    return {
      totalReward,
      totalRegret,
      averageReward: avgReward,
      averageRegret: avgRegret,
      bestArmFound: bestArm,
      optimalArm: this.optimalArm,
      foundOptimal,
      optimalPullRate,
      strategy: this.strategy,
      numTrials: this.numTrials,
      numArms: this.numArms
    };
  }
  
  /**
   * Get arm statistics
   */
  getArmStats(): ArmStats[] {
    return this.armStats.map(s => ({ ...s }));
  }
  
  /**
   * Get trial history
   */
  getTrialHistory(): Trial[] {
    return [...this.trialHistory];
  }
  
  /**
   * Get true rewards (for visualization)
   */
  getTrueRewards(): number[] {
    return [...this.trueRewards];
  }
  
  /**
   * Reset the algorithm
   */
  override reset(): void {
    super.reset();
    this.armStats = Array.from({ length: this.numArms }, (_, i) => ({
      armIndex: i,
      pulls: 0,
      totalReward: 0,
      meanReward: 0,
      alpha: 1,
      beta: 1
    }));
    this.trialHistory = [];
    this.rewardHistory = [];
    this.cumulativeRewardHistory = [];
    this.regretHistory = [];
    this.cumulativeRegretHistory = [];
    this.armPullCounts = new Array(this.numArms).fill(0);
  }
}
