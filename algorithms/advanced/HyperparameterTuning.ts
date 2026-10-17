/**
 * Hyperparameter Tuning
 * 
 * Implements various hyperparameter optimization strategies including
 * grid search, random search, and Bayesian optimization for finding
 * optimal model configurations.
 * 
 * Requirements: 10.1, 10.2, 10.3, 10.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics } from '~/types/algorithm';

export interface HyperparameterSpace {
  name: string;
  type: 'continuous' | 'discrete' | 'categorical';
  min?: number;
  max?: number;
  values?: (number | string)[];
  logScale?: boolean;
}

export interface HyperparameterConfig {
  [key: string]: number | string;
}

export interface TuningResult {
  config: HyperparameterConfig;
  score: number;
  iteration: number;
  validationScore?: number;
}

export interface TuningParams {
  searchStrategy: 'grid' | 'random' | 'bayesian';
  maxIterations: number;
  crossValidationFolds: number;
  scoringMetric: 'accuracy' | 'f1' | 'mse' | 'mae';
  earlyStoppingPatience?: number;
}

export class HyperparameterTuning extends BaseAlgorithm<number[][], number[], HyperparameterConfig> {
  name = 'Hyperparameter Tuning';
  
  // Tuning configuration
  private searchStrategy: 'grid' | 'random' | 'bayesian';
  private maxIterations: number;
  private crossValidationFolds: number;
  private scoringMetric: 'accuracy' | 'f1' | 'mse' | 'mae';
  private earlyStoppingPatience: number;
  
  // Hyperparameter space
  private hyperparameterSpace: HyperparameterSpace[] = [];
  
  // Training data
  private X_train: number[][] = [];
  private y_train: number[] = [];
  
  // Tuning results
  private tuningResults: TuningResult[] = [];
  private bestConfig: HyperparameterConfig = {};
  private bestScore: number = -Infinity;
  
  // Bayesian optimization state
  private gaussianProcessMean: number[] = [];
  private gaussianProcessVariance: number[] = [];
  private acquisitionHistory: number[] = [];
  
  constructor(params: TuningParams = {
    searchStrategy: 'random',
    maxIterations: 50,
    crossValidationFolds: 5,
    scoringMetric: 'accuracy',
    earlyStoppingPatience: 10
  }) {
    super();
    this.searchStrategy = params.searchStrategy;
    this.maxIterations = params.maxIterations;
    this.crossValidationFolds = params.crossValidationFolds;
    this.scoringMetric = params.scoringMetric;
    this.earlyStoppingPatience = params.earlyStoppingPatience || 10;
    
    // Define default hyperparameter space for a simple model
    this.initializeDefaultSpace();
  }

  /**
   * Initialize default hyperparameter space
   */
  private initializeDefaultSpace(): void {
    this.hyperparameterSpace = [
      {
        name: 'learning_rate',
        type: 'continuous',
        min: 0.001,
        max: 0.1,
        logScale: true
      },
      {
        name: 'batch_size',
        type: 'discrete',
        values: [16, 32, 64, 128]
      },
      {
        name: 'hidden_units',
        type: 'discrete',
        values: [32, 64, 128, 256]
      },
      {
        name: 'dropout_rate',
        type: 'continuous',
        min: 0.0,
        max: 0.5
      },
      {
        name: 'optimizer',
        type: 'categorical',
        values: ['adam', 'sgd', 'rmsprop']
      }
    ];
  }
  
  /**
   * Set custom hyperparameter space
   */
  setHyperparameterSpace(space: HyperparameterSpace[]): void {
    this.hyperparameterSpace = space;
  }
  
  /**
   * Generate random hyperparameter configuration
   */
  private generateRandomConfig(): HyperparameterConfig {
    const config: HyperparameterConfig = {};
    
    for (const param of this.hyperparameterSpace) {
      if (param.type === 'continuous') {
        const min = param.min!;
        const max = param.max!;
        
        if (param.logScale) {
          const logMin = Math.log(min);
          const logMax = Math.log(max);
          const logValue = Math.random() * (logMax - logMin) + logMin;
          config[param.name] = Math.exp(logValue);
        } else {
          config[param.name] = Math.random() * (max - min) + min;
        }
      } else if (param.type === 'discrete' || param.type === 'categorical') {
        const values = param.values!;
        const randomIndex = Math.floor(Math.random() * values.length);
        config[param.name] = values[randomIndex];
      }
    }
    
    return config;
  }
  
  /**
   * Generate all possible configurations for grid search
   */
  private generateGridConfigs(): HyperparameterConfig[] {
    const configs: HyperparameterConfig[] = [];
    
    // For simplicity, we'll sample from continuous spaces
    const samplesPerParam = Math.max(2, Math.floor(Math.pow(this.maxIterations, 1 / this.hyperparameterSpace.length)));
    
    const paramValues: any[][] = [];
    
    for (const param of this.hyperparameterSpace) {
      if (param.type === 'continuous') {
        const min = param.min!;
        const max = param.max!;
        const values: number[] = [];
        
        for (let i = 0; i < samplesPerParam; i++) {
          if (param.logScale) {
            const logMin = Math.log(min);
            const logMax = Math.log(max);
            const logValue = logMin + (i / (samplesPerParam - 1)) * (logMax - logMin);
            values.push(Math.exp(logValue));
          } else {
            values.push(min + (i / (samplesPerParam - 1)) * (max - min));
          }
        }
        paramValues.push(values);
      } else {
        paramValues.push(param.values!);
      }
    }
    
    // Generate cartesian product (limited to maxIterations)
    const generateCombinations = (arrays: any[][], current: any[] = []): any[][] => {
      if (arrays.length === 0) return [current];
      if (configs.length >= this.maxIterations) return [];
      
      const results: any[][] = [];
      const firstArray = arrays[0];
      const remainingArrays = arrays.slice(1);
      
      for (const value of firstArray) {
        if (configs.length >= this.maxIterations) break;
        const combinations = generateCombinations(remainingArrays, [...current, value]);
        results.push(...combinations);
      }
      
      return results;
    };
    
    const combinations = generateCombinations(paramValues);
    
    for (const combination of combinations.slice(0, this.maxIterations)) {
      const config: HyperparameterConfig = {};
      for (let i = 0; i < this.hyperparameterSpace.length; i++) {
        config[this.hyperparameterSpace[i].name] = combination[i];
      }
      configs.push(config);
    }
    
    return configs;
  }
  
  /**
   * Evaluate a hyperparameter configuration using cross-validation
   */
  private async evaluateConfig(config: HyperparameterConfig): Promise<number> {
    const foldSize = Math.floor(this.X_train.length / this.crossValidationFolds);
    const scores: number[] = [];
    
    for (let fold = 0; fold < this.crossValidationFolds; fold++) {
      // Create train/validation split
      const valStart = fold * foldSize;
      const valEnd = Math.min((fold + 1) * foldSize, this.X_train.length);
      
      const X_val = this.X_train.slice(valStart, valEnd);
      const y_val = this.y_train.slice(valStart, valEnd);
      
      const X_fold_train = [
        ...this.X_train.slice(0, valStart),
        ...this.X_train.slice(valEnd)
      ];
      const y_fold_train = [
        ...this.y_train.slice(0, valStart),
        ...this.y_train.slice(valEnd)
      ];
      
      // Train simple model with current config
      const model = this.createSimpleModel(config);
      await this.trainModel(model, X_fold_train, y_fold_train);
      
      // Evaluate on validation set
      const predictions = this.predictWithModel(model, X_val);
      const score = this.computeScore(y_val, predictions);
      scores.push(score);
    }
    
    // Return average score
    return scores.reduce((a, b) => a + b, 0) / scores.length;
  }
  
  /**
   * Create a simple model with given hyperparameters
   */
  private createSimpleModel(config: HyperparameterConfig): any {
    return {
      learningRate: config.learning_rate as number || 0.01,
      hiddenUnits: config.hidden_units as number || 64,
      dropoutRate: config.dropout_rate as number || 0.1,
      optimizer: config.optimizer as string || 'adam',
      weights: this.initializeWeights(config.hidden_units as number || 64),
      bias: 0
    };
  }
  
  /**
   * Initialize model weights
   */
  private initializeWeights(hiddenUnits: number): number[][] {
    const inputDim = this.X_train[0]?.length || 2;
    const weights: number[][] = [];
    
    // Input to hidden
    const inputToHidden: number[] = [];
    for (let i = 0; i < hiddenUnits; i++) {
      inputToHidden.push((Math.random() - 0.5) * 2 / Math.sqrt(inputDim));
    }
    weights.push(inputToHidden);
    
    // Hidden to output
    const hiddenToOutput: number[] = [];
    for (let i = 0; i < hiddenUnits; i++) {
      hiddenToOutput.push((Math.random() - 0.5) * 2 / Math.sqrt(hiddenUnits));
    }
    weights.push(hiddenToOutput);
    
    return weights;
  }
  
  /**
   * Train simple model (simplified for demonstration)
   */
  private async trainModel(model: any, X: number[][], y: number[]): Promise<void> {
    const epochs = 10;
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let i = 0; i < X.length; i++) {
        // Forward pass (simplified)
        const input = X[i];
        const target = y[i];
        
        // Hidden layer
        let hiddenSum = 0;
        for (let j = 0; j < input.length; j++) {
          hiddenSum += input[j] * model.weights[0][j % model.weights[0].length];
        }
        const hidden = Math.max(0, hiddenSum); // ReLU
        
        // Output
        const output = hidden * model.weights[1][0] + model.bias;
        
        // Loss and gradient (simplified)
        const error = output - target;
        const gradient = error * model.learningRate;
        
        // Update weights (simplified)
        model.bias -= gradient;
        for (let j = 0; j < model.weights[1].length; j++) {
          model.weights[1][j] -= gradient * hidden;
        }
      }
      
      // Allow UI updates
      if (epoch % 5 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
  }
  
  /**
   * Make predictions with model
   */
  private predictWithModel(model: any, X: number[][]): number[] {
    const predictions: number[] = [];
    
    for (const input of X) {
      // Forward pass
      let hiddenSum = 0;
      for (let j = 0; j < input.length; j++) {
        hiddenSum += input[j] * model.weights[0][j % model.weights[0].length];
      }
      const hidden = Math.max(0, hiddenSum);
      const output = hidden * model.weights[1][0] + model.bias;
      
      predictions.push(output);
    }
    
    return predictions;
  }
  
  /**
   * Compute score based on metric
   */
  private computeScore(yTrue: number[], yPred: number[]): number {
    switch (this.scoringMetric) {
      case 'mse':
        let mse = 0;
        for (let i = 0; i < yTrue.length; i++) {
          const diff = yTrue[i] - yPred[i];
          mse += diff * diff;
        }
        return -mse / yTrue.length; // Negative because we want to maximize
        
      case 'mae':
        let mae = 0;
        for (let i = 0; i < yTrue.length; i++) {
          mae += Math.abs(yTrue[i] - yPred[i]);
        }
        return -mae / yTrue.length; // Negative because we want to maximize
        
      case 'accuracy':
        let correct = 0;
        for (let i = 0; i < yTrue.length; i++) {
          const predicted = yPred[i] > 0.5 ? 1 : 0;
          if (predicted === yTrue[i]) correct++;
        }
        return correct / yTrue.length;
        
      case 'f1':
        // Simplified F1 calculation
        let tp = 0, fp = 0, fn = 0;
        for (let i = 0; i < yTrue.length; i++) {
          const predicted = yPred[i] > 0.5 ? 1 : 0;
          const actual = yTrue[i];
          
          if (predicted === 1 && actual === 1) tp++;
          else if (predicted === 1 && actual === 0) fp++;
          else if (predicted === 0 && actual === 1) fn++;
        }
        
        const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
        const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
        return precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
        
      default:
        return 0;
    }
  }
  
  /**
   * Bayesian optimization using Gaussian Process (simplified)
   */
  private selectNextConfigBayesian(): HyperparameterConfig {
    if (this.tuningResults.length < 3) {
      // Not enough data for GP, use random
      return this.generateRandomConfig();
    }
    
    // Simplified acquisition function (Upper Confidence Bound)
    let bestAcquisition = -Infinity;
    let bestConfig: HyperparameterConfig = {};
    
    // Sample candidate configurations
    for (let i = 0; i < 20; i++) {
      const candidate = this.generateRandomConfig();
      const acquisition = this.computeAcquisition(candidate);
      
      if (acquisition > bestAcquisition) {
        bestAcquisition = acquisition;
        bestConfig = candidate;
      }
    }
    
    this.acquisitionHistory.push(bestAcquisition);
    return bestConfig;
  }
  
  /**
   * Compute acquisition function value (simplified UCB)
   */
  private computeAcquisition(config: HyperparameterConfig): number {
    // Simplified: use distance to previous configurations
    let minDistance = Infinity;
    
    for (const result of this.tuningResults) {
      const distance = this.configDistance(config, result.config);
      minDistance = Math.min(minDistance, distance);
    }
    
    // UCB = mean + beta * std
    const beta = 2.0;
    const mean = this.bestScore * 0.8; // Simplified mean estimation
    const std = minDistance * 0.1; // Simplified std estimation
    
    return mean + beta * std;
  }
  
  /**
   * Compute distance between configurations
   */
  private configDistance(config1: HyperparameterConfig, config2: HyperparameterConfig): number {
    let distance = 0;
    
    for (const param of this.hyperparameterSpace) {
      const val1 = config1[param.name];
      const val2 = config2[param.name];
      
      if (param.type === 'continuous') {
        const diff = (val1 as number) - (val2 as number);
        distance += diff * diff;
      } else {
        distance += val1 === val2 ? 0 : 1;
      }
    }
    
    return Math.sqrt(distance);
  }

  /**
   * Run hyperparameter tuning
   */
  async fit(X: number[][], y: number[]): Promise<void> {
    // Validate input
    this.validateInput(X);
    this.validateTarget(X, y);
    
    // Store training data
    this.X_train = X.map(row => [...row]);
    this.y_train = [...y];
    
    // Initialize results
    this.tuningResults = [];
    this.bestScore = this.scoringMetric === 'mse' || this.scoringMetric === 'mae' ? Infinity : -Infinity;
    this.bestConfig = {};
    
    // Initialize training history
    this.trainingHistory = {
      epochs: [],
      loss: []
    };
    
    // Generate configurations based on strategy
    let configs: HyperparameterConfig[] = [];
    
    if (this.searchStrategy === 'grid') {
      configs = this.generateGridConfigs();
    } else if (this.searchStrategy === 'random') {
      for (let i = 0; i < this.maxIterations; i++) {
        configs.push(this.generateRandomConfig());
      }
    }
    
    // Evaluate configurations
    let patienceCounter = 0;
    
    for (let i = 0; i < this.maxIterations; i++) {
      let config: HyperparameterConfig;
      
      if (this.searchStrategy === 'bayesian') {
        config = this.selectNextConfigBayesian();
      } else {
        if (i >= configs.length) break;
        config = configs[i];
      }
      
      // Evaluate configuration
      const score = await this.evaluateConfig(config);
      
      // Store result
      const result: TuningResult = {
        config,
        score,
        iteration: i,
        validationScore: score
      };
      
      this.tuningResults.push(result);
      
      // Update best configuration
      const isBetter = this.scoringMetric === 'mse' || this.scoringMetric === 'mae' 
        ? score > this.bestScore  // Higher is better (less negative)
        : score > this.bestScore;
      
      if (isBetter) {
        this.bestScore = score;
        this.bestConfig = { ...config };
        patienceCounter = 0;
      } else {
        patienceCounter++;
      }
      
      // Record history
      this.trainingHistory.epochs.push(i);
      this.trainingHistory.loss.push(-score); // Convert to loss for visualization
      
      // Early stopping
      if (patienceCounter >= this.earlyStoppingPatience) {
        break;
      }
      
      // Allow UI updates
      if (i % 5 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    this.trained = true;
  }
  
  /**
   * Get best hyperparameter configuration
   */
  predict(): HyperparameterConfig {
    this.ensureTrained();
    return { ...this.bestConfig };
  }
  
  /**
   * Get best hyperparameters (alias for predict)
   */
  getBestParameters(): HyperparameterConfig {
    return this.predict();
  }
  
  /**
   * Get all tuning results
   */
  getTuningResults(): TuningResult[] {
    return this.tuningResults.map(result => ({
      config: { ...result.config },
      score: result.score,
      iteration: result.iteration,
      validationScore: result.validationScore
    }));
  }

  /**
   * Get visualization data
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained) {
      return [];
    }
    
    const visualizations: VisualizationData[] = [];
    
    // Optimization progress
    const scores = this.tuningResults.map(r => r.score);
    const iterations = this.tuningResults.map(r => r.iteration);
    
    // Best score over time
    const bestScores: number[] = [];
    let currentBest = this.scoringMetric === 'mse' || this.scoringMetric === 'mae' ? Infinity : -Infinity;
    
    for (const result of this.tuningResults) {
      const isBetter = this.scoringMetric === 'mse' || this.scoringMetric === 'mae' 
        ? result.score > currentBest
        : result.score > currentBest;
      
      if (isBetter) {
        currentBest = result.score;
      }
      bestScores.push(currentBest);
    }
    
    // Combined line chart data with both progress and bestScore
    visualizations.push({
      type: 'line',
      data: {
        progress: {
          x: iterations,
          y: scores,
          mode: 'lines+markers',
          name: 'Validation Score',
          line: { color: '#3b82f6', width: 2 }
        },
        bestScore: {
          x: iterations,
          y: bestScores,
          mode: 'lines',
          name: 'Best Score',
          line: { color: '#10b981', width: 3 }
        }
      }
    });
    
    // Hyperparameter importance (simplified)
    const paramImportance = this.computeParameterImportance();
    
    if (paramImportance.length > 0) {
      const importanceMap: Record<string, number> = {};
      for (const param of paramImportance) {
        importanceMap[param.name] = param.importance;
      }
      
      visualizations.push({
        type: 'bar',
        data: {
          importance: importanceMap
        }
      });
    }
    
    // Acquisition function (for Bayesian optimization)
    if (this.searchStrategy === 'bayesian' && this.acquisitionHistory.length > 0) {
      visualizations.push({
        type: 'line',
        data: {
          acquisition: {
            x: Array.from({length: this.acquisitionHistory.length}, (_, i) => i),
            y: this.acquisitionHistory,
            mode: 'lines+markers',
            name: 'Acquisition Function',
            line: { color: '#ef4444', width: 2 }
          }
        }
      });
    }
    
    return visualizations;
  }
  
  /**
   * Compute parameter importance (simplified)
   */
  private computeParameterImportance(): { name: string; importance: number }[] {
    const importance: { name: string; importance: number }[] = [];
    
    for (const param of this.hyperparameterSpace) {
      let totalVariance = 0;
      const values: number[] = [];
      
      // Collect values for this parameter
      for (const result of this.tuningResults) {
        const value = result.config[param.name];
        if (typeof value === 'number') {
          values.push(value);
        }
      }
      
      if (values.length > 1) {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        totalVariance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
      }
      
      importance.push({
        name: param.name,
        importance: totalVariance
      });
    }
    
    // Normalize importance scores
    const maxImportance = Math.max(...importance.map(p => p.importance));
    if (maxImportance > 0) {
      for (const param of importance) {
        param.importance /= maxImportance;
      }
    }
    
    return importance.sort((a, b) => b.importance - a.importance);
  }
  
  /**
   * Get metrics
   */
  getMetrics(): Metrics {
    if (!this.trained) {
      return {};
    }
    
    const totalIterations = this.tuningResults.length;
    const bestIteration = this.tuningResults.findIndex(r => r.score === this.bestScore);
    
    // Compute convergence metrics
    const improvementIterations = this.tuningResults.filter((result, i) => {
      if (i === 0) return true;
      const prevBest = Math.max(...this.tuningResults.slice(0, i).map(r => r.score));
      return result.score > prevBest;
    }).length;
    
    return {
      bestScore: this.bestScore,
      totalIterations,
      bestIteration,
      improvementIterations,
      convergenceRate: improvementIterations / totalIterations,
      searchStrategy: this.searchStrategy,
      scoringMetric: this.scoringMetric,
      hyperparameterSpaceSize: this.hyperparameterSpace.length,
      bestConfig: JSON.stringify(this.bestConfig)
    };
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.X_train = [];
    this.y_train = [];
    this.tuningResults = [];
    this.bestConfig = {};
    this.bestScore = -Infinity;
    this.gaussianProcessMean = [];
    this.gaussianProcessVariance = [];
    this.acquisitionHistory = [];
  }
  
  /**
   * Get model parameters
   */
  getParameters() {
    return {
      searchStrategy: this.searchStrategy,
      maxIterations: this.maxIterations,
      crossValidationFolds: this.crossValidationFolds,
      scoringMetric: this.scoringMetric,
      earlyStoppingPatience: this.earlyStoppingPatience,
      hyperparameterSpace: this.hyperparameterSpace.map(param => ({ ...param }))
    };
  }
}