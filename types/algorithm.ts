/**
 * Core algorithm types and interfaces for ML learning platform
 */

import type { VisualizationType } from './visualization';

/**
 * Re-export DataPoint from dataset types for convenience
 */
export type { DataPoint } from './dataset';

/**
 * Visualization data structure for rendering charts and plots
 */
export interface VisualizationData {
  type: VisualizationType;
  title?: string;
  data: any;
  layout?: any;
  config?: any;
}

/**
 * Performance metrics for model evaluation
 */
export interface Metrics {
  // Regression metrics
  mse?: number;
  rmse?: number;
  r2Score?: number;
  mae?: number;
  
  // Classification metrics
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1Score?: number;
  confusionMatrix?: number[][];
  rocCurve?: { fpr: number[]; tpr: number[]; auc: number };
  
  // Clustering metrics
  inertia?: number;
  silhouetteScore?: number;
  
  // Custom metrics
  [key: string]: any;
}

/**
 * Training history tracking
 */
export interface TrainingHistory {
  epochs: number[];
  loss: number[];
  valLoss?: number[];
  accuracy?: number[];
  valAccuracy?: number[];
  [key: string]: any;
}

/**
 * Algorithm results structure
 */
export interface AlgorithmResults {
  predictions?: number[] | string[];
  model?: ModelParameters;
  metrics?: Metrics;
  visualizationData?: VisualizationData | VisualizationData[];
  trainingHistory?: TrainingHistory;
  executionTime?: number;
}

/**
 * Model parameters (algorithm-specific)
 */
export interface ModelParameters {
  weights?: number[];
  bias?: number;
  centers?: number[][];
  tree?: any;
  [key: string]: any;
}

/**
 * Base interface that all ML algorithms must implement
 */
export interface Algorithm<
  TX = number[][],
  TY = number[],
  TResult = number[] | string[]
> {
  /**
   * Algorithm name
   */
  name: string;
  
  /**
   * Train the algorithm on the provided data
   * @param X - Feature matrix (n_samples x n_features)
   * @param y - Target values (optional for unsupervised learning)
   * @returns Promise that resolves when training is complete
   */
  fit(X: TX, y?: TY): Promise<void>;
  
  /**
   * Make predictions on new data
   * @param X - Feature matrix (n_samples x n_features)
   * @returns Predicted values
   */
  predict(X: TX): TResult;
  
  /**
   * Get visualization data for rendering
   * @returns Visualization data structure
   */
  getVisualizationData(): VisualizationData | VisualizationData[];
  
  /**
   * Get performance metrics
   * @returns Metrics object
   */
  getMetrics(): Metrics;
  
  /**
   * Reset the algorithm to initial state
   */
  reset(): void;
}

/**
 * Abstract base class providing common functionality for algorithms
 */
export abstract class BaseAlgorithm<
  TX = number[][],
  TY = number[],
  TResult = number[] | string[]
> implements Algorithm<TX, TY, TResult> {
  abstract name: string;
  
  trained: boolean = false;
  protected trainingHistory: TrainingHistory = {
    epochs: [],
    loss: []
  };
  
  /**
   * Check if the algorithm has been trained
   */
  protected ensureTrained(): void {
    if (!this.trained) {
      throw new Error(`${this.name} must be trained before making predictions`);
    }
  }
  
  /**
   * Validate input data dimensions
   */
  protected validateInput(X: number[][], expectedFeatures?: number): void {
    if (!X || X.length === 0) {
      throw new Error('Input data cannot be empty');
    }
    
    const features = X[0]?.length || 0;
    if (features === 0) {
      throw new Error('Input data must have at least one feature');
    }
    
    // Check all rows have same number of features
    for (let i = 0; i < X.length; i++) {
      const row = X[i];
      if (!row || row.length !== features) {
        throw new Error(`Inconsistent number of features at row ${i}`);
      }
    }
    
    if (expectedFeatures !== undefined && features !== expectedFeatures) {
      throw new Error(`Expected ${expectedFeatures} features, got ${features}`);
    }
    
    // Check for invalid numbers in input
    for (let i = 0; i < X.length; i++) {
      const row = X[i];
      if (!row) continue;
      for (let j = 0; j < row.length; j++) {
        const val = row[j];
        if (val === undefined || !isFinite(val) || isNaN(val)) {
          throw new Error(`Invalid value at row ${i}, column ${j}: ${val}`);
        }
      }
    }
  }
  
  /**
   * Validate target values match input size
   */
  protected validateTarget(X: number[][], y: number[]): void {
    if (y.length !== X.length) {
      throw new Error(`Target size (${y.length}) must match input size (${X.length})`);
    }
    
    // Check for invalid numbers in target
    for (let i = 0; i < y.length; i++) {
      const val = y[i];
      if (val === undefined || !isFinite(val) || isNaN(val)) {
        throw new Error(`Invalid target value at index ${i}: ${val}`);
      }
    }
  }
  
  /**
   * Check for numerical issues in values
   */
  protected checkNumericalStability(values: number[], context: string): void {
    for (let i = 0; i < values.length; i++) {
      const val = values[i];
      if (val === undefined) {
        throw new Error(`Undefined value in ${context} at index ${i}`);
      }
      if (isNaN(val)) {
        throw new Error(`NaN detected in ${context} at index ${i}`);
      }
      if (!isFinite(val)) {
        throw new Error(`Infinity detected in ${context} at index ${i}`);
      }
      if (Math.abs(val) > 1e100) {
        throw new Error(`Numerical overflow in ${context} at index ${i}`);
      }
    }
  }
  
  abstract fit(X: TX, y?: TY): Promise<void>;
  abstract predict(X: TX): TResult;
  abstract getVisualizationData(): VisualizationData | VisualizationData[];
  abstract getMetrics(): Metrics;
  
  /**
   * Reset algorithm to initial state
   */
  reset(): void {
    this.trained = false;
    this.trainingHistory = {
      epochs: [],
      loss: []
    };
  }
}
