/**
 * Linear Regression Algorithm
 * 
 * Implements simple linear regression using gradient descent optimization.
 * Supports single or multiple features.
 * 
 * Requirements: 2.3, 4.1, 4.2
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics, TrainingHistory } from '~/types/algorithm';
import { mean, sum } from '~/utils/math';

export interface LinearRegressionParams {
  learningRate: number;
  epochs: number;
  normalize: boolean;
}

export class LinearRegression extends BaseAlgorithm {
  name = 'Linear Regression';
  
  // Model parameters
  private weights: number[] = [];
  private bias: number = 0;
  
  // Training data (stored for visualization)
  private X_train: number[][] = [];
  private y_train: number[] = [];
  
  // Normalization parameters
  private featureMeans: number[] = [];
  private featureStds: number[] = [];
  private normalized: boolean = false;
  
  // Hyperparameters
  private learningRate: number;
  private epochs: number;
  private shouldNormalize: boolean;
  
  constructor(params: LinearRegressionParams = {
    learningRate: 0.01,
    epochs: 100,
    normalize: false
  }) {
    super();
    this.learningRate = params.learningRate;
    this.epochs = params.epochs;
    this.shouldNormalize = params.normalize;
  }
  
  /**
   * Normalize features using z-score normalization
   */
  private normalizeFeatures(X: number[][]): number[][] {
    const n = X[0].length;
    const m = X.length;
    
    this.featureMeans = [];
    this.featureStds = [];
    
    const X_norm: number[][] = X.map(row => [...row]);
    
    for (let j = 0; j < n; j++) {
      const column = X.map(row => row[j]);
      const colMean = mean(column);
      const colStd = Math.sqrt(
        sum(column.map(val => Math.pow(val - colMean, 2))) / m
      );
      
      this.featureMeans.push(colMean);
      this.featureStds.push(colStd === 0 ? 1 : colStd);
      
      for (let i = 0; i < m; i++) {
        X_norm[i][j] = (X[i][j] - colMean) / this.featureStds[j];
      }
    }
    
    this.normalized = true;
    return X_norm;
  }
  
  /**
   * Compute predictions for input X
   */
  private computePredictions(X: number[][]): number[] {
    return X.map(row => {
      let prediction = this.bias;
      for (let j = 0; j < row.length; j++) {
        prediction += this.weights[j] * row[j];
      }
      return prediction;
    });
  }
  
  /**
   * Compute mean squared error loss
   */
  private computeLoss(predictions: number[], y: number[]): number {
    const m = y.length;
    let totalError = 0;
    
    for (let i = 0; i < m; i++) {
      const error = predictions[i] - y[i];
      totalError += error * error;
    }
    
    return totalError / (2 * m);
  }
  
  /**
   * Train the model using gradient descent
   */
  async fit(X: number[][], y: number[]): Promise<void> {
    // Validate input
    this.validateInput(X);
    this.validateTarget(X, y);
    
    // Store training data
    this.X_train = X.map(row => [...row]);
    this.y_train = [...y];
    
    // Normalize features if requested
    let X_processed = X.map(row => [...row]);
    if (this.shouldNormalize) {
      X_processed = this.normalizeFeatures(X);
    }
    
    const m = X.length;
    const n = X[0].length;
    
    // Initialize weights and bias
    this.weights = new Array(n).fill(0);
    this.bias = 0;
    
    // Reset training history
    this.trainingHistory = {
      epochs: [],
      loss: []
    };
    
    // Gradient descent
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      // Compute predictions
      const predictions = this.computePredictions(X_processed);
      
      // Compute loss
      const loss = this.computeLoss(predictions, y);
      
      // Store history
      this.trainingHistory.epochs.push(epoch);
      this.trainingHistory.loss.push(loss);
      
      // Compute gradients
      const weightGradients = new Array(n).fill(0);
      let biasGradient = 0;
      
      for (let i = 0; i < m; i++) {
        const error = predictions[i] - y[i];
        biasGradient += error;
        
        for (let j = 0; j < n; j++) {
          weightGradients[j] += error * X_processed[i][j];
        }
      }
      
      // Update parameters
      this.bias -= (this.learningRate / m) * biasGradient;
      
      for (let j = 0; j < n; j++) {
        this.weights[j] -= (this.learningRate / m) * weightGradients[j];
      }
      
      // Allow UI to update periodically
      if (epoch % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    this.trained = true;
  }
  
  /**
   * Make predictions on new data
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    this.validateInput(X, this.weights.length);
    
    // Normalize if model was trained with normalization
    let X_processed = X.map(row => [...row]);
    if (this.normalized) {
      for (let i = 0; i < X.length; i++) {
        for (let j = 0; j < X[0].length; j++) {
          X_processed[i][j] = (X[i][j] - this.featureMeans[j]) / this.featureStds[j];
        }
      }
    }
    
    return this.computePredictions(X_processed);
  }
  
  /**
   * Get visualization data for plotting
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained) {
      return [];
    }
    
    const visualizations: VisualizationData[] = [];
    
    // For 1D regression, create regression line visualization
    if (this.weights.length === 1) {
      // Get data range
      const xValues = this.X_train.map(row => row[0]);
      const xMin = Math.min(...xValues);
      const xMax = Math.max(...xValues);
      
      // Generate line points
      const lineX = [];
      const lineY = [];
      const numPoints = 100;
      
      for (let i = 0; i <= numPoints; i++) {
        const x = xMin + (i / numPoints) * (xMax - xMin);
        const y = this.predict([[x]])[0];
        lineX.push(x);
        lineY.push(y);
      }
      
      visualizations.push({
        type: 'scatter2d',
        data: {
          scatter: {
            x: xValues,
            y: this.y_train,
            mode: 'markers',
            name: 'Data Points',
            marker: { color: '#3b82f6', size: 8 }
          },
          line: {
            x: lineX,
            y: lineY,
            mode: 'lines',
            name: 'Regression Line',
            line: { color: '#ef4444', width: 2 }
          }
        }
      });
      
      // Residuals visualization
      const predictions = this.predict(this.X_train);
      const residuals = this.y_train.map((y, i) => y - predictions[i]);
      
      visualizations.push({
        type: 'scatter2d',
        data: {
          residuals: {
            x: xValues,
            y: residuals,
            mode: 'markers',
            name: 'Residuals',
            marker: { color: '#8b5cf6', size: 6 }
          },
          zero: {
            x: [xMin, xMax],
            y: [0, 0],
            mode: 'lines',
            name: 'Zero Line',
            line: { color: '#6b7280', width: 1, dash: 'dash' }
          }
        }
      });
    }
    
    // Loss curve
    visualizations.push({
      type: 'line',
      data: {
        x: this.trainingHistory.epochs,
        y: this.trainingHistory.loss,
        mode: 'lines',
        name: 'Training Loss',
        line: { color: '#10b981', width: 2 }
      }
    });
    
    return visualizations;
  }
  
  /**
   * Calculate performance metrics
   */
  getMetrics(): Metrics {
    if (!this.trained) {
      return {};
    }
    
    const predictions = this.predict(this.X_train);
    const n = this.y_train.length;
    
    // Mean Squared Error (MSE)
    let mse = 0;
    for (let i = 0; i < n; i++) {
      const error = this.y_train[i] - predictions[i];
      mse += error * error;
    }
    mse /= n;
    
    // Root Mean Squared Error (RMSE)
    const rmse = Math.sqrt(mse);
    
    // R² Score (coefficient of determination)
    const yMean = mean(this.y_train);
    let ssTotal = 0;
    let ssResidual = 0;
    
    for (let i = 0; i < n; i++) {
      ssTotal += Math.pow(this.y_train[i] - yMean, 2);
      ssResidual += Math.pow(this.y_train[i] - predictions[i], 2);
    }
    
    const r2Score = 1 - (ssResidual / ssTotal);
    
    // Mean Absolute Error (MAE)
    let mae = 0;
    for (let i = 0; i < n; i++) {
      mae += Math.abs(this.y_train[i] - predictions[i]);
    }
    mae /= n;
    
    return {
      mse,
      rmse,
      r2Score,
      mae
    };
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.weights = [];
    this.bias = 0;
    this.X_train = [];
    this.y_train = [];
    this.featureMeans = [];
    this.featureStds = [];
    this.normalized = false;
  }
  
  /**
   * Get model parameters
   */
  getParameters() {
    return {
      weights: [...this.weights],
      bias: this.bias,
      normalized: this.normalized,
      featureMeans: [...this.featureMeans],
      featureStds: [...this.featureStds]
    };
  }
}
