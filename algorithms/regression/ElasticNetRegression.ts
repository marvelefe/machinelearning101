/**
 * Elastic Net Regression Algorithm
 * 
 * Implements Elastic Net regression which combines L1 (Lasso) and L2 (Ridge) regularization.
 * This provides a balance between feature selection and coefficient shrinkage.
 * 
 * Requirements: 5.1, 5.2, 5.3, 5.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics } from '~/types/algorithm';
import { mean, sum } from '~/utils/math';

export interface ElasticNetRegressionParams {
  alpha: number;
  l1Ratio: number;
  learningRate: number;
  epochs: number;
  normalize: boolean;
}

export class ElasticNetRegression extends BaseAlgorithm {
  name = 'Elastic Net Regression';
  
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
  private alpha: number; // Overall regularization strength
  private l1Ratio: number; // Balance between L1 and L2 (0 = Ridge, 1 = Lasso)
  private learningRate: number;
  private epochs: number;
  private shouldNormalize: boolean;
  
  constructor(params: ElasticNetRegressionParams = {
    alpha: 1.0,
    l1Ratio: 0.5,
    learningRate: 0.01,
    epochs: 100,
    normalize: true
  }) {
    super();
    this.alpha = params.alpha;
    this.l1Ratio = params.l1Ratio;
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
   * Compute mean squared error loss with Elastic Net regularization
   */
  private computeLoss(predictions: number[], y: number[]): number {
    const m = y.length;
    let totalError = 0;
    
    // MSE term
    for (let i = 0; i < m; i++) {
      const error = predictions[i] - y[i];
      totalError += error * error;
    }
    
    // Elastic Net regularization: alpha * (l1_ratio * L1 + (1 - l1_ratio) * L2)
    let l1Penalty = 0;
    let l2Penalty = 0;
    
    for (let j = 0; j < this.weights.length; j++) {
      l1Penalty += Math.abs(this.weights[j]);
      l2Penalty += this.weights[j] * this.weights[j];
    }
    
    const elasticNetPenalty = this.l1Ratio * l1Penalty + (1 - this.l1Ratio) * l2Penalty / 2;
    
    return (totalError / (2 * m)) + (this.alpha / m) * elasticNetPenalty;
  }
  
  /**
   * Soft thresholding function for L1 regularization component
   */
  private softThreshold(value: number, threshold: number): number {
    if (value > threshold) return value - threshold;
    if (value < -threshold) return value + threshold;
    return 0;
  }
  
  /**
   * Train the model using gradient descent with Elastic Net regularization
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
    
    // Gradient descent with Elastic Net regularization
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
      
      // Update bias (no regularization)
      this.bias -= (this.learningRate / m) * biasGradient;
      
      // Update weights with Elastic Net regularization
      for (let j = 0; j < n; j++) {
        // Standard gradient
        const gradient = weightGradients[j] / m;
        
        // L2 component (Ridge)
        const l2Component = (1 - this.l1Ratio) * this.alpha / m * this.weights[j];
        
        // Update with L2 regularization
        const updatedWeight = this.weights[j] - this.learningRate * (gradient + l2Component);
        
        // Apply L1 component (Lasso) using soft thresholding
        const l1Threshold = this.learningRate * this.l1Ratio * this.alpha / m;
        this.weights[j] = this.softThreshold(updatedWeight, l1Threshold);
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
            name: `Elastic Net (α=${this.alpha}, l1=${this.l1Ratio})`,
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
    
    // Weight visualization
    if (this.weights.length <= 20) {
      const weightIndices = this.weights.map((_, i) => `w${i + 1}`);
      const weightValues = this.weights.map(w => w);
      
      visualizations.push({
        type: 'bar',
        data: {
          x: weightIndices,
          y: weightValues,
          name: 'Coefficients',
          marker: { color: '#f59e0b' }
        }
      });
    }
    
    // Regularization comparison visualization (if multiple features)
    if (this.weights.length > 1) {
      const l1Norm = this.weights.reduce((sum, w) => sum + Math.abs(w), 0);
      const l2Norm = Math.sqrt(this.weights.reduce((sum, w) => sum + w * w, 0));
      
      visualizations.push({
        type: 'bar',
        data: {
          x: ['L1 Penalty', 'L2 Penalty', 'Combined'],
          y: [
            this.l1Ratio * l1Norm,
            (1 - this.l1Ratio) * l2Norm,
            this.l1Ratio * l1Norm + (1 - this.l1Ratio) * l2Norm
          ],
          name: 'Regularization Components',
          marker: { color: ['#ef4444', '#3b82f6', '#10b981'] }
        }
      });
    }
    
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
    
    // L1 and L2 norms
    let l1Norm = 0;
    let l2Norm = 0;
    let nonZeroCount = 0;
    
    for (let j = 0; j < this.weights.length; j++) {
      l1Norm += Math.abs(this.weights[j]);
      l2Norm += this.weights[j] * this.weights[j];
      if (Math.abs(this.weights[j]) > 1e-6) nonZeroCount++;
    }
    l2Norm = Math.sqrt(l2Norm);
    
    return {
      mse,
      rmse,
      r2Score,
      mae,
      alpha: this.alpha,
      l1Ratio: this.l1Ratio,
      l1Norm,
      l2Norm,
      nonZeroFeatures: nonZeroCount,
      totalFeatures: this.weights.length
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
      alpha: this.alpha,
      l1Ratio: this.l1Ratio,
      normalized: this.normalized,
      featureMeans: [...this.featureMeans],
      featureStds: [...this.featureStds]
    };
  }
}