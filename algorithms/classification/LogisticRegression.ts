/**
 * Logistic Regression Algorithm
 * 
 * Implements binary logistic regression using gradient descent optimization.
 * Uses sigmoid activation for binary classification.
 * 
 * Requirements: 2.3, 3.1, 4.1
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics, TrainingHistory } from '~/types/algorithm';
import { mean, sum } from '~/utils/math';

export interface LogisticRegressionParams {
  learningRate: number;
  epochs: number;
  regularization?: number;
  normalize: boolean;
}

export class LogisticRegression extends BaseAlgorithm {
  name = 'Logistic Regression';
  
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
  private regularization: number;
  private shouldNormalize: boolean;
  
  // Training history with accuracy
  protected override trainingHistory: TrainingHistory & { accuracy: number[] } = {
    epochs: [],
    loss: [],
    accuracy: []
  };
  
  constructor(params: LogisticRegressionParams = {
    learningRate: 0.01,
    epochs: 100,
    regularization: 0,
    normalize: false
  }) {
    super();
    this.learningRate = params.learningRate;
    this.epochs = params.epochs;
    this.regularization = params.regularization || 0;
    this.shouldNormalize = params.normalize;
  }
  
  /**
   * Sigmoid activation function
   * Maps any real value to (0, 1)
   */
  private sigmoid(z: number): number {
    // Clip to prevent overflow
    if (z > 20) return 1.0;
    if (z < -20) return 0.0;
    return 1 / (1 + Math.exp(-z));
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
   * Compute probability predictions for input X
   */
  private computeProbabilities(X: number[][]): number[] {
    return X.map(row => {
      let z = this.bias;
      for (let j = 0; j < row.length; j++) {
        z += this.weights[j] * row[j];
      }
      return this.sigmoid(z);
    });
  }
  
  /**
   * Compute binary cross-entropy loss
   */
  private computeLoss(probabilities: number[], y: number[]): number {
    const m = y.length;
    let totalLoss = 0;
    
    for (let i = 0; i < m; i++) {
      // Clip probabilities to prevent log(0)
      const p = Math.max(1e-15, Math.min(1 - 1e-15, probabilities[i]));
      totalLoss += -y[i] * Math.log(p) - (1 - y[i]) * Math.log(1 - p);
    }
    
    let loss = totalLoss / m;
    
    // Add L2 regularization term if specified
    if (this.regularization > 0) {
      const regTerm = (this.regularization / (2 * m)) * 
        sum(this.weights.map(w => w * w));
      loss += regTerm;
    }
    
    return loss;
  }
  
  /**
   * Compute accuracy
   */
  private computeAccuracy(probabilities: number[], y: number[]): number {
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      const prediction = probabilities[i] >= 0.5 ? 1 : 0;
      if (prediction === y[i]) {
        correct++;
      }
    }
    return correct / y.length;
  }
  
  /**
   * Train the model using gradient descent
   */
  async fit(X: number[][], y: number[]): Promise<void> {
    // Validate input
    this.validateInput(X);
    this.validateTarget(X, y);
    
    // Validate binary labels
    const uniqueLabels = [...new Set(y)];
    if (uniqueLabels.length !== 2) {
      throw new Error('Logistic Regression requires exactly 2 classes');
    }
    if (!uniqueLabels.includes(0) || !uniqueLabels.includes(1)) {
      throw new Error('Labels must be 0 and 1 for binary classification');
    }
    
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
      loss: [],
      accuracy: []
    };
    
    // Gradient descent
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      // Compute probabilities
      const probabilities = this.computeProbabilities(X_processed);
      
      // Compute loss
      const loss = this.computeLoss(probabilities, y);
      
      // Compute accuracy
      const accuracy = this.computeAccuracy(probabilities, y);
      
      // Store history
      this.trainingHistory.epochs.push(epoch);
      this.trainingHistory.loss.push(loss);
      this.trainingHistory.accuracy.push(accuracy);
      
      // Compute gradients
      const weightGradients = new Array(n).fill(0);
      let biasGradient = 0;
      
      for (let i = 0; i < m; i++) {
        const error = probabilities[i] - y[i];
        biasGradient += error;
        
        for (let j = 0; j < n; j++) {
          weightGradients[j] += error * X_processed[i][j];
        }
      }
      
      // Update parameters with regularization
      this.bias -= (this.learningRate / m) * biasGradient;
      
      for (let j = 0; j < n; j++) {
        const regTerm = this.regularization > 0 ? 
          (this.regularization / m) * this.weights[j] : 0;
        this.weights[j] -= (this.learningRate / m) * (weightGradients[j] + regTerm);
      }
      
      // Allow UI to update periodically
      if (epoch % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    this.trained = true;
  }
  
  /**
   * Predict probabilities for new data
   */
  predictProba(X: number[][]): number[] {
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
    
    return this.computeProbabilities(X_processed);
  }
  
  /**
   * Make binary predictions on new data (0 or 1)
   */
  predict(X: number[][]): number[] {
    const probabilities = this.predictProba(X);
    return probabilities.map(p => p >= 0.5 ? 1 : 0);
  }
  
  /**
   * Generate decision boundary points for 2D visualization
   */
  private generateDecisionBoundary(): { x: number[], y: number[], z: number[][] } | null {
    if (this.weights.length !== 2) {
      return null; // Only for 2D features
    }
    
    // Get data range
    const x1Values = this.X_train.map(row => row[0]);
    const x2Values = this.X_train.map(row => row[1]);
    
    const x1Min = Math.min(...x1Values) - 1;
    const x1Max = Math.max(...x1Values) + 1;
    const x2Min = Math.min(...x2Values) - 1;
    const x2Max = Math.max(...x2Values) + 1;
    
    // Create grid
    const resolution = 50;
    const x1Grid: number[] = [];
    const x2Grid: number[] = [];
    const probGrid: number[][] = [];
    
    for (let i = 0; i <= resolution; i++) {
      x1Grid.push(x1Min + (i / resolution) * (x1Max - x1Min));
    }
    
    for (let j = 0; j <= resolution; j++) {
      x2Grid.push(x2Min + (j / resolution) * (x2Max - x2Min));
    }
    
    // Compute probabilities for each grid point
    for (let j = 0; j <= resolution; j++) {
      const row: number[] = [];
      for (let i = 0; i <= resolution; i++) {
        const point = [[x1Grid[i], x2Grid[j]]];
        const prob = this.predictProba(point)[0];
        row.push(prob);
      }
      probGrid.push(row);
    }
    
    return { x: x1Grid, y: x2Grid, z: probGrid };
  }
  
  /**
   * Get visualization data for plotting
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained) {
      return [];
    }
    
    const visualizations: VisualizationData[] = [];
    
    // For 2D classification, create scatter plot with decision boundary
    if (this.weights.length === 2) {
      const x1Values = this.X_train.map(row => row[0]);
      const x2Values = this.X_train.map(row => row[1]);
      
      // Separate by class
      const class0X1 = x1Values.filter((_, i) => this.y_train[i] === 0);
      const class0X2 = x2Values.filter((_, i) => this.y_train[i] === 0);
      const class1X1 = x1Values.filter((_, i) => this.y_train[i] === 1);
      const class1X2 = x2Values.filter((_, i) => this.y_train[i] === 1);
      
      visualizations.push({
        type: 'scatter2d',
        data: {
          class0: {
            x: class0X1,
            y: class0X2,
            mode: 'markers',
            name: 'Class 0',
            marker: { color: '#3b82f6', size: 8, symbol: 'circle' }
          },
          class1: {
            x: class1X1,
            y: class1X2,
            mode: 'markers',
            name: 'Class 1',
            marker: { color: '#ef4444', size: 8, symbol: 'x' }
          }
        }
      });
      
      // Add probability heatmap (decision boundary)
      const boundaryData = this.generateDecisionBoundary();
      if (boundaryData) {
        visualizations.push({
          type: 'heatmap',
          data: {
            x: boundaryData.x,
            y: boundaryData.y,
            z: boundaryData.z,
            colorscale: 'RdBu',
            reversescale: true,
            showscale: true,
            opacity: 0.6,
            name: 'Decision Boundary'
          }
        });
      }
    }
    
    // Loss curve
    if (this.trainingHistory.epochs.length > 0) {
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
    }
    
    // Accuracy curve
    if (this.trainingHistory.epochs.length > 0) {
      visualizations.push({
        type: 'line',
        data: {
          x: this.trainingHistory.epochs,
          y: this.trainingHistory.accuracy,
          mode: 'lines',
          name: 'Training Accuracy',
          line: { color: '#8b5cf6', width: 2 }
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
    const probabilities = this.predictProba(this.X_train);
    const n = this.y_train.length;
    
    // Confusion matrix
    let tp = 0, tn = 0, fp = 0, fn = 0;
    
    for (let i = 0; i < n; i++) {
      if (this.y_train[i] === 1 && predictions[i] === 1) tp++;
      else if (this.y_train[i] === 0 && predictions[i] === 0) tn++;
      else if (this.y_train[i] === 0 && predictions[i] === 1) fp++;
      else if (this.y_train[i] === 1 && predictions[i] === 0) fn++;
    }
    
    const confusionMatrix = [
      [tn, fp],
      [fn, tp]
    ];
    
    // Accuracy
    const accuracy = (tp + tn) / n;
    
    // Precision
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    
    // Recall
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    
    // F1 Score
    const f1Score = precision + recall > 0 ? 
      2 * (precision * recall) / (precision + recall) : 0;
    
    return {
      accuracy,
      precision,
      recall,
      f1Score,
      confusionMatrix
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
    this.trainingHistory = {
      epochs: [],
      loss: [],
      accuracy: []
    };
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
