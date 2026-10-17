/**
 * Lasso Regression Algorithm
 * 
 * Implements Lasso regression (L1 regularization) which can set coefficients to exactly zero,
 * performing automatic feature selection while preventing overfitting.
 * 
 * Requirements: 5.1, 5.2, 5.3, 5.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics } from '~/types/algorithm';
import { mean, sum } from '~/utils/math';

export interface LassoRegressionParams {
  alpha: number;
  learningRate: number;
  epochs: number;
  normalize: boolean;
}

export class LassoRegression extends BaseAlgorithm {
  name = 'Lasso Regression';
  
  private weights: number[] = [];
  private bias: number = 0;
  private X_train: number[][] = [];
  private y_train: number[] = [];
  private featureMeans: number[] = [];
  private featureStds: number[] = [];
  private normalized: boolean = false;
  private alpha: number;
  private learningRate: number;
  private epochs: number;
  private shouldNormalize: boolean;
  
  constructor(params: LassoRegressionParams = {
    alpha: 1.0,
    learningRate: 0.01,
    epochs: 100,
    normalize: true
  }) {
    super();
    this.alpha = params.alpha;
    this.learningRate = params.learningRate;
    this.epochs = params.epochs;
    this.shouldNormalize = params.normalize;
  }
  
  private normalizeFeatures(X: number[][]): number[][] {
    const n = X[0].length;
    const m = X.length;
    
    this.featureMeans = [];
    this.featureStds = [];
    
    const X_norm: number[][] = X.map(row => [...row]);
    
    for (let j = 0; j < n; j++) {
      const column = X.map(row => row[j]);
      const colMean = mean(column);
      const colStd = Math.sqrt(sum(column.map(val => Math.pow(val - colMean, 2))) / m);
      
      this.featureMeans.push(colMean);
      this.featureStds.push(colStd === 0 ? 1 : colStd);
      
      for (let i = 0; i < m; i++) {
        X_norm[i][j] = (X[i][j] - colMean) / this.featureStds[j];
      }
    }
    
    this.normalized = true;
    return X_norm;
  }
  
  private computePredictions(X: number[][]): number[] {
    return X.map(row => {
      let prediction = this.bias;
      for (let j = 0; j < row.length; j++) {
        prediction += this.weights[j] * row[j];
      }
      return prediction;
    });
  }
  
  private computeLoss(predictions: number[], y: number[]): number {
    const m = y.length;
    let totalError = 0;
    
    for (let i = 0; i < m; i++) {
      const error = predictions[i] - y[i];
      totalError += error * error;
    }
    
    let l1Penalty = 0;
    for (let j = 0; j < this.weights.length; j++) {
      l1Penalty += Math.abs(this.weights[j]);
    }
    
    return (totalError / (2 * m)) + (this.alpha / m) * l1Penalty;
  }
  
  private softThreshold(value: number, threshold: number): number {
    if (value > threshold) return value - threshold;
    if (value < -threshold) return value + threshold;
    return 0;
  }
  
  async fit(X: number[][], y: number[]): Promise<void> {
    this.validateInput(X);
    this.validateTarget(X, y);
    
    this.X_train = X.map(row => [...row]);
    this.y_train = [...y];
    
    let X_processed = X.map(row => [...row]);
    if (this.shouldNormalize) {
      X_processed = this.normalizeFeatures(X);
    }
    
    const m = X.length;
    const n = X[0].length;
    
    this.weights = new Array(n).fill(0);
    this.bias = 0;
    
    this.trainingHistory = {
      epochs: [],
      loss: []
    };
    
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      const predictions = this.computePredictions(X_processed);
      const loss = this.computeLoss(predictions, y);
      
      this.trainingHistory.epochs.push(epoch);
      this.trainingHistory.loss.push(loss);
      
      const weightGradients = new Array(n).fill(0);
      let biasGradient = 0;
      
      for (let i = 0; i < m; i++) {
        const error = predictions[i] - y[i];
        biasGradient += error;
        
        for (let j = 0; j < n; j++) {
          weightGradients[j] += error * X_processed[i][j];
        }
      }
      
      this.bias -= (this.learningRate / m) * biasGradient;
      
      for (let j = 0; j < n; j++) {
        const gradient = weightGradients[j] / m;
        const update = this.weights[j] - this.learningRate * gradient;
        this.weights[j] = this.softThreshold(update, this.learningRate * this.alpha / m);
      }
      
      if (epoch % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    this.trained = true;
  }
  
  predict(X: number[][]): number[] {
    this.ensureTrained();
    this.validateInput(X, this.weights.length);
    
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
  
  getVisualizationData(): VisualizationData[] {
    if (!this.trained) return [];
    
    const visualizations: VisualizationData[] = [];
    
    if (this.weights.length === 1) {
      const xValues = this.X_train.map(row => row[0]);
      const xMin = Math.min(...xValues);
      const xMax = Math.max(...xValues);
      
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
            name: `Lasso (α=${this.alpha})`,
            line: { color: '#ef4444', width: 2 }
          }
        }
      });
    }
    
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
    
    return visualizations;
  }
  
  getMetrics(): Metrics {
    if (!this.trained) return {};
    
    const predictions = this.predict(this.X_train);
    const n = this.y_train.length;
    
    let mse = 0;
    for (let i = 0; i < n; i++) {
      const error = this.y_train[i] - predictions[i];
      mse += error * error;
    }
    mse /= n;
    
    const rmse = Math.sqrt(mse);
    
    const yMean = mean(this.y_train);
    let ssTotal = 0;
    let ssResidual = 0;
    
    for (let i = 0; i < n; i++) {
      ssTotal += Math.pow(this.y_train[i] - yMean, 2);
      ssResidual += Math.pow(this.y_train[i] - predictions[i], 2);
    }
    
    const r2Score = 1 - (ssResidual / ssTotal);
    
    let mae = 0;
    for (let i = 0; i < n; i++) {
      mae += Math.abs(this.y_train[i] - predictions[i]);
    }
    mae /= n;
    
    let l1Norm = 0;
    let nonZeroCount = 0;
    for (let j = 0; j < this.weights.length; j++) {
      l1Norm += Math.abs(this.weights[j]);
      if (Math.abs(this.weights[j]) > 1e-6) nonZeroCount++;
    }
    
    return {
      mse,
      rmse,
      r2Score,
      mae,
      alpha: this.alpha,
      l1Norm,
      nonZeroFeatures: nonZeroCount,
      totalFeatures: this.weights.length
    };
  }
  
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
  
  getParameters() {
    return {
      weights: [...this.weights],
      bias: this.bias,
      alpha: this.alpha,
      normalized: this.normalized,
      featureMeans: [...this.featureMeans],
      featureStds: [...this.featureStds]
    };
  }
}
