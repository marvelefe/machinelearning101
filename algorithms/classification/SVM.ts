/**
 * Support Vector Machine (SVM) Algorithm
 * 
 * Implements binary SVM classification using Sequential Minimal Optimization (SMO).
 * Supports linear, RBF, and polynomial kernels for non-linear classification.
 * 
 * Requirements: 6.1, 6.2, 6.3, 6.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics } from '~/types/algorithm';
import { mean, sum } from '~/utils/math';

export interface SVMParams {
  C: number;
  kernel: 'linear' | 'rbf' | 'poly';
  gamma: 'scale' | 'auto' | number;
  degree?: number;
  maxIter?: number;
  tol?: number;
}

export class SVM extends BaseAlgorithm {
  name = 'Support Vector Machine';
  
  // Model parameters
  private alphas: number[] = [];
  private b: number = 0;
  private supportVectorIndices: number[] = [];
  private supportVectors: number[][] = [];
  private supportVectorLabels: number[] = [];
  
  // Training data (stored for kernel computations and visualization)
  private X_train: number[][] = [];
  private y_train: number[] = [];
  
  // Normalization parameters
  private featureMeans: number[] = [];
  private featureStds: number[] = [];
  private normalized: boolean = false;
  
  // Hyperparameters
  private C: number;
  private kernel: 'linear' | 'rbf' | 'poly';
  private gamma: number;
  private degree: number;
  private maxIter: number;
  private tol: number;
  
  constructor(params: SVMParams = {
    C: 1.0,
    kernel: 'rbf',
    gamma: 'scale',
    degree: 3,
    maxIter: 1000,
    tol: 1e-3
  }) {
    super();
    this.C = params.C;
    this.kernel = params.kernel;
    this.degree = params.degree || 3;
    this.maxIter = params.maxIter || 1000;
    this.tol = params.tol || 1e-3;
    
    // Gamma will be set during fit based on data
    this.gamma = typeof params.gamma === 'number' ? params.gamma : 0;
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
      const variance = sum(column.map(val => Math.pow(val - colMean, 2))) / m;
      const colStd = Math.sqrt(variance);
      
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
   * Kernel function
   */
  private kernelFunction(x1: number[], x2: number[]): number {
    switch (this.kernel) {
      case 'linear':
        return this.linearKernel(x1, x2);
      case 'rbf':
        return this.rbfKernel(x1, x2);
      case 'poly':
        return this.polyKernel(x1, x2);
      default:
        return this.linearKernel(x1, x2);
    }
  }
  
  /**
   * Linear kernel: K(x1, x2) = x1 · x2
   */
  private linearKernel(x1: number[], x2: number[]): number {
    let result = 0;
    for (let i = 0; i < x1.length; i++) {
      result += x1[i] * x2[i];
    }
    return result;
  }
  
  /**
   * RBF kernel: K(x1, x2) = exp(-gamma * ||x1 - x2||^2)
   */
  private rbfKernel(x1: number[], x2: number[]): number {
    let squaredDistance = 0;
    for (let i = 0; i < x1.length; i++) {
      const diff = x1[i] - x2[i];
      squaredDistance += diff * diff;
    }
    return Math.exp(-this.gamma * squaredDistance);
  }
  
  /**
   * Polynomial kernel: K(x1, x2) = (x1 · x2 + 1)^degree
   */
  private polyKernel(x1: number[], x2: number[]): number {
    let dotProduct = 0;
    for (let i = 0; i < x1.length; i++) {
      dotProduct += x1[i] * x2[i];
    }
    return Math.pow(dotProduct + 1, this.degree);
  }
  
  /**
   * Compute decision function value for a single point
   */
  private decisionFunction(x: number[]): number {
    let result = this.b;
    for (let i = 0; i < this.supportVectors.length; i++) {
      result += this.alphas[this.supportVectorIndices[i]] * 
                this.supportVectorLabels[i] * 
                this.kernelFunction(this.supportVectors[i], x);
    }
    return result;
  }
  
  /**
   * Simplified SMO algorithm for training
   */
  async fit(X: number[][], y: number[]): Promise<void> {
    // Validate input
    this.validateInput(X);
    this.validateTarget(X, y);
    
    // Validate binary labels and convert to {-1, +1}
    const uniqueLabels = [...new Set(y)];
    if (uniqueLabels.length !== 2) {
      throw new Error('SVM requires exactly 2 classes');
    }
    
    // Convert labels to {-1, +1}
    const labelMap = new Map<number, number>();
    labelMap.set(uniqueLabels[0], -1);
    labelMap.set(uniqueLabels[1], 1);
    
    const y_converted = y.map(label => labelMap.get(label)!);
    
    // Normalize features
    const X_processed = this.normalizeFeatures(X);
    
    // Store training data
    this.X_train = X_processed.map(row => [...row]);
    this.y_train = y_converted;
    
    const m = X.length;
    const n = X[0].length;
    
    // Set gamma based on parameter
    if (this.gamma === 0) {
      // Calculate variance for 'scale' or use 1/n_features for 'auto'
      const allValues = X_processed.flat();
      const variance = sum(allValues.map(v => v * v)) / allValues.length;
      this.gamma = 1 / (n * Math.max(variance, 1e-10));
    }
    
    // Initialize alphas
    this.alphas = new Array(m).fill(0);
    this.b = 0;
    
    // Precompute kernel matrix for efficiency (for small datasets)
    const kernelMatrix: number[][] = [];
    for (let i = 0; i < m; i++) {
      kernelMatrix[i] = [];
      for (let j = 0; j < m; j++) {
        kernelMatrix[i][j] = this.kernelFunction(X_processed[i], X_processed[j]);
      }
    }
    
    // Simplified SMO algorithm
    let numChanged = 0;
    let examineAll = true;
    let iter = 0;
    
    while ((numChanged > 0 || examineAll) && iter < this.maxIter) {
      numChanged = 0;
      
      if (examineAll) {
        // Loop over all training examples
        for (let i = 0; i < m; i++) {
          numChanged += this.examineExample(i, kernelMatrix);
        }
      } else {
        // Loop over examples where alpha is not 0 or C (non-bound examples)
        for (let i = 0; i < m; i++) {
          if (this.alphas[i] > 0 && this.alphas[i] < this.C) {
            numChanged += this.examineExample(i, kernelMatrix);
          }
        }
      }
      
      if (examineAll) {
        examineAll = false;
      } else if (numChanged === 0) {
        examineAll = true;
      }
      
      iter++;
      
      // Allow UI to update periodically
      if (iter % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    // Extract support vectors
    this.supportVectorIndices = [];
    this.supportVectors = [];
    this.supportVectorLabels = [];
    
    for (let i = 0; i < m; i++) {
      if (this.alphas[i] > 1e-5) {
        this.supportVectorIndices.push(i);
        this.supportVectors.push(this.X_train[i]);
        this.supportVectorLabels.push(this.y_train[i]);
      }
    }
    
    this.trained = true;
  }
  
  /**
   * Examine a single example in SMO algorithm
   */
  private examineExample(i2: number, kernelMatrix: number[][]): number {
    const y2 = this.y_train[i2];
    const alpha2 = this.alphas[i2];
    const E2 = this.getError(i2, kernelMatrix);
    const r2 = E2 * y2;
    
    if ((r2 < -this.tol && alpha2 < this.C) || (r2 > this.tol && alpha2 > 0)) {
      // Try to find a second alpha to optimize
      const nonBoundIndices = this.alphas
        .map((alpha, idx) => ({ alpha, idx }))
        .filter(({ alpha }) => alpha > 0 && alpha < this.C)
        .map(({ idx }) => idx);
      
      if (nonBoundIndices.length > 1) {
        // Choose the alpha with maximum |E1 - E2|
        let maxDiff = 0;
        let i1 = -1;
        
        for (const idx of nonBoundIndices) {
          if (idx === i2) continue;
          const E1 = this.getError(idx, kernelMatrix);
          const diff = Math.abs(E1 - E2);
          if (diff > maxDiff) {
            maxDiff = diff;
            i1 = idx;
          }
        }
        
        if (i1 >= 0 && this.takeStep(i1, i2, kernelMatrix)) {
          return 1;
        }
      }
      
      // Try all non-bound examples
      for (const i1 of nonBoundIndices) {
        if (i1 === i2) continue;
        if (this.takeStep(i1, i2, kernelMatrix)) {
          return 1;
        }
      }
      
      // Try all examples
      for (let i1 = 0; i1 < this.alphas.length; i1++) {
        if (i1 === i2) continue;
        if (this.takeStep(i1, i2, kernelMatrix)) {
          return 1;
        }
      }
    }
    
    return 0;
  }
  
  /**
   * Get prediction error for example i
   */
  private getError(i: number, kernelMatrix: number[][]): number {
    let prediction = this.b;
    for (let j = 0; j < this.alphas.length; j++) {
      prediction += this.alphas[j] * this.y_train[j] * kernelMatrix[j][i];
    }
    return prediction - this.y_train[i];
  }
  
  /**
   * Take a step in SMO algorithm
   */
  private takeStep(i1: number, i2: number, kernelMatrix: number[][]): boolean {
    if (i1 === i2) return false;
    
    const alpha1 = this.alphas[i1];
    const alpha2 = this.alphas[i2];
    const y1 = this.y_train[i1];
    const y2 = this.y_train[i2];
    const E1 = this.getError(i1, kernelMatrix);
    const E2 = this.getError(i2, kernelMatrix);
    const s = y1 * y2;
    
    // Compute bounds
    let L: number, H: number;
    if (y1 !== y2) {
      L = Math.max(0, alpha2 - alpha1);
      H = Math.min(this.C, this.C + alpha2 - alpha1);
    } else {
      L = Math.max(0, alpha1 + alpha2 - this.C);
      H = Math.min(this.C, alpha1 + alpha2);
    }
    
    if (L === H) return false;
    
    // Compute eta
    const k11 = kernelMatrix[i1][i1];
    const k12 = kernelMatrix[i1][i2];
    const k22 = kernelMatrix[i2][i2];
    const eta = 2 * k12 - k11 - k22;
    
    let a2: number;
    if (eta < 0) {
      a2 = alpha2 - y2 * (E1 - E2) / eta;
      if (a2 < L) a2 = L;
      else if (a2 > H) a2 = H;
    } else {
      // Unusual case, use objective function
      a2 = alpha2;
    }
    
    if (Math.abs(a2 - alpha2) < this.tol * (a2 + alpha2 + this.tol)) {
      return false;
    }
    
    const a1 = alpha1 + s * (alpha2 - a2);
    
    // Update threshold b
    const b1 = E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + this.b;
    const b2 = E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + this.b;
    
    if (a1 > 0 && a1 < this.C) {
      this.b = b1;
    } else if (a2 > 0 && a2 < this.C) {
      this.b = b2;
    } else {
      this.b = (b1 + b2) / 2;
    }
    
    // Update alphas
    this.alphas[i1] = a1;
    this.alphas[i2] = a2;
    
    return true;
  }
  
  /**
   * Make predictions on new data
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    this.validateInput(X, this.X_train[0].length);
    
    // Normalize if model was trained with normalization
    let X_processed = X.map(row => [...row]);
    if (this.normalized) {
      for (let i = 0; i < X.length; i++) {
        for (let j = 0; j < X[0].length; j++) {
          X_processed[i][j] = (X[i][j] - this.featureMeans[j]) / this.featureStds[j];
        }
      }
    }
    
    return X_processed.map(x => {
      const decision = this.decisionFunction(x);
      return decision >= 0 ? 1 : 0;
    });
  }
  
  /**
   * Get decision function values (for confidence/margin)
   */
  decisionFunctionValues(X: number[][]): number[] {
    this.ensureTrained();
    this.validateInput(X, this.X_train[0].length);
    
    // Normalize if model was trained with normalization
    let X_processed = X.map(row => [...row]);
    if (this.normalized) {
      for (let i = 0; i < X.length; i++) {
        for (let j = 0; j < X[0].length; j++) {
          X_processed[i][j] = (X[i][j] - this.featureMeans[j]) / this.featureStds[j];
        }
      }
    }
    
    return X_processed.map(x => this.decisionFunction(x));
  }
  
  /**
   * Generate decision boundary for 2D visualization
   */
  private generateDecisionBoundary(): { x: number[], y: number[], z: number[][] } | null {
    if (this.X_train[0].length !== 2) {
      return null; // Only for 2D features
    }
    
    // Get data range from original (unnormalized) training data
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
    const decisionGrid: number[][] = [];
    
    for (let i = 0; i <= resolution; i++) {
      x1Grid.push(x1Min + (i / resolution) * (x1Max - x1Min));
    }
    
    for (let j = 0; j <= resolution; j++) {
      x2Grid.push(x2Min + (j / resolution) * (x2Max - x2Min));
    }
    
    // Compute decision function for each grid point
    for (let j = 0; j <= resolution; j++) {
      const row: number[] = [];
      for (let i = 0; i <= resolution; i++) {
        const point = [x1Grid[i], x2Grid[j]];
        const decision = this.decisionFunction(point);
        row.push(decision);
      }
      decisionGrid.push(row);
    }
    
    return { x: x1Grid, y: x2Grid, z: decisionGrid };
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
    if (this.X_train[0].length === 2) {
      const x1Values = this.X_train.map(row => row[0]);
      const x2Values = this.X_train.map(row => row[1]);
      
      // Separate by class
      const class0X1 = x1Values.filter((_, i) => this.y_train[i] === -1);
      const class0X2 = x2Values.filter((_, i) => this.y_train[i] === -1);
      const class1X1 = x1Values.filter((_, i) => this.y_train[i] === 1);
      const class1X2 = x2Values.filter((_, i) => this.y_train[i] === 1);
      
      // Support vectors
      const svX1 = this.supportVectors.map(sv => sv[0]);
      const svX2 = this.supportVectors.map(sv => sv[1]);
      
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
          },
          supportVectors: {
            x: svX1,
            y: svX2,
            mode: 'markers',
            name: 'Support Vectors',
            marker: { 
              color: 'rgba(0, 0, 0, 0)',
              size: 14,
              symbol: 'circle-open',
              line: { color: '#000000', width: 2 }
            }
          }
        }
      });
      
      // Add decision boundary heatmap
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
            opacity: 0.5,
            name: 'Decision Boundary'
          }
        });
      }
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
    
    const predictions = this.predict(this.X_train.map((row, i) => {
      // Denormalize for prediction
      return row.map((val, j) => val * this.featureStds[j] + this.featureMeans[j]);
    }));
    
    const n = this.y_train.length;
    
    // Convert y_train back to {0, 1} for metrics
    const y_binary = this.y_train.map(y => y === 1 ? 1 : 0);
    
    // Confusion matrix
    let tp = 0, tn = 0, fp = 0, fn = 0;
    
    for (let i = 0; i < n; i++) {
      if (y_binary[i] === 1 && predictions[i] === 1) tp++;
      else if (y_binary[i] === 0 && predictions[i] === 0) tn++;
      else if (y_binary[i] === 0 && predictions[i] === 1) fp++;
      else if (y_binary[i] === 1 && predictions[i] === 0) fn++;
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
      confusionMatrix,
      numSupportVectors: this.supportVectors.length
    };
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.alphas = [];
    this.b = 0;
    this.supportVectorIndices = [];
    this.supportVectors = [];
    this.supportVectorLabels = [];
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
      alphas: [...this.alphas],
      b: this.b,
      supportVectors: this.supportVectors.map(sv => [...sv]),
      supportVectorLabels: [...this.supportVectorLabels],
      numSupportVectors: this.supportVectors.length,
      kernel: this.kernel,
      C: this.C,
      gamma: this.gamma
    };
  }
}
