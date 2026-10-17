/**
 * K-Nearest Neighbors (KNN) Algorithm
 * 
 * Implements KNN classification using distance-based voting.
 * Supports multiple distance metrics (Euclidean, Manhattan).
 * 
 * Requirements: 6.1, 6.2, 6.3, 6.4
 */

import { BaseAlgorithm } from '../../types/algorithm';
import type { VisualizationData, Metrics } from '../../types/algorithm';

export interface KNNParams {
  k: number;
  distanceMetric: 'euclidean' | 'manhattan';
}

export class KNN extends BaseAlgorithm {
  name = 'K-Nearest Neighbors';
  
  // Model parameters
  private k: number;
  private distanceMetric: 'euclidean' | 'manhattan';
  
  // Training data (stored for lazy learning)
  private X_train: number[][] = [];
  private y_train: number[] = [];
  
  // Unique classes
  private classes: number[] = [];
  
  constructor(params: KNNParams = {
    k: 5,
    distanceMetric: 'euclidean'
  }) {
    super();
    this.k = params.k;
    this.distanceMetric = params.distanceMetric;
  }
  
  /**
   * Calculate Euclidean distance between two points
   */
  private euclideanDistance(point1: number[], point2: number[]): number {
    let sum = 0;
    for (let i = 0; i < point1.length; i++) {
      sum += Math.pow(point1[i] - point2[i], 2);
    }
    return Math.sqrt(sum);
  }
  
  /**
   * Calculate Manhattan distance between two points
   */
  private manhattanDistance(point1: number[], point2: number[]): number {
    let sum = 0;
    for (let i = 0; i < point1.length; i++) {
      sum += Math.abs(point1[i] - point2[i]);
    }
    return sum;
  }
  
  /**
   * Calculate distance between two points based on selected metric
   */
  private calculateDistance(point1: number[], point2: number[]): number {
    if (this.distanceMetric === 'euclidean') {
      return this.euclideanDistance(point1, point2);
    } else {
      return this.manhattanDistance(point1, point2);
    }
  }
  
  /**
   * Find K nearest neighbors for a given point
   */
  private findKNearestNeighbors(point: number[]): { indices: number[], distances: number[] } {
    // Calculate distances to all training points
    const distances = this.X_train.map((trainPoint, index) => ({
      index,
      distance: this.calculateDistance(point, trainPoint)
    }));
    
    // Sort by distance
    distances.sort((a, b) => a.distance - b.distance);
    
    // Take K nearest
    const kNearest = distances.slice(0, this.k);
    
    return {
      indices: kNearest.map(d => d.index),
      distances: kNearest.map(d => d.distance)
    };
  }
  
  /**
   * Predict class for a single point using majority voting
   */
  private predictSingle(point: number[]): number {
    const { indices } = this.findKNearestNeighbors(point);
    
    // Get labels of K nearest neighbors
    const neighborLabels = indices.map(i => this.y_train[i]);
    
    // Count votes for each class
    const votes: { [key: number]: number } = {};
    for (const label of neighborLabels) {
      votes[label] = (votes[label] || 0) + 1;
    }
    
    // Find class with most votes
    let maxVotes = 0;
    let predictedClass = neighborLabels[0];
    
    for (const [classLabel, voteCount] of Object.entries(votes)) {
      if (voteCount > maxVotes) {
        maxVotes = voteCount;
        predictedClass = parseInt(classLabel);
      }
    }
    
    return predictedClass;
  }
  
  /**
   * Train the model (store training data)
   * KNN is a lazy learner - no actual training happens
   */
  async fit(X: number[][], y: number[]): Promise<void> {
    // Validate input
    this.validateInput(X);
    this.validateTarget(X, y);
    
    // Store training data
    this.X_train = X.map(row => [...row]);
    this.y_train = [...y];
    
    // Get unique classes
    this.classes = [...new Set(y)].sort((a, b) => a - b);
    
    this.trained = true;
  }
  
  /**
   * Make predictions on new data
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    this.validateInput(X, this.X_train[0].length);
    
    return X.map(point => this.predictSingle(point));
  }
  
  /**
   * Generate decision boundary for 2D visualization
   */
  private generateDecisionBoundary(): { x: number[], y: number[], z: number[][] } | null {
    if (this.X_train[0].length !== 2) {
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
    const predGrid: number[][] = [];
    
    for (let i = 0; i <= resolution; i++) {
      x1Grid.push(x1Min + (i / resolution) * (x1Max - x1Min));
    }
    
    for (let j = 0; j <= resolution; j++) {
      x2Grid.push(x2Min + (j / resolution) * (x2Max - x2Min));
    }
    
    // Compute predictions for each grid point
    for (let j = 0; j <= resolution; j++) {
      const row: number[] = [];
      for (let i = 0; i <= resolution; i++) {
        const point = [x1Grid[i], x2Grid[j]];
        const prediction = this.predictSingle(point);
        row.push(prediction);
      }
      predGrid.push(row);
    }
    
    return { x: x1Grid, y: x2Grid, z: predGrid };
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
      
      // Create scatter data for each class
      const scatterData: any = {};
      const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'];
      const symbols = ['circle', 'x', 'diamond', 'square', 'triangle-up'];
      
      for (let classIdx = 0; classIdx < this.classes.length; classIdx++) {
        const classLabel = this.classes[classIdx];
        const classX1 = x1Values.filter((_, i) => this.y_train[i] === classLabel);
        const classX2 = x2Values.filter((_, i) => this.y_train[i] === classLabel);
        
        scatterData[`class${classIdx}`] = {
          x: classX1,
          y: classX2,
          mode: 'markers',
          name: `Class ${classLabel}`,
          marker: { 
            color: colors[classIdx % colors.length], 
            size: 8, 
            symbol: symbols[classIdx % symbols.length]
          }
        };
      }
      
      visualizations.push({
        type: 'scatter2d',
        data: scatterData
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
            colorscale: 'Viridis',
            showscale: true,
            opacity: 0.4,
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
    
    const predictions = this.predict(this.X_train);
    const n = this.y_train.length;
    
    // For binary classification
    if (this.classes.length === 2) {
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
    } else {
      // Multi-class classification
      // Build confusion matrix
      const confusionMatrix: number[][] = [];
      for (let i = 0; i < this.classes.length; i++) {
        confusionMatrix.push(new Array(this.classes.length).fill(0));
      }
      
      for (let i = 0; i < n; i++) {
        const trueIdx = this.classes.indexOf(this.y_train[i]);
        const predIdx = this.classes.indexOf(predictions[i]);
        confusionMatrix[trueIdx][predIdx]++;
      }
      
      // Calculate accuracy
      let correct = 0;
      for (let i = 0; i < n; i++) {
        if (predictions[i] === this.y_train[i]) correct++;
      }
      const accuracy = correct / n;
      
      // Calculate macro-averaged precision, recall, F1
      let totalPrecision = 0;
      let totalRecall = 0;
      
      for (let classIdx = 0; classIdx < this.classes.length; classIdx++) {
        let tp = confusionMatrix[classIdx][classIdx];
        let fp = 0;
        let fn = 0;
        
        for (let i = 0; i < this.classes.length; i++) {
          if (i !== classIdx) {
            fp += confusionMatrix[i][classIdx];
            fn += confusionMatrix[classIdx][i];
          }
        }
        
        const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
        const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
        
        totalPrecision += precision;
        totalRecall += recall;
      }
      
      const avgPrecision = totalPrecision / this.classes.length;
      const avgRecall = totalRecall / this.classes.length;
      const f1Score = avgPrecision + avgRecall > 0 ?
        2 * (avgPrecision * avgRecall) / (avgPrecision + avgRecall) : 0;
      
      return {
        accuracy,
        precision: avgPrecision,
        recall: avgRecall,
        f1Score,
        confusionMatrix
      };
    }
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.X_train = [];
    this.y_train = [];
    this.classes = [];
  }
  
  /**
   * Get model parameters
   */
  getParameters() {
    return {
      k: this.k,
      distanceMetric: this.distanceMetric,
      numTrainingSamples: this.X_train.length,
      numFeatures: this.X_train.length > 0 ? this.X_train[0].length : 0,
      classes: [...this.classes]
    };
  }
}
