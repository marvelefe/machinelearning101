/**
 * Outlier Detection & Handling Algorithm
 * Implements various outlier detection and handling techniques
 * Requirements: 8.1, 8.2, 8.3, 8.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics, TrainingHistory } from '~/types/algorithm'; 


export type OutlierMethod = 'z-score' | 'iqr' | 'isolation-forest' | 'local-outlier-factor' | 'dbscan' | 'none';
export type OutlierHandling = 'remove' | 'cap' | 'transform' | 'keep';

export interface OutlierDetectionResult {
  cleanedData: number[][];
  originalData: number[][];
  outlierIndices: number[];
  outlierScores: number[];
  method: OutlierMethod;
  handling: OutlierHandling;
  statistics: {
    totalSamples: number;
    outliersDetected: number;
    outlierPercentage: number;
    threshold: number;
  };
}

export class OutlierDetection extends BaseAlgorithm {
  name = 'Outlier Detection';
  
  private data: number[][] = [];
  private cleanedData: number[][] = [];
  private outlierIndices: number[] = [];
  private outlierScores: number[] = [];
  private method: OutlierMethod = 'none';
  private handling: OutlierHandling = 'keep';
  private threshold: number = 3.0;

  constructor() {
    super();
  }

  /**
   * Set the data for outlier detection
   */
  setData(data: number[][], threshold: number = 3.0, handling: OutlierHandling = 'keep'): void {
    this.data = data;
    this.threshold = threshold;
    this.handling = handling;
    this.cleanedData = [];
    this.outlierIndices = [];
    this.outlierScores = [];
    this.trained = false;
  }

  /**
   * Fit the outlier detector
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    this.setData(X, this.threshold, this.handling);
    
    // Apply default Z-score method
    this.detectByZScore();
    this.trained = true;
  }

  /**
   * Detect outliers using Z-score method
   */
  detectByZScore(threshold: number = 3.0): OutlierDetectionResult {
    if (this.data.length === 0) {
      throw new Error('No data for outlier detection');
    }

    this.method = 'z-score';
    this.threshold = threshold;
    const n = this.data.length;
    const d = this.data[0].length;

    // Calculate mean and standard deviation for each feature
    const means: number[] = [];
    const stds: number[] = [];

    for (let j = 0; j < d; j++) {
      const column = this.data.map(row => row[j]);
      const mean = column.reduce((sum, val) => sum + val, 0) / n;
      const variance = column.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
      const std = Math.sqrt(variance);
      
      means.push(mean);
      stds.push(std);
    }

    // Calculate Z-scores and identify outliers
    this.outlierScores = [];
    this.outlierIndices = [];

    for (let i = 0; i < n; i++) {
      let maxZScore = 0;
      
      for (let j = 0; j < d; j++) {
        const zScore = stds[j] > 0 ? Math.abs((this.data[i][j] - means[j]) / stds[j]) : 0;
        maxZScore = Math.max(maxZScore, zScore);
      }
      
      this.outlierScores.push(maxZScore);
      
      if (maxZScore > threshold) {
        this.outlierIndices.push(i);
      }
    }

    this.applyHandling();
    return this.getResult();
  }

  /**
   * Detect outliers using Interquartile Range (IQR) method
   */
  detectByIQR(multiplier: number = 1.5): OutlierDetectionResult {
    if (this.data.length === 0) {
      throw new Error('No data for outlier detection');
    }

    this.method = 'iqr';
    this.threshold = multiplier;
    const n = this.data.length;
    const d = this.data[0].length;

    // Calculate IQR for each feature
    const q1Values: number[] = [];
    const q3Values: number[] = [];
    const iqrValues: number[] = [];

    for (let j = 0; j < d; j++) {
      const column = this.data.map(row => row[j]).sort((a, b) => a - b);
      const q1Index = Math.floor(column.length * 0.25);
      const q3Index = Math.floor(column.length * 0.75);
      
      const q1 = column[q1Index];
      const q3 = column[q3Index];
      const iqr = q3 - q1;
      
      q1Values.push(q1);
      q3Values.push(q3);
      iqrValues.push(iqr);
    }

    // Identify outliers
    this.outlierScores = [];
    this.outlierIndices = [];

    for (let i = 0; i < n; i++) {
      let isOutlier = false;
      let maxOutlierScore = 0;
      
      for (let j = 0; j < d; j++) {
        const value = this.data[i][j];
        const lowerBound = q1Values[j] - multiplier * iqrValues[j];
        const upperBound = q3Values[j] + multiplier * iqrValues[j];
        
        if (value < lowerBound || value > upperBound) {
          isOutlier = true;
          // Calculate how far outside the bounds
          const lowerDistance = Math.max(0, lowerBound - value);
          const upperDistance = Math.max(0, value - upperBound);
          const outlierDistance = Math.max(lowerDistance, upperDistance);
          const normalizedScore = iqrValues[j] > 0 ? outlierDistance / iqrValues[j] : 0;
          maxOutlierScore = Math.max(maxOutlierScore, normalizedScore);
        }
      }
      
      this.outlierScores.push(maxOutlierScore);
      
      if (isOutlier) {
        this.outlierIndices.push(i);
      }
    }

    this.applyHandling();
    return this.getResult();
  }

  /**
   * Detect outliers using simplified Isolation Forest
   */
  detectByIsolationForest(contamination: number = 0.1): OutlierDetectionResult {
    if (this.data.length === 0) {
      throw new Error('No data for outlier detection');
    }

    this.method = 'isolation-forest';
    this.threshold = contamination;
    const n = this.data.length;
    const d = this.data[0].length;

    // Simplified isolation forest: use random cuts
    const numTrees = 10;
    const maxDepth = Math.ceil(Math.log2(n));
    const isolationScores: number[] = new Array(n).fill(0);

    for (let tree = 0; tree < numTrees; tree++) {
      const pathLengths = this.buildIsolationTree(this.data, maxDepth);
      
      for (let i = 0; i < n; i++) {
        isolationScores[i] += pathLengths[i];
      }
    }

    // Average path lengths
    for (let i = 0; i < n; i++) {
      isolationScores[i] /= numTrees;
    }

    // Convert to anomaly scores (shorter paths = more anomalous)
    const avgPathLength = isolationScores.reduce((sum, val) => sum + val, 0) / n;
    this.outlierScores = isolationScores.map(pathLength => 
      Math.pow(2, -pathLength / avgPathLength)
    );

    // Select top contamination% as outliers
    const sortedIndices = this.outlierScores
      .map((score, index) => ({ score, index }))
      .sort((a, b) => b.score - a.score);
    
    const numOutliers = Math.floor(n * contamination);
    this.outlierIndices = sortedIndices
      .slice(0, numOutliers)
      .map(item => item.index)
      .sort((a, b) => a - b);

    this.applyHandling();
    return this.getResult();
  }

  /**
   * Detect outliers using Local Outlier Factor (simplified)
   */
  detectByLOF(k: number = 5, contamination: number = 0.1): OutlierDetectionResult {
    if (this.data.length === 0) {
      throw new Error('No data for outlier detection');
    }

    this.method = 'local-outlier-factor';
    this.threshold = contamination;
    const n = this.data.length;
    k = Math.min(k, n - 1);

    // Calculate distances between all points
    const distances: number[][] = [];
    for (let i = 0; i < n; i++) {
      distances[i] = [];
      for (let j = 0; j < n; j++) {
        distances[i][j] = this.euclideanDistance(this.data[i], this.data[j]);
      }
    }

    // Calculate k-distance and k-neighbors for each point
    const kDistances: number[] = [];
    const kNeighbors: number[][] = [];

    for (let i = 0; i < n; i++) {
      const sortedDistances = distances[i]
        .map((dist, index) => ({ dist, index }))
        .filter(item => item.index !== i)
        .sort((a, b) => a.dist - b.dist);
      
      kDistances[i] = sortedDistances[k - 1].dist;
      kNeighbors[i] = sortedDistances.slice(0, k).map(item => item.index);
    }

    // Calculate reachability distances
    const reachabilityDistances: number[][] = [];
    for (let i = 0; i < n; i++) {
      reachabilityDistances[i] = [];
      for (let j = 0; j < n; j++) {
        reachabilityDistances[i][j] = Math.max(distances[i][j], kDistances[j]);
      }
    }

    // Calculate Local Reachability Density (LRD)
    const lrd: number[] = [];
    for (let i = 0; i < n; i++) {
      const avgReachDist = kNeighbors[i]
        .reduce((sum, neighbor) => sum + reachabilityDistances[i][neighbor], 0) / k;
      lrd[i] = avgReachDist > 0 ? 1 / avgReachDist : Infinity;
    }

    // Calculate Local Outlier Factor (LOF)
    this.outlierScores = [];
    for (let i = 0; i < n; i++) {
      const avgNeighborLRD = kNeighbors[i]
        .reduce((sum, neighbor) => sum + lrd[neighbor], 0) / k;
      const lof = lrd[i] > 0 ? avgNeighborLRD / lrd[i] : 1;
      this.outlierScores.push(lof);
    }

    // Select top contamination% as outliers
    const sortedIndices = this.outlierScores
      .map((score, index) => ({ score, index }))
      .sort((a, b) => b.score - a.score);
    
    const numOutliers = Math.floor(n * contamination);
    this.outlierIndices = sortedIndices
      .slice(0, numOutliers)
      .map(item => item.index)
      .sort((a, b) => a - b);

    this.applyHandling();
    return this.getResult();
  }

  /**
   * Detect outliers using DBSCAN clustering (points in small clusters or noise)
   */
  detectByDBSCAN(eps: number = 0.5, minPts: number = 5): OutlierDetectionResult {
    if (this.data.length === 0) {
      throw new Error('No data for outlier detection');
    }

    this.method = 'dbscan';
    this.threshold = eps;
    const n = this.data.length;

    // DBSCAN clustering
    const labels = new Array(n).fill(-1); // -1 = unvisited, -2 = noise
    const visited = new Array(n).fill(false);
    let clusterId = 0;

    for (let i = 0; i < n; i++) {
      if (visited[i]) continue;
      
      visited[i] = true;
      const neighbors = this.getNeighbors(i, eps);
      
      if (neighbors.length < minPts) {
        labels[i] = -2; // Mark as noise
      } else {
        this.expandCluster(i, neighbors, clusterId, eps, minPts, labels, visited);
        clusterId++;
      }
    }

    // Points labeled as noise (-2) are outliers
    this.outlierIndices = [];
    this.outlierScores = [];
    
    for (let i = 0; i < n; i++) {
      if (labels[i] === -2) {
        this.outlierIndices.push(i);
        this.outlierScores.push(1.0); // Binary: outlier or not
      } else {
        this.outlierScores.push(0.0);
      }
    }

    this.applyHandling();
    return this.getResult();
  }

  /**
   * Apply the specified handling method to outliers
   */
  private applyHandling(): void {
    switch (this.handling) {
      case 'remove':
        this.cleanedData = this.data.filter((_, index) => !this.outlierIndices.includes(index));
        break;
        
      case 'cap':
        this.cleanedData = this.data.map((row, index) => {
          if (this.outlierIndices.includes(index)) {
            return this.capOutlierValues(row);
          }
          return [...row];
        });
        break;
        
      case 'transform':
        this.cleanedData = this.data.map((row, index) => {
          if (this.outlierIndices.includes(index)) {
            return this.transformOutlierValues(row);
          }
          return [...row];
        });
        break;
        
      case 'keep':
      default:
        this.cleanedData = this.data.map(row => [...row]);
        break;
    }
  }

  /**
   * Cap outlier values to reasonable bounds
   */
  private capOutlierValues(row: number[]): number[] {
    const d = row.length;
    const cappedRow: number[] = [];
    
    for (let j = 0; j < d; j++) {
      const column = this.data.map(r => r[j]);
      const sorted = [...column].sort((a, b) => a - b);
      const q1 = sorted[Math.floor(sorted.length * 0.25)];
      const q3 = sorted[Math.floor(sorted.length * 0.75)];
      const iqr = q3 - q1;
      
      const lowerBound = q1 - 1.5 * iqr;
      const upperBound = q3 + 1.5 * iqr;
      
      cappedRow.push(Math.max(lowerBound, Math.min(upperBound, row[j])));
    }
    
    return cappedRow;
  }

  /**
   * Transform outlier values using log transformation
   */
  private transformOutlierValues(row: number[]): number[] {
    return row.map(val => {
      if (val > 0) {
        return Math.log(1 + val);
      } else if (val < 0) {
        return -Math.log(1 + Math.abs(val));
      }
      return val;
    });
  }

  /**
   * Make predictions (detect outliers in new data)
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    
    // For simplicity, return outlier scores for new data
    // In practice, you'd apply the same detection method
    const tempData = this.data;
    this.data = X;
    
    let scores: number[] = [];
    switch (this.method) {
      case 'z-score':
        scores = this.detectByZScore(this.threshold).outlierScores;
        break;
      case 'iqr':
        scores = this.detectByIQR(this.threshold).outlierScores;
        break;
      default:
        scores = X.map(() => 0);
    }
    
    this.data = tempData;
    return scores;
  }

  /**
   * Get visualization data
   */
  getVisualizationData(): VisualizationData[] {
    const result = this.getResult();
    const visualizations: VisualizationData[] = [];

    // Outlier scores histogram
    visualizations.push({
      type: 'scatter2d',
      data: {
        x: Array.from({ length: result.outlierScores.length }, (_, i) => i),
        y: result.outlierScores,
        mode: 'markers',
        marker: {
          color: result.outlierScores.map((_, i) => 
            result.outlierIndices.includes(i) ? 'red' : 'blue'
          ),
          size: 6
        },
        name: 'Outlier Scores'
      },
      layout: {
        title: `Outlier Scores - ${this.method.toUpperCase()} Method`,
        xaxis: { title: 'Sample Index' },
        yaxis: { title: 'Outlier Score' }
      }
    });

    // 2D scatter plot if data has at least 2 dimensions
    if (result.originalData[0]?.length >= 2) {
      const colors = result.originalData.map((_, i) => 
        result.outlierIndices.includes(i) ? 'red' : 'blue'
      );
      
      visualizations.push({
        type: 'scatter2d',
        data: {
          x: result.originalData.map(row => row[0]),
          y: result.originalData.map(row => row[1]),
          mode: 'markers',
          marker: {
            color: colors,
            size: 8,
            opacity: 0.7
          },
          name: 'Data Points'
        },
        layout: {
          title: 'Data Points with Outliers Highlighted',
          xaxis: { title: 'Feature 1' },
          yaxis: { title: 'Feature 2' }
        }
      });
    }

    return visualizations;
  }

  /**
   * Get performance metrics
   */
  getMetrics(): Metrics {
    const result = this.getResult();
    
    return {
      totalSamples: result.statistics.totalSamples,
      outliersDetected: result.statistics.outliersDetected,
      outlierPercentage: result.statistics.outlierPercentage,
      cleanedSamples: result.cleanedData.length,
      dataReduction: ((result.statistics.totalSamples - result.cleanedData.length) / result.statistics.totalSamples) * 100
    };
  }

  /**
   * Get the current detection result
   */
  getResult(): OutlierDetectionResult {
    const totalSamples = this.data.length;
    const outliersDetected = this.outlierIndices.length;
    const outlierPercentage = totalSamples > 0 ? (outliersDetected / totalSamples) * 100 : 0;
    
    return {
      cleanedData: this.cleanedData.length > 0 ? this.cleanedData : this.data,
      originalData: this.data,
      outlierIndices: this.outlierIndices,
      outlierScores: this.outlierScores,
      method: this.method,
      handling: this.handling,
      statistics: {
        totalSamples,
        outliersDetected,
        outlierPercentage,
        threshold: this.threshold
      }
    };
  }

  /**
   * Reset to initial state
   */
  override reset(): void {
    super.reset();
    this.data = [];
    this.cleanedData = [];
    this.outlierIndices = [];
    this.outlierScores = [];
    this.method = 'none';
    this.handling = 'keep';
    this.threshold = 3.0;
  }

  // Helper methods
  private buildIsolationTree(data: number[][], maxDepth: number): number[] {
    const pathLengths = new Array(data.length).fill(0);
    
    const buildTree = (indices: number[], depth: number): void => {
      if (depth >= maxDepth || indices.length <= 1) {
        // Leaf node - set path length
        for (const idx of indices) {
          pathLengths[idx] = depth;
        }
        return;
      }
      
      // Random split
      const feature = Math.floor(Math.random() * data[0].length);
      const values = indices.map(idx => data[idx][feature]);
      const minVal = Math.min(...values);
      const maxVal = Math.max(...values);
      
      if (minVal === maxVal) {
        // Can't split further
        for (const idx of indices) {
          pathLengths[idx] = depth;
        }
        return;
      }
      
      const splitValue = minVal + Math.random() * (maxVal - minVal);
      
      const leftIndices = indices.filter(idx => data[idx][feature] < splitValue);
      const rightIndices = indices.filter(idx => data[idx][feature] >= splitValue);
      
      if (leftIndices.length > 0) {
        buildTree(leftIndices, depth + 1);
      }
      if (rightIndices.length > 0) {
        buildTree(rightIndices, depth + 1);
      }
    };
    
    buildTree(Array.from({ length: data.length }, (_, i) => i), 0);
    return pathLengths;
  }

  private euclideanDistance(point1: number[], point2: number[]): number {
    let sum = 0;
    for (let i = 0; i < point1.length; i++) {
      sum += Math.pow(point1[i] - point2[i], 2);
    }
    return Math.sqrt(sum);
  }

  private getNeighbors(pointIndex: number, eps: number): number[] {
    const neighbors: number[] = [];
    const n = this.data.length;
    
    for (let i = 0; i < n; i++) {
      if (i !== pointIndex) {
        const distance = this.euclideanDistance(this.data[pointIndex], this.data[i]);
        if (distance <= eps) {
          neighbors.push(i);
        }
      }
    }
    
    return neighbors;
  }

  private expandCluster(
    pointIndex: number,
    neighbors: number[],
    clusterId: number,
    eps: number,
    minPts: number,
    labels: number[],
    visited: boolean[]
  ): void {
    labels[pointIndex] = clusterId;
    
    for (let i = 0; i < neighbors.length; i++) {
      const neighbor = neighbors[i];
      
      if (!visited[neighbor]) {
        visited[neighbor] = true;
        const neighborNeighbors = this.getNeighbors(neighbor, eps);
        
        if (neighborNeighbors.length >= minPts) {
          neighbors.push(...neighborNeighbors);
        }
      }
      
      if (labels[neighbor] === -1 || labels[neighbor] === -2) {
        labels[neighbor] = clusterId;
      }
    }
  }

  /**
   * Generate sample data with outliers
   */
  static generateDataWithOutliers(
    rows: number,
    cols: number,
    outlierPercentage: number = 0.1,
    outlierMagnitude: number = 3
  ): number[][] {
    const data: number[][] = [];
    const numOutliers = Math.floor(rows * outlierPercentage);
    
    // Generate normal data
    for (let i = 0; i < rows - numOutliers; i++) {
      const row: number[] = [];
      for (let j = 0; j < cols; j++) {
        // Normal distribution approximation
        const u1 = Math.random();
        const u2 = Math.random();
        const normal = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        row.push(normal);
      }
      data.push(row);
    }
    
    // Generate outliers
    for (let i = 0; i < numOutliers; i++) {
      const row: number[] = [];
      for (let j = 0; j < cols; j++) {
        // Outlier: normal value + large offset
        const u1 = Math.random();
        const u2 = Math.random();
        const normal = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        const outlier = normal + (Math.random() > 0.5 ? 1 : -1) * outlierMagnitude;
        row.push(outlier);
      }
      data.push(row);
    }
    
    // Shuffle the data
    for (let i = data.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [data[i], data[j]] = [data[j], data[i]];
    }
    
    return data;
  }
}