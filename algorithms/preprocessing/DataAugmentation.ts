/**
 * Data Augmentation Algorithm
 * Implements various data augmentation techniques to increase dataset size and diversity
 * Requirements: 8.1, 8.2, 8.3, 8.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics, TrainingHistory } from '~/types/algorithm'; 


export type AugmentationMethod = 'noise' | 'rotation' | 'scaling' | 'translation' | 'smote' | 'mixup' | 'none';

export interface DataAugmentationResult {
  augmentedData: number[][];
  originalData: number[][];
  augmentedLabels?: number[];
  originalLabels?: number[];
  method: AugmentationMethod;
  statistics: {
    originalSamples: number;
    augmentedSamples: number;
    augmentationRatio: number;
    totalSamples: number;
  };
}

export class DataAugmentation extends BaseAlgorithm {
  name = 'Data Augmentation';
  
  private data: number[][] = [];
  private labels: number[] = [];
  private augmentedData: number[][] = [];
  private augmentedLabels: number[] = [];
  private method: AugmentationMethod = 'none';
  private augmentationFactor: number = 2;

  constructor() {
    super();
  }

  /**
   * Set the data for augmentation
   */
  setData(data: number[][], labels?: number[], augmentationFactor: number = 2): void {
    this.data = data;
    this.labels = labels || [];
    this.augmentationFactor = augmentationFactor;
    this.augmentedData = [];
    this.augmentedLabels = [];
    this.trained = false;
  }

  /**
   * Fit the data augmentation
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    this.setData(X, y, this.augmentationFactor);
    
    // Apply default noise augmentation
    this.augmentWithNoise();
    this.trained = true;
  }

  /**
   * Augment data by adding Gaussian noise
   */
  augmentWithNoise(noiseLevel: number = 0.1, factor?: number): DataAugmentationResult {
    if (this.data.length === 0) {
      throw new Error('No data for augmentation');
    }

    this.method = 'noise';
    const augFactor = factor || this.augmentationFactor;
    const n = this.data.length;
    const d = this.data[0].length;

    // Calculate feature-wise standard deviations for noise scaling
    const stds: number[] = [];
    for (let j = 0; j < d; j++) {
      const column = this.data.map(row => row[j]);
      const mean = column.reduce((sum, val) => sum + val, 0) / n;
      const variance = column.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
      stds.push(Math.sqrt(variance));
    }

    // Generate augmented samples
    this.augmentedData = [...this.data];
    this.augmentedLabels = [...this.labels];

    for (let rep = 1; rep < augFactor; rep++) {
      for (let i = 0; i < n; i++) {
        const noisyRow: number[] = [];
        
        for (let j = 0; j < d; j++) {
          // Add Gaussian noise scaled by feature standard deviation
          const noise = this.generateGaussianNoise() * noiseLevel * stds[j];
          noisyRow.push(this.data[i][j] + noise);
        }
        
        this.augmentedData.push(noisyRow);
        if (this.labels.length > 0) {
          this.augmentedLabels.push(this.labels[i]);
        }
      }
    }

    return this.getResult();
  }

  /**
   * Augment data with rotation (for 2D data)
   */
  augmentWithRotation(maxAngle: number = 30, factor?: number): DataAugmentationResult {
    if (this.data.length === 0) {
      throw new Error('No data for augmentation');
    }

    if (this.data[0].length < 2) {
      throw new Error('Rotation requires at least 2D data');
    }

    this.method = 'rotation';
    const augFactor = factor || this.augmentationFactor;
    const n = this.data.length;

    this.augmentedData = [...this.data];
    this.augmentedLabels = [...this.labels];

    for (let rep = 1; rep < augFactor; rep++) {
      for (let i = 0; i < n; i++) {
        // Random rotation angle
        const angle = (Math.random() - 0.5) * 2 * maxAngle * (Math.PI / 180);
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        
        const rotatedRow = [...this.data[i]];
        
        // Apply rotation to first two dimensions
        const x = this.data[i][0];
        const y = this.data[i][1];
        
        rotatedRow[0] = x * cos - y * sin;
        rotatedRow[1] = x * sin + y * cos;
        
        this.augmentedData.push(rotatedRow);
        if (this.labels.length > 0) {
          this.augmentedLabels.push(this.labels[i]);
        }
      }
    }

    return this.getResult();
  }

  /**
   * Augment data with scaling
   */
  augmentWithScaling(scaleRange: [number, number] = [0.8, 1.2], factor?: number): DataAugmentationResult {
    if (this.data.length === 0) {
      throw new Error('No data for augmentation');
    }

    this.method = 'scaling';
    const augFactor = factor || this.augmentationFactor;
    const n = this.data.length;
    const d = this.data[0].length;

    this.augmentedData = [...this.data];
    this.augmentedLabels = [...this.labels];

    for (let rep = 1; rep < augFactor; rep++) {
      for (let i = 0; i < n; i++) {
        const scaledRow: number[] = [];
        
        for (let j = 0; j < d; j++) {
          // Random scaling factor for each dimension
          const scale = scaleRange[0] + Math.random() * (scaleRange[1] - scaleRange[0]);
          scaledRow.push(this.data[i][j] * scale);
        }
        
        this.augmentedData.push(scaledRow);
        if (this.labels.length > 0) {
          this.augmentedLabels.push(this.labels[i]);
        }
      }
    }

    return this.getResult();
  }

  /**
   * Augment data with translation
   */
  augmentWithTranslation(translationRange: number = 0.1, factor?: number): DataAugmentationResult {
    if (this.data.length === 0) {
      throw new Error('No data for augmentation');
    }

    this.method = 'translation';
    const augFactor = factor || this.augmentationFactor;
    const n = this.data.length;
    const d = this.data[0].length;

    // Calculate feature ranges for translation scaling
    const ranges: number[] = [];
    for (let j = 0; j < d; j++) {
      const column = this.data.map(row => row[j]);
      const min = Math.min(...column);
      const max = Math.max(...column);
      ranges.push(max - min);
    }

    this.augmentedData = [...this.data];
    this.augmentedLabels = [...this.labels];

    for (let rep = 1; rep < augFactor; rep++) {
      for (let i = 0; i < n; i++) {
        const translatedRow: number[] = [];
        
        for (let j = 0; j < d; j++) {
          // Random translation scaled by feature range
          const translation = (Math.random() - 0.5) * 2 * translationRange * ranges[j];
          translatedRow.push(this.data[i][j] + translation);
        }
        
        this.augmentedData.push(translatedRow);
        if (this.labels.length > 0) {
          this.augmentedLabels.push(this.labels[i]);
        }
      }
    }

    return this.getResult();
  }

  /**
   * SMOTE (Synthetic Minority Oversampling Technique)
   */
  augmentWithSMOTE(k: number = 5, factor?: number): DataAugmentationResult {
    if (this.data.length === 0 || this.labels.length === 0) {
      throw new Error('SMOTE requires data and labels');
    }

    this.method = 'smote';
    const augFactor = factor || this.augmentationFactor;
    const n = this.data.length;
    const d = this.data[0].length;

    // Group data by class
    const classes = Array.from(new Set(this.labels));
    const classData: { [key: number]: number[][] } = {};
    const classIndices: { [key: number]: number[] } = {};

    for (const cls of classes) {
      classData[cls] = [];
      classIndices[cls] = [];
    }

    for (let i = 0; i < n; i++) {
      const cls = this.labels[i];
      classData[cls].push(this.data[i]);
      classIndices[cls].push(i);
    }

    this.augmentedData = [...this.data];
    this.augmentedLabels = [...this.labels];

    // Generate synthetic samples for each class
    for (const cls of classes) {
      const classPoints = classData[cls];
      const numSamples = Math.floor(classPoints.length * (augFactor - 1));

      for (let s = 0; s < numSamples; s++) {
        // Select random point from class
        const randomIndex = Math.floor(Math.random() * classPoints.length);
        const point = classPoints[randomIndex];

        // Find k nearest neighbors in the same class
        const distances = classPoints.map((neighbor, idx) => ({
          distance: this.euclideanDistance(point, neighbor),
          index: idx,
          point: neighbor
        })).filter(item => item.index !== randomIndex)
          .sort((a, b) => a.distance - b.distance)
          .slice(0, Math.min(k, classPoints.length - 1));

        if (distances.length === 0) continue;

        // Select random neighbor
        const neighborIndex = Math.floor(Math.random() * distances.length);
        const neighbor = distances[neighborIndex].point;

        // Generate synthetic sample
        const syntheticSample: number[] = [];
        const lambda = Math.random(); // Random interpolation factor

        for (let j = 0; j < d; j++) {
          syntheticSample.push(point[j] + lambda * (neighbor[j] - point[j]));
        }

        this.augmentedData.push(syntheticSample);
        this.augmentedLabels.push(cls);
      }
    }

    return this.getResult();
  }

  /**
   * Mixup augmentation
   */
  augmentWithMixup(alpha: number = 0.2, factor?: number): DataAugmentationResult {
    if (this.data.length === 0) {
      throw new Error('No data for augmentation');
    }

    this.method = 'mixup';
    const augFactor = factor || this.augmentationFactor;
    const n = this.data.length;
    const d = this.data[0].length;

    this.augmentedData = [...this.data];
    this.augmentedLabels = [...this.labels];

    const numAugmented = Math.floor(n * (augFactor - 1));

    for (let s = 0; s < numAugmented; s++) {
      // Select two random samples
      const idx1 = Math.floor(Math.random() * n);
      const idx2 = Math.floor(Math.random() * n);

      // Generate mixing coefficient from Beta distribution (approximated)
      const lambda = this.betaSample(alpha, alpha);

      // Mix the samples
      const mixedSample: number[] = [];
      for (let j = 0; j < d; j++) {
        mixedSample.push(lambda * this.data[idx1][j] + (1 - lambda) * this.data[idx2][j]);
      }

      this.augmentedData.push(mixedSample);

      // Mix labels if available
      if (this.labels.length > 0) {
        // For simplicity, use the label of the more dominant sample
        const label = lambda > 0.5 ? this.labels[idx1] : this.labels[idx2];
        this.augmentedLabels.push(label);
      }
    }

    return this.getResult();
  }

  /**
   * Make predictions (transform new data using learned augmentation)
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    
    // For augmentation, prediction means applying the same augmentation to new data
    const tempData = this.data;
    const tempLabels = this.labels;
    
    this.data = X;
    this.labels = [];
    
    let result: number[] = [];
    switch (this.method) {
      case 'noise':
        result = this.augmentWithNoise().augmentedData.flat();
        break;
      case 'rotation':
        result = this.augmentWithRotation().augmentedData.flat();
        break;
      case 'scaling':
        result = this.augmentWithScaling().augmentedData.flat();
        break;
      case 'translation':
        result = this.augmentWithTranslation().augmentedData.flat();
        break;
      default:
        result = X.flat();
    }
    
    this.data = tempData;
    this.labels = tempLabels;
    return result;
  }

  /**
   * Get visualization data
   */
  getVisualizationData(): VisualizationData[] {
    const result = this.getResult();
    const visualizations: VisualizationData[] = [];

    // Show original vs augmented data distribution
    if (result.originalData[0]?.length >= 2) {
      // Original data
      visualizations.push({
        type: 'scatter2d',
        data: {
          x: result.originalData.map(row => row[0]),
          y: result.originalData.map(row => row[1]),
          mode: 'markers',
          marker: {
            color: 'blue',
            size: 6,
            opacity: 0.7
          },
          name: 'Original Data'
        },
        layout: {
          title: `Data Augmentation - ${this.method.toUpperCase()} Method`,
          xaxis: { title: 'Feature 1' },
          yaxis: { title: 'Feature 2' }
        }
      });

      // Augmented data
      const augmentedOnly = result.augmentedData.slice(result.originalData.length);
      if (augmentedOnly.length > 0) {
        visualizations.push({
          type: 'scatter2d',
          data: {
            x: augmentedOnly.map(row => row[0]),
            y: augmentedOnly.map(row => row[1]),
            mode: 'markers',
            marker: {
              color: 'red',
              size: 4,
              opacity: 0.5
            },
            name: 'Augmented Data'
          },
          layout: {
            title: 'Augmented Data Points',
            xaxis: { title: 'Feature 1' },
            yaxis: { title: 'Feature 2' }
          }
        });
      }
    }

    // Sample count comparison
    visualizations.push({
      type: 'scatter2d',
      data: {
        x: ['Original', 'Augmented', 'Total'],
        y: [
          result.statistics.originalSamples,
          result.statistics.augmentedSamples - result.statistics.originalSamples,
          result.statistics.totalSamples
        ],
        type: 'bar',
        marker: {
          color: ['blue', 'red', 'green']
        },
        name: 'Sample Counts'
      },
      layout: {
        title: 'Dataset Size Comparison',
        xaxis: { title: 'Dataset Type' },
        yaxis: { title: 'Number of Samples' }
      }
    });

    return visualizations;
  }

  /**
   * Get performance metrics
   */
  getMetrics(): Metrics {
    const result = this.getResult();
    
    return {
      originalSamples: result.statistics.originalSamples,
      augmentedSamples: result.statistics.augmentedSamples,
      totalSamples: result.statistics.totalSamples,
      augmentationRatio: result.statistics.augmentationRatio,
      datasetIncrease: ((result.statistics.totalSamples - result.statistics.originalSamples) / result.statistics.originalSamples) * 100
    };
  }

  /**
   * Get the current augmentation result
   */
  getResult(): DataAugmentationResult {
    const originalSamples = this.data.length;
    const augmentedSamples = this.augmentedData.length;
    const totalSamples = augmentedSamples;
    const augmentationRatio = originalSamples > 0 ? augmentedSamples / originalSamples : 1;
    
    return {
      augmentedData: this.augmentedData.length > 0 ? this.augmentedData : this.data,
      originalData: this.data,
      augmentedLabels: this.augmentedLabels.length > 0 ? this.augmentedLabels : undefined,
      originalLabels: this.labels.length > 0 ? this.labels : undefined,
      method: this.method,
      statistics: {
        originalSamples,
        augmentedSamples,
        augmentationRatio,
        totalSamples
      }
    };
  }

  /**
   * Reset to initial state
   */
  override reset(): void {
    super.reset();
    this.data = [];
    this.labels = [];
    this.augmentedData = [];
    this.augmentedLabels = [];
    this.method = 'none';
    this.augmentationFactor = 2;
  }

  // Helper methods
  private generateGaussianNoise(): number {
    // Box-Muller transform for Gaussian noise
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  private euclideanDistance(point1: number[], point2: number[]): number {
    let sum = 0;
    for (let i = 0; i < point1.length; i++) {
      sum += Math.pow(point1[i] - point2[i], 2);
    }
    return Math.sqrt(sum);
  }

  private betaSample(alpha: number, beta: number): number {
    // Simplified Beta distribution sampling using rejection method
    // For simplicity, we'll use a uniform approximation
    const u1 = Math.random();
    const u2 = Math.random();
    
    // Simple approximation: weighted average of two uniform samples
    const weight = alpha / (alpha + beta);
    return weight * u1 + (1 - weight) * u2;
  }

  /**
   * Generate sample data for augmentation demonstration
   */
  static generateSampleData(
    rows: number,
    cols: number,
    numClasses: number = 2,
    clusterSpread: number = 1
  ): { data: number[][], labels: number[] } {
    const data: number[][] = [];
    const labels: number[] = [];
    
    // Generate cluster centers
    const centers: number[][] = [];
    for (let c = 0; c < numClasses; c++) {
      const center: number[] = [];
      for (let j = 0; j < cols; j++) {
        center.push((Math.random() - 0.5) * 10);
      }
      centers.push(center);
    }
    
    // Generate samples around centers
    for (let i = 0; i < rows; i++) {
      const classLabel = Math.floor(Math.random() * numClasses);
      const center = centers[classLabel];
      
      const row: number[] = [];
      for (let j = 0; j < cols; j++) {
        // Add Gaussian noise around center
        const u1 = Math.random();
        const u2 = Math.random();
        const noise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * clusterSpread;
        row.push(center[j] + noise);
      }
      
      data.push(row);
      labels.push(classLabel);
    }
    
    return { data, labels };
  }

  /**
   * Generate imbalanced dataset for SMOTE demonstration
   */
  static generateImbalancedData(
    majoritySize: number,
    minoritySize: number,
    cols: number = 2
  ): { data: number[][], labels: number[] } {
    const data: number[][] = [];
    const labels: number[] = [];
    
    // Majority class (class 0) - centered around origin
    for (let i = 0; i < majoritySize; i++) {
      const row: number[] = [];
      for (let j = 0; j < cols; j++) {
        const u1 = Math.random();
        const u2 = Math.random();
        const noise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        row.push(noise);
      }
      data.push(row);
      labels.push(0);
    }
    
    // Minority class (class 1) - centered away from origin
    for (let i = 0; i < minoritySize; i++) {
      const row: number[] = [];
      for (let j = 0; j < cols; j++) {
        const u1 = Math.random();
        const u2 = Math.random();
        const noise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        row.push(5 + noise); // Offset from majority class
      }
      data.push(row);
      labels.push(1);
    }
    
    // Shuffle the data
    const combined = data.map((row, i) => ({ data: row, label: labels[i] }));
    for (let i = combined.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [combined[i], combined[j]] = [combined[j], combined[i]];
    }
    
    return {
      data: combined.map(item => item.data),
      labels: combined.map(item => item.label)
    };
  }
}