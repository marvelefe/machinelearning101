/**
 * Feature Selection Algorithm
 * Implements various feature selection techniques to identify the most relevant features
 * Requirements: 8.1, 8.2, 8.3, 8.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics, TrainingHistory } from '~/types/algorithm'; 


export type SelectionMethod = 'variance' | 'correlation' | 'mutual-info' | 'chi2' | 'f-score' | 'rfe' | 'none';

export interface FeatureSelectionResult {
  selectedFeatures: number[][];
  originalFeatures: number[][];
  selectedIndices: number[];
  featureScores: number[];
  method: SelectionMethod;
  statistics: {
    originalFeatureCount: number;
    selectedFeatureCount: number;
    selectionRatio: number;
    averageScore: number;
    scoreThreshold?: number;
  };
}

export class FeatureSelection extends BaseAlgorithm {
  name = 'Feature Selection';
  
  private data: number[][] = [];
  private targetValues: number[] = [];
  private selectedData: number[][] = [];
  private selectedIndices: number[] = [];
  private featureScores: number[] = [];
  private method: SelectionMethod = 'none';
  private threshold: number = 0.5;

  constructor() {
    super();
  }

  /**
   * Set the data for feature selection
   */
  setData(data: number[][], target: number[], threshold: number = 0.5): void {
    this.data = data;
    this.targetValues = target;
    this.threshold = threshold;
    this.selectedData = [];
    this.selectedIndices = [];
    this.featureScores = [];
    this.trained = false;
  }

  /**
   * Fit the feature selector
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    this.setData(X, y || [], this.threshold);
    
    // Apply default variance threshold selection
    this.selectByVarianceThreshold();
    this.trained = true;
  }

  /**
   * Select features based on variance threshold
   */
  selectByVarianceThreshold(threshold: number = 0.1): FeatureSelectionResult {
    if (this.data.length === 0) {
      throw new Error('No data for feature selection');
    }

    this.method = 'variance';
    this.threshold = threshold;
    const numFeatures = this.data[0].length;
    
    // Calculate variance for each feature
    this.featureScores = [];
    for (let j = 0; j < numFeatures; j++) {
      const column = this.data.map(row => row[j]);
      const mean = column.reduce((sum, val) => sum + val, 0) / column.length;
      const variance = column.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / column.length;
      this.featureScores.push(variance);
    }

    // Select features above threshold
    this.selectedIndices = [];
    this.featureScores.forEach((score, index) => {
      if (score > threshold) {
        this.selectedIndices.push(index);
      }
    });

    // If no features selected, select top 50%
    if (this.selectedIndices.length === 0) {
      const sortedIndices = this.featureScores
        .map((score, index) => ({ score, index }))
        .sort((a, b) => b.score - a.score)
        .slice(0, Math.max(1, Math.floor(numFeatures / 2)))
        .map(item => item.index);
      this.selectedIndices = sortedIndices;
    }

    // Extract selected features
    this.selectedData = this.data.map(row => 
      this.selectedIndices.map(index => row[index])
    );

    return this.getResult();
  }

  /**
   * Select features based on correlation with target
   */
  selectByCorrelation(threshold: number = 0.3): FeatureSelectionResult {
    if (this.data.length === 0 || this.targetValues.length === 0) {
      throw new Error('Data and target values required for correlation-based selection');
    }

    this.method = 'correlation';
    this.threshold = threshold;
    const numFeatures = this.data[0].length;
    
    // Calculate correlation with target for each feature
    this.featureScores = [];
    for (let j = 0; j < numFeatures; j++) {
      const column = this.data.map(row => row[j]);
      const correlation = this.calculateCorrelation(column, this.targetValues);
      this.featureScores.push(Math.abs(correlation));
    }

    // Select features above threshold
    this.selectedIndices = [];
    this.featureScores.forEach((score, index) => {
      if (score > threshold) {
        this.selectedIndices.push(index);
      }
    });

    // If no features selected, select top features
    if (this.selectedIndices.length === 0) {
      const sortedIndices = this.featureScores
        .map((score, index) => ({ score, index }))
        .sort((a, b) => b.score - a.score)
        .slice(0, Math.max(1, Math.floor(numFeatures / 2)))
        .map(item => item.index);
      this.selectedIndices = sortedIndices;
    }

    // Extract selected features
    this.selectedData = this.data.map(row => 
      this.selectedIndices.map(index => row[index])
    );

    return this.getResult();
  }

  /**
   * Select features using mutual information
   */
  selectByMutualInformation(k: number = 5): FeatureSelectionResult {
    if (this.data.length === 0 || this.targetValues.length === 0) {
      throw new Error('Data and target values required for mutual information selection');
    }

    this.method = 'mutual-info';
    const numFeatures = this.data[0].length;
    
    // Calculate mutual information for each feature
    this.featureScores = [];
    for (let j = 0; j < numFeatures; j++) {
      const column = this.data.map(row => row[j]);
      const mutualInfo = this.calculateMutualInformation(column, this.targetValues);
      this.featureScores.push(mutualInfo);
    }

    // Select top k features
    const sortedIndices = this.featureScores
      .map((score, index) => ({ score, index }))
      .sort((a, b) => b.score - a.score)
      .slice(0, Math.min(k, numFeatures))
      .map(item => item.index);
    
    this.selectedIndices = sortedIndices;

    // Extract selected features
    this.selectedData = this.data.map(row => 
      this.selectedIndices.map(index => row[index])
    );

    return this.getResult();
  }

  /**
   * Select features using F-score (ANOVA F-test)
   */
  selectByFScore(k: number = 5): FeatureSelectionResult {
    if (this.data.length === 0 || this.targetValues.length === 0) {
      throw new Error('Data and target values required for F-score selection');
    }

    this.method = 'f-score';
    const numFeatures = this.data[0].length;
    
    // Calculate F-score for each feature
    this.featureScores = [];
    for (let j = 0; j < numFeatures; j++) {
      const column = this.data.map(row => row[j]);
      const fScore = this.calculateFScore(column, this.targetValues);
      this.featureScores.push(fScore);
    }

    // Select top k features
    const sortedIndices = this.featureScores
      .map((score, index) => ({ score, index }))
      .sort((a, b) => b.score - a.score)
      .slice(0, Math.min(k, numFeatures))
      .map(item => item.index);
    
    this.selectedIndices = sortedIndices;

    // Extract selected features
    this.selectedData = this.data.map(row => 
      this.selectedIndices.map(index => row[index])
    );

    return this.getResult();
  }

  /**
   * Recursive Feature Elimination (simplified version)
   */
  selectByRFE(k: number = 5): FeatureSelectionResult {
    if (this.data.length === 0 || this.targetValues.length === 0) {
      throw new Error('Data and target values required for RFE selection');
    }

    this.method = 'rfe';
    const numFeatures = this.data[0].length;
    
    // Start with all features and iteratively remove the worst
    let currentFeatures = Array.from({ length: numFeatures }, (_, i) => i);
    const featureImportance: number[] = new Array(numFeatures).fill(0);
    
    while (currentFeatures.length > k) {
      // Calculate importance for current features using correlation
      const currentScores: number[] = [];
      for (const featureIndex of currentFeatures) {
        const column = this.data.map(row => row[featureIndex]);
        const correlation = Math.abs(this.calculateCorrelation(column, this.targetValues));
        currentScores.push(correlation);
        featureImportance[featureIndex] = correlation;
      }
      
      // Find and remove the least important feature
      const minScoreIndex = currentScores.indexOf(Math.min(...currentScores));
      currentFeatures.splice(minScoreIndex, 1);
    }
    
    this.selectedIndices = currentFeatures;
    this.featureScores = featureImportance;

    // Extract selected features
    this.selectedData = this.data.map(row => 
      this.selectedIndices.map(index => row[index])
    );

    return this.getResult();
  }

  /**
   * Make predictions (transform new data using selected features)
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    
    // Transform new data by selecting the same features
    const transformedData = X.map(row => 
      this.selectedIndices.map(index => row[index] || 0)
    );
    
    return transformedData.flat();
  }

  /**
   * Get visualization data
   */
  getVisualizationData(): VisualizationData[] {
    const result = this.getResult();
    
    return [
      {
        type: 'scatter2d',
        data: {
          x: Array.from({ length: result.featureScores.length }, (_, i) => i),
          y: result.featureScores,
          mode: 'markers+lines',
          marker: {
            color: result.featureScores.map((_, i) => 
              result.selectedIndices.includes(i) ? 'red' : 'blue'
            ),
            size: 8
          },
          name: 'Feature Scores'
        },
        layout: {
          title: `Feature Selection: ${this.method.charAt(0).toUpperCase() + this.method.slice(1)} Method`,
          xaxis: { title: 'Feature Index' },
          yaxis: { title: 'Feature Score' }
        }
      },
      {
        type: 'heatmap',
        data: {
          z: result.selectedFeatures.slice(0, Math.min(20, result.selectedFeatures.length)),
          x: result.selectedIndices.map(i => `Feature ${i}`),
          y: result.selectedFeatures.map((_, i) => `Sample ${i + 1}`).slice(0, Math.min(20, result.selectedFeatures.length)),
          colorscale: 'Viridis',
          showscale: true
        },
        layout: {
          title: 'Selected Features Heatmap',
          xaxis: { title: 'Selected Features' },
          yaxis: { title: 'Samples' }
        }
      }
    ];
  }

  /**
   * Get performance metrics
   */
  getMetrics(): Metrics {
    const result = this.getResult();
    
    return {
      originalFeatures: result.statistics.originalFeatureCount,
      selectedFeatures: result.statistics.selectedFeatureCount,
      selectionRatio: result.statistics.selectionRatio,
      averageScore: result.statistics.averageScore,
      dimensionalityReduction: (1 - result.statistics.selectionRatio) * 100
    };
  }

  /**
   * Get the current selection result
   */
  getResult(): FeatureSelectionResult {
    const originalFeatureCount = this.data[0]?.length || 0;
    const selectedFeatureCount = this.selectedIndices.length;
    const selectionRatio = originalFeatureCount > 0 ? selectedFeatureCount / originalFeatureCount : 0;
    const averageScore = this.featureScores.length > 0 
      ? this.featureScores.reduce((sum, score) => sum + score, 0) / this.featureScores.length 
      : 0;
    
    return {
      selectedFeatures: this.selectedData.length > 0 ? this.selectedData : this.data,
      originalFeatures: this.data,
      selectedIndices: this.selectedIndices,
      featureScores: this.featureScores,
      method: this.method,
      statistics: {
        originalFeatureCount,
        selectedFeatureCount,
        selectionRatio,
        averageScore,
        scoreThreshold: this.threshold
      }
    };
  }

  /**
   * Reset to initial state
   */
  override reset(): void {
    super.reset();
    this.data = [];
    this.targetValues = [];
    this.selectedData = [];
    this.selectedIndices = [];
    this.featureScores = [];
    this.method = 'none';
    this.threshold = 0.5;
  }

  // Helper methods
  private calculateCorrelation(x: number[], y: number[]): number {
    const n = x.length;
    if (n !== y.length || n === 0) return 0;
    
    const meanX = x.reduce((sum, val) => sum + val, 0) / n;
    const meanY = y.reduce((sum, val) => sum + val, 0) / n;
    
    let numerator = 0;
    let denomX = 0;
    let denomY = 0;
    
    for (let i = 0; i < n; i++) {
      const diffX = x[i] - meanX;
      const diffY = y[i] - meanY;
      numerator += diffX * diffY;
      denomX += diffX * diffX;
      denomY += diffY * diffY;
    }
    
    const denominator = Math.sqrt(denomX * denomY);
    return denominator === 0 ? 0 : numerator / denominator;
  }

  private calculateMutualInformation(x: number[], y: number[]): number {
    // Simplified mutual information calculation
    // Discretize continuous variables into bins
    const bins = 5;
    const xBinned = this.discretize(x, bins);
    const yBinned = this.discretize(y, bins);
    
    // Calculate joint and marginal probabilities
    const jointProb = new Map<string, number>();
    const xProb = new Map<number, number>();
    const yProb = new Map<number, number>();
    
    const n = x.length;
    
    for (let i = 0; i < n; i++) {
      const xBin = xBinned[i];
      const yBin = yBinned[i];
      const joint = `${xBin},${yBin}`;
      
      jointProb.set(joint, (jointProb.get(joint) || 0) + 1);
      xProb.set(xBin, (xProb.get(xBin) || 0) + 1);
      yProb.set(yBin, (yProb.get(yBin) || 0) + 1);
    }
    
    // Convert counts to probabilities
    jointProb.forEach((count, key) => jointProb.set(key, count / n));
    xProb.forEach((count, key) => xProb.set(key, count / n));
    yProb.forEach((count, key) => yProb.set(key, count / n));
    
    // Calculate mutual information
    let mi = 0;
    jointProb.forEach((pxy, key) => {
      const [xBin, yBin] = key.split(',').map(Number);
      const px = xProb.get(xBin) || 0;
      const py = yProb.get(yBin) || 0;
      
      if (pxy > 0 && px > 0 && py > 0) {
        mi += pxy * Math.log2(pxy / (px * py));
      }
    });
    
    return mi;
  }

  private calculateFScore(x: number[], y: number[]): number {
    // Simplified F-score calculation for continuous features
    // Group by target classes
    const classes = Array.from(new Set(y));
    if (classes.length < 2) return 0;
    
    const groups: number[][] = classes.map(() => []);
    for (let i = 0; i < y.length; i++) {
      const classIndex = classes.indexOf(y[i]);
      groups[classIndex].push(x[i]);
    }
    
    // Calculate overall mean
    const overallMean = x.reduce((sum, val) => sum + val, 0) / x.length;
    
    // Calculate between-group and within-group variance
    let betweenGroupVar = 0;
    let withinGroupVar = 0;
    
    for (const group of groups) {
      if (group.length === 0) continue;
      
      const groupMean = group.reduce((sum, val) => sum + val, 0) / group.length;
      betweenGroupVar += group.length * Math.pow(groupMean - overallMean, 2);
      
      for (const val of group) {
        withinGroupVar += Math.pow(val - groupMean, 2);
      }
    }
    
    const dfBetween = classes.length - 1;
    const dfWithin = x.length - classes.length;
    
    if (dfBetween === 0 || dfWithin === 0) return 0;
    
    const msBetween = betweenGroupVar / dfBetween;
    const msWithin = withinGroupVar / dfWithin;
    
    return msWithin === 0 ? 0 : msBetween / msWithin;
  }

  private discretize(values: number[], bins: number): number[] {
    const min = Math.min(...values);
    const max = Math.max(...values);
    const binSize = (max - min) / bins;
    
    return values.map(val => {
      const bin = Math.floor((val - min) / binSize);
      return Math.min(bin, bins - 1);
    });
  }

  /**
   * Generate sample data with some irrelevant features
   */
  static generateDataWithIrrelevantFeatures(
    rows: number,
    relevantFeatures: number,
    irrelevantFeatures: number,
    noiseLevel: number = 0.1
  ): { data: number[][], target: number[] } {
    const data: number[][] = [];
    const target: number[] = [];
    
    for (let i = 0; i < rows; i++) {
      const row: number[] = [];
      
      // Generate relevant features
      const relevantValues: number[] = [];
      for (let j = 0; j < relevantFeatures; j++) {
        const value = Math.random() * 10;
        relevantValues.push(value);
        row.push(value);
      }
      
      // Generate irrelevant features (random noise)
      for (let j = 0; j < irrelevantFeatures; j++) {
        row.push(Math.random() * 10);
      }
      
      // Generate target based on relevant features
      const targetValue = relevantValues.reduce((sum, val) => sum + val, 0) / relevantValues.length
        + (Math.random() - 0.5) * noiseLevel;
      
      data.push(row);
      target.push(targetValue);
    }
    
    return { data, target };
  }
}