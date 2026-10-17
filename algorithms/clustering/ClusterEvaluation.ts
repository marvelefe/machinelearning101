/**
 * Cluster Evaluation Metrics
 * 
 * Implements various metrics for evaluating clustering quality,
 * both internal (without ground truth) and external (with ground truth).
 * 
 * Requirements: 7.1, 7.2, 7.3, 7.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics, TrainingHistory } from '~/types/algorithm';
import { mean, sum } from '~/utils/math';

export interface ClusterEvaluationParams {
  // No specific parameters needed for evaluation
}

interface ClusteringResult {
  labels: number[];
  data: number[][];
  trueLabels?: number[];
}

interface EvaluationResults {
  internal: InternalMetrics;
  external?: ExternalMetrics;
  summary: string[];
}

interface InternalMetrics {
  silhouetteScore: number;
  calinskiHarabaszIndex: number;
  daviesBouldinIndex: number;
  inertia: number;
  dunn: number;
  silhouettePerCluster: number[];
}

interface ExternalMetrics {
  adjustedRandIndex: number;
  normalizedMutualInfo: number;
  fowlkesMallowsIndex: number;
  homogeneity: number;
  completeness: number;
  vMeasure: number;
  accuracy?: number;
}

export class ClusterEvaluation extends BaseAlgorithm {
  name = 'Cluster Evaluation Metrics';
  
  // Stored results
  private evaluationResults: EvaluationResults | null = null;
  private clusteringResult: ClusteringResult | null = null;
  
  constructor(params: ClusterEvaluationParams = {}) {
    super();
  }
  
  /**
   * Calculate Euclidean distance between two points
   */
  private euclideanDistance(point1: number[], point2: number[]): number {
    let sum = 0;
    for (let i = 0; i < point1.length; i++) {
      const diff = point1[i] - point2[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }
  
  /**
   * Calculate cluster centroids
   */
  private calculateCentroids(data: number[][], labels: number[]): number[][] {
    const uniqueLabels = [...new Set(labels)];
    const centroids: number[][] = [];
    
    for (const label of uniqueLabels) {
      const clusterPoints = data.filter((_, i) => labels[i] === label);
      if (clusterPoints.length > 0) {
        const centroid: number[] = [];
        for (let d = 0; d < data[0].length; d++) {
          centroid.push(mean(clusterPoints.map(p => p[d])));
        }
        centroids.push(centroid);
      }
    }
    
    return centroids;
  }
  
  /**
   * Calculate silhouette score
   */
  private calculateSilhouetteScore(data: number[][], labels: number[]): { score: number; perCluster: number[] } {
    const uniqueLabels = [...new Set(labels)];
    const n = data.length;
    
    if (uniqueLabels.length <= 1) {
      return { score: 0, perCluster: [] };
    }
    
    let totalScore = 0;
    const clusterScores: number[] = [];
    
    for (const clusterLabel of uniqueLabels) {
      const clusterPoints = data.filter((_, i) => labels[i] === clusterLabel);
      const clusterIndices = labels.map((label, i) => label === clusterLabel ? i : -1).filter(i => i !== -1);
      
      if (clusterPoints.length <= 1) {
        clusterScores.push(0);
        continue;
      }
      
      let clusterScore = 0;
      
      for (const pointIndex of clusterIndices) {
        const point = data[pointIndex];
        
        // Calculate a(i): mean distance to points in same cluster
        const sameClusterDistances = clusterIndices
          .filter(i => i !== pointIndex)
          .map(i => this.euclideanDistance(point, data[i]));
        
        const a = mean(sameClusterDistances);
        
        // Calculate b(i): mean distance to points in nearest other cluster
        let minB = Infinity;
        
        for (const otherLabel of uniqueLabels) {
          if (otherLabel === clusterLabel) continue;
          
          const otherClusterIndices = labels.map((label, i) => label === otherLabel ? i : -1).filter(i => i !== -1);
          if (otherClusterIndices.length > 0) {
            const otherClusterDistances = otherClusterIndices.map(i => this.euclideanDistance(point, data[i]));
            const b = mean(otherClusterDistances);
            minB = Math.min(minB, b);
          }
        }
        
        if (minB !== Infinity) {
          const s = (minB - a) / Math.max(a, minB);
          clusterScore += s;
          totalScore += s;
        }
      }
      
      clusterScores.push(clusterScore / clusterIndices.length);
    }
    
    return { score: totalScore / n, perCluster: clusterScores };
  }
  
  /**
   * Calculate Calinski-Harabasz Index (Variance Ratio Criterion)
   */
  private calculateCalinskiHarabaszIndex(data: number[][], labels: number[]): number {
    const uniqueLabels = [...new Set(labels)];
    const n = data.length;
    const k = uniqueLabels.length;
    
    if (k <= 1 || k >= n) {
      return 0;
    }
    
    // Calculate overall centroid
    const overallCentroid: number[] = [];
    for (let d = 0; d < data[0].length; d++) {
      overallCentroid.push(mean(data.map(p => p[d])));
    }
    
    // Calculate cluster centroids
    const centroids = this.calculateCentroids(data, labels);
    
    // Calculate between-cluster sum of squares (BCSS)
    let bcss = 0;
    for (let c = 0; c < centroids.length; c++) {
      const clusterSize = labels.filter(label => label === uniqueLabels[c]).length;
      const distance = this.euclideanDistance(centroids[c], overallCentroid);
      bcss += clusterSize * distance * distance;
    }
    
    // Calculate within-cluster sum of squares (WCSS)
    let wcss = 0;
    for (let i = 0; i < n; i++) {
      const clusterIndex = uniqueLabels.indexOf(labels[i]);
      if (clusterIndex !== -1) {
        const distance = this.euclideanDistance(data[i], centroids[clusterIndex]);
        wcss += distance * distance;
      }
    }
    
    // CH Index = (BCSS / (k-1)) / (WCSS / (n-k))
    return (bcss / (k - 1)) / (wcss / (n - k));
  }
  
  /**
   * Calculate Davies-Bouldin Index
   */
  private calculateDaviesBouldinIndex(data: number[][], labels: number[]): number {
    const uniqueLabels = [...new Set(labels)];
    const k = uniqueLabels.length;
    
    if (k <= 1) {
      return 0;
    }
    
    const centroids = this.calculateCentroids(data, labels);
    
    // Calculate average within-cluster distances
    const avgWithinClusterDistances: number[] = [];
    
    for (let c = 0; c < k; c++) {
      const clusterPoints = data.filter((_, i) => labels[i] === uniqueLabels[c]);
      if (clusterPoints.length <= 1) {
        avgWithinClusterDistances.push(0);
        continue;
      }
      
      let totalDistance = 0;
      let count = 0;
      
      for (const point of clusterPoints) {
        const distance = this.euclideanDistance(point, centroids[c]);
        totalDistance += distance;
        count++;
      }
      
      avgWithinClusterDistances.push(totalDistance / count);
    }
    
    // Calculate DB index
    let dbSum = 0;
    
    for (let i = 0; i < k; i++) {
      let maxRatio = 0;
      
      for (let j = 0; j < k; j++) {
        if (i !== j) {
          const centroidDistance = this.euclideanDistance(centroids[i], centroids[j]);
          if (centroidDistance > 0) {
            const ratio = (avgWithinClusterDistances[i] + avgWithinClusterDistances[j]) / centroidDistance;
            maxRatio = Math.max(maxRatio, ratio);
          }
        }
      }
      
      dbSum += maxRatio;
    }
    
    return dbSum / k;
  }
  
  /**
   * Calculate inertia (within-cluster sum of squares)
   */
  private calculateInertia(data: number[][], labels: number[]): number {
    const centroids = this.calculateCentroids(data, labels);
    const uniqueLabels = [...new Set(labels)];
    
    let inertia = 0;
    
    for (let i = 0; i < data.length; i++) {
      const clusterIndex = uniqueLabels.indexOf(labels[i]);
      if (clusterIndex !== -1) {
        const distance = this.euclideanDistance(data[i], centroids[clusterIndex]);
        inertia += distance * distance;
      }
    }
    
    return inertia;
  }
  
  /**
   * Calculate Dunn Index
   */
  private calculateDunnIndex(data: number[][], labels: number[]): number {
    const uniqueLabels = [...new Set(labels)];
    const k = uniqueLabels.length;
    
    if (k <= 1) {
      return 0;
    }
    
    // Calculate minimum inter-cluster distance
    let minInterClusterDistance = Infinity;
    
    for (let i = 0; i < k; i++) {
      for (let j = i + 1; j < k; j++) {
        const cluster1Points = data.filter((_, idx) => labels[idx] === uniqueLabels[i]);
        const cluster2Points = data.filter((_, idx) => labels[idx] === uniqueLabels[j]);
        
        let minDistance = Infinity;
        for (const p1 of cluster1Points) {
          for (const p2 of cluster2Points) {
            const distance = this.euclideanDistance(p1, p2);
            minDistance = Math.min(minDistance, distance);
          }
        }
        
        minInterClusterDistance = Math.min(minInterClusterDistance, minDistance);
      }
    }
    
    // Calculate maximum intra-cluster distance
    let maxIntraClusterDistance = 0;
    
    for (let i = 0; i < k; i++) {
      const clusterPoints = data.filter((_, idx) => labels[idx] === uniqueLabels[i]);
      
      let maxDistance = 0;
      for (let p = 0; p < clusterPoints.length; p++) {
        for (let q = p + 1; q < clusterPoints.length; q++) {
          const distance = this.euclideanDistance(clusterPoints[p], clusterPoints[q]);
          maxDistance = Math.max(maxDistance, distance);
        }
      }
      
      maxIntraClusterDistance = Math.max(maxIntraClusterDistance, maxDistance);
    }
    
    return maxIntraClusterDistance > 0 ? minInterClusterDistance / maxIntraClusterDistance : 0;
  }
  
  /**
   * Calculate Adjusted Rand Index
   */
  private calculateAdjustedRandIndex(trueLabels: number[], predLabels: number[]): number {
    const n = trueLabels.length;
    
    // Create contingency table
    const trueUnique = [...new Set(trueLabels)];
    const predUnique = [...new Set(predLabels)];
    
    const contingencyTable: number[][] = [];
    for (let i = 0; i < trueUnique.length; i++) {
      contingencyTable[i] = new Array(predUnique.length).fill(0);
    }
    
    for (let i = 0; i < n; i++) {
      const trueIdx = trueUnique.indexOf(trueLabels[i]);
      const predIdx = predUnique.indexOf(predLabels[i]);
      contingencyTable[trueIdx][predIdx]++;
    }
    
    // Calculate marginal sums
    const trueSums = contingencyTable.map(row => sum(row));
    const predSums: number[] = new Array(predUnique.length).fill(0);
    for (let j = 0; j < predUnique.length; j++) {
      for (let i = 0; i < trueUnique.length; i++) {
        predSums[j] += contingencyTable[i][j];
      }
    }
    
    // Calculate ARI components
    let sumCij2 = 0;
    for (let i = 0; i < trueUnique.length; i++) {
      for (let j = 0; j < predUnique.length; j++) {
        const nij = contingencyTable[i][j];
        sumCij2 += nij * (nij - 1) / 2;
      }
    }
    
    let sumAi2 = 0;
    for (const ai of trueSums) {
      sumAi2 += ai * (ai - 1) / 2;
    }
    
    let sumBj2 = 0;
    for (const bj of predSums) {
      sumBj2 += bj * (bj - 1) / 2;
    }
    
    const expectedIndex = (sumAi2 * sumBj2) / (n * (n - 1) / 2);
    const maxIndex = (sumAi2 + sumBj2) / 2;
    
    if (maxIndex - expectedIndex === 0) {
      return 0;
    }
    
    return (sumCij2 - expectedIndex) / (maxIndex - expectedIndex);
  }
  
  /**
   * Calculate Normalized Mutual Information
   */
  private calculateNormalizedMutualInfo(trueLabels: number[], predLabels: number[]): number {
    const n = trueLabels.length;
    
    // Create contingency table
    const trueUnique = [...new Set(trueLabels)];
    const predUnique = [...new Set(predLabels)];
    
    const contingencyTable: number[][] = [];
    for (let i = 0; i < trueUnique.length; i++) {
      contingencyTable[i] = new Array(predUnique.length).fill(0);
    }
    
    for (let i = 0; i < n; i++) {
      const trueIdx = trueUnique.indexOf(trueLabels[i]);
      const predIdx = predUnique.indexOf(predLabels[i]);
      contingencyTable[trueIdx][predIdx]++;
    }
    
    // Calculate marginal probabilities
    const trueSums = contingencyTable.map(row => sum(row));
    const predSums: number[] = new Array(predUnique.length).fill(0);
    for (let j = 0; j < predUnique.length; j++) {
      for (let i = 0; i < trueUnique.length; i++) {
        predSums[j] += contingencyTable[i][j];
      }
    }
    
    // Calculate mutual information
    let mi = 0;
    for (let i = 0; i < trueUnique.length; i++) {
      for (let j = 0; j < predUnique.length; j++) {
        const nij = contingencyTable[i][j];
        if (nij > 0) {
          const pij = nij / n;
          const pi = trueSums[i] / n;
          const pj = predSums[j] / n;
          mi += pij * Math.log(pij / (pi * pj));
        }
      }
    }
    
    // Calculate entropies
    let hTrue = 0;
    for (const ai of trueSums) {
      if (ai > 0) {
        const pi = ai / n;
        hTrue -= pi * Math.log(pi);
      }
    }
    
    let hPred = 0;
    for (const bj of predSums) {
      if (bj > 0) {
        const pj = bj / n;
        hPred -= pj * Math.log(pj);
      }
    }
    
    // Normalized MI
    return hTrue + hPred > 0 ? (2 * mi) / (hTrue + hPred) : 0;
  }
  
  /**
   * Calculate Fowlkes-Mallows Index
   */
  private calculateFowlkesMallowsIndex(trueLabels: number[], predLabels: number[]): number {
    const n = trueLabels.length;
    
    // Count pairs
    let tp = 0; // True positives: same cluster in both
    let fp = 0; // False positives: same in pred, different in true
    let fn = 0; // False negatives: different in pred, same in true
    
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const sameTrue = trueLabels[i] === trueLabels[j];
        const samePred = predLabels[i] === predLabels[j];
        
        if (sameTrue && samePred) {
          tp++;
        } else if (!sameTrue && samePred) {
          fp++;
        } else if (sameTrue && !samePred) {
          fn++;
        }
      }
    }
    
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    
    return precision + recall > 0 ? Math.sqrt(precision * recall) : 0;
  }
  
  /**
   * Calculate homogeneity and completeness
   */
  private calculateHomogeneityCompleteness(trueLabels: number[], predLabels: number[]): { homogeneity: number; completeness: number; vMeasure: number } {
    const n = trueLabels.length;
    
    // Create contingency table
    const trueUnique = [...new Set(trueLabels)];
    const predUnique = [...new Set(predLabels)];
    
    const contingencyTable: number[][] = [];
    for (let i = 0; i < trueUnique.length; i++) {
      contingencyTable[i] = new Array(predUnique.length).fill(0);
    }
    
    for (let i = 0; i < n; i++) {
      const trueIdx = trueUnique.indexOf(trueLabels[i]);
      const predIdx = predUnique.indexOf(predLabels[i]);
      contingencyTable[trueIdx][predIdx]++;
    }
    
    // Calculate marginal sums
    const trueSums = contingencyTable.map(row => sum(row));
    const predSums: number[] = new Array(predUnique.length).fill(0);
    for (let j = 0; j < predUnique.length; j++) {
      for (let i = 0; i < trueUnique.length; i++) {
        predSums[j] += contingencyTable[i][j];
      }
    }
    
    // Calculate conditional entropies
    let hTrueGivenPred = 0;
    for (let j = 0; j < predUnique.length; j++) {
      if (predSums[j] > 0) {
        let entropy = 0;
        for (let i = 0; i < trueUnique.length; i++) {
          if (contingencyTable[i][j] > 0) {
            const p = contingencyTable[i][j] / predSums[j];
            entropy -= p * Math.log(p);
          }
        }
        hTrueGivenPred += (predSums[j] / n) * entropy;
      }
    }
    
    let hPredGivenTrue = 0;
    for (let i = 0; i < trueUnique.length; i++) {
      if (trueSums[i] > 0) {
        let entropy = 0;
        for (let j = 0; j < predUnique.length; j++) {
          if (contingencyTable[i][j] > 0) {
            const p = contingencyTable[i][j] / trueSums[i];
            entropy -= p * Math.log(p);
          }
        }
        hPredGivenTrue += (trueSums[i] / n) * entropy;
      }
    }
    
    // Calculate entropies
    let hTrue = 0;
    for (const ai of trueSums) {
      if (ai > 0) {
        const pi = ai / n;
        hTrue -= pi * Math.log(pi);
      }
    }
    
    let hPred = 0;
    for (const bj of predSums) {
      if (bj > 0) {
        const pj = bj / n;
        hPred -= pj * Math.log(pj);
      }
    }
    
    // Calculate metrics
    const homogeneity = hTrue > 0 ? 1 - hTrueGivenPred / hTrue : 1;
    const completeness = hPred > 0 ? 1 - hPredGivenTrue / hPred : 1;
    const vMeasure = homogeneity + completeness > 0 ? (2 * homogeneity * completeness) / (homogeneity + completeness) : 0;
    
    return { homogeneity, completeness, vMeasure };
  }
  
  /**
   * Evaluate clustering results
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    throw new Error('ClusterEvaluation does not support fit(). Use evaluateClustering() instead.');
  }
  
  /**
   * Not applicable for evaluation
   */
  predict(X: number[][]): number[] {
    throw new Error('ClusterEvaluation does not support predict(). Use evaluateClustering() instead.');
  }
  
  /**
   * Evaluate clustering results
   */
  async evaluateClustering(data: number[][], labels: number[], trueLabels?: number[]): Promise<EvaluationResults> {
    // Validate input
    this.validateInput(data);
    
    if (labels.length !== data.length) {
      throw new Error('Labels length must match data length');
    }
    
    if (trueLabels && trueLabels.length !== data.length) {
      throw new Error('True labels length must match data length');
    }
    
    // Store clustering result
    this.clusteringResult = {
      labels: [...labels],
      data: data.map(row => [...row]),
      trueLabels: trueLabels ? [...trueLabels] : undefined
    };
    
    // Calculate internal metrics
    const silhouetteResult = this.calculateSilhouetteScore(data, labels);
    const calinskiHarabaszIndex = this.calculateCalinskiHarabaszIndex(data, labels);
    const daviesBouldinIndex = this.calculateDaviesBouldinIndex(data, labels);
    const inertia = this.calculateInertia(data, labels);
    const dunn = this.calculateDunnIndex(data, labels);
    
    const internal: InternalMetrics = {
      silhouetteScore: silhouetteResult.score,
      calinskiHarabaszIndex,
      daviesBouldinIndex,
      inertia,
      dunn,
      silhouettePerCluster: silhouetteResult.perCluster
    };
    
    // Calculate external metrics if true labels provided
    let external: ExternalMetrics | undefined;
    if (trueLabels) {
      const adjustedRandIndex = this.calculateAdjustedRandIndex(trueLabels, labels);
      const normalizedMutualInfo = this.calculateNormalizedMutualInfo(trueLabels, labels);
      const fowlkesMallowsIndex = this.calculateFowlkesMallowsIndex(trueLabels, labels);
      const { homogeneity, completeness, vMeasure } = this.calculateHomogeneityCompleteness(trueLabels, labels);
      
      // Calculate accuracy (for perfect matching)
      let accuracy: number | undefined;
      if (new Set(trueLabels).size === new Set(labels).size) {
        const matches = labels.filter((label, i) => label === trueLabels[i]).length;
        accuracy = matches / labels.length;
      }
      
      external = {
        adjustedRandIndex,
        normalizedMutualInfo,
        fowlkesMallowsIndex,
        homogeneity,
        completeness,
        vMeasure,
        accuracy
      };
    }
    
    // Generate summary
    const summary: string[] = [];
    const numClusters = new Set(labels).size;
    
    summary.push(`Found ${numClusters} clusters in ${data.length} data points`);
    
    // Internal metrics summary
    if (internal.silhouetteScore > 0.7) {
      summary.push('Excellent cluster separation (Silhouette > 0.7)');
    } else if (internal.silhouetteScore > 0.5) {
      summary.push('Good cluster separation (Silhouette > 0.5)');
    } else if (internal.silhouetteScore > 0.25) {
      summary.push('Weak cluster separation (Silhouette > 0.25)');
    } else {
      summary.push('Poor cluster separation (Silhouette ≤ 0.25)');
    }
    
    if (internal.daviesBouldinIndex < 1.0) {
      summary.push('Good cluster compactness (Davies-Bouldin < 1.0)');
    } else {
      summary.push('Poor cluster compactness (Davies-Bouldin ≥ 1.0)');
    }
    
    // External metrics summary
    if (external) {
      if (external.adjustedRandIndex > 0.8) {
        summary.push('Excellent agreement with ground truth (ARI > 0.8)');
      } else if (external.adjustedRandIndex > 0.6) {
        summary.push('Good agreement with ground truth (ARI > 0.6)');
      } else if (external.adjustedRandIndex > 0.3) {
        summary.push('Moderate agreement with ground truth (ARI > 0.3)');
      } else {
        summary.push('Poor agreement with ground truth (ARI ≤ 0.3)');
      }
    }
    
    this.evaluationResults = {
      internal,
      external,
      summary
    };
    
    this.trained = true;
    
    return this.evaluationResults;
  }
  
  /**
   * Get visualization data for plotting
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained || !this.clusteringResult) {
      return [];
    }
    
    const visualizations: VisualizationData[] = [];
    const { data, labels } = this.clusteringResult;
    
    // Clustering visualization
    if (data[0].length >= 2) {
      const clusterData: any = {};
      const uniqueLabels = [...new Set(labels)];
      const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];
      
      for (let c = 0; c < uniqueLabels.length; c++) {
        const clusterPoints = data.filter((_, i) => labels[i] === uniqueLabels[c]);
        
        if (clusterPoints.length > 0) {
          const x = clusterPoints.map(p => p[0]);
          const y = clusterPoints.map(p => p[1]);
          
          clusterData[`cluster${c}`] = {
            x,
            y,
            mode: 'markers',
            name: `Cluster ${uniqueLabels[c]}`,
            marker: { 
              color: colors[c % colors.length], 
              size: 8,
              opacity: 0.8
            }
          };
        }
      }
      
      visualizations.push({
        type: 'scatter2d',
        data: clusterData
      });
    }
    
    // Metrics comparison chart
    if (this.evaluationResults) {
      const { internal, external } = this.evaluationResults;
      
      const metricsData = {
        x: ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin', 'Dunn'],
        y: [
          internal.silhouetteScore,
          internal.calinskiHarabaszIndex / 100, // Normalize for display
          1 / (internal.daviesBouldinIndex + 1), // Invert and normalize (lower is better)
          internal.dunn
        ],
        type: 'bar',
        name: 'Internal Metrics',
        marker: { color: '#3b82f6' }
      };
      
      visualizations.push({
        type: 'bar',
        data: metricsData
      });
      
      if (external) {
        const externalData = {
          x: ['ARI', 'NMI', 'FMI', 'V-Measure'],
          y: [
            external.adjustedRandIndex,
            external.normalizedMutualInfo,
            external.fowlkesMallowsIndex,
            external.vMeasure
          ],
          type: 'bar',
          name: 'External Metrics',
          marker: { color: '#ef4444' }
        };
        
        visualizations.push({
          type: 'bar',
          data: externalData
        });
      }
    }
    
    return visualizations;
  }
  
  /**
   * Calculate performance metrics
   */
  getMetrics(): Metrics {
    if (!this.trained || !this.evaluationResults) {
      return {};
    }
    
    const { internal, external, summary } = this.evaluationResults;
    
    return {
      ...internal,
      ...external,
      summary,
      numClusters: new Set(this.clusteringResult?.labels || []).size,
      numPoints: this.clusteringResult?.data.length || 0
    };
  }
  
  /**
   * Get evaluation results
   */
  getEvaluationResults(): EvaluationResults | null {
    return this.evaluationResults;
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.evaluationResults = null;
    this.clusteringResult = null;
  }
  
  /**
   * Get model parameters
   */
  getParameters() {
    return {
      evaluationResults: this.evaluationResults,
      clusteringResult: this.clusteringResult
    };
  }
}