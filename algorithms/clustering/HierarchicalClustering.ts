/**
 * Hierarchical Clustering Algorithm
 * 
 * Implements agglomerative hierarchical clustering with multiple linkage methods.
 * Supports single, complete, average, and Ward's linkage criteria.
 * 
 * Requirements: 7.1, 7.2, 7.3, 7.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics } from '~/types/algorithm';
import { mean } from '~/utils/math';

export interface HierarchicalClusteringParams {
  nClusters: number;
  linkage: 'single' | 'complete' | 'average' | 'ward';
}

interface MergeStep {
  cluster1: number;
  cluster2: number;
  distance: number;
  newCluster: number;
}

export class HierarchicalClustering extends BaseAlgorithm {
  name = 'Hierarchical Clustering';
  
  // Model parameters
  private assignments: number[] = [];
  private clusters: Map<number, number[]> = new Map();
  
  // Training data (stored for visualization)
  private X_train: number[][] = [];
  
  // Hyperparameters
  private nClusters: number;
  private linkage: 'single' | 'complete' | 'average' | 'ward';
  
  // Dendrogram data
  private mergeHistory: MergeStep[] = [];
  private nextClusterId: number = 0;
  
  constructor(params: HierarchicalClusteringParams = {
    nClusters: 3,
    linkage: 'ward'
  }) {
    super();
    this.nClusters = params.nClusters;
    this.linkage = params.linkage;
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
   * Calculate centroid of a cluster
   */
  private calculateCentroid(points: number[][]): number[] {
    if (points.length === 0) return [];
    
    const numFeatures = points[0].length;
    const centroid: number[] = [];
    
    for (let j = 0; j < numFeatures; j++) {
      const values = points.map(p => p[j]);
      centroid.push(mean(values));
    }
    
    return centroid;
  }
  
  /**
   * Calculate distance between two clusters based on linkage method
   */
  private calculateClusterDistance(
    cluster1Points: number[][],
    cluster2Points: number[][]
  ): number {
    if (this.linkage === 'single') {
      // Minimum distance between any two points
      let minDist = Infinity;
      for (const p1 of cluster1Points) {
        for (const p2 of cluster2Points) {
          const dist = this.euclideanDistance(p1, p2);
          minDist = Math.min(minDist, dist);
        }
      }
      return minDist;
    } else if (this.linkage === 'complete') {
      // Maximum distance between any two points
      let maxDist = -Infinity;
      for (const p1 of cluster1Points) {
        for (const p2 of cluster2Points) {
          const dist = this.euclideanDistance(p1, p2);
          maxDist = Math.max(maxDist, dist);
        }
      }
      return maxDist;
    } else if (this.linkage === 'average') {
      // Average distance between all pairs of points
      let totalDist = 0;
      let count = 0;
      for (const p1 of cluster1Points) {
        for (const p2 of cluster2Points) {
          totalDist += this.euclideanDistance(p1, p2);
          count++;
        }
      }
      return totalDist / count;
    } else if (this.linkage === 'ward') {
      // Ward's method: minimize within-cluster variance
      const centroid1 = this.calculateCentroid(cluster1Points);
      const centroid2 = this.calculateCentroid(cluster2Points);
      const mergedPoints = [...cluster1Points, ...cluster2Points];
      const mergedCentroid = this.calculateCentroid(mergedPoints);
      
      // Calculate increase in variance
      let variance1 = 0;
      for (const p of cluster1Points) {
        variance1 += Math.pow(this.euclideanDistance(p, centroid1), 2);
      }
      
      let variance2 = 0;
      for (const p of cluster2Points) {
        variance2 += Math.pow(this.euclideanDistance(p, centroid2), 2);
      }
      
      let mergedVariance = 0;
      for (const p of mergedPoints) {
        mergedVariance += Math.pow(this.euclideanDistance(p, mergedCentroid), 2);
      }
      
      return mergedVariance - variance1 - variance2;
    }
    
    return 0;
  }
  
  /**
   * Train the hierarchical clustering model
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    // Validate input
    this.validateInput(X);
    
    if (this.nClusters > X.length) {
      throw new Error(`Number of clusters (${this.nClusters}) cannot exceed number of samples (${X.length})`);
    }
    
    if (this.nClusters < 1) {
      throw new Error('Number of clusters must be at least 1');
    }
    
    // Store training data
    this.X_train = X.map(row => [...row]);
    
    // Initialize: each point is its own cluster
    this.clusters.clear();
    this.mergeHistory = [];
    this.nextClusterId = X.length;
    
    for (let i = 0; i < X.length; i++) {
      this.clusters.set(i, [i]);
    }
    
    // Build distance matrix
    const n = X.length;
    const distanceMatrix: Map<string, number> = new Map();
    
    const getKey = (i: number, j: number) => {
      return i < j ? `${i},${j}` : `${j},${i}`;
    };
    
    // Calculate initial distances
    const clusterIds = Array.from(this.clusters.keys());
    for (let i = 0; i < clusterIds.length; i++) {
      for (let j = i + 1; j < clusterIds.length; j++) {
        const id1 = clusterIds[i];
        const id2 = clusterIds[j];
        const points1 = this.clusters.get(id1)!.map(idx => X[idx]);
        const points2 = this.clusters.get(id2)!.map(idx => X[idx]);
        const dist = this.calculateClusterDistance(points1, points2);
        distanceMatrix.set(getKey(id1, id2), dist);
      }
    }
    
    // Agglomerative clustering: merge until we have desired number of clusters
    while (this.clusters.size > this.nClusters) {
      // Find closest pair of clusters
      let minDist = Infinity;
      let bestPair: [number, number] = [-1, -1];
      
      const currentClusterIds = Array.from(this.clusters.keys());
      for (let i = 0; i < currentClusterIds.length; i++) {
        for (let j = i + 1; j < currentClusterIds.length; j++) {
          const id1 = currentClusterIds[i];
          const id2 = currentClusterIds[j];
          const key = getKey(id1, id2);
          const dist = distanceMatrix.get(key);
          
          if (dist !== undefined && dist < minDist) {
            minDist = dist;
            bestPair = [id1, id2];
          }
        }
      }
      
      if (bestPair[0] === -1) break;
      
      // Merge the two closest clusters
      const [id1, id2] = bestPair;
      const cluster1 = this.clusters.get(id1)!;
      const cluster2 = this.clusters.get(id2)!;
      const mergedCluster = [...cluster1, ...cluster2];
      
      // Create new cluster
      const newId = this.nextClusterId++;
      this.clusters.set(newId, mergedCluster);
      
      // Record merge
      this.mergeHistory.push({
        cluster1: id1,
        cluster2: id2,
        distance: minDist,
        newCluster: newId
      });
      
      // Remove old clusters
      this.clusters.delete(id1);
      this.clusters.delete(id2);
      
      // Update distance matrix
      // Remove distances involving old clusters
      for (const otherId of this.clusters.keys()) {
        if (otherId === newId) continue;
        distanceMatrix.delete(getKey(id1, otherId));
        distanceMatrix.delete(getKey(id2, otherId));
      }
      
      // Calculate distances from new cluster to all others
      for (const otherId of this.clusters.keys()) {
        if (otherId === newId) continue;
        
        const otherCluster = this.clusters.get(otherId)!;
        const points1 = mergedCluster.map(idx => X[idx]);
        const points2 = otherCluster.map(idx => X[idx]);
        const dist = this.calculateClusterDistance(points1, points2);
        distanceMatrix.set(getKey(newId, otherId), dist);
      }
      
      // Allow UI to update periodically
      if (this.mergeHistory.length % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    // Create final assignments
    this.assignments = new Array(X.length).fill(-1);
    let clusterLabel = 0;
    
    for (const pointIndices of this.clusters.values()) {
      for (const idx of pointIndices) {
        this.assignments[idx] = clusterLabel;
      }
      clusterLabel++;
    }
    
    this.trained = true;
  }
  
  /**
   * Predict cluster assignments for new data
   * Note: For hierarchical clustering, we assign to nearest existing cluster centroid
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    this.validateInput(X, this.X_train[0].length);
    
    // Calculate centroids of existing clusters
    const centroids: number[][] = [];
    for (let c = 0; c < this.nClusters; c++) {
      const clusterPoints = this.X_train.filter((_, i) => this.assignments[i] === c);
      if (clusterPoints.length > 0) {
        centroids.push(this.calculateCentroid(clusterPoints));
      }
    }
    
    // Assign new points to nearest centroid
    const predictions: number[] = [];
    for (const point of X) {
      let minDist = Infinity;
      let bestCluster = 0;
      
      for (let c = 0; c < centroids.length; c++) {
        const dist = this.euclideanDistance(point, centroids[c]);
        if (dist < minDist) {
          minDist = dist;
          bestCluster = c;
        }
      }
      
      predictions.push(bestCluster);
    }
    
    return predictions;
  }
  
  /**
   * Calculate silhouette score for clustering quality
   */
  private calculateSilhouetteScore(): number {
    const n = this.X_train.length;
    
    if (this.nClusters === 1 || n <= this.nClusters) {
      return 0;
    }
    
    let totalScore = 0;
    
    for (let i = 0; i < n; i++) {
      const point = this.X_train[i];
      const cluster = this.assignments[i];
      
      // Calculate a(i): mean distance to points in same cluster
      const sameClusterPoints = this.X_train.filter((_, j) => 
        this.assignments[j] === cluster && i !== j
      );
      
      if (sameClusterPoints.length === 0) {
        continue;
      }
      
      const a = mean(sameClusterPoints.map(p => this.euclideanDistance(point, p)));
      
      // Calculate b(i): mean distance to points in nearest other cluster
      let minB = Infinity;
      
      for (let c = 0; c < this.nClusters; c++) {
        if (c === cluster) continue;
        
        const otherClusterPoints = this.X_train.filter((_, j) => 
          this.assignments[j] === c
        );
        
        if (otherClusterPoints.length > 0) {
          const b = mean(otherClusterPoints.map(p => this.euclideanDistance(point, p)));
          minB = Math.min(minB, b);
        }
      }
      
      const s = (minB - a) / Math.max(a, minB);
      totalScore += s;
    }
    
    return totalScore / n;
  }
  
  /**
   * Calculate Davies-Bouldin score (lower is better)
   */
  private calculateDaviesBouldinScore(): number {
    if (this.nClusters === 1) return 0;
    
    // Calculate centroids and within-cluster scatter
    const centroids: number[][] = [];
    const scatters: number[] = [];
    
    for (let c = 0; c < this.nClusters; c++) {
      const clusterPoints = this.X_train.filter((_, i) => this.assignments[i] === c);
      if (clusterPoints.length === 0) continue;
      
      const centroid = this.calculateCentroid(clusterPoints);
      centroids.push(centroid);
      
      // Within-cluster scatter
      const scatter = mean(clusterPoints.map(p => this.euclideanDistance(p, centroid)));
      scatters.push(scatter);
    }
    
    // Calculate DB index
    let dbSum = 0;
    
    for (let i = 0; i < centroids.length; i++) {
      let maxRatio = -Infinity;
      
      for (let j = 0; j < centroids.length; j++) {
        if (i === j) continue;
        
        const centroidDist = this.euclideanDistance(centroids[i], centroids[j]);
        const ratio = (scatters[i] + scatters[j]) / centroidDist;
        maxRatio = Math.max(maxRatio, ratio);
      }
      
      dbSum += maxRatio;
    }
    
    return dbSum / centroids.length;
  }
  
  /**
   * Calculate Calinski-Harabasz score (higher is better)
   */
  private calculateCalinskiHarabaszScore(): number {
    if (this.nClusters === 1 || this.X_train.length <= this.nClusters) return 0;
    
    const n = this.X_train.length;
    const overallCentroid = this.calculateCentroid(this.X_train);
    
    // Between-cluster variance
    let betweenVariance = 0;
    for (let c = 0; c < this.nClusters; c++) {
      const clusterPoints = this.X_train.filter((_, i) => this.assignments[i] === c);
      if (clusterPoints.length === 0) continue;
      
      const centroid = this.calculateCentroid(clusterPoints);
      const dist = this.euclideanDistance(centroid, overallCentroid);
      betweenVariance += clusterPoints.length * dist * dist;
    }
    
    // Within-cluster variance
    let withinVariance = 0;
    for (let c = 0; c < this.nClusters; c++) {
      const clusterPoints = this.X_train.filter((_, i) => this.assignments[i] === c);
      if (clusterPoints.length === 0) continue;
      
      const centroid = this.calculateCentroid(clusterPoints);
      for (const point of clusterPoints) {
        const dist = this.euclideanDistance(point, centroid);
        withinVariance += dist * dist;
      }
    }
    
    // CH score
    const ch = (betweenVariance / (this.nClusters - 1)) / (withinVariance / (n - this.nClusters));
    return ch;
  }
  
  /**
   * Get visualization data for plotting
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained) {
      return [];
    }
    
    const visualizations: VisualizationData[] = [];
    const numFeatures = this.X_train[0].length;
    
    if (numFeatures >= 2) {
      // 2D scatter plot
      const clusterData: any = {};
      const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];
      
      for (let c = 0; c < this.nClusters; c++) {
        const clusterPoints = this.X_train.filter((_, i) => this.assignments[i] === c);
        const x = clusterPoints.map(p => p[0]);
        const y = clusterPoints.map(p => p[1]);
        
        clusterData[`cluster${c}`] = {
          x,
          y,
          mode: 'markers',
          name: `Cluster ${c}`,
          marker: { 
            color: colors[c % colors.length], 
            size: 8,
            opacity: 0.7
          }
        };
      }
      
      visualizations.push({
        type: 'scatter2d',
        data: clusterData
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
    
    const silhouetteScore = this.calculateSilhouetteScore();
    const daviesBouldinScore = this.calculateDaviesBouldinScore();
    const calinskiHarabaszScore = this.calculateCalinskiHarabaszScore();
    
    // Calculate cluster sizes
    const clusterSizes: number[] = new Array(this.nClusters).fill(0);
    for (const assignment of this.assignments) {
      clusterSizes[assignment]++;
    }
    
    return {
      silhouetteScore,
      daviesBouldinScore,
      calinskiHarabaszScore,
      clusterSizes,
      nMerges: this.mergeHistory.length
    };
  }
  
  /**
   * Get merge history for dendrogram visualization
   */
  getMergeHistory(): MergeStep[] {
    return this.mergeHistory;
  }
  
  /**
   * Get current assignments
   */
  getAssignments(): number[] {
    return [...this.assignments];
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.assignments = [];
    this.clusters.clear();
    this.X_train = [];
    this.mergeHistory = [];
    this.nextClusterId = 0;
  }
  
  /**
   * Get model parameters
   */
  getParameters() {
    return {
      nClusters: this.nClusters,
      linkage: this.linkage,
      assignments: [...this.assignments]
    };
  }
}
