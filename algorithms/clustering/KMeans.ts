/**
 * K-Means Clustering Algorithm
 * 
 * Implements k-means clustering with k-means++ initialization.
 * Supports iterative assignment and update steps with convergence detection.
 * 
 * Requirements: 2.4, 4.6
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics, TrainingHistory } from '~/types/algorithm';
import { mean, sum } from '~/utils/math';

export interface KMeansParams {
  k: number;
  maxIterations: number;
  tolerance: number;
  initialization: 'random' | 'kmeans++';
  randomSeed?: number;
}

interface ClusterHistory {
  iteration: number;
  centroids: number[][];
  assignments: number[];
  inertia: number;
}

export class KMeans extends BaseAlgorithm {
  name = 'K-Means Clustering';
  
  // Model parameters
  private centroids: number[][] = [];
  private assignments: number[] = [];
  
  // Training data (stored for visualization)
  private X_train: number[][] = [];
  
  // Hyperparameters
  private k: number;
  private maxIterations: number;
  private tolerance: number;
  private initialization: 'random' | 'kmeans++';
  private randomSeed?: number;
  
  // Training history with cluster assignments
  private clusterHistory: ClusterHistory[] = [];
  protected override trainingHistory: TrainingHistory & { inertia: number[] } = {
    epochs: [],
    loss: [],
    inertia: []
  };
  
  // Random number generator state
  private rng: number = 0;
  
  constructor(params: KMeansParams = {
    k: 3,
    maxIterations: 100,
    tolerance: 1e-4,
    initialization: 'kmeans++'
  }) {
    super();
    this.k = params.k;
    this.maxIterations = params.maxIterations;
    this.tolerance = params.tolerance;
    this.initialization = params.initialization;
    this.randomSeed = params.randomSeed;
  }
  
  /**
   * Simple seeded random number generator
   */
  private random(): number {
    if (this.randomSeed !== undefined) {
      this.rng = (this.rng * 9301 + 49297) % 233280;
      return this.rng / 233280;
    }
    return Math.random();
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
   * Initialize centroids using random selection
   */
  private initializeRandomCentroids(X: number[][]): number[][] {
    const n = X.length;
    const centroids: number[][] = [];
    const selectedIndices = new Set<number>();
    
    while (centroids.length < this.k) {
      const idx = Math.floor(this.random() * n);
      if (!selectedIndices.has(idx)) {
        selectedIndices.add(idx);
        centroids.push([...X[idx]]);
      }
    }
    
    return centroids;
  }
  
  /**
   * Initialize centroids using k-means++ algorithm
   * This provides better initial centroids by spreading them out
   */
  private initializeKMeansPlusPlusCentroids(X: number[][]): number[][] {
    const n = X.length;
    const centroids: number[][] = [];
    
    // Choose first centroid randomly
    const firstIdx = Math.floor(this.random() * n);
    centroids.push([...X[firstIdx]]);
    
    // Choose remaining centroids
    for (let c = 1; c < this.k; c++) {
      const distances: number[] = [];
      let totalDistance = 0;
      
      // Calculate distance to nearest centroid for each point
      for (let i = 0; i < n; i++) {
        let minDist = Infinity;
        for (const centroid of centroids) {
          const dist = this.euclideanDistance(X[i], centroid);
          minDist = Math.min(minDist, dist);
        }
        const distSquared = minDist * minDist;
        distances.push(distSquared);
        totalDistance += distSquared;
      }
      
      // Choose next centroid with probability proportional to distance squared
      let target = this.random() * totalDistance;
      let cumSum = 0;
      
      for (let i = 0; i < n; i++) {
        cumSum += distances[i];
        if (cumSum >= target) {
          centroids.push([...X[i]]);
          break;
        }
      }
    }
    
    return centroids;
  }
  
  /**
   * Assign each point to the nearest centroid
   */
  private assignClusters(X: number[][], centroids: number[][]): number[] {
    const assignments: number[] = [];
    
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
      
      assignments.push(bestCluster);
    }
    
    return assignments;
  }
  
  /**
   * Update centroids based on current assignments
   */
  private updateCentroids(X: number[][], assignments: number[]): number[][] {
    const n = X[0].length; // number of features
    const newCentroids: number[][] = [];
    
    for (let c = 0; c < this.k; c++) {
      // Find all points assigned to this cluster
      const clusterPoints = X.filter((_, i) => assignments[i] === c);
      
      if (clusterPoints.length === 0) {
        // If cluster is empty, reinitialize randomly
        const randomIdx = Math.floor(this.random() * X.length);
        newCentroids.push([...X[randomIdx]]);
      } else {
        // Calculate mean of all points in cluster
        const centroid: number[] = [];
        for (let j = 0; j < n; j++) {
          const values = clusterPoints.map(point => point[j]);
          centroid.push(mean(values));
        }
        newCentroids.push(centroid);
      }
    }
    
    return newCentroids;
  }
  
  /**
   * Calculate inertia (within-cluster sum of squares)
   */
  private calculateInertia(X: number[][], centroids: number[][], assignments: number[]): number {
    let inertia = 0;
    
    for (let i = 0; i < X.length; i++) {
      const clusterIdx = assignments[i];
      const dist = this.euclideanDistance(X[i], centroids[clusterIdx]);
      inertia += dist * dist;
    }
    
    return inertia;
  }
  
  /**
   * Calculate centroid shift (for convergence detection)
   */
  private calculateCentroidShift(oldCentroids: number[][], newCentroids: number[][]): number {
    let maxShift = 0;
    
    for (let c = 0; c < this.k; c++) {
      const shift = this.euclideanDistance(oldCentroids[c], newCentroids[c]);
      maxShift = Math.max(maxShift, shift);
    }
    
    return maxShift;
  }
  
  /**
   * Train the k-means model
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    // Validate input
    this.validateInput(X);
    
    if (this.k > X.length) {
      throw new Error(`Number of clusters (${this.k}) cannot exceed number of samples (${X.length})`);
    }
    
    if (this.k < 1) {
      throw new Error('Number of clusters must be at least 1');
    }
    
    // Store training data
    this.X_train = X.map(row => [...row]);
    
    // Initialize random seed if provided
    if (this.randomSeed !== undefined) {
      this.rng = this.randomSeed;
    }
    
    // Initialize centroids
    if (this.initialization === 'kmeans++') {
      this.centroids = this.initializeKMeansPlusPlusCentroids(X);
    } else {
      this.centroids = this.initializeRandomCentroids(X);
    }
    
    // Reset history
    this.trainingHistory = {
      epochs: [],
      loss: [],
      inertia: []
    };
    this.clusterHistory = [];
    
    // Initial assignment
    this.assignments = this.assignClusters(X, this.centroids);
    let inertia = this.calculateInertia(X, this.centroids, this.assignments);
    
    // Store initial state
    this.clusterHistory.push({
      iteration: 0,
      centroids: this.centroids.map(c => [...c]),
      assignments: [...this.assignments],
      inertia
    });
    
    this.trainingHistory.epochs.push(0);
    this.trainingHistory.loss.push(inertia);
    this.trainingHistory.inertia.push(inertia);
    
    // Iterative optimization
    let converged = false;
    
    for (let iteration = 1; iteration <= this.maxIterations; iteration++) {
      // Update centroids
      const oldCentroids = this.centroids.map(c => [...c]);
      this.centroids = this.updateCentroids(X, this.assignments);
      
      // Assign clusters
      this.assignments = this.assignClusters(X, this.centroids);
      
      // Calculate inertia
      inertia = this.calculateInertia(X, this.centroids, this.assignments);
      
      // Store history
      this.trainingHistory.epochs.push(iteration);
      this.trainingHistory.loss.push(inertia);
      this.trainingHistory.inertia.push(inertia);
      
      this.clusterHistory.push({
        iteration,
        centroids: this.centroids.map(c => [...c]),
        assignments: [...this.assignments],
        inertia
      });
      
      // Check convergence
      const centroidShift = this.calculateCentroidShift(oldCentroids, this.centroids);
      if (centroidShift < this.tolerance) {
        converged = true;
        break;
      }
      
      // Allow UI to update periodically
      if (iteration % 5 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    this.trained = true;
  }
  
  /**
   * Predict cluster assignments for new data
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    this.validateInput(X, this.centroids[0].length);
    
    return this.assignClusters(X, this.centroids);
  }
  
  /**
   * Calculate silhouette score for clustering quality
   * Measures how similar points are to their own cluster vs other clusters
   */
  private calculateSilhouetteScore(): number {
    const n = this.X_train.length;
    
    if (this.k === 1 || n <= this.k) {
      return 0; // Silhouette score not meaningful for single cluster or too few points
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
        continue; // Skip if point is alone in cluster
      }
      
      const a = mean(sameClusterPoints.map(p => this.euclideanDistance(point, p)));
      
      // Calculate b(i): mean distance to points in nearest other cluster
      let minB = Infinity;
      
      for (let c = 0; c < this.k; c++) {
        if (c === cluster) continue;
        
        const otherClusterPoints = this.X_train.filter((_, j) => 
          this.assignments[j] === c
        );
        
        if (otherClusterPoints.length > 0) {
          const b = mean(otherClusterPoints.map(p => this.euclideanDistance(point, p)));
          minB = Math.min(minB, b);
        }
      }
      
      // Silhouette coefficient for this point
      const s = (minB - a) / Math.max(a, minB);
      totalScore += s;
    }
    
    return totalScore / n;
  }
  
  /**
   * Get visualization data for plotting
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained) {
      return [];
    }
    
    const visualizations: VisualizationData[] = [];
    
    // Scatter plot with cluster colors (for 2D or 3D data)
    const numFeatures = this.X_train[0].length;
    
    if (numFeatures === 2) {
      // 2D scatter plot
      const clusterData: any = {};
      const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];
      
      for (let c = 0; c < this.k; c++) {
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
      
      // Add centroids
      clusterData.centroids = {
        x: this.centroids.map(c => c[0]),
        y: this.centroids.map(c => c[1]),
        mode: 'markers',
        name: 'Centroids',
        marker: {
          color: '#000000',
          size: 15,
          symbol: 'x',
          line: { width: 2, color: '#ffffff' }
        }
      };
      
      visualizations.push({
        type: 'scatter2d',
        data: clusterData
      });
    } else if (numFeatures >= 3) {
      // 3D scatter plot (use first 3 features)
      const clusterData: any = {};
      const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];
      
      for (let c = 0; c < this.k; c++) {
        const clusterPoints = this.X_train.filter((_, i) => this.assignments[i] === c);
        const x = clusterPoints.map(p => p[0]);
        const y = clusterPoints.map(p => p[1]);
        const z = clusterPoints.map(p => p[2]);
        
        clusterData[`cluster${c}`] = {
          x,
          y,
          z,
          mode: 'markers',
          type: 'scatter3d',
          name: `Cluster ${c}`,
          marker: { 
            color: colors[c % colors.length], 
            size: 5,
            opacity: 0.7
          }
        };
      }
      
      // Add centroids
      clusterData.centroids = {
        x: this.centroids.map(c => c[0]),
        y: this.centroids.map(c => c[1]),
        z: this.centroids.map(c => c[2]),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Centroids',
        marker: {
          color: '#000000',
          size: 10,
          symbol: 'x'
        }
      };
      
      visualizations.push({
        type: 'scatter3d',
        data: clusterData
      });
    }
    
    // Inertia curve over iterations
    visualizations.push({
      type: 'line',
      data: {
        x: this.trainingHistory.epochs,
        y: this.trainingHistory.inertia,
        mode: 'lines+markers',
        name: 'Inertia',
        line: { color: '#10b981', width: 2 },
        marker: { size: 4 }
      }
    });
    
    return visualizations;
  }
  
  /**
   * Calculate performance metrics
   */
  getMetrics(): Metrics {
    if (!this.trained) {
      return {};
    }
    
    const finalInertia = this.trainingHistory.inertia[this.trainingHistory.inertia.length - 1];
    const silhouetteScore = this.calculateSilhouetteScore();
    
    // Calculate cluster sizes
    const clusterSizes: number[] = new Array(this.k).fill(0);
    for (const assignment of this.assignments) {
      clusterSizes[assignment]++;
    }
    
    return {
      inertia: finalInertia,
      silhouetteScore,
      iterations: this.trainingHistory.epochs.length - 1,
      clusterSizes
    };
  }
  
  /**
   * Get cluster history for step-by-step visualization
   */
  getClusterHistory(): ClusterHistory[] {
    return this.clusterHistory;
  }
  
  /**
   * Get current centroids
   */
  getCentroids(): number[][] {
    return this.centroids.map(c => [...c]);
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
    this.centroids = [];
    this.assignments = [];
    this.X_train = [];
    this.clusterHistory = [];
    this.trainingHistory = {
      epochs: [],
      loss: [],
      inertia: []
    };
  }
  
  /**
   * Get model parameters
   */
  getParameters() {
    return {
      k: this.k,
      centroids: this.centroids.map(c => [...c]),
      assignments: [...this.assignments],
      initialization: this.initialization
    };
  }
}
