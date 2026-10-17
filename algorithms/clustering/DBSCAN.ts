/**
 * DBSCAN (Density-Based Spatial Clustering of Applications with Noise) Algorithm
 * 
 * Implements density-based clustering that can find arbitrarily shaped clusters
 * and identify outliers as noise points.
 * 
 * Requirements: 7.1, 7.2, 7.3, 7.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics, TrainingHistory } from '~/types/algorithm';
import { mean } from '~/utils/math';

export interface DBSCANParams {
  eps: number;
  minPts: number;
  metric: 'euclidean' | 'manhattan';
}

interface DBSCANPoint {
  index: number;
  features: number[];
  cluster: number;
  visited: boolean;
  isCore: boolean;
  neighbors: number[];
}

export class DBSCAN extends BaseAlgorithm {
  name = 'DBSCAN Clustering';
  
  // Model parameters
  private clusters: number[][] = [];
  private assignments: number[] = [];
  private noisePoints: number[] = [];
  
  // Training data (stored for visualization)
  private X_train: number[][] = [];
  private points: DBSCANPoint[] = [];
  
  // Hyperparameters
  private eps: number;
  private minPts: number;
  private metric: 'euclidean' | 'manhattan';
  
  // Results
  private numClusters: number = 0;
  
  constructor(params: DBSCANParams = {
    eps: 0.5,
    minPts: 5,
    metric: 'euclidean'
  }) {
    super();
    this.eps = params.eps;
    this.minPts = params.minPts;
    this.metric = params.metric;
  }
  
  /**
   * Calculate distance between two points
   */
  private calculateDistance(point1: number[], point2: number[]): number {
    if (this.metric === 'euclidean') {
      let sum = 0;
      for (let i = 0; i < point1.length; i++) {
        const diff = point1[i] - point2[i];
        sum += diff * diff;
      }
      return Math.sqrt(sum);
    } else if (this.metric === 'manhattan') {
      let sum = 0;
      for (let i = 0; i < point1.length; i++) {
        sum += Math.abs(point1[i] - point2[i]);
      }
      return sum;
    }
    return 0;
  }
  
  /**
   * Find all neighbors within eps distance
   */
  private findNeighbors(pointIndex: number, X: number[][]): number[] {
    const neighbors: number[] = [];
    const point = X[pointIndex];
    
    for (let i = 0; i < X.length; i++) {
      if (i !== pointIndex) {
        const distance = this.calculateDistance(point, X[i]);
        if (distance <= this.eps) {
          neighbors.push(i);
        }
      }
    }
    
    return neighbors;
  }
  
  /**
   * Expand cluster from a core point
   */
  private expandCluster(
    pointIndex: number,
    neighbors: number[],
    clusterId: number,
    X: number[][]
  ): void {
    // Add point to cluster
    this.points[pointIndex].cluster = clusterId;
    
    // Process all neighbors
    let i = 0;
    while (i < neighbors.length) {
      const neighborIndex = neighbors[i];
      const neighbor = this.points[neighborIndex];
      
      // Mark as visited if not already
      if (!neighbor.visited) {
        neighbor.visited = true;
        
        // Find neighbors of this neighbor
        const neighborNeighbors = this.findNeighbors(neighborIndex, X);
        neighbor.neighbors = neighborNeighbors;
        
        // If it's a core point, add its neighbors to the list
        if (neighborNeighbors.length >= this.minPts) {
          neighbor.isCore = true;
          // Add new neighbors to the list (union operation)
          for (const newNeighbor of neighborNeighbors) {
            if (!neighbors.includes(newNeighbor)) {
              neighbors.push(newNeighbor);
            }
          }
        }
      }
      
      // If neighbor is not assigned to any cluster, assign it to this cluster
      if (neighbor.cluster === -1) {
        neighbor.cluster = clusterId;
      }
      
      i++;
    }
  }
  
  /**
   * Train the DBSCAN model
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    // Validate input
    this.validateInput(X);
    
    if (this.eps <= 0) {
      throw new Error('eps must be positive');
    }
    
    if (this.minPts < 1) {
      throw new Error('minPts must be at least 1');
    }
    
    // Store training data
    this.X_train = X.map(row => [...row]);
    
    // Initialize points
    this.points = X.map((features, index) => ({
      index,
      features: [...features],
      cluster: -1, // -1 means unassigned, -2 means noise
      visited: false,
      isCore: false,
      neighbors: []
    }));
    
    // Reset results
    this.clusters = [];
    this.assignments = [];
    this.noisePoints = [];
    this.numClusters = 0;
    
    let clusterId = 0;
    
    // Process each point
    for (let i = 0; i < this.points.length; i++) {
      const point = this.points[i];
      
      if (point.visited) {
        continue;
      }
      
      point.visited = true;
      
      // Find neighbors
      const neighbors = this.findNeighbors(i, X);
      point.neighbors = neighbors;
      
      // Check if it's a core point
      if (neighbors.length >= this.minPts) {
        point.isCore = true;
        // Start a new cluster
        this.expandCluster(i, neighbors, clusterId, X);
        clusterId++;
      } else {
        // Mark as noise (for now)
        point.cluster = -2;
      }
      
      // Allow UI to update periodically
      if (i % 50 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    this.numClusters = clusterId;
    
    // Post-process: border points
    // Points that are not core but are within eps of a core point
    // should be assigned to that cluster (not noise)
    for (const point of this.points) {
      if (point.cluster === -2) { // Currently marked as noise
        // Check if it's within eps of any core point
        for (const otherPoint of this.points) {
          if (otherPoint.isCore && otherPoint.cluster >= 0) {
            const distance = this.calculateDistance(point.features, otherPoint.features);
            if (distance <= this.eps) {
              point.cluster = otherPoint.cluster;
              break;
            }
          }
        }
      }
    }
    
    // Build final assignments and cluster information
    this.assignments = this.points.map(p => p.cluster);
    
    // Group points by cluster
    for (let c = 0; c < this.numClusters; c++) {
      const clusterPoints = this.points
        .filter(p => p.cluster === c)
        .map(p => p.index);
      this.clusters.push(clusterPoints);
    }
    
    // Collect noise points
    this.noisePoints = this.points
      .filter(p => p.cluster === -2)
      .map(p => p.index);
    
    this.trained = true;
  }
  
  /**
   * Predict cluster assignments for new data
   * Note: DBSCAN doesn't naturally support prediction on new data
   * This implementation assigns new points to the nearest cluster or marks as noise
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    this.validateInput(X, this.X_train[0].length);
    
    const predictions: number[] = [];
    
    for (const newPoint of X) {
      let bestCluster = -2; // Default to noise
      let minDistance = Infinity;
      
      // Find the closest core point
      for (const point of this.points) {
        if (point.isCore) {
          const distance = this.calculateDistance(newPoint, point.features);
          if (distance <= this.eps && distance < minDistance) {
            minDistance = distance;
            bestCluster = point.cluster;
          }
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
    if (this.numClusters <= 1) {
      return 0; // Silhouette score not meaningful for single cluster
    }
    
    const validPoints = this.points.filter(p => p.cluster >= 0);
    if (validPoints.length === 0) {
      return 0;
    }
    
    let totalScore = 0;
    
    for (const point of validPoints) {
      const cluster = point.cluster;
      
      // Calculate a(i): mean distance to points in same cluster
      const sameClusterPoints = validPoints.filter(p => p.cluster === cluster && p.index !== point.index);
      
      if (sameClusterPoints.length === 0) {
        continue; // Skip if point is alone in cluster
      }
      
      const a = mean(sameClusterPoints.map(p => 
        this.calculateDistance(point.features, p.features)
      ));
      
      // Calculate b(i): mean distance to points in nearest other cluster
      let minB = Infinity;
      
      for (let c = 0; c < this.numClusters; c++) {
        if (c === cluster) continue;
        
        const otherClusterPoints = validPoints.filter(p => p.cluster === c);
        
        if (otherClusterPoints.length > 0) {
          const b = mean(otherClusterPoints.map(p => 
            this.calculateDistance(point.features, p.features)
          ));
          minB = Math.min(minB, b);
        }
      }
      
      if (minB !== Infinity) {
        // Silhouette coefficient for this point
        const s = (minB - a) / Math.max(a, minB);
        totalScore += s;
      }
    }
    
    return totalScore / validPoints.length;
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
      // 2D or 3D scatter plot
      const clusterData: any = {};
      const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];
      
      // Add cluster points
      for (let c = 0; c < this.numClusters; c++) {
        const clusterPoints = this.X_train.filter((_, i) => this.assignments[i] === c);
        
        if (clusterPoints.length > 0) {
          const x = clusterPoints.map(p => p[0]);
          const y = clusterPoints.map(p => p[1]);
          
          if (numFeatures === 2) {
            clusterData[`cluster${c}`] = {
              x,
              y,
              mode: 'markers',
              name: `Cluster ${c}`,
              marker: { 
                color: colors[c % colors.length], 
                size: 8,
                opacity: 0.8
              }
            };
          } else {
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
                opacity: 0.8
              }
            };
          }
        }
      }
      
      // Add noise points
      const noisePointsData = this.X_train.filter((_, i) => this.assignments[i] === -2);
      if (noisePointsData.length > 0) {
        const x = noisePointsData.map(p => p[0]);
        const y = noisePointsData.map(p => p[1]);
        
        if (numFeatures === 2) {
          clusterData.noise = {
            x,
            y,
            mode: 'markers',
            name: 'Noise',
            marker: {
              color: '#6b7280',
              size: 6,
              opacity: 0.6,
              symbol: 'x'
            }
          };
        } else {
          const z = noisePointsData.map(p => p[2]);
          clusterData.noise = {
            x,
            y,
            z,
            mode: 'markers',
            type: 'scatter3d',
            name: 'Noise',
            marker: {
              color: '#6b7280',
              size: 4,
              opacity: 0.6,
              symbol: 'x'
            }
          };
        }
      }
      
      visualizations.push({
        type: numFeatures === 2 ? 'scatter2d' : 'scatter3d',
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
    
    // Calculate cluster sizes
    const clusterSizes: number[] = [];
    for (let c = 0; c < this.numClusters; c++) {
      const size = this.assignments.filter(a => a === c).length;
      clusterSizes.push(size);
    }
    
    // Calculate core points ratio
    const corePointsCount = this.points.filter(p => p.isCore).length;
    const corePointsRatio = corePointsCount / this.points.length;
    
    return {
      numClusters: this.numClusters,
      silhouetteScore,
      clusterSizes,
      noisePoints: this.noisePoints.length,
      noiseRatio: this.noisePoints.length / this.X_train.length,
      corePoints: corePointsCount,
      corePointsRatio,
      totalPoints: this.X_train.length
    };
  }
  
  /**
   * Get cluster assignments
   */
  getAssignments(): number[] {
    return [...this.assignments];
  }
  
  /**
   * Get noise points
   */
  getNoisePoints(): number[] {
    return [...this.noisePoints];
  }
  
  /**
   * Get core points
   */
  getCorePoints(): number[] {
    return this.points.filter(p => p.isCore).map(p => p.index);
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.clusters = [];
    this.assignments = [];
    this.noisePoints = [];
    this.X_train = [];
    this.points = [];
    this.numClusters = 0;
  }
  
  /**
   * Get model parameters
   */
  getParameters() {
    return {
      eps: this.eps,
      minPts: this.minPts,
      metric: this.metric,
      numClusters: this.numClusters,
      assignments: [...this.assignments],
      noisePoints: [...this.noisePoints]
    };
  }
}