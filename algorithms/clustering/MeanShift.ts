/**
 * Mean Shift Clustering Algorithm
 * 
 * Implements mean shift clustering that finds modes in the data density
 * and groups points that converge to the same mode.
 * 
 * Requirements: 7.1, 7.2, 7.3, 7.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics, TrainingHistory } from '~/types/algorithm';
import { mean } from '~/utils/math';

export interface MeanShiftParams {
  bandwidth: number;
  maxIterations: number;
  tolerance: number;
  kernel: 'gaussian' | 'flat';
  binSeeding: boolean;
}

interface MeanShiftPoint {
  index: number;
  original: number[];
  current: number[];
  converged: boolean;
  iterations: number;
  mode: number[];
  cluster: number;
}

interface ModeInfo {
  mode: number[];
  points: number[];
  density: number;
}

export class MeanShift extends BaseAlgorithm {
  name = 'Mean Shift Clustering';
  
  // Model parameters
  private modes: number[][] = [];
  private assignments: number[] = [];
  private points: MeanShiftPoint[] = [];
  
  // Training data (stored for visualization)
  private X_train: number[][] = [];
  
  // Hyperparameters
  private bandwidth: number;
  private maxIterations: number;
  private tolerance: number;
  private kernel: 'gaussian' | 'flat';
  private binSeeding: boolean;
  
  // Results
  private numClusters: number = 0;
  private modeInfo: ModeInfo[] = [];
  
  constructor(params: MeanShiftParams = {
    bandwidth: 1.0,
    maxIterations: 300,
    tolerance: 1e-3,
    kernel: 'gaussian',
    binSeeding: false
  }) {
    super();
    this.bandwidth = params.bandwidth;
    this.maxIterations = params.maxIterations;
    this.tolerance = params.tolerance;
    this.kernel = params.kernel;
    this.binSeeding = params.binSeeding;
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
   * Calculate kernel weight based on distance
   */
  private kernelWeight(distance: number): number {
    if (this.kernel === 'gaussian') {
      // Gaussian kernel: exp(-0.5 * (d/h)^2)
      const normalized = distance / this.bandwidth;
      return Math.exp(-0.5 * normalized * normalized);
    } else {
      // Flat (uniform) kernel: 1 if d <= h, 0 otherwise
      return distance <= this.bandwidth ? 1.0 : 0.0;
    }
  }
  
  /**
   * Calculate mean shift vector for a point
   */
  private calculateMeanShift(point: number[], X: number[][]): { shift: number[]; density: number } {
    const d = point.length;
    const shift: number[] = new Array(d).fill(0);
    let totalWeight = 0;
    
    // Calculate weighted mean of neighbors
    for (const neighbor of X) {
      const distance = this.euclideanDistance(point, neighbor);
      const weight = this.kernelWeight(distance);
      
      if (weight > 0) {
        totalWeight += weight;
        for (let i = 0; i < d; i++) {
          shift[i] += weight * neighbor[i];
        }
      }
    }
    
    // Normalize by total weight
    if (totalWeight > 0) {
      for (let i = 0; i < d; i++) {
        shift[i] = shift[i] / totalWeight - point[i];
      }
    }
    
    return { shift, density: totalWeight };
  }
  
  /**
   * Perform mean shift for a single point
   */
  private meanShiftPoint(startPoint: number[], X: number[][]): MeanShiftPoint {
    const point: MeanShiftPoint = {
      index: -1,
      original: [...startPoint],
      current: [...startPoint],
      converged: false,
      iterations: 0,
      mode: [],
      cluster: -1
    };
    
    for (let iter = 0; iter < this.maxIterations; iter++) {
      const { shift, density } = this.calculateMeanShift(point.current, X);
      
      // Calculate shift magnitude
      const shiftMagnitude = Math.sqrt(shift.reduce((sum, s) => sum + s * s, 0));
      
      // Update current position
      for (let i = 0; i < point.current.length; i++) {
        point.current[i] += shift[i];
      }
      
      point.iterations = iter + 1;
      
      // Check convergence
      if (shiftMagnitude < this.tolerance) {
        point.converged = true;
        break;
      }
    }
    
    point.mode = [...point.current];
    return point;
  }
  
  /**
   * Generate seed points for mean shift
   */
  private generateSeeds(X: number[][]): number[][] {
    if (!this.binSeeding) {
      // Use all data points as seeds
      return X.map(point => [...point]);
    }
    
    // Bin seeding: create a grid of seed points
    const d = X[0].length;
    const seeds: number[][] = [];
    
    // Find data bounds
    const mins: number[] = new Array(d).fill(Infinity);
    const maxs: number[] = new Array(d).fill(-Infinity);
    
    for (const point of X) {
      for (let i = 0; i < d; i++) {
        mins[i] = Math.min(mins[i], point[i]);
        maxs[i] = Math.max(maxs[i], point[i]);
      }
    }
    
    // Create grid based on bandwidth
    const gridSize = this.bandwidth / 2;
    const gridCounts: number[] = [];
    
    for (let i = 0; i < d; i++) {
      const range = maxs[i] - mins[i];
      gridCounts[i] = Math.max(1, Math.ceil(range / gridSize));
    }
    
    // Generate grid points
    const generateGridPoints = (dim: number, current: number[]): void => {
      if (dim === d) {
        seeds.push([...current]);
        return;
      }
      
      for (let i = 0; i < gridCounts[dim]; i++) {
        const value = mins[dim] + (i + 0.5) * (maxs[dim] - mins[dim]) / gridCounts[dim];
        current[dim] = value;
        generateGridPoints(dim + 1, current);
      }
    };
    
    generateGridPoints(0, new Array(d));
    
    // Filter seeds that have nearby data points
    const filteredSeeds = seeds.filter(seed => {
      return X.some(point => this.euclideanDistance(seed, point) <= this.bandwidth);
    });
    
    return filteredSeeds.length > 0 ? filteredSeeds : X.map(point => [...point]);
  }
  
  /**
   * Merge nearby modes
   */
  private mergeModes(modes: number[][]): number[][] {
    if (modes.length <= 1) return modes;
    
    const merged: number[][] = [];
    const used: boolean[] = new Array(modes.length).fill(false);
    
    for (let i = 0; i < modes.length; i++) {
      if (used[i]) continue;
      
      const cluster: number[][] = [modes[i]];
      used[i] = true;
      
      // Find nearby modes to merge
      for (let j = i + 1; j < modes.length; j++) {
        if (used[j]) continue;
        
        const distance = this.euclideanDistance(modes[i], modes[j]);
        if (distance < this.bandwidth / 2) {
          cluster.push(modes[j]);
          used[j] = true;
        }
      }
      
      // Calculate centroid of merged modes
      const d = modes[0].length;
      const centroid: number[] = new Array(d).fill(0);
      
      for (const mode of cluster) {
        for (let k = 0; k < d; k++) {
          centroid[k] += mode[k];
        }
      }
      
      for (let k = 0; k < d; k++) {
        centroid[k] /= cluster.length;
      }
      
      merged.push(centroid);
    }
    
    return merged;
  }
  
  /**
   * Assign points to clusters based on nearest mode
   */
  private assignClusters(): void {
    this.assignments = [];
    
    for (const point of this.points) {
      let bestCluster = 0;
      let minDistance = Infinity;
      
      for (let c = 0; c < this.modes.length; c++) {
        const distance = this.euclideanDistance(point.mode, this.modes[c]);
        if (distance < minDistance) {
          minDistance = distance;
          bestCluster = c;
        }
      }
      
      point.cluster = bestCluster;
      this.assignments.push(bestCluster);
    }
  }
  
  /**
   * Calculate mode information
   */
  private calculateModeInfo(): void {
    this.modeInfo = [];
    
    for (let c = 0; c < this.modes.length; c++) {
      const mode = this.modes[c];
      const clusterPoints = this.points
        .filter(p => p.cluster === c)
        .map(p => p.index);
      
      // Calculate density at mode
      const { density } = this.calculateMeanShift(mode, this.X_train);
      
      this.modeInfo.push({
        mode: [...mode],
        points: clusterPoints,
        density
      });
    }
  }
  
  /**
   * Train the Mean Shift model
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    // Validate input
    this.validateInput(X);
    
    if (this.bandwidth <= 0) {
      throw new Error('Bandwidth must be positive');
    }
    
    // Store training data
    this.X_train = X.map(row => [...row]);
    
    // Generate seed points
    const seeds = this.generateSeeds(X);
    
    // Reset results
    this.points = [];
    this.modes = [];
    this.assignments = [];
    this.modeInfo = [];
    
    // Perform mean shift for each seed
    const allModes: number[][] = [];
    
    for (let i = 0; i < seeds.length; i++) {
      const seed = seeds[i];
      const shiftedPoint = this.meanShiftPoint(seed, X);
      
      // Find original point index if seed is a data point
      if (i < X.length) {
        shiftedPoint.index = i;
      } else {
        // For bin seeds, find nearest data point
        let nearestIndex = 0;
        let minDistance = Infinity;
        
        for (let j = 0; j < X.length; j++) {
          const distance = this.euclideanDistance(seed, X[j]);
          if (distance < minDistance) {
            minDistance = distance;
            nearestIndex = j;
          }
        }
        
        shiftedPoint.index = nearestIndex;
      }
      
      this.points.push(shiftedPoint);
      allModes.push(shiftedPoint.mode);
      
      // Allow UI to update periodically
      if (i % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    // Merge nearby modes
    this.modes = this.mergeModes(allModes);
    this.numClusters = this.modes.length;
    
    // Assign points to clusters
    this.assignClusters();
    
    // Calculate mode information
    this.calculateModeInfo();
    
    this.trained = true;
  }
  
  /**
   * Predict cluster assignments for new data
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    this.validateInput(X, this.X_train[0].length);
    
    const predictions: number[] = [];
    
    for (const point of X) {
      // Perform mean shift for the new point
      const shiftedPoint = this.meanShiftPoint(point, this.X_train);
      
      // Find nearest mode
      let bestCluster = 0;
      let minDistance = Infinity;
      
      for (let c = 0; c < this.modes.length; c++) {
        const distance = this.euclideanDistance(shiftedPoint.mode, this.modes[c]);
        if (distance < minDistance) {
          minDistance = distance;
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
    if (this.numClusters <= 1) {
      return 0;
    }
    
    const n = this.X_train.length;
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
      
      for (let c = 0; c < this.numClusters; c++) {
        if (c === cluster) continue;
        
        const otherClusterPoints = this.X_train.filter((_, j) => 
          this.assignments[j] === c
        );
        
        if (otherClusterPoints.length > 0) {
          const b = mean(otherClusterPoints.map(p => this.euclideanDistance(point, p)));
          minB = Math.min(minB, b);
        }
      }
      
      if (minB !== Infinity) {
        const s = (minB - a) / Math.max(a, minB);
        totalScore += s;
      }
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
      
      // Add modes
      if (numFeatures === 2) {
        clusterData.modes = {
          x: this.modes.map(m => m[0]),
          y: this.modes.map(m => m[1]),
          mode: 'markers',
          name: 'Modes',
          marker: {
            color: '#000000',
            size: 15,
            symbol: 'star',
            line: { width: 2, color: '#ffffff' }
          }
        };
      } else {
        clusterData.modes = {
          x: this.modes.map(m => m[0]),
          y: this.modes.map(m => m[1]),
          z: this.modes.map(m => m[2]),
          mode: 'markers',
          type: 'scatter3d',
          name: 'Modes',
          marker: {
            color: '#000000',
            size: 10,
            symbol: 'diamond'
          }
        };
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
    const clusterSizes: number[] = new Array(this.numClusters).fill(0);
    for (const assignment of this.assignments) {
      clusterSizes[assignment]++;
    }
    
    // Calculate convergence statistics
    const convergedPoints = this.points.filter(p => p.converged).length;
    const convergenceRate = convergedPoints / this.points.length;
    const avgIterations = mean(this.points.map(p => p.iterations));
    
    // Calculate mode densities
    const modeDensities = this.modeInfo.map(info => info.density);
    
    return {
      numClusters: this.numClusters,
      silhouetteScore,
      clusterSizes,
      convergenceRate,
      avgIterations,
      modeDensities,
      totalPoints: this.X_train.length,
      bandwidth: this.bandwidth
    };
  }
  
  /**
   * Get modes (cluster centers)
   */
  getModes(): number[][] {
    return this.modes.map(mode => [...mode]);
  }
  
  /**
   * Get mode information
   */
  getModeInfo(): ModeInfo[] {
    return this.modeInfo.map(info => ({
      mode: [...info.mode],
      points: [...info.points],
      density: info.density
    }));
  }
  
  /**
   * Get cluster assignments
   */
  getAssignments(): number[] {
    return [...this.assignments];
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.modes = [];
    this.assignments = [];
    this.points = [];
    this.X_train = [];
    this.modeInfo = [];
    this.numClusters = 0;
  }
  
  /**
   * Get model parameters
   */
  getParameters() {
    return {
      bandwidth: this.bandwidth,
      kernel: this.kernel,
      binSeeding: this.binSeeding,
      numClusters: this.numClusters,
      modes: this.getModes(),
      assignments: [...this.assignments]
    };
  }
}