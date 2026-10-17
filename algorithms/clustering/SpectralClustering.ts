/**
 * Spectral Clustering Algorithm
 * 
 * Implements spectral clustering using graph-based similarity and eigenvalue decomposition.
 * Can find non-convex clusters by working in the spectral embedding space.
 * 
 * Requirements: 7.1, 7.2, 7.3, 7.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics, TrainingHistory } from '~/types/algorithm';
import { mean } from '~/utils/math';

export interface SpectralClusteringParams {
  k: number;
  sigma: number;
  affinity: 'rbf' | 'nearest_neighbors' | 'polynomial';
  nNeighbors: number;
  degree: number;
  gamma: number;
  laplacian: 'unnormalized' | 'rw' | 'symmetric';
  eigenSolver: 'arpack' | 'lobpcg';
  randomSeed?: number;
}

interface EigenResult {
  eigenvalues: number[];
  eigenvectors: number[][];
}

export class SpectralClustering extends BaseAlgorithm {
  name = 'Spectral Clustering';
  
  // Model parameters
  private assignments: number[] = [];
  private embedding: number[][] = [];
  private affinityMatrix: number[][] = [];
  private laplacianMatrix: number[][] = [];
  
  // Training data (stored for visualization)
  private X_train: number[][] = [];
  
  // Hyperparameters
  private k: number;
  private sigma: number;
  private affinity: 'rbf' | 'nearest_neighbors' | 'polynomial';
  private nNeighbors: number;
  private degree: number;
  private gamma: number;
  private laplacian: 'unnormalized' | 'rw' | 'symmetric';
  private eigenSolver: 'arpack' | 'lobpcg';
  private randomSeed?: number;
  
  // Results
  private eigenvalues: number[] = [];
  private eigenvectors: number[][] = [];
  
  // Random number generator state
  private rng: number = 0;
  
  constructor(params: SpectralClusteringParams = {
    k: 3,
    sigma: 1.0,
    affinity: 'rbf',
    nNeighbors: 10,
    degree: 3,
    gamma: 1.0,
    laplacian: 'rw',
    eigenSolver: 'arpack'
  }) {
    super();
    this.k = params.k;
    this.sigma = params.sigma;
    this.affinity = params.affinity;
    this.nNeighbors = params.nNeighbors;
    this.degree = params.degree;
    this.gamma = params.gamma;
    this.laplacian = params.laplacian;
    this.eigenSolver = params.eigenSolver;
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
   * Build affinity matrix based on the chosen affinity type
   */
  private buildAffinityMatrix(X: number[][]): number[][] {
    const n = X.length;
    const affinity: number[][] = [];
    
    // Initialize matrix
    for (let i = 0; i < n; i++) {
      affinity[i] = new Array(n).fill(0);
    }
    
    if (this.affinity === 'rbf') {
      // RBF (Gaussian) kernel: exp(-gamma * ||x - y||^2)
      for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
          if (i === j) {
            affinity[i][j] = 1.0;
          } else {
            const distance = this.euclideanDistance(X[i], X[j]);
            const similarity = Math.exp(-this.gamma * distance * distance);
            affinity[i][j] = similarity;
            affinity[j][i] = similarity;
          }
        }
      }
    } else if (this.affinity === 'nearest_neighbors') {
      // k-nearest neighbors graph
      for (let i = 0; i < n; i++) {
        // Find k nearest neighbors
        const distances: { index: number; distance: number }[] = [];
        
        for (let j = 0; j < n; j++) {
          if (i !== j) {
            const distance = this.euclideanDistance(X[i], X[j]);
            distances.push({ index: j, distance });
          }
        }
        
        // Sort by distance and take k nearest
        distances.sort((a, b) => a.distance - b.distance);
        const neighbors = distances.slice(0, this.nNeighbors);
        
        // Set affinity for neighbors
        for (const neighbor of neighbors) {
          const similarity = Math.exp(-neighbor.distance * neighbor.distance / (2 * this.sigma * this.sigma));
          affinity[i][neighbor.index] = similarity;
          affinity[neighbor.index][i] = similarity; // Make symmetric
        }
      }
    } else if (this.affinity === 'polynomial') {
      // Polynomial kernel: (gamma * <x, y> + 1)^degree
      for (let i = 0; i < n; i++) {
        for (let j = i; j < n; j++) {
          let dotProduct = 0;
          for (let d = 0; d < X[i].length; d++) {
            dotProduct += X[i][d] * X[j][d];
          }
          
          const similarity = Math.pow(this.gamma * dotProduct + 1, this.degree);
          affinity[i][j] = similarity;
          affinity[j][i] = similarity;
        }
      }
    }
    
    return affinity;
  }
  
  /**
   * Build Laplacian matrix from affinity matrix
   */
  private buildLaplacianMatrix(affinity: number[][]): number[][] {
    const n = affinity.length;
    const laplacian: number[][] = [];
    
    // Initialize matrix
    for (let i = 0; i < n; i++) {
      laplacian[i] = new Array(n).fill(0);
    }
    
    // Calculate degree matrix
    const degrees: number[] = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        degrees[i] += affinity[i][j];
      }
    }
    
    if (this.laplacian === 'unnormalized') {
      // L = D - A
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i === j) {
            laplacian[i][j] = degrees[i] - affinity[i][j];
          } else {
            laplacian[i][j] = -affinity[i][j];
          }
        }
      }
    } else if (this.laplacian === 'rw') {
      // L_rw = D^(-1) * L = I - D^(-1) * A
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i === j) {
            laplacian[i][j] = 1.0 - (degrees[i] > 0 ? affinity[i][j] / degrees[i] : 0);
          } else {
            laplacian[i][j] = degrees[i] > 0 ? -affinity[i][j] / degrees[i] : 0;
          }
        }
      }
    } else if (this.laplacian === 'symmetric') {
      // L_sym = D^(-1/2) * L * D^(-1/2) = I - D^(-1/2) * A * D^(-1/2)
      const sqrtDegrees: number[] = degrees.map(d => d > 0 ? Math.sqrt(d) : 0);
      
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i === j) {
            laplacian[i][j] = 1.0;
          } else {
            const normalization = sqrtDegrees[i] * sqrtDegrees[j];
            laplacian[i][j] = normalization > 0 ? -affinity[i][j] / normalization : 0;
          }
        }
      }
    }
    
    return laplacian;
  }
  
  /**
   * Compute eigenvalues and eigenvectors using power iteration (simplified)
   * In practice, would use more sophisticated methods like Lanczos or ARPACK
   */
  private computeEigenDecomposition(matrix: number[][], k: number): EigenResult {
    const n = matrix.length;
    const maxIterations = 1000;
    const tolerance = 1e-10;
    
    const eigenvalues: number[] = [];
    const eigenvectors: number[][] = [];
    
    // For spectral clustering, we need the smallest eigenvalues
    // We'll use inverse iteration to find them
    
    for (let eigenIndex = 0; eigenIndex < k; eigenIndex++) {
      // Initialize random vector
      let vector: number[] = new Array(n);
      for (let i = 0; i < n; i++) {
        vector[i] = this.random() - 0.5;
      }
      
      // Normalize
      let norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
      if (norm > 0) {
        vector = vector.map(v => v / norm);
      }
      
      let eigenvalue = 0;
      
      // Power iteration (simplified - would use inverse iteration for smallest eigenvalues)
      for (let iter = 0; iter < maxIterations; iter++) {
        // Multiply by matrix
        const newVector: number[] = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) {
            newVector[i] += matrix[i][j] * vector[j];
          }
        }
        
        // Orthogonalize against previous eigenvectors
        for (const prevEigenvector of eigenvectors) {
          let dotProduct = 0;
          for (let i = 0; i < n; i++) {
            dotProduct += newVector[i] * prevEigenvector[i];
          }
          
          for (let i = 0; i < n; i++) {
            newVector[i] -= dotProduct * prevEigenvector[i];
          }
        }
        
        // Normalize
        norm = Math.sqrt(newVector.reduce((sum, v) => sum + v * v, 0));
        if (norm < tolerance) {
          break;
        }
        
        const normalizedVector = newVector.map(v => v / norm);
        
        // Calculate eigenvalue (Rayleigh quotient)
        let numerator = 0;
        let denominator = 0;
        
        for (let i = 0; i < n; i++) {
          let matrixVectorProduct = 0;
          for (let j = 0; j < n; j++) {
            matrixVectorProduct += matrix[i][j] * normalizedVector[j];
          }
          numerator += normalizedVector[i] * matrixVectorProduct;
          denominator += normalizedVector[i] * normalizedVector[i];
        }
        
        const newEigenvalue = denominator > 0 ? numerator / denominator : 0;
        
        // Check convergence
        if (Math.abs(newEigenvalue - eigenvalue) < tolerance) {
          break;
        }
        
        eigenvalue = newEigenvalue;
        vector = normalizedVector;
      }
      
      eigenvalues.push(eigenvalue);
      eigenvectors.push([...vector]);
    }
    
    return { eigenvalues, eigenvectors };
  }
  
  /**
   * Perform K-Means clustering on the embedding
   */
  private kMeansEmbedding(embedding: number[][], k: number): number[] {
    const n = embedding.length;
    const d = embedding[0].length;
    
    // Initialize centroids randomly
    const centroids: number[][] = [];
    for (let c = 0; c < k; c++) {
      const centroid: number[] = [];
      for (let j = 0; j < d; j++) {
        centroid.push(this.random() - 0.5);
      }
      centroids.push(centroid);
    }
    
    let assignments: number[] = new Array(n).fill(0);
    const maxIterations = 100;
    
    for (let iter = 0; iter < maxIterations; iter++) {
      const newAssignments: number[] = [];
      
      // Assign points to nearest centroid
      for (let i = 0; i < n; i++) {
        let bestCluster = 0;
        let minDistance = Infinity;
        
        for (let c = 0; c < k; c++) {
          const distance = this.euclideanDistance(embedding[i], centroids[c]);
          if (distance < minDistance) {
            minDistance = distance;
            bestCluster = c;
          }
        }
        
        newAssignments.push(bestCluster);
      }
      
      // Check for convergence
      let changed = false;
      for (let i = 0; i < n; i++) {
        if (assignments[i] !== newAssignments[i]) {
          changed = true;
          break;
        }
      }
      
      assignments = newAssignments;
      
      if (!changed) {
        break;
      }
      
      // Update centroids
      for (let c = 0; c < k; c++) {
        const clusterPoints = embedding.filter((_, i) => assignments[i] === c);
        
        if (clusterPoints.length > 0) {
          for (let j = 0; j < d; j++) {
            centroids[c][j] = mean(clusterPoints.map(p => p[j]));
          }
        }
      }
    }
    
    return assignments;
  }
  
  /**
   * Train the Spectral Clustering model
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
    
    // Step 1: Build affinity matrix
    console.log('Building affinity matrix...');
    this.affinityMatrix = this.buildAffinityMatrix(X);
    
    // Step 2: Build Laplacian matrix
    console.log('Building Laplacian matrix...');
    this.laplacianMatrix = this.buildLaplacianMatrix(this.affinityMatrix);
    
    // Step 3: Compute eigendecomposition
    console.log('Computing eigendecomposition...');
    const eigenResult = this.computeEigenDecomposition(this.laplacianMatrix, this.k);
    this.eigenvalues = eigenResult.eigenvalues;
    this.eigenvectors = eigenResult.eigenvectors;
    
    // Step 4: Create embedding from eigenvectors
    console.log('Creating spectral embedding...');
    this.embedding = [];
    for (let i = 0; i < X.length; i++) {
      const embeddingPoint: number[] = [];
      for (let j = 0; j < this.k; j++) {
        embeddingPoint.push(this.eigenvectors[j][i]);
      }
      this.embedding.push(embeddingPoint);
    }
    
    // Step 5: Normalize embedding rows (for symmetric Laplacian)
    if (this.laplacian === 'symmetric') {
      for (let i = 0; i < this.embedding.length; i++) {
        const norm = Math.sqrt(this.embedding[i].reduce((sum, v) => sum + v * v, 0));
        if (norm > 0) {
          for (let j = 0; j < this.embedding[i].length; j++) {
            this.embedding[i][j] /= norm;
          }
        }
      }
    }
    
    // Step 6: Apply K-Means to the embedding
    console.log('Clustering in spectral space...');
    this.assignments = this.kMeansEmbedding(this.embedding, this.k);
    
    this.trained = true;
  }
  
  /**
   * Predict cluster assignments for new data
   * Note: Spectral clustering doesn't naturally support out-of-sample prediction
   * This is a simplified approach using nearest neighbors in the original space
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    this.validateInput(X, this.X_train[0].length);
    
    const predictions: number[] = [];
    
    for (const newPoint of X) {
      // Find nearest neighbor in training data
      let nearestIndex = 0;
      let minDistance = Infinity;
      
      for (let i = 0; i < this.X_train.length; i++) {
        const distance = this.euclideanDistance(newPoint, this.X_train[i]);
        if (distance < minDistance) {
          minDistance = distance;
          nearestIndex = i;
        }
      }
      
      predictions.push(this.assignments[nearestIndex]);
    }
    
    return predictions;
  }
  
  /**
   * Calculate silhouette score for clustering quality
   */
  private calculateSilhouetteScore(): number {
    if (this.k <= 1) {
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
    
    // Original space clustering
    const numFeatures = this.X_train[0].length;
    
    if (numFeatures >= 2) {
      const clusterData: any = {};
      const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];
      
      // Add cluster points in original space
      for (let c = 0; c < this.k; c++) {
        const clusterPoints = this.X_train.filter((_, i) => this.assignments[i] === c);
        
        if (clusterPoints.length > 0) {
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
    
    // Spectral embedding visualization (if 2D or more)
    if (this.embedding.length > 0 && this.embedding[0].length >= 2) {
      const embeddingData: any = {};
      const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];
      
      for (let c = 0; c < this.k; c++) {
        const clusterEmbeddings = this.embedding.filter((_, i) => this.assignments[i] === c);
        
        if (clusterEmbeddings.length > 0) {
          const x = clusterEmbeddings.map(p => p[0]);
          const y = clusterEmbeddings.map(p => p[1]);
          
          embeddingData[`cluster${c}`] = {
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
        }
      }
      
      visualizations.push({
        type: 'scatter2d',
        data: embeddingData
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
    const clusterSizes: number[] = new Array(this.k).fill(0);
    for (const assignment of this.assignments) {
      clusterSizes[assignment]++;
    }
    
    // Calculate spectral gap (difference between k-th and (k+1)-th eigenvalues)
    let spectralGap = 0;
    if (this.eigenvalues.length > this.k) {
      spectralGap = Math.abs(this.eigenvalues[this.k] - this.eigenvalues[this.k - 1]);
    }
    
    return {
      silhouetteScore,
      clusterSizes,
      eigenvalues: [...this.eigenvalues],
      spectralGap,
      numClusters: this.k,
      affinity: this.affinity,
      laplacian: this.laplacian
    };
  }
  
  /**
   * Get spectral embedding
   */
  getEmbedding(): number[][] {
    return this.embedding.map(row => [...row]);
  }
  
  /**
   * Get affinity matrix
   */
  getAffinityMatrix(): number[][] {
    return this.affinityMatrix.map(row => [...row]);
  }
  
  /**
   * Get Laplacian matrix
   */
  getLaplacianMatrix(): number[][] {
    return this.laplacianMatrix.map(row => [...row]);
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
    this.assignments = [];
    this.embedding = [];
    this.affinityMatrix = [];
    this.laplacianMatrix = [];
    this.X_train = [];
    this.eigenvalues = [];
    this.eigenvectors = [];
  }
  
  /**
   * Get model parameters
   */
  getParameters() {
    return {
      k: this.k,
      sigma: this.sigma,
      affinity: this.affinity,
      laplacian: this.laplacian,
      assignments: [...this.assignments],
      eigenvalues: [...this.eigenvalues],
      embedding: this.getEmbedding()
    };
  }
}