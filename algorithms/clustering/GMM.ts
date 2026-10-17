/**
 * Gaussian Mixture Model (GMM) Algorithm
 * 
 * Implements probabilistic clustering using Expectation-Maximization (EM) algorithm.
 * Models data as a mixture of Gaussian distributions.
 * 
 * Requirements: 7.1, 7.2, 7.3, 7.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics, TrainingHistory } from '~/types/algorithm';
import { mean, sum } from '~/utils/math';

export interface GMMParams {
  k: number;
  maxIterations: number;
  tolerance: number;
  initialization: 'random' | 'kmeans++';
  covarianceType: 'full' | 'diagonal';
  randomSeed?: number;
}

interface GaussianComponent {
  mean: number[];
  covariance: number[][];
  weight: number;
}

interface EMHistory {
  iteration: number;
  components: GaussianComponent[];
  logLikelihood: number;
  responsibilities: number[][];
}

export class GMM extends BaseAlgorithm {
  name = 'Gaussian Mixture Model';
  
  // Model parameters
  private components: GaussianComponent[] = [];
  private responsibilities: number[][] = [];
  
  // Training data (stored for visualization)
  private X_train: number[][] = [];
  
  // Hyperparameters
  private k: number;
  private maxIterations: number;
  private tolerance: number;
  private initialization: 'random' | 'kmeans++';
  private covarianceType: 'full' | 'diagonal';
  private randomSeed?: number;
  
  // Training history
  private emHistory: EMHistory[] = [];
  protected override trainingHistory: TrainingHistory & { logLikelihood: number[] } = {
    epochs: [],
    loss: [],
    logLikelihood: []
  };
  
  // Random number generator state
  private rng: number = 0;
  
  constructor(params: GMMParams = {
    k: 3,
    maxIterations: 100,
    tolerance: 1e-6,
    initialization: 'kmeans++',
    covarianceType: 'full'
  }) {
    super();
    this.k = params.k;
    this.maxIterations = params.maxIterations;
    this.tolerance = params.tolerance;
    this.initialization = params.initialization;
    this.covarianceType = params.covarianceType;
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
   * Calculate multivariate Gaussian probability density
   */
  private gaussianPDF(x: number[], mean: number[], covariance: number[][]): number {
    const d = x.length;
    const diff = x.map((xi, i) => xi - mean[i]);
    
    // Calculate determinant and inverse of covariance matrix
    const det = this.determinant(covariance);
    if (det <= 0) {
      return 1e-10; // Avoid numerical issues
    }
    
    const invCov = this.invertMatrix(covariance);
    
    // Calculate quadratic form: (x-μ)ᵀ Σ⁻¹ (x-μ)
    let quadForm = 0;
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        quadForm += diff[i] * invCov[i][j] * diff[j];
      }
    }
    
    // Calculate PDF
    const normalization = Math.pow(2 * Math.PI, d / 2) * Math.sqrt(det);
    return Math.exp(-0.5 * quadForm) / normalization;
  }
  
  /**
   * Calculate matrix determinant (2x2 or using cofactor expansion)
   */
  private determinant(matrix: number[][]): number {
    const n = matrix.length;
    
    if (n === 1) {
      return matrix[0][0];
    }
    
    if (n === 2) {
      return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }
    
    // For larger matrices, use cofactor expansion (simplified for 2D case)
    // In practice, would use more efficient methods like LU decomposition
    let det = 0;
    for (let i = 0; i < n; i++) {
      const subMatrix = matrix.slice(1).map(row => 
        row.filter((_, j) => j !== i)
      );
      det += Math.pow(-1, i) * matrix[0][i] * this.determinant(subMatrix);
    }
    
    return det;
  }
  
  /**
   * Invert matrix (simplified for 2x2, identity for diagonal)
   */
  private invertMatrix(matrix: number[][]): number[][] {
    const n = matrix.length;
    
    if (n === 1) {
      return [[1 / matrix[0][0]]];
    }
    
    if (n === 2) {
      const det = this.determinant(matrix);
      if (Math.abs(det) < 1e-10) {
        // Singular matrix, return identity
        return [[1, 0], [0, 1]];
      }
      
      return [
        [matrix[1][1] / det, -matrix[0][1] / det],
        [-matrix[1][0] / det, matrix[0][0] / det]
      ];
    }
    
    // For larger matrices or diagonal case, use simplified approach
    const inv: number[][] = [];
    for (let i = 0; i < n; i++) {
      inv[i] = [];
      for (let j = 0; j < n; j++) {
        if (i === j) {
          inv[i][j] = 1 / (matrix[i][j] || 1e-10);
        } else {
          inv[i][j] = 0;
        }
      }
    }
    
    return inv;
  }
  
  /**
   * Initialize components using K-Means++ or random
   */
  private initializeComponents(X: number[][]): void {
    const n = X.length;
    const d = X[0].length;
    
    // Initialize random seed if provided
    if (this.randomSeed !== undefined) {
      this.rng = this.randomSeed;
    }
    
    this.components = [];
    
    if (this.initialization === 'kmeans++') {
      // Use K-Means++ initialization for means
      const means: number[][] = [];
      
      // Choose first mean randomly
      const firstIdx = Math.floor(this.random() * n);
      means.push([...X[firstIdx]]);
      
      // Choose remaining means
      for (let c = 1; c < this.k; c++) {
        const distances: number[] = [];
        let totalDistance = 0;
        
        for (let i = 0; i < n; i++) {
          let minDist = Infinity;
          for (const mean of means) {
            const dist = this.euclideanDistance(X[i], mean);
            minDist = Math.min(minDist, dist);
          }
          const distSquared = minDist * minDist;
          distances.push(distSquared);
          totalDistance += distSquared;
        }
        
        let target = this.random() * totalDistance;
        let cumSum = 0;
        
        for (let i = 0; i < n; i++) {
          cumSum += distances[i];
          if (cumSum >= target) {
            means.push([...X[i]]);
            break;
          }
        }
      }
      
      // Initialize components with K-Means++ means
      for (let c = 0; c < this.k; c++) {
        this.components.push({
          mean: means[c],
          covariance: this.initializeCovariance(d),
          weight: 1 / this.k
        });
      }
    } else {
      // Random initialization
      for (let c = 0; c < this.k; c++) {
        const randomIdx = Math.floor(this.random() * n);
        this.components.push({
          mean: [...X[randomIdx]],
          covariance: this.initializeCovariance(d),
          weight: 1 / this.k
        });
      }
    }
  }
  
  /**
   * Initialize covariance matrix
   */
  private initializeCovariance(d: number): number[][] {
    const cov: number[][] = [];
    for (let i = 0; i < d; i++) {
      cov[i] = [];
      for (let j = 0; j < d; j++) {
        if (i === j) {
          cov[i][j] = 1.0; // Identity matrix scaled
        } else if (this.covarianceType === 'full') {
          cov[i][j] = 0.0;
        } else {
          cov[i][j] = 0.0; // Diagonal covariance
        }
      }
    }
    return cov;
  }
  
  /**
   * Calculate Euclidean distance
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
   * E-step: Calculate responsibilities (posterior probabilities)
   */
  private eStep(X: number[][]): number {
    const n = X.length;
    this.responsibilities = [];
    let logLikelihood = 0;
    
    for (let i = 0; i < n; i++) {
      const point = X[i];
      const responsibilities: number[] = [];
      let totalProb = 0;
      
      // Calculate probability for each component
      for (let c = 0; c < this.k; c++) {
        const component = this.components[c];
        const prob = component.weight * this.gaussianPDF(point, component.mean, component.covariance);
        responsibilities.push(prob);
        totalProb += prob;
      }
      
      // Normalize responsibilities
      if (totalProb > 0) {
        for (let c = 0; c < this.k; c++) {
          responsibilities[c] /= totalProb;
        }
        logLikelihood += Math.log(totalProb);
      } else {
        // Handle numerical issues
        for (let c = 0; c < this.k; c++) {
          responsibilities[c] = 1 / this.k;
        }
      }
      
      this.responsibilities.push(responsibilities);
    }
    
    return logLikelihood;
  }
  
  /**
   * M-step: Update component parameters
   */
  private mStep(X: number[][]): void {
    const n = X.length;
    const d = X[0].length;
    
    for (let c = 0; c < this.k; c++) {
      // Calculate effective number of points for this component
      let Nc = 0;
      for (let i = 0; i < n; i++) {
        Nc += this.responsibilities[i][c];
      }
      
      if (Nc < 1e-10) {
        // Component has no support, reinitialize
        const randomIdx = Math.floor(this.random() * n);
        this.components[c] = {
          mean: [...X[randomIdx]],
          covariance: this.initializeCovariance(d),
          weight: 1e-10
        };
        continue;
      }
      
      // Update weight
      this.components[c].weight = Nc / n;
      
      // Update mean
      const newMean: number[] = new Array(d).fill(0);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) {
          newMean[j] += this.responsibilities[i][c] * X[i][j];
        }
      }
      for (let j = 0; j < d; j++) {
        newMean[j] /= Nc;
      }
      this.components[c].mean = newMean;
      
      // Update covariance
      const newCov: number[][] = [];
      for (let i = 0; i < d; i++) {
        newCov[i] = new Array(d).fill(0);
      }
      
      for (let i = 0; i < n; i++) {
        const diff = X[i].map((xi, j) => xi - newMean[j]);
        for (let j = 0; j < d; j++) {
          for (let k = 0; k < d; k++) {
            if (this.covarianceType === 'full' || j === k) {
              newCov[j][k] += this.responsibilities[i][c] * diff[j] * diff[k];
            }
          }
        }
      }
      
      for (let j = 0; j < d; j++) {
        for (let k = 0; k < d; k++) {
          if (this.covarianceType === 'full' || j === k) {
            newCov[j][k] /= Nc;
            // Add regularization to avoid singular matrices
            if (j === k) {
              newCov[j][k] += 1e-6;
            }
          }
        }
      }
      
      this.components[c].covariance = newCov;
    }
  }
  
  /**
   * Train the GMM model using EM algorithm
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    // Validate input
    this.validateInput(X);
    
    if (this.k > X.length) {
      throw new Error(`Number of components (${this.k}) cannot exceed number of samples (${X.length})`);
    }
    
    if (this.k < 1) {
      throw new Error('Number of components must be at least 1');
    }
    
    // Store training data
    this.X_train = X.map(row => [...row]);
    
    // Initialize components
    this.initializeComponents(X);
    
    // Reset history
    this.trainingHistory = {
      epochs: [],
      loss: [],
      logLikelihood: []
    };
    this.emHistory = [];
    
    let prevLogLikelihood = -Infinity;
    
    // EM iterations
    for (let iteration = 0; iteration < this.maxIterations; iteration++) {
      // E-step
      const logLikelihood = this.eStep(X);
      
      // M-step
      this.mStep(X);
      
      // Store history
      this.trainingHistory.epochs.push(iteration);
      this.trainingHistory.loss.push(-logLikelihood); // Negative log-likelihood as loss
      this.trainingHistory.logLikelihood.push(logLikelihood);
      
      this.emHistory.push({
        iteration,
        components: this.components.map(c => ({
          mean: [...c.mean],
          covariance: c.covariance.map(row => [...row]),
          weight: c.weight
        })),
        logLikelihood,
        responsibilities: this.responsibilities.map(r => [...r])
      });
      
      // Check convergence
      const improvement = logLikelihood - prevLogLikelihood;
      if (Math.abs(improvement) < this.tolerance) {
        console.log(`GMM converged after ${iteration + 1} iterations`);
        break;
      }
      
      prevLogLikelihood = logLikelihood;
      
      // Allow UI to update periodically
      if (iteration % 5 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    this.trained = true;
  }
  
  /**
   * Predict cluster assignments (hard assignment)
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    this.validateInput(X, this.components[0].mean.length);
    
    const predictions: number[] = [];
    
    for (const point of X) {
      let bestComponent = 0;
      let maxProb = -Infinity;
      
      for (let c = 0; c < this.k; c++) {
        const component = this.components[c];
        const prob = component.weight * this.gaussianPDF(point, component.mean, component.covariance);
        if (prob > maxProb) {
          maxProb = prob;
          bestComponent = c;
        }
      }
      
      predictions.push(bestComponent);
    }
    
    return predictions;
  }
  
  /**
   * Predict probabilities (soft assignment)
   */
  predictProba(X: number[][]): number[][] {
    this.ensureTrained();
    this.validateInput(X, this.components[0].mean.length);
    
    const probabilities: number[][] = [];
    
    for (const point of X) {
      const probs: number[] = [];
      let totalProb = 0;
      
      // Calculate probability for each component
      for (let c = 0; c < this.k; c++) {
        const component = this.components[c];
        const prob = component.weight * this.gaussianPDF(point, component.mean, component.covariance);
        probs.push(prob);
        totalProb += prob;
      }
      
      // Normalize probabilities
      if (totalProb > 0) {
        for (let c = 0; c < this.k; c++) {
          probs[c] /= totalProb;
        }
      } else {
        for (let c = 0; c < this.k; c++) {
          probs[c] = 1 / this.k;
        }
      }
      
      probabilities.push(probs);
    }
    
    return probabilities;
  }
  
  /**
   * Calculate silhouette score for clustering quality
   */
  private calculateSilhouetteScore(): number {
    const assignments = this.predict(this.X_train);
    const n = this.X_train.length;
    
    if (this.k === 1 || n <= this.k) {
      return 0;
    }
    
    let totalScore = 0;
    
    for (let i = 0; i < n; i++) {
      const point = this.X_train[i];
      const cluster = assignments[i];
      
      // Calculate a(i): mean distance to points in same cluster
      const sameClusterPoints = this.X_train.filter((_, j) => 
        assignments[j] === cluster && i !== j
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
          assignments[j] === c
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
      // Get hard assignments for coloring
      const assignments = this.predict(this.X_train);
      
      // 2D scatter plot with probability contours
      const clusterData: any = {};
      const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];
      
      // Add cluster points
      for (let c = 0; c < this.k; c++) {
        const clusterPoints = this.X_train.filter((_, i) => assignments[i] === c);
        
        if (clusterPoints.length > 0) {
          const x = clusterPoints.map(p => p[0]);
          const y = clusterPoints.map(p => p[1]);
          
          clusterData[`cluster${c}`] = {
            x,
            y,
            mode: 'markers',
            name: `Component ${c}`,
            marker: { 
              color: colors[c % colors.length], 
              size: 8,
              opacity: 0.7
            }
          };
        }
      }
      
      // Add component centers
      clusterData.centers = {
        x: this.components.map(c => c.mean[0]),
        y: this.components.map(c => c.mean[1]),
        mode: 'markers',
        name: 'Component Centers',
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
    }
    
    // Log-likelihood curve
    visualizations.push({
      type: 'line',
      data: {
        x: this.trainingHistory.epochs,
        y: this.trainingHistory.logLikelihood,
        mode: 'lines+markers',
        name: 'Log-Likelihood',
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
    
    const assignments = this.predict(this.X_train);
    const finalLogLikelihood = this.trainingHistory.logLikelihood[this.trainingHistory.logLikelihood.length - 1];
    const silhouetteScore = this.calculateSilhouetteScore();
    
    // Calculate cluster sizes
    const clusterSizes: number[] = new Array(this.k).fill(0);
    for (const assignment of assignments) {
      clusterSizes[assignment]++;
    }
    
    // Calculate AIC and BIC
    const n = this.X_train.length;
    const d = this.X_train[0].length;
    const numParams = this.k * (d + d * (d + 1) / 2 + 1) - 1; // means + covariances + weights - 1
    const aic = 2 * numParams - 2 * finalLogLikelihood;
    const bic = Math.log(n) * numParams - 2 * finalLogLikelihood;
    
    return {
      logLikelihood: finalLogLikelihood,
      silhouetteScore,
      iterations: this.trainingHistory.epochs.length,
      clusterSizes,
      aic,
      bic,
      numComponents: this.k,
      converged: this.trainingHistory.epochs.length < this.maxIterations
    };
  }
  
  /**
   * Get component parameters
   */
  getComponents(): GaussianComponent[] {
    return this.components.map(c => ({
      mean: [...c.mean],
      covariance: c.covariance.map(row => [...row]),
      weight: c.weight
    }));
  }
  
  /**
   * Get responsibilities (soft assignments)
   */
  getResponsibilities(): number[][] {
    return this.responsibilities.map(r => [...r]);
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.components = [];
    this.responsibilities = [];
    this.X_train = [];
    this.emHistory = [];
    this.trainingHistory = {
      epochs: [],
      loss: [],
      logLikelihood: []
    };
  }
  
  /**
   * Get model parameters
   */
  getParameters() {
    return {
      k: this.k,
      components: this.getComponents(),
      responsibilities: this.getResponsibilities(),
      initialization: this.initialization,
      covarianceType: this.covarianceType
    };
  }
}