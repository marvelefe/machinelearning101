/**
 * t-SNE (t-Distributed Stochastic Neighbor Embedding)
 * 
 * Implements t-SNE for non-linear dimensionality reduction and visualization.
 * Particularly effective for visualizing high-dimensional data in 2D or 3D.
 * 
 * Requirements: 6.2, 7.1, 7.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics, TrainingHistory } from '~/types/algorithm';
import { mean, std } from '~/utils/math';

export interface TSNEParams {
  nComponents: number;
  perplexity: number;
  learningRate: number;
  maxIterations: number;
}

export class TSNE extends BaseAlgorithm {
  name = 't-SNE';
  
  // Model parameters
  private Y: number[][] = []; // Low-dimensional embedding
  private P: number[][] = []; // High-dimensional probabilities
  
  // Training data
  private X_train: number[][] = [];
  private iterationHistory: number[][][] = []; // Store embeddings at different iterations
  
  // Hyperparameters
  private nComponents: number;
  private perplexity: number;
  private learningRate: number;
  private maxIterations: number;
  
  constructor(params: TSNEParams = {
    nComponents: 2,
    perplexity: 30,
    learningRate: 200,
    maxIterations: 1000
  }) {
    super();
    this.nComponents = params.nComponents;
    this.perplexity = params.perplexity;
    this.learningRate = params.learningRate;
    this.maxIterations = params.maxIterations;
  }
  
  /**
   * Compute Euclidean distance between two points
   */
  private euclideanDistance(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - b[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }
  
  /**
   * Compute pairwise distances
   */
  private computeDistances(X: number[][]): number[][] {
    const n = X.length;
    const distances: number[][] = Array(n).fill(0).map(() => Array(n).fill(0));
    
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dist = this.euclideanDistance(X[i], X[j]);
        distances[i][j] = dist;
        distances[j][i] = dist;
      }
    }
    
    return distances;
  }

  /**
   * Compute Gaussian kernel with specific variance
   */
  private computeGaussianKernel(distances: number[], beta: number): number[] {
    const n = distances.length;
    const P = new Array(n);
    
    for (let i = 0; i < n; i++) {
      P[i] = Math.exp(-distances[i] * distances[i] * beta);
    }
    
    return P;
  }
  
  /**
   * Compute Shannon entropy
   */
  private computeEntropy(P: number[]): number {
    let entropy = 0;
    for (let i = 0; i < P.length; i++) {
      if (P[i] > 1e-10) {
        entropy -= P[i] * Math.log2(P[i]);
      }
    }
    return entropy;
  }
  
  /**
   * Binary search for beta (precision) that gives desired perplexity
   */
  private findBeta(distances: number[], targetPerplexity: number): number {
    let betaMin = -Infinity;
    let betaMax = Infinity;
    let beta = 1.0;
    
    const tolerance = 1e-5;
    const maxTries = 50;
    
    for (let iter = 0; iter < maxTries; iter++) {
      const P = this.computeGaussianKernel(distances, beta);
      const sum = P.reduce((a, b) => a + b, 0);
      
      if (sum === 0) {
        beta = beta * 2;
        continue;
      }
      
      const Pnorm = P.map(p => p / sum);
      const entropy = this.computeEntropy(Pnorm);
      const perplexity = Math.pow(2, entropy);
      
      const perplexityDiff = perplexity - targetPerplexity;
      
      if (Math.abs(perplexityDiff) < tolerance) {
        break;
      }
      
      if (perplexityDiff > 0) {
        betaMin = beta;
        beta = betaMax === Infinity ? beta * 2 : (beta + betaMax) / 2;
      } else {
        betaMax = beta;
        beta = betaMin === -Infinity ? beta / 2 : (beta + betaMin) / 2;
      }
    }
    
    return beta;
  }
  
  /**
   * Compute high-dimensional probability matrix P
   */
  private computeP(distances: number[][]): number[][] {
    const n = distances.length;
    const P: number[][] = Array(n).fill(0).map(() => Array(n).fill(0));
    
    for (let i = 0; i < n; i++) {
      const beta = this.findBeta(distances[i], this.perplexity);
      const Pi = this.computeGaussianKernel(distances[i], beta);
      const sum = Pi.reduce((a, b) => a + b, 0) - Pi[i]; // Exclude self
      
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          P[i][j] = Pi[j] / sum;
        }
      }
    }
    
    // Symmetrize
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const pij = (P[i][j] + P[j][i]) / (2 * n);
        P[i][j] = Math.max(pij, 1e-12);
        P[j][i] = P[i][j];
      }
    }
    
    return P;
  }

  /**
   * Compute low-dimensional probability matrix Q using Student t-distribution
   */
  private computeQ(Y: number[][]): number[][] {
    const n = Y.length;
    const Q: number[][] = Array(n).fill(0).map(() => Array(n).fill(0));
    
    // Compute pairwise distances in low-dimensional space
    let sum = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dist = this.euclideanDistance(Y[i], Y[j]);
        const qij = 1 / (1 + dist * dist); // Student t-distribution with df=1
        Q[i][j] = qij;
        Q[j][i] = qij;
        sum += 2 * qij;
      }
    }
    
    // Normalize
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        Q[i][j] = Math.max(Q[i][j] / sum, 1e-12);
      }
    }
    
    return Q;
  }
  
  /**
   * Compute KL divergence (cost function)
   */
  private computeCost(P: number[][], Q: number[][]): number {
    const n = P.length;
    let cost = 0;
    
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j && P[i][j] > 1e-12 && Q[i][j] > 1e-12) {
          cost += P[i][j] * Math.log(P[i][j] / Q[i][j]);
        }
      }
    }
    
    return cost;
  }
  
  /**
   * Compute gradient of cost function
   */
  private computeGradient(P: number[][], Q: number[][], Y: number[][]): number[][] {
    const n = Y.length;
    const d = Y[0].length;
    const gradient: number[][] = Array(n).fill(0).map(() => Array(d).fill(0));
    
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          const dist = this.euclideanDistance(Y[i], Y[j]);
          const mult = (P[i][j] - Q[i][j]) / (1 + dist * dist);
          
          for (let k = 0; k < d; k++) {
            gradient[i][k] += 4 * mult * (Y[i][k] - Y[j][k]);
          }
        }
      }
    }
    
    return gradient;
  }

  /**
   * Train t-SNE
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    // Validate input
    this.validateInput(X);
    
    const n = X.length;
    
    if (n < 3) {
      throw new Error('t-SNE requires at least 3 data points');
    }
    
    if (this.perplexity >= n) {
      throw new Error(`Perplexity (${this.perplexity}) must be less than number of samples (${n})`);
    }
    
    // Store training data
    this.X_train = X.map(row => [...row]);
    
    // Initialize training history
    this.trainingHistory = {
      epochs: [],
      loss: []
    };
    this.iterationHistory = [];
    
    // Compute pairwise distances in high-dimensional space
    const distances = this.computeDistances(X);
    
    // Compute high-dimensional probabilities P
    this.P = this.computeP(distances);
    
    // Initialize low-dimensional embedding Y randomly
    this.Y = Array(n).fill(0).map(() => 
      Array(this.nComponents).fill(0).map(() => (Math.random() - 0.5) * 0.0001)
    );
    
    // Gradient descent with momentum
    let velocity: number[][] = Array(n).fill(0).map(() => Array(this.nComponents).fill(0));
    const momentum = 0.5;
    const finalMomentum = 0.8;
    const momentumSwitchIter = 250;
    
    // Store snapshots for animation
    const snapshotInterval = Math.floor(this.maxIterations / 10);
    
    for (let iter = 0; iter < this.maxIterations; iter++) {
      // Compute Q and gradient
      const Q = this.computeQ(this.Y);
      const gradient = this.computeGradient(this.P, Q, this.Y);
      
      // Compute cost
      const cost = this.computeCost(this.P, Q);
      
      // Store history
      if (iter % 10 === 0) {
        this.trainingHistory.epochs.push(iter);
        this.trainingHistory.loss.push(cost);
      }
      
      // Store snapshot for animation
      if (iter % snapshotInterval === 0 || iter === this.maxIterations - 1) {
        this.iterationHistory.push(this.Y.map(row => [...row]));
      }
      
      // Update with momentum
      const currentMomentum = iter < momentumSwitchIter ? momentum : finalMomentum;
      
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < this.nComponents; j++) {
          velocity[i][j] = currentMomentum * velocity[i][j] - this.learningRate * gradient[i][j];
          this.Y[i][j] += velocity[i][j];
        }
      }
      
      // Allow UI to update periodically
      if (iter % 50 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    this.trained = true;
  }
  
  /**
   * Transform is not supported for t-SNE (it's not a parametric method)
   */
  predict(X: number[][]): number[] {
    throw new Error('t-SNE does not support transform/predict on new data. It must be rerun on the entire dataset.');
  }
  
  /**
   * Get the low-dimensional embedding
   */
  getEmbedding(): number[][] {
    return this.Y.map(row => [...row]);
  }

  /**
   * Get visualization data
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained) {
      return [];
    }
    
    const visualizations: VisualizationData[] = [];
    
    // Loss curve
    visualizations.push({
      type: 'line',
      data: {
        loss: {
          x: this.trainingHistory.epochs,
          y: this.trainingHistory.loss,
          mode: 'lines',
          name: 'KL Divergence',
          line: { color: '#ef4444', width: 2 }
        }
      }
    });
    
    // Final 2D embedding
    if (this.nComponents >= 2) {
      const x = this.Y.map(point => point[0]);
      const y = this.Y.map(point => point[1]);
      
      visualizations.push({
        type: 'scatter2d',
        data: {
          embedding: {
            x,
            y,
            mode: 'markers',
            name: 't-SNE Embedding',
            marker: {
              size: 8,
              color: '#8b5cf6',
              opacity: 0.7
            }
          }
        }
      });
    }
    
    // 3D embedding if available
    if (this.nComponents >= 3) {
      const x = this.Y.map(point => point[0]);
      const y = this.Y.map(point => point[1]);
      const z = this.Y.map(point => point[2]);
      
      visualizations.push({
        type: 'scatter3d',
        data: {
          x,
          y,
          z,
          mode: 'markers',
          name: '3D Embedding',
          marker: {
            size: 5,
            color: '#8b5cf6',
            opacity: 0.7
          },
          type: 'scatter3d'
        }
      });
    }
    
    return visualizations;
  }
  
  /**
   * Get animation frames for iteration visualization
   */
  getAnimationFrames(): number[][][] {
    return this.iterationHistory.map(frame => frame.map(row => [...row]));
  }
  
  /**
   * Get metrics
   */
  getMetrics(): Metrics {
    if (!this.trained) {
      return {};
    }
    
    const finalCost = this.trainingHistory.loss[this.trainingHistory.loss.length - 1];
    
    return {
      finalCost,
      iterations: this.maxIterations,
      perplexity: this.perplexity,
      nComponents: this.nComponents,
      nSamples: this.X_train.length,
      nFeatures: this.X_train[0].length
    };
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.Y = [];
    this.P = [];
    this.X_train = [];
    this.iterationHistory = [];
  }
  
  /**
   * Get model parameters
   */
  getParameters() {
    return {
      embedding: this.Y.map(row => [...row]),
      perplexity: this.perplexity,
      learningRate: this.learningRate,
      maxIterations: this.maxIterations,
      nComponents: this.nComponents
    };
  }
}
