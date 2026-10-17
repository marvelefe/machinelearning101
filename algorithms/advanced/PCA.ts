/**
 * Principal Component Analysis (PCA)
 * 
 * Implements PCA for dimensionality reduction using eigenvalue decomposition.
 * Reduces high-dimensional data to lower dimensions while preserving variance.
 * 
 * Requirements: 6.1, 7.1, 7.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics } from '~/types/algorithm';
import { mean, matrixTranspose, matrixMultiply, covariance, std } from '~/utils/math';

export interface PCAParams {
  nComponents: number;
  standardize: boolean;
}

export class PCA extends BaseAlgorithm {
  name = 'Principal Component Analysis';
  
  // Model parameters
  private components: number[][] = []; // Principal components (eigenvectors)
  private explainedVariance: number[] = []; // Variance explained by each component
  private explainedVarianceRatio: number[] = []; // Ratio of variance explained
  private mean: number[] = []; // Feature means
  private std: number[] = []; // Feature standard deviations
  
  // Training data
  private X_train: number[][] = [];
  private X_transformed: number[][] = [];
  
  // Hyperparameters
  private nComponents: number;
  private shouldStandardize: boolean;
  
  constructor(params: PCAParams = {
    nComponents: 2,
    standardize: true
  }) {
    super();
    this.nComponents = params.nComponents;
    this.shouldStandardize = params.standardize;
  }

  /**
   * Standardize features (z-score normalization)
   */
  private standardizeFeatures(X: number[][]): number[][] {
    const n = X[0].length;
    const m = X.length;
    
    this.mean = [];
    this.std = [];
    
    const X_std: number[][] = X.map(row => [...row]);
    
    for (let j = 0; j < n; j++) {
      const column = X.map(row => row[j]);
      const colMean = mean(column);
      const colStd = std(column, true);
      
      this.mean.push(colMean);
      this.std.push(colStd === 0 ? 1 : colStd);
      
      for (let i = 0; i < m; i++) {
        X_std[i][j] = (X[i][j] - colMean) / this.std[j];
      }
    }
    
    return X_std;
  }
  
  /**
   * Center features (subtract mean)
   */
  private centerFeatures(X: number[][]): number[][] {
    const n = X[0].length;
    const m = X.length;
    
    this.mean = [];
    
    const X_centered: number[][] = X.map(row => [...row]);
    
    for (let j = 0; j < n; j++) {
      const column = X.map(row => row[j]);
      const colMean = mean(column);
      
      this.mean.push(colMean);
      
      for (let i = 0; i < m; i++) {
        X_centered[i][j] = X[i][j] - colMean;
      }
    }
    
    return X_centered;
  }
  
  /**
   * Compute covariance matrix
   */
  private computeCovarianceMatrix(X: number[][]): number[][] {
    const n = X[0].length;
    const m = X.length;
    
    const cov: number[][] = Array(n).fill(0).map(() => Array(n).fill(0));
    
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const col_i = X.map(row => row[i]);
        const col_j = X.map(row => row[j]);
        
        let sum = 0;
        for (let k = 0; k < m; k++) {
          sum += col_i[k] * col_j[k];
        }
        
        const covValue = sum / (m - 1);
        cov[i][j] = covValue;
        cov[j][i] = covValue; // Symmetric
      }
    }
    
    return cov;
  }

  /**
   * Compute eigenvalues and eigenvectors using power iteration
   * Simplified implementation for educational purposes
   */
  private computeEigenDecomposition(matrix: number[][], k: number): {
    eigenvalues: number[];
    eigenvectors: number[][];
  } {
    const n = matrix.length;
    const eigenvalues: number[] = [];
    const eigenvectors: number[][] = [];
    
    // Copy matrix for deflation
    const A = matrix.map(row => [...row]);
    
    for (let comp = 0; comp < k; comp++) {
      // Initialize random vector
      let v = Array(n).fill(0).map(() => Math.random() - 0.5);
      
      // Normalize
      let norm = Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
      v = v.map(val => val / norm);
      
      // Power iteration
      const maxIter = 1000;
      const tolerance = 1e-6;
      let eigenvalue = 0;
      
      for (let iter = 0; iter < maxIter; iter++) {
        // Multiply A * v
        const Av = Array(n).fill(0);
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) {
            Av[i] += A[i][j] * v[j];
          }
        }
        
        // Compute eigenvalue (Rayleigh quotient)
        let numerator = 0;
        let denominator = 0;
        for (let i = 0; i < n; i++) {
          numerator += v[i] * Av[i];
          denominator += v[i] * v[i];
        }
        const newEigenvalue = numerator / denominator;
        
        // Normalize Av
        norm = Math.sqrt(Av.reduce((sum, val) => sum + val * val, 0));
        const newV = Av.map(val => val / norm);
        
        // Check convergence
        if (Math.abs(newEigenvalue - eigenvalue) < tolerance) {
          eigenvalue = newEigenvalue;
          v = newV;
          break;
        }
        
        eigenvalue = newEigenvalue;
        v = newV;
      }
      
      eigenvalues.push(Math.max(0, eigenvalue)); // Ensure non-negative
      eigenvectors.push(v);
      
      // Deflate matrix (remove this component)
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          A[i][j] -= eigenvalue * v[i] * v[j];
        }
      }
    }
    
    return { eigenvalues, eigenvectors };
  }

  /**
   * Train PCA on the data
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    // Validate input
    this.validateInput(X);
    
    const m = X.length;
    const n = X[0].length;
    
    if (this.nComponents > n) {
      throw new Error(`Number of components (${this.nComponents}) cannot exceed number of features (${n})`);
    }
    
    if (this.nComponents > m) {
      console.warn(`Number of components (${this.nComponents}) exceeds number of samples (${m}). Results may be unreliable.`);
    }
    
    // Store training data
    this.X_train = X.map(row => [...row]);
    
    // Standardize or center features
    let X_processed: number[][];
    if (this.shouldStandardize) {
      X_processed = this.standardizeFeatures(X);
    } else {
      X_processed = this.centerFeatures(X);
      this.std = Array(n).fill(1);
    }
    
    // Compute covariance matrix
    const covMatrix = this.computeCovarianceMatrix(X_processed);
    
    // Compute eigenvalues and eigenvectors
    const { eigenvalues, eigenvectors } = this.computeEigenDecomposition(
      covMatrix,
      Math.min(this.nComponents, n)
    );
    
    // Store components (eigenvectors as rows)
    this.components = eigenvectors;
    
    // Calculate explained variance
    const totalVariance = eigenvalues.reduce((sum, val) => sum + val, 0);
    this.explainedVariance = eigenvalues;
    this.explainedVarianceRatio = eigenvalues.map(val => 
      totalVariance > 0 ? val / totalVariance : 0
    );
    
    // Transform training data
    this.X_transformed = this.transformData(X_processed);
    
    this.trained = true;
    
    // Allow UI to update
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  
  /**
   * Transform data to principal component space
   */
  private transformData(X: number[][]): number[][] {
    const m = X.length;
    const k = this.components.length;
    
    const X_transformed: number[][] = Array(m).fill(0).map(() => Array(k).fill(0));
    
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < k; j++) {
        let sum = 0;
        for (let l = 0; l < X[i].length; l++) {
          sum += X[i][l] * this.components[j][l];
        }
        X_transformed[i][j] = sum;
      }
    }
    
    return X_transformed;
  }

  /**
   * Transform new data to principal component space
   */
  predict(X: number[][]): number[] {
    // For PCA, predict returns the transformed data
    // We'll return flattened array for compatibility
    const transformed = this.transform(X);
    return transformed.flat();
  }
  
  /**
   * Transform new data to principal component space
   */
  transform(X: number[][]): number[][] {
    this.ensureTrained();
    this.validateInput(X, this.X_train[0].length);
    
    // Standardize or center using training parameters
    const X_processed: number[][] = X.map(row => [...row]);
    
    for (let i = 0; i < X.length; i++) {
      for (let j = 0; j < X[0].length; j++) {
        X_processed[i][j] = (X[i][j] - this.mean[j]) / this.std[j];
      }
    }
    
    return this.transformData(X_processed);
  }
  
  /**
   * Get visualization data
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained) {
      return [];
    }
    
    const visualizations: VisualizationData[] = [];
    
    // Scree plot (explained variance ratio)
    const componentNumbers = this.explainedVarianceRatio.map((_, i) => i + 1);
    visualizations.push({
      type: 'line',
      data: {
        scree: {
          x: componentNumbers,
          y: this.explainedVarianceRatio,
          mode: 'lines+markers',
          name: 'Explained Variance Ratio',
          line: { color: '#3b82f6', width: 2 },
          marker: { size: 8, color: '#3b82f6' }
        },
        cumulative: {
          x: componentNumbers,
          y: this.explainedVarianceRatio.reduce((acc: number[], val, i) => {
            acc.push((acc[i - 1] || 0) + val);
            return acc;
          }, []),
          mode: 'lines+markers',
          name: 'Cumulative Variance',
          line: { color: '#10b981', width: 2, dash: 'dash' },
          marker: { size: 8, color: '#10b981' }
        }
      }
    });
    
    // 2D projection (if we have at least 2 components)
    if (this.nComponents >= 2 && this.X_transformed.length > 0) {
      const pc1 = this.X_transformed.map(row => row[0]);
      const pc2 = this.X_transformed.map(row => row[1]);
      
      visualizations.push({
        type: 'scatter2d',
        data: {
          projection: {
            x: pc1,
            y: pc2,
            mode: 'markers',
            name: 'Projected Data',
            marker: { 
              color: '#8b5cf6',
              size: 8,
              opacity: 0.7
            }
          }
        }
      });
    }
    
    // 3D projection (if we have at least 3 components)
    if (this.nComponents >= 3 && this.X_transformed.length > 0) {
      const pc1 = this.X_transformed.map(row => row[0]);
      const pc2 = this.X_transformed.map(row => row[1]);
      const pc3 = this.X_transformed.map(row => row[2]);
      
      visualizations.push({
        type: 'scatter3d',
        data: {
          x: pc1,
          y: pc2,
          z: pc3,
          mode: 'markers',
          name: '3D Projection',
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
   * Get metrics
   */
  getMetrics(): Metrics {
    if (!this.trained) {
      return {};
    }
    
    const cumulativeVariance = this.explainedVarianceRatio.reduce((acc, val) => acc + val, 0);
    
    return {
      explainedVariance: this.explainedVariance,
      explainedVarianceRatio: this.explainedVarianceRatio,
      cumulativeVariance,
      nComponents: this.components.length,
      nFeatures: this.X_train[0].length,
      nSamples: this.X_train.length
    };
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.components = [];
    this.explainedVariance = [];
    this.explainedVarianceRatio = [];
    this.mean = [];
    this.std = [];
    this.X_train = [];
    this.X_transformed = [];
  }
  
  /**
   * Get model parameters
   */
  getParameters() {
    return {
      components: this.components.map(c => [...c]),
      explainedVariance: [...this.explainedVariance],
      explainedVarianceRatio: [...this.explainedVarianceRatio],
      mean: [...this.mean],
      std: [...this.std],
      nComponents: this.nComponents
    };
  }
  
  /**
   * Get the principal components (eigenvectors)
   */
  getComponents(): number[][] {
    return this.components.map(c => [...c]);
  }
  
  /**
   * Get explained variance ratio for each component
   */
  getExplainedVarianceRatio(): number[] {
    return [...this.explainedVarianceRatio];
  }
  
  /**
   * Get transformed training data
   */
  getTransformedData(): number[][] {
    return this.X_transformed.map(row => [...row]);
  }
}
