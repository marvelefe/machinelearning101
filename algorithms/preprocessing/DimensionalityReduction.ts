/**
 * Dimensionality Reduction Algorithm
 * Implements various dimensionality reduction techniques including PCA, LDA, and others
 * Requirements: 8.1, 8.2, 8.3, 8.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics, TrainingHistory } from '~/types/algorithm'; 


export type ReductionMethod = 'pca' | 'lda' | 'ica' | 'factor-analysis' | 'truncated-svd' | 'none';

export interface DimensionalityReductionResult {
  reducedData: number[][];
  originalData: number[][];
  components: number[][];
  explainedVariance: number[];
  cumulativeVariance: number[];
  method: ReductionMethod;
  statistics: {
    originalDimensions: number;
    reducedDimensions: number;
    varianceRetained: number;
    compressionRatio: number;
  };
}

export class DimensionalityReduction extends BaseAlgorithm {
  name = 'Dimensionality Reduction';
  
  private data: number[][] = [];
  private targetValues: number[] = [];
  private reducedData: number[][] = [];
  private components: number[][] = [];
  private explainedVariance: number[] = [];
  private method: ReductionMethod = 'none';
  private nComponents: number = 2;
  private mean: number[] = [];

  constructor() {
    super();
  }

  /**
   * Set the data for dimensionality reduction
   */
  setData(data: number[][], target?: number[], nComponents: number = 2): void {
    this.data = data;
    this.targetValues = target || [];
    this.nComponents = Math.min(nComponents, data[0]?.length || 2);
    this.reducedData = [];
    this.components = [];
    this.explainedVariance = [];
    this.mean = [];
    this.trained = false;
  }

  /**
   * Fit the dimensionality reduction algorithm
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    this.setData(X, y, this.nComponents);
    
    // Apply default PCA
    this.applyPCA();
    this.trained = true;
  }

  /**
   * Apply Principal Component Analysis (PCA)
   */
  applyPCA(nComponents?: number): DimensionalityReductionResult {
    if (this.data.length === 0) {
      throw new Error('No data for PCA');
    }

    this.method = 'pca';
    if (nComponents) {
      this.nComponents = Math.min(nComponents, this.data[0].length);
    }

    const n = this.data.length;
    const d = this.data[0].length;

    // Center the data (subtract mean)
    this.mean = [];
    for (let j = 0; j < d; j++) {
      const columnSum = this.data.reduce((sum, row) => sum + row[j], 0);
      this.mean[j] = columnSum / n;
    }

    const centeredData = this.data.map(row => 
      row.map((val, j) => val - this.mean[j])
    );

    // Compute covariance matrix
    const covMatrix: number[][] = [];
    for (let i = 0; i < d; i++) {
      covMatrix[i] = [];
      for (let j = 0; j < d; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += centeredData[k][i] * centeredData[k][j];
        }
        covMatrix[i][j] = sum / (n - 1);
      }
    }

    // Compute eigenvalues and eigenvectors using power iteration (simplified)
    const { eigenvalues, eigenvectors } = this.computeEigenDecomposition(covMatrix);

    // Sort by eigenvalues (descending)
    const sortedIndices = eigenvalues
      .map((val, idx) => ({ val, idx }))
      .sort((a, b) => b.val - a.val)
      .map(item => item.idx);

    // Select top components
    this.components = [];
    this.explainedVariance = [];
    
    for (let i = 0; i < this.nComponents; i++) {
      const idx = sortedIndices[i];
      this.components.push(eigenvectors[idx]);
      this.explainedVariance.push(Math.max(0, eigenvalues[idx]));
    }

    // Transform data
    this.reducedData = centeredData.map(row => {
      const transformedRow: number[] = [];
      for (let i = 0; i < this.nComponents; i++) {
        let sum = 0;
        for (let j = 0; j < d; j++) {
          sum += row[j] * this.components[i][j];
        }
        transformedRow.push(sum);
      }
      return transformedRow;
    });

    return this.getResult();
  }

  /**
   * Apply Linear Discriminant Analysis (LDA)
   */
  applyLDA(nComponents?: number): DimensionalityReductionResult {
    if (this.data.length === 0 || this.targetValues.length === 0) {
      throw new Error('Data and target values required for LDA');
    }

    this.method = 'lda';
    const classes = Array.from(new Set(this.targetValues));
    const maxComponents = Math.min(classes.length - 1, this.data[0].length);
    
    if (nComponents) {
      this.nComponents = Math.min(nComponents, maxComponents);
    } else {
      this.nComponents = Math.min(this.nComponents, maxComponents);
    }

    const n = this.data.length;
    const d = this.data[0].length;

    // Calculate overall mean
    this.mean = [];
    for (let j = 0; j < d; j++) {
      const columnSum = this.data.reduce((sum, row) => sum + row[j], 0);
      this.mean[j] = columnSum / n;
    }

    // Calculate class means and within-class scatter matrix
    const classMeans: number[][] = [];
    const classData: number[][][] = [];
    
    for (const cls of classes) {
      const classIndices = this.targetValues
        .map((val, idx) => val === cls ? idx : -1)
        .filter(idx => idx !== -1);
      
      const classRows = classIndices.map(idx => this.data[idx]);
      classData.push(classRows);
      
      const classMean = [];
      for (let j = 0; j < d; j++) {
        const sum = classRows.reduce((s, row) => s + row[j], 0);
        classMean.push(sum / classRows.length);
      }
      classMeans.push(classMean);
    }

    // Within-class scatter matrix
    const withinScatter: number[][] = Array(d).fill(0).map(() => Array(d).fill(0));
    
    for (let c = 0; c < classes.length; c++) {
      const classRows = classData[c];
      const classMean = classMeans[c];
      
      for (const row of classRows) {
        for (let i = 0; i < d; i++) {
          for (let j = 0; j < d; j++) {
            withinScatter[i][j] += (row[i] - classMean[i]) * (row[j] - classMean[j]);
          }
        }
      }
    }

    // Between-class scatter matrix
    const betweenScatter: number[][] = Array(d).fill(0).map(() => Array(d).fill(0));
    
    for (let c = 0; c < classes.length; c++) {
      const classMean = classMeans[c];
      const classSize = classData[c].length;
      
      for (let i = 0; i < d; i++) {
        for (let j = 0; j < d; j++) {
          betweenScatter[i][j] += classSize * (classMean[i] - this.mean[i]) * (classMean[j] - this.mean[j]);
        }
      }
    }

    // Solve generalized eigenvalue problem: Sw^-1 * Sb
    // Simplified: use pseudo-inverse approximation
    const invWithinScatter = this.pseudoInverse(withinScatter);
    const ldaMatrix = this.matrixMultiply(invWithinScatter, betweenScatter);

    // Compute eigendecomposition
    const { eigenvalues, eigenvectors } = this.computeEigenDecomposition(ldaMatrix);

    // Sort by eigenvalues (descending)
    const sortedIndices = eigenvalues
      .map((val, idx) => ({ val, idx }))
      .sort((a, b) => b.val - a.val)
      .map(item => item.idx);

    // Select top components
    this.components = [];
    this.explainedVariance = [];
    
    for (let i = 0; i < this.nComponents; i++) {
      const idx = sortedIndices[i];
      this.components.push(eigenvectors[idx]);
      this.explainedVariance.push(Math.max(0, eigenvalues[idx]));
    }

    // Transform data
    this.reducedData = this.data.map(row => {
      const centeredRow = row.map((val, j) => val - this.mean[j]);
      const transformedRow: number[] = [];
      
      for (let i = 0; i < this.nComponents; i++) {
        let sum = 0;
        for (let j = 0; j < d; j++) {
          sum += centeredRow[j] * this.components[i][j];
        }
        transformedRow.push(sum);
      }
      return transformedRow;
    });

    return this.getResult();
  }

  /**
   * Apply Independent Component Analysis (ICA) - simplified version
   */
  applyICA(nComponents?: number): DimensionalityReductionResult {
    if (this.data.length === 0) {
      throw new Error('No data for ICA');
    }

    this.method = 'ica';
    if (nComponents) {
      this.nComponents = Math.min(nComponents, this.data[0].length);
    }

    // First apply PCA for whitening
    const pcaResult = this.applyPCA(this.nComponents);
    const whitenedData = pcaResult.reducedData;

    // Apply FastICA algorithm (simplified)
    this.components = [];
    this.explainedVariance = [];
    
    // Initialize random components
    for (let i = 0; i < this.nComponents; i++) {
      const component = Array(this.nComponents).fill(0).map(() => Math.random() - 0.5);
      // Normalize
      const norm = Math.sqrt(component.reduce((sum, val) => sum + val * val, 0));
      this.components.push(component.map(val => val / norm));
      this.explainedVariance.push(1 / this.nComponents); // Equal weights for ICA
    }

    // Simple ICA iterations (FastICA approximation)
    for (let iter = 0; iter < 10; iter++) {
      for (let i = 0; i < this.nComponents; i++) {
        const w = this.components[i];
        
        // Update rule (simplified)
        const newW = Array(this.nComponents).fill(0);
        
        for (const row of whitenedData) {
          const projection = row.reduce((sum, val, j) => sum + val * w[j], 0);
          const g = Math.tanh(projection); // Non-linear function
          const gPrime = 1 - g * g; // Derivative
          
          for (let j = 0; j < this.nComponents; j++) {
            newW[j] += row[j] * g - w[j] * gPrime;
          }
        }
        
        // Normalize
        const norm = Math.sqrt(newW.reduce((sum, val) => sum + val * val, 0));
        if (norm > 0) {
          this.components[i] = newW.map(val => val / norm);
        }
      }
    }

    // Transform data using ICA components
    this.reducedData = whitenedData.map(row => {
      const transformedRow: number[] = [];
      for (let i = 0; i < this.nComponents; i++) {
        let sum = 0;
        for (let j = 0; j < this.nComponents; j++) {
          sum += row[j] * this.components[i][j];
        }
        transformedRow.push(sum);
      }
      return transformedRow;
    });

    return this.getResult();
  }

  /**
   * Apply Truncated SVD
   */
  applyTruncatedSVD(nComponents?: number): DimensionalityReductionResult {
    if (this.data.length === 0) {
      throw new Error('No data for Truncated SVD');
    }

    this.method = 'truncated-svd';
    if (nComponents) {
      this.nComponents = Math.min(nComponents, Math.min(this.data.length, this.data[0].length));
    }

    // SVD: X = U * Σ * V^T
    // For dimensionality reduction, we use V^T (right singular vectors)
    
    // Simplified SVD using eigendecomposition of X^T * X
    const n = this.data.length;
    const d = this.data[0].length;
    
    // Compute X^T * X
    const XTX: number[][] = [];
    for (let i = 0; i < d; i++) {
      XTX[i] = [];
      for (let j = 0; j < d; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += this.data[k][i] * this.data[k][j];
        }
        XTX[i][j] = sum;
      }
    }

    // Eigendecomposition
    const { eigenvalues, eigenvectors } = this.computeEigenDecomposition(XTX);

    // Sort by eigenvalues (descending)
    const sortedIndices = eigenvalues
      .map((val, idx) => ({ val, idx }))
      .sort((a, b) => b.val - a.val)
      .map(item => item.idx);

    // Select top components
    this.components = [];
    this.explainedVariance = [];
    
    for (let i = 0; i < this.nComponents; i++) {
      const idx = sortedIndices[i];
      this.components.push(eigenvectors[idx]);
      this.explainedVariance.push(Math.sqrt(Math.max(0, eigenvalues[idx])));
    }

    // Transform data
    this.reducedData = this.data.map(row => {
      const transformedRow: number[] = [];
      for (let i = 0; i < this.nComponents; i++) {
        let sum = 0;
        for (let j = 0; j < d; j++) {
          sum += row[j] * this.components[i][j];
        }
        transformedRow.push(sum);
      }
      return transformedRow;
    });

    return this.getResult();
  }

  /**
   * Make predictions (transform new data using fitted components)
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    
    // Transform new data using learned components
    const transformedData = X.map(row => {
      const centeredRow = this.method === 'pca' || this.method === 'lda' 
        ? row.map((val, j) => val - (this.mean[j] || 0))
        : row;
      
      const transformedRow: number[] = [];
      for (let i = 0; i < this.nComponents; i++) {
        let sum = 0;
        for (let j = 0; j < Math.min(row.length, this.components[i].length); j++) {
          sum += centeredRow[j] * this.components[i][j];
        }
        transformedRow.push(sum);
      }
      return transformedRow;
    });
    
    return transformedData.flat();
  }

  /**
   * Get visualization data
   */
  getVisualizationData(): VisualizationData[] {
    const result = this.getResult();
    const visualizations: VisualizationData[] = [];

    // Scree plot (explained variance)
    visualizations.push({
      type: 'line',
      data: {
        x: Array.from({ length: result.explainedVariance.length }, (_, i) => i + 1),
        y: result.explainedVariance,
        mode: 'lines+markers',
        name: 'Explained Variance',
        line: { color: 'blue' }
      },
      layout: {
        title: 'Scree Plot - Explained Variance by Component',
        xaxis: { title: 'Component' },
        yaxis: { title: 'Explained Variance' }
      }
    });

    // Cumulative variance plot
    const cumulativeVariance = result.cumulativeVariance;
    visualizations.push({
      type: 'line',
      data: {
        x: Array.from({ length: cumulativeVariance.length }, (_, i) => i + 1),
        y: cumulativeVariance,
        mode: 'lines+markers',
        name: 'Cumulative Variance',
        line: { color: 'red' }
      },
      layout: {
        title: 'Cumulative Explained Variance',
        xaxis: { title: 'Number of Components' },
        yaxis: { title: 'Cumulative Variance Ratio' }
      }
    });

    // 2D scatter plot of reduced data
    if (result.reducedData[0]?.length >= 2) {
      const colors = this.targetValues.length > 0 
        ? this.targetValues.map(val => `hsl(${(val * 137.5) % 360}, 70%, 50%)`)
        : Array(result.reducedData.length).fill('blue');

      visualizations.push({
        type: 'scatter2d',
        data: {
          x: result.reducedData.map(row => row[0]),
          y: result.reducedData.map(row => row[1]),
          mode: 'markers',
          marker: {
            color: colors,
            size: 6,
            opacity: 0.7
          },
          name: 'Reduced Data'
        },
        layout: {
          title: `${this.method.toUpperCase()} - 2D Projection`,
          xaxis: { title: 'First Component' },
          yaxis: { title: 'Second Component' }
        }
      });
    }

    return visualizations;
  }

  /**
   * Get performance metrics
   */
  getMetrics(): Metrics {
    const result = this.getResult();
    
    return {
      originalDimensions: result.statistics.originalDimensions,
      reducedDimensions: result.statistics.reducedDimensions,
      varianceRetained: result.statistics.varianceRetained,
      compressionRatio: result.statistics.compressionRatio,
      dimensionalityReduction: (1 - result.statistics.compressionRatio) * 100
    };
  }

  /**
   * Get the current reduction result
   */
  getResult(): DimensionalityReductionResult {
    const originalDimensions = this.data[0]?.length || 0;
    const reducedDimensions = this.nComponents;
    const compressionRatio = originalDimensions > 0 ? reducedDimensions / originalDimensions : 1;
    
    // Calculate cumulative variance
    const totalVariance = this.explainedVariance.reduce((sum, val) => sum + val, 0);
    const cumulativeVariance: number[] = [];
    let cumSum = 0;
    
    for (const variance of this.explainedVariance) {
      cumSum += variance;
      cumulativeVariance.push(totalVariance > 0 ? cumSum / totalVariance : 0);
    }
    
    const varianceRetained = cumulativeVariance[cumulativeVariance.length - 1] || 0;
    
    return {
      reducedData: this.reducedData.length > 0 ? this.reducedData : this.data,
      originalData: this.data,
      components: this.components,
      explainedVariance: this.explainedVariance,
      cumulativeVariance,
      method: this.method,
      statistics: {
        originalDimensions,
        reducedDimensions,
        varianceRetained,
        compressionRatio
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
    this.reducedData = [];
    this.components = [];
    this.explainedVariance = [];
    this.method = 'none';
    this.nComponents = 2;
    this.mean = [];
  }

  // Helper methods for matrix operations
  private computeEigenDecomposition(matrix: number[][]): { eigenvalues: number[], eigenvectors: number[][] } {
    const n = matrix.length;
    const eigenvalues: number[] = [];
    const eigenvectors: number[][] = [];
    
    // Simplified eigendecomposition using power iteration
    for (let i = 0; i < Math.min(n, 10); i++) {
      // Random initial vector
      let vector = Array(n).fill(0).map(() => Math.random() - 0.5);
      
      // Power iteration
      for (let iter = 0; iter < 50; iter++) {
        // Multiply by matrix
        const newVector = Array(n).fill(0);
        for (let j = 0; j < n; j++) {
          for (let k = 0; k < n; k++) {
            newVector[j] += matrix[j][k] * vector[k];
          }
        }
        
        // Normalize
        const norm = Math.sqrt(newVector.reduce((sum, val) => sum + val * val, 0));
        if (norm > 1e-10) {
          vector = newVector.map(val => val / norm);
        }
      }
      
      // Calculate eigenvalue
      let eigenvalue = 0;
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += matrix[j][k] * vector[k];
        }
        eigenvalue += vector[j] * sum;
      }
      
      eigenvalues.push(eigenvalue);
      eigenvectors.push([...vector]);
      
      // Deflate matrix (remove this eigenvalue/eigenvector)
      for (let j = 0; j < n; j++) {
        for (let k = 0; k < n; k++) {
          matrix[j][k] -= eigenvalue * vector[j] * vector[k];
        }
      }
    }
    
    return { eigenvalues, eigenvectors };
  }

  private pseudoInverse(matrix: number[][]): number[][] {
    const n = matrix.length;
    const result: number[][] = Array(n).fill(0).map(() => Array(n).fill(0));
    
    // Simplified pseudo-inverse using diagonal approximation
    for (let i = 0; i < n; i++) {
      const diagonalElement = matrix[i][i];
      if (Math.abs(diagonalElement) > 1e-10) {
        result[i][i] = 1 / diagonalElement;
      }
    }
    
    return result;
  }

  private matrixMultiply(A: number[][], B: number[][]): number[][] {
    const n = A.length;
    const m = B[0].length;
    const p = B.length;
    
    const result: number[][] = Array(n).fill(0).map(() => Array(m).fill(0));
    
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        for (let k = 0; k < p; k++) {
          result[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    
    return result;
  }

  /**
   * Generate high-dimensional sample data
   */
  static generateHighDimensionalData(
    rows: number,
    dimensions: number,
    intrinsicDimensions: number = 2,
    noiseLevel: number = 0.1
  ): { data: number[][], target?: number[] } {
    const data: number[][] = [];
    const target: number[] = [];
    
    for (let i = 0; i < rows; i++) {
      // Generate intrinsic low-dimensional data
      const intrinsicData = Array(intrinsicDimensions).fill(0).map(() => Math.random() * 10 - 5);
      
      // Embed in high-dimensional space with linear combinations + noise
      const row: number[] = [];
      for (let j = 0; j < dimensions; j++) {
        let value = 0;
        
        // Linear combination of intrinsic dimensions
        for (let k = 0; k < intrinsicDimensions; k++) {
          const weight = Math.random() * 2 - 1; // Random weight
          value += weight * intrinsicData[k];
        }
        
        // Add noise
        value += (Math.random() - 0.5) * noiseLevel;
        row.push(value);
      }
      
      data.push(row);
      
      // Generate target based on first intrinsic dimension
      target.push(intrinsicData[0] > 0 ? 1 : 0);
    }
    
    return { data, target };
  }
}