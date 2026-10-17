/**
 * Image Segmentation Algorithm
 * 
 * Implements pixel-wise classification for semantic segmentation.
 * Uses a simplified approach with k-means clustering on pixel features
 * (color, position, intensity) to segment images into regions.
 * Designed for browser-based learning with segmentation mask visualization.
 * 
 * Requirements: 2.2, 2.4, 2.5
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics } from '~/types/algorithm';
import { mean, euclideanDistance } from '~/utils/math';

export interface ImageSegmentationParams {
  numSegments: number;
  maxIterations: number;
  useColorFeatures: boolean;
  usePositionFeatures: boolean;
  positionWeight: number;
}

export interface SegmentationMask {
  width: number;
  height: number;
  segments: number[][];
  colors: string[];
}

export class ImageSegmentation extends BaseAlgorithm {
  name = 'Image Segmentation';
  
  // Model parameters - cluster centers for each segment
  private clusterCenters: number[][] = [];
  
  // Training data
  private X_train: number[][] = [];
  private imageHeight: number = 0;
  private imageWidth: number = 0;
  
  // Hyperparameters
  private numSegments: number;
  private maxIterations: number;
  private useColorFeatures: boolean;
  private usePositionFeatures: boolean;
  private positionWeight: number;
  
  // Segmentation results for visualization
  private segmentationMask: number[][] = [];
  private segmentColors: string[] = [];
  
  constructor(params: ImageSegmentationParams = {
    numSegments: 4,
    maxIterations: 50,
    useColorFeatures: true,
    usePositionFeatures: true,
    positionWeight: 0.3
  }) {
    super();
    this.numSegments = params.numSegments;
    this.maxIterations = params.maxIterations;
    this.useColorFeatures = params.useColorFeatures;
    this.usePositionFeatures = params.usePositionFeatures;
    this.positionWeight = params.positionWeight;
  }
  
  /**
   * Extract features for a pixel
   */
  private extractPixelFeatures(
    image: number[][],
    row: number,
    col: number
  ): number[] {
    const features: number[] = [];
    
    // Color/intensity feature
    if (this.useColorFeatures) {
      features.push(image[row][col]);
    }
    
    // Position features (normalized)
    if (this.usePositionFeatures) {
      const normalizedRow = row / this.imageHeight;
      const normalizedCol = col / this.imageWidth;
      features.push(normalizedRow * this.positionWeight);
      features.push(normalizedCol * this.positionWeight);
    }
    
    return features;
  }
  
  /**
   * Initialize cluster centers using k-means++
   */
  private initializeCenters(pixelFeatures: number[][]): void {
    this.clusterCenters = [];
    
    // Choose first center randomly
    const firstIdx = Math.floor(Math.random() * pixelFeatures.length);
    this.clusterCenters.push([...pixelFeatures[firstIdx]]);
    
    // Choose remaining centers using k-means++
    for (let k = 1; k < this.numSegments; k++) {
      const distances: number[] = [];
      
      // Compute distance to nearest center for each point
      for (const features of pixelFeatures) {
        let minDist = Infinity;
        
        for (const center of this.clusterCenters) {
          const dist = euclideanDistance(features, center);
          minDist = Math.min(minDist, dist);
        }
        
        distances.push(minDist * minDist);
      }
      
      // Choose next center with probability proportional to distance squared
      const totalDist = distances.reduce((sum, d) => sum + d, 0);
      let threshold = Math.random() * totalDist;
      
      for (let i = 0; i < distances.length; i++) {
        threshold -= distances[i];
        if (threshold <= 0) {
          this.clusterCenters.push([...pixelFeatures[i]]);
          break;
        }
      }
    }
  }
  
  /**
   * Assign each pixel to nearest cluster
   */
  private assignClusters(pixelFeatures: number[][]): number[] {
    const assignments: number[] = [];
    
    for (const features of pixelFeatures) {
      let minDist = Infinity;
      let bestCluster = 0;
      
      for (let k = 0; k < this.numSegments; k++) {
        const dist = euclideanDistance(features, this.clusterCenters[k]);
        if (dist < minDist) {
          minDist = dist;
          bestCluster = k;
        }
      }
      
      assignments.push(bestCluster);
    }
    
    return assignments;
  }
  
  /**
   * Update cluster centers based on assignments
   */
  private updateCenters(
    pixelFeatures: number[][],
    assignments: number[]
  ): boolean {
    const newCenters: number[][] = [];
    let centersChanged = false;
    
    for (let k = 0; k < this.numSegments; k++) {
      // Find all pixels assigned to this cluster
      const clusterPixels: number[][] = [];
      
      for (let i = 0; i < assignments.length; i++) {
        if (assignments[i] === k) {
          clusterPixels.push(pixelFeatures[i]);
        }
      }
      
      if (clusterPixels.length === 0) {
        // Keep old center if no pixels assigned
        newCenters.push([...this.clusterCenters[k]]);
        continue;
      }
      
      // Compute mean of cluster pixels
      const featureSize = pixelFeatures[0].length;
      const newCenter: number[] = [];
      
      for (let f = 0; f < featureSize; f++) {
        const featureValues = clusterPixels.map(p => p[f]);
        newCenter.push(mean(featureValues));
      }
      
      // Check if center changed
      const oldCenter = this.clusterCenters[k];
      for (let f = 0; f < featureSize; f++) {
        if (Math.abs(newCenter[f] - oldCenter[f]) > 1e-6) {
          centersChanged = true;
          break;
        }
      }
      
      newCenters.push(newCenter);
    }
    
    this.clusterCenters = newCenters;
    return centersChanged;
  }
  
  /**
   * Generate distinct colors for segments
   */
  private generateSegmentColors(): string[] {
    const colors: string[] = [];
    const goldenRatio = 0.618033988749895;
    let hue = Math.random();
    
    for (let i = 0; i < this.numSegments; i++) {
      hue = (hue + goldenRatio) % 1;
      const saturation = 0.6 + Math.random() * 0.2;
      const lightness = 0.5 + Math.random() * 0.2;
      
      colors.push(this.hslToHex(hue, saturation, lightness));
    }
    
    return colors;
  }
  
  /**
   * Convert HSL to hex color
   */
  private hslToHex(h: number, s: number, l: number): string {
    const hueToRgb = (p: number, q: number, t: number): number => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1/6) return p + (q - p) * 6 * t;
      if (t < 1/2) return q;
      if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
      return p;
    };
    
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    
    const r = Math.round(hueToRgb(p, q, h + 1/3) * 255);
    const g = Math.round(hueToRgb(p, q, h) * 255);
    const b = Math.round(hueToRgb(p, q, h - 1/3) * 255);
    
    return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
  }
  
  /**
   * Train the segmentation model
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    this.validateInput(X);
    
    // Convert flat input to 2D image
    const numFeatures = X[0].length;
    const imageSize = Math.sqrt(numFeatures);
    
    if (!Number.isInteger(imageSize)) {
      throw new Error('Input features must form a square image');
    }
    
    this.imageHeight = imageSize;
    this.imageWidth = imageSize;
    
    // Store training data
    this.X_train = X.map(row => [...row]);
    
    // Convert first sample to 2D image
    const image: number[][] = [];
    for (let i = 0; i < imageSize; i++) {
      const row: number[] = [];
      for (let j = 0; j < imageSize; j++) {
        row.push(X[0][i * imageSize + j]);
      }
      image.push(row);
    }
    
    // Extract features for all pixels
    const pixelFeatures: number[][] = [];
    for (let i = 0; i < this.imageHeight; i++) {
      for (let j = 0; j < this.imageWidth; j++) {
        const features = this.extractPixelFeatures(image, i, j);
        pixelFeatures.push(features);
      }
    }
    
    // Initialize cluster centers
    this.initializeCenters(pixelFeatures);
    
    // Generate segment colors
    this.segmentColors = this.generateSegmentColors();
    
    // Reset training history
    this.trainingHistory = {
      epochs: [],
      loss: []
    };
    
    // K-means clustering iterations
    for (let iter = 0; iter < this.maxIterations; iter++) {
      // Assign pixels to clusters
      const assignments = this.assignClusters(pixelFeatures);
      
      // Compute inertia (sum of squared distances to centers)
      let inertia = 0;
      for (let i = 0; i < pixelFeatures.length; i++) {
        const dist = euclideanDistance(
          pixelFeatures[i],
          this.clusterCenters[assignments[i]]
        );
        inertia += dist * dist;
      }
      
      // Store history
      this.trainingHistory.epochs.push(iter);
      this.trainingHistory.loss.push(inertia);
      
      // Update centers
      const centersChanged = this.updateCenters(pixelFeatures, assignments);
      
      // Allow UI to update periodically
      if (iter % 5 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
      
      // Check convergence
      if (!centersChanged) {
        break;
      }
    }
    
    // Create final segmentation mask
    const finalAssignments = this.assignClusters(pixelFeatures);
    this.segmentationMask = [];
    
    let idx = 0;
    for (let i = 0; i < this.imageHeight; i++) {
      const row: number[] = [];
      for (let j = 0; j < this.imageWidth; j++) {
        row.push(finalAssignments[idx++]);
      }
      this.segmentationMask.push(row);
    }
    
    this.trained = true;
  }
  
  /**
   * Segment a new image
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    this.validateInput(X, this.imageHeight * this.imageWidth);
    
    const predictions: number[] = [];
    
    for (const sample of X) {
      // Convert to 2D image
      const image: number[][] = [];
      for (let i = 0; i < this.imageHeight; i++) {
        const row: number[] = [];
        for (let j = 0; j < this.imageWidth; j++) {
          row.push(sample[i * this.imageWidth + j]);
        }
        image.push(row);
      }
      
      // Extract features and assign to clusters
      for (let i = 0; i < this.imageHeight; i++) {
        for (let j = 0; j < this.imageWidth; j++) {
          const features = this.extractPixelFeatures(image, i, j);
          
          let minDist = Infinity;
          let bestCluster = 0;
          
          for (let k = 0; k < this.numSegments; k++) {
            const dist = euclideanDistance(features, this.clusterCenters[k]);
            if (dist < minDist) {
              minDist = dist;
              bestCluster = k;
            }
          }
          
          predictions.push(bestCluster);
        }
      }
    }
    
    return predictions;
  }
  
  /**
   * Get segmentation mask for an image
   */
  getSegmentationMask(X: number[]): SegmentationMask {
    this.ensureTrained();
    
    // Convert to 2D image
    const image: number[][] = [];
    for (let i = 0; i < this.imageHeight; i++) {
      const row: number[] = [];
      for (let j = 0; j < this.imageWidth; j++) {
        row.push(X[i * this.imageWidth + j]);
      }
      image.push(row);
    }
    
    // Segment the image
    const segments: number[][] = [];
    for (let i = 0; i < this.imageHeight; i++) {
      const row: number[] = [];
      for (let j = 0; j < this.imageWidth; j++) {
        const features = this.extractPixelFeatures(image, i, j);
        
        let minDist = Infinity;
        let bestCluster = 0;
        
        for (let k = 0; k < this.numSegments; k++) {
          const dist = euclideanDistance(features, this.clusterCenters[k]);
          if (dist < minDist) {
            minDist = dist;
            bestCluster = k;
          }
        }
        
        row.push(bestCluster);
      }
      segments.push(row);
    }
    
    return {
      width: this.imageWidth,
      height: this.imageHeight,
      segments,
      colors: this.segmentColors
    };
  }
  
  /**
   * Get visualization data for segmentation masks
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained) {
      return [];
    }
    
    const visualizations: VisualizationData[] = [];
    
    // Original image
    if (this.X_train.length > 0) {
      const originalImage: number[][] = [];
      for (let i = 0; i < this.imageHeight; i++) {
        const row: number[] = [];
        for (let j = 0; j < this.imageWidth; j++) {
          row.push(this.X_train[0][i * this.imageWidth + j]);
        }
        originalImage.push(row);
      }
      
      visualizations.push({
        type: 'heatmap',
        data: {
          z: originalImage,
          colorscale: 'Greys',
          showscale: false,
          name: 'Original Image'
        }
      });
    }
    
    // Segmentation mask
    visualizations.push({
      type: 'heatmap',
      data: {
        mask: this.segmentationMask,
        colors: this.segmentColors,
        width: this.imageWidth,
        height: this.imageHeight,
        numSegments: this.numSegments
      }
    });
    
    // Inertia curve
    visualizations.push({
      type: 'line',
      data: {
        x: this.trainingHistory.epochs,
        y: this.trainingHistory.loss,
        mode: 'lines',
        name: 'Inertia',
        line: { color: '#10b981', width: 2 }
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
    
    // Compute final inertia
    const finalInertia = this.trainingHistory.loss[this.trainingHistory.loss.length - 1];
    
    // Compute segment sizes
    const segmentSizes: number[] = Array(this.numSegments).fill(0);
    for (const row of this.segmentationMask) {
      for (const segment of row) {
        segmentSizes[segment]++;
      }
    }
    
    const totalPixels = this.imageHeight * this.imageWidth;
    const segmentPercentages = segmentSizes.map(size => (size / totalPixels) * 100);
    
    // Compute segment compactness (average distance to center)
    const avgDistances: number[] = [];
    
    if (this.X_train.length > 0) {
      const image: number[][] = [];
      for (let i = 0; i < this.imageHeight; i++) {
        const row: number[] = [];
        for (let j = 0; j < this.imageWidth; j++) {
          row.push(this.X_train[0][i * this.imageWidth + j]);
        }
        image.push(row);
      }
      
      for (let k = 0; k < this.numSegments; k++) {
        const distances: number[] = [];
        
        for (let i = 0; i < this.imageHeight; i++) {
          for (let j = 0; j < this.imageWidth; j++) {
            if (this.segmentationMask[i][j] === k) {
              const features = this.extractPixelFeatures(image, i, j);
              const dist = euclideanDistance(features, this.clusterCenters[k]);
              distances.push(dist);
            }
          }
        }
        
        avgDistances.push(distances.length > 0 ? mean(distances) : 0);
      }
    }
    
    return {
      inertia: finalInertia,
      numSegments: this.numSegments,
      segmentSizes,
      segmentPercentages,
      avgDistanceToCenter: avgDistances.length > 0 ? mean(avgDistances) : 0,
      iterations: this.trainingHistory.epochs.length
    };
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.clusterCenters = [];
    this.X_train = [];
    this.imageHeight = 0;
    this.imageWidth = 0;
    this.segmentationMask = [];
    this.segmentColors = [];
    this.trainingHistory = {
      epochs: [],
      loss: []
    };
  }
}
