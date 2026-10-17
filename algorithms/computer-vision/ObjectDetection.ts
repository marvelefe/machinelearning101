/**
 * Object Detection Algorithm
 * 
 * Implements a simplified object detection system for educational purposes.
 * Uses a sliding window approach with a simple classifier to detect objects
 * and predict bounding boxes. Designed for browser-based learning with
 * bounding box visualization.
 * 
 * Requirements: 2.2, 2.4, 2.5
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics } from '~/types/algorithm';
import { mean } from '~/utils/math';

export interface ObjectDetectionParams {
  windowSize: number;
  stride: number;
  confidenceThreshold: number;
  iouThreshold: number;
  learningRate: number;
  epochs: number;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  class: number;
  className?: string;
}

interface DetectionSample {
  image: number[][];
  boxes: BoundingBox[];
}

export class ObjectDetection extends BaseAlgorithm {
  name = 'Object Detection';
  
  // Model parameters - simple linear classifier for each window
  private weights: number[] = [];
  private bias: number = 0;
  
  // Training data
  private X_train: DetectionSample[] = [];
  private imageHeight: number = 0;
  private imageWidth: number = 0;
  private numClasses: number = 0;
  
  // Hyperparameters
  private windowSize: number;
  private stride: number;
  private confidenceThreshold: number;
  private iouThreshold: number;
  private learningRate: number;
  private epochs: number;
  
  // Detection results for visualization
  private detections: BoundingBox[] = [];
  private groundTruthBoxes: BoundingBox[] = [];
  
  constructor(params: ObjectDetectionParams = {
    windowSize: 8,
    stride: 4,
    confidenceThreshold: 0.5,
    iouThreshold: 0.3,
    learningRate: 0.01,
    epochs: 50
  }) {
    super();
    this.windowSize = params.windowSize;
    this.stride = params.stride;
    this.confidenceThreshold = params.confidenceThreshold;
    this.iouThreshold = params.iouThreshold;
    this.learningRate = params.learningRate;
    this.epochs = params.epochs;
  }
  
  /**
   * Sigmoid activation function
   */
  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }
  
  /**
   * Initialize model weights
   */
  private initializeWeights(featureSize: number): void {
    this.weights = [];
    const scale = Math.sqrt(2.0 / featureSize);
    
    for (let i = 0; i < featureSize; i++) {
      this.weights.push((Math.random() * 2 - 1) * scale);
    }
    
    this.bias = 0;
  }
  
  /**
   * Extract features from an image window
   */
  private extractWindowFeatures(
    image: number[][],
    x: number,
    y: number,
    size: number
  ): number[] {
    const features: number[] = [];
    
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const row = y + i;
        const col = x + j;
        
        if (row < image.length && col < image[0].length) {
          features.push(image[row][col]);
        } else {
          features.push(0); // Padding
        }
      }
    }
    
    return features;
  }
  
  /**
   * Compute Intersection over Union (IoU) between two boxes
   */
  private computeIoU(box1: BoundingBox, box2: BoundingBox): number {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);
    
    const intersectionWidth = Math.max(0, x2 - x1);
    const intersectionHeight = Math.max(0, y2 - y1);
    const intersectionArea = intersectionWidth * intersectionHeight;
    
    const box1Area = box1.width * box1.height;
    const box2Area = box2.width * box2.height;
    const unionArea = box1Area + box2Area - intersectionArea;
    
    return unionArea > 0 ? intersectionArea / unionArea : 0;
  }
  
  /**
   * Non-Maximum Suppression to remove overlapping boxes
   */
  private nonMaxSuppression(boxes: BoundingBox[]): BoundingBox[] {
    if (boxes.length === 0) return [];
    
    // Sort by confidence (descending)
    const sortedBoxes = [...boxes].sort((a, b) => b.confidence - a.confidence);
    const selectedBoxes: BoundingBox[] = [];
    
    while (sortedBoxes.length > 0) {
      const currentBox = sortedBoxes.shift()!;
      selectedBoxes.push(currentBox);
      
      // Remove boxes with high IoU
      for (let i = sortedBoxes.length - 1; i >= 0; i--) {
        const iou = this.computeIoU(currentBox, sortedBoxes[i]);
        if (iou > this.iouThreshold) {
          sortedBoxes.splice(i, 1);
        }
      }
    }
    
    return selectedBoxes;
  }
  
  /**
   * Check if a window contains an object based on ground truth boxes
   */
  private windowContainsObject(
    x: number,
    y: number,
    size: number,
    groundTruthBoxes: BoundingBox[]
  ): boolean {
    const windowBox: BoundingBox = {
      x,
      y,
      width: size,
      height: size,
      confidence: 1,
      class: 0
    };
    
    for (const gtBox of groundTruthBoxes) {
      const iou = this.computeIoU(windowBox, gtBox);
      if (iou > 0.5) {
        return true;
      }
    }
    
    return false;
  }
  
  /**
   * Predict confidence score for a window
   */
  private predictWindow(features: number[]): number {
    let score = this.bias;
    
    for (let i = 0; i < features.length; i++) {
      score += features[i] * this.weights[i];
    }
    
    return this.sigmoid(score);
  }
  
  /**
   * Train the object detector
   */
  async fit(X: number[][], y: number[]): Promise<void> {
    // For object detection, we expect X to be flattened images
    // and y to encode bounding box information
    this.validateInput(X);
    
    // Convert flat input to 2D images
    const numFeatures = X[0].length;
    const imageSize = Math.sqrt(numFeatures);
    
    if (!Number.isInteger(imageSize)) {
      throw new Error('Input features must form a square image');
    }
    
    this.imageHeight = imageSize;
    this.imageWidth = imageSize;
    
    // Convert to 2D images with bounding boxes
    // For simplicity, we'll create synthetic bounding boxes based on y values
    // In a real scenario, y would contain box coordinates
    this.X_train = X.map((sample, idx) => {
      const image: number[][] = [];
      for (let i = 0; i < imageSize; i++) {
        const row: number[] = [];
        for (let j = 0; j < imageSize; j++) {
          row.push(sample[i * imageSize + j]);
        }
        image.push(row);
      }
      
      // Create a simple bounding box based on the label
      // This is a simplified approach for educational purposes
      const boxes: BoundingBox[] = [];
      if (y[idx] > 0) {
        // Create a box in the center for positive samples
        const boxSize = Math.floor(imageSize * 0.4);
        const boxX = Math.floor((imageSize - boxSize) / 2);
        const boxY = Math.floor((imageSize - boxSize) / 2);
        
        boxes.push({
          x: boxX,
          y: boxY,
          width: boxSize,
          height: boxSize,
          confidence: 1,
          class: y[idx]
        });
      }
      
      return { image, boxes };
    });
    
    this.numClasses = Math.max(...y) + 1;
    
    // Initialize weights
    const featureSize = this.windowSize * this.windowSize;
    this.initializeWeights(featureSize);
    
    // Reset training history
    this.trainingHistory = {
      epochs: [],
      loss: [],
      accuracy: []
    };
    
    // Training loop
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      let totalLoss = 0;
      let correct = 0;
      let total = 0;
      
      // For each training image
      for (const sample of this.X_train) {
        const { image, boxes } = sample;
        
        // Slide window across image
        for (let y = 0; y <= this.imageHeight - this.windowSize; y += this.stride) {
          for (let x = 0; x <= this.imageWidth - this.windowSize; x += this.stride) {
            // Extract features
            const features = this.extractWindowFeatures(image, x, y, this.windowSize);
            
            // Check if window contains object
            const hasObject = this.windowContainsObject(x, y, this.windowSize, boxes);
            const target = hasObject ? 1 : 0;
            
            // Forward pass
            const prediction = this.predictWindow(features);
            
            // Compute loss (binary cross-entropy)
            const loss = target === 1
              ? -Math.log(Math.max(1e-15, prediction))
              : -Math.log(Math.max(1e-15, 1 - prediction));
            
            totalLoss += loss;
            
            // Accuracy
            const predicted = prediction > 0.5 ? 1 : 0;
            if (predicted === target) {
              correct++;
            }
            total++;
            
            // Backpropagation
            const error = prediction - target;
            
            // Update weights
            for (let i = 0; i < features.length; i++) {
              this.weights[i] -= this.learningRate * error * features[i];
            }
            this.bias -= this.learningRate * error;
          }
        }
      }
      
      // Store history
      const avgLoss = totalLoss / total;
      const accuracy = correct / total;
      
      this.trainingHistory.epochs.push(epoch);
      this.trainingHistory.loss.push(avgLoss);
      this.trainingHistory.accuracy!.push(accuracy);
      
      // Allow UI to update periodically
      if (epoch % 5 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    // Run detection on first training sample for visualization
    if (this.X_train.length > 0) {
      const sample = this.X_train[0];
      this.detections = this.detectObjects(sample.image);
      this.groundTruthBoxes = sample.boxes;
    }
    
    this.trained = true;
  }
  
  /**
   * Detect objects in an image
   */
  private detectObjects(image: number[][]): BoundingBox[] {
    const candidateBoxes: BoundingBox[] = [];
    
    // Slide window across image
    for (let y = 0; y <= image.length - this.windowSize; y += this.stride) {
      for (let x = 0; x <= image[0].length - this.windowSize; x += this.stride) {
        // Extract features
        const features = this.extractWindowFeatures(image, x, y, this.windowSize);
        
        // Predict confidence
        const confidence = this.predictWindow(features);
        
        // Add box if confidence exceeds threshold
        if (confidence > this.confidenceThreshold) {
          candidateBoxes.push({
            x,
            y,
            width: this.windowSize,
            height: this.windowSize,
            confidence,
            class: 1,
            className: 'Object'
          });
        }
      }
    }
    
    // Apply non-maximum suppression
    return this.nonMaxSuppression(candidateBoxes);
  }
  
  /**
   * Make predictions on new data
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    this.validateInput(X, this.imageHeight * this.imageWidth);
    
    // Convert to 2D images and detect objects
    const predictions: number[] = [];
    
    for (const sample of X) {
      const image: number[][] = [];
      for (let i = 0; i < this.imageHeight; i++) {
        const row: number[] = [];
        for (let j = 0; j < this.imageWidth; j++) {
          row.push(sample[i * this.imageWidth + j]);
        }
        image.push(row);
      }
      
      const detections = this.detectObjects(image);
      // Return 1 if any objects detected, 0 otherwise
      predictions.push(detections.length > 0 ? 1 : 0);
    }
    
    return predictions;
  }
  
  /**
   * Detect objects and return bounding boxes
   */
  detectWithBoxes(X: number[]): BoundingBox[] {
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
    
    return this.detectObjects(image);
  }
  
  /**
   * Get visualization data for bounding box overlay
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained) {
      return [];
    }
    
    const visualizations: VisualizationData[] = [];
    
    // Bounding box overlay visualization
    if (this.X_train.length > 0) {
      const sample = this.X_train[0];
      
      visualizations.push({
        type: 'heatmap',
        data: {
          image: sample.image,
          detections: this.detections,
          groundTruth: this.groundTruthBoxes,
          imageWidth: this.imageWidth,
          imageHeight: this.imageHeight
        }
      });
    }
    
    // Loss curve
    visualizations.push({
      type: 'line',
      data: {
        x: this.trainingHistory.epochs,
        y: this.trainingHistory.loss,
        mode: 'lines',
        name: 'Training Loss',
        line: { color: '#10b981', width: 2 }
      }
    });
    
    // Accuracy curve
    if (this.trainingHistory.accuracy) {
      visualizations.push({
        type: 'line',
        data: {
          x: this.trainingHistory.epochs,
          y: this.trainingHistory.accuracy,
          mode: 'lines',
          name: 'Training Accuracy',
          line: { color: '#8b5cf6', width: 2 }
      }
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
    
    // Compute detection metrics
    let truePositives = 0;
    let falsePositives = 0;
    let falseNegatives = 0;
    
    for (const sample of this.X_train) {
      const detections = this.detectObjects(sample.image);
      const groundTruth = sample.boxes;
      
      // Match detections to ground truth
      const matchedGT = new Set<number>();
      
      for (const detection of detections) {
        let matched = false;
        
        for (let i = 0; i < groundTruth.length; i++) {
          if (matchedGT.has(i)) continue;
          
          const iou = this.computeIoU(detection, groundTruth[i]);
          if (iou > 0.5) {
            truePositives++;
            matchedGT.add(i);
            matched = true;
            break;
          }
        }
        
        if (!matched) {
          falsePositives++;
        }
      }
      
      // Count unmatched ground truth boxes
      falseNegatives += groundTruth.length - matchedGT.size;
    }
    
    // Calculate metrics
    const precision = truePositives + falsePositives > 0
      ? truePositives / (truePositives + falsePositives)
      : 0;
    
    const recall = truePositives + falseNegatives > 0
      ? truePositives / (truePositives + falseNegatives)
      : 0;
    
    const f1Score = precision + recall > 0
      ? 2 * (precision * recall) / (precision + recall)
      : 0;
    
    const avgDetections = mean(
      this.X_train.map(s => this.detectObjects(s.image).length)
    );
    
    return {
      precision,
      recall,
      f1Score,
      truePositives,
      falsePositives,
      falseNegatives,
      averageDetectionsPerImage: avgDetections,
      confidenceThreshold: this.confidenceThreshold,
      iouThreshold: this.iouThreshold
    };
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.weights = [];
    this.bias = 0;
    this.X_train = [];
    this.imageHeight = 0;
    this.imageWidth = 0;
    this.numClasses = 0;
    this.detections = [];
    this.groundTruthBoxes = [];
    this.trainingHistory = {
      epochs: [],
      loss: [],
      accuracy: []
    };
  }
}
