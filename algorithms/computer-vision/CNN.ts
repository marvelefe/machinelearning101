/**
 * Convolutional Neural Network (CNN)
 * 
 * Implements a simplified CNN for educational purposes with visualization support.
 * Includes convolutional layers, pooling, and fully connected layers.
 * Designed for browser-based learning with filter and feature map visualization.
 * 
 * Requirements: 2.2, 2.4, 2.5
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics } from '~/types/algorithm';
import { mean, sum } from '~/utils/math';

export interface CNNParams {
  numFilters: number;
  filterSize: number;
  poolSize: number;
  learningRate: number;
  epochs: number;
  hiddenSize: number;
}

interface ConvLayer {
  filters: number[][][]; // [filterIdx][row][col]
  biases: number[];
}

interface PoolLayer {
  poolSize: number;
}

interface DenseLayer {
  weights: number[][];
  biases: number[];
}

interface FeatureMap {
  data: number[][];
  filterIndex: number;
}

export class CNN extends BaseAlgorithm {
  name = 'Convolutional Neural Network';
  
  // Network layers
  private convLayer: ConvLayer | null = null;
  private poolLayer: PoolLayer;
  private denseLayer: DenseLayer | null = null;
  private outputLayer: DenseLayer | null = null;
  
  // Training data
  private X_train: number[][][] = []; // [sample][row][col]
  private y_train: number[] = [];
  
  // Input dimensions
  private inputHeight: number = 0;
  private inputWidth: number = 0;
  private numClasses: number = 0;
  
  // Hyperparameters
  private numFilters: number;
  private filterSize: number;
  private poolSize: number;
  private learningRate: number;
  private epochs: number;
  private hiddenSize: number;
  
  // Visualization data
  private currentFeatureMaps: FeatureMap[] = [];
  private currentPooledMaps: FeatureMap[] = [];
  
  constructor(params: CNNParams = {
    numFilters: 4,
    filterSize: 3,
    poolSize: 2,
    learningRate: 0.01,
    epochs: 50,
    hiddenSize: 16
  }) {
    super();
    this.numFilters = params.numFilters;
    this.filterSize = params.filterSize;
    this.poolSize = params.poolSize;
    this.learningRate = params.learningRate;
    this.epochs = params.epochs;
    this.hiddenSize = params.hiddenSize;
    this.poolLayer = { poolSize: this.poolSize };
  }
  
  /**
   * ReLU activation function
   */
  private relu(x: number): number {
    return Math.max(0, x);
  }
  
  /**
   * ReLU derivative
   */
  private reluDerivative(x: number): number {
    return x > 0 ? 1 : 0;
  }
  
  /**
   * Softmax activation for output layer
   */
  private softmax(logits: number[]): number[] {
    const maxLogit = Math.max(...logits);
    const expScores = logits.map(x => Math.exp(x - maxLogit));
    const sumExp = sum(expScores);
    return expScores.map(x => x / sumExp);
  }
  
  /**
   * Initialize convolutional layer filters
   */
  private initializeConvLayer(): void {
    const filters: number[][][] = [];
    const biases: number[] = [];
    
    const scale = Math.sqrt(2.0 / (this.filterSize * this.filterSize));
    
    for (let f = 0; f < this.numFilters; f++) {
      const filter: number[][] = [];
      for (let i = 0; i < this.filterSize; i++) {
        const row: number[] = [];
        for (let j = 0; j < this.filterSize; j++) {
          row.push((Math.random() * 2 - 1) * scale);
        }
        filter.push(row);
      }
      filters.push(filter);
      biases.push(0);
    }
    
    this.convLayer = { filters, biases };
  }
  
  /**
   * Initialize dense layers
   */
  private initializeDenseLayers(inputSize: number): void {
    // Hidden layer
    const hiddenWeights: number[][] = [];
    const hiddenBiases: number[] = [];
    const hiddenScale = Math.sqrt(2.0 / inputSize);
    
    for (let i = 0; i < this.hiddenSize; i++) {
      const row: number[] = [];
      for (let j = 0; j < inputSize; j++) {
        row.push((Math.random() * 2 - 1) * hiddenScale);
      }
      hiddenWeights.push(row);
      hiddenBiases.push(0);
    }
    
    this.denseLayer = { weights: hiddenWeights, biases: hiddenBiases };
    
    // Output layer
    const outputWeights: number[][] = [];
    const outputBiases: number[] = [];
    const outputScale = Math.sqrt(2.0 / this.hiddenSize);
    
    for (let i = 0; i < this.numClasses; i++) {
      const row: number[] = [];
      for (let j = 0; j < this.hiddenSize; j++) {
        row.push((Math.random() * 2 - 1) * outputScale);
      }
      outputWeights.push(row);
      outputBiases.push(0);
    }
    
    this.outputLayer = { weights: outputWeights, biases: outputBiases };
  }
  
  /**
   * Perform convolution operation
   */
  private convolve(input: number[][], filter: number[][], bias: number): number[][] {
    const inputHeight = input.length;
    const inputWidth = input[0].length;
    const outputHeight = inputHeight - this.filterSize + 1;
    const outputWidth = inputWidth - this.filterSize + 1;
    
    const output: number[][] = [];
    
    for (let i = 0; i < outputHeight; i++) {
      const row: number[] = [];
      for (let j = 0; j < outputWidth; j++) {
        let sum = bias;
        
        // Apply filter
        for (let fi = 0; fi < this.filterSize; fi++) {
          for (let fj = 0; fj < this.filterSize; fj++) {
            sum += input[i + fi][j + fj] * filter[fi][fj];
          }
        }
        
        row.push(this.relu(sum));
      }
      output.push(row);
    }
    
    return output;
  }
  
  /**
   * Perform max pooling operation
   */
  private maxPool(input: number[][]): number[][] {
    const inputHeight = input.length;
    const inputWidth = input[0].length;
    const outputHeight = Math.floor(inputHeight / this.poolSize);
    const outputWidth = Math.floor(inputWidth / this.poolSize);
    
    const output: number[][] = [];
    
    for (let i = 0; i < outputHeight; i++) {
      const row: number[] = [];
      for (let j = 0; j < outputWidth; j++) {
        let maxVal = -Infinity;
        
        // Find max in pool window
        for (let pi = 0; pi < this.poolSize; pi++) {
          for (let pj = 0; pj < this.poolSize; pj++) {
            const val = input[i * this.poolSize + pi][j * this.poolSize + pj];
            if (val > maxVal) {
              maxVal = val;
            }
          }
        }
        
        row.push(maxVal);
      }
      output.push(row);
    }
    
    return output;
  }
  
  /**
   * Flatten 3D feature maps to 1D vector
   */
  private flatten(featureMaps: number[][][]): number[] {
    const flattened: number[] = [];
    
    for (const map of featureMaps) {
      for (const row of map) {
        for (const val of row) {
          flattened.push(val);
        }
      }
    }
    
    return flattened;
  }
  
  /**
   * Forward pass through the network
   */
  private forward(input: number[][]): {
    convMaps: number[][][];
    pooledMaps: number[][][];
    flattened: number[];
    hidden: number[];
    hiddenActivated: number[];
    output: number[];
    probabilities: number[];
  } {
    if (!this.convLayer || !this.denseLayer || !this.outputLayer) {
      throw new Error('Network not initialized');
    }
    
    // Convolutional layer
    const convMaps: number[][][] = [];
    for (let f = 0; f < this.numFilters; f++) {
      const convMap = this.convolve(input, this.convLayer.filters[f], this.convLayer.biases[f]);
      convMaps.push(convMap);
    }
    
    // Pooling layer
    const pooledMaps: number[][][] = [];
    for (const convMap of convMaps) {
      const pooledMap = this.maxPool(convMap);
      pooledMaps.push(pooledMap);
    }
    
    // Flatten
    const flattened = this.flatten(pooledMaps);
    
    // Dense hidden layer
    const hidden: number[] = [];
    for (let i = 0; i < this.hiddenSize; i++) {
      let sum = this.denseLayer.biases[i];
      for (let j = 0; j < flattened.length; j++) {
        sum += flattened[j] * this.denseLayer.weights[i][j];
      }
      hidden.push(sum);
    }
    
    const hiddenActivated = hidden.map(h => this.relu(h));
    
    // Output layer
    const output: number[] = [];
    for (let i = 0; i < this.numClasses; i++) {
      let sum = this.outputLayer.biases[i];
      for (let j = 0; j < this.hiddenSize; j++) {
        sum += hiddenActivated[j] * this.outputLayer.weights[i][j];
      }
      output.push(sum);
    }
    
    const probabilities = this.softmax(output);
    
    return {
      convMaps,
      pooledMaps,
      flattened,
      hidden,
      hiddenActivated,
      output,
      probabilities
    };
  }
  
  /**
   * Compute cross-entropy loss
   */
  private computeLoss(probabilities: number[][], y: number[]): number {
    const m = y.length;
    let totalLoss = 0;
    
    for (let i = 0; i < m; i++) {
      const trueClass = y[i];
      const prob = Math.max(1e-15, Math.min(1 - 1e-15, probabilities[i][trueClass]));
      totalLoss += -Math.log(prob);
    }
    
    return totalLoss / m;
  }
  
  /**
   * Compute accuracy
   */
  private computeAccuracy(probabilities: number[][], y: number[]): number {
    let correct = 0;
    for (let i = 0; i < y.length; i++) {
      const predicted = probabilities[i].indexOf(Math.max(...probabilities[i]));
      if (predicted === y[i]) {
        correct++;
      }
    }
    return correct / y.length;
  }
  
  /**
   * Simplified backpropagation (gradient descent on output and dense layers only)
   * Full CNN backpropagation is complex; this is a simplified version for educational purposes
   */
  private backpropagate(
    input: number[][],
    y: number,
    forwardResult: ReturnType<typeof this.forward>
  ): void {
    if (!this.denseLayer || !this.outputLayer) {
      throw new Error('Network not initialized');
    }
    
    const { flattened, hiddenActivated, probabilities } = forwardResult;
    
    // Output layer gradients
    const outputGrad: number[] = [...probabilities];
    outputGrad[y] -= 1;
    
    // Update output layer
    for (let i = 0; i < this.numClasses; i++) {
      this.outputLayer.biases[i] -= this.learningRate * outputGrad[i];
      for (let j = 0; j < this.hiddenSize; j++) {
        this.outputLayer.weights[i][j] -= this.learningRate * outputGrad[i] * hiddenActivated[j];
      }
    }
    
    // Hidden layer gradients
    const hiddenGrad: number[] = Array(this.hiddenSize).fill(0);
    for (let j = 0; j < this.hiddenSize; j++) {
      for (let k = 0; k < this.numClasses; k++) {
        hiddenGrad[j] += this.outputLayer.weights[k][j] * outputGrad[k];
      }
      hiddenGrad[j] *= this.reluDerivative(forwardResult.hidden[j]);
    }
    
    // Update dense layer
    for (let i = 0; i < this.hiddenSize; i++) {
      this.denseLayer.biases[i] -= this.learningRate * hiddenGrad[i];
      for (let j = 0; j < flattened.length; j++) {
        this.denseLayer.weights[i][j] -= this.learningRate * hiddenGrad[i] * flattened[j];
      }
    }
    
    // Note: For simplicity, we're not backpropagating through conv/pool layers
    // In a full implementation, you would compute gradients for filters as well
  }
  
  /**
   * Train the CNN using backpropagation
   */
  async fit(X: number[][], y: number[]): Promise<void> {
    // Validate input
    this.validateInput(X);
    this.validateTarget(X, y);
    
    // Convert flat input to 2D images
    // Assume square images: sqrt(features) x sqrt(features)
    const numFeatures = X[0].length;
    const imageSize = Math.sqrt(numFeatures);
    
    if (!Number.isInteger(imageSize)) {
      throw new Error('Input features must form a square image (e.g., 64 features = 8x8 image)');
    }
    
    this.inputHeight = imageSize;
    this.inputWidth = imageSize;
    
    // Convert to 2D images
    this.X_train = X.map(sample => {
      const image: number[][] = [];
      for (let i = 0; i < imageSize; i++) {
        const row: number[] = [];
        for (let j = 0; j < imageSize; j++) {
          row.push(sample[i * imageSize + j]);
        }
        image.push(row);
      }
      return image;
    });
    
    this.y_train = [...y];
    
    // Determine number of classes
    this.numClasses = Math.max(...y) + 1;
    
    if (this.numClasses < 2) {
      throw new Error('Need at least 2 classes for classification');
    }
    
    // Initialize network
    this.initializeConvLayer();
    
    // Calculate flattened size after conv and pool
    const convHeight = this.inputHeight - this.filterSize + 1;
    const convWidth = this.inputWidth - this.filterSize + 1;
    const poolHeight = Math.floor(convHeight / this.poolSize);
    const poolWidth = Math.floor(convWidth / this.poolSize);
    const flattenedSize = this.numFilters * poolHeight * poolWidth;
    
    this.initializeDenseLayers(flattenedSize);
    
    // Reset training history
    this.trainingHistory = {
      epochs: [],
      loss: [],
      accuracy: []
    };
    
    // Training loop
    const m = this.X_train.length;
    
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      // Forward pass for all samples
      const allProbabilities: number[][] = [];
      
      for (let i = 0; i < m; i++) {
        const forwardResult = this.forward(this.X_train[i]);
        allProbabilities.push(forwardResult.probabilities);
        
        // Backpropagation for this sample
        this.backpropagate(this.X_train[i], this.y_train[i], forwardResult);
      }
      
      // Compute loss and accuracy
      const loss = this.computeLoss(allProbabilities, this.y_train);
      const accuracy = this.computeAccuracy(allProbabilities, this.y_train);
      
      // Store history
      this.trainingHistory.epochs.push(epoch);
      this.trainingHistory.loss.push(loss);
      this.trainingHistory.accuracy!.push(accuracy);
      
      // Allow UI to update periodically
      if (epoch % 5 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    // Store feature maps for visualization (using first training sample)
    if (this.X_train.length > 0) {
      const result = this.forward(this.X_train[0]);
      this.currentFeatureMaps = result.convMaps.map((map, idx) => ({
        data: map,
        filterIndex: idx
      }));
      this.currentPooledMaps = result.pooledMaps.map((map, idx) => ({
        data: map,
        filterIndex: idx
      }));
    }
    
    this.trained = true;
  }
  
  /**
   * Make predictions on new data
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    this.validateInput(X, this.inputHeight * this.inputWidth);
    
    // Convert to 2D images
    const images = X.map(sample => {
      const image: number[][] = [];
      for (let i = 0; i < this.inputHeight; i++) {
        const row: number[] = [];
        for (let j = 0; j < this.inputWidth; j++) {
          row.push(sample[i * this.inputWidth + j]);
        }
        image.push(row);
      }
      return image;
    });
    
    // Predict
    return images.map(image => {
      const result = this.forward(image);
      return result.probabilities.indexOf(Math.max(...result.probabilities));
    });
  }
  
  /**
   * Get visualization data for filter visualization and feature maps
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained || !this.convLayer) {
      return [];
    }
    
    const visualizations: VisualizationData[] = [];
    
    // Convolutional filters visualization
    visualizations.push({
      type: 'heatmap',
      data: {
        filters: this.convLayer.filters.map((filter, idx) => ({
          z: filter,
          name: `Filter ${idx + 1}`,
          type: 'heatmap',
          colorscale: 'RdBu',
          showscale: true
        }))
      }
    });
    
    // Feature maps visualization
    if (this.currentFeatureMaps.length > 0) {
      visualizations.push({
        type: 'heatmap',
        data: {
          featureMaps: this.currentFeatureMaps.map((fm, idx) => ({
            z: fm.data,
            name: `Feature Map ${idx + 1}`,
            type: 'heatmap',
            colorscale: 'Viridis',
            showscale: true
          }))
        }
      });
    }
    
    // Pooled feature maps visualization
    if (this.currentPooledMaps.length > 0) {
      visualizations.push({
        type: 'heatmap',
        data: {
          pooledMaps: this.currentPooledMaps.map((pm, idx) => ({
            z: pm.data,
            name: `Pooled Map ${idx + 1}`,
            type: 'heatmap',
            colorscale: 'Viridis',
            showscale: true
          }))
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
    
    // Convert training images back to flat format for prediction
    const X_flat = this.X_train.map(image => {
      const flat: number[] = [];
      for (const row of image) {
        flat.push(...row);
      }
      return flat;
    });
    
    const predictions = this.predict(X_flat);
    const n = this.y_train.length;
    
    // Confusion matrix
    const confusionMatrix: number[][] = Array(this.numClasses).fill(0)
      .map(() => Array(this.numClasses).fill(0));
    
    for (let i = 0; i < n; i++) {
      const trueClass = this.y_train[i];
      const predClass = predictions[i];
      confusionMatrix[trueClass][predClass]++;
    }
    
    // Overall accuracy
    let correct = 0;
    for (let i = 0; i < n; i++) {
      if (predictions[i] === this.y_train[i]) {
        correct++;
      }
    }
    const accuracy = correct / n;
    
    // Per-class precision and recall
    const precisions: number[] = [];
    const recalls: number[] = [];
    
    for (let c = 0; c < this.numClasses; c++) {
      let tp = confusionMatrix[c][c];
      let fp = 0;
      let fn = 0;
      
      for (let i = 0; i < this.numClasses; i++) {
        if (i !== c) {
          fp += confusionMatrix[i][c];
          fn += confusionMatrix[c][i];
        }
      }
      
      const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      
      precisions.push(precision);
      recalls.push(recall);
    }
    
    // Macro-averaged precision and recall
    const precision = mean(precisions);
    const recall = mean(recalls);
    
    // F1 Score
    const f1Score = precision + recall > 0 ? 
      2 * (precision * recall) / (precision + recall) : 0;
    
    return {
      accuracy,
      precision,
      recall,
      f1Score,
      confusionMatrix,
      architecture: `Conv(${this.numFilters}x${this.filterSize}x${this.filterSize})-Pool(${this.poolSize})-Dense(${this.hiddenSize})-Output(${this.numClasses})`
    };
  }
  
  /**
   * Get filters for visualization
   */
  getFilters(): number[][][] {
    if (!this.convLayer) {
      return [];
    }
    return this.convLayer.filters.map(filter => filter.map(row => [...row]));
  }
  
  /**
   * Get feature maps for a specific input
   */
  getFeatureMaps(input: number[]): FeatureMap[] {
    this.ensureTrained();
    
    // Convert to 2D image
    const image: number[][] = [];
    for (let i = 0; i < this.inputHeight; i++) {
      const row: number[] = [];
      for (let j = 0; j < this.inputWidth; j++) {
        row.push(input[i * this.inputWidth + j]);
      }
      image.push(row);
    }
    
    const result = this.forward(image);
    return result.convMaps.map((map, idx) => ({
      data: map,
      filterIndex: idx
    }));
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.convLayer = null;
    this.denseLayer = null;
    this.outputLayer = null;
    this.X_train = [];
    this.y_train = [];
    this.inputHeight = 0;
    this.inputWidth = 0;
    this.numClasses = 0;
    this.currentFeatureMaps = [];
    this.currentPooledMaps = [];
    this.trainingHistory = {
      epochs: [],
      loss: [],
      accuracy: []
    };
  }
}
