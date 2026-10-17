/**
 * Transfer Learning Algorithm
 * 
 * Implements transfer learning for educational purposes, demonstrating how
 * pre-trained features can be reused for new classification tasks.
 * Uses a simplified approach with feature extraction and fine-tuning.
 * 
 * Requirements: 2.2, 2.4, 2.5
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics } from '~/types/algorithm';
import { mean, sum } from '~/utils/math';

export interface TransferLearningParams {
  featureExtractorLayers: number;
  fineTuneLayers: number;
  learningRate: number;
  epochs: number;
  freezeFeatures: boolean;
}

interface FeatureExtractor {
  layers: number[][][]; // [layerIdx][neuronIdx][weightIdx]
  biases: number[][];
}

interface Classifier {
  weights: number[][];
  biases: number[];
}

export class TransferLearning extends BaseAlgorithm {
  name = 'Transfer Learning';
  
  // Pre-trained feature extractor (simulated)
  private featureExtractor: FeatureExtractor | null = null;
  
  // Task-specific classifier
  private classifier: Classifier | null = null;
  
  // Training data
  private X_train: number[][] = [];
  private y_train: number[] = [];
  
  // Extracted features (cached)
  private extractedFeatures: number[][] = [];
  
  // Model configuration
  private inputSize: number = 0;
  private featureSize: number = 64; // Size of extracted features
  private numClasses: number = 0;
  
  // Hyperparameters
  private featureExtractorLayers: number;
  private fineTuneLayers: number;
  private learningRate: number;
  private epochs: number;
  private freezeFeatures: boolean;
  
  // Training history
  protected override trainingHistory = {
    epochs: [] as number[],
    loss: [] as number[],
    accuracy: [] as number[],
    featureExtractorLoss: [] as number[]
  };
  
  constructor(params: TransferLearningParams = {
    featureExtractorLayers: 2,
    fineTuneLayers: 1,
    learningRate: 0.001,
    epochs: 50,
    freezeFeatures: true
  }) {
    super();
    this.featureExtractorLayers = params.featureExtractorLayers;
    this.fineTuneLayers = params.fineTuneLayers;
    this.learningRate = params.learningRate;
    this.epochs = params.epochs;
    this.freezeFeatures = params.freezeFeatures;
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
   * Initialize pre-trained feature extractor (simulated)
   * In a real scenario, this would load weights from a pre-trained model
   */
  private initializeFeatureExtractor(): void {
    const layers: number[][][] = [];
    const biases: number[][] = [];
    
    let currentSize = this.inputSize;
    
    for (let l = 0; l < this.featureExtractorLayers; l++) {
      const nextSize = l === this.featureExtractorLayers - 1 ? 
        this.featureSize : 
        Math.floor(currentSize * 0.75);
      
      const layerWeights: number[][] = [];
      const layerBiases: number[] = [];
      const scale = Math.sqrt(2.0 / currentSize);
      
      for (let i = 0; i < nextSize; i++) {
        const neuronWeights: number[] = [];
        for (let j = 0; j < currentSize; j++) {
          // Simulate pre-trained weights with specific patterns
          neuronWeights.push((Math.random() * 2 - 1) * scale);
        }
        layerWeights.push(neuronWeights);
        layerBiases.push((Math.random() * 2 - 1) * 0.1);
      }
      
      layers.push(layerWeights);
      biases.push(layerBiases);
      currentSize = nextSize;
    }
    
    this.featureExtractor = { layers, biases };
  }
  
  /**
   * Initialize task-specific classifier
   */
  private initializeClassifier(): void {
    const weights: number[][] = [];
    const biases: number[] = [];
    const scale = Math.sqrt(2.0 / this.featureSize);
    
    for (let i = 0; i < this.numClasses; i++) {
      const row: number[] = [];
      for (let j = 0; j < this.featureSize; j++) {
        row.push((Math.random() * 2 - 1) * scale);
      }
      weights.push(row);
      biases.push(0);
    }
    
    this.classifier = { weights, biases };
  }
  
  /**
   * Extract features using the feature extractor
   */
  private extractFeatures(x: number[]): number[] {
    if (!this.featureExtractor) {
      throw new Error('Feature extractor not initialized');
    }
    
    let activation = [...x];
    
    // Forward pass through feature extractor layers
    for (let l = 0; l < this.featureExtractorLayers; l++) {
      const layer = this.featureExtractor.layers[l];
      const bias = this.featureExtractor.biases[l];
      const nextActivation: number[] = [];
      
      for (let i = 0; i < layer.length; i++) {
        let sum = bias[i];
        for (let j = 0; j < activation.length; j++) {
          sum += activation[j] * layer[i][j];
        }
        nextActivation.push(this.relu(sum));
      }
      
      activation = nextActivation;
    }
    
    return activation;
  }
  
  /**
   * Forward pass through classifier
   */
  private classifyFeatures(features: number[]): number[] {
    if (!this.classifier) {
      throw new Error('Classifier not initialized');
    }
    
    const logits: number[] = [];
    
    for (let i = 0; i < this.numClasses; i++) {
      let sum = this.classifier.biases[i];
      for (let j = 0; j < features.length; j++) {
        sum += features[j] * this.classifier.weights[i][j];
      }
      logits.push(sum);
    }
    
    return this.softmax(logits);
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
   * Update classifier weights using gradient descent
   */
  private updateClassifier(
    features: number[],
    probabilities: number[],
    trueClass: number
  ): void {
    if (!this.classifier) {
      throw new Error('Classifier not initialized');
    }
    
    // Compute output gradient
    const outputGrad = [...probabilities];
    outputGrad[trueClass] -= 1;
    
    // Update classifier weights and biases
    for (let i = 0; i < this.numClasses; i++) {
      this.classifier.biases[i] -= this.learningRate * outputGrad[i];
      for (let j = 0; j < this.featureSize; j++) {
        this.classifier.weights[i][j] -= this.learningRate * outputGrad[i] * features[j];
      }
    }
  }
  
  /**
   * Fine-tune feature extractor (if not frozen)
   */
  private fineTuneFeatureExtractor(
    x: number[],
    features: number[],
    classifierGrad: number[]
  ): void {
    if (!this.featureExtractor || this.freezeFeatures) {
      return;
    }
    
    // Simplified fine-tuning: only update the last layer of feature extractor
    const lastLayerIdx = this.featureExtractorLayers - 1;
    const lastLayer = this.featureExtractor.layers[lastLayerIdx];
    const lastBias = this.featureExtractor.biases[lastLayerIdx];
    
    // Compute gradient for last feature layer
    const featureGrad: number[] = Array(this.featureSize).fill(0);
    
    for (let i = 0; i < this.featureSize; i++) {
      for (let c = 0; c < this.numClasses; c++) {
        if (this.classifier) {
          featureGrad[i] += this.classifier.weights[c][i] * classifierGrad[c];
        }
      }
      featureGrad[i] *= this.reluDerivative(features[i]);
    }
    
    // Get input to last layer (would need to cache activations in full implementation)
    // For simplicity, we'll use a small learning rate and update based on current features
    const fineTuneLR = this.learningRate * 0.1; // Smaller LR for fine-tuning
    
    for (let i = 0; i < lastLayer.length; i++) {
      lastBias[i] -= fineTuneLR * featureGrad[i];
      // Note: Full implementation would need cached activations from previous layer
    }
  }
  
  /**
   * Train the model using transfer learning
   */
  async fit(X: number[][], y: number[]): Promise<void> {
    // Validate input
    this.validateInput(X);
    this.validateTarget(X, y);
    
    // Store training data
    this.X_train = X.map(row => [...row]);
    this.y_train = [...y];
    
    // Determine dimensions
    const m = X.length;
    this.inputSize = X[0].length;
    this.numClasses = Math.max(...y) + 1;
    
    if (this.numClasses < 2) {
      throw new Error('Need at least 2 classes for classification');
    }
    
    // Initialize feature extractor (simulating pre-trained model)
    this.initializeFeatureExtractor();
    
    // Extract features from all training samples
    this.extractedFeatures = X.map(x => this.extractFeatures(x));
    
    // Initialize task-specific classifier
    this.initializeClassifier();
    
    // Reset training history
    this.trainingHistory = {
      epochs: [],
      loss: [],
      accuracy: [],
      featureExtractorLoss: []
    };
    
    // Training loop
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      const allProbabilities: number[][] = [];
      
      // Forward pass and update for each sample
      for (let i = 0; i < m; i++) {
        const features = this.extractedFeatures[i];
        const probabilities = this.classifyFeatures(features);
        allProbabilities.push(probabilities);
        
        // Update classifier
        this.updateClassifier(features, probabilities, y[i]);
        
        // Fine-tune feature extractor if not frozen
        if (!this.freezeFeatures) {
          const outputGrad = [...probabilities];
          outputGrad[y[i]] -= 1;
          this.fineTuneFeatureExtractor(X[i], features, outputGrad);
        }
      }
      
      // Compute metrics
      const loss = this.computeLoss(allProbabilities, y);
      const accuracy = this.computeAccuracy(allProbabilities, y);
      
      // Store history
      this.trainingHistory.epochs.push(epoch);
      this.trainingHistory.loss.push(loss);
      this.trainingHistory.accuracy.push(accuracy);
      
      // Allow UI to update periodically
      if (epoch % 5 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    this.trained = true;
  }
  
  /**
   * Make predictions on new data
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    this.validateInput(X, this.inputSize);
    
    return X.map(x => {
      const features = this.extractFeatures(x);
      const probabilities = this.classifyFeatures(features);
      return probabilities.indexOf(Math.max(...probabilities));
    });
  }
  
  /**
   * Get visualization data
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained) {
      return [];
    }
    
    const visualizations: VisualizationData[] = [];
    
    // Feature space visualization (2D projection of extracted features)
    if (this.extractedFeatures.length > 0) {
      // Use first 2 dimensions of features for visualization
      const feature1 = this.extractedFeatures.map(f => f[0]);
      const feature2 = this.extractedFeatures.map(f => f[1]);
      
      const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];
      const symbols = ['circle', 'x', 'square', 'diamond', 'triangle-up', 'triangle-down'];
      
      const classData: any = {};
      
      for (let c = 0; c < this.numClasses; c++) {
        const classFeature1 = feature1.filter((_, i) => this.y_train[i] === c);
        const classFeature2 = feature2.filter((_, i) => this.y_train[i] === c);
        
        classData[`class${c}`] = {
          x: classFeature1,
          y: classFeature2,
          mode: 'markers',
          name: `Class ${c}`,
          marker: {
            color: colors[c % colors.length],
            size: 8,
            symbol: symbols[c % symbols.length]
          }
        };
      }
      
      visualizations.push({
        type: 'scatter2d',
        data: classData
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
    
    // Confusion matrix
    const confusionMatrix = this.computeConfusionMatrix();
    visualizations.push({
      type: 'confusion-matrix',
      data: confusionMatrix
    });
    
    return visualizations;
  }
  
  /**
   * Compute confusion matrix
   */
  private computeConfusionMatrix(): number[][] {
    const predictions = this.predict(this.X_train);
    const matrix: number[][] = Array(this.numClasses).fill(0)
      .map(() => Array(this.numClasses).fill(0));
    
    for (let i = 0; i < this.y_train.length; i++) {
      const trueClass = this.y_train[i];
      const predClass = predictions[i];
      matrix[trueClass][predClass]++;
    }
    
    return matrix;
  }
  
  /**
   * Calculate performance metrics
   */
  getMetrics(): Metrics {
    if (!this.trained) {
      return {};
    }
    
    const predictions = this.predict(this.X_train);
    const n = this.y_train.length;
    
    // Confusion matrix
    const confusionMatrix = this.computeConfusionMatrix();
    
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
      transferLearningMode: this.freezeFeatures ? 'Feature Extraction' : 'Fine-Tuning',
      featureSize: this.featureSize
    };
  }
  
  /**
   * Get extracted features for visualization
   */
  getExtractedFeatures(): number[][] {
    return this.extractedFeatures.map(f => [...f]);
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.featureExtractor = null;
    this.classifier = null;
    this.X_train = [];
    this.y_train = [];
    this.extractedFeatures = [];
    this.inputSize = 0;
    this.numClasses = 0;
    this.trainingHistory = {
      epochs: [],
      loss: [],
      accuracy: [],
      featureExtractorLoss: []
    };
  }
}
