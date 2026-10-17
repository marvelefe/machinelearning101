/**
 * Image Classification Algorithm
 * 
 * Implements basic image classification using a simple neural network approach.
 * For browser-based learning, this uses simplified image features (pixel intensities,
 * color histograms, or pre-extracted features) rather than full CNN processing.
 * 
 * Requirements: 2.2, 2.4, 11.2, 11.3
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics } from '~/types/algorithm';
import { mean, sum } from '~/utils/math';

export interface ImageClassificationParams {
  learningRate: number;
  epochs: number;
  hiddenSize: number;
  regularization?: number;
}

export class ImageClassification extends BaseAlgorithm {
  name = 'Image Classification';
  
  // Model parameters
  private weightsInputHidden: number[][] = [];
  private weightsHiddenOutput: number[][] = [];
  private biasHidden: number[] = [];
  private biasOutput: number[] = [];
  
  // Training data (stored for visualization)
  private X_train: number[][] = [];
  private y_train: number[] = [];
  
  // Model configuration
  private inputSize: number = 0;
  private hiddenSize: number;
  private outputSize: number = 0;
  private numClasses: number = 0;
  
  // Hyperparameters
  private learningRate: number;
  private epochs: number;
  private regularization: number;
  
  // Training history
  protected override trainingHistory = {
    epochs: [] as number[],
    loss: [] as number[],
    accuracy: [] as number[]
  };
  
  constructor(params: ImageClassificationParams = {
    learningRate: 0.01,
    epochs: 100,
    hiddenSize: 10,
    regularization: 0.01
  }) {
    super();
    this.learningRate = params.learningRate;
    this.epochs = params.epochs;
    this.hiddenSize = params.hiddenSize;
    this.regularization = params.regularization || 0.01;
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
   * Initialize weights using Xavier initialization
   */
  private initializeWeights(): void {
    // Input to hidden weights
    this.weightsInputHidden = [];
    for (let i = 0; i < this.inputSize; i++) {
      const row: number[] = [];
      const scale = Math.sqrt(2.0 / this.inputSize);
      for (let j = 0; j < this.hiddenSize; j++) {
        row.push((Math.random() * 2 - 1) * scale);
      }
      this.weightsInputHidden.push(row);
    }
    
    // Hidden to output weights
    this.weightsHiddenOutput = [];
    for (let i = 0; i < this.hiddenSize; i++) {
      const row: number[] = [];
      const scale = Math.sqrt(2.0 / this.hiddenSize);
      for (let j = 0; j < this.outputSize; j++) {
        row.push((Math.random() * 2 - 1) * scale);
      }
      this.weightsHiddenOutput.push(row);
    }
    
    // Biases
    this.biasHidden = new Array(this.hiddenSize).fill(0);
    this.biasOutput = new Array(this.outputSize).fill(0);
  }
  
  /**
   * Forward pass through the network
   */
  private forward(x: number[]): { hidden: number[], hiddenActivated: number[], output: number[], probabilities: number[] } {
    // Input to hidden
    const hidden: number[] = [];
    for (let j = 0; j < this.hiddenSize; j++) {
      let sum = this.biasHidden[j];
      for (let i = 0; i < this.inputSize; i++) {
        sum += x[i] * this.weightsInputHidden[i][j];
      }
      hidden.push(sum);
    }
    
    // Apply ReLU activation
    const hiddenActivated = hidden.map(h => this.relu(h));
    
    // Hidden to output
    const output: number[] = [];
    for (let k = 0; k < this.outputSize; k++) {
      let sum = this.biasOutput[k];
      for (let j = 0; j < this.hiddenSize; j++) {
        sum += hiddenActivated[j] * this.weightsHiddenOutput[j][k];
      }
      output.push(sum);
    }
    
    // Apply softmax
    const probabilities = this.softmax(output);
    
    return { hidden, hiddenActivated, output, probabilities };
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
    
    let loss = totalLoss / m;
    
    // Add L2 regularization
    if (this.regularization > 0) {
      let regTerm = 0;
      
      // Regularize input-hidden weights
      for (let i = 0; i < this.inputSize; i++) {
        for (let j = 0; j < this.hiddenSize; j++) {
          regTerm += this.weightsInputHidden[i][j] ** 2;
        }
      }
      
      // Regularize hidden-output weights
      for (let j = 0; j < this.hiddenSize; j++) {
        for (let k = 0; k < this.outputSize; k++) {
          regTerm += this.weightsHiddenOutput[j][k] ** 2;
        }
      }
      
      loss += (this.regularization / (2 * m)) * regTerm;
    }
    
    return loss;
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
   * Train the model using backpropagation
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
    this.outputSize = this.numClasses;
    
    // Validate labels
    if (this.numClasses < 2) {
      throw new Error('Need at least 2 classes for classification');
    }
    
    // Initialize weights
    this.initializeWeights();
    
    // Reset training history
    this.trainingHistory = {
      epochs: [],
      loss: [],
      accuracy: []
    };
    
    // Training loop
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      // Forward pass for all samples
      const allProbabilities: number[][] = [];
      const allForwardResults: Array<{
        hidden: number[],
        hiddenActivated: number[],
        output: number[],
        probabilities: number[]
      }> = [];
      
      for (let i = 0; i < m; i++) {
        const result = this.forward(X[i]);
        allForwardResults.push(result);
        allProbabilities.push(result.probabilities);
      }
      
      // Compute loss and accuracy
      const loss = this.computeLoss(allProbabilities, y);
      const accuracy = this.computeAccuracy(allProbabilities, y);
      
      // Store history
      this.trainingHistory.epochs.push(epoch);
      this.trainingHistory.loss.push(loss);
      this.trainingHistory.accuracy.push(accuracy);
      
      // Backpropagation
      // Initialize gradient accumulators
      const gradWeightsInputHidden: number[][] = Array(this.inputSize).fill(0).map(() => Array(this.hiddenSize).fill(0));
      const gradWeightsHiddenOutput: number[][] = Array(this.hiddenSize).fill(0).map(() => Array(this.outputSize).fill(0));
      const gradBiasHidden: number[] = Array(this.hiddenSize).fill(0);
      const gradBiasOutput: number[] = Array(this.outputSize).fill(0);
      
      for (let i = 0; i < m; i++) {
        const { hidden, hiddenActivated, probabilities } = allForwardResults[i];
        const trueClass = y[i];
        
        // Output layer gradients
        const outputGrad: number[] = [...probabilities];
        outputGrad[trueClass] -= 1; // Derivative of cross-entropy with softmax
        
        // Gradients for hidden-output weights and output bias
        for (let j = 0; j < this.hiddenSize; j++) {
          for (let k = 0; k < this.outputSize; k++) {
            gradWeightsHiddenOutput[j][k] += hiddenActivated[j] * outputGrad[k];
          }
        }
        
        for (let k = 0; k < this.outputSize; k++) {
          gradBiasOutput[k] += outputGrad[k];
        }
        
        // Hidden layer gradients
        const hiddenGrad: number[] = Array(this.hiddenSize).fill(0);
        for (let j = 0; j < this.hiddenSize; j++) {
          for (let k = 0; k < this.outputSize; k++) {
            hiddenGrad[j] += this.weightsHiddenOutput[j][k] * outputGrad[k];
          }
          hiddenGrad[j] *= this.reluDerivative(hidden[j]);
        }
        
        // Gradients for input-hidden weights and hidden bias
        for (let i_idx = 0; i_idx < this.inputSize; i_idx++) {
          for (let j = 0; j < this.hiddenSize; j++) {
            gradWeightsInputHidden[i_idx][j] += X[i][i_idx] * hiddenGrad[j];
          }
        }
        
        for (let j = 0; j < this.hiddenSize; j++) {
          gradBiasHidden[j] += hiddenGrad[j];
        }
      }
      
      // Update weights with regularization
      for (let i = 0; i < this.inputSize; i++) {
        for (let j = 0; j < this.hiddenSize; j++) {
          const regGrad = this.regularization * this.weightsInputHidden[i][j];
          this.weightsInputHidden[i][j] -= (this.learningRate / m) * (gradWeightsInputHidden[i][j] + regGrad);
        }
      }
      
      for (let j = 0; j < this.hiddenSize; j++) {
        for (let k = 0; k < this.outputSize; k++) {
          const regGrad = this.regularization * this.weightsHiddenOutput[j][k];
          this.weightsHiddenOutput[j][k] -= (this.learningRate / m) * (gradWeightsHiddenOutput[j][k] + regGrad);
        }
      }
      
      // Update biases
      for (let j = 0; j < this.hiddenSize; j++) {
        this.biasHidden[j] -= (this.learningRate / m) * gradBiasHidden[j];
      }
      
      for (let k = 0; k < this.outputSize; k++) {
        this.biasOutput[k] -= (this.learningRate / m) * gradBiasOutput[k];
      }
      
      // Allow UI to update periodically
      if (epoch % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    this.trained = true;
  }
  
  /**
   * Predict class probabilities for new data
   */
  predictProba(X: number[][]): number[][] {
    this.ensureTrained();
    this.validateInput(X, this.inputSize);
    
    const probabilities: number[][] = [];
    for (let i = 0; i < X.length; i++) {
      const result = this.forward(X[i]);
      probabilities.push(result.probabilities);
    }
    
    return probabilities;
  }
  
  /**
   * Make class predictions on new data
   */
  predict(X: number[][]): number[] {
    const probabilities = this.predictProba(X);
    return probabilities.map(probs => probs.indexOf(Math.max(...probs)));
  }
  
  /**
   * Get visualization data for plotting
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained) {
      return [];
    }
    
    const visualizations: VisualizationData[] = [];
    
    // For 2D features, create scatter plot by class
    if (this.inputSize === 2) {
      const x1Values = this.X_train.map(row => row[0]);
      const x2Values = this.X_train.map(row => row[1]);
      
      const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];
      const symbols = ['circle', 'x', 'square', 'diamond', 'triangle-up', 'triangle-down'];
      
      const classData: any = {};
      
      for (let c = 0; c < this.numClasses; c++) {
        const classX1 = x1Values.filter((_, i) => this.y_train[i] === c);
        const classX2 = x2Values.filter((_, i) => this.y_train[i] === c);
        
        classData[`class${c}`] = {
          x: classX1,
          y: classX2,
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
    const matrix: number[][] = Array(this.numClasses).fill(0).map(() => Array(this.numClasses).fill(0));
    
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
          fp += confusionMatrix[i][c]; // Predicted as c but actually i
          fn += confusionMatrix[c][i]; // Actually c but predicted as i
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
      confusionMatrix
    };
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.weightsInputHidden = [];
    this.weightsHiddenOutput = [];
    this.biasHidden = [];
    this.biasOutput = [];
    this.X_train = [];
    this.y_train = [];
    this.inputSize = 0;
    this.outputSize = 0;
    this.numClasses = 0;
    this.trainingHistory = {
      epochs: [],
      loss: [],
      accuracy: []
    };
  }
}
