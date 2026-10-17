/**
 * Autoencoder Neural Network
 * 
 * Implements an autoencoder for unsupervised learning and dimensionality reduction.
 * An autoencoder learns to compress data into a lower-dimensional representation
 * and then reconstruct the original data from this compressed form.
 * 
 * Requirements: 10.1, 10.2, 10.3, 10.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics } from '~/types/algorithm';

export interface AutoencoderParams {
  encoderLayers: number[];
  learningRate: number;
  epochs: number;
  activation: 'sigmoid' | 'relu' | 'tanh';
  regularization?: number;
}

export interface EncoderDecoderState {
  encoderWeights: number[][][];
  decoderWeights: number[][][];
  encoderBiases: number[][];
  decoderBiases: number[][];
  latentRepresentation: number[][];
  reconstructions: number[][];
}

export class Autoencoder extends BaseAlgorithm<number[][], number[], number[][]> {
  name = 'Autoencoder';
  
  // Network architecture
  private encoderLayers: number[] = [];
  private decoderLayers: number[] = [];
  private encoderWeights: number[][][] = [];
  private decoderWeights: number[][][] = [];
  private encoderBiases: number[][] = [];
  private decoderBiases: number[][] = [];
  
  // Training data
  private X_train: number[][] = [];
  
  // Visualization data
  private latentRepresentation: number[][] = [];
  private reconstructions: number[][] = [];
  private reconstructionErrors: number[] = [];
  
  // Hyperparameters
  private learningRate: number;
  private epochs: number;
  private activation: 'sigmoid' | 'relu' | 'tanh';
  private regularization: number;
  
  constructor(params: AutoencoderParams = {
    encoderLayers: [8, 4, 2],
    learningRate: 0.01,
    epochs: 200,
    activation: 'relu',
    regularization: 0.001
  }) {
    super();
    this.encoderLayers = params.encoderLayers;
    this.learningRate = params.learningRate;
    this.epochs = params.epochs;
    this.activation = params.activation;
    this.regularization = params.regularization || 0.001;
  }

  /**
   * Activation functions
   */
  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
  }
  
  private sigmoidDerivative(x: number): number {
    const s = this.sigmoid(x);
    return s * (1 - s);
  }
  
  private relu(x: number): number {
    return Math.max(0, x);
  }
  
  private reluDerivative(x: number): number {
    return x > 0 ? 1 : 0;
  }
  
  private tanh(x: number): number {
    return Math.tanh(x);
  }
  
  private tanhDerivative(x: number): number {
    const t = Math.tanh(x);
    return 1 - t * t;
  }
  
  private activate(x: number): number {
    switch (this.activation) {
      case 'sigmoid':
        return this.sigmoid(x);
      case 'relu':
        return this.relu(x);
      case 'tanh':
        return this.tanh(x);
      default:
        return this.relu(x);
    }
  }
  
  private activateDerivative(x: number): number {
    switch (this.activation) {
      case 'sigmoid':
        return this.sigmoidDerivative(x);
      case 'relu':
        return this.reluDerivative(x);
      case 'tanh':
        return this.tanhDerivative(x);
      default:
        return this.reluDerivative(x);
    }
  }
  
  /**
   * Initialize network weights and biases
   */
  private initializeNetwork(inputSize: number): void {
    // Build encoder layers: [input, hidden1, hidden2, ..., latent]
    const fullEncoderLayers = [inputSize, ...this.encoderLayers];
    
    // Build decoder layers (mirror of encoder): [latent, ..., hidden2, hidden1, output]
    this.decoderLayers = [...this.encoderLayers].reverse();
    this.decoderLayers.push(inputSize);
    
    // Initialize encoder
    this.encoderWeights = [];
    this.encoderBiases = [];
    
    for (let l = 1; l < fullEncoderLayers.length; l++) {
      const layerWeights: number[][] = [];
      const layerBiases: number[] = [];
      
      const prevSize = fullEncoderLayers[l - 1];
      const currSize = fullEncoderLayers[l];
      
      // Xavier initialization
      const scale = Math.sqrt(2.0 / prevSize);
      
      for (let j = 0; j < currSize; j++) {
        const neuronWeights: number[] = [];
        for (let i = 0; i < prevSize; i++) {
          neuronWeights.push((Math.random() - 0.5) * 2 * scale);
        }
        layerWeights.push(neuronWeights);
        layerBiases.push(0);
      }
      
      this.encoderWeights.push(layerWeights);
      this.encoderBiases.push(layerBiases);
    }
    
    // Initialize decoder
    this.decoderWeights = [];
    this.decoderBiases = [];
    
    for (let l = 1; l < this.decoderLayers.length; l++) {
      const layerWeights: number[][] = [];
      const layerBiases: number[] = [];
      
      const prevSize = this.decoderLayers[l - 1];
      const currSize = this.decoderLayers[l];
      
      const scale = Math.sqrt(2.0 / prevSize);
      
      for (let j = 0; j < currSize; j++) {
        const neuronWeights: number[] = [];
        for (let i = 0; i < prevSize; i++) {
          neuronWeights.push((Math.random() - 0.5) * 2 * scale);
        }
        layerWeights.push(neuronWeights);
        layerBiases.push(0);
      }
      
      this.decoderWeights.push(layerWeights);
      this.decoderBiases.push(layerBiases);
    }
  }

  /**
   * Forward pass through encoder
   */
  private encode(input: number[]): number[] {
    let currentActivation = input;
    
    for (let l = 0; l < this.encoderWeights.length; l++) {
      const nextActivation: number[] = [];
      
      for (let j = 0; j < this.encoderWeights[l].length; j++) {
        let z = this.encoderBiases[l][j];
        
        for (let i = 0; i < currentActivation.length; i++) {
          z += this.encoderWeights[l][j][i] * currentActivation[i];
        }
        
        nextActivation.push(this.activate(z));
      }
      
      currentActivation = nextActivation;
    }
    
    return currentActivation;
  }
  
  /**
   * Forward pass through decoder
   */
  private decode(latent: number[]): number[] {
    let currentActivation = latent;
    
    for (let l = 0; l < this.decoderWeights.length; l++) {
      const nextActivation: number[] = [];
      
      for (let j = 0; j < this.decoderWeights[l].length; j++) {
        let z = this.decoderBiases[l][j];
        
        for (let i = 0; i < currentActivation.length; i++) {
          z += this.decoderWeights[l][j][i] * currentActivation[i];
        }
        
        // Use sigmoid for output layer to ensure values in [0,1]
        const a = (l === this.decoderWeights.length - 1) 
          ? this.sigmoid(z)
          : this.activate(z);
        
        nextActivation.push(a);
      }
      
      currentActivation = nextActivation;
    }
    
    return currentActivation;
  }
  
  /**
   * Full forward pass (encode + decode)
   */
  private forward(input: number[]): {
    latent: number[];
    reconstruction: number[];
  } {
    const latent = this.encode(input);
    const reconstruction = this.decode(latent);
    
    return { latent, reconstruction };
  }
  
  /**
   * Compute reconstruction loss (MSE)
   */
  private computeLoss(input: number[], reconstruction: number[]): number {
    let loss = 0;
    for (let i = 0; i < input.length; i++) {
      const diff = input[i] - reconstruction[i];
      loss += diff * diff;
    }
    return loss / input.length;
  }
  
  /**
   * Backward pass (simplified backpropagation)
   */
  private backward(input: number[], reconstruction: number[]): void {
    // Compute output error
    const outputError: number[] = [];
    for (let i = 0; i < input.length; i++) {
      outputError.push(2 * (reconstruction[i] - input[i]) / input.length);
    }
    
    // Backpropagate through decoder
    let currentError = outputError;
    
    for (let l = this.decoderWeights.length - 1; l >= 0; l--) {
      const nextError: number[] = new Array(this.decoderWeights[l][0].length).fill(0);
      
      for (let j = 0; j < this.decoderWeights[l].length; j++) {
        // Update biases
        this.decoderBiases[l][j] -= this.learningRate * currentError[j];
        
        // Update weights and propagate error
        for (let i = 0; i < this.decoderWeights[l][j].length; i++) {
          const gradient = currentError[j];
          this.decoderWeights[l][j][i] -= this.learningRate * gradient;
          nextError[i] += currentError[j] * this.decoderWeights[l][j][i];
        }
      }
      
      currentError = nextError;
    }
    
    // Backpropagate through encoder
    for (let l = this.encoderWeights.length - 1; l >= 0; l--) {
      const nextError: number[] = new Array(this.encoderWeights[l][0].length).fill(0);
      
      for (let j = 0; j < this.encoderWeights[l].length; j++) {
        // Update biases
        this.encoderBiases[l][j] -= this.learningRate * currentError[j];
        
        // Update weights and propagate error
        for (let i = 0; i < this.encoderWeights[l][j].length; i++) {
          const gradient = currentError[j];
          this.encoderWeights[l][j][i] -= this.learningRate * gradient;
          nextError[i] += currentError[j] * this.encoderWeights[l][j][i];
        }
      }
      
      currentError = nextError;
    }
  }

  /**
   * Train the autoencoder
   */
  async fit(X: number[][]): Promise<void> {
    // Validate input
    this.validateInput(X);
    
    // Store training data
    this.X_train = X.map(row => [...row]);
    
    // Initialize network
    const inputSize = X[0].length;
    this.initializeNetwork(inputSize);
    
    // Initialize training history
    this.trainingHistory = {
      epochs: [],
      loss: []
    };
    
    // Training loop
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      let totalLoss = 0;
      
      // Process each sample
      for (let i = 0; i < X.length; i++) {
        const input = X[i];
        
        // Forward pass
        const { latent, reconstruction } = this.forward(input);
        
        // Compute loss
        const loss = this.computeLoss(input, reconstruction);
        totalLoss += loss;
        
        // Backward pass
        this.backward(input, reconstruction);
      }
      
      // Record history
      const avgLoss = totalLoss / X.length;
      this.trainingHistory.epochs.push(epoch);
      this.trainingHistory.loss.push(avgLoss);
      
      // Allow UI to update periodically
      if (epoch % 20 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    // Generate final latent representations and reconstructions
    this.latentRepresentation = X.map(input => this.encode(input));
    this.reconstructions = X.map(input => this.decode(this.encode(input)));
    this.reconstructionErrors = X.map((input, i) => 
      this.computeLoss(input, this.reconstructions[i])
    );
    
    this.trained = true;
  }
  
  /**
   * Encode data to latent space
   */
  predict(X: number[][]): number[][] {
    this.ensureTrained();
    this.validateInput(X, this.X_train[0].length);
    
    return X.map(input => this.encode(input));
  }
  
  /**
   * Reconstruct data from input
   */
  reconstruct(X: number[][]): number[][] {
    this.ensureTrained();
    this.validateInput(X, this.X_train[0].length);
    
    return X.map(input => {
      const latent = this.encode(input);
      return this.decode(latent);
    });
  }
  
  /**
   * Generate new data from latent space
   */
  generate(latentSamples: number[][]): number[][] {
    this.ensureTrained();
    
    return latentSamples.map(latent => this.decode(latent));
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
          name: 'Reconstruction Loss',
          line: { color: '#ef4444', width: 2 }
        }
      }
    });
    
    // Latent space visualization (if 2D or 3D)
    const latentDim = this.latentRepresentation[0]?.length || 0;
    
    if (latentDim === 2) {
      visualizations.push({
        type: 'scatter',
        data: {
          latent: {
            x: this.latentRepresentation.map(point => point[0]),
            y: this.latentRepresentation.map(point => point[1]),
            mode: 'markers',
            name: 'Latent Space',
            marker: { 
              color: this.reconstructionErrors,
              colorscale: 'Viridis',
              size: 8,
              colorbar: { title: 'Reconstruction Error' }
            }
          }
        }
      });
    } else if (latentDim === 3) {
      visualizations.push({
        type: 'scatter3d',
        data: {
          latent: {
            x: this.latentRepresentation.map(point => point[0]),
            y: this.latentRepresentation.map(point => point[1]),
            z: this.latentRepresentation.map(point => point[2]),
            mode: 'markers',
            name: 'Latent Space',
            marker: { 
              color: this.reconstructionErrors,
              colorscale: 'Viridis',
              size: 6,
              colorbar: { title: 'Reconstruction Error' }
            }
          }
        }
      });
    }
    
    // Reconstruction comparison (first few samples)
    const numSamples = Math.min(5, this.X_train.length);
    const sampleIndices = Array.from({length: numSamples}, (_, i) => i);
    
    visualizations.push({
      type: 'heatmap',
      data: {
        original: {
          z: sampleIndices.map(i => this.X_train[i]),
          colorscale: 'Blues',
          name: 'Original Data'
        },
        reconstructed: {
          z: sampleIndices.map(i => this.reconstructions[i]),
          colorscale: 'Reds',
          name: 'Reconstructed Data'
        }
      }
    });
    
    return visualizations;
  }
  
  /**
   * Get metrics
   */
  getMetrics(): Metrics {
    if (!this.trained) {
      return {};
    }
    
    const finalLoss = this.trainingHistory.loss[this.trainingHistory.loss.length - 1];
    const avgReconstructionError = this.reconstructionErrors.reduce((a, b) => a + b, 0) / this.reconstructionErrors.length;
    
    // Compute compression ratio
    const inputDim = this.X_train[0].length;
    const latentDim = this.encoderLayers[this.encoderLayers.length - 1];
    const compressionRatio = inputDim / latentDim;
    
    // Compute total parameters
    let totalParams = 0;
    
    // Encoder parameters
    for (let l = 0; l < this.encoderWeights.length; l++) {
      for (let j = 0; j < this.encoderWeights[l].length; j++) {
        totalParams += this.encoderWeights[l][j].length + 1; // weights + bias
      }
    }
    
    // Decoder parameters
    for (let l = 0; l < this.decoderWeights.length; l++) {
      for (let j = 0; j < this.decoderWeights[l].length; j++) {
        totalParams += this.decoderWeights[l][j].length + 1; // weights + bias
      }
    }
    
    return {
      finalLoss,
      avgReconstructionError,
      compressionRatio,
      inputDimension: inputDim,
      latentDimension: latentDim,
      totalParameters: totalParams,
      architecture: `${inputDim}-${this.encoderLayers.join('-')}-${inputDim}`
    };
  }
  
  /**
   * Get autoencoder state for visualization
   */
  getAutoencoderState(): EncoderDecoderState {
    return {
      encoderWeights: this.encoderWeights.map(layer => layer.map(neuron => [...neuron])),
      decoderWeights: this.decoderWeights.map(layer => layer.map(neuron => [...neuron])),
      encoderBiases: this.encoderBiases.map(layer => [...layer]),
      decoderBiases: this.decoderBiases.map(layer => [...layer]),
      latentRepresentation: this.latentRepresentation.map(point => [...point]),
      reconstructions: this.reconstructions.map(point => [...point])
    };
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.encoderWeights = [];
    this.decoderWeights = [];
    this.encoderBiases = [];
    this.decoderBiases = [];
    this.X_train = [];
    this.latentRepresentation = [];
    this.reconstructions = [];
    this.reconstructionErrors = [];
  }
  
  /**
   * Get model parameters
   */
  getParameters() {
    return {
      encoderLayers: [...this.encoderLayers],
      decoderLayers: [...this.decoderLayers],
      learningRate: this.learningRate,
      epochs: this.epochs,
      activation: this.activation,
      regularization: this.regularization
    };
  }
}