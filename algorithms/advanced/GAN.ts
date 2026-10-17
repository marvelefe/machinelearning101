/**
 * Generative Adversarial Network (GAN)
 * 
 * Implements a basic GAN with generator and discriminator networks.
 * The generator learns to create realistic data while the discriminator
 * learns to distinguish between real and generated data.
 * 
 * Requirements: 10.1, 10.2, 10.3, 10.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics } from '~/types/algorithm';

export interface GANParams {
  generatorLayers: number[];
  discriminatorLayers: number[];
  latentDim: number;
  learningRate: number;
  epochs: number;
  batchSize: number;
  activation: 'relu' | 'leaky_relu' | 'tanh';
}

export interface GANState {
  generatorWeights: number[][][];
  discriminatorWeights: number[][][];
  generatorBiases: number[][];
  discriminatorBiases: number[][];
  generatedSamples: number[][];
  discriminatorScores: { real: number[]; fake: number[] };
}

// GAN generates samples with `predict(numSamples)`, so its fit/predict input
// types are unrelated; widen the generic input to cover both.
export class GAN extends BaseAlgorithm<number | number[][], number[], number[][]> {
  name = 'Generative Adversarial Network';
  
  // Network architectures
  private generatorLayers: number[] = [];
  private discriminatorLayers: number[] = [];
  private latentDim: number;
  
  // Generator network
  private generatorWeights: number[][][] = [];
  private generatorBiases: number[][] = [];
  
  // Discriminator network
  private discriminatorWeights: number[][][] = [];
  private discriminatorBiases: number[][] = [];
  
  // Training data
  private X_train: number[][] = [];
  
  // Generated samples and scores
  private generatedSamples: number[][] = [];
  private discriminatorScores: { real: number[]; fake: number[] } = { real: [], fake: [] };
  
  // Training history
  private generatorLoss: number[] = [];
  private discriminatorLoss: number[] = [];
  
  // Hyperparameters
  private learningRate: number;
  private epochs: number;
  private batchSize: number;
  private activation: 'relu' | 'leaky_relu' | 'tanh';
  
  constructor(params: GANParams = {
    generatorLayers: [64, 32],
    discriminatorLayers: [32, 16],
    latentDim: 10,
    learningRate: 0.0002,
    epochs: 200,
    batchSize: 32,
    activation: 'leaky_relu'
  }) {
    super();
    this.generatorLayers = params.generatorLayers;
    this.discriminatorLayers = params.discriminatorLayers;
    this.latentDim = params.latentDim;
    this.learningRate = params.learningRate;
    this.epochs = params.epochs;
    this.batchSize = params.batchSize;
    this.activation = params.activation;
  }

  /**
   * Activation functions
   */
  private relu(x: number): number {
    return Math.max(0, x);
  }
  
  private reluDerivative(x: number): number {
    return x > 0 ? 1 : 0;
  }
  
  private leakyRelu(x: number, alpha: number = 0.2): number {
    return x > 0 ? x : alpha * x;
  }
  
  private leakyReluDerivative(x: number, alpha: number = 0.2): number {
    return x > 0 ? 1 : alpha;
  }
  
  private tanh(x: number): number {
    return Math.tanh(x);
  }
  
  private tanhDerivative(x: number): number {
    const t = Math.tanh(x);
    return 1 - t * t;
  }
  
  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
  }
  
  private sigmoidDerivative(x: number): number {
    const s = this.sigmoid(x);
    return s * (1 - s);
  }
  
  private activate(x: number): number {
    switch (this.activation) {
      case 'relu':
        return this.relu(x);
      case 'leaky_relu':
        return this.leakyRelu(x);
      case 'tanh':
        return this.tanh(x);
      default:
        return this.leakyRelu(x);
    }
  }
  
  private activateDerivative(x: number): number {
    switch (this.activation) {
      case 'relu':
        return this.reluDerivative(x);
      case 'leaky_relu':
        return this.leakyReluDerivative(x);
      case 'tanh':
        return this.tanhDerivative(x);
      default:
        return this.leakyReluDerivative(x);
    }
  }
  
  /**
   * Initialize network weights and biases
   */
  private initializeGenerator(outputDim: number): void {
    const fullLayers = [this.latentDim, ...this.generatorLayers, outputDim];
    
    this.generatorWeights = [];
    this.generatorBiases = [];
    
    for (let l = 1; l < fullLayers.length; l++) {
      const layerWeights: number[][] = [];
      const layerBiases: number[] = [];
      
      const prevSize = fullLayers[l - 1];
      const currSize = fullLayers[l];
      
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
      
      this.generatorWeights.push(layerWeights);
      this.generatorBiases.push(layerBiases);
    }
  }
  
  private initializeDiscriminator(inputDim: number): void {
    const fullLayers = [inputDim, ...this.discriminatorLayers, 1];
    
    this.discriminatorWeights = [];
    this.discriminatorBiases = [];
    
    for (let l = 1; l < fullLayers.length; l++) {
      const layerWeights: number[][] = [];
      const layerBiases: number[] = [];
      
      const prevSize = fullLayers[l - 1];
      const currSize = fullLayers[l];
      
      const scale = Math.sqrt(2.0 / prevSize);
      
      for (let j = 0; j < currSize; j++) {
        const neuronWeights: number[] = [];
        for (let i = 0; i < prevSize; i++) {
          neuronWeights.push((Math.random() - 0.5) * 2 * scale);
        }
        layerWeights.push(neuronWeights);
        layerBiases.push(0);
      }
      
      this.discriminatorWeights.push(layerWeights);
      this.discriminatorBiases.push(layerBiases);
    }
  }

  /**
   * Forward pass through generator
   */
  private generateSample(noise: number[]): number[] {
    let currentActivation = noise;
    
    for (let l = 0; l < this.generatorWeights.length; l++) {
      const nextActivation: number[] = [];
      
      for (let j = 0; j < this.generatorWeights[l].length; j++) {
        let z = this.generatorBiases[l][j];
        
        for (let i = 0; i < currentActivation.length; i++) {
          z += this.generatorWeights[l][j][i] * currentActivation[i];
        }
        
        // Use tanh for output layer to ensure values in [-1, 1]
        const a = (l === this.generatorWeights.length - 1) 
          ? this.tanh(z)
          : this.activate(z);
        
        nextActivation.push(a);
      }
      
      currentActivation = nextActivation;
    }
    
    return currentActivation;
  }
  
  /**
   * Forward pass through discriminator
   */
  private discriminate(sample: number[]): number {
    let currentActivation = sample;
    
    for (let l = 0; l < this.discriminatorWeights.length; l++) {
      const nextActivation: number[] = [];
      
      for (let j = 0; j < this.discriminatorWeights[l].length; j++) {
        let z = this.discriminatorBiases[l][j];
        
        for (let i = 0; i < currentActivation.length; i++) {
          z += this.discriminatorWeights[l][j][i] * currentActivation[i];
        }
        
        // Use sigmoid for output layer (probability)
        const a = (l === this.discriminatorWeights.length - 1) 
          ? this.sigmoid(z)
          : this.activate(z);
        
        nextActivation.push(a);
      }
      
      currentActivation = nextActivation;
    }
    
    return currentActivation[0];
  }
  
  /**
   * Generate random noise vector
   */
  private generateNoise(): number[] {
    const noise: number[] = [];
    for (let i = 0; i < this.latentDim; i++) {
      noise.push((Math.random() - 0.5) * 2); // Uniform [-1, 1]
    }
    return noise;
  }
  
  /**
   * Sample random batch from training data
   */
  private sampleBatch(data: number[][], batchSize: number): number[][] {
    const batch: number[][] = [];
    for (let i = 0; i < batchSize; i++) {
      const idx = Math.floor(Math.random() * data.length);
      batch.push([...data[idx]]);
    }
    return batch;
  }
  
  /**
   * Update discriminator weights
   */
  private updateDiscriminator(realBatch: number[][], fakeBatch: number[][]): number {
    let totalLoss = 0;
    
    // Train on real samples (target = 1)
    for (const sample of realBatch) {
      const prediction = this.discriminate(sample);
      const loss = -Math.log(prediction + 1e-10);
      totalLoss += loss;
      
      // Simplified gradient update (real samples should get score 1)
      const error = prediction - 1;
      this.backpropagateDiscriminator(sample, error);
    }
    
    // Train on fake samples (target = 0)
    for (const sample of fakeBatch) {
      const prediction = this.discriminate(sample);
      const loss = -Math.log(1 - prediction + 1e-10);
      totalLoss += loss;
      
      // Simplified gradient update (fake samples should get score 0)
      const error = prediction - 0;
      this.backpropagateDiscriminator(sample, error);
    }
    
    return totalLoss / (realBatch.length + fakeBatch.length);
  }
  
  /**
   * Update generator weights
   */
  private updateGenerator(batchSize: number): number {
    let totalLoss = 0;
    
    for (let i = 0; i < batchSize; i++) {
      const noise = this.generateNoise();
      const fakeSample = this.generateSample(noise);
      const discriminatorScore = this.discriminate(fakeSample);
      
      // Generator wants discriminator to output 1 (think fake is real)
      const loss = -Math.log(discriminatorScore + 1e-10);
      totalLoss += loss;
      
      // Backpropagate through both networks
      const error = discriminatorScore - 1;
      this.backpropagateGenerator(noise, fakeSample, error);
    }
    
    return totalLoss / batchSize;
  }
  
  /**
   * Simplified backpropagation for discriminator
   */
  private backpropagateDiscriminator(sample: number[], error: number): void {
    // Very simplified gradient update
    const learningRate = this.learningRate * 0.5; // Slower learning for discriminator
    
    // Update output layer
    const lastLayer = this.discriminatorWeights.length - 1;
    for (let j = 0; j < this.discriminatorWeights[lastLayer].length; j++) {
      this.discriminatorBiases[lastLayer][j] -= learningRate * error;
      
      for (let i = 0; i < this.discriminatorWeights[lastLayer][j].length; i++) {
        this.discriminatorWeights[lastLayer][j][i] -= learningRate * error * sample[i];
      }
    }
  }
  
  /**
   * Simplified backpropagation for generator
   */
  private backpropagateGenerator(noise: number[], fakeSample: number[], error: number): void {
    // Very simplified gradient update
    const learningRate = this.learningRate;
    
    // Update output layer
    const lastLayer = this.generatorWeights.length - 1;
    for (let j = 0; j < this.generatorWeights[lastLayer].length; j++) {
      this.generatorBiases[lastLayer][j] -= learningRate * error;
      
      for (let i = 0; i < this.generatorWeights[lastLayer][j].length; i++) {
        this.generatorWeights[lastLayer][j][i] -= learningRate * error * noise[i];
      }
    }
  }

  /**
   * Train the GAN
   */
  async fit(X: number[][]): Promise<void> {
    // Validate input
    this.validateInput(X);
    
    // Store training data (normalize to [-1, 1])
    this.X_train = X.map(row => row.map(val => (val - 0.5) * 2));
    
    // Initialize networks
    const dataDim = X[0].length;
    this.initializeGenerator(dataDim);
    this.initializeDiscriminator(dataDim);
    
    // Initialize training history
    this.trainingHistory = {
      epochs: [],
      loss: []
    };
    
    this.generatorLoss = [];
    this.discriminatorLoss = [];
    
    // Training loop
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      let epochGenLoss = 0;
      let epochDiscLoss = 0;
      
      const numBatches = Math.ceil(this.X_train.length / this.batchSize);
      
      for (let batch = 0; batch < numBatches; batch++) {
        // Get real batch
        const realBatch = this.sampleBatch(this.X_train, this.batchSize);
        
        // Generate fake batch
        const fakeBatch: number[][] = [];
        for (let i = 0; i < this.batchSize; i++) {
          const noise = this.generateNoise();
          const fakeSample = this.generateSample(noise);
          fakeBatch.push(fakeSample);
        }
        
        // Update discriminator
        const discLoss = this.updateDiscriminator(realBatch, fakeBatch);
        epochDiscLoss += discLoss;
        
        // Update generator
        const genLoss = this.updateGenerator(this.batchSize);
        epochGenLoss += genLoss;
      }
      
      // Record history
      const avgGenLoss = epochGenLoss / numBatches;
      const avgDiscLoss = epochDiscLoss / numBatches;
      
      this.trainingHistory.epochs.push(epoch);
      this.trainingHistory.loss.push(avgGenLoss + avgDiscLoss);
      this.generatorLoss.push(avgGenLoss);
      this.discriminatorLoss.push(avgDiscLoss);
      
      // Allow UI to update periodically
      if (epoch % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    // Generate final samples for visualization
    this.generateFinalSamples();
    
    this.trained = true;
  }
  
  /**
   * Generate samples from the trained generator
   */
  predict(numSamples: number = 100): number[][] {
    this.ensureTrained();
    
    const samples: number[][] = [];
    for (let i = 0; i < numSamples; i++) {
      const noise = this.generateNoise();
      const sample = this.generateSample(noise);
      // Denormalize from [-1, 1] to [0, 1]
      const denormalized = sample.map(val => (val + 1) / 2);
      samples.push(denormalized);
    }
    
    return samples;
  }
  
  /**
   * Generate final samples and compute discriminator scores
   */
  private generateFinalSamples(): void {
    const numSamples = Math.min(100, this.X_train.length);
    
    // Generate fake samples
    this.generatedSamples = [];
    for (let i = 0; i < numSamples; i++) {
      const noise = this.generateNoise();
      const sample = this.generateSample(noise);
      this.generatedSamples.push(sample);
    }
    
    // Compute discriminator scores
    this.discriminatorScores.real = [];
    this.discriminatorScores.fake = [];
    
    // Scores for real samples
    for (let i = 0; i < Math.min(numSamples, this.X_train.length); i++) {
      const score = this.discriminate(this.X_train[i]);
      this.discriminatorScores.real.push(score);
    }
    
    // Scores for fake samples
    for (const sample of this.generatedSamples) {
      const score = this.discriminate(sample);
      this.discriminatorScores.fake.push(score);
    }
  }

  /**
   * Get visualization data
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained) {
      return [];
    }
    
    const visualizations: VisualizationData[] = [];
    
    // Loss curves
    visualizations.push({
      type: 'line',
      data: {
        generatorLoss: {
          x: this.trainingHistory.epochs,
          y: this.generatorLoss,
          mode: 'lines',
          name: 'Generator Loss',
          line: { color: '#ef4444', width: 2 }
        },
        discriminatorLoss: {
          x: this.trainingHistory.epochs,
          y: this.discriminatorLoss,
          mode: 'lines',
          name: 'Discriminator Loss',
          line: { color: '#3b82f6', width: 2 }
        }
      }
    });
    
    // Discriminator scores distribution
    visualizations.push({
      type: 'histogram',
      data: {
        real_scores: {
          x: this.discriminatorScores.real,
          name: 'Real Data Scores',
          opacity: 0.7,
          marker: { color: '#10b981' }
        },
        fake_scores: {
          x: this.discriminatorScores.fake,
          name: 'Generated Data Scores',
          opacity: 0.7,
          marker: { color: '#f59e0b' }
        }
      }
    });
    
    // Generated vs Real data comparison (if 2D)
    if (this.X_train[0].length === 2) {
      const realSample = this.X_train.slice(0, 100);
      const fakeSample = this.generatedSamples.slice(0, 100);
      
      visualizations.push({
        type: 'scatter',
        data: {
          realData: {
            x: realSample.map(point => (point[0] + 1) / 2), // Denormalize
            y: realSample.map(point => (point[1] + 1) / 2),
            mode: 'markers',
            name: 'Real Data',
            marker: { color: '#10b981', size: 6, opacity: 0.7 }
          },
          generatedData: {
            x: fakeSample.map(point => (point[0] + 1) / 2),
            y: fakeSample.map(point => (point[1] + 1) / 2),
            mode: 'markers',
            name: 'Generated Data',
            marker: { color: '#f59e0b', size: 6, opacity: 0.7 }
          }
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
    
    const finalGenLoss = this.generatorLoss[this.generatorLoss.length - 1];
    const finalDiscLoss = this.discriminatorLoss[this.discriminatorLoss.length - 1];
    
    // Compute discriminator accuracy
    const realAccuracy = this.discriminatorScores.real.filter(score => score > 0.5).length / this.discriminatorScores.real.length;
    const fakeAccuracy = this.discriminatorScores.fake.filter(score => score <= 0.5).length / this.discriminatorScores.fake.length;
    const discriminatorAccuracy = (realAccuracy + fakeAccuracy) / 2;
    
    // Compute average discriminator scores
    const avgRealScore = this.discriminatorScores.real.reduce((a, b) => a + b, 0) / this.discriminatorScores.real.length;
    const avgFakeScore = this.discriminatorScores.fake.reduce((a, b) => a + b, 0) / this.discriminatorScores.fake.length;
    
    return {
      generatorLoss: finalGenLoss,
      discriminatorLoss: finalDiscLoss,
      discriminatorAccuracy,
      avgRealScore,
      avgFakeScore,
      generatorArchitecture: `${this.latentDim}-${this.generatorLayers.join('-')}-${this.X_train[0].length}`,
      discriminatorArchitecture: `${this.X_train[0].length}-${this.discriminatorLayers.join('-')}-1`,
      latentDimension: this.latentDim
    };
  }
  
  /**
   * Get GAN state for visualization
   */
  getGANState(): GANState {
    return {
      generatorWeights: this.generatorWeights.map(layer => layer.map(neuron => [...neuron])),
      discriminatorWeights: this.discriminatorWeights.map(layer => layer.map(neuron => [...neuron])),
      generatorBiases: this.generatorBiases.map(layer => [...layer]),
      discriminatorBiases: this.discriminatorBiases.map(layer => [...layer]),
      generatedSamples: this.generatedSamples.map(sample => [...sample]),
      discriminatorScores: {
        real: [...this.discriminatorScores.real],
        fake: [...this.discriminatorScores.fake]
      }
    };
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.generatorWeights = [];
    this.discriminatorWeights = [];
    this.generatorBiases = [];
    this.discriminatorBiases = [];
    this.X_train = [];
    this.generatedSamples = [];
    this.discriminatorScores = { real: [], fake: [] };
    this.generatorLoss = [];
    this.discriminatorLoss = [];
  }
  
  /**
   * Get model parameters
   */
  getParameters() {
    return {
      generatorLayers: [...this.generatorLayers],
      discriminatorLayers: [...this.discriminatorLayers],
      latentDim: this.latentDim,
      learningRate: this.learningRate,
      epochs: this.epochs,
      batchSize: this.batchSize,
      activation: this.activation
    };
  }
}