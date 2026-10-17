/**
 * Text Generation Algorithm
 * 
 * Implements sequence generation using a character-level language model
 * with LSTM-like recurrent neural network. Generates text one character
 * at a time based on learned patterns from training corpus.
 * 
 * Requirements: 3.2, 3.4, 3.5
 */

import { BaseAlgorithm, type VisualizationData, type Metrics } from '~/types/algorithm';

export interface TextGenerationOptions {
  sequenceLength: number; // Length of input sequences for training
  temperature: number; // Sampling temperature (higher = more random)
  maxLength: number; // Maximum length of generated text
  learningRate: number;
  epochs: number;
  hiddenSize: number; // Size of hidden layer
}

export interface GenerationStep {
  step: number;
  character: string;
  probabilities: { [char: string]: number };
  topPredictions: Array<{ char: string; prob: number }>;
  context: string;
}

export interface GeneratedText {
  text: string;
  steps: GenerationStep[];
  seed: string;
}

/**
 * Text Generation using character-level RNN
 * Requirements: 3.2, 3.4
 */
export class TextGeneration extends BaseAlgorithm {
  name = 'Text Generation';
  
  private options: TextGenerationOptions;
  private charToIndex: Map<string, number> = new Map();
  private indexToChar: Map<number, string> = new Map();
  private vocabSize: number = 0;
  
  // Model parameters (simplified RNN)
  private Wxh: number[][] = []; // Input to hidden weights
  private Whh: number[][] = []; // Hidden to hidden weights
  private Why: number[][] = []; // Hidden to output weights
  private bh: number[] = []; // Hidden bias
  private by: number[] = []; // Output bias
  
  private trainingText: string = '';
  private generatedTexts: GeneratedText[] = [];
  
  constructor(options: Partial<TextGenerationOptions> = {}) {
    super();
    this.options = {
      sequenceLength: options.sequenceLength ?? 25,
      temperature: options.temperature ?? 0.8,
      maxLength: options.maxLength ?? 200,
      learningRate: options.learningRate ?? 0.01,
      epochs: options.epochs ?? 100,
      hiddenSize: options.hiddenSize ?? 100
    };
  }
  
  /**
   * Train text generation model on corpus
   * Requirements: 3.2, 3.4
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    throw new Error('Use trainOnText() method for text generation instead of fit()');
  }
  
  /**
   * Train on text corpus
   * Requirements: 3.2, 3.4
   * 
   * @param text - Training text corpus
   */
  async trainOnText(text: string): Promise<void> {
    if (!text || text.length < this.options.sequenceLength) {
      throw new Error(`Training text must be at least ${this.options.sequenceLength} characters long`);
    }
    
    this.trainingText = text;
    
    // Build character vocabulary
    this.buildVocabulary(text);
    
    if (this.vocabSize === 0) {
      throw new Error('Vocabulary is empty');
    }
    
    // Initialize model parameters
    this.initializeWeights();
    
    // Train the model
    await this.trainModel(text);
    
    this.trained = true;
  }
  
  /**
   * Build character vocabulary from text
   * Requirements: 3.4
   */
  private buildVocabulary(text: string): void {
    this.charToIndex.clear();
    this.indexToChar.clear();
    
    // Get unique characters
    const uniqueChars = Array.from(new Set(text));
    
    // Build mappings
    uniqueChars.forEach((char, index) => {
      this.charToIndex.set(char, index);
      this.indexToChar.set(index, char);
    });
    
    this.vocabSize = uniqueChars.length;
  }
  
  /**
   * Initialize weight matrices with Xavier initialization
   * Requirements: 3.4
   */
  private initializeWeights(): void {
    const { hiddenSize } = this.options;
    const vocabSize = this.vocabSize;
    
    // Xavier initialization scale
    const scaleWxh = Math.sqrt(2.0 / (vocabSize + hiddenSize));
    const scaleWhh = Math.sqrt(2.0 / (hiddenSize + hiddenSize));
    const scaleWhy = Math.sqrt(2.0 / (hiddenSize + vocabSize));
    
    // Input to hidden weights
    this.Wxh = Array(hiddenSize).fill(0).map(() =>
      Array(vocabSize).fill(0).map(() => (Math.random() - 0.5) * 2 * scaleWxh)
    );
    
    // Hidden to hidden weights
    this.Whh = Array(hiddenSize).fill(0).map(() =>
      Array(hiddenSize).fill(0).map(() => (Math.random() - 0.5) * 2 * scaleWhh)
    );
    
    // Hidden to output weights
    this.Why = Array(vocabSize).fill(0).map(() =>
      Array(hiddenSize).fill(0).map(() => (Math.random() - 0.5) * 2 * scaleWhy)
    );
    
    // Biases
    this.bh = Array(hiddenSize).fill(0);
    this.by = Array(vocabSize).fill(0);
  }
  
  /**
   * Convert character to one-hot vector
   * Requirements: 3.4
   */
  private charToOneHot(char: string): number[] {
    const vector = Array(this.vocabSize).fill(0);
    const index = this.charToIndex.get(char);
    if (index !== undefined) {
      vector[index] = 1;
    }
    return vector;
  }
  
  /**
   * Tanh activation function
   */
  private tanh(x: number): number {
    return Math.tanh(x);
  }
  
  /**
   * Softmax function
   */
  private softmax(scores: number[]): number[] {
    const maxScore = Math.max(...scores);
    const expScores = scores.map(s => Math.exp(s - maxScore));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    return expScores.map(s => s / sumExp);
  }
  
  /**
   * Forward pass through RNN
   * Requirements: 3.4
   */
  private forward(inputs: number[][], hPrev: number[]): {
    hiddenStates: number[][];
    outputs: number[][];
    finalHidden: number[];
  } {
    const hiddenStates: number[][] = [];
    const outputs: number[][] = [];
    let h = [...hPrev];
    
    for (const x of inputs) {
      // Hidden state: h = tanh(Wxh * x + Whh * h_prev + bh)
      const hNew = Array(this.options.hiddenSize).fill(0);
      
      for (let i = 0; i < this.options.hiddenSize; i++) {
        let sum = this.bh[i];
        
        // Wxh * x
        for (let j = 0; j < this.vocabSize; j++) {
          sum += this.Wxh[i][j] * x[j];
        }
        
        // Whh * h_prev
        for (let j = 0; j < this.options.hiddenSize; j++) {
          sum += this.Whh[i][j] * h[j];
        }
        
        hNew[i] = this.tanh(sum);
      }
      
      hiddenStates.push(hNew);
      h = hNew;
      
      // Output: y = Why * h + by
      const y = Array(this.vocabSize).fill(0);
      for (let i = 0; i < this.vocabSize; i++) {
        let sum = this.by[i];
        for (let j = 0; j < this.options.hiddenSize; j++) {
          sum += this.Why[i][j] * h[j];
        }
        y[i] = sum;
      }
      
      outputs.push(y);
    }
    
    return { hiddenStates, outputs, finalHidden: h };
  }
  
  /**
   * Train the model using truncated backpropagation through time
   * Requirements: 3.4
   */
  private async trainModel(text: string): Promise<void> {
    const { sequenceLength, learningRate, epochs } = this.options;
    const textLength = text.length;
    
    // Initialize hidden state
    let hPrev = Array(this.options.hiddenSize).fill(0);
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
      let numSequences = 0;
      
      // Process text in sequences
      for (let i = 0; i <= textLength - sequenceLength - 1; i += sequenceLength) {
        const sequence = text.substring(i, i + sequenceLength);
        const target = text.substring(i + 1, i + sequenceLength + 1);
        
        // Convert to one-hot vectors
        const inputs = Array.from(sequence).map(char => this.charToOneHot(char));
        const targets = Array.from(target).map(char => {
          const idx = this.charToIndex.get(char);
          return idx !== undefined ? idx : 0;
        });
        
        // Forward pass
        const { hiddenStates, outputs, finalHidden } = this.forward(inputs, hPrev);
        
        // Compute loss
        let loss = 0;
        for (let t = 0; t < outputs.length; t++) {
          const probs = this.softmax(outputs[t]);
          loss += -Math.log(probs[targets[t]] + 1e-10);
        }
        
        totalLoss += loss;
        numSequences++;
        
        // Backward pass (simplified gradient descent)
        this.backwardSimplified(inputs, targets, hiddenStates, outputs, learningRate);
        
        // Update hidden state for next sequence
        hPrev = finalHidden;
        
        // Reset hidden state periodically
        if (i % (sequenceLength * 10) === 0) {
          hPrev = Array(this.options.hiddenSize).fill(0);
        }
      }
      
      // Record training history
      const avgLoss = totalLoss / numSequences;
      this.trainingHistory.epochs.push(epoch + 1);
      this.trainingHistory.loss.push(avgLoss);
      
      // Decay learning rate
      if ((epoch + 1) % 20 === 0) {
        this.options.learningRate *= 0.95;
      }
    }
  }
  
  /**
   * Simplified backward pass
   * Requirements: 3.4
   */
  private backwardSimplified(
    inputs: number[][],
    targets: number[],
    hiddenStates: number[][],
    outputs: number[][],
    lr: number
  ): void {
    // Simplified gradient updates (not full BPTT)
    for (let t = 0; t < outputs.length; t++) {
      const probs = this.softmax(outputs[t]);
      const dy = [...probs];
      dy[targets[t]] -= 1; // Gradient of cross-entropy loss
      
      // Update Why and by
      for (let i = 0; i < this.vocabSize; i++) {
        this.by[i] -= lr * dy[i];
        for (let j = 0; j < this.options.hiddenSize; j++) {
          this.Why[i][j] -= lr * dy[i] * hiddenStates[t][j];
        }
      }
      
      // Compute gradient for hidden layer
      const dh = Array(this.options.hiddenSize).fill(0);
      for (let i = 0; i < this.options.hiddenSize; i++) {
        for (let j = 0; j < this.vocabSize; j++) {
          dh[i] += this.Why[j][i] * dy[j];
        }
        // Gradient through tanh
        dh[i] *= (1 - hiddenStates[t][i] * hiddenStates[t][i]);
      }
      
      // Update Wxh and bh
      for (let i = 0; i < this.options.hiddenSize; i++) {
        this.bh[i] -= lr * dh[i];
        for (let j = 0; j < this.vocabSize; j++) {
          this.Wxh[i][j] -= lr * dh[i] * inputs[t][j];
        }
      }
      
      // Update Whh (using previous hidden state if available)
      if (t > 0) {
        for (let i = 0; i < this.options.hiddenSize; i++) {
          for (let j = 0; j < this.options.hiddenSize; j++) {
            this.Whh[i][j] -= lr * dh[i] * hiddenStates[t - 1][j];
          }
        }
      }
    }
  }
  
  /**
   * Sample next character from probability distribution
   * Requirements: 3.4
   */
  private sampleCharacter(probs: number[], temperature: number): number {
    // Apply temperature
    const adjustedProbs = probs.map(p => Math.pow(p, 1 / temperature));
    const sum = adjustedProbs.reduce((a, b) => a + b, 0);
    const normalizedProbs = adjustedProbs.map(p => p / sum);
    
    // Sample from distribution
    const rand = Math.random();
    let cumulative = 0;
    
    for (let i = 0; i < normalizedProbs.length; i++) {
      cumulative += normalizedProbs[i];
      if (rand < cumulative) {
        return i;
      }
    }
    
    return normalizedProbs.length - 1;
  }
  
  /**
   * Generate text starting from seed
   * Requirements: 3.2, 3.4, 3.5
   */
  generateText(seed: string, maxLength?: number): GeneratedText {
    this.ensureTrained();
    
    const length = maxLength || this.options.maxLength;
    const steps: GenerationStep[] = [];
    let generated = seed;
    let h = Array(this.options.hiddenSize).fill(0);
    
    // Process seed to get initial hidden state
    for (const char of seed) {
      const x = this.charToOneHot(char);
      const { finalHidden } = this.forward([x], h);
      h = finalHidden;
    }
    
    // Generate characters
    for (let step = 0; step < length; step++) {
      // Get last character
      const lastChar = generated[generated.length - 1];
      const x = this.charToOneHot(lastChar);
      
      // Forward pass
      const { outputs, finalHidden } = this.forward([x], h);
      const probs = this.softmax(outputs[0]);
      
      // Sample next character
      const nextCharIndex = this.sampleCharacter(probs, this.options.temperature);
      const nextChar = this.indexToChar.get(nextCharIndex) || '';
      
      // Get top predictions
      const probsWithChars = probs.map((prob, idx) => ({
        char: this.indexToChar.get(idx) || '',
        prob
      }));
      probsWithChars.sort((a, b) => b.prob - a.prob);
      const topPredictions = probsWithChars.slice(0, 5);
      
      // Create probability map
      const probMap: { [char: string]: number } = {};
      probs.forEach((prob, idx) => {
        const char = this.indexToChar.get(idx);
        if (char) {
          probMap[char] = prob;
        }
      });
      
      // Record step
      steps.push({
        step: step + 1,
        character: nextChar,
        probabilities: probMap,
        topPredictions,
        context: generated.slice(-20) // Last 20 characters as context
      });
      
      // Update generated text and hidden state
      generated += nextChar;
      h = finalHidden;
      
      // Stop if we generate a natural ending
      if (nextChar === '\n' && step > 50) {
        break;
      }
    }
    
    const result: GeneratedText = {
      text: generated,
      steps,
      seed
    };
    
    this.generatedTexts.push(result);
    
    return result;
  }
  
  /**
   * Predict method (not used for text generation)
   * Requirements: 3.2
   */
  predict(X: number[][]): string[] {
    throw new Error('Use generateText() method for text generation');
  }
  
  /**
   * Get visualization data for step-by-step generation display
   * Requirements: 3.5
   */
  getVisualizationData(): VisualizationData | VisualizationData[] {
    const visualizations: VisualizationData[] = [];
    
    // Training loss curve
    if (this.trainingHistory.loss.length > 0) {
      visualizations.push({
        type: 'line',
        data: {
          labels: this.trainingHistory.epochs,
          datasets: [{
            label: 'Training Loss',
            data: this.trainingHistory.loss,
            borderColor: 'rgb(99, 102, 241)',
            backgroundColor: 'rgba(99, 102, 241, 0.1)',
            tension: 0.4
          }]
        },
        config: {
          type: 'line',
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              title: {
                display: true,
                text: 'Training Loss Over Epochs'
              }
            },
            scales: {
              y: {
                beginAtZero: false,
                title: {
                  display: true,
                  text: 'Loss'
                }
              },
              x: {
                title: {
                  display: true,
                  text: 'Epoch'
                }
              }
            }
          }
        }
      });
    }
    
    // Character probability distribution for last generation
    if (this.generatedTexts.length > 0) {
      const lastGeneration = this.generatedTexts[this.generatedTexts.length - 1];
      if (lastGeneration.steps.length > 0) {
        const lastStep = lastGeneration.steps[lastGeneration.steps.length - 1];
        
        visualizations.push({
          type: 'line',
          data: {
            labels: lastStep.topPredictions.map(p => p.char === '\n' ? '\\n' : p.char),
            datasets: [{
              label: 'Top Character Probabilities',
              data: lastStep.topPredictions.map(p => p.prob),
              backgroundColor: 'rgba(16, 185, 129, 0.5)',
              borderColor: 'rgb(16, 185, 129)',
              borderWidth: 2
            }]
          },
          config: {
            type: 'bar',
            options: {
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                title: {
                  display: true,
                  text: 'Character Prediction Probabilities (Last Step)'
                }
              },
              scales: {
                y: {
                  beginAtZero: true,
                  max: 1,
                  title: {
                    display: true,
                    text: 'Probability'
                  }
                }
              }
            }
          }
        });
      }
    }
    
    return visualizations;
  }
  
  /**
   * Get text generation metrics
   * Requirements: 3.4
   */
  getMetrics(): Metrics {
    const metrics: Metrics = {
      vocabularySize: this.vocabSize,
      hiddenSize: this.options.hiddenSize,
      sequenceLength: this.options.sequenceLength,
      trainingTextLength: this.trainingText.length,
      epochs: this.options.epochs,
      finalLoss: this.trainingHistory.loss[this.trainingHistory.loss.length - 1] || 0
    };
    
    if (this.generatedTexts.length > 0) {
      metrics.generatedTextsCount = this.generatedTexts.length;
      metrics.averageGeneratedLength = this.generatedTexts.reduce((sum, t) => sum + t.text.length, 0) / this.generatedTexts.length;
    }
    
    return metrics;
  }
  
  /**
   * Reset text generation state
   */
  override reset(): void {
    super.reset();
    this.charToIndex.clear();
    this.indexToChar.clear();
    this.vocabSize = 0;
    this.Wxh = [];
    this.Whh = [];
    this.Why = [];
    this.bh = [];
    this.by = [];
    this.trainingText = '';
    this.generatedTexts = [];
  }
}
