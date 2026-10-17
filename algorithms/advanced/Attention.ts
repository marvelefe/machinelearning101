/**
 * Attention Mechanisms
 * 
 * Implements various attention mechanisms including self-attention and
 * multi-head attention. Demonstrates how attention allows models to focus
 * on relevant parts of input sequences.
 * 
 * Requirements: 10.1, 10.2, 10.3, 10.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics } from '~/types/algorithm';

export interface AttentionParams {
  sequenceLength: number;
  embeddingDim: number;
  numHeads: number;
  attentionType: 'self' | 'cross' | 'multi-head';
  temperature: number;
  dropout?: number;
}

export interface AttentionWeights {
  weights: number[][];
  queries: number[][];
  keys: number[][];
  values: number[][];
  output: number[][];
}

export interface MultiHeadAttention {
  heads: AttentionWeights[];
  concatenated: number[][];
  projected: number[][];
}

export class Attention extends BaseAlgorithm<number[][][], number[][], number[][][]> {
  name = 'Attention Mechanisms';
  
  // Model parameters
  private sequenceLength: number;
  private embeddingDim: number;
  private numHeads: number;
  private attentionType: 'self' | 'cross' | 'multi-head';
  private temperature: number;
  private dropout?: number;
  
  // Learned parameters
  private queryWeights: number[][];
  private keyWeights: number[][];
  private valueWeights: number[][];
  private outputWeights: number[][];
  
  // Multi-head specific
  private headDim: number;
  private multiHeadWeights: {
    queries: number[][][];
    keys: number[][][];
    values: number[][][];
    output: number[][];
  } = { queries: [], keys: [], values: [], output: [] };
  
  // Training data
  private sequences: number[][][] = [];
  private targets: number[][] = [];
  
  // Attention visualization data
  private attentionWeights: number[][][] = [];
  private attentionOutputs: number[][][] = [];
  private headAttentions: AttentionWeights[][] = [];
  
  constructor(params: AttentionParams = {
    sequenceLength: 8,
    embeddingDim: 16,
    numHeads: 4,
    attentionType: 'self',
    temperature: 1.0,
    dropout: 0.1
  }) {
    super();
    this.sequenceLength = params.sequenceLength;
    this.embeddingDim = params.embeddingDim;
    this.numHeads = params.numHeads;
    this.attentionType = params.attentionType;
    this.temperature = params.temperature;
    this.dropout = params.dropout;
    this.headDim = Math.floor(this.embeddingDim / this.numHeads);
  }

  /**
   * Initialize attention weights
   */
  private initializeWeights(): void {
    const scale = Math.sqrt(2.0 / this.embeddingDim);
    
    // Single-head attention weights
    this.queryWeights = this.randomMatrix(this.embeddingDim, this.embeddingDim, scale);
    this.keyWeights = this.randomMatrix(this.embeddingDim, this.embeddingDim, scale);
    this.valueWeights = this.randomMatrix(this.embeddingDim, this.embeddingDim, scale);
    this.outputWeights = this.randomMatrix(this.embeddingDim, this.embeddingDim, scale);
    
    // Multi-head attention weights
    if (this.attentionType === 'multi-head') {
      this.multiHeadWeights.queries = [];
      this.multiHeadWeights.keys = [];
      this.multiHeadWeights.values = [];
      
      for (let h = 0; h < this.numHeads; h++) {
        this.multiHeadWeights.queries.push(
          this.randomMatrix(this.embeddingDim, this.headDim, scale)
        );
        this.multiHeadWeights.keys.push(
          this.randomMatrix(this.embeddingDim, this.headDim, scale)
        );
        this.multiHeadWeights.values.push(
          this.randomMatrix(this.embeddingDim, this.headDim, scale)
        );
      }
      
      this.multiHeadWeights.output = this.randomMatrix(
        this.embeddingDim, this.embeddingDim, scale
      );
    }
  }
  
  /**
   * Create random matrix with given scale
   */
  private randomMatrix(rows: number, cols: number, scale: number): number[][] {
    const matrix: number[][] = [];
    for (let i = 0; i < rows; i++) {
      const row: number[] = [];
      for (let j = 0; j < cols; j++) {
        row.push((Math.random() - 0.5) * 2 * scale);
      }
      matrix.push(row);
    }
    return matrix;
  }
  
  /**
   * Matrix multiplication
   */
  private matmul(A: number[][], B: number[][]): number[][] {
    const result: number[][] = [];
    for (let i = 0; i < A.length; i++) {
      const row: number[] = [];
      for (let j = 0; j < B[0].length; j++) {
        let sum = 0;
        for (let k = 0; k < B.length; k++) {
          sum += A[i][k] * B[k][j];
        }
        row.push(sum);
      }
      result.push(row);
    }
    return result;
  }
  
  /**
   * Softmax function
   */
  private softmax(logits: number[]): number[] {
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map(x => Math.exp((x - maxLogit) / this.temperature));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    return expLogits.map(x => x / sumExp);
  }
  
  /**
   * Compute scaled dot-product attention
   */
  private scaledDotProductAttention(
    queries: number[][],
    keys: number[][],
    values: number[][]
  ): AttentionWeights {
    const seqLen = queries.length;
    const keyDim = keys[0].length;
    
    // Compute attention scores: Q * K^T
    const scores: number[][] = [];
    for (let i = 0; i < seqLen; i++) {
      const scoreRow: number[] = [];
      for (let j = 0; j < seqLen; j++) {
        let score = 0;
        for (let k = 0; k < keyDim; k++) {
          score += queries[i][k] * keys[j][k];
        }
        // Scale by sqrt(d_k)
        scoreRow.push(score / Math.sqrt(keyDim));
      }
      scores.push(scoreRow);
    }
    
    // Apply softmax to get attention weights
    const attentionWeights: number[][] = [];
    for (let i = 0; i < seqLen; i++) {
      attentionWeights.push(this.softmax(scores[i]));
    }
    
    // Compute weighted sum of values
    const output: number[][] = [];
    for (let i = 0; i < seqLen; i++) {
      const outputRow: number[] = [];
      for (let d = 0; d < values[0].length; d++) {
        let weightedSum = 0;
        for (let j = 0; j < seqLen; j++) {
          weightedSum += attentionWeights[i][j] * values[j][d];
        }
        outputRow.push(weightedSum);
      }
      output.push(outputRow);
    }
    
    return {
      weights: attentionWeights,
      queries,
      keys,
      values,
      output
    };
  }
  
  /**
   * Self-attention mechanism
   */
  private selfAttention(sequence: number[][]): AttentionWeights {
    // Linear projections
    const queries = this.matmul(sequence, this.queryWeights);
    const keys = this.matmul(sequence, this.keyWeights);
    const values = this.matmul(sequence, this.valueWeights);
    
    return this.scaledDotProductAttention(queries, keys, values);
  }
  
  /**
   * Multi-head attention mechanism
   */
  private multiHeadAttention(sequence: number[][]): MultiHeadAttention {
    const heads: AttentionWeights[] = [];
    
    // Compute attention for each head
    for (let h = 0; h < this.numHeads; h++) {
      const queries = this.matmul(sequence, this.multiHeadWeights.queries[h]);
      const keys = this.matmul(sequence, this.multiHeadWeights.keys[h]);
      const values = this.matmul(sequence, this.multiHeadWeights.values[h]);
      
      const headAttention = this.scaledDotProductAttention(queries, keys, values);
      heads.push(headAttention);
    }
    
    // Concatenate head outputs
    const concatenated: number[][] = [];
    for (let i = 0; i < this.sequenceLength; i++) {
      const concatRow: number[] = [];
      for (let h = 0; h < this.numHeads; h++) {
        concatRow.push(...heads[h].output[i]);
      }
      concatenated.push(concatRow);
    }
    
    // Final linear projection
    const projected = this.matmul(concatenated, this.multiHeadWeights.output);
    
    return {
      heads,
      concatenated,
      projected
    };
  }
  
  /**
   * Apply attention mechanism
   */
  private applyAttention(sequence: number[][]): number[][] {
    if (this.attentionType === 'multi-head') {
      const multiHead = this.multiHeadAttention(sequence);
      this.headAttentions.push(multiHead.heads);
      return multiHead.projected;
    } else {
      const attention = this.selfAttention(sequence);
      this.attentionWeights.push(attention.weights);
      return attention.output;
    }
  }
  
  /**
   * Simple sequence classification task for demonstration
   */
  private sequenceClassification(attentionOutput: number[][]): number[] {
    // Global average pooling
    const pooled: number[] = [];
    for (let d = 0; d < attentionOutput[0].length; d++) {
      let sum = 0;
      for (let i = 0; i < attentionOutput.length; i++) {
        sum += attentionOutput[i][d];
      }
      pooled.push(sum / attentionOutput.length);
    }
    
    // Simple linear classifier (for demonstration)
    const weights = Array.from({length: pooled.length}, () => Math.random() - 0.5);
    const bias = Math.random() - 0.5;
    
    let logit = bias;
    for (let i = 0; i < pooled.length; i++) {
      logit += pooled[i] * weights[i];
    }
    
    const probability = 1 / (1 + Math.exp(-logit));
    return [1 - probability, probability];
  }

  /**
   * Train the attention model
   */
  async fit(X: number[][][], y?: number[][]): Promise<void> {
    // Validate input
    if (!X || X.length === 0) {
      throw new Error('Input sequences cannot be empty');
    }
    
    // Store training data
    this.sequences = X.map(seq => seq.map(token => [...token]));
    this.targets = y ? y.map(target => [...target]) : [];
    
    // Initialize weights
    this.initializeWeights();
    
    // Initialize training history
    this.trainingHistory = {
      epochs: [],
      loss: []
    };
    
    // Clear visualization data
    this.attentionWeights = [];
    this.attentionOutputs = [];
    this.headAttentions = [];
    
    // Process each sequence
    for (let epoch = 0; epoch < 10; epoch++) {
      let totalLoss = 0;
      
      for (let i = 0; i < this.sequences.length; i++) {
        const sequence = this.sequences[i];
        
        // Apply attention
        const attentionOutput = this.applyAttention(sequence);
        this.attentionOutputs.push(attentionOutput);
        
        // Simple loss computation (for demonstration)
        if (this.targets.length > 0) {
          const prediction = this.sequenceClassification(attentionOutput);
          const target = this.targets[i];
          
          // Cross-entropy loss
          let loss = 0;
          for (let j = 0; j < target.length; j++) {
            loss -= target[j] * Math.log(prediction[j] + 1e-10);
          }
          totalLoss += loss;
        }
      }
      
      // Record history
      this.trainingHistory.epochs.push(epoch);
      this.trainingHistory.loss.push(totalLoss / this.sequences.length);
      
      // Allow UI to update
      if (epoch % 2 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    this.trained = true;
  }
  
  /**
   * Apply attention to new sequences
   */
  predict(X: number[][][]): number[][][] {
    this.ensureTrained();
    
    const results: number[][][] = [];
    
    for (const sequence of X) {
      const attentionOutput = this.applyAttention(sequence);
      results.push(attentionOutput);
    }
    
    return results;
  }
  
  /**
   * Get attention weights for visualization
   */
  getAttentionWeights(): number[][][] {
    return this.attentionWeights.map(weights => 
      weights.map(row => [...row])
    );
  }
  
  /**
   * Get multi-head attention weights
   */
  getMultiHeadWeights(): AttentionWeights[][] {
    return this.headAttentions.map(heads =>
      heads.map(head => ({
        weights: head.weights.map(row => [...row]),
        queries: head.queries.map(row => [...row]),
        keys: head.keys.map(row => [...row]),
        values: head.values.map(row => [...row]),
        output: head.output.map(row => [...row])
      }))
    );
  }

  /**
   * Get visualization data
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained) {
      return [];
    }
    
    const visualizations: VisualizationData[] = [];
    
    // Attention heatmap (first sequence)
    if (this.attentionWeights.length > 0) {
      visualizations.push({
        type: 'heatmap',
        data: {
          attentionWeights: this.attentionWeights[0]
        }
      });
    }
    
    // Multi-head attention comparison
    if (this.headAttentions.length > 0 && this.headAttentions[0].length > 0) {
      const firstSequenceHeads = this.headAttentions[0];
      const multiHeadData: number[][][] = [];
      
      for (let h = 0; h < Math.min(4, firstSequenceHeads.length); h++) {
        multiHeadData.push(firstSequenceHeads[h].weights);
      }
      
      visualizations.push({
        type: 'heatmap',
        data: {
          multiHead: multiHeadData
        }
      });
    }
    
    // Attention distribution and entropy
    if (this.attentionWeights.length > 0) {
      const avgAttention = this.computeAverageAttention();
      const entropyByPosition = this.computeEntropyByPosition();
      
      visualizations.push({
        type: 'line',
        data: {
          distribution: {
            x: Array.from({length: avgAttention.length}, (_, i) => i),
            y: avgAttention
          },
          entropy: {
            x: Array.from({length: entropyByPosition.length}, (_, i) => i),
            y: entropyByPosition
          }
        }
      });
    }
    
    return visualizations;
  }
  
  /**
   * Compute entropy by query position
   */
  private computeEntropyByPosition(): number[] {
    if (this.attentionWeights.length === 0) return [];
    
    const seqLen = this.attentionWeights[0].length;
    const entropyByPos: number[] = [];
    
    for (let i = 0; i < seqLen; i++) {
      let totalEntropy = 0;
      
      for (const weights of this.attentionWeights) {
        let entropy = 0;
        for (const weight of weights[i]) {
          if (weight > 0) {
            entropy -= weight * Math.log(weight);
          }
        }
        totalEntropy += entropy;
      }
      
      entropyByPos.push(totalEntropy / this.attentionWeights.length);
    }
    
    return entropyByPos;
  }
  
  /**
   * Compute average attention weights across positions
   */
  private computeAverageAttention(): number[] {
    if (this.attentionWeights.length === 0) return [];
    
    const seqLen = this.attentionWeights[0].length;
    const avgAttention: number[] = [];
    
    for (let i = 0; i < seqLen; i++) {
      let sum = 0;
      let count = 0;
      
      for (const weights of this.attentionWeights) {
        for (let j = 0; j < seqLen; j++) {
          sum += weights[i][j];
          count++;
        }
      }
      
      avgAttention.push(sum / count);
    }
    
    return avgAttention;
  }
  
  /**
   * Get metrics
   */
  getMetrics(): Metrics {
    if (!this.trained) {
      return {};
    }
    
    const finalLoss = this.trainingHistory.loss[this.trainingHistory.loss.length - 1];
    
    // Compute attention statistics
    let totalAttentionEntropy = 0;
    let maxAttentionWeight = 0;
    let minAttentionWeight = 1;
    let entropyCount = 0;
    
    // Use appropriate attention weights based on type
    const weightsToAnalyze = this.attentionType === 'multi-head' && this.headAttentions.length > 0
      ? this.headAttentions.flatMap(heads => heads.map(h => h.weights))
      : this.attentionWeights;
    
    for (const weights of weightsToAnalyze) {
      for (const row of weights) {
        for (const weight of row) {
          maxAttentionWeight = Math.max(maxAttentionWeight, weight);
          minAttentionWeight = Math.min(minAttentionWeight, weight);
        }
        
        // Compute entropy of attention distribution
        let entropy = 0;
        for (const weight of row) {
          if (weight > 0) {
            entropy -= weight * Math.log(weight);
          }
        }
        totalAttentionEntropy += entropy;
        entropyCount++;
      }
    }
    
    const attentionEntropy = entropyCount > 0 
      ? totalAttentionEntropy / entropyCount
      : 0;
    
    return {
      finalLoss,
      attentionEntropy,
      maxAttentionWeight,
      minAttentionWeight,
      sequenceLength: this.sequenceLength,
      embeddingDim: this.embeddingDim,
      numHeads: this.numHeads,
      attentionType: this.attentionType,
      totalParameters: this.getTotalParameters()
    };
  }
  
  /**
   * Get total number of parameters
   */
  private getTotalParameters(): number {
    let total = 0;
    
    // Single-head parameters
    total += this.embeddingDim * this.embeddingDim * 4; // Q, K, V, O matrices
    
    // Multi-head parameters
    if (this.attentionType === 'multi-head') {
      total += this.numHeads * this.embeddingDim * this.headDim * 3; // Q, K, V for each head
      total += this.embeddingDim * this.embeddingDim; // Output projection
    }
    
    return total;
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.queryWeights = [];
    this.keyWeights = [];
    this.valueWeights = [];
    this.outputWeights = [];
    this.multiHeadWeights = { queries: [], keys: [], values: [], output: [] };
    this.sequences = [];
    this.targets = [];
    this.attentionWeights = [];
    this.attentionOutputs = [];
    this.headAttentions = [];
  }
  
  /**
   * Get model parameters
   */
  getParameters() {
    return {
      sequenceLength: this.sequenceLength,
      embeddingDim: this.embeddingDim,
      numHeads: this.numHeads,
      attentionType: this.attentionType,
      temperature: this.temperature,
      dropout: this.dropout,
      headDim: this.headDim
    };
  }
}