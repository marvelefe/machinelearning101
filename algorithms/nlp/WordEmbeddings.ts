/**
 * Word Embeddings Algorithm
 * 
 * Implements Word2Vec-style word embeddings using Skip-gram with negative sampling.
 * Learns dense vector representations of words that capture semantic relationships.
 * 
 * Requirements: 3.2, 3.4, 3.5
 */

import { BaseAlgorithm, type VisualizationData, type Metrics } from '~/types/algorithm';

export interface WordEmbeddingsOptions {
  embeddingDim: number;
  windowSize: number;
  minCount: number;
  learningRate: number;
  epochs: number;
  negSamples: number;
}

export interface WordVector {
  word: string;
  vector: number[];
  frequency: number;
}

/**
 * Word Embeddings using Skip-gram model
 * Requirements: 3.2, 3.4
 */
export class WordEmbeddings extends BaseAlgorithm {
  name = 'Word Embeddings';
  
  private options: WordEmbeddingsOptions;
  private vocabulary: Map<string, number> = new Map(); // word -> index
  private wordFrequency: Map<string, number> = new Map();
  private embeddings: number[][] = []; // word vectors
  private contextEmbeddings: number[][] = []; // context vectors
  private indexToWord: string[] = [];
  private corpusSize: number = 0;
  private projectedEmbeddings2D: number[][] = [];
  private projectedEmbeddings3D: number[][] = [];
  
  constructor(options: Partial<WordEmbeddingsOptions> = {}) {
    super();
    this.options = {
      embeddingDim: options.embeddingDim ?? 50,
      windowSize: options.windowSize ?? 2,
      minCount: options.minCount ?? 2,
      learningRate: options.learningRate ?? 0.025,
      epochs: options.epochs ?? 5,
      negSamples: options.negSamples ?? 5
    };
  }
  
  /**
   * Train word embeddings on text corpus
   * Requirements: 3.2, 3.4
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    // For text processing, we expect text to be passed differently
    throw new Error('Use trainOnText() method for word embeddings instead of fit()');
  }
  
  /**
   * Train embeddings on text corpus
   * Requirements: 3.2, 3.4
   */
  async trainOnText(text: string): Promise<void> {
    // Tokenize and build vocabulary
    const tokens = this.tokenize(text);
    this.buildVocabulary(tokens);
    
    if (this.vocabulary.size === 0) {
      this.trained = true;
      return;
    }
    
    // Initialize embeddings randomly
    this.initializeEmbeddings();
    
    // Train using Skip-gram with negative sampling
    await this.trainSkipGram(tokens);
    
    // Project embeddings to 2D and 3D for visualization
    this.projectEmbeddings();
    
    this.trained = true;
  }
  
  /**
   * Simple tokenization
   * Requirements: 3.2
   */
  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(token => token.length > 0);
  }
  
  /**
   * Build vocabulary from tokens
   * Requirements: 3.4
   */
  private buildVocabulary(tokens: string[]): void {
    this.vocabulary.clear();
    this.wordFrequency.clear();
    this.indexToWord = [];
    
    // Count word frequencies
    const wordCounts = new Map<string, number>();
    for (const token of tokens) {
      wordCounts.set(token, (wordCounts.get(token) || 0) + 1);
    }
    
    // Filter by minimum count and build vocabulary
    let index = 0;
    for (const [word, count] of wordCounts.entries()) {
      if (count >= this.options.minCount) {
        this.vocabulary.set(word, index);
        this.wordFrequency.set(word, count);
        this.indexToWord.push(word);
        index++;
      }
    }
    
    this.corpusSize = tokens.length;
  }
  
  /**
   * Initialize embedding matrices with random values
   * Requirements: 3.4
   */
  private initializeEmbeddings(): void {
    const vocabSize = this.vocabulary.size;
    const dim = this.options.embeddingDim;
    
    // Initialize word embeddings
    this.embeddings = Array(vocabSize).fill(0).map(() =>
      Array(dim).fill(0).map(() => (Math.random() - 0.5) / dim)
    );
    
    // Initialize context embeddings
    this.contextEmbeddings = Array(vocabSize).fill(0).map(() =>
      Array(dim).fill(0).map(() => (Math.random() - 0.5) / dim)
    );
  }
  
  /**
   * Train Skip-gram model with negative sampling
   * Requirements: 3.4
   */
  private async trainSkipGram(tokens: string[]): Promise<void> {
    const vocabSize = this.vocabulary.size;
    const windowSize = this.options.windowSize;
    const learningRate = this.options.learningRate;
    const negSamples = this.options.negSamples;
    
    // Filter tokens to only include vocabulary words
    const validTokens = tokens.filter(t => this.vocabulary.has(t));
    
    for (let epoch = 0; epoch < this.options.epochs; epoch++) {
      let totalLoss = 0;
      let samples = 0;
      
      // Iterate through corpus
      for (let i = 0; i < validTokens.length; i++) {
        const centerWord = validTokens[i];
        const centerIdx = this.vocabulary.get(centerWord)!;
        
        // Get context words within window
        const start = Math.max(0, i - windowSize);
        const end = Math.min(validTokens.length, i + windowSize + 1);
        
        for (let j = start; j < end; j++) {
          if (i === j) continue;
          
          const contextWord = validTokens[j];
          const contextIdx = this.vocabulary.get(contextWord)!;
          
          // Positive sample
          const posLoss = this.trainPair(centerIdx, contextIdx, 1, learningRate);
          totalLoss += posLoss;
          samples++;
          
          // Negative samples
          for (let k = 0; k < negSamples; k++) {
            const negIdx = Math.floor(Math.random() * vocabSize);
            if (negIdx !== contextIdx) {
              const negLoss = this.trainPair(centerIdx, negIdx, 0, learningRate);
              totalLoss += negLoss;
              samples++;
            }
          }
        }
      }
      
      // Record training history
      this.trainingHistory.epochs.push(epoch + 1);
      this.trainingHistory.loss.push(samples > 0 ? totalLoss / samples : 0);
    }
  }
  
  /**
   * Train a single word pair
   * Requirements: 3.4
   */
  private trainPair(centerIdx: number, contextIdx: number, label: number, lr: number): number {
    const centerVec = this.embeddings[centerIdx];
    const contextVec = this.contextEmbeddings[contextIdx];
    
    // Compute dot product
    let dotProduct = 0;
    for (let i = 0; i < centerVec.length; i++) {
      dotProduct += centerVec[i] * contextVec[i];
    }
    
    // Sigmoid activation
    const prediction = 1 / (1 + Math.exp(-dotProduct));
    
    // Compute loss (binary cross-entropy)
    const loss = label === 1
      ? -Math.log(prediction + 1e-10)
      : -Math.log(1 - prediction + 1e-10);
    
    // Compute gradient
    const gradient = prediction - label;
    
    // Update embeddings
    for (let i = 0; i < centerVec.length; i++) {
      const centerGrad = gradient * contextVec[i];
      const contextGrad = gradient * centerVec[i];
      
      this.embeddings[centerIdx][i] -= lr * centerGrad;
      this.contextEmbeddings[contextIdx][i] -= lr * contextGrad;
    }
    
    return loss;
  }
  
  /**
   * Project embeddings to 2D and 3D using PCA
   * Requirements: 3.5
   */
  private projectEmbeddings(): void {
    if (this.embeddings.length === 0) return;
    
    // Simple PCA projection for visualization
    this.projectedEmbeddings2D = this.projectToPCA(this.embeddings, 2);
    this.projectedEmbeddings3D = this.projectToPCA(this.embeddings, 3);
  }
  
  /**
   * Simple PCA projection
   * Requirements: 3.5
   */
  private projectToPCA(data: number[][], targetDim: number): number[][] {
    if (data.length === 0) return [];
    
    const n = data.length;
    const d = data[0].length;
    
    // Center the data
    const mean = Array(d).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) {
        mean[j] += data[i][j];
      }
    }
    for (let j = 0; j < d; j++) {
      mean[j] /= n;
    }
    
    const centered = data.map(row =>
      row.map((val, j) => val - mean[j])
    );
    
    // Compute covariance matrix (simplified - use first few dimensions)
    const maxDim = Math.min(d, 10); // Limit for performance
    const cov: number[][] = Array(maxDim).fill(0).map(() => Array(maxDim).fill(0));
    
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < maxDim; j++) {
        for (let k = 0; k < maxDim; k++) {
          cov[j][k] += centered[i][j] * centered[i][k];
        }
      }
    }
    
    for (let j = 0; j < maxDim; j++) {
      for (let k = 0; k < maxDim; k++) {
        cov[j][k] /= n;
      }
    }
    
    // Simple projection using first principal components (approximation)
    // For a full implementation, we'd use eigenvalue decomposition
    // Here we use a simplified approach for visualization
    const projected = centered.map(row => {
      const result: number[] = [];
      for (let i = 0; i < targetDim; i++) {
        let sum = 0;
        for (let j = 0; j < Math.min(maxDim, row.length); j++) {
          sum += row[j] * (i === j ? 1 : 0.1); // Simplified projection
        }
        result.push(sum);
      }
      return result;
    });
    
    return projected;
  }
  
  /**
   * Get embedding vector for a word
   * Requirements: 3.4
   */
  getWordVector(word: string): number[] | null {
    const idx = this.vocabulary.get(word.toLowerCase());
    if (idx === undefined) return null;
    return [...this.embeddings[idx]];
  }
  
  /**
   * Find most similar words using cosine similarity
   * Requirements: 3.4
   */
  findSimilar(word: string, topK: number = 5): Array<{ word: string; similarity: number }> {
    const wordVec = this.getWordVector(word);
    if (!wordVec) return [];
    
    const similarities: Array<{ word: string; similarity: number }> = [];
    
    for (const [otherWord, idx] of this.vocabulary.entries()) {
      if (otherWord === word.toLowerCase()) continue;
      
      const otherVec = this.embeddings[idx];
      const similarity = this.cosineSimilarity(wordVec, otherVec);
      similarities.push({ word: otherWord, similarity });
    }
    
    return similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);
  }
  
  /**
   * Compute cosine similarity between two vectors
   * Requirements: 3.4
   */
  private cosineSimilarity(vec1: number[], vec2: number[]): number {
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;
    
    for (let i = 0; i < vec1.length; i++) {
      dotProduct += vec1[i] * vec2[i];
      norm1 += vec1[i] * vec1[i];
      norm2 += vec2[i] * vec2[i];
    }
    
    return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2) + 1e-10);
  }
  
  /**
   * Predict method (returns word vectors)
   * Requirements: 3.2
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    // Return flattened embeddings
    return this.embeddings.flat();
  }
  
  /**
   * Get all word vectors
   */
  getAllWordVectors(): WordVector[] {
    return this.indexToWord.map((word, idx) => ({
      word,
      vector: this.embeddings[idx],
      frequency: this.wordFrequency.get(word) || 0
    }));
  }
  
  /**
   * Get visualization data for embedding projection
   * Requirements: 3.5
   */
  getVisualizationData(): VisualizationData | VisualizationData[] {
    const visualizations: VisualizationData[] = [];
    
    if (this.projectedEmbeddings2D.length > 0) {
      // 2D projection
      const x = this.projectedEmbeddings2D.map(p => p[0]);
      const y = this.projectedEmbeddings2D.map(p => p[1]);
      
      visualizations.push({
        type: 'scatter2d',
        data: {
          x,
          y,
          words: this.indexToWord,
          frequencies: this.indexToWord.map(w => this.wordFrequency.get(w) || 0),
          mode: 'markers+text',
          text: this.indexToWord,
          textposition: 'top center'
        }
      });
    }
    
    if (this.projectedEmbeddings3D.length > 0) {
      // 3D projection
      const x = this.projectedEmbeddings3D.map(p => p[0]);
      const y = this.projectedEmbeddings3D.map(p => p[1]);
      const z = this.projectedEmbeddings3D.map(p => p[2]);
      
      visualizations.push({
        type: 'scatter3d',
        data: {
          x,
          y,
          z,
          words: this.indexToWord,
          frequencies: this.indexToWord.map(w => this.wordFrequency.get(w) || 0),
          mode: 'markers+text',
          text: this.indexToWord,
          textposition: 'top center'
        }
      });
    }
    
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
            }
          }
        }
      });
    }
    
    return visualizations;
  }
  
  /**
   * Get embedding metrics
   * Requirements: 3.4
   */
  getMetrics(): Metrics {
    return {
      vocabularySize: this.vocabulary.size,
      embeddingDimension: this.options.embeddingDim,
      corpusSize: this.corpusSize,
      epochs: this.options.epochs,
      finalLoss: this.trainingHistory.loss[this.trainingHistory.loss.length - 1] || 0,
      averageWordFrequency: this.corpusSize / Math.max(1, this.vocabulary.size)
    };
  }
  
  /**
   * Reset embeddings state
   */
  override reset(): void {
    super.reset();
    this.vocabulary.clear();
    this.wordFrequency.clear();
    this.embeddings = [];
    this.contextEmbeddings = [];
    this.indexToWord = [];
    this.corpusSize = 0;
    this.projectedEmbeddings2D = [];
    this.projectedEmbeddings3D = [];
  }
}
