/**
 * Sentiment Analysis Algorithm
 * 
 * Implements text classification for sentiment analysis using a simple
 * bag-of-words approach with logistic regression. Classifies text as
 * positive, negative, or neutral sentiment.
 * 
 * Requirements: 3.2, 3.4, 3.5
 */

import { BaseAlgorithm, type VisualizationData, type Metrics } from '~/types/algorithm';

export interface SentimentAnalysisOptions {
  sentimentClasses: number; // 2 (positive/negative) or 3 (positive/neutral/negative)
  learningRate: number;
  epochs: number;
  regularization: number;
  minWordFrequency: number;
}

export interface SentimentPrediction {
  text: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  confidence: number;
  scores: { [key: string]: number };
  highlightedWords: Array<{ word: string; sentiment: string; weight: number }>;
}

/**
 * Sentiment Analysis using Bag-of-Words with Logistic Regression
 * Requirements: 3.2, 3.4
 */
export class SentimentAnalysis extends BaseAlgorithm {
  name = 'Sentiment Analysis';
  
  private options: SentimentAnalysisOptions;
  private vocabulary: Map<string, number> = new Map(); // word -> index
  private wordWeights: number[][] = []; // [class][word] weights
  private biases: number[] = [];
  private indexToWord: string[] = [];
  private trainingTexts: string[] = [];
  private trainingLabels: number[] = [];
  private predictions: SentimentPrediction[] = [];
  
  // Sentiment lexicon for feature engineering
  private positiveWords = new Set([
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love',
    'best', 'perfect', 'beautiful', 'awesome', 'brilliant', 'outstanding',
    'superb', 'terrific', 'fabulous', 'delightful', 'happy', 'joy', 'pleased'
  ]);
  
  private negativeWords = new Set([
    'bad', 'terrible', 'awful', 'horrible', 'poor', 'worst', 'hate',
    'disappointing', 'disappointed', 'sad', 'angry', 'annoying', 'frustrating',
    'useless', 'waste', 'boring', 'dull', 'ugly', 'disgusting', 'pathetic'
  ]);
  
  constructor(options: Partial<SentimentAnalysisOptions> = {}) {
    super();
    this.options = {
      sentimentClasses: options.sentimentClasses ?? 3,
      learningRate: options.learningRate ?? 0.1,
      epochs: options.epochs ?? 50,
      regularization: options.regularization ?? 0.01,
      minWordFrequency: options.minWordFrequency ?? 2
    };
  }
  
  /**
   * Train sentiment classifier on labeled text data
   * Requirements: 3.2, 3.4
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    throw new Error('Use trainOnText() method for sentiment analysis instead of fit()');
  }
  
  /**
   * Train on text corpus with sentiment labels
   * Requirements: 3.2, 3.4
   * 
   * @param texts - Array of text samples
   * @param labels - Array of sentiment labels (0: negative, 1: neutral, 2: positive for 3-class)
   *                 or (0: negative, 1: positive for 2-class)
   */
  async trainOnText(texts: string[], labels: number[]): Promise<void> {
    if (texts.length !== labels.length) {
      throw new Error('Number of texts must match number of labels');
    }
    
    if (texts.length === 0) {
      throw new Error('Training data cannot be empty');
    }
    
    this.trainingTexts = texts;
    this.trainingLabels = labels;
    
    // Build vocabulary from training texts
    this.buildVocabulary(texts);
    
    if (this.vocabulary.size === 0) {
      throw new Error('Vocabulary is empty after filtering');
    }
    
    // Initialize weights
    this.initializeWeights();
    
    // Train using gradient descent
    await this.trainClassifier(texts, labels);
    
    this.trained = true;
  }
  
  /**
   * Tokenize and normalize text
   * Requirements: 3.2
   */
  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(token => token.length > 1); // Filter out single characters
  }
  
  /**
   * Build vocabulary from texts
   * Requirements: 3.4
   */
  private buildVocabulary(texts: string[]): void {
    this.vocabulary.clear();
    this.indexToWord = [];
    
    // Count word frequencies
    const wordCounts = new Map<string, number>();
    for (const text of texts) {
      const tokens = this.tokenize(text);
      for (const token of tokens) {
        wordCounts.set(token, (wordCounts.get(token) || 0) + 1);
      }
    }
    
    // Build vocabulary with minimum frequency threshold
    let index = 0;
    for (const [word, count] of wordCounts.entries()) {
      if (count >= this.options.minWordFrequency) {
        this.vocabulary.set(word, index);
        this.indexToWord.push(word);
        index++;
      }
    }
  }
  
  /**
   * Initialize weight matrices
   * Requirements: 3.4
   */
  private initializeWeights(): void {
    const vocabSize = this.vocabulary.size;
    const numClasses = this.options.sentimentClasses;
    
    // Initialize weights for each class
    this.wordWeights = Array(numClasses).fill(0).map(() =>
      Array(vocabSize).fill(0).map(() => (Math.random() - 0.5) * 0.01)
    );
    
    // Initialize biases
    this.biases = Array(numClasses).fill(0);
  }
  
  /**
   * Convert text to feature vector (bag-of-words)
   * Requirements: 3.4
   */
  private textToFeatures(text: string): number[] {
    const features = Array(this.vocabulary.size).fill(0);
    const tokens = this.tokenize(text);
    
    for (const token of tokens) {
      const idx = this.vocabulary.get(token);
      if (idx !== undefined) {
        features[idx] += 1; // Count occurrences
      }
    }
    
    // Normalize by document length
    const total = tokens.length || 1;
    for (let i = 0; i < features.length; i++) {
      features[i] /= total;
    }
    
    return features;
  }
  
  /**
   * Compute softmax probabilities
   * Requirements: 3.4
   */
  private softmax(scores: number[]): number[] {
    const maxScore = Math.max(...scores);
    const expScores = scores.map(s => Math.exp(s - maxScore));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    return expScores.map(s => s / sumExp);
  }
  
  /**
   * Train classifier using gradient descent
   * Requirements: 3.4
   */
  private async trainClassifier(texts: string[], labels: number[]): Promise<void> {
    const numClasses = this.options.sentimentClasses;
    const lr = this.options.learningRate;
    const reg = this.options.regularization;
    
    // Convert texts to features
    const features = texts.map(text => this.textToFeatures(text));
    
    for (let epoch = 0; epoch < this.options.epochs; epoch++) {
      let totalLoss = 0;
      
      // Shuffle training data
      const indices = Array.from({ length: texts.length }, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
      
      // Process each sample
      for (const idx of indices) {
        const x = features[idx];
        const trueLabel = labels[idx];
        
        // Forward pass: compute scores for each class
        const scores = this.biases.map((bias, classIdx) => {
          let score = bias;
          for (let i = 0; i < x.length; i++) {
            score += this.wordWeights[classIdx][i] * x[i];
          }
          return score;
        });
        
        // Compute probabilities
        const probs = this.softmax(scores);
        
        // Compute loss (cross-entropy)
        totalLoss += -Math.log(probs[trueLabel] + 1e-10);
        
        // Backward pass: compute gradients and update weights
        for (let classIdx = 0; classIdx < numClasses; classIdx++) {
          const error = probs[classIdx] - (classIdx === trueLabel ? 1 : 0);
          
          // Update bias
          this.biases[classIdx] -= lr * error;
          
          // Update weights with L2 regularization
          for (let i = 0; i < x.length; i++) {
            const gradient = error * x[i] + reg * this.wordWeights[classIdx][i];
            this.wordWeights[classIdx][i] -= lr * gradient;
          }
        }
      }
      
      // Record training history
      this.trainingHistory.epochs.push(epoch + 1);
      this.trainingHistory.loss.push(totalLoss / texts.length);
      
      // Decay learning rate
      if ((epoch + 1) % 10 === 0) {
        this.options.learningRate *= 0.9;
      }
    }
  }
  
  /**
   * Predict sentiment for new texts
   * Requirements: 3.2, 3.4
   */
  predict(X: number[][]): string[] {
    throw new Error('Use predictSentiment() method for sentiment analysis');
  }
  
  /**
   * Predict sentiment for text samples
   * Requirements: 3.2, 3.4
   */
  predictSentiment(texts: string[]): SentimentPrediction[] {
    this.ensureTrained();
    
    this.predictions = texts.map(text => {
      const features = this.textToFeatures(text);
      
      // Compute scores
      const scores = this.biases.map((bias, classIdx) => {
        let score = bias;
        for (let i = 0; i < features.length; i++) {
          score += this.wordWeights[classIdx][i] * features[i];
        }
        return score;
      });
      
      // Get probabilities
      const probs = this.softmax(scores);
      
      // Get predicted class
      const predictedClass = probs.indexOf(Math.max(...probs));
      const confidence = probs[predictedClass];
      
      // Map class to sentiment label
      let sentiment: 'positive' | 'negative' | 'neutral';
      if (this.options.sentimentClasses === 2) {
        sentiment = predictedClass === 0 ? 'negative' : 'positive';
      } else {
        sentiment = predictedClass === 0 ? 'negative' : 
                   predictedClass === 1 ? 'neutral' : 'positive';
      }
      
      // Get word contributions for highlighting
      const highlightedWords = this.getWordContributions(text, predictedClass);
      
      // Build scores object
      const scoresObj: { [key: string]: number } = {};
      if (this.options.sentimentClasses === 2) {
        scoresObj.negative = probs[0];
        scoresObj.positive = probs[1];
      } else {
        scoresObj.negative = probs[0];
        scoresObj.neutral = probs[1];
        scoresObj.positive = probs[2];
      }
      
      return {
        text,
        sentiment,
        confidence,
        scores: scoresObj,
        highlightedWords
      };
    });
    
    return this.predictions;
  }
  
  /**
   * Get word contributions to sentiment prediction
   * Requirements: 3.5
   */
  private getWordContributions(text: string, predictedClass: number): Array<{ word: string; sentiment: string; weight: number }> {
    const tokens = this.tokenize(text);
    const contributions: Array<{ word: string; sentiment: string; weight: number }> = [];
    
    for (const token of tokens) {
      const idx = this.vocabulary.get(token);
      if (idx !== undefined) {
        const weight = this.wordWeights[predictedClass][idx];
        
        // Determine sentiment contribution
        let sentiment: string;
        if (Math.abs(weight) < 0.01) {
          sentiment = 'neutral';
        } else if (weight > 0) {
          sentiment = predictedClass === 2 || (predictedClass === 1 && this.options.sentimentClasses === 2) 
            ? 'positive' : 'negative';
        } else {
          sentiment = predictedClass === 0 ? 'negative' : 'positive';
        }
        
        contributions.push({
          word: token,
          sentiment,
          weight: Math.abs(weight)
        });
      }
    }
    
    // Sort by weight magnitude
    contributions.sort((a, b) => b.weight - a.weight);
    
    return contributions;
  }
  
  /**
   * Get top words for each sentiment class
   * Requirements: 3.4
   */
  getTopWords(topK: number = 10): { [key: string]: Array<{ word: string; weight: number }> } {
    this.ensureTrained();
    
    const result: { [key: string]: Array<{ word: string; weight: number }> } = {};
    const classNames = this.options.sentimentClasses === 2 
      ? ['negative', 'positive']
      : ['negative', 'neutral', 'positive'];
    
    for (let classIdx = 0; classIdx < this.options.sentimentClasses; classIdx++) {
      const words = this.indexToWord.map((word, idx) => ({
        word,
        weight: this.wordWeights[classIdx][idx]
      }));
      
      words.sort((a, b) => b.weight - a.weight);
      result[classNames[classIdx]] = words.slice(0, topK);
    }
    
    return result;
  }
  
  /**
   * Get visualization data for sentiment highlighting
   * Requirements: 3.5
   */
  getVisualizationData(): VisualizationData | VisualizationData[] {
    const visualizations: VisualizationData[] = [];
    
    // Sentiment predictions visualization
    if (this.predictions.length > 0) {
      visualizations.push({
        type: 'scatter2d',
        data: {
          predictions: this.predictions,
          type: 'sentiment-highlight'
        }
      });
      
      // Sentiment distribution
      const sentimentCounts = { positive: 0, negative: 0, neutral: 0 };
      for (const pred of this.predictions) {
        sentimentCounts[pred.sentiment]++;
      }
      
      visualizations.push({
        type: 'line',
        data: {
          labels: Object.keys(sentimentCounts),
          datasets: [{
            label: 'Sentiment Distribution',
            data: Object.values(sentimentCounts),
            backgroundColor: [
              'rgba(239, 68, 68, 0.5)',   // red for negative
              'rgba(156, 163, 175, 0.5)', // gray for neutral
              'rgba(34, 197, 94, 0.5)'    // green for positive
            ],
            borderColor: [
              'rgb(239, 68, 68)',
              'rgb(156, 163, 175)',
              'rgb(34, 197, 94)'
            ],
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
                text: 'Sentiment Distribution'
              }
            }
          }
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
            },
            scales: {
              y: {
                beginAtZero: false
              }
            }
          }
        }
      });
    }
    
    // Top words visualization
    if (this.trained) {
      const topWords = this.getTopWords(15);
      const datasets = Object.entries(topWords).map(([sentiment, words], idx) => {
        const colors = {
          negative: 'rgb(239, 68, 68)',
          neutral: 'rgb(156, 163, 175)',
          positive: 'rgb(34, 197, 94)'
        };
        
        return {
          label: sentiment.charAt(0).toUpperCase() + sentiment.slice(1),
          data: words.map(w => w.weight),
          backgroundColor: colors[sentiment as keyof typeof colors] + '80',
          borderColor: colors[sentiment as keyof typeof colors],
          borderWidth: 2
        };
      });
      
      visualizations.push({
        type: 'line',
        data: {
          labels: topWords[Object.keys(topWords)[0]].map(w => w.word),
          datasets
        },
        config: {
          type: 'bar',
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              title: {
                display: true,
                text: 'Top Words by Sentiment'
              }
            }
          }
        }
      });
    }
    
    return visualizations;
  }
  
  /**
   * Get sentiment analysis metrics
   * Requirements: 3.4
   */
  getMetrics(): Metrics {
    const metrics: Metrics = {
      vocabularySize: this.vocabulary.size,
      sentimentClasses: this.options.sentimentClasses,
      trainingSize: this.trainingTexts.length,
      epochs: this.options.epochs,
      finalLoss: this.trainingHistory.loss[this.trainingHistory.loss.length - 1] || 0
    };
    
    // If we have predictions and labels, compute accuracy
    if (this.predictions.length > 0 && this.trainingLabels.length > 0) {
      const classNames = this.options.sentimentClasses === 2 
        ? ['negative', 'positive']
        : ['negative', 'neutral', 'positive'];
      
      let correct = 0;
      for (let i = 0; i < Math.min(this.predictions.length, this.trainingLabels.length); i++) {
        const predictedLabel = classNames.indexOf(this.predictions[i].sentiment);
        if (predictedLabel === this.trainingLabels[i]) {
          correct++;
        }
      }
      
      metrics.accuracy = correct / Math.min(this.predictions.length, this.trainingLabels.length);
    }
    
    return metrics;
  }
  
  /**
   * Reset sentiment analyzer state
   */
  override reset(): void {
    super.reset();
    this.vocabulary.clear();
    this.wordWeights = [];
    this.biases = [];
    this.indexToWord = [];
    this.trainingTexts = [];
    this.trainingLabels = [];
    this.predictions = [];
  }
}
