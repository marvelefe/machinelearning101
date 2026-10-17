/**
 * Text Preprocessing & Tokenization Algorithm
 * 
 * Implements word tokenization, sentence splitting, and text normalization
 * for natural language processing tasks.
 * 
 * Requirements: 3.2, 3.4, 3.5
 */

import { BaseAlgorithm, type VisualizationData, type Metrics } from '~/types/algorithm';

export interface TokenizationOptions {
  lowercase: boolean;
  removePunctuation: boolean;
  removeStopwords: boolean;
  stemming: boolean;
  tokenizeBy: 'word' | 'sentence' | 'character';
}

export interface Token {
  text: string;
  start: number;
  end: number;
  type: 'word' | 'punctuation' | 'stopword' | 'number' | 'whitespace';
  normalized?: string;
}

export interface Sentence {
  text: string;
  start: number;
  end: number;
  tokens: Token[];
}

/**
 * Common English stopwords for filtering
 */
const STOPWORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
  'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
  'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
  'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how'
]);

export class Tokenizer extends BaseAlgorithm {
  name = 'Text Tokenization';
  
  private options: TokenizationOptions;
  private inputText: string = '';
  private tokens: Token[] = [];
  private sentences: Sentence[] = [];
  private vocabulary: Set<string> = new Set();
  private tokenCounts: Map<string, number> = new Map();
  
  constructor(options: Partial<TokenizationOptions> = {}) {
    super();
    this.options = {
      lowercase: options.lowercase ?? true,
      removePunctuation: options.removePunctuation ?? false,
      removeStopwords: options.removeStopwords ?? false,
      stemming: options.stemming ?? false,
      tokenizeBy: options.tokenizeBy ?? 'word'
    };
  }
  
  /**
   * Process the input text and perform tokenization
   * Requirements: 3.2, 3.4
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    // For text processing, we expect the text to be passed differently
    // This is a placeholder to maintain interface compatibility
    throw new Error('Use processText() method for tokenization instead of fit()');
  }
  
  /**
   * Process text input and perform tokenization
   * Requirements: 3.2, 3.4
   */
  async processText(text: string): Promise<void> {
    this.inputText = text;
    this.tokens = [];
    this.sentences = [];
    this.vocabulary.clear();
    this.tokenCounts.clear();
    
    if (!text || text.trim().length === 0) {
      this.trained = true;
      return;
    }
    
    // Perform tokenization based on mode
    switch (this.options.tokenizeBy) {
      case 'sentence':
        this.tokenizeBySentence();
        break;
      case 'character':
        this.tokenizeByCharacter();
        break;
      case 'word':
      default:
        this.tokenizeByWord();
        break;
    }
    
    // Build vocabulary and count tokens
    this.buildVocabulary();
    
    this.trained = true;
  }
  
  /**
   * Tokenize text by words
   * Requirements: 3.2
   */
  private tokenizeByWord(): void {
    const text = this.inputText;
    let position = 0;
    
    // Regular expression to match words, punctuation, numbers, and whitespace
    const tokenRegex = /(\w+|[^\w\s]|\s+)/g;
    let match;
    
    while ((match = tokenRegex.exec(text)) !== null) {
      const tokenText = match[0];
      const start = match.index;
      const end = start + tokenText.length;
      
      // Determine token type
      let type: Token['type'] = 'word';
      if (/^\s+$/.test(tokenText)) {
        type = 'whitespace';
      } else if (/^\d+$/.test(tokenText)) {
        type = 'number';
      } else if (/^[^\w\s]$/.test(tokenText)) {
        type = 'punctuation';
      } else if (this.options.lowercase && STOPWORDS.has(tokenText.toLowerCase())) {
        type = 'stopword';
      }
      
      // Apply normalization
      let normalized = tokenText;
      if (this.options.lowercase && type === 'word') {
        normalized = tokenText.toLowerCase();
      }
      
      // Skip based on options
      if (this.options.removePunctuation && type === 'punctuation') {
        continue;
      }
      if (this.options.removeStopwords && type === 'stopword') {
        continue;
      }
      
      // Apply stemming (simple suffix removal)
      if (this.options.stemming && type === 'word') {
        normalized = this.simpleStem(normalized);
      }
      
      this.tokens.push({
        text: tokenText,
        start,
        end,
        type,
        normalized
      });
    }
  }
  
  /**
   * Tokenize text by sentences
   * Requirements: 3.2
   */
  private tokenizeBySentence(): void {
    const text = this.inputText;
    
    // Split by sentence boundaries (., !, ?)
    const sentenceRegex = /[^.!?]+[.!?]+/g;
    let match;
    let position = 0;
    
    while ((match = sentenceRegex.exec(text)) !== null) {
      const sentenceText = match[0];
      const start = match.index;
      const end = start + sentenceText.length;
      
      // Tokenize words within the sentence
      const savedOptions = { ...this.options };
      this.options.tokenizeBy = 'word';
      const savedText = this.inputText;
      this.inputText = sentenceText;
      
      const sentenceTokens: Token[] = [];
      this.tokenizeByWord();
      sentenceTokens.push(...this.tokens);
      
      // Restore
      this.inputText = savedText;
      this.options = savedOptions;
      
      this.sentences.push({
        text: sentenceText.trim(),
        start,
        end,
        tokens: sentenceTokens
      });
      
      // Add sentence as a token
      this.tokens.push({
        text: sentenceText.trim(),
        start,
        end,
        type: 'word',
        normalized: this.options.lowercase ? sentenceText.trim().toLowerCase() : sentenceText.trim()
      });
    }
    
    // Handle remaining text if no sentence boundary found
    if (this.sentences.length === 0 && text.trim().length > 0) {
      this.sentences.push({
        text: text.trim(),
        start: 0,
        end: text.length,
        tokens: []
      });
      
      this.tokens.push({
        text: text.trim(),
        start: 0,
        end: text.length,
        type: 'word',
        normalized: this.options.lowercase ? text.trim().toLowerCase() : text.trim()
      });
    }
  }
  
  /**
   * Tokenize text by characters
   * Requirements: 3.2
   */
  private tokenizeByCharacter(): void {
    const text = this.inputText;
    
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      let type: Token['type'] = 'word';
      
      if (/\s/.test(char)) {
        type = 'whitespace';
      } else if (/[^\w\s]/.test(char)) {
        type = 'punctuation';
      } else if (/\d/.test(char)) {
        type = 'number';
      }
      
      this.tokens.push({
        text: char,
        start: i,
        end: i + 1,
        type,
        normalized: this.options.lowercase ? char.toLowerCase() : char
      });
    }
  }
  
  /**
   * Simple stemming algorithm (Porter-like suffix removal)
   * Requirements: 3.2
   */
  private simpleStem(word: string): string {
    // Simple suffix removal rules
    const suffixes = ['ing', 'ed', 'es', 's', 'ly', 'er', 'est'];
    
    for (const suffix of suffixes) {
      if (word.endsWith(suffix) && word.length > suffix.length + 2) {
        return word.slice(0, -suffix.length);
      }
    }
    
    return word;
  }
  
  /**
   * Build vocabulary from tokens
   * Requirements: 3.4
   */
  private buildVocabulary(): void {
    for (const token of this.tokens) {
      if (token.type === 'word' || token.type === 'number') {
        const key = token.normalized || token.text;
        this.vocabulary.add(key);
        this.tokenCounts.set(key, (this.tokenCounts.get(key) || 0) + 1);
      }
    }
  }
  
  /**
   * Predict method (not applicable for tokenization)
   * Requirements: 3.2
   */
  predict(X: number[][]): string[] {
    this.ensureTrained();
    return this.tokens.map(t => t.text);
  }
  
  /**
   * Get tokens as array
   */
  getTokens(): Token[] {
    return this.tokens;
  }
  
  /**
   * Get sentences (if sentence tokenization was used)
   */
  getSentences(): Sentence[] {
    return this.sentences;
  }
  
  /**
   * Get vocabulary
   */
  getVocabulary(): string[] {
    return Array.from(this.vocabulary);
  }
  
  /**
   * Get token frequency counts
   */
  getTokenCounts(): Map<string, number> {
    return this.tokenCounts;
  }
  
  /**
   * Get visualization data for token highlighting
   * Requirements: 3.5
   */
  getVisualizationData(): VisualizationData | VisualizationData[] {
    const visualizations: VisualizationData[] = [];
    
    // Token highlighting data
    visualizations.push({
      type: 'scatter2d',
      data: {
        tokens: this.tokens,
        text: this.inputText,
        sentences: this.sentences
      }
    });
    
    // Token frequency distribution
    const sortedTokens = Array.from(this.tokenCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20); // Top 20 tokens
    
    if (sortedTokens.length > 0) {
      visualizations.push({
        type: 'line',
        data: {
          labels: sortedTokens.map(([token]) => token),
          datasets: [{
            label: 'Token Frequency',
            data: sortedTokens.map(([, count]) => count),
            backgroundColor: 'rgba(99, 102, 241, 0.5)',
            borderColor: 'rgb(99, 102, 241)',
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
                text: 'Top 20 Token Frequencies'
              }
            }
          }
        }
      });
    }
    
    return visualizations;
  }
  
  /**
   * Get tokenization metrics
   * Requirements: 3.4
   */
  getMetrics(): Metrics {
    const wordTokens = this.tokens.filter(t => t.type === 'word' || t.type === 'number');
    const stopwordTokens = this.tokens.filter(t => t.type === 'stopword');
    const punctuationTokens = this.tokens.filter(t => t.type === 'punctuation');
    
    return {
      totalTokens: this.tokens.length,
      wordTokens: wordTokens.length,
      stopwordTokens: stopwordTokens.length,
      punctuationTokens: punctuationTokens.length,
      vocabularySize: this.vocabulary.size,
      sentenceCount: this.sentences.length,
      averageTokensPerSentence: this.sentences.length > 0 
        ? this.tokens.length / this.sentences.length 
        : 0,
      uniqueTokenRatio: this.tokens.length > 0 
        ? this.vocabulary.size / this.tokens.length 
        : 0
    };
  }
  
  /**
   * Reset tokenizer state
   */
  override reset(): void {
    super.reset();
    this.inputText = '';
    this.tokens = [];
    this.sentences = [];
    this.vocabulary.clear();
    this.tokenCounts.clear();
  }
}
