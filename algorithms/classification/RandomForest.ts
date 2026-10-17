/**
 * Random Forest Classification Algorithm
 * 
 * Implements Random Forest using an ensemble of decision trees with bootstrap sampling
 * and random feature selection. Combines predictions through majority voting.
 * 
 * Requirements: 6.1, 6.2, 6.3, 6.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics, TrainingHistory } from '~/types/algorithm';
import { mean, mode } from '~/utils/math';

export interface RandomForestParams {
  nEstimators: number;
  maxDepth?: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  maxFeatures?: number | 'sqrt' | 'log2';
  bootstrap: boolean;
  randomState?: number;
}

interface DecisionNode {
  featureIndex?: number;
  threshold?: number;
  left?: DecisionNode;
  right?: DecisionNode;
  prediction?: number;
  samples?: number;
  impurity?: number;
}

class DecisionTree {
  private root: DecisionNode | null = null;
  private maxDepth: number;
  private minSamplesSplit: number;
  private minSamplesLeaf: number;
  private maxFeatures: number;
  private randomState: number;

  constructor(
    maxDepth: number = 10,
    minSamplesSplit: number = 2,
    minSamplesLeaf: number = 1,
    maxFeatures: number = -1,
    randomState: number = 42
  ) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.minSamplesLeaf = minSamplesLeaf;
    this.maxFeatures = maxFeatures;
    this.randomState = randomState;
  }

  /**
   * Simple random number generator for reproducibility
   */
  private random(): number {
    this.randomState = (this.randomState * 9301 + 49297) % 233280;
    return this.randomState / 233280;
  }

  /**
   * Calculate Gini impurity
   */
  private calculateGini(y: number[]): number {
    if (y.length === 0) return 0;

    const counts = new Map<number, number>();
    for (const label of y) {
      counts.set(label, (counts.get(label) || 0) + 1);
    }

    let gini = 1;
    const total = y.length;
    for (const count of counts.values()) {
      const prob = count / total;
      gini -= prob * prob;
    }

    return gini;
  }

  /**
   * Find the best split for a node
   */
  private findBestSplit(X: number[][], y: number[], featureIndices: number[]): {
    featureIndex: number;
    threshold: number;
    gain: number;
  } | null {
    let bestGain = 0;
    let bestFeature = -1;
    let bestThreshold = 0;

    const parentImpurity = this.calculateGini(y);

    for (const featureIndex of featureIndices) {
      const values = X.map(row => row[featureIndex]);
      const uniqueValues = [...new Set(values)].sort((a, b) => a - b);

      for (let i = 0; i < uniqueValues.length - 1; i++) {
        const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;

        const leftIndices: number[] = [];
        const rightIndices: number[] = [];

        for (let j = 0; j < X.length; j++) {
          if (X[j][featureIndex] <= threshold) {
            leftIndices.push(j);
          } else {
            rightIndices.push(j);
          }
        }

        if (leftIndices.length < this.minSamplesLeaf || 
            rightIndices.length < this.minSamplesLeaf) {
          continue;
        }

        const leftY = leftIndices.map(idx => y[idx]);
        const rightY = rightIndices.map(idx => y[idx]);

        const leftImpurity = this.calculateGini(leftY);
        const rightImpurity = this.calculateGini(rightY);

        const weightedImpurity = 
          (leftY.length / y.length) * leftImpurity +
          (rightY.length / y.length) * rightImpurity;

        const gain = parentImpurity - weightedImpurity;

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = featureIndex;
          bestThreshold = threshold;
        }
      }
    }

    if (bestFeature === -1) return null;

    return {
      featureIndex: bestFeature,
      threshold: bestThreshold,
      gain: bestGain
    };
  }

  /**
   * Get random subset of features
   */
  private getRandomFeatures(nFeatures: number): number[] {
    const features = Array.from({ length: nFeatures }, (_, i) => i);
    const nSelected = this.maxFeatures === -1 ? nFeatures : 
                     Math.min(this.maxFeatures, nFeatures);

    // Fisher-Yates shuffle
    for (let i = features.length - 1; i > 0; i--) {
      const j = Math.floor(this.random() * (i + 1));
      [features[i], features[j]] = [features[j], features[i]];
    }

    return features.slice(0, nSelected);
  }

  /**
   * Build decision tree recursively
   */
  private buildTree(
    X: number[][],
    y: number[],
    depth: number = 0
  ): DecisionNode {
    const samples = y.length;
    const uniqueClasses = [...new Set(y)];

    // Create leaf node if stopping criteria met
    if (depth >= this.maxDepth ||
        samples < this.minSamplesSplit ||
        uniqueClasses.length === 1) {
      
      const counts = new Map<number, number>();
      for (const label of y) {
        counts.set(label, (counts.get(label) || 0) + 1);
      }
      
      let maxCount = 0;
      let prediction = 0;
      for (const [label, count] of counts.entries()) {
        if (count > maxCount) {
          maxCount = count;
          prediction = label;
        }
      }

      return {
        prediction,
        samples,
        impurity: this.calculateGini(y)
      };
    }

    // Find best split
    const featureIndices = this.getRandomFeatures(X[0].length);
    const bestSplit = this.findBestSplit(X, y, featureIndices);

    if (!bestSplit) {
      // No valid split found, create leaf
      const counts = new Map<number, number>();
      for (const label of y) {
        counts.set(label, (counts.get(label) || 0) + 1);
      }
      
      let maxCount = 0;
      let prediction = 0;
      for (const [label, count] of counts.entries()) {
        if (count > maxCount) {
          maxCount = count;
          prediction = label;
        }
      }

      return {
        prediction,
        samples,
        impurity: this.calculateGini(y)
      };
    }

    // Split data
    const leftIndices: number[] = [];
    const rightIndices: number[] = [];

    for (let i = 0; i < X.length; i++) {
      if (X[i][bestSplit.featureIndex] <= bestSplit.threshold) {
        leftIndices.push(i);
      } else {
        rightIndices.push(i);
      }
    }

    const leftX = leftIndices.map(idx => X[idx]);
    const leftY = leftIndices.map(idx => y[idx]);
    const rightX = rightIndices.map(idx => X[idx]);
    const rightY = rightIndices.map(idx => y[idx]);

    // Recursively build subtrees
    const leftChild = this.buildTree(leftX, leftY, depth + 1);
    const rightChild = this.buildTree(rightX, rightY, depth + 1);

    return {
      featureIndex: bestSplit.featureIndex,
      threshold: bestSplit.threshold,
      left: leftChild,
      right: rightChild,
      samples,
      impurity: this.calculateGini(y)
    };
  }

  /**
   * Train the decision tree
   */
  fit(X: number[][], y: number[]): void {
    this.root = this.buildTree(X, y);
  }

  /**
   * Predict single sample
   */
  private predictSample(x: number[]): number {
    let node = this.root;
    
    while (node && node.featureIndex !== undefined) {
      if (x[node.featureIndex] <= node.threshold!) {
        node = node.left!;
      } else {
        node = node.right!;
      }
    }

    return node?.prediction || 0;
  }

  /**
   * Make predictions
   */
  predict(X: number[][]): number[] {
    return X.map(x => this.predictSample(x));
  }
}

export class RandomForest extends BaseAlgorithm {
  name = 'Random Forest';
  
  // Model parameters
  private trees: DecisionTree[] = [];
  private nEstimators: number;
  private maxDepth: number;
  private minSamplesSplit: number;
  private minSamplesLeaf: number;
  private maxFeatures: number;
  private bootstrap: boolean;
  private randomState: number;
  
  // Training data (stored for visualization)
  private X_train: number[][] = [];
  private y_train: number[] = [];
  private classes: number[] = [];
  
  // Training history
  protected override trainingHistory: TrainingHistory & { 
    oobScore: number[];
    featureImportances: number[][];
  } = {
    epochs: [],
    loss: [],
    oobScore: [],
    featureImportances: []
  };
  
  constructor(params: RandomForestParams = {
    nEstimators: 100,
    maxDepth: 10,
    minSamplesSplit: 2,
    minSamplesLeaf: 1,
    maxFeatures: 'sqrt',
    bootstrap: true,
    randomState: 42
  }) {
    super();
    this.nEstimators = params.nEstimators;
    this.maxDepth = params.maxDepth || 10;
    this.minSamplesSplit = params.minSamplesSplit;
    this.minSamplesLeaf = params.minSamplesLeaf;
    this.bootstrap = params.bootstrap;
    this.randomState = params.randomState || 42;
    
    // Set max features
    if (typeof params.maxFeatures === 'number') {
      this.maxFeatures = params.maxFeatures;
    } else {
      this.maxFeatures = -1; // Will be set during fit
    }
  }
  
  /**
   * Simple random number generator for reproducibility
   */
  private random(): number {
    this.randomState = (this.randomState * 9301 + 49297) % 233280;
    return this.randomState / 233280;
  }
  
  /**
   * Bootstrap sampling
   */
  private bootstrapSample(X: number[][], y: number[]): {
    X_sample: number[][];
    y_sample: number[];
    oobIndices: number[];
  } {
    const n = X.length;
    const indices: number[] = [];
    const oobIndices: number[] = [];
    const inBag = new Set<number>();
    
    // Sample with replacement
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(this.random() * n);
      indices.push(idx);
      inBag.add(idx);
    }
    
    // Find out-of-bag samples
    for (let i = 0; i < n; i++) {
      if (!inBag.has(i)) {
        oobIndices.push(i);
      }
    }
    
    const X_sample = indices.map(idx => X[idx]);
    const y_sample = indices.map(idx => y[idx]);
    
    return { X_sample, y_sample, oobIndices };
  }
  
  /**
   * Train the Random Forest
   */
  async fit(X: number[][], y: number[]): Promise<void> {
    // Validate input
    this.validateInput(X);
    this.validateTarget(X, y);
    
    // Store training data
    this.X_train = X.map(row => [...row]);
    this.y_train = [...y];
    this.classes = [...new Set(y)].sort((a, b) => a - b);
    
    const nFeatures = X[0].length;
    
    // Set max features if not specified
    if (this.maxFeatures === -1) {
      this.maxFeatures = Math.floor(Math.sqrt(nFeatures));
    }
    
    // Reset training history
    this.trainingHistory = {
      epochs: [],
      loss: [],
      oobScore: [],
      featureImportances: []
    };
    
    // Initialize trees
    this.trees = [];
    const oobPredictions: Map<number, number[]> = new Map();
    
    // Train each tree
    for (let i = 0; i < this.nEstimators; i++) {
      // Bootstrap sampling
      let X_sample: number[][];
      let y_sample: number[];
      let oobIndices: number[] = [];
      
      if (this.bootstrap) {
        const sample = this.bootstrapSample(X, y);
        X_sample = sample.X_sample;
        y_sample = sample.y_sample;
        oobIndices = sample.oobIndices;
      } else {
        X_sample = X.map(row => [...row]);
        y_sample = [...y];
      }
      
      // Create and train tree
      const tree = new DecisionTree(
        this.maxDepth,
        this.minSamplesSplit,
        this.minSamplesLeaf,
        this.maxFeatures,
        this.randomState + i
      );
      
      tree.fit(X_sample, y_sample);
      this.trees.push(tree);
      
      // Calculate OOB predictions for this tree
      if (this.bootstrap && oobIndices.length > 0) {
        const oobX = oobIndices.map(idx => X[idx]);
        const oobPreds = tree.predict(oobX);
        
        for (let j = 0; j < oobIndices.length; j++) {
          const idx = oobIndices[j];
          if (!oobPredictions.has(idx)) {
            oobPredictions.set(idx, []);
          }
          oobPredictions.get(idx)!.push(oobPreds[j]);
        }
      }
      
      // Calculate OOB score
      let oobScore = 0;
      if (this.bootstrap && oobPredictions.size > 0) {
        let correct = 0;
        let total = 0;
        
        for (const [idx, preds] of oobPredictions.entries()) {
          if (preds.length > 0) {
            const prediction = mode(preds);
            if (prediction === y[idx]) {
              correct++;
            }
            total++;
          }
        }
        
        oobScore = total > 0 ? correct / total : 0;
      }
      
      // Store history
      this.trainingHistory.epochs.push(i);
      this.trainingHistory.loss.push(0); // Not applicable for Random Forest
      this.trainingHistory.oobScore.push(oobScore);
      
      // Allow UI to update periodically
      if (i % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    
    this.trained = true;
  }
  
  /**
   * Make predictions using majority voting
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    this.validateInput(X, this.X_train[0].length);
    
    const predictions: number[][] = [];
    
    // Get predictions from each tree
    for (const tree of this.trees) {
      predictions.push(tree.predict(X));
    }
    
    // Majority voting
    const finalPredictions: number[] = [];
    for (let i = 0; i < X.length; i++) {
      const votes = predictions.map(preds => preds[i]);
      finalPredictions.push(mode(votes));
    }
    
    return finalPredictions;
  }
  
  /**
   * Predict class probabilities
   */
  predictProba(X: number[][]): number[][] {
    this.ensureTrained();
    this.validateInput(X, this.X_train[0].length);
    
    const predictions: number[][] = [];
    
    // Get predictions from each tree
    for (const tree of this.trees) {
      predictions.push(tree.predict(X));
    }
    
    // Calculate probabilities
    const probabilities: number[][] = [];
    for (let i = 0; i < X.length; i++) {
      const votes = predictions.map(preds => preds[i]);
      const classCounts = new Map<number, number>();
      
      // Count votes for each class
      for (const vote of votes) {
        classCounts.set(vote, (classCounts.get(vote) || 0) + 1);
      }
      
      // Convert to probabilities
      const probs: number[] = [];
      for (const cls of this.classes) {
        const count = classCounts.get(cls) || 0;
        probs.push(count / this.trees.length);
      }
      
      probabilities.push(probs);
    }
    
    return probabilities;
  }
  
  /**
   * Generate decision boundary for 2D visualization
   */
  private generateDecisionBoundary(): { x: number[], y: number[], z: number[][] } | null {
    if (this.X_train[0].length !== 2) {
      return null; // Only for 2D features
    }
    
    // Get data range
    const x1Values = this.X_train.map(row => row[0]);
    const x2Values = this.X_train.map(row => row[1]);
    
    const x1Min = Math.min(...x1Values) - 1;
    const x1Max = Math.max(...x1Values) + 1;
    const x2Min = Math.min(...x2Values) - 1;
    const x2Max = Math.max(...x2Values) + 1;
    
    // Create grid
    const resolution = 50;
    const x1Grid: number[] = [];
    const x2Grid: number[] = [];
    const predGrid: number[][] = [];
    
    for (let i = 0; i <= resolution; i++) {
      x1Grid.push(x1Min + (i / resolution) * (x1Max - x1Min));
    }
    
    for (let j = 0; j <= resolution; j++) {
      x2Grid.push(x2Min + (j / resolution) * (x2Max - x2Min));
    }
    
    // Compute predictions for each grid point
    for (let j = 0; j <= resolution; j++) {
      const row: number[] = [];
      for (let i = 0; i <= resolution; i++) {
        const point = [[x1Grid[i], x2Grid[j]]];
        const pred = this.predict(point)[0];
        row.push(pred);
      }
      predGrid.push(row);
    }
    
    return { x: x1Grid, y: x2Grid, z: predGrid };
  }
  
  /**
   * Get visualization data for plotting
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained) {
      return [];
    }
    
    const visualizations: VisualizationData[] = [];
    
    // For 2D classification, create scatter plot with decision boundary
    if (this.X_train[0].length === 2) {
      const x1Values = this.X_train.map(row => row[0]);
      const x2Values = this.X_train.map(row => row[1]);
      
      // Create scatter plot for each class
      const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'];
      const symbols = ['circle', 'x', 'triangle-up', 'square', 'diamond'];
      
      for (const cls of this.classes) {
        const classIndices = this.y_train.map((y, i) => y === cls ? i : -1).filter(i => i !== -1);
        const classX1 = classIndices.map(i => x1Values[i]);
        const classX2 = classIndices.map(i => x2Values[i]);
        
        visualizations.push({
          type: 'scatter2d',
          data: {
            x: classX1,
            y: classX2,
            mode: 'markers',
            name: `Class ${cls}`,
            marker: { 
              color: colors[cls % colors.length], 
              size: 8, 
              symbol: symbols[cls % symbols.length] 
            }
          }
        });
      }
      
      // Add decision boundary
      const boundaryData = this.generateDecisionBoundary();
      if (boundaryData) {
        visualizations.push({
          type: 'heatmap',
          data: {
            x: boundaryData.x,
            y: boundaryData.y,
            z: boundaryData.z,
            colorscale: 'Viridis',
            showscale: true,
            opacity: 0.3,
            name: 'Decision Boundary'
          }
        });
      }
    }
    
    // OOB Score over trees
    if (this.bootstrap && this.trainingHistory.oobScore.length > 0) {
      visualizations.push({
        type: 'line',
        data: {
          x: this.trainingHistory.epochs,
          y: this.trainingHistory.oobScore,
          mode: 'lines',
          name: 'Out-of-Bag Score',
          line: { color: '#10b981', width: 2 }
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
    
    const predictions = this.predict(this.X_train);
    const n = this.y_train.length;
    
    // Multi-class confusion matrix
    const confusionMatrix: number[][] = [];
    for (let i = 0; i < this.classes.length; i++) {
      confusionMatrix.push(new Array(this.classes.length).fill(0));
    }
    
    for (let i = 0; i < n; i++) {
      const trueIdx = this.classes.indexOf(this.y_train[i]);
      const predIdx = this.classes.indexOf(predictions[i]);
      if (trueIdx !== -1 && predIdx !== -1) {
        confusionMatrix[trueIdx][predIdx]++;
      }
    }
    
    // Accuracy
    let correct = 0;
    for (let i = 0; i < n; i++) {
      if (this.y_train[i] === predictions[i]) {
        correct++;
      }
    }
    const accuracy = correct / n;
    
    // For binary classification, calculate additional metrics
    let precision = 0;
    let recall = 0;
    let f1Score = 0;
    
    if (this.classes.length === 2) {
      const tp = confusionMatrix[1][1];
      const fp = confusionMatrix[0][1];
      const fn = confusionMatrix[1][0];
      
      precision = tp + fp > 0 ? tp / (tp + fp) : 0;
      recall = tp + fn > 0 ? tp / (tp + fn) : 0;
      f1Score = precision + recall > 0 ? 
        2 * (precision * recall) / (precision + recall) : 0;
    }
    
    const metrics: Metrics = {
      accuracy,
      confusionMatrix
    };
    
    if (this.classes.length === 2) {
      metrics.precision = precision;
      metrics.recall = recall;
      metrics.f1Score = f1Score;
    }
    
    return metrics;
  }
  
  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.trees = [];
    this.X_train = [];
    this.y_train = [];
    this.classes = [];
    this.trainingHistory = {
      epochs: [],
      loss: [],
      oobScore: [],
      featureImportances: []
    };
  }
  
  /**
   * Get model parameters
   */
  getParameters() {
    return {
      nEstimators: this.nEstimators,
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      minSamplesLeaf: this.minSamplesLeaf,
      maxFeatures: this.maxFeatures,
      bootstrap: this.bootstrap,
      nTrees: this.trees.length,
      classes: [...this.classes]
    };
  }
}