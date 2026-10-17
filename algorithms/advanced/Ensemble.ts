/**
 * Model Ensembling
 *
 * Implements various ensemble methods including bagging, boosting, and stacking
 * to combine multiple models for improved performance and robustness.
 *
 * Requirements: 10.1, 10.2, 10.3, 10.4
 */

import { BaseAlgorithm } from "~/types/algorithm";
import type { VisualizationData, Metrics } from "~/types/algorithm";

export interface EnsembleParams {
  ensembleType: "bagging" | "boosting" | "stacking" | "voting";
  numModels: number;
  baseModelType: "linear" | "tree" | "neural";
  votingType: "hard" | "soft";
  learningRate?: number; // for boosting
  maxDepth?: number; // for tree models
}

export interface BaseModel {
  id: string;
  type: string;
  weights: number[];
  bias: number;
  predictions: number[];
  weight: number; // ensemble weight
  error: number;
}

export interface EnsembleState {
  models: BaseModel[];
  finalPredictions: number[];
  modelWeights: number[];
  trainingErrors: number[];
  validationErrors: number[];
}

export class Ensemble extends BaseAlgorithm {
  name = "Model Ensembling";

  // Ensemble configuration
  private ensembleType: "bagging" | "boosting" | "stacking" | "voting";
  private numModels: number;
  private baseModelType: "linear" | "tree" | "neural";
  private votingType: "hard" | "soft";
  private learningRate: number;
  private maxDepth: number;

  // Training data
  private X_train: number[][] = [];
  private y_train: number[] = [];

  // Ensemble components
  private baseModels: BaseModel[] = [];
  private modelWeights: number[] = [];
  private sampleWeights: number[] = [];

  // Ensemble predictions
  private ensemblePredictions: number[] = [];
  private individualPredictions: number[][] = [];

  // Training history
  private trainingErrors: number[] = [];
  private validationErrors: number[] = [];

  constructor(
    params: EnsembleParams = {
      ensembleType: "bagging",
      numModels: 5,
      baseModelType: "linear",
      votingType: "soft",
      learningRate: 0.1,
      maxDepth: 3,
    }
  ) {
    super();
    this.ensembleType = params.ensembleType;
    this.numModels = params.numModels;
    this.baseModelType = params.baseModelType;
    this.votingType = params.votingType;
    this.learningRate = params.learningRate || 0.1;
    this.maxDepth = params.maxDepth || 3;
  }

  /**
   * Create a base model
   */
  private createBaseModel(id: string): BaseModel {
    const inputDim = this.X_train[0]?.length || 2;

    return {
      id,
      type: this.baseModelType,
      weights: Array.from(
        { length: inputDim },
        () => (Math.random() - 0.5) * 0.1
      ),
      bias: (Math.random() - 0.5) * 0.1,
      predictions: [],
      weight: 1.0,
      error: 0,
    };
  }

  /**
   * Train a linear base model
   */
  private trainLinearModel(
    model: BaseModel,
    X: number[][],
    y: number[],
    sampleWeights?: number[]
  ): void {
    const learningRate = 0.01;
    const epochs = 50;

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let i = 0; i < X.length; i++) {
        const input = X[i];
        const target = y[i];
        const weight = sampleWeights ? sampleWeights[i] : 1.0;

        // Forward pass
        let prediction = model.bias;
        for (let j = 0; j < input.length; j++) {
          prediction += model.weights[j] * input[j];
        }

        // Sigmoid activation for classification
        prediction = 1 / (1 + Math.exp(-prediction));

        // Compute error
        const error = (prediction - target) * weight;

        // Update weights
        model.bias -= learningRate * error;
        for (let j = 0; j < input.length; j++) {
          model.weights[j] -= learningRate * error * input[j];
        }
      }
    }
  }

  /**
   * Train a simple tree model (decision stump)
   */
  private trainTreeModel(
    model: BaseModel,
    X: number[][],
    y: number[],
    sampleWeights?: number[]
  ): void {
    let bestFeature = 0;
    let bestThreshold = 0;
    let bestError = Infinity;

    // Find best split
    for (let feature = 0; feature < X[0].length; feature++) {
      const values = X.map((x) => x[feature]).sort((a, b) => a - b);

      for (let i = 0; i < values.length - 1; i++) {
        const threshold = (values[i] + values[i + 1]) / 2;

        // Compute weighted error for this split
        let error = 0;
        for (let j = 0; j < X.length; j++) {
          const prediction = X[j][feature] > threshold ? 1 : 0;
          const weight = sampleWeights ? sampleWeights[j] : 1.0;
          if (prediction !== y[j]) {
            error += weight;
          }
        }

        if (error < bestError) {
          bestError = error;
          bestFeature = feature;
          bestThreshold = threshold;
        }
      }
    }

    // Store decision rule in weights (simplified representation)
    model.weights = Array(X[0].length).fill(0);
    model.weights[bestFeature] = 1;
    model.bias = -bestThreshold;
    model.error = bestError;
  }

  /**
   * Make prediction with a base model
   */
  private predictWithModel(model: BaseModel, X: number[][]): number[] {
    const predictions: number[] = [];

    for (const input of X) {
      if (model.type === "linear" || model.type === "neural") {
        // Linear/neural model
        let prediction = model.bias;
        for (let j = 0; j < input.length; j++) {
          prediction += model.weights[j] * input[j];
        }

        // Sigmoid activation
        prediction = 1 / (1 + Math.exp(-prediction));
        predictions.push(prediction);
      } else if (model.type === "tree") {
        // Tree model (decision stump)
        const featureIndex = model.weights.findIndex((w) => w !== 0);
        const threshold = -model.bias;
        const prediction = input[featureIndex] > threshold ? 1 : 0;
        predictions.push(prediction);
      }
    }

    return predictions;
  }

  /**
   * Bootstrap sampling for bagging
   */
  private bootstrapSample(
    X: number[][],
    y: number[]
  ): { X: number[][]; y: number[] } {
    const n = X.length;
    const sampledX: number[][] = [];
    const sampledY: number[] = [];

    for (let i = 0; i < n; i++) {
      const randomIndex = Math.floor(Math.random() * n);
      sampledX.push([...X[randomIndex]]);
      sampledY.push(y[randomIndex]);
    }

    return { X: sampledX, y: sampledY };
  }

  /**
   * Update sample weights for boosting
   */
  private updateSampleWeights(
    predictions: number[],
    y: number[],
    modelWeight: number
  ): void {
    const n = this.sampleWeights.length;
    let totalWeight = 0;

    for (let i = 0; i < n; i++) {
      const error = predictions[i] !== y[i] ? 1 : 0;
      this.sampleWeights[i] *= Math.exp(modelWeight * error);
      totalWeight += this.sampleWeights[i];
    }

    // Normalize weights
    for (let i = 0; i < n; i++) {
      this.sampleWeights[i] /= totalWeight;
    }
  }

  /**
   * Compute model weight for boosting
   */
  private computeModelWeight(predictions: number[], y: number[]): number {
    let error = 0;
    let total = 0;

    for (let i = 0; i < predictions.length; i++) {
      const weight = this.sampleWeights[i];
      if (predictions[i] !== y[i]) {
        error += weight;
      }
      total += weight;
    }

    const errorRate = error / total;

    // Avoid division by zero
    if (errorRate >= 0.5) return 0;
    if (errorRate <= 0) return 10;

    return 0.5 * Math.log((1 - errorRate) / errorRate);
  }

  /**
   * Train ensemble using bagging
   */
  private async trainBagging(): Promise<void> {
    for (let i = 0; i < this.numModels; i++) {
      const model = this.createBaseModel(`bagging_${i}`);

      // Bootstrap sample
      const { X, y } = this.bootstrapSample(this.X_train, this.y_train);

      // Train base model
      if (this.baseModelType === "linear" || this.baseModelType === "neural") {
        this.trainLinearModel(model, X, y);
      } else {
        this.trainTreeModel(model, X, y);
      }

      // Store predictions
      model.predictions = this.predictWithModel(model, this.X_train);
      this.baseModels.push(model);

      // Allow UI updates
      if (i % 2 === 0) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }

    // Equal weights for bagging
    this.modelWeights = Array(this.numModels).fill(1.0 / this.numModels);
  }

  /**
   * Train ensemble using boosting (AdaBoost)
   */
  private async trainBoosting(): Promise<void> {
    // Initialize sample weights uniformly
    const n = this.X_train.length;
    this.sampleWeights = Array(n).fill(1.0 / n);

    for (let i = 0; i < this.numModels; i++) {
      const model = this.createBaseModel(`boosting_${i}`);

      // Train base model with weighted samples
      if (this.baseModelType === "tree") {
        this.trainTreeModel(
          model,
          this.X_train,
          this.y_train,
          this.sampleWeights
        );
      } else {
        this.trainLinearModel(
          model,
          this.X_train,
          this.y_train,
          this.sampleWeights
        );
      }

      // Get predictions
      const predictions = this.predictWithModel(model, this.X_train);

      // Convert soft predictions to hard predictions for boosting
      const hardPredictions = predictions.map((p) => (p > 0.5 ? 1 : 0));

      // Compute model weight
      const modelWeight = this.computeModelWeight(
        hardPredictions,
        this.y_train
      );

      if (modelWeight <= 0) break; // Stop if model is not better than random

      model.weight = modelWeight;
      model.predictions = predictions;
      this.baseModels.push(model);
      this.modelWeights.push(modelWeight);

      // Update sample weights
      this.updateSampleWeights(hardPredictions, this.y_train, modelWeight);

      // Allow UI updates
      if (i % 2 === 0) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }

    // Normalize model weights
    const totalWeight = this.modelWeights.reduce((a, b) => a + b, 0);
    if (totalWeight > 0) {
      this.modelWeights = this.modelWeights.map((w) => w / totalWeight);
    }
  }

  /**
   * Train ensemble using voting
   */
  private async trainVoting(): Promise<void> {
    for (let i = 0; i < this.numModels; i++) {
      const model = this.createBaseModel(`voting_${i}`);

      // Add some diversity by using different random seeds or slight variations
      if (this.baseModelType === "linear" || this.baseModelType === "neural") {
        this.trainLinearModel(model, this.X_train, this.y_train);
      } else {
        this.trainTreeModel(model, this.X_train, this.y_train);
      }

      model.predictions = this.predictWithModel(model, this.X_train);
      this.baseModels.push(model);

      // Allow UI updates
      if (i % 2 === 0) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }

    // Equal weights for voting
    this.modelWeights = Array(this.numModels).fill(1.0 / this.numModels);
  }

  /**
   * Combine predictions from base models
   */
  private combinePredictions(predictions: number[][]): number[] {
    const n = predictions[0].length;
    const combined: number[] = [];

    for (let i = 0; i < n; i++) {
      if (this.votingType === "hard") {
        // Hard voting: majority vote
        let votes = 0;
        for (let j = 0; j < predictions.length; j++) {
          const hardPred = predictions[j][i] > 0.5 ? 1 : 0;
          votes += hardPred * this.modelWeights[j];
        }
        combined.push(votes > 0.5 ? 1 : 0);
      } else {
        // Soft voting: weighted average
        let weightedSum = 0;
        for (let j = 0; j < predictions.length; j++) {
          weightedSum += predictions[j][i] * this.modelWeights[j];
        }
        combined.push(weightedSum);
      }
    }

    return combined;
  }

  /**
   * Train the ensemble
   */
  async fit(X: number[][], y: number[]): Promise<void> {
    // Validate input
    this.validateInput(X);
    this.validateTarget(X, y);

    // Store training data
    this.X_train = X.map((row) => [...row]);
    this.y_train = [...y];

    // Initialize
    this.baseModels = [];
    this.modelWeights = [];
    this.sampleWeights = [];

    // Initialize training history
    this.trainingHistory = {
      epochs: [],
      loss: [],
    };

    // Train ensemble based on type
    switch (this.ensembleType) {
      case "bagging":
        await this.trainBagging();
        break;
      case "boosting":
        await this.trainBoosting();
        break;
      case "voting":
      case "stacking":
        await this.trainVoting();
        break;
    }

    // Compute ensemble predictions
    const allPredictions = this.baseModels.map((model) => model.predictions);
    this.ensemblePredictions = this.combinePredictions(allPredictions);
    this.individualPredictions = allPredictions;

    // Compute training history
    for (let i = 0; i < this.baseModels.length; i++) {
      this.trainingHistory.epochs.push(i);

      // Compute ensemble error up to this point
      const currentPredictions = allPredictions.slice(0, i + 1);
      const currentWeights = this.modelWeights.slice(0, i + 1);

      // Normalize current weights
      const totalWeight = currentWeights.reduce((a, b) => a + b, 0);
      const normalizedWeights = currentWeights.map((w) => w / totalWeight);

      // Compute ensemble prediction
      const ensemblePred: number[] = [];
      for (let j = 0; j < this.X_train.length; j++) {
        let weightedSum = 0;
        for (let k = 0; k < currentPredictions.length; k++) {
          weightedSum += currentPredictions[k][j] * normalizedWeights[k];
        }
        ensemblePred.push(weightedSum);
      }

      // Compute error
      let error = 0;
      for (let j = 0; j < this.y_train.length; j++) {
        const pred = ensemblePred[j] > 0.5 ? 1 : 0;
        if (pred !== this.y_train[j]) error++;
      }

      this.trainingHistory.loss.push(error / this.y_train.length);
    }

    this.trained = true;
  }

  /**
   * Make ensemble predictions
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    this.validateInput(X, this.X_train[0].length);

    // Get predictions from all base models
    const allPredictions: number[][] = [];
    for (const model of this.baseModels) {
      const predictions = this.predictWithModel(model, X);
      allPredictions.push(predictions);
    }

    // Combine predictions
    return this.combinePredictions(allPredictions);
  }

  /**
   * Get individual model predictions
   */
  getIndividualPredictions(X: number[][]): number[][] {
    this.ensureTrained();
    this.validateInput(X, this.X_train[0].length);

    const predictions: number[][] = [];
    for (const model of this.baseModels) {
      predictions.push(this.predictWithModel(model, X));
    }

    return predictions;
  }

  /**
   * Get visualization data
   */
  getVisualizationData(): VisualizationData[] {
    if (!this.trained) {
      return [];
    }

    const visualizations: VisualizationData[] = [];

    // Ensemble error over models
    visualizations.push({
      type: "line",
      data: {
        ensemble_error: {
          x: this.trainingHistory.epochs,
          y: this.trainingHistory.loss,
          mode: "lines+markers",
          name: "Ensemble Error",
          line: { color: "#ef4444", width: 2 },
        },
      },
    });

    // Model weights
    visualizations.push({
      type: "bar",
      data: {
        model_weights: {
          x: this.baseModels.map((m) => m.id),
          y: this.modelWeights,
          name: "Model Weights",
          marker: { color: "#3b82f6" },
        },
      },
    });

    // Individual vs ensemble predictions (first 50 samples)
    const numSamples = Math.min(50, this.X_train.length);
    const sampleIndices = Array.from({ length: numSamples }, (_, i) => i);

    const plotData: any = {
      ensemble: {
        x: sampleIndices,
        y: this.ensemblePredictions.slice(0, numSamples),
        mode: "lines+markers",
        name: "Ensemble",
        line: { color: "#10b981", width: 3 },
      },
    };

    // Add individual model predictions
    for (let i = 0; i < Math.min(3, this.baseModels.length); i++) {
      plotData[`model_${i}`] = {
        x: sampleIndices,
        y: this.individualPredictions[i].slice(0, numSamples),
        mode: "lines",
        name: `Model ${i + 1}`,
        line: { width: 1, dash: "dot" },
        opacity: 0.7,
      };
    }

    visualizations.push({
      type: "line",
      data: plotData,
    });

    // Model diversity (correlation matrix)
    if (this.baseModels.length > 1) {
      const correlations = this.computeModelCorrelations();

      visualizations.push({
        type: "heatmap",
        data: {
          correlations: {
            z: correlations,
            colorscale: "RdBu",
            name: "Model Correlations",
          },
        },
      });
    }

    return visualizations;
  }

  /**
   * Compute correlation matrix between models
   */
  private computeModelCorrelations(): number[][] {
    const n = this.baseModels.length;
    const correlations: number[][] = [];

    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j < n; j++) {
        if (i === j) {
          row.push(1.0);
        } else {
          const corr = this.computeCorrelation(
            this.individualPredictions[i],
            this.individualPredictions[j]
          );
          row.push(corr);
        }
      }
      correlations.push(row);
    }

    return correlations;
  }

  /**
   * Compute correlation between two arrays
   */
  private computeCorrelation(x: number[], y: number[]): number {
    const n = x.length;
    const meanX = x.reduce((a, b) => a + b, 0) / n;
    const meanY = y.reduce((a, b) => a + b, 0) / n;

    let numerator = 0;
    let denomX = 0;
    let denomY = 0;

    for (let i = 0; i < n; i++) {
      const dx = x[i] - meanX;
      const dy = y[i] - meanY;
      numerator += dx * dy;
      denomX += dx * dx;
      denomY += dy * dy;
    }

    const denominator = Math.sqrt(denomX * denomY);
    return denominator === 0 ? 0 : numerator / denominator;
  }

  /**
   * Get metrics
   */
  getMetrics(): Metrics {
    if (!this.trained) {
      return {};
    }

    // Compute ensemble accuracy
    let correct = 0;
    for (let i = 0; i < this.y_train.length; i++) {
      const pred = this.ensemblePredictions[i] > 0.5 ? 1 : 0;
      if (pred === this.y_train[i]) correct++;
    }
    const ensembleAccuracy = correct / this.y_train.length;

    // Compute individual model accuracies
    const individualAccuracies: number[] = [];
    for (let j = 0; j < this.baseModels.length; j++) {
      let modelCorrect = 0;
      for (let i = 0; i < this.y_train.length; i++) {
        const pred = this.individualPredictions[j][i] > 0.5 ? 1 : 0;
        if (pred === this.y_train[i]) modelCorrect++;
      }
      individualAccuracies.push(modelCorrect / this.y_train.length);
    }

    const avgIndividualAccuracy =
      individualAccuracies.reduce((a, b) => a + b, 0) /
      individualAccuracies.length;
    const bestIndividualAccuracy = Math.max(...individualAccuracies);

    // Compute model diversity
    const avgCorrelation = this.computeAverageCorrelation();

    return {
      ensembleAccuracy,
      avgIndividualAccuracy,
      bestIndividualAccuracy,
      improvementOverBest: ensembleAccuracy - bestIndividualAccuracy,
      improvementOverAverage: ensembleAccuracy - avgIndividualAccuracy,
      numModels: this.baseModels.length,
      ensembleType: this.ensembleType,
      avgModelCorrelation: avgCorrelation,
      modelDiversity: 1 - avgCorrelation,
    };
  }

  /**
   * Compute average correlation between models
   */
  private computeAverageCorrelation(): number {
    if (this.baseModels.length < 2) return 0;

    let totalCorr = 0;
    let count = 0;

    for (let i = 0; i < this.baseModels.length; i++) {
      for (let j = i + 1; j < this.baseModels.length; j++) {
        const corr = this.computeCorrelation(
          this.individualPredictions[i],
          this.individualPredictions[j]
        );
        totalCorr += Math.abs(corr);
        count++;
      }
    }

    return count > 0 ? totalCorr / count : 0;
  }

  /**
   * Get model weights
   */
  getModelWeights(): number[] {
    return [...this.modelWeights];
  }

  /**
   * Get ensemble state
   */
  getEnsembleState(): EnsembleState {
    return {
      models: this.baseModels.map((model) => ({
        id: model.id,
        type: model.type,
        weights: [...model.weights],
        bias: model.bias,
        predictions: [...model.predictions],
        weight: model.weight,
        error: model.error,
      })),
      finalPredictions: [...this.ensemblePredictions],
      modelWeights: [...this.modelWeights],
      trainingErrors: [...this.trainingHistory.loss],
      validationErrors: [],
    };
  }

  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.X_train = [];
    this.y_train = [];
    this.baseModels = [];
    this.modelWeights = [];
    this.sampleWeights = [];
    this.ensemblePredictions = [];
    this.individualPredictions = [];
    this.trainingErrors = [];
    this.validationErrors = [];
  }

  /**
   * Get model parameters
   */
  getParameters() {
    return {
      ensembleType: this.ensembleType,
      numModels: this.numModels,
      baseModelType: this.baseModelType,
      votingType: this.votingType,
      learningRate: this.learningRate,
      maxDepth: this.maxDepth,
    };
  }
}
