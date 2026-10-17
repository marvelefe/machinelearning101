/**
 * Simple Feedforward Neural Network
 *
 * Implements a basic feedforward neural network with backpropagation.
 * Designed for educational purposes with visualization support.
 *
 * Requirements: 6.3, 6.4, 7.1, 7.4
 */

import { BaseAlgorithm } from "~/types/algorithm";
import type {
  VisualizationData,
  Metrics,
  TrainingHistory,
} from "~/types/algorithm";
import { mean } from "~/utils/math";

export interface NeuralNetworkParams {
  hiddenLayers: number[];
  learningRate: number;
  epochs: number;
  activation: "sigmoid" | "relu" | "tanh";
}

export interface LayerActivations {
  layer: number;
  activations: number[];
  preActivations: number[];
}

export interface NetworkState {
  weights: number[][][];
  biases: number[][];
  activations: LayerActivations[];
  gradients: number[][][];
}

export class NeuralNetwork extends BaseAlgorithm {
  name = "Neural Network";

  // Network architecture
  private layers: number[] = []; // [input, hidden1, hidden2, ..., output]
  private weights: number[][][] = []; // weights[layer][neuron][input]
  private biases: number[][] = []; // biases[layer][neuron]

  // Training data
  private X_train: number[][] = [];
  private y_train: number[] = [];

  // Visualization data
  private currentActivations: LayerActivations[] = [];
  private currentGradients: number[][][] = [];

  // Hyperparameters
  private learningRate: number;
  private epochs: number;
  private activation: "sigmoid" | "relu" | "tanh";

  constructor(
    params: NeuralNetworkParams = {
      hiddenLayers: [4, 3],
      learningRate: 0.1,
      epochs: 100,
      activation: "sigmoid",
    }
  ) {
    super();
    this.learningRate = params.learningRate;
    this.epochs = params.epochs;
    this.activation = params.activation;
    this.layers = params.hiddenLayers;
  }

  /**
   * Activation functions
   */
  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
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

  /**
   * Apply activation function
   */
  private activate(x: number): number {
    switch (this.activation) {
      case "sigmoid":
        return this.sigmoid(x);
      case "relu":
        return this.relu(x);
      case "tanh":
        return this.tanh(x);
      default:
        return this.sigmoid(x);
    }
  }

  /**
   * Apply activation derivative
   */
  private activateDerivative(x: number): number {
    switch (this.activation) {
      case "sigmoid":
        return this.sigmoidDerivative(x);
      case "relu":
        return this.reluDerivative(x);
      case "tanh":
        return this.tanhDerivative(x);
      default:
        return this.sigmoidDerivative(x);
    }
  }

  /**
   * Initialize network weights and biases
   */
  private initializeNetwork(inputSize: number, outputSize: number): void {
    // Build layer sizes: [input, hidden1, hidden2, ..., output]
    this.layers = [inputSize, ...this.layers, outputSize];

    // Initialize weights and biases
    this.weights = [];
    this.biases = [];

    for (let l = 1; l < this.layers.length; l++) {
      const layerWeights: number[][] = [];
      const layerBiases: number[] = [];

      const prevSize = this.layers[l - 1];
      const currSize = this.layers[l];

      // Xavier/He initialization
      const scale = Math.sqrt(2.0 / prevSize);

      for (let j = 0; j < currSize; j++) {
        const neuronWeights: number[] = [];
        for (let i = 0; i < prevSize; i++) {
          neuronWeights.push((Math.random() - 0.5) * 2 * scale);
        }
        layerWeights.push(neuronWeights);
        layerBiases.push(0);
      }

      this.weights.push(layerWeights);
      this.biases.push(layerBiases);
    }
  }

  /**
   * Forward pass through the network
   */
  private forward(input: number[]): {
    activations: number[][];
    preActivations: number[][];
  } {
    const activations: number[][] = [input];
    const preActivations: number[][] = [input];

    let currentActivation = input;

    for (let l = 0; l < this.weights.length; l++) {
      const nextPreActivation: number[] = [];
      const nextActivation: number[] = [];

      for (let j = 0; j < this.weights[l].length; j++) {
        let z = this.biases[l][j];

        for (let i = 0; i < currentActivation.length; i++) {
          z += this.weights[l][j][i] * currentActivation[i];
        }

        nextPreActivation.push(z);

        // Use sigmoid for output layer, specified activation for hidden layers
        const a =
          l === this.weights.length - 1
            ? this.sigmoid(z) // Output layer always sigmoid for binary classification
            : this.activate(z);

        nextActivation.push(a);
      }

      preActivations.push(nextPreActivation);
      activations.push(nextActivation);
      currentActivation = nextActivation;
    }

    return { activations, preActivations };
  }

  /**
   * Backward pass (backpropagation)
   */
  private backward(
    input: number[],
    target: number,
    activations: number[][],
    preActivations: number[][]
  ): {
    weightGradients: number[][][];
    biasGradients: number[][];
  } {
    const numLayers = this.weights.length;
    const weightGradients: number[][][] = [];
    const biasGradients: number[][] = [];

    // Initialize gradients
    for (let l = 0; l < numLayers; l++) {
      weightGradients.push(
        this.weights[l].map((neuron) => neuron.map(() => 0))
      );
      biasGradients.push(this.biases[l].map(() => 0));
    }

    // Output layer error (using binary cross-entropy derivative)
    const outputLayer = numLayers;
    const output = activations[outputLayer][0];
    let delta = [output - target]; // For sigmoid + BCE

    // Backpropagate through layers
    for (let l = numLayers - 1; l >= 0; l--) {
      const currentActivations = activations[l];
      const nextDelta: number[] = [];

      // Compute gradients for this layer
      for (let j = 0; j < this.weights[l].length; j++) {
        biasGradients[l][j] = delta[j];

        for (let i = 0; i < this.weights[l][j].length; i++) {
          weightGradients[l][j][i] = delta[j] * currentActivations[i];
        }
      }

      // Propagate error to previous layer
      if (l > 0) {
        for (let i = 0; i < currentActivations.length; i++) {
          let error = 0;
          for (let j = 0; j < this.weights[l].length; j++) {
            error += delta[j] * this.weights[l][j][i];
          }
          // Apply activation derivative
          const derivative = this.activateDerivative(preActivations[l][i]);
          nextDelta.push(error * derivative);
        }
        delta = nextDelta;
      }
    }

    // Store for visualization
    this.currentGradients = weightGradients;

    return { weightGradients, biasGradients };
  }

  /**
   * Train the network
   */
  async fit(X: number[][], y: number[]): Promise<void> {
    // Validate input
    this.validateInput(X);
    this.validateTarget(X, y);

    // Store training data
    this.X_train = X.map((row) => [...row]);
    this.y_train = [...y];

    // Initialize network
    const inputSize = X[0].length;
    const outputSize = 1; // Binary classification
    this.initializeNetwork(inputSize, outputSize);

    // Initialize training history
    this.trainingHistory = {
      epochs: [],
      loss: [],
      accuracy: [],
    };

    // Training loop
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      let totalLoss = 0;
      let correct = 0;

      // Mini-batch gradient descent (batch size = 1, i.e., SGD)
      for (let i = 0; i < X.length; i++) {
        const input = X[i];
        const target = y[i];

        // Forward pass
        const { activations, preActivations } = this.forward(input);
        const output = activations[activations.length - 1][0];

        // Compute loss (binary cross-entropy)
        const loss = -(
          target * Math.log(output + 1e-10) +
          (1 - target) * Math.log(1 - output + 1e-10)
        );
        totalLoss += loss;

        // Compute accuracy
        const prediction = output >= 0.5 ? 1 : 0;
        if (prediction === target) correct++;

        // Backward pass
        const { weightGradients, biasGradients } = this.backward(
          input,
          target,
          activations,
          preActivations
        );

        // Update weights and biases
        for (let l = 0; l < this.weights.length; l++) {
          for (let j = 0; j < this.weights[l].length; j++) {
            this.biases[l][j] -= this.learningRate * biasGradients[l][j];

            for (let k = 0; k < this.weights[l][j].length; k++) {
              this.weights[l][j][k] -=
                this.learningRate * weightGradients[l][j][k];
            }
          }
        }
      }

      // Record history
      const avgLoss = totalLoss / X.length;
      const accuracy = correct / X.length;

      this.trainingHistory.epochs.push(epoch);
      this.trainingHistory.loss.push(avgLoss);
      this.trainingHistory.accuracy!.push(accuracy);

      // Allow UI to update periodically
      if (epoch % 10 === 0) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }

    // Store final activations for visualization
    if (X.length > 0) {
      const { activations, preActivations } = this.forward(X[0]);
      this.currentActivations = activations.map((acts, layer) => ({
        layer,
        activations: acts,
        preActivations: preActivations[layer],
      }));
    }

    this.trained = true;
  }

  /**
   * Make predictions
   */
  predict(X: number[][]): number[] {
    this.ensureTrained();
    this.validateInput(X, this.layers[0]);

    return X.map((input) => {
      const { activations } = this.forward(input);
      const output = activations[activations.length - 1][0];
      return output >= 0.5 ? 1 : 0;
    });
  }

  /**
   * Get probability predictions
   */
  predictProba(X: number[][]): number[] {
    this.ensureTrained();
    this.validateInput(X, this.layers[0]);

    return X.map((input) => {
      const { activations } = this.forward(input);
      return activations[activations.length - 1][0];
    });
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
      type: "line",
      data: {
        loss: {
          x: this.trainingHistory.epochs,
          y: this.trainingHistory.loss,
          mode: "lines",
          name: "Loss",
          line: { color: "#ef4444", width: 2 },
        },
      },
    });

    // Accuracy curve
    if (this.trainingHistory.accuracy) {
      visualizations.push({
        type: "line",
        data: {
          accuracy: {
            x: this.trainingHistory.epochs,
            y: this.trainingHistory.accuracy,
            mode: "lines",
            name: "Accuracy",
            line: { color: "#10b981", width: 2 },
          },
        },
      });
    }

    // Network architecture visualization
    visualizations.push({
      type: "network",
      data: {
        layers: this.layers,
        weights: this.weights,
        biases: this.biases,
        activations: this.currentActivations,
        gradients: this.currentGradients,
      },
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

    // Compute metrics on training data
    const predictions = this.predict(this.X_train);

    let correct = 0;
    let tp = 0,
      fp = 0,
      tn = 0,
      fn = 0;

    for (let i = 0; i < this.y_train.length; i++) {
      const pred = predictions[i];
      const actual = this.y_train[i];

      if (pred === actual) correct++;

      if (pred === 1 && actual === 1) tp++;
      if (pred === 1 && actual === 0) fp++;
      if (pred === 0 && actual === 0) tn++;
      if (pred === 0 && actual === 1) fn++;
    }

    const accuracy = correct / this.y_train.length;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1Score =
      precision + recall > 0
        ? (2 * (precision * recall)) / (precision + recall)
        : 0;

    const finalLoss =
      this.trainingHistory.loss[this.trainingHistory.loss.length - 1];

    return {
      accuracy,
      precision,
      recall,
      f1Score,
      finalLoss,
      confusionMatrix: [
        [tn, fp],
        [fn, tp],
      ],
      architecture: this.layers.join("-"),
      totalParameters: this.getTotalParameters(),
    };
  }

  /**
   * Get total number of parameters
   */
  private getTotalParameters(): number {
    let total = 0;

    for (let l = 0; l < this.weights.length; l++) {
      for (let j = 0; j < this.weights[l].length; j++) {
        total += this.weights[l][j].length; // weights
        total += 1; // bias
      }
    }

    return total;
  }

  /**
   * Get network state for visualization
   */
  getNetworkState(): NetworkState {
    return {
      weights: this.weights.map((layer) => layer.map((neuron) => [...neuron])),
      biases: this.biases.map((layer) => [...layer]),
      activations: this.currentActivations.map((la) => ({
        layer: la.layer,
        activations: [...la.activations],
        preActivations: [...la.preActivations],
      })),
      gradients: this.currentGradients.map((layer) =>
        layer.map((neuron) => [...neuron])
      ),
    };
  }

  /**
   * Forward pass for a single input (for visualization)
   */
  forwardSingle(input: number[]): LayerActivations[] {
    const { activations, preActivations } = this.forward(input);

    return activations.map((acts, layer) => ({
      layer,
      activations: acts,
      preActivations: preActivations[layer],
    }));
  }

  /**
   * Reset the model
   */
  override reset(): void {
    super.reset();
    this.weights = [];
    this.biases = [];
    this.X_train = [];
    this.y_train = [];
    this.currentActivations = [];
    this.currentGradients = [];
  }

  /**
   * Get model parameters
   */
  getParameters() {
    return {
      layers: [...this.layers],
      weights: this.weights.map((layer) => layer.map((neuron) => [...neuron])),
      biases: this.biases.map((layer) => [...layer]),
      learningRate: this.learningRate,
      epochs: this.epochs,
      activation: this.activation,
    };
  }
}
