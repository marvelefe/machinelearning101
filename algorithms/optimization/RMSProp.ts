/**
 * RMSProp Optimizer (Root Mean Square Propagation)
 * 
 * Implements RMSProp optimization algorithm that adapts learning rates
 * based on recent gradient magnitudes to handle different parameter scales.
 * 
 * Requirements: 9.1, 9.2, 9.3, 9.4
 */

import { BaseAlgorithm } from '~/types/algorithm';
import type { VisualizationData, Metrics, TrainingHistory } from '~/types/algorithm'; 


/**
 * Loss function type: takes parameters and returns loss value
 */
export type LossFunction = (params: number[]) => number;

/**
 * Gradient function type: takes parameters and returns gradient vector
 */
export type GradientFunction = (params: number[]) => number[];

/**
 * Optimization step result
 */
export interface OptimizationStep {
  iteration: number;
  params: number[];
  loss: number;
  gradient: number[];
  learningRate: number;
  squaredGradientAvg: number[];
  adaptiveLearningRates: number[];
}

/**
 * RMSProp Configuration
 */
export interface RMSPropConfig {
  learningRate: number;
  beta: number;
  epsilon: number;
  maxIterations: number;
  tolerance: number;
  lossFunction: string;
  startX: number;
  startY: number;
}

/**
 * RMSProp Optimizer Algorithm
 * 
 * RMSProp adapts learning rates based on recent gradient magnitudes:
 * E[g²] = βE[g²] + (1-β)g²
 * θ = θ - α * g / √(E[g²] + ε)
 * 
 * This implementation demonstrates:
 * - Adaptive learning rates per parameter
 * - Handling of different parameter scales
 * - Gradient magnitude normalization
 * - Superior performance on non-stationary objectives
 */
export class RMSProp extends BaseAlgorithm {
  name = 'RMSProp Optimizer';
  
  private config: RMSPropConfig;
  private params: number[] = [0, 0];
  private squaredGradientAvg: number[] = [0, 0];
  private history: OptimizationStep[] = [];
  private lossFunction: LossFunction;
  private gradientFunction: GradientFunction;
  
  constructor(config: Partial<RMSPropConfig> = {}) {
    super();
    
    this.config = {
      learningRate: 0.01,
      beta: 0.9,
      epsilon: 1e-8,
      maxIterations: 100,
      tolerance: 1e-6,
      lossFunction: 'quadratic-bowl',
      startX: -3,
      startY: 2,
      ...config
    };
    
    this.params = [this.config.startX, this.config.startY];
    this.squaredGradientAvg = [0, 0];
    this.setupLossFunction();
  }
  
  /**
   * Setup loss and gradient functions based on configuration
   */
  private setupLossFunction(): void {
    switch (this.config.lossFunction) {
      case 'quadratic-bowl':
        this.lossFunction = (params) => {
          const [x, y] = params;
          return x * x + y * y;
        };
        this.gradientFunction = (params) => {
          const [x, y] = params;
          return [2 * x, 2 * y];
        };
        break;
        
      case 'elongated-valley':
        this.lossFunction = (params) => {
          const [x, y] = params;
          return 10 * x * x + y * y;
        };
        this.gradientFunction = (params) => {
          const [x, y] = params;
          return [20 * x, 2 * y];
        };
        break;
        
      case 'rosenbrock':
        this.lossFunction = (params) => {
          const [x, y] = params;
          const a = 1, b = 100;
          return (a - x) * (a - x) + b * (y - x * x) * (y - x * x);
        };
        this.gradientFunction = (params) => {
          const [x, y] = params;
          const a = 1, b = 100;
          const dx = -2 * (a - x) - 4 * b * x * (y - x * x);
          const dy = 2 * b * (y - x * x);
          return [dx, dy];
        };
        break;
        
      case 'saddle-point':
        this.lossFunction = (params) => {
          const [x, y] = params;
          return x * x - y * y;
        };
        this.gradientFunction = (params) => {
          const [x, y] = params;
          return [2 * x, -2 * y];
        };
        break;
        
      case 'ackley':
        this.lossFunction = (params) => {
          const [x, y] = params;
          const a = 20, b = 0.2, c = 2 * Math.PI;
          const term1 = -a * Math.exp(-b * Math.sqrt(0.5 * (x * x + y * y)));
          const term2 = -Math.exp(0.5 * (Math.cos(c * x) + Math.cos(c * y)));
          return term1 + term2 + a + Math.E;
        };
        this.gradientFunction = (params) => {
          const [x, y] = params;
          const a = 20, b = 0.2, c = 2 * Math.PI;
          const norm = Math.sqrt(x * x + y * y);
          const exp1 = Math.exp(-b * norm);
          const exp2 = Math.exp(0.5 * (Math.cos(c * x) + Math.cos(c * y)));
          
          const dx = a * b * exp1 * x / norm + 0.5 * c * exp2 * Math.sin(c * x);
          const dy = a * b * exp1 * y / norm + 0.5 * c * exp2 * Math.sin(c * y);
          
          return [dx, dy];
        };
        break;
        
      default:
        // Default to quadratic bowl
        this.lossFunction = (params) => {
          const [x, y] = params;
          return x * x + y * y;
        };
        this.gradientFunction = (params) => {
          const [x, y] = params;
          return [2 * x, 2 * y];
        };
    }
  }
  
  /**
   * Train the RMSProp optimizer
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    this.reset();
    this.setupLossFunction();
    
    // Reset to starting position
    this.params = [this.config.startX, this.config.startY];
    this.squaredGradientAvg = [0, 0];
    this.history = [];
    
    let prevLoss = Infinity;
    
    for (let iteration = 0; iteration < this.config.maxIterations; iteration++) {
      // Calculate current loss and gradient
      const currentLoss = this.lossFunction(this.params);
      const gradient = this.gradientFunction(this.params);
      
      // Update squared gradient average: E[g²] = βE[g²] + (1-β)g²
      this.squaredGradientAvg = this.squaredGradientAvg.map((avg, i) => 
        this.config.beta * avg + (1 - this.config.beta) * gradient[i] * gradient[i]
      );
      
      // Calculate adaptive learning rates for each parameter
      const adaptiveLearningRates = this.squaredGradientAvg.map(avg => 
        this.config.learningRate / (Math.sqrt(avg) + this.config.epsilon)
      );
      
      // Record step before update
      this.history.push({
        iteration,
        params: [...this.params],
        loss: currentLoss,
        gradient: [...gradient],
        learningRate: this.config.learningRate,
        squaredGradientAvg: [...this.squaredGradientAvg],
        adaptiveLearningRates: [...adaptiveLearningRates]
      });
      
      // Check for convergence
      if (Math.abs(prevLoss - currentLoss) < this.config.tolerance) {
        break;
      }
      
      // RMSProp update: θ = θ - α * g / √(E[g²] + ε)
      this.params = this.params.map((param, i) => 
        param - adaptiveLearningRates[i] * gradient[i]
      );
      
      prevLoss = currentLoss;
    }
    
    this.trained = true;
  }
  
  /**
   * Predict using current parameters
   */
  predict(X: number[][]): number[] {
    return [...this.params];
  }
  
  /**
   * Get visualization data for the optimization process
   */
  getVisualizationData(): VisualizationData[] {
    const visualizations: VisualizationData[] = [];
    
    // 1. Loss landscape contour plot with optimization path
    const contourData = this.generateContourData();
    visualizations.push({
      type: 'contour',
      data: {
        ...contourData,
        path: this.history.map(step => ({
          x: step.params[0],
          y: step.params[1],
          loss: step.loss
        }))
      },
      layout: {
        title: 'RMSProp Optimization Path',
        xaxis: { title: 'Parameter X' },
        yaxis: { title: 'Parameter Y' }
      }
    });
    
    // 2. Loss over iterations
    visualizations.push({
      type: 'line',
      data: {
        x: this.history.map(step => step.iteration),
        y: this.history.map(step => step.loss),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Loss'
      },
      layout: {
        title: 'Loss Over Iterations',
        xaxis: { title: 'Iteration' },
        yaxis: { title: 'Loss' }
      }
    });
    
    // 3. Parameter evolution
    visualizations.push({
      type: 'line',
      data: [
        {
          x: this.history.map(step => step.iteration),
          y: this.history.map(step => step.params[0]),
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Parameter X'
        },
        {
          x: this.history.map(step => step.iteration),
          y: this.history.map(step => step.params[1]),
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Parameter Y'
        }
      ],
      layout: {
        title: 'Parameter Evolution',
        xaxis: { title: 'Iteration' },
        yaxis: { title: 'Parameter Value' }
      }
    });
    
    // 4. Adaptive learning rates (RMSProp-specific)
    visualizations.push({
      type: 'line',
      data: [
        {
          x: this.history.map(step => step.iteration),
          y: this.history.map(step => step.adaptiveLearningRates[0]),
          type: 'scatter',
          mode: 'lines',
          name: 'Adaptive LR X'
        },
        {
          x: this.history.map(step => step.iteration),
          y: this.history.map(step => step.adaptiveLearningRates[1]),
          type: 'scatter',
          mode: 'lines',
          name: 'Adaptive LR Y'
        }
      ],
      layout: {
        title: 'Adaptive Learning Rates',
        xaxis: { title: 'Iteration' },
        yaxis: { title: 'Learning Rate' }
      }
    });
    
    // 5. Squared gradient averages
    visualizations.push({
      type: 'line',
      data: [
        {
          x: this.history.map(step => step.iteration),
          y: this.history.map(step => Math.sqrt(step.squaredGradientAvg[0])),
          type: 'scatter',
          mode: 'lines',
          name: 'RMS Gradient X'
        },
        {
          x: this.history.map(step => step.iteration),
          y: this.history.map(step => Math.sqrt(step.squaredGradientAvg[1])),
          type: 'scatter',
          mode: 'lines',
          name: 'RMS Gradient Y'
        }
      ],
      layout: {
        title: 'RMS Gradient Evolution',
        xaxis: { title: 'Iteration' },
        yaxis: { title: 'RMS Gradient' }
      }
    });
    
    return visualizations;
  }
  
  /**
   * Generate contour data for loss landscape visualization
   */
  private generateContourData() {
    const xRange = [-5, 5];
    const yRange = [-5, 5];
    const resolution = 50;
    
    const x = [];
    const y = [];
    const z = [];
    
    for (let i = 0; i < resolution; i++) {
      const xVal = xRange[0] + (xRange[1] - xRange[0]) * i / (resolution - 1);
      x.push(xVal);
      
      const zRow = [];
      for (let j = 0; j < resolution; j++) {
        const yVal = yRange[0] + (yRange[1] - yRange[0]) * j / (resolution - 1);
        if (i === 0) y.push(yVal);
        
        const loss = this.lossFunction([xVal, yVal]);
        zRow.push(loss);
      }
      z.push(zRow);
    }
    
    return { x, y, z };
  }
  
  /**
   * Get optimization metrics
   */
  getMetrics(): Metrics {
    if (this.history.length === 0) {
      return {};
    }
    
    const finalStep = this.history[this.history.length - 1];
    const initialLoss = this.history[0].loss;
    const finalLoss = finalStep.loss;
    
    return {
      finalLoss,
      initialLoss,
      lossReduction: initialLoss - finalLoss,
      iterations: this.history.length,
      convergenceRate: (initialLoss - finalLoss) / this.history.length,
      finalParams: finalStep.params,
      gradientNorm: Math.sqrt(
        finalStep.gradient.reduce((sum, g) => sum + g * g, 0)
      ),
      finalAdaptiveLearningRates: finalStep.adaptiveLearningRates,
      finalSquaredGradientAvg: finalStep.squaredGradientAvg
    };
  }
  
  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<RMSPropConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.params = [this.config.startX, this.config.startY];
    this.setupLossFunction();
  }
  
  /**
   * Reset the optimizer
   */
  override reset(): void {
    super.reset();
    this.params = [this.config.startX, this.config.startY];
    this.squaredGradientAvg = [0, 0];
    this.history = [];
  }
}