/**
 * Stochastic Gradient Descent (SGD) Optimizer
 * 
 * Implements basic SGD optimization for educational purposes.
 * Shows how gradient descent works with different loss functions and learning rates.
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
}

/**
 * SGD Configuration
 */
export interface SGDConfig {
  learningRate: number;
  maxIterations: number;
  tolerance: number;
  lossFunction: string;
  startX: number;
  startY: number;
}

/**
 * Stochastic Gradient Descent Algorithm
 * 
 * Basic gradient descent: θ = θ - α∇L(θ)
 * 
 * This implementation demonstrates:
 * - Parameter updates using gradients
 * - Learning rate effects on convergence
 * - Different loss landscapes
 * - Optimization path visualization
 */
export class SGD extends BaseAlgorithm {
  name = 'Stochastic Gradient Descent';
  
  private config: SGDConfig;
  private params: number[] = [0, 0];
  private history: OptimizationStep[] = [];
  private lossFunction: LossFunction;
  private gradientFunction: GradientFunction;
  
  constructor(config: Partial<SGDConfig> = {}) {
    super();
    
    this.config = {
      learningRate: 0.1,
      maxIterations: 100,
      tolerance: 1e-6,
      lossFunction: 'quadratic-bowl',
      startX: -3,
      startY: 2,
      ...config
    };
    
    this.params = [this.config.startX, this.config.startY];
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
   * Train the SGD optimizer
   * For SGD, this means running the optimization process
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    this.reset();
    this.setupLossFunction();
    
    // Reset to starting position
    this.params = [this.config.startX, this.config.startY];
    this.history = [];
    
    let prevLoss = Infinity;
    
    for (let iteration = 0; iteration < this.config.maxIterations; iteration++) {
      // Calculate current loss and gradient
      const currentLoss = this.lossFunction(this.params);
      const gradient = this.gradientFunction(this.params);
      
      // Record step
      this.history.push({
        iteration,
        params: [...this.params],
        loss: currentLoss,
        gradient: [...gradient],
        learningRate: this.config.learningRate
      });
      
      // Check for convergence
      if (Math.abs(prevLoss - currentLoss) < this.config.tolerance) {
        break;
      }
      
      // SGD update: θ = θ - α∇L(θ)
      this.params = this.params.map((param, i) => 
        param - this.config.learningRate * gradient[i]
      );
      
      prevLoss = currentLoss;
    }
    
    this.trained = true;
  }
  
  /**
   * Predict using current parameters
   * For optimization, this returns the current parameter values
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
        title: 'SGD Optimization Path',
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
      )
    };
  }
  
  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<SGDConfig>): void {
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
    this.history = [];
  }
}