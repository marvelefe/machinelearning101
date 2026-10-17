/**
 * Adam Optimizer (Adaptive Moment Estimation)
 * 
 * Implements the Adam optimization algorithm that combines momentum
 * with adaptive learning rates for each parameter.
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
  firstMoment: number[];
  secondMoment: number[];
}

/**
 * Adam Configuration
 */
export interface AdamConfig {
  learningRate: number;
  beta1: number;
  beta2: number;
  epsilon: number;
  maxIterations: number;
  tolerance: number;
  lossFunction: string;
  startX: number;
  startY: number;
}

/**
 * Adam Optimizer Algorithm
 * 
 * Adam combines momentum and RMSProp:
 * m = β₁m + (1-β₁)g
 * v = β₂v + (1-β₂)g²
 * m̂ = m / (1-β₁ᵗ)
 * v̂ = v / (1-β₂ᵗ)
 * θ = θ - α * m̂ / (√v̂ + ε)
 * 
 * This implementation demonstrates:
 * - Adaptive learning rates per parameter
 * - Momentum for acceleration
 * - Bias correction for initialization
 * - Superior convergence on various loss landscapes
 */
export class Adam extends BaseAlgorithm {
  name = 'Adam Optimizer';
  
  private config: AdamConfig;
  private params: number[] = [0, 0];
  private firstMoment: number[] = [0, 0];
  private secondMoment: number[] = [0, 0];
  private history: OptimizationStep[] = [];
  private lossFunction: LossFunction;
  private gradientFunction: GradientFunction;
  private iteration: number = 0;
  
  constructor(config: Partial<AdamConfig> = {}) {
    super();
    
    this.config = {
      learningRate: 0.01,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
      maxIterations: 100,
      tolerance: 1e-6,
      lossFunction: 'quadratic-bowl',
      startX: -3,
      startY: 2,
      ...config
    };
    
    this.params = [this.config.startX, this.config.startY];
    this.firstMoment = [0, 0];
    this.secondMoment = [0, 0];
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
        
      case 'beale':
        this.lossFunction = (params) => {
          const [x, y] = params;
          const term1 = (1.5 - x + x * y) ** 2;
          const term2 = (2.25 - x + x * y * y) ** 2;
          const term3 = (2.625 - x + x * y * y * y) ** 2;
          return term1 + term2 + term3;
        };
        this.gradientFunction = (params) => {
          const [x, y] = params;
          const dx = 2 * (1.5 - x + x * y) * (-1 + y) +
                    2 * (2.25 - x + x * y * y) * (-1 + y * y) +
                    2 * (2.625 - x + x * y * y * y) * (-1 + y * y * y);
          const dy = 2 * (1.5 - x + x * y) * x +
                    2 * (2.25 - x + x * y * y) * (2 * x * y) +
                    2 * (2.625 - x + x * y * y * y) * (3 * x * y * y);
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
   * Train the Adam optimizer
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    this.reset();
    this.setupLossFunction();
    
    // Reset to starting position
    this.params = [this.config.startX, this.config.startY];
    this.firstMoment = [0, 0];
    this.secondMoment = [0, 0];
    this.history = [];
    this.iteration = 0;
    
    let prevLoss = Infinity;
    
    for (let iter = 0; iter < this.config.maxIterations; iter++) {
      this.iteration = iter + 1;
      
      // Calculate current loss and gradient
      const currentLoss = this.lossFunction(this.params);
      const gradient = this.gradientFunction(this.params);
      
      // Update biased first moment: m = β₁m + (1-β₁)g
      this.firstMoment = this.firstMoment.map((m, i) => 
        this.config.beta1 * m + (1 - this.config.beta1) * gradient[i]
      );
      
      // Update biased second moment: v = β₂v + (1-β₂)g²
      this.secondMoment = this.secondMoment.map((v, i) => 
        this.config.beta2 * v + (1 - this.config.beta2) * gradient[i] * gradient[i]
      );
      
      // Compute bias-corrected moments
      const firstMomentCorrected = this.firstMoment.map(m => 
        m / (1 - Math.pow(this.config.beta1, this.iteration))
      );
      
      const secondMomentCorrected = this.secondMoment.map(v => 
        v / (1 - Math.pow(this.config.beta2, this.iteration))
      );
      
      // Record step before update
      this.history.push({
        iteration: iter,
        params: [...this.params],
        loss: currentLoss,
        gradient: [...gradient],
        learningRate: this.config.learningRate,
        firstMoment: [...this.firstMoment],
        secondMoment: [...this.secondMoment]
      });
      
      // Check for convergence
      if (Math.abs(prevLoss - currentLoss) < this.config.tolerance) {
        break;
      }
      
      // Adam update: θ = θ - α * m̂ / (√v̂ + ε)
      this.params = this.params.map((param, i) => 
        param - this.config.learningRate * firstMomentCorrected[i] / 
        (Math.sqrt(secondMomentCorrected[i]) + this.config.epsilon)
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
        title: 'Adam Optimization Path',
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
    
    // 4. Moment evolution (Adam-specific)
    visualizations.push({
      type: 'line',
      data: [
        {
          x: this.history.map(step => step.iteration),
          y: this.history.map(step => step.firstMoment[0]),
          type: 'scatter',
          mode: 'lines',
          name: 'First Moment X'
        },
        {
          x: this.history.map(step => step.iteration),
          y: this.history.map(step => step.firstMoment[1]),
          type: 'scatter',
          mode: 'lines',
          name: 'First Moment Y'
        },
        {
          x: this.history.map(step => step.iteration),
          y: this.history.map(step => Math.sqrt(step.secondMoment[0])),
          type: 'scatter',
          mode: 'lines',
          name: 'Second Moment X (sqrt)'
        },
        {
          x: this.history.map(step => step.iteration),
          y: this.history.map(step => Math.sqrt(step.secondMoment[1])),
          type: 'scatter',
          mode: 'lines',
          name: 'Second Moment Y (sqrt)'
        }
      ],
      layout: {
        title: 'Moment Evolution',
        xaxis: { title: 'Iteration' },
        yaxis: { title: 'Moment Value' }
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
      finalFirstMoment: finalStep.firstMoment,
      finalSecondMoment: finalStep.secondMoment
    };
  }
  
  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<AdamConfig>): void {
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
    this.firstMoment = [0, 0];
    this.secondMoment = [0, 0];
    this.history = [];
    this.iteration = 0;
  }
}