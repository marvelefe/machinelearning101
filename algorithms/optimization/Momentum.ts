/**
 * Momentum and Nesterov Accelerated Gradient Algorithms
 * 
 * Implements momentum-based optimization methods that accelerate
 * convergence by accumulating gradients over time.
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
 * Momentum method types
 */
export type MomentumType = 'none' | 'standard' | 'nesterov';

/**
 * Optimization step result
 */
export interface OptimizationStep {
  iteration: number;
  params: number[];
  loss: number;
  gradient: number[];
  velocity: number[];
  learningRate: number;
  momentumType: MomentumType;
}

/**
 * Momentum Configuration
 */
export interface MomentumConfig {
  learningRate: number;
  momentum: number;
  momentumType: MomentumType;
  maxIterations: number;
  tolerance: number;
  lossFunction: string;
  startX: number;
  startY: number;
}

/**
 * Momentum and Nesterov Optimizer Algorithm
 * 
 * Implements three variants:
 * 1. Standard SGD: v = 0, θ = θ - α∇L(θ)
 * 2. SGD with Momentum: v = βv + ∇L(θ), θ = θ - αv
 * 3. Nesterov Momentum: v = βv + ∇L(θ - αβv), θ = θ - αv
 * 
 * This implementation demonstrates:
 * - Acceleration through momentum
 * - Lookahead capability of Nesterov
 * - Comparison of different momentum methods
 * - Visualization of velocity and optimization paths
 */
export class Momentum extends BaseAlgorithm {
  name = 'Momentum Optimizer';
  
  private config: MomentumConfig;
  private params: number[] = [0, 0];
  private velocity: number[] = [0, 0];
  private history: OptimizationStep[] = [];
  private lossFunction: LossFunction;
  private gradientFunction: GradientFunction;
  
  constructor(config: Partial<MomentumConfig> = {}) {
    super();
    
    this.config = {
      learningRate: 0.01,
      momentum: 0.9,
      momentumType: 'standard',
      maxIterations: 200,
      tolerance: 1e-6,
      lossFunction: 'quadratic-bowl',
      startX: -3,
      startY: 2,
      ...config
    };
    
    this.params = [this.config.startX, this.config.startY];
    this.velocity = [0, 0];
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
        
      case 'zigzag':
        this.lossFunction = (params) => {
          const [x, y] = params;
          return Math.abs(x) + Math.abs(y) + 0.1 * (x * x + y * y);
        };
        this.gradientFunction = (params) => {
          const [x, y] = params;
          const dx = Math.sign(x) + 0.2 * x;
          const dy = Math.sign(y) + 0.2 * y;
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
   * Train with momentum-based optimization
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    this.reset();
    this.setupLossFunction();
    
    // Reset to starting position
    this.params = [this.config.startX, this.config.startY];
    this.velocity = [0, 0];
    this.history = [];
    
    let prevLoss = Infinity;
    
    for (let iteration = 0; iteration < this.config.maxIterations; iteration++) {
      let currentLoss: number;
      let gradient: number[];
      
      if (this.config.momentumType === 'nesterov') {
        // Nesterov: Look ahead by computing gradient at θ - αβv
        const lookaheadParams = this.params.map((param, i) => 
          param - this.config.learningRate * this.config.momentum * this.velocity[i]
        );
        currentLoss = this.lossFunction(lookaheadParams);
        gradient = this.gradientFunction(lookaheadParams);
      } else {
        // Standard SGD or Momentum: compute gradient at current position
        currentLoss = this.lossFunction(this.params);
        gradient = this.gradientFunction(this.params);
      }
      
      // Record step before update
      this.history.push({
        iteration,
        params: [...this.params],
        loss: currentLoss,
        gradient: [...gradient],
        velocity: [...this.velocity],
        learningRate: this.config.learningRate,
        momentumType: this.config.momentumType
      });
      
      // Check for convergence
      if (Math.abs(prevLoss - currentLoss) < this.config.tolerance) {
        break;
      }
      
      // Update based on momentum type
      switch (this.config.momentumType) {
        case 'none':
          // Standard SGD: θ = θ - α∇L(θ)
          this.params = this.params.map((param, i) => 
            param - this.config.learningRate * gradient[i]
          );
          this.velocity = [0, 0]; // No velocity for visualization consistency
          break;
          
        case 'standard':
          // SGD with Momentum: v = βv + ∇L(θ), θ = θ - αv
          this.velocity = this.velocity.map((v, i) => 
            this.config.momentum * v + gradient[i]
          );
          this.params = this.params.map((param, i) => 
            param - this.config.learningRate * this.velocity[i]
          );
          break;
          
        case 'nesterov':
          // Nesterov Momentum: v = βv + ∇L(θ - αβv), θ = θ - αv
          this.velocity = this.velocity.map((v, i) => 
            this.config.momentum * v + gradient[i]
          );
          this.params = this.params.map((param, i) => 
            param - this.config.learningRate * this.velocity[i]
          );
          break;
      }
      
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
        title: `${this.getMomentumTypeName()} Optimization Path`,
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
    
    // 4. Velocity evolution (momentum-specific)
    if (this.config.momentumType !== 'none') {
      visualizations.push({
        type: 'line',
        data: [
          {
            x: this.history.map(step => step.iteration),
            y: this.history.map(step => step.velocity[0]),
            type: 'scatter',
            mode: 'lines',
            name: 'Velocity X'
          },
          {
            x: this.history.map(step => step.iteration),
            y: this.history.map(step => step.velocity[1]),
            type: 'scatter',
            mode: 'lines',
            name: 'Velocity Y'
          }
        ],
        layout: {
          title: 'Velocity Evolution',
          xaxis: { title: 'Iteration' },
          yaxis: { title: 'Velocity' }
        }
      });
    }
    
    // 5. Gradient vs velocity comparison
    if (this.config.momentumType !== 'none') {
      visualizations.push({
        type: 'line',
        data: [
          {
            x: this.history.map(step => step.iteration),
            y: this.history.map(step => 
              Math.sqrt(step.gradient.reduce((sum, g) => sum + g * g, 0))
            ),
            type: 'scatter',
            mode: 'lines',
            name: 'Gradient Magnitude'
          },
          {
            x: this.history.map(step => step.iteration),
            y: this.history.map(step => 
              Math.sqrt(step.velocity.reduce((sum, v) => sum + v * v, 0))
            ),
            type: 'scatter',
            mode: 'lines',
            name: 'Velocity Magnitude'
          }
        ],
        layout: {
          title: 'Gradient vs Velocity Magnitude',
          xaxis: { title: 'Iteration' },
          yaxis: { title: 'Magnitude' }
        }
      });
    }
    
    return visualizations;
  }
  
  /**
   * Get human-readable momentum type name
   */
  private getMomentumTypeName(): string {
    switch (this.config.momentumType) {
      case 'none': return 'Standard SGD';
      case 'standard': return 'SGD with Momentum';
      case 'nesterov': return 'Nesterov Accelerated Gradient';
      default: return 'Unknown';
    }
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
    const initialStep = this.history[0];
    const finalLoss = finalStep.loss;
    const initialLoss = initialStep.loss;
    
    // Calculate average velocity magnitude
    const avgVelocityMagnitude = this.config.momentumType !== 'none' 
      ? this.history.reduce((sum, step) => 
          sum + Math.sqrt(step.velocity.reduce((vSum, v) => vSum + v * v, 0)), 0
        ) / this.history.length
      : 0;
    
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
      momentumType: this.config.momentumType,
      momentum: this.config.momentum,
      finalVelocity: finalStep.velocity,
      averageVelocityMagnitude: avgVelocityMagnitude
    };
  }
  
  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<MomentumConfig>): void {
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
    this.velocity = [0, 0];
    this.history = [];
  }
}