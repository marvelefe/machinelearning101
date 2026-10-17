/**
 * Learning Rate Scheduling Algorithms
 * 
 * Implements various learning rate scheduling strategies to improve
 * optimization convergence and stability.
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
 * Learning rate schedule types
 */
export type ScheduleType = 'constant' | 'step' | 'exponential' | 'cosine' | 'polynomial' | 'plateau';

/**
 * Optimization step result
 */
export interface OptimizationStep {
  iteration: number;
  params: number[];
  loss: number;
  gradient: number[];
  learningRate: number;
  scheduledLearningRate: number;
}

/**
 * Learning Rate Scheduler Configuration
 */
export interface LearningRateSchedulerConfig {
  initialLearningRate: number;
  scheduleType: ScheduleType;
  maxIterations: number;
  tolerance: number;
  lossFunction: string;
  startX: number;
  startY: number;
  
  // Step schedule parameters
  stepSize?: number;
  gamma?: number;
  
  // Exponential schedule parameters
  decayRate?: number;
  
  // Cosine schedule parameters
  minLearningRate?: number;
  
  // Polynomial schedule parameters
  power?: number;
  
  // Plateau schedule parameters
  patience?: number;
  factor?: number;
  threshold?: number;
}

/**
 * Learning Rate Scheduler Algorithm
 * 
 * Demonstrates various learning rate scheduling strategies:
 * - Constant: Fixed learning rate
 * - Step: Reduce LR by factor every N steps
 * - Exponential: Exponential decay
 * - Cosine: Cosine annealing
 * - Polynomial: Polynomial decay
 * - Plateau: Reduce when loss plateaus
 */
export class LearningRateScheduler extends BaseAlgorithm {
  name = 'Learning Rate Scheduler';
  
  private config: LearningRateSchedulerConfig;
  private params: number[] = [0, 0];
  private history: OptimizationStep[] = [];
  private lossFunction: LossFunction;
  private gradientFunction: GradientFunction;
  private currentLearningRate: number;
  private plateauCounter: number = 0;
  private bestLoss: number = Infinity;
  
  constructor(config: Partial<LearningRateSchedulerConfig> = {}) {
    super();
    
    this.config = {
      initialLearningRate: 0.1,
      scheduleType: 'constant',
      maxIterations: 200,
      tolerance: 1e-6,
      lossFunction: 'quadratic-bowl',
      startX: -3,
      startY: 2,
      
      // Step schedule
      stepSize: 50,
      gamma: 0.5,
      
      // Exponential schedule
      decayRate: 0.95,
      
      // Cosine schedule
      minLearningRate: 0.001,
      
      // Polynomial schedule
      power: 1.0,
      
      // Plateau schedule
      patience: 10,
      factor: 0.5,
      threshold: 1e-4,
      
      ...config
    };
    
    this.params = [this.config.startX, this.config.startY];
    this.currentLearningRate = this.config.initialLearningRate;
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
        
      case 'noisy-quadratic':
        this.lossFunction = (params) => {
          const [x, y] = params;
          const noise = 0.1 * (Math.random() - 0.5);
          return x * x + y * y + noise;
        };
        this.gradientFunction = (params) => {
          const [x, y] = params;
          const noiseX = 0.05 * (Math.random() - 0.5);
          const noiseY = 0.05 * (Math.random() - 0.5);
          return [2 * x + noiseX, 2 * y + noiseY];
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
   * Calculate learning rate based on schedule
   */
  private calculateLearningRate(iteration: number, currentLoss: number): number {
    const { scheduleType, initialLearningRate } = this.config;
    
    switch (scheduleType) {
      case 'constant':
        return initialLearningRate;
        
      case 'step':
        const stepDecays = Math.floor(iteration / (this.config.stepSize || 50));
        return initialLearningRate * Math.pow(this.config.gamma || 0.5, stepDecays);
        
      case 'exponential':
        return initialLearningRate * Math.pow(this.config.decayRate || 0.95, iteration);
        
      case 'cosine':
        const minLR = this.config.minLearningRate || 0.001;
        const cosineDecay = 0.5 * (1 + Math.cos(Math.PI * iteration / this.config.maxIterations));
        return minLR + (initialLearningRate - minLR) * cosineDecay;
        
      case 'polynomial':
        const power = this.config.power || 1.0;
        const decayFactor = Math.pow(1 - iteration / this.config.maxIterations, power);
        return initialLearningRate * decayFactor;
        
      case 'plateau':
        // Check if loss has improved
        if (currentLoss < this.bestLoss - (this.config.threshold || 1e-4)) {
          this.bestLoss = currentLoss;
          this.plateauCounter = 0;
        } else {
          this.plateauCounter++;
        }
        
        // Reduce learning rate if plateau detected
        if (this.plateauCounter >= (this.config.patience || 10)) {
          this.currentLearningRate *= (this.config.factor || 0.5);
          this.plateauCounter = 0;
        }
        
        return this.currentLearningRate;
        
      default:
        return initialLearningRate;
    }
  }
  
  /**
   * Train with learning rate scheduling
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    this.reset();
    this.setupLossFunction();
    
    // Reset to starting position
    this.params = [this.config.startX, this.config.startY];
    this.history = [];
    this.currentLearningRate = this.config.initialLearningRate;
    this.plateauCounter = 0;
    this.bestLoss = Infinity;
    
    let prevLoss = Infinity;
    
    for (let iteration = 0; iteration < this.config.maxIterations; iteration++) {
      // Calculate current loss and gradient
      const currentLoss = this.lossFunction(this.params);
      const gradient = this.gradientFunction(this.params);
      
      // Calculate scheduled learning rate
      const scheduledLR = this.calculateLearningRate(iteration, currentLoss);
      
      // Record step before update
      this.history.push({
        iteration,
        params: [...this.params],
        loss: currentLoss,
        gradient: [...gradient],
        learningRate: this.config.initialLearningRate,
        scheduledLearningRate: scheduledLR
      });
      
      // Check for convergence
      if (Math.abs(prevLoss - currentLoss) < this.config.tolerance) {
        break;
      }
      
      // SGD update with scheduled learning rate
      this.params = this.params.map((param, i) => 
        param - scheduledLR * gradient[i]
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
        title: 'Optimization Path with Learning Rate Scheduling',
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
    
    // 3. Learning rate schedule
    visualizations.push({
      type: 'line',
      data: {
        x: this.history.map(step => step.iteration),
        y: this.history.map(step => step.scheduledLearningRate),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Scheduled Learning Rate'
      },
      layout: {
        title: `Learning Rate Schedule (${this.config.scheduleType})`,
        xaxis: { title: 'Iteration' },
        yaxis: { title: 'Learning Rate' }
      }
    });
    
    // 4. Parameter evolution
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
    
    // 5. Gradient magnitude evolution
    visualizations.push({
      type: 'line',
      data: {
        x: this.history.map(step => step.iteration),
        y: this.history.map(step => 
          Math.sqrt(step.gradient.reduce((sum, g) => sum + g * g, 0))
        ),
        type: 'scatter',
        mode: 'lines',
        name: 'Gradient Magnitude'
      },
      layout: {
        title: 'Gradient Magnitude Over Time',
        xaxis: { title: 'Iteration' },
        yaxis: { title: 'Gradient Magnitude' }
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
        
        // Use deterministic version for contour
        const loss = this.config.lossFunction === 'noisy-quadratic' 
          ? xVal * xVal + yVal * yVal 
          : this.lossFunction([xVal, yVal]);
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
    
    // Calculate average learning rate
    const avgLearningRate = this.history.reduce((sum, step) => 
      sum + step.scheduledLearningRate, 0) / this.history.length;
    
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
      scheduleType: this.config.scheduleType,
      initialLearningRate: this.config.initialLearningRate,
      finalLearningRate: finalStep.scheduledLearningRate,
      averageLearningRate: avgLearningRate
    };
  }
  
  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<LearningRateSchedulerConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.params = [this.config.startX, this.config.startY];
    this.currentLearningRate = this.config.initialLearningRate;
    this.setupLossFunction();
  }
  
  /**
   * Reset the optimizer
   */
  override reset(): void {
    super.reset();
    this.params = [this.config.startX, this.config.startY];
    this.history = [];
    this.currentLearningRate = this.config.initialLearningRate;
    this.plateauCounter = 0;
    this.bestLoss = Infinity;
  }
}