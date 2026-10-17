/**
 * Gradient Descent Optimizer Classes
 * 
 * Implements various gradient descent optimizers for visualization and learning:
 * - SGD (Stochastic Gradient Descent)
 * - SGD with Momentum
 * - Adam (Adaptive Moment Estimation)
 * - RMSProp (Root Mean Square Propagation)
 * 
 * Requirements: 4.1, 4.2, 4.3, 4.4
 */

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
  learningRate?: number;
}

/**
 * Optimizer configuration
 */
export interface OptimizerConfig {
  learningRate: number;
  maxIterations?: number;
  tolerance?: number;
  trackHistory?: boolean;
}

/**
 * Base optimizer interface
 */
export interface Optimizer {
  name: string;
  step(gradient: number[]): number[];
  reset(): void;
  getHistory(): OptimizationStep[];
}

/**
 * Base Optimizer class with common functionality
 */
export abstract class BaseOptimizer implements Optimizer {
  abstract name: string;
  
  protected params: number[];
  protected iteration: number = 0;
  protected history: OptimizationStep[] = [];
  protected trackHistory: boolean;
  
  constructor(
    initialParams: number[],
    protected config: OptimizerConfig
  ) {
    this.params = [...initialParams];
    this.trackHistory = config.trackHistory ?? true;
  }
  
  /**
   * Perform one optimization step
   */
  abstract step(gradient: number[]): number[];
  
  /**
   * Reset optimizer state
   */
  reset(): void {
    this.iteration = 0;
    this.history = [];
  }
  
  /**
   * Get optimization history
   */
  getHistory(): OptimizationStep[] {
    return this.history;
  }
  
  /**
   * Record step in history
   */
  protected recordStep(params: number[], loss: number, gradient: number[]): void {
    if (this.trackHistory) {
      this.history.push({
        iteration: this.iteration,
        params: [...params],
        loss,
        gradient: [...gradient],
        learningRate: this.config.learningRate
      });
    }
  }
}

/**
 * Stochastic Gradient Descent (SGD)
 * 
 * Basic gradient descent: θ = θ - α∇L(θ)
 */
export class SGD extends BaseOptimizer {
  name = 'SGD';
  
  step(gradient: number[]): number[] {
    const newParams = this.params.map((param, i) => 
      param - this.config.learningRate * gradient[i]
    );
    
    this.params = newParams;
    this.iteration++;
    
    return [...newParams];
  }
}

/**
 * SGD with Momentum
 * 
 * Adds momentum to accelerate convergence:
 * v = βv + ∇L(θ)
 * θ = θ - αv
 */
export class SGDMomentum extends BaseOptimizer {
  name = 'SGD with Momentum';
  
  private velocity: number[];
  private momentum: number;
  
  constructor(
    initialParams: number[],
    config: OptimizerConfig & { momentum?: number }
  ) {
    super(initialParams, config);
    this.velocity = new Array(initialParams.length).fill(0);
    this.momentum = config.momentum ?? 0.9;
  }
  
  step(gradient: number[]): number[] {
    // Update velocity: v = βv + ∇L
    this.velocity = this.velocity.map((v, i) => 
      this.momentum * v + gradient[i]
    );
    
    // Update parameters: θ = θ - αv
    const newParams = this.params.map((param, i) => 
      param - this.config.learningRate * this.velocity[i]
    );
    
    this.params = newParams;
    this.iteration++;
    
    return [...newParams];
  }
  
  override reset(): void {
    super.reset();
    this.velocity = new Array(this.params.length).fill(0);
  }
}

/**
 * RMSProp (Root Mean Square Propagation)
 * 
 * Adapts learning rate based on recent gradient magnitudes:
 * E[g²] = βE[g²] + (1-β)g²
 * θ = θ - α * g / √(E[g²] + ε)
 */
export class RMSProp extends BaseOptimizer {
  name = 'RMSProp';
  
  private squaredGradientAvg: number[];
  private beta: number;
  private epsilon: number;
  
  constructor(
    initialParams: number[],
    config: OptimizerConfig & { beta?: number; epsilon?: number }
  ) {
    super(initialParams, config);
    this.squaredGradientAvg = new Array(initialParams.length).fill(0);
    this.beta = config.beta ?? 0.9;
    this.epsilon = config.epsilon ?? 1e-8;
  }
  
  step(gradient: number[]): number[] {
    // Update squared gradient average: E[g²] = βE[g²] + (1-β)g²
    this.squaredGradientAvg = this.squaredGradientAvg.map((avg, i) => 
      this.beta * avg + (1 - this.beta) * gradient[i] * gradient[i]
    );
    
    // Update parameters: θ = θ - α * g / √(E[g²] + ε)
    const newParams = this.params.map((param, i) => 
      param - this.config.learningRate * gradient[i] / 
      (Math.sqrt(this.squaredGradientAvg[i]) + this.epsilon)
    );
    
    this.params = newParams;
    this.iteration++;
    
    return [...newParams];
  }
  
  override reset(): void {
    super.reset();
    this.squaredGradientAvg = new Array(this.params.length).fill(0);
  }
}

/**
 * Adam (Adaptive Moment Estimation)
 * 
 * Combines momentum and RMSProp:
 * m = β₁m + (1-β₁)g
 * v = β₂v + (1-β₂)g²
 * m̂ = m / (1-β₁ᵗ)
 * v̂ = v / (1-β₂ᵗ)
 * θ = θ - α * m̂ / (√v̂ + ε)
 */
export class Adam extends BaseOptimizer {
  name = 'Adam';
  
  private firstMoment: number[];
  private secondMoment: number[];
  private beta1: number;
  private beta2: number;
  private epsilon: number;
  
  constructor(
    initialParams: number[],
    config: OptimizerConfig & { beta1?: number; beta2?: number; epsilon?: number }
  ) {
    super(initialParams, config);
    this.firstMoment = new Array(initialParams.length).fill(0);
    this.secondMoment = new Array(initialParams.length).fill(0);
    this.beta1 = config.beta1 ?? 0.9;
    this.beta2 = config.beta2 ?? 0.999;
    this.epsilon = config.epsilon ?? 1e-8;
  }
  
  step(gradient: number[]): number[] {
    this.iteration++;
    
    // Update biased first moment: m = β₁m + (1-β₁)g
    this.firstMoment = this.firstMoment.map((m, i) => 
      this.beta1 * m + (1 - this.beta1) * gradient[i]
    );
    
    // Update biased second moment: v = β₂v + (1-β₂)g²
    this.secondMoment = this.secondMoment.map((v, i) => 
      this.beta2 * v + (1 - this.beta2) * gradient[i] * gradient[i]
    );
    
    // Compute bias-corrected moments
    const firstMomentCorrected = this.firstMoment.map(m => 
      m / (1 - Math.pow(this.beta1, this.iteration))
    );
    
    const secondMomentCorrected = this.secondMoment.map(v => 
      v / (1 - Math.pow(this.beta2, this.iteration))
    );
    
    // Update parameters: θ = θ - α * m̂ / (√v̂ + ε)
    const newParams = this.params.map((param, i) => 
      param - this.config.learningRate * firstMomentCorrected[i] / 
      (Math.sqrt(secondMomentCorrected[i]) + this.epsilon)
    );
    
    this.params = newParams;
    
    return [...newParams];
  }
  
  override reset(): void {
    super.reset();
    this.firstMoment = new Array(this.params.length).fill(0);
    this.secondMoment = new Array(this.params.length).fill(0);
  }
}

/**
 * Optimizer factory
 */
export function createOptimizer(
  type: 'sgd' | 'momentum' | 'rmsprop' | 'adam',
  initialParams: number[],
  config: OptimizerConfig & { momentum?: number; beta?: number; beta1?: number; beta2?: number; epsilon?: number }
): Optimizer {
  switch (type) {
    case 'sgd':
      return new SGD(initialParams, config);
    case 'momentum':
      return new SGDMomentum(initialParams, config);
    case 'rmsprop':
      return new RMSProp(initialParams, config);
    case 'adam':
      return new Adam(initialParams, config);
    default:
      throw new Error(`Unknown optimizer type: ${type}`);
  }
}
