# Developer Guide

Welcome to the ML Learning Platform developer documentation! This guide will help you understand the codebase, contribute effectively, and add new modules.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Architecture](#project-architecture)
3. [Adding New Modules](#adding-new-modules)
4. [Algorithm Implementation](#algorithm-implementation)
5. [Component Development](#component-development)
6. [Testing](#testing)
7. [Performance Considerations](#performance-considerations)
8. [Contribution Guidelines](#contribution-guidelines)

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Basic understanding of Vue 3, TypeScript, and Nuxt
- Familiarity with machine learning concepts (helpful but not required)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ml-learning-platform

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Generate static site
npm run generate
```

### Development Workflow

1. Create a feature branch from `main`
2. Make your changes following our coding standards
3. Test thoroughly (unit tests, visual testing, accessibility)
4. Run `npm run build` to ensure no errors
5. Submit a pull request with a clear description

## Project Architecture

### Directory Structure

```
.
├── algorithms/          # ML algorithm implementations
│   ├── advanced/       # Neural networks, PCA, t-SNE
│   ├── classification/ # Logistic regression
│   ├── clustering/     # K-means
│   ├── optimization/   # Gradient descent
│   ├── preprocessing/  # Feature scaling, imputation
│   └── regression/     # Linear regression
│
├── components/         # Vue components
│   ├── layout/        # Layout components
│   ├── modules/       # Module-specific components
│   ├── shared/        # Reusable UI components
│   └── visualizations/ # Visualization components
│
├── composables/        # Vue composables
├── content/           # Module content (markdown + JSON)
├── pages/             # Nuxt pages (auto-routing)
├── stores/            # Pinia stores
├── types/             # TypeScript type definitions
├── utils/             # Utility functions
└── workers/           # Web Workers for computation
```

### Key Architectural Patterns

#### 1. Module-Based Architecture

Each learning module is self-contained with:
- Algorithm implementation (`algorithms/[domain]/`)
- Page component (`pages/modules/[domain]/[module].vue`)
- Content files (`content/modules/[domain]/[module]/`)
- Configuration (`config.json`)

#### 2. State Management (Pinia)

- **module.ts**: Algorithm parameters, datasets, results
- **progress.ts**: User progress tracking
- **analytics.ts**: Analytics and consent
- **toast.ts**: Toast notifications

#### 3. Visualization Strategy

- **Plotly.js**: Scientific plots, 3D visualizations
- **D3.js**: Custom complex visualizations
- **Chart.js**: Simple charts
- **Canvas**: Performance-critical animations

#### 4. Computation Pattern

Heavy computations run in Web Workers to avoid blocking the UI:

```typescript
// Main thread
const worker = new Worker('/service-workers/algorithm-worker.ts');
worker.postMessage({ type: 'train', data, params });

// Worker thread
self.onmessage = (e) => {
  const result = runAlgorithm(e.data);
  self.postMessage({ type: 'complete', result });
};
```

## Adding New Modules

Follow these steps to add a new learning module:

### Step 1: Create Content Structure

Create a new directory in `content/modules/[domain]/[module-name]/`:

```bash
mkdir -p content/modules/[domain]/[module-name]
cd content/modules/[domain]/[module-name]
```

Create three files:

#### config.json
```json
{
  "id": "module-name",
  "name": "Module Display Name",
  "domain": "regression|classification|clustering|preprocessing|optimization|advanced",
  "difficulty": "beginner|intermediate|advanced",
  "estimatedTime": 30,
  "description": "Brief description of the module",
  "prerequisites": ["prerequisite-module-id"],
  "learningObjectives": [
    "Objective 1",
    "Objective 2"
  ],
  "interactive": {
    "algorithm": "AlgorithmClassName",
    "defaultParams": {
      "param1": 0.01,
      "param2": 100
    },
    "controls": [
      {
        "id": "param1",
        "type": "slider",
        "label": "Parameter 1",
        "min": 0.001,
        "max": 0.1,
        "step": 0.001,
        "tooltip": "Description of what this parameter does"
      }
    ],
    "datasets": ["dataset-id-1", "dataset-id-2"],
    "visualizations": [
      {
        "id": "main-viz",
        "type": "scatter2d",
        "title": "Visualization Title"
      }
    ]
  }
}
```

#### content.md
```markdown
# Module Name

## Introduction

Brief introduction to the concept...

## Key Concepts

### Concept 1

Explanation...

### Concept 2

Explanation...

## How It Works

Step-by-step explanation...

## When to Use

Practical guidance...

## Summary

Key takeaways...
```

#### quiz.json
```json
{
  "questions": [
    {
      "id": "q1",
      "text": "Question text?",
      "type": "multiple-choice",
      "options": [
        "Option A",
        "Option B",
        "Option C",
        "Option D"
      ],
      "correctAnswer": 0,
      "explanation": "Explanation of the correct answer"
    }
  ]
}
```

### Step 2: Implement the Algorithm

Create a new algorithm class in `algorithms/[domain]/[AlgorithmName].ts`:

```typescript
import type { Algorithm, AlgorithmResults } from '~/types/algorithm';

/**
 * [Algorithm Name] implementation
 * 
 * @description Brief description of the algorithm
 * Requirements: X.X, Y.Y
 */
export class AlgorithmName implements Algorithm {
  name = 'Algorithm Name';
  
  private param1: number;
  private param2: number;
  
  // Store results
  private results: AlgorithmResults | null = null;
  
  constructor(param1: number = 0.01, param2: number = 100) {
    this.param1 = param1;
    this.param2 = param2;
  }
  
  /**
   * Train the algorithm
   */
  async fit(X: number[][], y?: number[]): Promise<void> {
    // Implementation
    // For heavy computation, consider using Web Workers
  }
  
  /**
   * Make predictions
   */
  predict(X: number[][]): number[] | string[] {
    // Implementation
    return [];
  }
  
  /**
   * Get visualization data
   */
  getVisualizationData(): any {
    // Return data formatted for visualizations
    return {
      type: 'scatter2d',
      data: {
        // Visualization-specific data
      }
    };
  }
  
  /**
   * Get performance metrics
   */
  getMetrics(): Record<string, number> {
    // Return relevant metrics
    return {
      metric1: 0,
      metric2: 0
    };
  }
  
  /**
   * Reset the algorithm state
   */
  reset(): void {
    this.results = null;
  }
}
```

### Step 3: Create the Page Component

Create `pages/modules/[domain]/[module-name].vue`:

```vue
<script setup lang="ts">
import { ref, computed } from 'vue';
import { useModuleStore } from '~/stores/module';
import { AlgorithmName } from '~/algorithms/[domain]/AlgorithmName';

// Load module configuration
const config = await import(`~/content/modules/[domain]/[module-name]/config.json`);
const content = await import(`~/content/modules/[domain]/[module-name]/content.md`);
const quiz = await import(`~/content/modules/[domain]/[module-name]/quiz.json`);

// Initialize store
const moduleStore = useModuleStore();
moduleStore.initModule(config.id, config.interactive.defaultParams);

// Create algorithm instance
const algorithm = ref<AlgorithmName | null>(null);

// Handle training
const handleTrain = async () => {
  const params = moduleStore.parameters;
  algorithm.value = new AlgorithmName(params.param1, params.param2);
  
  const dataset = moduleStore.dataset;
  await algorithm.value.fit(dataset.X, dataset.y);
  
  moduleStore.setResults({
    visualizationData: algorithm.value.getVisualizationData(),
    metrics: algorithm.value.getMetrics()
  });
};

// SEO
useSEO({
  title: config.name,
  description: config.description
});
</script>

<template>
  <ModuleLayout
    :config="config"
    :content="content"
    :quiz="quiz"
  >
    <template #interactive>
      <InteractiveSection>
        <template #controls>
          <ControlPanel
            :controls="config.interactive.controls"
            :values="moduleStore.parameters"
            @update="moduleStore.setParameter"
            @train="handleTrain"
          />
        </template>
        
        <template #visualization>
          <VisualizationPanel
            v-if="moduleStore.results"
            :data="moduleStore.results.visualizationData"
          />
        </template>
      </InteractiveSection>
    </template>
  </ModuleLayout>
</template>
```

### Step 4: Add Navigation

Update the domain index page in `pages/modules/[domain]/index.vue` to include your new module in the list.

### Step 5: Test

1. Visual testing: Verify the module renders correctly
2. Functionality: Test all interactive controls
3. Accessibility: Test keyboard navigation and screen readers
4. Performance: Check computation speed and rendering
5. Responsive: Test on different screen sizes

## Algorithm Implementation

### Best Practices

1. **Type Safety**: Use TypeScript interfaces and types
2. **Documentation**: Add JSDoc comments with requirement references
3. **Error Handling**: Validate inputs and handle edge cases
4. **Performance**: Use typed arrays for numerical operations
5. **Web Workers**: Move heavy computations off the main thread

### Example: Complete Algorithm Implementation

```typescript
import type { Algorithm, AlgorithmResults, Metrics } from '~/types/algorithm';

/**
 * Example Algorithm Implementation
 * 
 * Requirements: 2.3, 4.1, 4.2
 */
export class ExampleAlgorithm implements Algorithm {
  name = 'Example Algorithm';
  
  private weights: number[] = [];
  private bias: number = 0;
  private history: { loss: number[]; epochs: number[] } = {
    loss: [],
    epochs: []
  };
  
  constructor(
    private learningRate: number = 0.01,
    private epochs: number = 100
  ) {
    this.validateParams();
  }
  
  /**
   * Validate constructor parameters
   */
  private validateParams(): void {
    if (this.learningRate <= 0 || this.learningRate > 1) {
      throw new Error('Learning rate must be between 0 and 1');
    }
    if (this.epochs < 1) {
      throw new Error('Epochs must be at least 1');
    }
  }
  
  /**
   * Train the model
   */
  async fit(X: number[][], y: number[]): Promise<void> {
    // Validate input
    if (X.length === 0 || y.length === 0) {
      throw new Error('Training data cannot be empty');
    }
    if (X.length !== y.length) {
      throw new Error('X and y must have the same length');
    }
    
    // Initialize weights
    const numFeatures = X[0].length;
    this.weights = new Array(numFeatures).fill(0);
    this.bias = 0;
    
    // Training loop
    for (let epoch = 0; epoch < this.epochs; epoch++) {
      // Forward pass
      const predictions = this.predict(X);
      
      // Calculate loss
      const loss = this.calculateLoss(predictions, y);
      this.history.loss.push(loss);
      this.history.epochs.push(epoch);
      
      // Backward pass (gradient descent)
      this.updateWeights(X, y, predictions);
      
      // Yield to prevent blocking (every 10 epochs)
      if (epoch % 10 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
  }
  
  /**
   * Make predictions
   */
  predict(X: number[][]): number[] {
    return X.map(features => {
      const sum = features.reduce((acc, val, idx) => 
        acc + val * this.weights[idx], 0
      ) + this.bias;
      return sum;
    });
  }
  
  /**
   * Calculate loss (MSE)
   */
  private calculateLoss(predictions: number[], actual: number[]): number {
    const mse = predictions.reduce((sum, pred, i) => {
      const error = pred - actual[i];
      return sum + error * error;
    }, 0) / predictions.length;
    return mse;
  }
  
  /**
   * Update weights using gradient descent
   */
  private updateWeights(
    X: number[][],
    y: number[],
    predictions: number[]
  ): void {
    const n = X.length;
    
    // Calculate gradients
    const weightGradients = new Array(this.weights.length).fill(0);
    let biasGradient = 0;
    
    for (let i = 0; i < n; i++) {
      const error = predictions[i] - y[i];
      biasGradient += error;
      
      for (let j = 0; j < this.weights.length; j++) {
        weightGradients[j] += error * X[i][j];
      }
    }
    
    // Update weights
    for (let j = 0; j < this.weights.length; j++) {
      this.weights[j] -= (this.learningRate * weightGradients[j]) / n;
    }
    this.bias -= (this.learningRate * biasGradient) / n;
  }
  
  /**
   * Get visualization data
   */
  getVisualizationData(): any {
    return {
      lossHistory: {
        type: 'line',
        data: {
          x: this.history.epochs,
          y: this.history.loss,
          name: 'Training Loss'
        }
      },
      weights: this.weights,
      bias: this.bias
    };
  }
  
  /**
   * Get performance metrics
   */
  getMetrics(): Metrics {
    const lastLoss = this.history.loss[this.history.loss.length - 1] || 0;
    
    return {
      mse: lastLoss,
      rmse: Math.sqrt(lastLoss),
      epochs: this.history.epochs.length
    };
  }
  
  /**
   * Reset algorithm state
   */
  reset(): void {
    this.weights = [];
    this.bias = 0;
    this.history = { loss: [], epochs: [] };
  }
}
```

### Using Web Workers for Heavy Computation

For computationally intensive algorithms, use Web Workers:

```typescript
// algorithms/[domain]/[Algorithm].worker.ts
import { AlgorithmName } from './AlgorithmName';

self.onmessage = async (e: MessageEvent) => {
  const { type, data, params } = e.data;
  
  try {
    if (type === 'train') {
      const algorithm = new AlgorithmName(params.param1, params.param2);
      
      // Train with progress updates
      await algorithm.fit(data.X, data.y);
      
      // Send results back
      self.postMessage({
        type: 'complete',
        results: {
          visualizationData: algorithm.getVisualizationData(),
          metrics: algorithm.getMetrics()
        }
      });
    }
  } catch (error) {
    self.postMessage({
      type: 'error',
      error: error.message
    });
  }
};
```

## Component Development

### Shared Component Guidelines

1. **Props**: Use TypeScript interfaces for prop types
2. **Accessibility**: Include ARIA labels and keyboard support
3. **Responsive**: Test on mobile, tablet, and desktop
4. **Reusability**: Keep components generic and configurable

### Example: Creating a Shared Component

```vue
<script setup lang="ts">
interface Props {
  modelValue: number;
  min?: number;
  max?: number;
  step?: number;
  label: string;
  tooltip?: string;
  disabled?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  min: 0,
  max: 100,
  step: 1,
  disabled: false
});

const emit = defineEmits<{
  'update:modelValue': [value: number];
}>();

const handleInput = (event: Event) => {
  const value = parseFloat((event.target as HTMLInputElement).value);
  emit('update:modelValue', value);
};
</script>

<template>
  <div class="slider-control">
    <label :for="`slider-${label}`" class="block text-sm font-medium mb-2">
      {{ label }}
      <Tooltip v-if="tooltip" :content="tooltip" />
    </label>
    
    <div class="flex items-center gap-4">
      <input
        :id="`slider-${label}`"
        type="range"
        :min="min"
        :max="max"
        :step="step"
        :value="modelValue"
        :disabled="disabled"
        :aria-label="label"
        :aria-valuemin="min"
        :aria-valuemax="max"
        :aria-valuenow="modelValue"
        class="flex-1"
        @input="handleInput"
      />
      
      <output
        :for="`slider-${label}`"
        class="text-sm font-mono w-16 text-right"
      >
        {{ modelValue }}
      </output>
    </div>
  </div>
</template>

<style scoped>
.slider-control input[type="range"] {
  @apply w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer;
}

.slider-control input[type="range"]::-webkit-slider-thumb {
  @apply appearance-none w-4 h-4 bg-blue-600 rounded-full cursor-pointer;
}

.slider-control input[type="range"]:focus {
  @apply outline-none ring-2 ring-blue-500 ring-offset-2;
}

.slider-control input[type="range"]:disabled {
  @apply opacity-50 cursor-not-allowed;
}
</style>
```

## Testing

### Unit Testing

Unit tests live in the root `tests/` directory, with one test file per algorithm. Use Vitest for unit tests:

```typescript
// tests/example-algorithm.test.ts
import { describe, it, expect } from 'vitest';
import { ExampleAlgorithm } from '../ExampleAlgorithm';

describe('ExampleAlgorithm', () => {
  it('should initialize with default parameters', () => {
    const algo = new ExampleAlgorithm();
    expect(algo.name).toBe('Example Algorithm');
  });
  
  it('should fit a simple linear relationship', async () => {
    const algo = new ExampleAlgorithm(0.01, 100);
    const X = [[1], [2], [3], [4]];
    const y = [2, 4, 6, 8];
    
    await algo.fit(X, y);
    const predictions = algo.predict([[5]]);
    
    expect(predictions[0]).toBeCloseTo(10, 0);
  });
  
  it('should throw error for invalid learning rate', () => {
    expect(() => new ExampleAlgorithm(-0.1, 100)).toThrow();
    expect(() => new ExampleAlgorithm(1.5, 100)).toThrow();
  });
  
  it('should throw error for empty dataset', async () => {
    const algo = new ExampleAlgorithm();
    await expect(algo.fit([], [])).rejects.toThrow('Training data cannot be empty');
  });
});
```

Run tests:
```bash
npm run test
```

### Component Testing

```typescript
// components/__tests__/Slider.test.ts
import { mount } from '@vue/test-utils';
import { describe, it, expect } from 'vitest';
import Slider from '../shared/Slider.vue';

describe('Slider', () => {
  it('renders with correct props', () => {
    const wrapper = mount(Slider, {
      props: {
        modelValue: 50,
        min: 0,
        max: 100,
        label: 'Test Slider'
      }
    });
    
    expect(wrapper.find('label').text()).toContain('Test Slider');
    expect(wrapper.find('input').element.value).toBe('50');
  });
  
  it('emits update event on input', async () => {
    const wrapper = mount(Slider, {
      props: {
        modelValue: 50,
        label: 'Test'
      }
    });
    
    await wrapper.find('input').setValue(75);
    expect(wrapper.emitted('update:modelValue')).toBeTruthy();
    expect(wrapper.emitted('update:modelValue')?.[0]).toEqual([75]);
  });
});
```

## Performance Considerations

### Bundle Size Optimization

1. **Code Splitting**: Algorithms and visualizations are lazy-loaded
2. **Tree Shaking**: Import only what you need
3. **Dynamic Imports**: Use for heavy dependencies

```typescript
// Lazy load visualization library
const Plotly = await import('plotly.js-dist-min');
```

### Computation Performance

1. **Typed Arrays**: Use for numerical operations
```typescript
const data = new Float64Array(1000);
```

2. **Web Workers**: For heavy computations
3. **Memoization**: Cache expensive calculations
```typescript
const memoized = computed(() => expensiveCalculation(params.value));
```

4. **Debouncing**: Limit update frequency
```typescript
const debouncedUpdate = useDebounceFn(updateVisualization, 100);
```

### Rendering Performance

1. **Virtual Scrolling**: For long lists
2. **Canvas**: For many data points (>1000)
3. **requestAnimationFrame**: For smooth animations
4. **v-once**: For static content

## Contribution Guidelines

### Code Style

- Follow TypeScript strict mode
- Use Composition API for Vue components
- Add JSDoc comments for public APIs
- Include requirement references in comments
- Use meaningful variable and function names

### Commit Messages

Follow conventional commits:
```
feat: add new k-means clustering module
fix: correct gradient descent convergence issue
docs: update algorithm implementation guide
style: format code with prettier
refactor: simplify dataset loading logic
test: add unit tests for linear regression
perf: optimize canvas rendering for large datasets
```

### Pull Request Process

1. Create a feature branch
2. Make your changes with clear commits
3. Add/update tests as needed
4. Update documentation
5. Run `npm run build` to verify
6. Submit PR with description of changes
7. Address review feedback

### Code Review Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] Accessibility is maintained
- [ ] Performance is acceptable
- [ ] No console errors or warnings
- [ ] Responsive design works on all screen sizes

## Resources

- [Vue 3 Documentation](https://vuejs.org/)
- [Nuxt Documentation](https://nuxt.com/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Pinia Documentation](https://pinia.vuejs.org/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Plotly.js](https://plotly.com/javascript/)
- [D3.js](https://d3js.org/)

## Getting Help

- Check existing documentation
- Search closed issues on GitHub
- Ask in discussions
- Review similar modules for patterns

## License

[Your License Here]
