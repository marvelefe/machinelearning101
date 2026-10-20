/**
 * Linear Regression Algorithm Tests
 *
 * Tests deterministic training, predictions, metrics, and visualization output.
 */

import { describe, it, expect } from 'vitest';
import { LinearRegression } from '../algorithms/regression/LinearRegression';

const X = [[1], [2], [3], [4]];
const y = [2, 4, 6, 8];

function createAlgorithm(): LinearRegression {
  return new LinearRegression({ learningRate: 0.1, epochs: 500, normalize: false });
}

describe('Linear Regression Algorithm', () => {
  it('fits a known linear relationship and predicts unseen values', async () => {
    const algorithm = createAlgorithm();

    await algorithm.fit(X, y);

    expect(algorithm.predict([[5]])[0]).toBeCloseTo(10, 1);
    algorithm.predict(X).forEach((prediction, index) => {
      expect(prediction).toBeCloseTo(y[index], 1);
    });
  });

  it('reports regression metrics for the fitted training data', async () => {
    const algorithm = createAlgorithm();
    await algorithm.fit(X, y);

    const metrics = algorithm.getMetrics();

    expect(metrics.mse).toBeDefined();
    expect(metrics.rmse).toBeDefined();
    expect(metrics.mae).toBeDefined();
    expect(metrics.r2Score).toBeDefined();
    expect(metrics.mse!).toBeLessThan(0.01);
    expect(metrics.mae!).toBeLessThan(0.1);
    expect(metrics.r2Score!).toBeGreaterThan(0.99);
  });

  it('provides regression, residual, and loss visualizations after training', async () => {
    const algorithm = createAlgorithm();
    await algorithm.fit(X, y);

    const visualizations = algorithm.getVisualizationData();

    expect(visualizations).toHaveLength(3);
    expect(visualizations[0].type).toBe('scatter2d');
    expect(visualizations[0].data.scatter.x).toEqual([1, 2, 3, 4]);
    expect(visualizations[0].data.scatter.y).toEqual(y);
    expect(visualizations[0].data.line.x).toHaveLength(101);
    expect(visualizations[1].data.residuals.x).toEqual([1, 2, 3, 4]);
    expect(visualizations[2].type).toBe('line');
    expect(visualizations[2].data.x).toHaveLength(500);
    expect(visualizations[2].data.y).toHaveLength(500);
  });
});
