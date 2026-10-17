<script setup lang="ts">
import { ref } from 'vue';
import type { LineSeries } from '~/components/visualizations/LineChart.vue';

// Sample data for Scatter2D
const scatterSeries = ref([
  {
    x: [1, 2, 3, 4, 5],
    y: [2, 4, 3, 5, 6],
    name: 'Series 1',
    color: '#3b82f6',
  },
  {
    x: [1, 2, 3, 4, 5],
    y: [1, 3, 2, 4, 5],
    name: 'Series 2',
    color: '#ef4444',
  },
]);

// Sample data for LineChart
const lineSeries = ref<LineSeries[]>([
  {
    x: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    y: [10, 8, 6, 5, 4, 3, 2.5, 2, 1.8, 1.5, 1.2],
    name: 'Training Loss',
    color: '#3b82f6',
  },
  {
    x: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    y: [10, 8.5, 7, 6, 5.5, 5, 4.8, 4.7, 4.6, 4.5, 4.5],
    name: 'Validation Loss',
    color: '#ef4444',
    dash: 'dash' as const,
  },
]);

// Sample data for Heatmap
const heatmapData = ref([
  [1, 20, 30],
  [20, 1, 60],
  [30, 60, 1],
]);
const heatmapLabels = ref(['Class A', 'Class B', 'Class C']);

// Sample data for 3D Scatter
const scatter3DSeries = ref([
  {
    x: [1, 2, 3, 4, 5],
    y: [2, 4, 3, 5, 6],
    z: [1, 3, 2, 4, 5],
    name: '3D Series',
    color: '#3b82f6',
  },
]);

// Sample data for Canvas Scatter (large dataset)
const generateLargeDataset = (count: number) => {
  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < count; i++) {
    x.push(Math.random() * 100);
    y.push(Math.random() * 100);
  }
  return { x, y };
};

const largeDataset1 = generateLargeDataset(1500);
const largeDataset2 = generateLargeDataset(1500);

const canvasSeries = ref([
  {
    ...largeDataset1,
    name: 'Cluster 1',
    color: '#3b82f6',
    size: 3,
  },
  {
    ...largeDataset2,
    name: 'Cluster 2',
    color: '#ef4444',
    size: 3,
  },
]);
</script>

<template>
  <div class="demo-page">
    <div class="container">
      <h1 class="title">Visualization Components Demo</h1>
      <p class="subtitle">Interactive visualization components for ML learning platform</p>

      <section class="demo-section">
        <h2>Scatter2D Plot</h2>
        <p>Interactive 2D scatter plot with zoom, pan, and hover tooltips</p>
        <div class="chart-wrapper">
          <Scatter2D
            :series="scatterSeries"
            title="Sample Scatter Plot"
            x-label="Feature X"
            y-label="Feature Y"
            :show-legend="true"
          />
        </div>
      </section>

      <section class="demo-section">
        <h2>Line Chart</h2>
        <p>Line chart for loss curves and training metrics</p>
        <div class="chart-wrapper">
          <LineChart
            :series="lineSeries"
            title="Training Progress"
            x-label="Epoch"
            y-label="Loss"
            :show-legend="true"
          />
        </div>
      </section>

      <section class="demo-section">
        <h2>Heatmap</h2>
        <p>Heatmap visualization for confusion matrices</p>
        <div class="chart-wrapper">
          <Heatmap
            :data="heatmapData"
            :x-labels="heatmapLabels"
            :y-labels="heatmapLabels"
            title="Confusion Matrix"
            :show-annotations="true"
          />
        </div>
      </section>

      <section class="demo-section">
        <h2>3D Scatter Plot</h2>
        <p>Interactive 3D scatter plot with rotation controls</p>
        <div class="chart-wrapper">
          <Plot3D
            type="scatter"
            :scatter-series="scatter3DSeries"
            title="3D Data Visualization"
            x-label="X Axis"
            y-label="Y Axis"
            z-label="Z Axis"
          />
        </div>
      </section>

      <section class="demo-section">
        <h2>Canvas Scatter (Performance)</h2>
        <p>High-performance canvas-based scatter plot for large datasets (3000 points)</p>
        <div class="chart-wrapper" style="height: 600px;">
          <CanvasScatter
            :series="canvasSeries"
            x-label="Feature X"
            y-label="Feature Y"
            :show-grid="true"
          />
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.demo-page {
  min-height: 100vh;
  background-color: #f9fafb;
  padding: 2rem 1rem;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
}

.title {
  font-size: 2.5rem;
  font-weight: 700;
  color: #111827;
  margin-bottom: 0.5rem;
}

.subtitle {
  font-size: 1.125rem;
  color: #6b7280;
  margin-bottom: 3rem;
}

.demo-section {
  background-color: white;
  border-radius: 0.5rem;
  padding: 2rem;
  margin-bottom: 2rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.demo-section h2 {
  font-size: 1.5rem;
  font-weight: 600;
  color: #111827;
  margin-bottom: 0.5rem;
}

.demo-section p {
  color: #6b7280;
  margin-bottom: 1.5rem;
}

.chart-wrapper {
  background-color: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 0.375rem;
  padding: 1rem;
  min-height: 400px;
}
</style>
