<template>
  <ClientOnly>
  <BaseChart
    :data="plotlyData"
    :layout="plotlyLayout"
    :config="plotlyConfig"
    :loading="loading"
    :error="error"
    :height="String(height)"
  />
  </ClientOnly>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import type { Layout, Config } from 'plotly.js-dist-min';
import BaseChart from './BaseChart.vue';

/**
 * SegmentationMask Visualization Component
 * 
 * Displays image segmentation results with colored regions for each segment.
 * Shows pixel-wise classification with distinct colors for different segments.
 * 
 * Requirements: 2.3, 16.1, 16.2
 */

interface Props {
  mask: number[][];
  colors: string[];
  width: number;
  height: number;
  numSegments: number;
  title?: string;
  showLegend?: boolean;
  loading?: boolean;
  error?: string | null;
  chartHeight?: string;
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Segmentation Mask',
  showLegend: true,
  loading: false,
  error: null,
  chartHeight: '500px',
});

const plotlyData = computed(() => {
  // Create a custom colorscale based on segment colors
  const colorscale: Array<[number, string]> = [];
  
  for (let i = 0; i < props.numSegments; i++) {
    const position = i / Math.max(1, props.numSegments - 1);
    colorscale.push([position, props.colors[i] || '#cccccc']);
  }
  
  // Ensure we have start and end points
  if (colorscale.length === 1) {
    colorscale.push([1, colorscale[0][1]]);
  }
  
  return [{
    z: props.mask,
    type: 'heatmap',
    colorscale: colorscale,
    showscale: props.showLegend,
    hovertemplate: 
      'X: %{x}<br>' +
      'Y: %{y}<br>' +
      'Segment: %{z}<br>' +
      '<extra></extra>',
    colorbar: props.showLegend ? {
      title: {
        text: 'Segment',
        side: 'right',
      },
      thickness: 20,
      len: 0.7,
      tickmode: 'linear',
      tick0: 0,
      dtick: 1,
      tickfont: { size: 10 },
    } : undefined,
    zmin: 0,
    zmax: props.numSegments - 1,
  }];
});

const plotlyLayout = computed<Partial<Layout>>(() => ({
  title: props.title ? {
    text: props.title,
    font: { size: 16, weight: 600 },
  } : undefined,
  xaxis: {
    title: '',
    showgrid: false,
    showticklabels: false,
    zeroline: false,
  },
  yaxis: {
    title: '',
    showgrid: false,
    showticklabels: false,
    zeroline: false,
    autorange: 'reversed',
    scaleanchor: 'x',
    scaleratio: 1,
  },
  margin: {
    l: 40,
    r: props.showLegend ? 100 : 40,
    t: props.title ? 60 : 40,
    b: 40,
  },
  annotations: getSegmentAnnotations(),
}));

const plotlyConfig = computed<Partial<Config>>(() => ({
  responsive: true,
  displayModeBar: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
  toImageButtonOptions: {
    format: 'png',
    filename: 'segmentation_mask',
    height: 800,
    width: 800,
    scale: 2,
  },
}));

/**
 * Get annotations showing segment statistics
 */
const getSegmentAnnotations = () => {
  if (!props.showLegend) return [];
  
  // Count pixels in each segment
  const segmentCounts: number[] = Array(props.numSegments).fill(0);
  
  for (const row of props.mask) {
    for (const segment of row) {
      if (segment >= 0 && segment < props.numSegments) {
        segmentCounts[segment]++;
      }
    }
  }
  
  const totalPixels = props.width * props.height;
  
  // Create legend annotations
  const annotations: any[] = [];
  const legendX = 1.15;
  const legendY = 0.9;
  const spacing = 0.08;
  
  for (let i = 0; i < props.numSegments; i++) {
    const percentage = ((segmentCounts[i] / totalPixels) * 100).toFixed(1);
    
    annotations.push({
      x: legendX,
      y: legendY - i * spacing,
      xref: 'paper',
      yref: 'paper',
      text: `Segment ${i}: ${percentage}%`,
      showarrow: false,
      font: {
        size: 11,
        color: props.colors[i] || '#666666',
      },
      xanchor: 'left',
      yanchor: 'middle',
    });
  }
  
  return annotations;
};
</script>
