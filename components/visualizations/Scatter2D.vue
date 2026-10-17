<template>
  <ClientOnly>
  <BaseChart
    :data="plotlyData"
    :layout="plotlyLayout"
    :config="plotlyConfig"
    :loading="loading"
    :error="error"
    :height="height"
    @plotly-click="handleClick"
    @plotly-hover="handleHover"
    @plotly-unhover="handleUnhover"
  />
  </ClientOnly>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import type { Layout, Config } from 'plotly.js-dist-min';
import BaseChart from './BaseChart.vue';

export interface ScatterSeries {
  x: number[];
  y: number[];
  name?: string;
  color?: string;
  mode?: 'markers' | 'lines' | 'lines+markers';
  marker?: {
    size?: number;
    symbol?: string;
    opacity?: number;
    line?: {
      color?: string;
      width?: number;
    };
  };
  line?: {
    color?: string;
    width?: number;
    dash?: 'solid' | 'dot' | 'dash' | 'dashdot';
  };
  hovertemplate?: string;
}

interface Props {
  series: ScatterSeries[];
  title?: string;
  xLabel?: string;
  yLabel?: string;
  showLegend?: boolean;
  loading?: boolean;
  error?: string | null;
  height?: string;
  xRange?: [number, number];
  yRange?: [number, number];
  showGrid?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  title: '',
  xLabel: 'X',
  yLabel: 'Y',
  showLegend: true,
  loading: false,
  error: null,
  height: '500px',
  showGrid: true,
});

const emit = defineEmits<{
  pointClick: [data: { seriesIndex: number; pointIndex: number; x: number; y: number }];
  pointHover: [data: { seriesIndex: number; pointIndex: number; x: number; y: number }];
}>();

const defaultColors = [
  '#3b82f6', // blue
  '#ef4444', // red
  '#10b981', // green
  '#f59e0b', // amber
  '#8b5cf6', // purple
  '#ec4899', // pink
  '#14b8a6', // teal
  '#f97316', // orange
];

const plotlyData = computed(() => {
  return props.series.map((s, index) => ({
    x: s.x,
    y: s.y,
    type: 'scatter',
    mode: s.mode || 'markers',
    name: s.name || `Series ${index + 1}`,
    marker: {
      size: s.marker?.size || 8,
      color: s.color || defaultColors[index % defaultColors.length],
      symbol: s.marker?.symbol || 'circle',
      opacity: s.marker?.opacity || 0.8,
      line: {
        color: 'white',
        width: 1,
      },
    },
    hovertemplate: s.hovertemplate || 
      `<b>${s.name || `Series ${index + 1}`}</b><br>` +
      `${props.xLabel}: %{x:.2f}<br>` +
      `${props.yLabel}: %{y:.2f}<br>` +
      '<extra></extra>',
  }));
});

const plotlyLayout = computed<Partial<Layout>>(() => ({
  title: props.title ? {
    text: props.title,
    font: { size: 16, weight: 600 },
  } : undefined,
  xaxis: {
    title: props.xLabel,
    showgrid: props.showGrid,
    gridcolor: '#e5e7eb',
    zeroline: true,
    zerolinecolor: '#9ca3af',
    range: props.xRange,
  },
  yaxis: {
    title: props.yLabel,
    showgrid: props.showGrid,
    gridcolor: '#e5e7eb',
    zeroline: true,
    zerolinecolor: '#9ca3af',
    range: props.yRange,
  },
  showlegend: props.showLegend,
  legend: {
    x: 1,
    xanchor: 'right',
    y: 1,
    bgcolor: 'rgba(255, 255, 255, 0.8)',
    bordercolor: '#e5e7eb',
    borderwidth: 1,
  },
  hovermode: 'closest',
  dragmode: 'zoom',
}));

const plotlyConfig = computed<Partial<Config>>(() => ({
  responsive: true,
  displayModeBar: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
  toImageButtonOptions: {
    format: 'png',
    filename: 'scatter_plot',
    height: 800,
    width: 1200,
    scale: 2,
  },
}));

const handleClick = (event: any) => {
  if (event.points && event.points.length > 0) {
    const point = event.points[0];
    emit('pointClick', {
      seriesIndex: point.curveNumber,
      pointIndex: point.pointIndex,
      x: point.x,
      y: point.y,
    });
  }
};

const handleHover = (event: any) => {
  if (event.points && event.points.length > 0) {
    const point = event.points[0];
    emit('pointHover', {
      seriesIndex: point.curveNumber,
      pointIndex: point.pointIndex,
      x: point.x,
      y: point.y,
    });
  }
};

const handleUnhover = () => {
  // Optional: emit unhover event if needed
};
</script>
