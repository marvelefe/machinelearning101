<script setup lang="ts">
import { computed } from 'vue';
import type { Layout, Config } from 'plotly.js-dist-min';

export interface Scatter3DSeries {
  x: number[];
  y: number[];
  z: number[];
  name?: string;
  color?: string;
  mode?: 'markers' | 'lines' | 'lines+markers' | 'markers+text';
  marker?: {
    size?: number;
    symbol?: string;
    opacity?: number;
  };
}

export interface Surface3DData {
  x: number[];
  y: number[];
  z: number[][];
  colorscale?: string | Array<[number, string]>;
  showscale?: boolean;
}

interface Props {
  type?: 'scatter' | 'surface';
  scatterSeries?: Scatter3DSeries[];
  surfaceData?: Surface3DData;
  title?: string;
  xLabel?: string;
  yLabel?: string;
  zLabel?: string;
  showLegend?: boolean;
  loading?: boolean;
  error?: string | null;
  height?: string;
  cameraPosition?: {
    eye?: { x: number; y: number; z: number };
    center?: { x: number; y: number; z: number };
  };
}

const props = withDefaults(defineProps<Props>(), {
  type: 'scatter',
  title: '',
  xLabel: 'X',
  yLabel: 'Y',
  zLabel: 'Z',
  showLegend: true,
  loading: false,
  error: null,
  height: '600px',
});

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
  if (props.type === 'scatter' && props.scatterSeries) {
    return props.scatterSeries.map((s, index) => ({
      x: s.x,
      y: s.y,
      z: s.z,
      type: 'scatter3d',
      mode: s.mode || 'markers',
      name: s.name || `Series ${index + 1}`,
      marker: {
        size: s.marker?.size || 5,
        color: s.color || defaultColors[index % defaultColors.length],
        symbol: s.marker?.symbol || 'circle',
        opacity: s.marker?.opacity || 0.8,
        line: {
          color: 'white',
          width: 0.5,
        },
      },
      hovertemplate: 
        `<b>${s.name || `Series ${index + 1}`}</b><br>` +
        `${props.xLabel}: %{x:.2f}<br>` +
        `${props.yLabel}: %{y:.2f}<br>` +
        `${props.zLabel}: %{z:.2f}<br>` +
        '<extra></extra>',
    }));
  } else if (props.type === 'surface' && props.surfaceData) {
    return [{
      x: props.surfaceData.x,
      y: props.surfaceData.y,
      z: props.surfaceData.z,
      type: 'surface',
      colorscale: props.surfaceData.colorscale || 'Viridis',
      showscale: props.surfaceData.showscale !== false,
      hovertemplate: 
        `${props.xLabel}: %{x:.2f}<br>` +
        `${props.yLabel}: %{y:.2f}<br>` +
        `${props.zLabel}: %{z:.2f}<br>` +
        '<extra></extra>',
      colorbar: {
        thickness: 15,
        len: 0.7,
        tickfont: { size: 10 },
      },
    }];
  }
  return [];
});

const plotlyLayout = computed<Partial<Layout>>(() => ({
  title: props.title ? {
    text: props.title,
    font: { size: 16, weight: 600 },
  } : undefined,
  scene: {
    xaxis: {
      title: props.xLabel,
      backgroundcolor: 'rgb(240, 240, 240)',
      gridcolor: 'white',
      showbackground: true,
      zerolinecolor: 'white',
    },
    yaxis: {
      title: props.yLabel,
      backgroundcolor: 'rgb(240, 240, 240)',
      gridcolor: 'white',
      showbackground: true,
      zerolinecolor: 'white',
    },
    zaxis: {
      title: props.zLabel,
      backgroundcolor: 'rgb(240, 240, 240)',
      gridcolor: 'white',
      showbackground: true,
      zerolinecolor: 'white',
    },
    camera: props.cameraPosition || {
      eye: { x: 1.5, y: 1.5, z: 1.5 },
      center: { x: 0, y: 0, z: 0 },
    },
  },
  showlegend: props.showLegend && props.type === 'scatter',
  legend: {
    x: 0,
    y: 1,
    bgcolor: 'rgba(255, 255, 255, 0.8)',
    bordercolor: '#e5e7eb',
    borderwidth: 1,
  },
  hovermode: 'closest',
}));

const plotlyConfig = computed<Partial<Config>>(() => ({
  responsive: true,
  displayModeBar: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d'],
  toImageButtonOptions: {
    format: 'png',
    filename: '3d_plot',
    height: 1000,
    width: 1000,
    scale: 2,
  },
}));
</script>

<template>
  <ClientOnly>
  <BaseChart
    :data="plotlyData"
    :layout="plotlyLayout"
    :config="plotlyConfig"
    :loading="loading"
    :error="error"
    :height="height"
  />
  </ClientOnly>
</template>
