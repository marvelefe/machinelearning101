// Visualization type definitions

export type VisualizationType =
  | 'scatter2d'
  | 'scatter3d'
  | 'scatter'
  | 'line'
  | 'bar'
  | 'histogram'
  | 'heatmap'
  | 'contour'
  | 'surface'
  | 'network'
  | 'tree'
  | 'loss-landscape'
  | 'confusion'
  | 'confusion-matrix'
  | 'roc-curve'
  | 'learning-curves'
  | 'cross-validation'
  | 'canvas-scatter'
  | 'policy';

export interface VisualizationConfig {
  type: VisualizationType;
  title?: string;
  width?: number;
  height?: string;
  showLegend?: boolean;
  showGrid?: boolean;
}

export interface ScatterPoint {
  x: number;
  y: number;
  label?: string | number;
}

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
  };
}

export interface LineSeries {
  x: number[];
  y: number[];
  name: string;
  color?: string;
  dash?: 'solid' | 'dot' | 'dash' | 'dashdot';
  width?: number;
  showMarkers?: boolean;
}

export interface HeatmapData {
  data: number[][];
  xLabels?: string[];
  yLabels?: string[];
  colorscale?: string | Array<[number, string]>;
  showScale?: boolean;
  showAnnotations?: boolean;
}

export interface Scatter3DPoint {
  x: number;
  y: number;
  z: number;
  label?: string | number;
}

export interface Scatter3DSeries {
  x: number[];
  y: number[];
  z: number[];
  name?: string;
  color?: string;
  mode?: 'markers' | 'lines' | 'lines+markers';
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

export interface CanvasScatterSeries {
  x: number[];
  y: number[];
  name: string;
  color: string;
  size?: number;
}

export interface VisualizationEventData {
  seriesIndex: number;
  pointIndex: number;
  x: number;
  y: number;
  z?: number;
}

export interface ConfusionMatrixData {
  matrix: number[][];
  classLabels?: string[];
  showMetrics?: boolean;
}

export interface ROCCurveData {
  fpr: number[];
  tpr: number[];
  auc?: number;
  showDiagonal?: boolean;
  showAUC?: boolean;
}

export interface LearningCurvesData {
  trainingSizes: number[];
  trainingScores: number[];
  validationScores: number[];
  metricName?: string;
  showIndicators?: boolean;
}

export interface CrossValidationData {
  foldScores: number[];
  metricName?: string;
  showStats?: boolean;
  showSplits?: boolean;
  numFolds?: number;
}
