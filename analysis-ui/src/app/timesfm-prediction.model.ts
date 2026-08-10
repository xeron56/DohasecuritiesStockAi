export interface PredictionCandle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  segment: 'context' | 'holdout';
}

export interface ForecastPoint {
  time: string;
  predicted: number;
  q10: number;
  q50: number;
  q90: number;
}

export interface BacktestPoint extends ForecastPoint {
  actual: number;
  error: number;
  absolute_error: number;
  absolute_percentage_error: number | null;
}

export interface AccuracyMetrics {
  accuracy_score: number;
  accuracy_definition: string;
  mae: number;
  rmse: number;
  mape_percent: number | null;
  smape_percent: number;
  r_squared: number | null;
  directional_accuracy_percent: number | null;
  interval_80_coverage_percent: number;
  naive_mae: number;
  skill_vs_naive_percent: number | null;
}

export interface PredictionModelMetadata {
  name: string;
  version: string;
  checkpoint: string;
  parameters: number;
  backend: 'torch';
  device: string;
  gpu_name: string | null;
  max_context: number;
  max_horizon: number;
  recursive_chunks: number;
}

export interface PredictionDataMetadata {
  vendor: string;
  endpoint: string;
  symbol: string;
  requested_resolution: string;
  server_resolution: string;
  resolution_label: string;
  first_timestamp: string;
  last_timestamp: string;
  total_points: number;
  context_points: number;
  holdout_points: number;
  split_ratio: number;
  target: 'close';
}

export interface LiveFeedMetadata {
  enabled: boolean;
  transport: 'sockjs_stomp';
  url: string;
  topic: string;
  stock_code: string;
  note: string;
}

export interface TimesFmPredictionResult {
  schema_version: '1.0';
  run_id: string;
  generated_at: string;
  symbol: string;
  currency: string;
  model: PredictionModelMetadata;
  data: PredictionDataMetadata;
  metrics: AccuracyMetrics;
  history: PredictionCandle[];
  backtest: BacktestPoint[];
  future: ForecastPoint[];
  live_feed: LiveFeedMetadata;
  disclaimer: string;
}

export interface LiveStockUpdate {
  stock_code: string;
  ltp: number;
  volume?: number;
  value?: number;
  trades?: number;
}

export interface LiveForecastMatch {
  observedAt: Date;
  targetTime: string;
  actual: number;
  predicted: number;
  absoluteError: number;
  accuracyPercent: number;
}
