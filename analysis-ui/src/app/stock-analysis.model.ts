export type AnalysisLanguage = 'en' | 'bn';
export type FactorStatus = 'positive' | 'caution' | 'negative' | 'neutral';
export type JobStatus = 'queued' | 'running' | 'completed' | 'failed';

export interface BilingualText {
  en: string;
  bn: string;
}

export interface StockOption {
  symbol: string;
  name: string;
  sector: string;
  latest_price: number | null;
  change_percent: number | null;
}

export interface MarketSnapshot {
  latest_price: number;
  change: number | null;
  change_percent: number | null;
  previous_close: number | null;
  fifty_two_week_low: number | null;
  fifty_two_week_high: number | null;
  as_of: string;
}

export interface ScoreMetric {
  key: string;
  label: BilingualText;
  display_value: string;
  score: number;
}

export interface FactorCard {
  key: string;
  status: FactorStatus;
  title: BilingualText;
  subtitle: BilingualText;
  explanation: BilingualText;
  metrics: ScoreMetric[];
}

export interface ValuationMethod {
  key: string;
  label: BilingualText;
  value: number | null;
  available: boolean;
}

export interface ValuationSummary {
  verdict: 'looks_cheap' | 'fair' | 'looks_expensive' | 'insufficient_data';
  verdict_label: BilingualText;
  current_price: number;
  rough_estimate: number | null;
  fair_range_low: number | null;
  fair_range_high: number | null;
  confidence: 'low' | 'medium' | 'high';
  summary: BilingualText;
  methods: ValuationMethod[];
}

export interface ReportSection {
  key: string;
  title: BilingualText;
  summary: BilingualText;
  bullets: BilingualText[];
}

export interface AgentReports {
  market_report: string;
  news_report: string;
  fundamentals_report: string;
  investment_plan: string;
  final_trade_decision: string;
  raw_state: Record<string, unknown>;
}

export interface StockAnalysis {
  schema_version: '1.0';
  analysis_id: string;
  symbol: string;
  company_name: string;
  sector: string;
  analysis_date: string;
  generated_at: string;
  market: MarketSnapshot;
  fundamental_score: number;
  score_label: BilingualText;
  headline: BilingualText;
  takeaways: BilingualText[];
  in_depth_title: BilingualText;
  in_depth_snippet: BilingualText;
  valuation: ValuationSummary;
  factors: FactorCard[];
  report_sections: ReportSection[];
  agent_reports: AgentReports;
  disclaimer: BilingualText;
}

export interface AnalysisJob {
  job_id: string;
  symbol: string;
  analysis_date: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  analysis_url: string | null;
  message: string;
}
