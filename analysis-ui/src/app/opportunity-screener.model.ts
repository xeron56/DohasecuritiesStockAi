export type OpportunityLabel =
  | 'Research first'
  | 'Watch'
  | 'Avoid'
  | 'Insufficient evidence';

export interface OpportunityFactorScores {
  quality_growth: number;
  valuation: number;
  financial_safety: number;
  momentum: number;
  underfollowed: number;
  data_quality: number;
}

export interface OpportunityMetrics {
  current_price: number | null;
  market_cap_raw: number | null;
  eps_ttm: number | null;
  nav_per_share: number | null;
  pe_ratio: number | null;
  price_to_book: number | null;
  roe_percent: number | null;
  debt_to_equity: number | null;
  dividend_yield_percent: number | null;
  director_holdings_percent: number | null;
  average_volume_20d: number | null;
  eps_growth_percent: number | null;
  nav_growth_percent: number | null;
  cash_conversion: number | null;
  twelve_month_return_percent: number | null;
  distance_from_52w_high_percent: number | null;
}

export interface OpportunityAIReview {
  verdict: OpportunityLabel;
  confidence: 'low' | 'medium' | 'high';
  thesis: string;
  what_market_may_be_missing: string;
  multi_year_path: string;
  valuation_discipline: string;
  catalysts: string[];
  risks: string[];
  checkpoints: string[];
}

export interface OpportunityCandidate {
  rank: number;
  symbol: string;
  company_name: string;
  sector: string;
  category: string;
  score: number;
  research_label: OpportunityLabel;
  factors: OpportunityFactorScores;
  metrics: OpportunityMetrics;
  why_it_ranked: string[];
  red_flags: string[];
  missing_evidence: string[];
  evidence_periods: Record<string, string>;
  ai_review: OpportunityAIReview | null;
}

export interface OpportunityMethodology {
  weights: Record<string, number>;
  initial_universe: number;
  eligible_universe: number;
  detailed_finalists: number;
  excluded_counts: Record<string, number>;
  notes: string[];
}

export interface OpportunityScanResult {
  schema_version: '1.0';
  scan_id: string;
  as_of: string;
  generated_at: string;
  horizon_years: number;
  ai_enabled: boolean;
  ai_provider: string | null;
  ai_model: string | null;
  candidates: OpportunityCandidate[];
  methodology: OpportunityMethodology;
  sources: { name: string; detail: string }[];
  disclaimer: string;
}
