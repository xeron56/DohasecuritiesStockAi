import { CommonModule } from '@angular/common';
import {
  AfterViewInit,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  computed,
  ElementRef,
  OnDestroy,
  OnInit,
  signal,
  ViewChild,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Title } from '@angular/platform-browser';
import { ArcElement, Chart, DoughnutController, Tooltip } from 'chart.js';
import { Subscription, switchMap, takeWhile, timer } from 'rxjs';

import {
  AnalysisJob,
  AnalysisLanguage,
  BilingualText,
  FactorStatus,
  StockAnalysis,
  StockOption,
} from './stock-analysis.model';
import { StockAnalysisService } from './stock-analysis.service';
import { TimesFmPredictionComponent } from './timesfm-prediction.component';

Chart.register(DoughnutController, ArcElement, Tooltip);

type DecisionTone = FactorStatus;

interface ReportDocument {
  key: string;
  title: string;
  role: string;
  icon: string;
  content: string;
  html: string;
  tone: DecisionTone;
  open: boolean;
}

interface ReportGroup {
  key: string;
  kicker: string;
  title: string;
  description: string;
  documents: ReportDocument[];
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, TimesFmPredictionComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AppComponent implements OnInit, AfterViewInit, OnDestroy {
  readonly predictionMode =
    new URLSearchParams(window.location.search).get('view') === 'timesfm';
  readonly Math = Math;
  readonly stocks = signal<StockOption[]>([]);
  readonly analysis = signal<StockAnalysis | null>(null);
  readonly language = signal<AnalysisLanguage>('en');
  readonly loadingStocks = signal(true);
  readonly loadingAnalysis = signal(false);
  readonly jobMessage = signal('');
  readonly errorMessage = signal('');
  readonly subscriptions: Subscription[] = [];

  readonly reportGroups = computed<ReportGroup[]>(() => {
    const report = this.analysis();
    return report ? this.buildReportGroups(report) : [];
  });
  readonly reportDocumentCount = computed(() =>
    this.reportGroups().reduce(
      (total, group) => total + group.documents.length,
      0,
    ),
  );

  selectedSymbol = 'GP';
  private scoreCanvas: ElementRef<HTMLCanvasElement> | null = null;
  private scoreChart: Chart<'doughnut'> | null = null;

  @ViewChild('scoreCanvas')
  set scoreCanvasRef(canvas: ElementRef<HTMLCanvasElement> | undefined) {
    this.scoreCanvas = canvas ?? null;
    this.renderScoreChart();
  }

  constructor(
    private readonly stockAnalysisService: StockAnalysisService,
    private readonly titleService: Title,
    private readonly cdr: ChangeDetectorRef,
  ) {}

  ngOnInit(): void {
    if (this.predictionMode) return;
    this.titleService.setTitle('DSE Investment Research Dossier');
    const query = new URLSearchParams(window.location.search);
    const requestedSymbol = query.get('symbol')?.trim().toUpperCase();
    const requestedDate = query.get('date')?.trim();
    if (requestedSymbol) this.selectedSymbol = requestedSymbol;
    this.loadStocks();
    if (requestedDate && /^\d{4}-\d{2}-\d{2}$/.test(requestedDate)) {
      this.loadAnalysis(this.selectedSymbol, requestedDate);
    } else {
      this.loadLatest(this.selectedSymbol);
    }
  }

  ngAfterViewInit(): void {
    this.renderScoreChart();
  }

  ngOnDestroy(): void {
    this.scoreChart?.destroy();
    this.subscriptions.forEach((subscription) => subscription.unsubscribe());
  }

  setLanguage(language: AnalysisLanguage): void {
    this.language.set(language);
    document.documentElement.lang = language;
  }

  copy(value: BilingualText): string {
    return value[this.language()];
  }

  selectStock(): void {
    this.loadLatest(this.selectedSymbol);
  }

  runAnalysis(force = false): void {
    this.loadingAnalysis.set(true);
    this.errorMessage.set('');
    this.jobMessage.set(
      this.language() === 'bn'
        ? 'বিশ্লেষণ সারিতে যোগ করা হচ্ছে…'
        : 'Queueing the multi-agent analysis…',
    );
    this.subscriptions.push(
      this.stockAnalysisService.createAnalysis(this.selectedSymbol, force).subscribe({
        next: (job) => this.pollJob(job),
        error: () => this.handleError('The analysis job could not be started.'),
      }),
    );
  }

  factorStatus(status: FactorStatus): string {
    const labels: Record<FactorStatus, BilingualText> = {
      positive: { en: 'Positive', bn: 'ইতিবাচক' },
      caution: { en: 'Watch', bn: 'নজর রাখুন' },
      negative: { en: 'Concern', bn: 'উদ্বেগ' },
      neutral: { en: 'Neutral', bn: 'নিরপেক্ষ' },
    };
    return this.copy(labels[status]);
  }

  getSectionStatus(sectionKey: string): FactorStatus {
    const report = this.analysis();
    if (!report) return 'neutral';
    const mapping: Record<string, string> = {
      earnings: 'profitability',
      financial_health: 'financial_health',
      business: 'business_quality',
      valuation: 'valuation',
      dividend: 'dividend',
    };
    return (
      report.factors.find(
        (factor) => factor.key === (mapping[sectionKey] || sectionKey),
      )?.status ?? 'neutral'
    );
  }

  metricStatus(score: number): FactorStatus {
    if (score >= 7) return 'positive';
    if (score < 4) return 'negative';
    return 'caution';
  }

  factorAverage(report: StockAnalysis, key: string): number {
    const metrics = report.factors.find((factor) => factor.key === key)?.metrics ?? [];
    if (!metrics.length) return 0;
    return metrics.reduce((total, metric) => total + metric.score, 0) / metrics.length;
  }

  scoreTone(score: number): DecisionTone {
    if (score >= 70) return 'positive';
    if (score >= 45) return 'caution';
    return 'negative';
  }

  finalDecision(report: StockAnalysis): string {
    const source =
      report.agent_reports.final_trade_decision ||
      this.stateString(report, 'final_trade_decision');
    return (
      this.extractDecision(source) ??
      report.ai_research?.trader_report.rating ??
      'Review'
    );
  }

  decisionTone(report: StockAnalysis): DecisionTone {
    const decision = this.finalDecision(report).toLowerCase();
    if (/buy|overweight|accumulate/.test(decision)) return 'positive';
    if (/sell|underweight|avoid/.test(decision)) return 'negative';
    if (/hold|watch|review/.test(decision)) return 'caution';
    return 'neutral';
  }

  formatTaka(value: number | null): string {
    return value === null
      ? '—'
      : `৳${value.toLocaleString('en-BD', { maximumFractionDigits: 1 })}`;
  }

  signedPercent(value: number | null): string {
    if (value === null) return '—';
    return `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;
  }

  fairValueGap(report: StockAnalysis): number | null {
    const estimate = report.valuation.rough_estimate;
    const price = report.valuation.current_price;
    if (estimate === null || price <= 0) return null;
    return ((estimate / price) - 1) * 100;
  }

  rangePosition(report: StockAnalysis): number {
    const low = report.market.fifty_two_week_low;
    const high = report.market.fifty_two_week_high;
    if (low === null || high === null || high <= low) return 50;
    return this.clamp(((report.market.latest_price - low) / (high - low)) * 100, 0, 100);
  }

  valuationScale(report: StockAnalysis): {
    current: number;
    low: number;
    high: number;
  } {
    const current = report.valuation.current_price;
    const fairLow = report.valuation.fair_range_low ?? current;
    const fairHigh = report.valuation.fair_range_high ?? current;
    const scaleLow = Math.min(current, fairLow) * 0.92;
    const scaleHigh = Math.max(current, fairHigh) * 1.08;
    const spread = scaleHigh - scaleLow || 1;
    const position = (value: number) =>
      this.clamp(((value - scaleLow) / spread) * 100, 2, 98);
    return {
      current: position(current),
      low: position(fairLow),
      high: position(fairHigh),
    };
  }

  hasAgentEvidence(report: StockAnalysis): boolean {
    return this.buildReportGroups(report).some((group) => group.documents.length > 0);
  }

  rawState(report: StockAnalysis): string {
    return JSON.stringify(report.agent_reports.raw_state, null, 2);
  }

  jumpTo(id: string): void {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  renderMarkdown(markdown: string): string {
    const escaped = this.escapeHtml(markdown || '').replace(/\r\n?/g, '\n');
    const lines = escaped.split('\n');
    const output: string[] = [];
    let index = 0;

    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) {
        index += 1;
        continue;
      }

      if (/^```/.test(line.trim())) {
        const code: string[] = [];
        index += 1;
        while (index < lines.length && !/^```/.test(lines[index].trim())) {
          code.push(lines[index]);
          index += 1;
        }
        index += 1;
        output.push(`<pre><code>${code.join('\n')}</code></pre>`);
        continue;
      }

      const heading = line.match(/^(#{1,6})\s+(.+)$/);
      if (heading) {
        const level = Math.min(5, heading[1].length + 1);
        output.push(`<h${level}>${this.inlineMarkdown(heading[2])}</h${level}>`);
        index += 1;
        continue;
      }

      if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)) {
        output.push('<hr>');
        index += 1;
        continue;
      }

      if (
        line.includes('|') &&
        index + 1 < lines.length &&
        /^\s*\|?\s*:?-{3,}/.test(lines[index + 1])
      ) {
        const headers = this.tableCells(line);
        index += 2;
        const rows: string[][] = [];
        while (index < lines.length && lines[index].includes('|') && lines[index].trim()) {
          rows.push(this.tableCells(lines[index]));
          index += 1;
        }
        output.push(
          `<div class="report-table-wrap"><table><thead><tr>${headers
            .map((cell) => `<th>${this.inlineMarkdown(cell)}</th>`)
            .join('')}</tr></thead><tbody>${rows
            .map(
              (row) =>
                `<tr>${row
                  .map((cell) => `<td>${this.inlineMarkdown(cell)}</td>`)
                  .join('')}</tr>`,
            )
            .join('')}</tbody></table></div>`,
        );
        continue;
      }

      if (/^\s*>\s?/.test(line)) {
        const quote: string[] = [];
        while (index < lines.length && /^\s*>\s?/.test(lines[index])) {
          quote.push(lines[index].replace(/^\s*>\s?/, ''));
          index += 1;
        }
        output.push(`<blockquote>${quote.map((item) => this.inlineMarkdown(item)).join('<br>')}</blockquote>`);
        continue;
      }

      const unordered = /^\s*[-*+]\s+/.test(line);
      const ordered = /^\s*\d+[.)]\s+/.test(line);
      if (unordered || ordered) {
        const tag = ordered ? 'ol' : 'ul';
        const items: string[] = [];
        const pattern = ordered ? /^\s*\d+[.)]\s+/ : /^\s*[-*+]\s+/;
        while (index < lines.length && pattern.test(lines[index])) {
          items.push(lines[index].replace(pattern, ''));
          index += 1;
        }
        output.push(
          `<${tag}>${items
            .map((item) => `<li>${this.inlineMarkdown(item)}</li>`)
            .join('')}</${tag}>`,
        );
        continue;
      }

      const paragraph: string[] = [line.trim()];
      index += 1;
      while (
        index < lines.length &&
        lines[index].trim() &&
        !this.isMarkdownBoundary(lines, index)
      ) {
        paragraph.push(lines[index].trim());
        index += 1;
      }
      output.push(`<p>${paragraph.map((item) => this.inlineMarkdown(item)).join('<br>')}</p>`);
    }

    return output.join('');
  }

  private buildReportGroups(report: StockAnalysis): ReportGroup[] {
    const state = this.asRecord(report.agent_reports.raw_state);
    const research = this.asRecord(state['investment_debate_state']);
    const risk = this.asRecord(state['risk_debate_state']);
    const direct = report.agent_reports;
    const document = (
      key: string,
      title: string,
      role: string,
      icon: string,
      content: unknown,
      tone: DecisionTone,
      open = false,
    ): ReportDocument | null => {
      const text = typeof content === 'string' ? content.trim() : '';
      return text
        ? {
            key,
            title,
            role,
            icon,
            content: text,
            html: this.renderMarkdown(text),
            tone,
            open,
          }
        : null;
    };
    const compact = (items: Array<ReportDocument | null>) =>
      items.filter((item): item is ReportDocument => item !== null);

    const groups: ReportGroup[] = [
      {
        key: 'analyst-desk',
        kicker: 'I · EVIDENCE COLLECTION',
        title: 'Analyst desk',
        description: 'Independent market, sentiment, news, and company-fundamentals reviews.',
        documents: compact([
          document('market', 'Market & technical report', 'Market Analyst', 'M', direct.market_report || state['market_report'], 'neutral'),
          document('sentiment', 'Market sentiment report', 'Sentiment Analyst', 'S', state['sentiment_report'], 'caution'),
          document('news', 'News & disclosure report', 'News Analyst', 'N', direct.news_report || state['news_report'], 'caution'),
          document('fundamentals', 'Fundamental report', 'Fundamentals Analyst', 'F', direct.fundamentals_report || state['fundamentals_report'], 'positive'),
        ]),
      },
      {
        key: 'research-debate',
        kicker: 'II · ADVERSARIAL RESEARCH',
        title: 'Bull vs bear debate',
        description: 'The upside case and downside case are kept separate before the research manager decides.',
        documents: compact([
          document('bull', 'Bull case', 'Bull Researcher', 'B+', research['bull_history'], 'positive'),
          document('bear', 'Bear case', 'Bear Researcher', 'B−', research['bear_history'], 'negative'),
          document('research-manager', 'Research manager decision', 'Research Manager', 'RM', research['judge_decision'] || direct.investment_plan || state['investment_plan'], 'caution', true),
        ]),
      },
      {
        key: 'trade-plan',
        kicker: 'III · EXECUTION PLAN',
        title: 'Trader plan',
        description: 'The research conclusion translated into an entry, sizing, timing, and monitoring plan.',
        documents: compact([
          document('trader', 'Transaction proposal', 'Trader', 'T', state['trader_investment_plan'] || state['trader_investment_decision'], 'caution', true),
        ]),
      },
      {
        key: 'risk-committee',
        kicker: 'IV–V · RISK & FINAL AUTHORITY',
        title: 'Risk committee and portfolio decision',
        description: 'Aggressive, conservative, and balanced risk views followed by the final portfolio-manager decision.',
        documents: compact([
          document('aggressive', 'Upside-oriented risk view', 'Aggressive Analyst', 'A', risk['aggressive_history'], 'positive'),
          document('conservative', 'Capital-preservation view', 'Conservative Analyst', 'C', risk['conservative_history'], 'negative'),
          document('neutral', 'Balanced risk view', 'Neutral Analyst', 'N', risk['neutral_history'], 'caution'),
          document('portfolio', 'Final portfolio decision', 'Portfolio Manager', 'PM', risk['judge_decision'] || direct.final_trade_decision || state['final_trade_decision'], this.decisionTone(report), true),
        ]),
      },
    ];

    return groups.filter((group) => group.documents.length > 0);
  }

  private extractDecision(content: string): string | null {
    if (!content) return null;
    const strongPatterns = [
      /(?:\*\*)?final\s+(?:transaction\s+)?(?:proposal|decision|recommendation|rating)(?:\*\*)?\s*[:\-]\s*(?:\*\*)?\s*(BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL|ACCUMULATE|AVOID)/gi,
      /(?:\*\*)?(?:rating|recommendation|decision)(?:\*\*)?\s*[:\-]\s*(?:\*\*)?\s*(BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL|ACCUMULATE|AVOID)/gi,
    ];
    for (const pattern of strongPatterns) {
      const matches = [...content.matchAll(pattern)];
      if (matches.length) return this.titleCase(matches[matches.length - 1][1]);
    }
    const fallback = [...content.matchAll(/\b(BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL|ACCUMULATE|AVOID)\b/gi)];
    return fallback.length ? this.titleCase(fallback[fallback.length - 1][1]) : null;
  }

  private titleCase(value: string): string {
    return value.charAt(0).toUpperCase() + value.slice(1).toLowerCase();
  }

  private stateString(report: StockAnalysis, key: string): string {
    const value = this.asRecord(report.agent_reports.raw_state)[key];
    return typeof value === 'string' ? value : '';
  }

  private asRecord(value: unknown): Record<string, unknown> {
    return value && typeof value === 'object' && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : {};
  }

  private clamp(value: number, low: number, high: number): number {
    return Math.max(low, Math.min(high, value));
  }

  private tableCells(line: string): string[] {
    return line
      .trim()
      .replace(/^\|/, '')
      .replace(/\|$/, '')
      .split('|')
      .map((cell) => cell.trim());
  }

  private isMarkdownBoundary(lines: string[], index: number): boolean {
    const line = lines[index];
    return (
      /^(#{1,6})\s+/.test(line) ||
      /^```/.test(line.trim()) ||
      /^\s*>\s?/.test(line) ||
      /^\s*[-*+]\s+/.test(line) ||
      /^\s*\d+[.)]\s+/.test(line) ||
      /^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line) ||
      (line.includes('|') &&
        index + 1 < lines.length &&
        /^\s*\|?\s*:?-{3,}/.test(lines[index + 1]))
    );
  }

  private inlineMarkdown(value: string): string {
    return value
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/__([^_]+)__/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*([^*]+)\*/g, '$1<em>$2</em>');
  }

  private escapeHtml(value: string): string {
    return value
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  private loadStocks(): void {
    this.loadingStocks.set(true);
    this.subscriptions.push(
      this.stockAnalysisService.getStocks().subscribe({
        next: (stocks) => {
          this.stocks.set(stocks);
          this.loadingStocks.set(false);
          this.cdr.markForCheck();
        },
        error: () => {
          this.loadingStocks.set(false);
          this.handleError('The DSE stock list could not be loaded.');
        },
      }),
    );
  }

  private loadLatest(symbol: string): void {
    this.loadAnalysis(symbol);
  }

  private loadAnalysis(symbol: string, analysisDate?: string): void {
    this.loadingAnalysis.set(true);
    this.errorMessage.set('');
    this.jobMessage.set('');
    const request = analysisDate
      ? this.stockAnalysisService.getAnalysis(symbol, analysisDate)
      : this.stockAnalysisService.getLatest(symbol);
    this.subscriptions.push(
      request.subscribe({
        next: (report) => this.applyAnalysis(report),
        error: () => {
          this.loadingAnalysis.set(false);
          this.analysis.set(null);
          this.errorMessage.set(
            this.language() === 'bn'
              ? 'এই শেয়ারের কোনো সম্পূর্ণ বিশ্লেষণ নেই। নতুন বিশ্লেষণ চালান।'
              : 'No completed analysis exists for this stock yet. Run a new analysis.',
          );
          this.cdr.markForCheck();
        },
      }),
    );
  }

  private pollJob(initialJob: AnalysisJob): void {
    this.jobMessage.set(initialJob.message);
    this.subscriptions.push(
      timer(0, 2000)
        .pipe(
          switchMap(() => this.stockAnalysisService.getJob(initialJob.job_id)),
          takeWhile(
            (job) => job.status === 'queued' || job.status === 'running',
            true,
          ),
        )
        .subscribe({
          next: (job) => {
            this.jobMessage.set(job.message);
            if (job.status === 'completed') this.loadLatest(job.symbol);
            if (job.status === 'failed') {
              this.handleError(job.message || 'The analysis failed.');
            }
            this.cdr.markForCheck();
          },
          error: () => this.handleError('The analysis job status could not be checked.'),
        }),
    );
  }

  private applyAnalysis(report: StockAnalysis): void {
    this.selectedSymbol = report.symbol;
    this.analysis.set(report);
    this.loadingAnalysis.set(false);
    this.jobMessage.set('');
    this.errorMessage.set('');
    const query = new URLSearchParams({
      symbol: report.symbol,
      date: report.analysis_date,
    });
    window.history.replaceState(null, '', `${window.location.pathname}?${query}`);
    this.cdr.markForCheck();
    setTimeout(() => this.renderScoreChart());
  }

  private renderScoreChart(): void {
    const report = this.analysis();
    if (!this.scoreCanvas || !report) return;
    const colors: Record<DecisionTone, string> = {
      positive: '#147d57',
      caution: '#bf741f',
      negative: '#c24949',
      neutral: '#64748b',
    };
    this.scoreChart?.destroy();
    this.scoreChart = new Chart(this.scoreCanvas.nativeElement, {
      type: 'doughnut',
      data: {
        labels: ['Score', 'Remaining'],
        datasets: [
          {
            data: [report.fundamental_score, 100 - report.fundamental_score],
            backgroundColor: [colors[this.scoreTone(report.fundamental_score)], '#e7e9e4'],
            borderWidth: 0,
            hoverOffset: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '80%',
        animation: { duration: 450 },
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false },
        },
      },
    });
  }

  private handleError(message: string): void {
    this.loadingAnalysis.set(false);
    this.jobMessage.set('');
    this.errorMessage.set(message);
    this.cdr.markForCheck();
  }
}
