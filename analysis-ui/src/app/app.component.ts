import { CommonModule } from '@angular/common';
import {
  AfterViewInit,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  ElementRef,
  OnDestroy,
  OnInit,
  ViewChild,
  signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Title } from '@angular/platform-browser';
import {
  ArcElement,
  Chart,
  DoughnutController,
  Tooltip,
} from 'chart.js';
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
  readonly loadingStocks = signal<boolean>(true);
  readonly loadingAnalysis = signal<boolean>(false);
  readonly jobMessage = signal<string>('');
  readonly errorMessage = signal<string>('');
  readonly fullReportOpen = signal<boolean>(false);
  readonly subscriptions: Subscription[] = [];

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
    this.titleService.setTitle('DSE AI Stock Analysis');
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
    this.fullReportOpen.set(false);
    this.loadLatest(this.selectedSymbol);
  }

  runAnalysis(force = false): void {
    this.loadingAnalysis.set(true);
    this.errorMessage.set('');
    this.jobMessage.set(
      this.language() === 'bn'
        ? 'বিশ্লেষণ সারিতে যোগ করা হচ্ছে…'
        : 'Queueing analysis…',
    );
    this.subscriptions.push(
      this.stockAnalysisService.createAnalysis(this.selectedSymbol, force).subscribe({
        next: (job) => this.pollJob(job),
        error: () => this.handleError('The analysis job could not be started.'),
      }),
    );
  }

  toggleFullReport(): void {
    this.fullReportOpen.update((open) => !open);
    if (!this.fullReportOpen()) return;
    setTimeout(() => {
      document.getElementById('full-analysis')?.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      });
    });
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

  formatTaka(value: number | null): string {
    return value === null
      ? '—'
      : `৳${value.toLocaleString('en-BD', { maximumFractionDigits: 1 })}`;
  }

  rawState(report: StockAnalysis): string {
    return JSON.stringify(report.agent_reports.raw_state, null, 2);
  }

  hasAgentEvidence(report: StockAnalysis): boolean {
    const evidence = report.agent_reports;
    return Boolean(
      evidence.market_report ||
        evidence.news_report ||
        evidence.fundamentals_report ||
        evidence.investment_plan ||
        evidence.final_trade_decision ||
        Object.keys(evidence.raw_state).length,
    );
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
          error: () =>
            this.handleError('The analysis job status could not be checked.'),
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
    this.scoreChart?.destroy();
    this.scoreChart = new Chart(this.scoreCanvas.nativeElement, {
      type: 'doughnut',
      data: {
        labels: ['Score', 'Remaining'],
        datasets: [
          {
            data: [report.fundamental_score, 100 - report.fundamental_score],
            backgroundColor: ['#527c9e', '#ebe6dc'],
            borderWidth: 0,
            hoverOffset: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '78%',
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
