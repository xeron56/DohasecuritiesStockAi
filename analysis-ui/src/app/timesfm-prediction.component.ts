import { CommonModule } from '@angular/common';
import {
  AfterViewInit,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  OnDestroy,
  OnInit,
  ViewChild,
  computed,
  signal,
} from '@angular/core';
import { Title } from '@angular/platform-browser';

import { Client, IMessage, IStompSocket } from '@stomp/stompjs';
import {
  ColorType,
  IChartApi,
  ISeriesApi,
  LineStyle,
  UTCTimestamp,
  createChart,
} from 'lightweight-charts';
import SockJS from 'sockjs-client';
import { Subscription } from 'rxjs';

import {
  BacktestPoint,
  LiveForecastMatch,
  LiveStockUpdate,
  TimesFmPredictionResult,
} from './timesfm-prediction.model';
import { TimesFmPredictionService } from './timesfm-prediction.service';

@Component({
  selector: 'app-timesfm-prediction',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './timesfm-prediction.component.html',
  styleUrl: './timesfm-prediction.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class TimesFmPredictionComponent
  implements OnInit, AfterViewInit, OnDestroy
{
  readonly result = signal<TimesFmPredictionResult | null>(null);
  readonly loading = signal(true);
  readonly errorMessage = signal('');
  readonly liveStatus = signal<'connecting' | 'connected' | 'offline'>(
    'connecting',
  );
  readonly livePrice = signal<number | null>(null);
  readonly liveMatches = signal<LiveForecastMatch[]>([]);
  readonly latestLiveMatch = computed(() => this.liveMatches().at(-1) ?? null);
  readonly recentBacktest = computed(() => this.result()?.backtest.slice(-12) ?? []);
  readonly subscriptions: Subscription[] = [];

  private chart: IChartApi | null = null;
  private liveSeries: ISeriesApi<'Line'> | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private stompClient: Client | null = null;
  private chartReady = false;

  @ViewChild('predictionChart')
  chartContainer?: ElementRef<HTMLDivElement>;

  constructor(
    private readonly predictionService: TimesFmPredictionService,
    private readonly titleService: Title,
  ) {}

  ngOnInit(): void {
    this.titleService.setTitle('DSE Stock Forecast');
    const query = new URLSearchParams(window.location.search);
    const runId = query.get('run')?.trim();
    const symbol = query.get('symbol')?.trim().toUpperCase();
    const resolution = query.get('resolution')?.trim();
    const request = runId
      ? this.predictionService.getPrediction(runId)
      : this.predictionService.getLatest(symbol, resolution);
    this.subscriptions.push(
      request.subscribe({
        next: (result) => {
          this.result.set(result);
          this.titleService.setTitle(`${result.model.name} · ${result.symbol}`);
          this.loading.set(false);
          this.errorMessage.set('');
          setTimeout(() => this.renderChart());
          this.connectLiveFeed(result);
        },
        error: () => {
          this.loading.set(false);
          this.errorMessage.set(
            'Prediction data is not available. Run a forecasting command first.',
          );
          this.liveStatus.set('offline');
        },
      }),
    );
  }

  ngAfterViewInit(): void {
    this.chartReady = true;
    this.renderChart();
  }

  ngOnDestroy(): void {
    this.subscriptions.forEach((subscription) => subscription.unsubscribe());
    void this.stompClient?.deactivate();
    this.resizeObserver?.disconnect();
    this.chart?.remove();
  }

  price(value: number | null): string {
    return value === null
      ? '—'
      : `৳${value.toLocaleString('en-BD', {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        })}`;
  }

  percent(value: number | null): string {
    return value === null ? '—' : `${value.toFixed(2)}%`;
  }

  signedPercent(value: number | null): string {
    if (value === null) return '—';
    return `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;
  }

  trackBacktest(_: number, point: BacktestPoint): string {
    return point.time;
  }

  private renderChart(): void {
    const result = this.result();
    const container = this.chartContainer?.nativeElement;
    if (!result || !container || !this.chartReady) return;

    this.resizeObserver?.disconnect();
    this.chart?.remove();
    this.chart = createChart(container, {
      autoSize: true,
      height: 520,
      layout: {
        background: { type: ColorType.Solid, color: '#111827' },
        textColor: '#9ca3af',
        fontFamily: 'Inter, ui-sans-serif, sans-serif',
      },
      grid: {
        vertLines: { color: '#1f2937' },
        horzLines: { color: '#1f2937' },
      },
      crosshair: {
        vertLine: { color: '#6b7280', labelBackgroundColor: '#374151' },
        horzLine: { color: '#6b7280', labelBackgroundColor: '#374151' },
      },
      rightPriceScale: { borderColor: '#374151' },
      timeScale: {
        borderColor: '#374151',
        timeVisible: result.data.server_resolution !== '1D',
        secondsVisible: false,
      },
    });

    const candles = this.chart.addCandlestickSeries({
      upColor: '#34d399',
      downColor: '#f87171',
      borderVisible: false,
      wickUpColor: '#34d399',
      wickDownColor: '#f87171',
    });
    candles.setData(
      result.history.map((point) => ({
        time: this.chartTime(point.time),
        open: point.open,
        high: point.high,
        low: point.low,
        close: point.close,
      })),
    );

    const predicted = this.chart.addLineSeries({
      color: '#f59e0b',
      lineWidth: 2,
      title: 'Held-out prediction',
      priceLineVisible: false,
    });
    predicted.setData(
      result.backtest.map((point) => ({
        time: this.chartTime(point.time),
        value: point.predicted,
      })),
    );

    const lower = this.chart.addLineSeries({
      color: 'rgba(245, 158, 11, 0.48)',
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      title: 'Q10',
      priceLineVisible: false,
      lastValueVisible: false,
    });
    const upper = this.chart.addLineSeries({
      color: 'rgba(245, 158, 11, 0.48)',
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      title: 'Q90',
      priceLineVisible: false,
      lastValueVisible: false,
    });
    lower.setData(
      result.backtest.map((point) => ({
        time: this.chartTime(point.time),
        value: point.q10,
      })),
    );
    upper.setData(
      result.backtest.map((point) => ({
        time: this.chartTime(point.time),
        value: point.q90,
      })),
    );

    const future = this.chart.addLineSeries({
      color: '#a78bfa',
      lineWidth: 3,
      lineStyle: LineStyle.Dashed,
      title: 'Future forecast',
      priceLineVisible: false,
    });
    const futureStart = result.history.at(-1);
    future.setData([
      ...(futureStart
        ? [
            {
              time: this.chartTime(futureStart.time),
              value: futureStart.close,
            },
          ]
        : []),
      ...result.future.map((point) => ({
        time: this.chartTime(point.time),
        value: point.predicted,
      })),
    ]);

    this.liveSeries = this.chart.addLineSeries({
      color: '#22d3ee',
      lineWidth: 2,
      title: 'Live actual',
      priceLineVisible: true,
      lastValueVisible: true,
    });
    this.chart.timeScale().fitContent();
    this.resizeObserver = new ResizeObserver(() => this.chart?.timeScale().fitContent());
    this.resizeObserver.observe(container);
  }

  private connectLiveFeed(result: TimesFmPredictionResult): void {
    if (!result.live_feed.enabled) {
      this.liveStatus.set('offline');
      return;
    }
    void this.stompClient?.deactivate();
    this.liveStatus.set('connecting');
    const socket = new SockJS(result.live_feed.url);
    this.stompClient = new Client({
      webSocketFactory: () => socket as unknown as IStompSocket,
      reconnectDelay: 5000,
      heartbeatIncoming: 10_000,
      heartbeatOutgoing: 10_000,
    });
    this.stompClient.onConnect = () => {
      this.liveStatus.set('connected');
      this.stompClient?.subscribe(result.live_feed.topic, (message) =>
        this.handleLiveMessage(message, result),
      );
    };
    this.stompClient.onWebSocketClose = () => this.liveStatus.set('offline');
    this.stompClient.onWebSocketError = () => this.liveStatus.set('offline');
    this.stompClient.onStompError = () => this.liveStatus.set('offline');
    this.stompClient.activate();
  }

  private handleLiveMessage(
    message: IMessage,
    result: TimesFmPredictionResult,
  ): void {
    let payload: unknown;
    try {
      payload = JSON.parse(message.body) as unknown;
    } catch {
      return;
    }
    const updates = Array.isArray(payload) ? payload : [payload];
    const update = updates.find(
      (candidate): candidate is LiveStockUpdate =>
        this.isLiveStockUpdate(candidate) &&
        candidate.stock_code.split("'")[0].toUpperCase() === result.symbol,
    );
    if (!update) return;

    const actual = Number(update.ltp);
    if (!Number.isFinite(actual) || actual <= 0 || !result.future.length) return;
    const now = Date.now();
    const target = result.future.reduce((closest, point) =>
      Math.abs(Date.parse(point.time) - now) <
      Math.abs(Date.parse(closest.time) - now)
        ? point
        : closest,
    );
    const absoluteError = Math.abs(actual - target.predicted);
    const accuracyPercent = Math.max(
      0,
      100 - (200 * absoluteError) / (Math.abs(actual) + Math.abs(target.predicted)),
    );
    this.livePrice.set(actual);
    this.liveMatches.update((matches) => [
      ...matches.slice(-49),
      {
        observedAt: new Date(),
        targetTime: target.time,
        actual,
        predicted: target.predicted,
        absoluteError,
        accuracyPercent,
      },
    ]);
    this.liveSeries?.update({ time: this.chartTime(target.time), value: actual });
  }

  private isLiveStockUpdate(value: unknown): value is LiveStockUpdate {
    if (typeof value !== 'object' || value === null) return false;
    const candidate = value as Partial<LiveStockUpdate>;
    return (
      typeof candidate.stock_code === 'string' &&
      Number.isFinite(Number(candidate.ltp))
    );
  }

  private chartTime(value: string): UTCTimestamp {
    return Math.floor(Date.parse(value) / 1000) as UTCTimestamp;
  }
}
