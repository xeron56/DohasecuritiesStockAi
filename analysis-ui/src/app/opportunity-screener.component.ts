import { CommonModule } from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  computed,
  signal,
} from '@angular/core';
import { Title } from '@angular/platform-browser';

import {
  OpportunityCandidate,
  OpportunityFactorScores,
  OpportunityScanResult,
} from './opportunity-screener.model';
import { OpportunityScreenerService } from './opportunity-screener.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './opportunity-screener.component.html',
  styleUrl: './opportunity-screener.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class OpportunityScreenerComponent {
  readonly result = signal<OpportunityScanResult | null>(null);
  readonly loading = signal(true);
  readonly errorMessage = signal('');
  readonly expandedSymbol = signal<string | null>(null);
  readonly candidates = computed<OpportunityCandidate[]>(
    () => this.result()?.candidates ?? [],
  );

  readonly factorLabels: Record<keyof OpportunityFactorScores, string> = {
    quality_growth: 'Quality & growth',
    valuation: 'Valuation',
    financial_safety: 'Financial safety',
    momentum: 'Value-trap check',
    underfollowed: 'Under-followed',
    data_quality: 'Data quality',
  };

  constructor(
    private readonly service: OpportunityScreenerService,
    title: Title,
  ) {
    title.setTitle('DSE Long-term Opportunity Research');
    const scanId = new URLSearchParams(window.location.search).get('run')?.trim();
    const request = scanId ? this.service.getScan(scanId) : this.service.getLatest();
    request.subscribe({
      next: (result) => {
        this.result.set(result);
        this.loading.set(false);
      },
      error: () => {
        this.errorMessage.set(
          'No saved opportunity scan is available. Run dohasecuritiesstockai-opportunities first.',
        );
        this.loading.set(false);
      },
    });
  }

  toggle(symbol: string): void {
    this.expandedSymbol.update((current) =>
      current === symbol ? null : symbol,
    );
  }

  factorEntries(
    factors: OpportunityFactorScores,
  ): { key: keyof OpportunityFactorScores; label: string; value: number }[] {
    return (Object.keys(this.factorLabels) as (keyof OpportunityFactorScores)[]).map(
      (key) => ({ key, label: this.factorLabels[key], value: factors[key] }),
    );
  }

  taka(value: number | null): string {
    return value === null
      ? '—'
      : `৳${value.toLocaleString('en-BD', { maximumFractionDigits: 2 })}`;
  }

  metric(value: number | null, suffix = ''): string {
    return value === null ? '—' : `${value.toFixed(1)}${suffix}`;
  }

  signed(value: number | null): string {
    if (value === null) return '—';
    return `${value > 0 ? '+' : ''}${value.toFixed(1)}%`;
  }

  labelClass(label: string): string {
    return label.toLowerCase().replaceAll(' ', '-');
  }
}
