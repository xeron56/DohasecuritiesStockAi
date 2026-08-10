import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../environments/environment';
import { AnalysisJob, StockAnalysis, StockOption } from './stock-analysis.model';

@Injectable({ providedIn: 'root' })
export class StockAnalysisService {
  private readonly apiUrl = environment.analysisApiUrl;

  constructor(private readonly http: HttpClient) {}

  getStocks(): Observable<StockOption[]> {
    return this.http.get<StockOption[]>(`${this.apiUrl}/stocks`);
  }

  getLatest(symbol: string): Observable<StockAnalysis> {
    return this.http.get<StockAnalysis>(
      `${this.apiUrl}/analyses/${encodeURIComponent(symbol)}/latest`,
    );
  }

  getAnalysis(symbol: string, analysisDate: string): Observable<StockAnalysis> {
    return this.http.get<StockAnalysis>(
      `${this.apiUrl}/analyses/${encodeURIComponent(symbol)}/${encodeURIComponent(analysisDate)}`,
    );
  }

  createAnalysis(symbol: string, force = false): Observable<AnalysisJob> {
    return this.http.post<AnalysisJob>(`${this.apiUrl}/analyses`, {
      symbol,
      force,
    });
  }

  getJob(jobId: string): Observable<AnalysisJob> {
    return this.http.get<AnalysisJob>(
      `${this.apiUrl}/analyses/jobs/${encodeURIComponent(jobId)}`,
    );
  }
}
