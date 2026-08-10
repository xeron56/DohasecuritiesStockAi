import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../environments/environment';
import { TimesFmPredictionResult } from './timesfm-prediction.model';

@Injectable({ providedIn: 'root' })
export class TimesFmPredictionService {
  private readonly apiUrl = environment.analysisApiUrl;

  constructor(private readonly http: HttpClient) {}

  getPrediction(runId: string): Observable<TimesFmPredictionResult> {
    return this.http.get<TimesFmPredictionResult>(
      `${this.apiUrl}/predictions/${encodeURIComponent(runId)}`,
    );
  }

  getLatest(
    symbol?: string,
    resolution?: string,
  ): Observable<TimesFmPredictionResult> {
    let params = new HttpParams();
    if (symbol) params = params.set('symbol', symbol);
    if (resolution) params = params.set('resolution', resolution);
    return this.http.get<TimesFmPredictionResult>(
      `${this.apiUrl}/predictions/latest`,
      { params },
    );
  }
}
