import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

import { environment } from '../environments/environment';
import { OpportunityScanResult } from './opportunity-screener.model';

@Injectable({ providedIn: 'root' })
export class OpportunityScreenerService {
  private readonly apiUrl = environment.analysisApiUrl;

  constructor(private readonly http: HttpClient) {}

  getScan(scanId: string): Observable<OpportunityScanResult> {
    return this.http.get<OpportunityScanResult>(
      `${this.apiUrl}/opportunities/${encodeURIComponent(scanId)}`,
    );
  }

  getLatest(): Observable<OpportunityScanResult> {
    return this.http.get<OpportunityScanResult>(
      `${this.apiUrl}/opportunities/latest`,
    );
  }
}
