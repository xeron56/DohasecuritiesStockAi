import { provideHttpClient } from '@angular/common/http';
import { bootstrapApplication } from '@angular/platform-browser';
import { AppComponent } from './app/app.component';
import { OpportunityScreenerComponent } from './app/opportunity-screener.component';

const rootComponent =
  new URLSearchParams(window.location.search).get('view') === 'opportunities'
    ? OpportunityScreenerComponent
    : AppComponent;

bootstrapApplication(rootComponent, {
  providers: [provideHttpClient()],
}).catch((error: unknown) => console.error(error));
