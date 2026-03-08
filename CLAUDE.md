# CellBioStats

## Architecture
- **Backend**: FastAPI in `backend/` — `main.py` is the entry point
- **Frontend**: Static HTML/CSS/JS in `frontend/` — served by FastAPI
- **No build tools**: Tailwind CSS and Plotly.js via CDN

## Running locally
```
uvicorn backend.main:app --reload
```

## Key conventions
- Statistical logic lives in `backend/stats_engine.py` — do not modify statistical methods without domain expertise
- Plot generation in `backend/plot_engine.py` — mirrors the original SuperPlot logic exactly
- Frontend is a wizard (upload -> configure -> results), not a dashboard
- Sessions stored in-memory (dict keyed by UUID) — no database
- All API routes prefixed with `/api/`

## Dependencies
- See `requirements.txt` — FastAPI, uvicorn, pandas, plotly, scipy, statsmodels, kaleido
- Frontend: Tailwind CSS (CDN), Plotly.js (CDN), Inter font (Google Fonts)

## Deployment
- Render (see `render.yaml`)
- Auto-deploys from GitHub main branch
