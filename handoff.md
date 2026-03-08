# Handoff: CellBioStats Web App Transformation

## Status: Phase 1-3 COMPLETE, Phase 4 MOSTLY COMPLETE

## What Was Done

### Backend (COMPLETE)
- `app.py` removed (old Dash app)
- `backend/stats_engine.py` - extracted `calculate_hierarchical_stats`, `perform_statistical_analysis`, `create_stats_table` verbatim
- `backend/plot_engine.py` - extracted `create_interactive_superplot` verbatim
- `backend/models.py` - Pydantic schemas for API requests
- `backend/routes.py` - All 5 API endpoints implemented and tested
- `backend/main.py` - FastAPI app with CORS, static file serving

### Frontend (COMPLETE)
- `frontend/index.html` - Three-section wizard (Upload -> Configure -> Results) with Tailwind CSS
- `frontend/css/app.css` - Custom styles, transitions, spinner animations
- `frontend/js/api.js` - Fetch wrappers for all endpoints
- `frontend/js/upload.js` - Drag-and-drop + file browse
- `frontend/js/mapping.js` - Column dropdowns, data preview, customization controls
- `frontend/js/plot.js` - Plotly.js rendering
- `frontend/js/results.js` - Stats display, download handlers
- `frontend/js/app.js` - Wizard state machine

### Config (COMPLETE)
- `requirements.txt` - Updated (plotly>=5,<6, kaleido==0.2.1)
- `render.yaml` - Render deployment config
- `CLAUDE.md` - Project conventions
- `README.md` - Updated with new setup/deployment instructions

## API Endpoints (ALL TESTED AND PASSING)

| Endpoint | Status | Notes |
|----------|--------|-------|
| POST /api/upload | PASS | CSV + XLSX |
| POST /api/analyze | PASS | With paired, custom options |
| GET /api/download/stats/{id} | PASS | .txt download |
| POST /api/download/plot | PASS | PNG/SVG/PDF all work |
| GET /api/color-maps | PASS | 38 palettes |
| Static files (/, /css, /js, /assets) | PASS | All served correctly |

## Key Dependency Note
- **plotly must be <6** and **kaleido must be 0.2.1** for server-side image export to work
- plotly 6.x uses kaleido 1.x which requires Chrome installed on the system
- kaleido 0.2.1 bundles its own chromium - works on Render without Chrome

## Remaining Tasks

### Must Do Before Deploy
1. **Browser test** - Open `http://localhost:8000` in a browser and walk through the full wizard flow:
   - Upload `sample_data.csv`
   - Map columns (Treatment, Value, Replicate)
   - Click Analyze
   - Verify plot renders correctly
   - Test all download buttons
   - Test paired toggle
   - Test plot customization (expand the collapsible, change settings, re-analyze)
   - Test drag-and-drop vs click-to-browse upload
   - Test with an XLSX file

2. **Error handling** - Test edge cases:
   - Upload a non-CSV/XLSX file (should show error)
   - Select wrong column types (e.g., text column as Value)
   - Upload a file with missing values

### Nice to Have
3. **Mobile responsiveness** - Test on mobile viewport, may need tweaks
4. **Loading states polish** - The spinners are implemented but could be more prominent
5. **Data preview scrolling** - For files with many columns, horizontal scroll might need tweaking

### Deploy
6. **Push to GitHub** - All changes need to be committed and pushed
7. **Render setup** - Connect repo on Render dashboard, or push and let auto-deploy from `render.yaml`
8. **Verify on Render** - Test the deployed version end-to-end

## How to Run Locally
```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
# Open http://localhost:8000
```

## Files That Can Be Removed (optional cleanup)
- `dist/CellBio Stats.exe` - Old PyInstaller build
- `assets/logo.png` - Duplicated in `frontend/assets/logo.png` (but keep for GitHub README image reference)
