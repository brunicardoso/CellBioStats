import uuid
import io
import pandas as pd
import plotly.express as px
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response

from .models import AnalyzeRequest, PlotDownloadRequest
from .stats_engine import create_stats_table
from .plot_engine import create_interactive_superplot

router = APIRouter(prefix="/api")

# In-memory session store
sessions: dict[str, dict] = {}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    filename = file.filename or ""

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a CSV or XLSX file.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    session_id = str(uuid.uuid4())
    sessions[session_id] = {"df": df, "stats_text": None}

    return {
        "session_id": session_id,
        "columns": list(df.columns),
        "rows": len(df),
        "filename": filename,
        "preview": df.head(5).to_dict(orient="records"),
    }


@router.post("/analyze")
async def analyze(req: AnalyzeRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please re-upload your file.")

    df = session["df"]

    for col in [req.treatment_col, req.value_col, req.replicate_col]:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found in data.")

    try:
        fig, stats_results, plot_data, ks_results = create_interactive_superplot(
            df,
            req.treatment_col,
            req.value_col,
            req.replicate_col,
            paired=req.paired,
            font_size=req.font_size,
            color_map=req.color_map,
            template=req.template,
            marker_size=req.marker_size,
            downsample_mode=req.downsample_mode,
            downsample_value=req.downsample_value,
            log_scale=req.log_scale,
            show_stars=req.show_stars,
            show_axes=req.show_axes,
        )

        stats_text = create_stats_table(
            stats_results,
            plot_data["data"],
            plot_data["x_col"],
            plot_data["y_col"],
            plot_data["replicate_col"],
        )

        session["stats_text"] = stats_text

        ks_info = None
        if ks_results:
            any_failed = any(r['failed'] for r in ks_results)
            if any_failed:
                summary = "Downsampled distribution may not represent the original data. Try using a larger sample size or percentage."
            else:
                summary = "Downsampled points represent the original distribution well."
            ks_info = {"warning": any_failed, "summary": summary}

        return {
            "plot_json": fig.to_json(),
            "stats_text": stats_text,
            "ks_info": ks_info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.get("/download/stats/{session_id}")
async def download_stats(session_id: str):
    session = sessions.get(session_id)
    if not session or not session.get("stats_text"):
        raise HTTPException(status_code=404, detail="No stats available. Run analysis first.")

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"statistical_analysis_summary_{timestamp}.txt"

    return Response(
        content=session["stats_text"],
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/download/plot")
async def download_plot(req: PlotDownloadRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    df = session["df"]

    try:
        fig, _, _, _ = create_interactive_superplot(
            df,
            req.treatment_col,
            req.value_col,
            req.replicate_col,
            paired=req.paired,
            font_size=req.font_size,
            color_map=req.color_map,
            template=req.template,
            marker_size=req.marker_size,
            downsample_mode=req.downsample_mode,
            downsample_value=req.downsample_value,
            log_scale=req.log_scale,
            show_stars=req.show_stars,
            show_axes=req.show_axes,
        )

        img_bytes = fig.to_image(format=req.format, scale=req.scale)

        media_types = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "svg": "image/svg+xml",
            "pdf": "application/pdf",
        }

        y_col_safe = req.value_col.replace(" ", "_")
        filename = f"superplot_{y_col_safe}.{req.format}"

        return Response(
            content=img_bytes,
            media_type=media_types.get(req.format, "application/octet-stream"),
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plot export error: {str(e)}")


@router.get("/color-maps")
async def get_color_maps():
    color_maps = [
        k for k in dir(px.colors.qualitative)
        if not k.startswith("_") and isinstance(getattr(px.colors.qualitative, k), list)
    ]
    return {"color_maps": sorted(color_maps)}
