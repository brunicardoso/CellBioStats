from pydantic import BaseModel
from typing import Optional


class AnalyzeRequest(BaseModel):
    session_id: str
    treatment_col: str
    value_col: str
    replicate_col: str
    paired: bool = False
    font_size: int = 18
    color_map: str = "Plotly"
    template: str = "plotly"
    marker_size: int = 6
    downsample_mode: str = "all"
    downsample_value: Optional[float] = None
    log_scale: bool = False
    show_stars: bool = True
    show_axes: bool = True


class PlotDownloadRequest(BaseModel):
    session_id: str
    treatment_col: str
    value_col: str
    replicate_col: str
    paired: bool = False
    font_size: int = 18
    color_map: str = "Plotly"
    template: str = "plotly"
    marker_size: int = 6
    downsample_mode: str = "all"
    downsample_value: Optional[float] = None
    log_scale: bool = False
    show_stars: bool = True
    show_axes: bool = True
    format: str = "png"
    scale: int = 1
