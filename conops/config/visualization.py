from pydantic import BaseModel


class VisualizationConfig(BaseModel):
    """Configuration for visualization settings."""

    # ACS mode color mapping
    mode_colors: dict[str, str] = {
        "SCIENCE": "green",
        "SLEWING": "orange",
        "SAA": "purple",
        "PASS": "cyan",
        "CHARGING": "yellow",
        "SAFE": "red",
    }

    # Plot appearance settings
    font_family: str = "Helvetica"
    title_font_size: int = 12
    label_font_size: int = 10
    legend_font_size: int = 8
    tick_font_size: int = 9

    # Sky pointing visualization settings
    figsize: tuple[int, int] = (14, 8)
    n_grid_points: int = 100
    constraint_alpha: float = 0.3
    time_step_seconds: float = 60.0

    # DITL timeline settings
    timeline_figsize: tuple[int, int] = (12, 5)
    timeline_font_size: int = 11
    orbit_period_seconds: float = 5762.0

    # Data management telemetry settings
    data_telemetry_figsize: tuple[int, int] = (12, 10)
