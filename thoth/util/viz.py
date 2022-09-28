import datetime
from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from thoth.anomaly.base import _TimeSeries
from thoth.anomaly.optimization import AnomalyOptimization, ValidationTimeSeries
from thoth.profiler import Metric


def _plot_line_plot(
    x: str,
    y: str,
    data: pd.DataFrame,
    title: str,
    ax: Any,
    hue: Optional[str] = None,
    style: Optional[str] = None,
) -> None:
    sns.lineplot(x=x, y=y, data=data, ax=ax, hue=hue, style=style)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
    ax.set(title=title)
    ax.tick_params(rotation=45)


def plot_time_series(time_series: _TimeSeries, zoom_n: int = 15) -> None:
    fig, axs = plt.subplots(nrows=2, figsize=(12, 8))
    fig.tight_layout(h_pad=10)
    base_kwargs = dict(x="ts", y="value")
    _plot_line_plot(
        **base_kwargs,
        data=pd.DataFrame(time_series.points),
        title=str(time_series.metric),
        ax=axs[0],
    )
    _plot_line_plot(
        **base_kwargs,
        data=pd.DataFrame(time_series.points[-zoom_n:]),
        title=f"Zoom last {zoom_n} points",
        ax=axs[1],
    )
    plt.show()


def plot_validation_time_series(
    validation_time_series: List[ValidationTimeSeries], metric: Metric, zoom_n: int = 15
) -> None:
    @dataclass
    class PlotData:
        ts: datetime.datetime
        value: float
        color: str
        style: Optional[str] = None
        error: Optional[float] = None

    true_values = [
        PlotData(point.ts, point.true_value, color="True", style="True")
        for point in validation_time_series[0].points
    ]
    last_n_ts: List[datetime.datetime] = [p.ts for p in true_values[-zoom_n:]]
    predicted_values = [
        PlotData(
            point.ts,
            point.predicted,
            color=ts.model_name + " (Model)",
            style="Model",
            error=point.error,
        )
        for ts in validation_time_series
        for point in ts.points
        if point.predicted
    ]
    values = true_values + predicted_values
    values_last_n = [p for p in values if p.ts in last_n_ts]

    fig, axs = plt.subplots(nrows=3, figsize=(12, 16))
    fig.tight_layout(h_pad=10)
    base_kwargs = dict(x="ts", y="value", hue="color", style="style")
    _plot_line_plot(
        **base_kwargs,
        title=str(metric),
        data=pd.DataFrame(data=values),
        ax=axs[0],
    )
    _plot_line_plot(
        **base_kwargs,
        title=f"Zoom last {zoom_n} points",
        data=pd.DataFrame(data=values_last_n),
        ax=axs[1],
    )
    _plot_line_plot(
        x="ts",
        y="error",
        hue="color",
        title="Model Errors",
        data=pd.DataFrame(data=predicted_values),
        ax=axs[2],
    )
    plt.show()


def tabulate_optimization_report_best_model(
    report: AnomalyOptimization,
) -> pd.DataFrame:
    @dataclass
    class Table:
        metric: str
        best_model: str
        threshold: float

    return pd.DataFrame(
        data=[
            Table(
                str(metric_report.metric),
                metric_report.best_model_name,
                metric_report.threshold,
            )
            for metric_report in report.metric_optimizations
        ]
    )


def tabulate_optimization_report_scores(
    report: AnomalyOptimization,
) -> pd.DataFrame:
    @dataclass
    class Table:
        metric: str
        model: str
        mean_error: float
        threshold: float
        below_threshold_proportion: float

    return pd.DataFrame(
        data=[
            Table(
                str(metric_report.metric),
                validation_time_series.model_name,
                validation_time_series.mean_error,
                validation_time_series.threshold,
                validation_time_series.below_threshold_proportion,
            )
            for metric_report in report.metric_optimizations
            for validation_time_series in metric_report.validation_results
        ]
    )
