from dataclasses import dataclass
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.graph_objs import Figure

from thoth.anomaly.optimization import AnomalyOptimization, MetricOptimization
from thoth.base import Point, TimeSeries
from thoth.profiler import Metric


def plot_ts(ts: TimeSeries) -> Figure:
    return px.line(
        data_frame=pd.DataFrame(ts.points), x="ts", y="value", title=ts.metric.name
    )


def plot_validation_results(metric_optimization: MetricOptimization) -> Figure:
    true_values = [
        {
            "ts": point.ts,
            "value": round(point.true_value, 2),
            "label": "True Value",
        }
        for point in metric_optimization.validation_results[0].points
    ]
    model_values = [
        {
            "ts": point.ts,
            "value": round(point.predicted, 2) if point.predicted else None,
            "label": validation_ts.model_name,
        }
        for validation_ts in metric_optimization.validation_results
        for point in validation_ts.points
    ]
    df = pd.DataFrame(true_values + model_values)
    return px.line(
        data_frame=df, x="ts", y="value", color="label", title="Validation Values"
    )


def plot_validation_errors(metric_optimization: MetricOptimization) -> Figure:
    error_values = [
        {
            "ts": point.ts,
            "error": point.error,
            "label": validation_ts.model_name,
        }
        for validation_ts in metric_optimization.validation_results
        for point in validation_ts.points
    ]
    df = pd.DataFrame(error_values)
    return px.line(
        data_frame=df, x="ts", y="error", color="label", title="Validation Errors"
    )


def plot_metric_scoring(
    metric: Metric, threshold: float, scoring_points: List[Point]
) -> Figure:
    fig = px.line(
        data_frame=pd.DataFrame(scoring_points),
        x="ts",
        y="value",
        title=f"Anomaly Scoring for {metric.name}",
    )
    fig.add_hline(
        y=threshold,
        line_width=1,
        line_dash="dash",
        line_color="red",
        annotation_text="threshold",
        annotation_position="top right",
        annotation_font={"color": "red"},
    )
    fig.add_hrect(
        y0=threshold,
        y1=1.0,
        annotation_text="anomaly zone",
        annotation_position="top right",
        annotation_font={"color": "red"},
        fillcolor="red",
        opacity=0.25,
        line_width=0,
    )
    return fig


def plot_predicted_values(
    metric: Metric,
    threshold: float,
    predicted_points: List[Point],
    observed_points: List[Point],
) -> Figure:
    x_predicted = [point.ts for point in predicted_points]
    min_ts = x_predicted[0]
    y_predicted = [point.value for point in predicted_points]
    y_predicted_upper = [y * (1 + threshold) for y in y_predicted]
    y_predicted_lower = [y * (1 - threshold) for y in y_predicted]
    x_observed = [point.ts for point in observed_points if point.ts >= min_ts]
    y_observed = [point.value for point in observed_points if point.ts >= min_ts]

    fig = go.Figure(
        [
            go.Scatter(
                x=x_predicted,
                y=y_predicted,
                name="predicted",
                # line=dict(color='rgb(0,100,80)'),
                mode="lines",
                line=dict(color="rgba(39, 174, 96,1.0)"),
            ),
            go.Scatter(
                x=x_predicted + x_predicted[::-1],  # x, then x reversed
                y=y_predicted_upper
                + y_predicted_lower[::-1],  # upper, then lower reversed
                fill="toself",
                # fillcolor='rgba(0,100,80,0.2)',
                # line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name="expected interval",
                fillcolor="green",
                mode="lines",
                line=dict(width=0),
                opacity=0.25,
                # showlegend=False
            ),
            go.Scatter(
                x=x_observed,
                y=y_observed,
                name="observed",
                # line=dict(color='rgb(0,100,80)'),
                mode="lines",
                line=dict(color="rgba(211, 84, 0,1.0)"),
            ),
        ],
    )
    fig.update_layout(title=f"Expected vs Observed for {metric.name}")
    return fig


def create_anomaly_optimization_table(
    optimization: AnomalyOptimization,
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
            for metric_report in optimization.metric_optimizations
            for validation_time_series in metric_report.validation_results
        ]
    )


def create_metric_optimization_table(
    metric_optimization: MetricOptimization,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": validation_result.model_name,
                "mean_error": round(validation_result.mean_error, 4),
                "threshold": round(validation_result.threshold, 2),
                "below_threshold_percentage": round(
                    validation_result.below_threshold_proportion, 2
                ),
            }
            for validation_result in metric_optimization.validation_results
        ]
    )
