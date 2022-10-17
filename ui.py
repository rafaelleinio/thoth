import dataclasses
from typing import List

import streamlit as st
import streamlit_permalink as stp
from sqlmodel import Session

import thoth as th
from thoth.base import convert_to_timeseries, TimeSeries, Point
from thoth.util import viz

ENGINE = th.build_engine()


def about_page():
    md = """
    # Welcome to thoth! 👋

    In a few easy steps you'll create an e2e data quality monitoring platform.

    ### Useful links:
    - [Thoth homepage](https://github.com/rafaelleinio/thoth)
    - [Notebook examples](https://github.com/rafaelleinio/thoth/tree/main/examples)
    - [Docs]()
    - [PyPI]()

    Made with ❤️ by [@rafaelleinio](https://github.com/rafaelleinio)

    """
    st.markdown(md)


def get_dataset_uris() -> List[str]:
    with Session(ENGINE) as session:
        datasets = th.get_datasets(session=session)
    return [d.uri for d in datasets]


def get_dataset(dataset_uri: str) -> th.Dataset:
    with Session(ENGINE) as session:
        dataset = th.get_dataset(dataset_uri=dataset_uri, session=session)
    return dataset


def get_metric_ts_from_profiling(
    metric: th.profiler.Metric, profiling: List[th.profiler.ProfilingReport]
) -> TimeSeries:
    return TimeSeries(
        metric=metric,
        points=[
            Point(ts=report.ts, value=report.get_profiling_value(metric=metric).value)
            for report in profiling
        ],
    )


def get_metric_score_ts_from_scoring(
    metric: th.profiler.Metric, scoring: List[th.anomaly.AnomalyScoring]
) -> TimeSeries:
    return TimeSeries(
        metric=metric,
        points=[
            Point(
                ts=anomaly_scoring.ts,
                value=anomaly_scoring.get_metric_score(metric=metric).value,
            )
            for anomaly_scoring in scoring
        ],
    )


def get_metric_predicted_ts_from_scoring(
    metric: th.profiler.Metric, scoring: List[th.anomaly.AnomalyScoring]
) -> TimeSeries:
    return TimeSeries(
        metric=metric,
        points=[
            Point(
                ts=anomaly_scoring.ts,
                value=anomaly_scoring.get_metric_score(metric=metric).predicted,
            )
            for anomaly_scoring in scoring
        ],
    )


@dataclasses.dataclass
class _InstanceTimeSeries:
    instance: str
    time_series: List[TimeSeries]


def build_profiling_view(dataset_uri: str, selected_instances: List[str]):
    st.markdown("## Profiling metrics")

    # fetch data
    with Session(ENGINE) as session:
        profiling = th.select_profiling(dataset_uri=dataset_uri, session=session)
    time_series = convert_to_timeseries(profiling)

    # time-serialize
    instance_time_series_collection = [
        _InstanceTimeSeries(
            instance=instance,
            time_series=[ts for ts in time_series if ts.metric.instance == instance],
        )
        for instance in selected_instances
    ]

    # build expander sections
    for instance_time_series in instance_time_series_collection:
        expander = st.expander(
            label=f"Instance '{instance_time_series.instance}'", expanded=True
        )
        for ts in instance_time_series.time_series:
            expander.plotly_chart(viz.plot_ts(ts=ts))


def build_optimization_view(dataset_uri: str, selected_instances: List[str]):
    st.markdown("## Dataset Anomaly Optimization")

    # fetch data
    with Session(ENGINE) as session:
        optimization = th.get_optimization(dataset_uri=dataset_uri, session=session)
    if not optimization:
        st.markdown("### ⚠️Optimization or Scoring not found for this dataset!")
        return

    st.markdown(f"### Target confidence = `{optimization.confidence}`")

    # time-serialize
    @dataclasses.dataclass
    class InstanceOptimization:
        instance: str
        metric_optimizations: List[th.anomaly.MetricOptimization]

    instance_optimizations = [
        InstanceOptimization(
            instance=instance,
            metric_optimizations=[
                mo
                for mo in optimization.metric_optimizations
                if mo.metric.instance == instance
            ],
        )
        for instance in selected_instances
    ]

    # build expander sections
    for instance_optimization in instance_optimizations:
        expander = st.expander(
            label=f"Instance '{instance_optimization.instance}'", expanded=True
        )
        for mo in instance_optimization.metric_optimizations:
            expander.markdown(f"#### Metric '{mo.metric.name}'")
            expander.markdown(
                f"- Best model for this metric: `{mo.best_model_name}`\n"
                f"- Best anomaly scoring threshold for the model: `{mo.threshold}`\n"
            )
            expander.plotly_chart(
                figure_or_data=viz.plot_validation_results(metric_optimization=mo)
            )
            expander.plotly_chart(
                figure_or_data=viz.plot_validation_errors(metric_optimization=mo)
            )
            expander.markdown("##### Models performance overview:")
            expander.table(
                data=viz.create_metric_optimization_table(metric_optimization=mo)
            )
            expander.markdown("___________________")


@dataclasses.dataclass
class MetricScoringTimeSeries:
    metric: th.profiler.Metric
    anomaly_threshold: float
    scoring_points: List[Point]
    observed_points: List[Point]
    predicted_points: List[Point]


@dataclasses.dataclass
class InstanceMetricScoring:
    instance: str
    metric_scoring_time_series_collection: List[MetricScoringTimeSeries]


def build_scoring_data(
    selected_instances: List[str],
    profiling: List[th.profiler.ProfilingReport],
    optimization: th.anomaly.AnomalyOptimization,
    scoring: List[th.anomaly.AnomalyScoring],
) -> List[InstanceMetricScoring]:
    instance_metric_scoring_collection = []
    for instance in selected_instances:
        metric_scoring_time_series_collection = []
        metrics = [
            metric_optimization.metric
            for metric_optimization in optimization.metric_optimizations
            if metric_optimization.metric.instance == instance
        ]
        for metric in metrics:
            metric_scoring_time_series_collection.append(
                MetricScoringTimeSeries(
                    metric=metric,
                    anomaly_threshold=optimization.get_metric_optimization(
                        metric=metric
                    ).threshold,
                    scoring_points=get_metric_score_ts_from_scoring(
                        metric=metric, scoring=scoring
                    ).points,
                    observed_points=get_metric_ts_from_profiling(
                        metric=metric, profiling=profiling
                    ).points,
                    predicted_points=get_metric_predicted_ts_from_scoring(
                        metric=metric, scoring=scoring
                    ).points,
                )
            )
        instance_metric_scoring_collection.append(
            InstanceMetricScoring(
                instance=instance,
                metric_scoring_time_series_collection=(
                    metric_scoring_time_series_collection
                ),
            )
        )
    return instance_metric_scoring_collection


def build_scoring_view(dataset_uri: str, selected_instances: List[str]):
    st.markdown("## Anomaly Scoring")

    # fetch data
    with Session(ENGINE) as session:
        optimization = th.get_optimization(dataset_uri=dataset_uri, session=session)
        scoring = th.get_scoring(dataset_uri=dataset_uri, session=session)
        if not optimization or not scoring:
            st.markdown("### ⚠️Optimization or Scoring not found for this dataset!")
            return
        start_ts = scoring[0].ts
        profiling = th.select_profiling(
            dataset_uri=dataset_uri, start_ts=start_ts, session=session
        )

    # time-serialize
    instance_metric_scoring_collection = build_scoring_data(
        selected_instances=selected_instances,
        optimization=optimization,
        profiling=profiling,
        scoring=scoring,
    )

    # build expander sections
    for instance_metric_scoring in instance_metric_scoring_collection:
        expander = st.expander(
            label=f"Instance '{instance_metric_scoring.instance}'", expanded=True
        )
        for (
            metric_scoring_time_series
        ) in instance_metric_scoring.metric_scoring_time_series_collection:
            expander.markdown(f"#### Metric '{metric_scoring_time_series.metric.name}'")
            expander.markdown(
                "🔴 **Anomaly detected for last timestamp batch "
                f"({metric_scoring_time_series.scoring_points[-1].ts.isoformat()})!**"
                if metric_scoring_time_series.scoring_points[-1].value
                > metric_scoring_time_series.anomaly_threshold
                else (
                    "🟢 Last timestamp batch "
                    f"({metric_scoring_time_series.scoring_points[-1].ts.isoformat()}) "
                    f"is according expectations"
                )
            )
            expander.plotly_chart(
                viz.plot_metric_scoring(
                    metric=metric_scoring_time_series.metric,
                    threshold=metric_scoring_time_series.anomaly_threshold,
                    scoring_points=metric_scoring_time_series.scoring_points,
                )
            )
            expander.plotly_chart(
                viz.plot_predicted_values(
                    metric=metric_scoring_time_series.metric,
                    threshold=metric_scoring_time_series.anomaly_threshold,
                    predicted_points=metric_scoring_time_series.predicted_points,
                    observed_points=metric_scoring_time_series.observed_points,
                )
            )
            expander.markdown("___________________")


def build_dataset_metadata_text(dataset: th.Dataset) -> str:
    metrics_list_text = "    - " + "\n    - ".join(
        f"`{str(metric)}`" for metric in sorted(dataset.metrics)
    )
    columns_list_text = "    - " + "\n    - ".join(f"`{c}`" for c in dataset.columns)
    return (
        f"- **Timestamp column**: `{dataset.ts_column}`\n"
        f"- **Profiling aggregation granularity**: `{dataset.granularity}`\n"
        f"- **Feature columns**: \n"
        f"{columns_list_text}\n"
        "- **Profiling metrics**:\n"
        f"{metrics_list_text}"
    )


def home_page():
    st.markdown("# Select dataset")
    with stp.form("dataset-form"):
        dataset_uri = stp.selectbox(
            label="Dataset:", options=get_dataset_uris(), url_key="dataset_uri"
        )
        dataset = get_dataset(dataset_uri=dataset_uri)

        expander = st.expander(label="Dataset metadata", expanded=False)
        expander.markdown(build_dataset_metadata_text(dataset=dataset))

        instances = list(dataset.get_instances())
        selected_instances = stp.multiselect(
            label="Select instances:",
            options=instances,
            default=instances,
            url_key="instances",
        )

        view_option = stp.radio(
            label="Select view:",
            options=["👤 Profiling", "📈 Optimization", "💯 Scoring"],
            index=0,
            url_key="view",
        )
        submit_button = stp.form_submit_button(label="✨ Get me the data!")

    if submit_button:
        if view_option == "👤 Profiling":
            build_profiling_view(
                dataset_uri=dataset_uri, selected_instances=selected_instances
            )
        if view_option == "📈 Optimization":
            build_optimization_view(
                dataset_uri=dataset_uri, selected_instances=selected_instances
            )
        if view_option == "💯 Scoring":
            build_scoring_view(
                dataset_uri=dataset_uri, selected_instances=selected_instances
            )


SUBPAGES = {
    "🏠️ Home": home_page,
    "❓ About": about_page,
}


def sidebar():
    with st.sidebar:
        st.image(
            "https://i.imgur.com/UJwvBFC.png",
            caption="data profiling monitoring platform",
        )
        # st.sidebar.subheader("Index")s
        option = stp.radio(
            "Index:",
            SUBPAGES.keys(),
            index=0,
            url_key="page",
            on_change=lambda: st.experimental_set_query_params(),
        )
        st.sidebar.markdown("---")
    SUBPAGES[option]()


def main():
    st.set_page_config(
        page_title="Thoth Dashboard",
        page_icon="https://i.imgur.com/aIYgdab.png",
        layout="wide",
    )
    sidebar()


if __name__ == "__main__":
    main()
