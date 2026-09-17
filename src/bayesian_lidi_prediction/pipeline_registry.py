"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from bayesian_lidi_prediction.pipelines import data_ingestion


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_ingestion_pipeline = data_ingestion.create_pipeline()
    pipelines = find_pipelines(raise_errors=True)
    pipelines["data_ingestion"] = data_ingestion_pipeline
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines
