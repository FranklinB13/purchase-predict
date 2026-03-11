from kedro.pipeline import Pipeline

from purchase_predict.pipelines.loading import create_pipeline as create_loading_pipeline
from purchase_predict.pipelines.processing import create_pipeline as create_processing_pipeline
from purchase_predict.pipelines.training import create_pipeline as create_training_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    loading_pipeline = create_loading_pipeline()
    processing_pipeline = create_processing_pipeline()
    training_pipeline = create_training_pipeline()

    return {
        "__default__": loading_pipeline + processing_pipeline + training_pipeline,
        "loading": loading_pipeline,
        "processing": processing_pipeline,
        "training": training_pipeline,
    }
