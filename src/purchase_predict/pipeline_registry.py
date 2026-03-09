from kedro.pipeline import Pipeline

from purchase_predict.pipelines.processing import create_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    processing_pipeline = create_pipeline()

    return {
        "__default__": processing_pipeline,
        "processing": processing_pipeline,
    }