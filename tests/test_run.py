from purchase_predict.pipeline_registry import register_pipelines


def test_register_pipelines_contains_expected_pipelines():
    pipelines = register_pipelines()

    assert "loading" in pipelines
    assert "processing" in pipelines
    assert "training" in pipelines
    assert "__default__" in pipelines