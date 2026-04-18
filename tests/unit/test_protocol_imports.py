"""Import smoke test — catches broken Protocol signatures early."""


def test_data_protocol_importable():
    from data.base import DataSource
    assert DataSource.__name__ == "DataSource"


def test_feature_protocol_importable():
    from features.base import Feature
    assert Feature.__name__ == "Feature"


def test_model_protocols_importable():
    from models.base import (
        LLMContextFlags,
        LLMContextProvider,
        PredictionBundle,
        Predictor,
    )
    bundle = PredictionBundle(
        direction="flat",
        prob_up=0.5,
        horizon_bars=1,
        feature_snapshot_hash="x",
        feature_registry_version="0.0.0",
        ml_model_version="stub",
        llm_prompt_version="stub",
    )
    assert bundle.direction == "flat"
    flags = LLMContextFlags(context_veto=False)
    assert flags.context_veto is False
