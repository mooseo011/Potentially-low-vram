from plvram.config import AppConfig
from plvram.engine.inference import InferenceEngine, backend_available


def test_engine_demo_mode_without_backend():
    cfg = AppConfig()
    engine = InferenceEngine(cfg)
    # On a box without torch/transformers/deepspeed this is demo mode.
    assert engine.demo == (not backend_available())


def test_demo_load_and_generate():
    cfg = AppConfig()
    engine = InferenceEngine(cfg)
    if not engine.demo:
        return  # real backend present; skip the demo-specific assertions
    result = engine.load()
    assert result.ok and result.demo
    text = "".join(engine.generate("hello there"))
    assert "demo mode" in text.lower()
    assert "hello there" in text
