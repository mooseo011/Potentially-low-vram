from plvram.config import AppConfig, OffloadConfig


def test_defaults_and_save_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cfg = AppConfig()
    assert cfg.offload.param_device == "cpu"
    cfg.model_id = "meta-llama/Llama-3-8B"
    cfg.offload.param_device = "nvme"
    path = cfg.save()
    assert path.exists()

    loaded = AppConfig.load()
    assert loaded.model_id == "meta-llama/Llama-3-8B"
    assert loaded.offload.param_device == "nvme"


def test_remember_model_dedupes_and_caps():
    cfg = AppConfig()
    for i in range(15):
        cfg.remember_model(f"model-{i}")
    cfg.remember_model("model-0")  # re-add existing -> moves to front
    assert cfg.recent_models[0] == "model-0"
    assert len(cfg.recent_models) <= 10
    # no duplicates
    assert len(set(cfg.recent_models)) == len(cfg.recent_models)


def test_from_dict_tolerates_partial():
    cfg = AppConfig.from_dict({"model_id": "x", "offload": {"param_device": "cpu"}})
    assert cfg.model_id == "x"
    assert isinstance(cfg.offload, OffloadConfig)
