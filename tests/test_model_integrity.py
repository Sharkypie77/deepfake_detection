import pytest
import torch

from modules.aasist import AASIST


def test_aasist_checkpoint_round_trip(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "aasist.pth"
    torch.save(AASIST().state_dict(), checkpoint_path)
    import importlib
    audio_analysis = importlib.import_module("modules.audio_analysis")
    monkeypatch.setattr(audio_analysis, "AASIST_CHECKPOINT_PATH", str(checkpoint_path))
    module = audio_analysis.AAISSTModule(device="cpu")

    assert isinstance(module.model, AASIST)


def test_aasist_forward_shape():
    model = AASIST().eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 64600))
    assert output.shape == (1, 2)


def test_official_aasist_checkpoint_if_available():
    checkpoint_path = __import__("pathlib").Path("data/models/aasist-official.pth")
    if not checkpoint_path.is_file():
        pytest.skip("official checkpoint is not present in CI")
    model = AASIST()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert missing == []
    assert unexpected == []


def test_aasist_missing_checkpoint_fails_loudly(monkeypatch):
    import importlib
    audio_analysis = importlib.import_module("modules.audio_analysis")
    monkeypatch.setattr(audio_analysis, "AASIST_CHECKPOINT_PATH", "missing-aasist-checkpoint.pth")

    with pytest.raises(FileNotFoundError, match="AASIST checkpoint not found"):
        audio_analysis.AAISSTModule(device="cpu")
