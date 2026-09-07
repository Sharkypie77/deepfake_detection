import hashlib
import json

from checkpoint_provenance import checkpoint_status


def test_checkpoint_without_manifest_is_unvalidated(tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"weights")
    status = checkpoint_status("vision", str(checkpoint))
    assert status == {"validated": False, "reason": "no manifest found"}


def test_valid_manifest_requires_checksum_and_metric(tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"weights")
    manifest = {
        "model_name": "vision",
        "training_dataset": "validated-set",
        "training_date": "2026-01-01",
        "validation_metric": 0.91,
        "checksum": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "validated": True,
    }
    checkpoint.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
    status = checkpoint_status("vision", str(checkpoint))
    assert status["validated"] is True
