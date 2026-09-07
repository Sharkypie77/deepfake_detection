"""Checkpoint provenance and validation policy."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

THRESHOLDS = {"aasist": 0.20, "vision": 0.70}


def checkpoint_status(model_name: str, checkpoint_path: str | None, checkpoint: Any = None) -> Dict[str, Any]:
    if not checkpoint_path:
        return {"validated": False, "reason": "no checkpoint configured"}
    path = Path(checkpoint_path).expanduser()
    manifest_path = path.with_suffix(".json")
    manifest = None
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    elif isinstance(checkpoint, dict):
        manifest = checkpoint.get("manifest") or checkpoint.get("metadata")
    if not manifest:
        return {"validated": False, "reason": "no manifest found"}
    required = ("model_name", "training_dataset", "training_date", "validation_metric", "checksum", "validated")
    missing = [field for field in required if field not in manifest]
    if missing:
        return {"validated": False, "reason": f"manifest missing: {', '.join(missing)}"}
    if manifest["model_name"].lower() != model_name.lower():
        return {"validated": False, "reason": "manifest model_name mismatch"}
    if not manifest["validated"]:
        return {"validated": False, "reason": "manifest validated=false"}
    metric = float(manifest["validation_metric"])
    if model_name.lower() == "aasist" and metric > THRESHOLDS["aasist"]:
        return {"validated": False, "reason": f"EER {metric} exceeds threshold {THRESHOLDS['aasist']}"}
    if model_name.lower() == "vision" and metric < THRESHOLDS["vision"]:
        return {"validated": False, "reason": f"AUC {metric} below threshold {THRESHOLDS['vision']}"}
    if manifest["checksum"] != hashlib.sha256(path.read_bytes()).hexdigest():
        return {"validated": False, "reason": "checkpoint checksum mismatch"}
    return {"validated": True, "reason": "validated manifest", "manifest": manifest}
