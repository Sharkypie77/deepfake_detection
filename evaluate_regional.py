"""Evaluate AASIST on a bounded IndicTTS regional-language subset.

Only the labeled train split is used for this eval-only baseline. The test
split's labels are intentionally hidden (``is_tts=-1``), so it cannot produce
an honest EER. Audio is decoded from dataset bytes and resampled to 16 kHz,
matching the AASIST inference contract.
"""

import argparse
import csv
import io
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from sklearn.metrics import roc_curve

from modules.aasist import AASIST


def compute_eer(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return None
    fpr, tpr, _ = roc_curve(labels, scores)
    index = int(np.nanargmin(np.abs(fpr - (1.0 - tpr))))
    return float((fpr[index] + 1.0 - tpr[index]) / 2.0)


def load_model(checkpoint_path, device):
    model = AASIST().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    state_dict = {key.removeprefix("model."): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Invalid AASIST checkpoint: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def waveform_from_bytes(audio_bytes):
    waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != 16000:
        source = torch.from_numpy(waveform).unsqueeze(0)
        target_length = round(source.shape[-1] * 16000 / sample_rate)
        waveform = torch.nn.functional.interpolate(
            source.unsqueeze(0), size=target_length, mode="linear", align_corners=False
        ).squeeze().numpy()
    samples = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
    samples = samples[:64600]
    if samples.numel() < 64600:
        samples = torch.nn.functional.pad(samples, (0, 64600 - samples.numel()))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.getenv("AASIST_CHECKPOINT_PATH", "data/models/aasist-official.pth"))
    parser.add_argument("--per-language", type=int, default=100)
    parser.add_argument("--output", default="data/eval/indictts_regional/predictions.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"AASIST checkpoint not found: {checkpoint}")
    device = torch.device(args.device)
    dataset = load_dataset(
        "SherryT997/IndicTTS-Deepfake-Challenge-Data",
        split="train",
        streaming=True,
    ).decode(False)

    selected = []
    language_counts = Counter()
    label_counts = Counter()
    for row in dataset:
        language = row["language"]
        label = int(row["is_tts"])
        if label not in (0, 1) or language_counts[language] >= args.per_language:
            continue
        audio_bytes = row["audio"].get("bytes")
        if not audio_bytes:
            continue
        selected.append((row["id"], language, label, audio_bytes))
        language_counts[language] += 1
        label_counts[(language, label)] += 1

    if not selected:
        raise RuntimeError("No labeled audio was selected from the IndicTTS train split")

    model = load_model(checkpoint, device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.inference_mode():
        for index, (sample_id, language, label, audio_bytes) in enumerate(selected, start=1):
            logits = model(waveform_from_bytes(audio_bytes).unsqueeze(0).to(device))
            score = float(torch.softmax(logits, dim=1)[0, 1].item())
            rows.append({"id": sample_id, "language": language, "label": label, "score": score})
            if index % 50 == 0:
                print(f"evaluated={index}/{len(selected)}", flush=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "language", "label", "score"))
        writer.writeheader()
        writer.writerows(rows)

    by_language = defaultdict(lambda: {"labels": [], "scores": []})
    for row in rows:
        by_language[row["language"]]["labels"].append(row["label"])
        by_language[row["language"]]["scores"].append(row["score"])
    report = {
        "dataset": "SherryT997/IndicTTS-Deepfake-Challenge-Data",
        "split": "train",
        "selection": {"max_per_language": args.per_language, "samples": len(rows)},
        "label_definition": "is_tts=0 real, is_tts=1 synthetic/TTS",
        "language_counts": dict(language_counts),
        "label_counts": {f"{language}:{label}": count for (language, label), count in label_counts.items()},
        "overall_eer": compute_eer([row["label"] for row in rows], [row["score"] for row in rows]),
        "per_language_eer": {
            language: compute_eer(values["labels"], values["scores"])
            for language, values in sorted(by_language.items())
        },
        "predictions": str(output_path),
        "checkpoint": str(checkpoint),
        "device": str(device),
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
