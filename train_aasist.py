"""Train/evaluate the official AASIST topology on ASVspoof LA.

This script is intentionally explicit about dataset paths and never creates a
validated provenance manifest. A manifest may be generated only from measured
held-out metrics after the run.
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import soundfile as sf

from modules.aasist import AASIST


class ASVspoofDataset(Dataset):
    def __init__(self, protocol: Path, audio_root: Path):
        self.items = []
        for line in protocol.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            file_id, label = fields[1], fields[-1]
            self.items.append((audio_root / f"{file_id}.flac", int(label == "spoof")))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        waveform, sample_rate = sf.read(path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = torch.from_numpy(waveform)
        if sample_rate != 16000:
            raise ValueError(f"Expected 16 kHz ASVspoof audio, got {sample_rate} for {path}")
        waveform = waveform[:64600]
        if waveform.numel() < 64600:
            waveform = nn.functional.pad(waveform, (0, 64600 - waveform.numel()))
        return waveform, torch.tensor(label, dtype=torch.long)


def train(args):
    dataset = ASVspoofDataset(Path(args.protocol), Path(args.audio_root))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = AASIST().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(args.epochs):
        for waveform, target in loader:
            logits = model(waveform.to(args.device))
            loss = criterion(logits, target.to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch={epoch + 1} loss={loss.item():.6f}")
    torch.save(model.state_dict(), args.output)
    print(f"checkpoint={args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, help="ASVspoof LA train protocol")
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output", default="data/models/aasist-trained.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train(parser.parse_args())
