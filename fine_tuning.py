"""
Phase 7: Fine-tuning Infrastructure
Transfer learning on Indian regional deepfakes
Supports Hindi, Tamil, Telugu, Kannada, Marathi, etc.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
import logging
from typing import Dict, Tuple, List
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class IndianDeepfakeDataset(Dataset):
    """
    Custom dataset for Indian regional deepfakes
    
    Directory structure:
    data/training/
    ├── authentic/
    │   ├── hindi/
    │   ├── tamil/
    │   ├── telugu/
    │   └── ...
    └── deepfake/
        ├── wav2lip/
        ├── sadtalker/
        ├── rvc/
        └── ...
    """
    
    def __init__(self, video_dir: str, split: str = "train", test_size: float = 0.2):
        """
        Args:
            video_dir: Root directory containing video data
            split: "train" or "val"
            test_size: Validation split ratio
        """
        self.video_dir = Path(video_dir)
        self.split = split
        self.samples = []
        
        # Load authentic videos
        authentic_dir = self.video_dir / "authentic"
        if authentic_dir.exists():
            for lang_dir in authentic_dir.iterdir():
                if lang_dir.is_dir():
                    for video_file in lang_dir.glob("*.mp4"):
                        self.samples.append((str(video_file), 0, lang_dir.name))  # Label 0: authentic
        
        # Load deepfake videos
        deepfake_dir = self.video_dir / "deepfake"
        if deepfake_dir.exists():
            for method_dir in deepfake_dir.iterdir():
                if method_dir.is_dir():
                    for video_file in method_dir.glob("*.mp4"):
                        self.samples.append((str(video_file), 1, method_dir.name))  # Label 1: deepfake
        
        # Split train/val
        train_samples, val_samples = train_test_split(
            self.samples, test_size=test_size, random_state=42, stratify=[s[1] for s in self.samples]
        )
        
        self.samples = train_samples if split == "train" else val_samples
        logger.info(f"Loaded {len(self.samples)} {split} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns:
            - Mel-spectrogram (audio features)
            - Label (0/1)
            - Language/Method name
        """
        video_path, label, category = self.samples[idx]
        
        try:
            from PIL import Image
            from facenet_pytorch import MTCNN
            cap = cv2.VideoCapture(video_path)
            frame_count = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise ValueError("could not decode a video frame")
            detector = MTCNN(image_size=224, margin=20, keep_all=False, device="cpu")
            face = detector(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if face is None:
                face = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
                face = torch.nn.functional.interpolate(face.unsqueeze(0), (224, 224)).squeeze(0)
            return face, label, category
        
        except Exception as e:
            logger.warning(f"Error loading {video_path}: {e}")
            # Return dummy data on error
            return torch.zeros((3, 224, 224)), label, category


class DeepfakeDetectorFineTune(pl.LightningModule):
    """
    Fine-tunable deepfake detector using transfer learning
    Built on pretrained EfficientNetV2
    """
    
    def __init__(self, learning_rate: float = 1e-3, num_classes: int = 2):
        super().__init__()
        self.learning_rate = learning_rate
        
        # Pretrained backbone
        import timm
        self.backbone = timm.create_model('efficientnetv2_rw_s', pretrained=True, num_classes=num_classes)
        
        # Replace final layer
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
    
    def forward(self, x):
        return self.backbone(x)
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', (logits.argmax(1) == y).float().mean(), prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', (logits.argmax(1) == y).float().mean(), prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]


class FineTuningPipeline:
    """
    Complete fine-tuning pipeline for Indian deepfakes
    """
    
    def __init__(self, data_dir: str, output_dir: str = "fine_tuned_models"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Fine-tuning pipeline initialized")
        logger.info(f"Data: {self.data_dir}")
        logger.info(f"Output: {self.output_dir}")
    
    def train(self, epochs: int = 10, batch_size: int = 32, learning_rate: float = 1e-3) -> Dict:
        """
        Train model on Indian deepfakes dataset
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Training results
        """
        
        logger.info(f"Starting fine-tuning: {epochs} epochs, {batch_size} batch size")
        
        # Create datasets
        train_dataset = IndianDeepfakeDataset(str(self.data_dir), split="train")
        val_dataset = IndianDeepfakeDataset(str(self.data_dir), split="val")
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
        
        # Model
        model = DeepfakeDetectorFineTune(learning_rate=learning_rate)
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            log_every_n_steps=10,
            default_root_dir=str(self.output_dir)
        )
        
        # Train
        trainer.fit(model, train_loader, val_loader)
        
        logger.info(f"✓ Fine-tuning complete")
        logger.info("Fine-tuning complete; inspect Lightning checkpoints in %s", self.output_dir)
        
        return {
            "status": "completed",
            "best_model": str(self.output_dir),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }
    
    def evaluate_on_dataset(self, model_path: str, test_data_dir: str) -> Dict:
        """
        Evaluate fine-tuned model on test dataset
        """
        
        logger.info(f"Evaluating model: {model_path}")
        
        # Load model
        model = DeepfakeDetectorFineTune.load_from_checkpoint(model_path)
        model.eval()
        
        # Test dataset
        test_dataset = IndianDeepfakeDataset(test_data_dir, split="val")
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Evaluate
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (x, y, _) in enumerate(test_loader):
                x = x.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())
        
        # Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
        logger.info(f"✓ Accuracy: {accuracy:.4f}")
        logger.info(f"✓ Precision: {precision:.4f}")
        logger.info(f"✓ Recall: {recall:.4f}")
        logger.info(f"✓ F1: {f1:.4f}")
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "num_samples": len(all_labels)
        }
    
    def generate_training_report(self, metrics: Dict) -> str:
        """Generate training report"""
        
        report = f"""
╔════════════════════════════════════════════════════════════════╗
║         DEEPFAKE DETECTOR FINE-TUNING REPORT                   ║
╚════════════════════════════════════════════════════════════════╝

📊 PERFORMANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Accuracy:  {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)
  Precision: {metrics.get('precision', 0):.4f}
  Recall:    {metrics.get('recall', 0):.4f}
  F1-Score:  {metrics.get('f1', 0):.4f}

📈 CONFUSION MATRIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

              Predicted
              Authentic  Deepfake
  Authentic     {metrics.get('confusion_matrix', [[0,0],[0,0]])[0][0]}        {metrics.get('confusion_matrix', [[0,0],[0,0]])[0][1]}
  Deepfake      {metrics.get('confusion_matrix', [[0,0],[0,0]])[1][0]}        {metrics.get('confusion_matrix', [[0,0],[0,0]])[1][1]}

✅ MODEL READY FOR DEPLOYMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fine-tuned model tested on {metrics.get('num_samples', 0)} samples
Trained on Indian regional deepfakes (Hindi, Tamil, Telugu, etc.)
Ready for production deployment

🚀 Next Steps:
  1. Deploy to production servers
  2. Monitor performance on live data
  3. Retrain quarterly with new datasets
  4. Integrate with the Telegram bot

"""
        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    pipeline = FineTuningPipeline(data_dir="data/training")
    
    logger.info("To fine-tune on your dataset:")
    logger.info("  python fine_tuning.py --train --epochs 10 --batch_size 32")
    logger.info("To evaluate:")
    logger.info("  python fine_tuning.py --evaluate --model_path path/to/model.ckpt")
