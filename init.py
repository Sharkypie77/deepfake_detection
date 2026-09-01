#!/usr/bin/env python3
"""
Initialize Deepfake Detection Suite
- Creates directories
- Initializes database
- Downloads model weights
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    logger.info("Creating directories...")
    
    dirs = [
        "data/uploads",
        "data/results",
        "data/models",
        "logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ {dir_path}")

def init_database():
    """Initialize SQLite database"""
    logger.info("Initializing database...")
    
    try:
        from models import init_db
        from config import DATABASE_URL
        
        engine, SessionLocal = init_db(DATABASE_URL)
        logger.info("✓ Database initialized")
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
        return False
    
    return True

def download_models():
    """Download pretrained model weights"""
    logger.info("Downloading pretrained models...")
    
    # Whisper
    try:
        logger.info("Downloading Whisper-base (138MB)...")
        import whisper
        whisper.load_model("base")
        logger.info("✓ Whisper-base downloaded")
    except Exception as e:
        logger.warning(f"Whisper download skipped: {e}")
    
    # EfficientNetV2
    try:
        logger.info("Downloading EfficientNetV2-S...")
        import torch
        from torchvision.models import efficientnet_v2_s
        model = efficientnet_v2_s(pretrained=True)
        logger.info("✓ EfficientNetV2-S downloaded")
    except Exception as e:
        logger.warning(f"EfficientNetV2-S download skipped: {e}")
    
    # MediaPipe (auto-downloads on first use)
    try:
        logger.info("Initializing MediaPipe...")
        import mediapipe as mp
        logger.info("✓ MediaPipe initialized")
    except Exception as e:
        logger.warning(f"MediaPipe initialization skipped: {e}")

def copy_env():
    """Copy .env.example to .env"""
    logger.info("Setting up environment file...")
    
    if Path(".env").exists():
        logger.info("✓ .env already exists, skipping")
    elif Path(".env.example").exists():
        import shutil
        shutil.copy(".env.example", ".env")
        logger.info("✓ Created .env from .env.example (update credentials manually)")
    else:
        logger.warning("✗ .env.example not found")

def main():
    """Run all initialization steps"""
    logger.info("\n" + "="*60)
    logger.info("🚀 Deepfake Detection Suite - Initialization")
    logger.info("="*60 + "\n")
    
    try:
        create_directories()
        copy_env()
        success = init_database()
        download_models()
        
        if success:
            logger.info("\n" + "="*60)
            logger.info("✅ Initialization complete!")
            logger.info("="*60)
            logger.info("\n📝 Next steps:")
            logger.info("  1. Edit .env with your configuration")
            logger.info("  2. For WhatsApp: Add WHATSAPP_API_TOKEN")
            logger.info("  3. Start API: uvicorn main:app --reload")
            logger.info("  4. Or test locally: python pipeline.py path/to/video.mp4\n")
            return 0
        else:
            logger.error("✗ Initialization failed")
            return 1
    
    except KeyboardInterrupt:
        logger.warning("\n✗ Initialization cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"✗ Initialization error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
