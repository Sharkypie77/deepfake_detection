"""
Deepfake Detection Suite - Core Configuration
Language-Agnostic Multimodal Verification System
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
VIDEO_UPLOAD_DIR = DATA_DIR / "uploads"
MODEL_WEIGHTS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Create directories
for directory in [VIDEO_UPLOAD_DIR, MODEL_WEIGHTS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'deepfake_detection.db'}")

# ============================================================================
# API CONFIGURATION
# ============================================================================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_TITLE = "Deepfake Detection Suite"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Language-agnostic multimodal deepfake detection for Indian languages & global content"

# ============================================================================
# TELEGRAM BOT CONFIGURATION
# ============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
# Get token from @BotFather on Telegram: https://t.me/botfather

# ============================================================================
# REDIS & CELERY CONFIGURATION
# ============================================================================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL

# ============================================================================
# MODEL CONFIGURATION & WEIGHTS
# ============================================================================
class ModelConfig:
    """Language-agnostic models for multimodal deepfake detection"""
    
    # Face Detection (Language-Agnostic)
    FACE_DETECTOR = "mtcnn"  # Options: "mtcnn", "retinaface"
    FACE_DETECTOR_CONFIDENCE = 0.9
    
    # Vision Module - Spatial Domain
    VISION_BACKBONE = "efficientnet_v2_s"  # EfficientNetV2-S for speed/accuracy balance
    VISION_INPUT_SIZE = (224, 224)
    VISION_MODEL_WEIGHTS = "imagenet"  # Pretrained on ImageNet
    
    # Vision Module - Frequency Domain
    FFT_ENABLED = True
    DCT_ENABLED = True
    FREQUENCY_BANDS = 16  # Number of frequency bands for analysis
    
    # Audio Module - AASIST (Language-Agnostic Anti-Spoofing)
    # AASIST works on raw waveforms, no language dependency
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_DURATION = 10  # seconds
    AUDIO_BACKEND = "librosa"
    AASIST_MODEL = "pretrained"  # Will use community pretrained weights
    
    # Audio Feature Extraction
    MEL_SPECTROGRAM_ENABLED = True
    MEL_BINS = 128
    CQCC_ENABLED = True
    MFCC_ENABLED = True
    
    # Lip-Sync Module (Language-Agnostic)
    # Uses MediaPipe face mesh + temporal frame-based viseme detection
    MEDIAPIPE_CONFIDENCE = 0.5
    VISEME_EXTRACTION = "landmark_distance"  # Euclidean distance between lip landmarks
    SYNC_THRESHOLD = 0.65  # Cosine similarity threshold for sync match
    
    # Speech-to-Text (Whisper - supports 99+ languages)
    WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
    WHISPER_LANGUAGE = "auto"  # Auto-detect (Hindi, Tamil, Telugu, English, etc.)
    
    # Compression Resilience
    COMPRESSION_AWARE = True
    AUGMENT_WITH_COMPRESSION = True
    CRF_RANGE = [28, 30, 32, 34, 36]  # H.264 quality factors for training augmentation
    
    # Multimodal Fusion
    FUSION_STRATEGY = "attention_weighted"  # Options: "weighted_sum", "attention_weighted", "meta_classifier"
    VISION_WEIGHT = 0.35
    AUDIO_WEIGHT = 0.35
    SYNC_WEIGHT = 0.30
    
    # Risk Score Calibration
    AUTHENTIC_THRESHOLD = 0.30  # 0-30%: Likely Authentic (Green)
    INCONCLUSIVE_THRESHOLD = 0.65  # 31-65%: Inconclusive (Amber)
    DEEPFAKE_THRESHOLD = 1.0  # 66-100%: High-Risk Deepfake (Red)

# ============================================================================
# PROCESSING PIPELINE CONFIGURATION
# ============================================================================
class ProcessingConfig:
    """Video processing and batch job settings"""
    
    MAX_VIDEO_SIZE_MB = 500
    SUPPORTED_FORMATS = ["mp4", "avi", "mov", "mkv", "flv", "webm"]
    
    # Frame extraction
    FRAME_SAMPLE_RATE = 1  # Extract every 1st frame (FPS-dependent)
    MIN_FRAMES = 30
    MAX_FRAMES = 300
    
    # Audio extraction
    EXTRACT_AUDIO = True
    AUDIO_DURATION = 10  # Process 10 seconds of audio
    
    # Batch processing
    BATCH_SIZE = 4
    NUM_WORKERS = 2
    TASK_TIMEOUT = 600  # 10 minutes per video
    
    # GPU/CPU
    DEVICE = "cuda"  # Will auto-fallback to cpu if not available
    HALF_PRECISION = True  # Use FP16 for faster inference

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "json"  # Options: "json", "plaintext"

# ============================================================================
# SECURITY & VALIDATION
# ============================================================================
class SecurityConfig:
    """Input validation and security settings"""
    
    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"}
    MAX_UPLOAD_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_WINDOW = 3600  # 1 hour
    ENABLE_CORS = True
    CORS_ORIGINS = ["*"]  # Update for production
    
    # WhatsApp security
    WHATSAPP_WEBHOOK_TIMEOUT = 5

# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================
class ExportConfig:
    """PDF and report generation settings"""
    
    GENERATE_PDF_REPORTS = True
    INCLUDE_WAVEFORM_ANALYSIS = True
    INCLUDE_FRAME_HEATMAPS = True
    INCLUDE_AUDIT_TRAIL = True
    PDF_QUALITY = "high"  # Options: "low", "medium", "high"
