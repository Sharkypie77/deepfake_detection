"""
SQLite Database Schema for Deepfake Detection Suite
Audit-trail enabled, structured for multimodal analysis
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import json

Base = declarative_base()

class BatchVideoMembership(Base):
    __tablename__ = "batch_video_membership"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(36), ForeignKey("batch_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    video_id = Column(String(36), ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    added_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String(50), default="pending", nullable=False)

    batch = relationship("BatchJob", back_populates="memberships")
    video = relationship("Video", back_populates="batch_memberships")
    __table_args__ = (Index("ix_batch_video_membership_batch_video", "batch_id", "video_id"),)

# ============================================================================
# VIDEO METADATA TABLE
# ============================================================================
class Video(Base):
    """Stores uploaded video metadata and processing status"""
    __tablename__ = "videos"
    
    id = Column(String(36), primary_key=True)  # UUID
    filename = Column(String(512), nullable=False)
    file_path = Column(String(1024), nullable=False)
    file_size_mb = Column(Float, nullable=False)
    duration_seconds = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    resolution = Column(String(20), nullable=True)  # e.g., "1920x1080"
    
    # Processing Status
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    progress_percent = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    
    # Submission Info
    submitted_by = Column(String(255), nullable=True)  # WhatsApp number, email, etc.
    submission_platform = Column(String(50))  # "web", "whatsapp", "api"
    submitted_at = Column(DateTime, default=datetime.utcnow)
    
    # Processing Timestamps
    processing_started_at = Column(DateTime, nullable=True)
    processing_completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    vision_result = relationship("VisionAnalysis", back_populates="video", uselist=False)
    audio_result = relationship("AudioAnalysis", back_populates="video", uselist=False)
    sync_result = relationship("SyncAnalysis", back_populates="video", uselist=False)
    final_result = relationship("FinalResult", back_populates="video", uselist=False)
    audit_logs = relationship("AuditLog", back_populates="video")
    batch_memberships = relationship("BatchVideoMembership", back_populates="video", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Video(id={self.id}, filename={self.filename}, status={self.status})>"


# ============================================================================
# VISION ANALYSIS RESULTS TABLE (Module 1)
# ============================================================================
class VisionAnalysis(Base):
    """Spatial-Frequency Vision AI Analysis Results"""
    __tablename__ = "vision_analysis"
    
    id = Column(String(36), primary_key=True)
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False)
    video = relationship("Video", back_populates="vision_result")
    
    # Spatial Domain Analysis
    boundary_warping_score = Column(Float, nullable=True)  # 0-1 likelihood of manipulation
    eye_blinking_regularity = Column(Float, nullable=True)  # 0-1 naturalness score
    hair_boundary_blur = Column(Float, nullable=True)  # 0-1 blur anomaly
    skin_texture_consistency = Column(Float, nullable=True)  # 0-1 consistency
    
    # Frequency Domain Analysis (FFT/DCT)
    gan_artifact_score = Column(Float, nullable=True)  # 0-1 likelihood of GAN artifacts
    fft_anomaly_regions = Column(JSON, nullable=True)  # High-frequency anomaly map
    dct_compression_artifacts = Column(Float, nullable=True)  # 0-1 artifact likelihood
    
    # Face Detection Info
    num_faces_detected = Column(Integer, nullable=True)
    face_landmarks_confidence = Column(Float, nullable=True)
    face_location_stability = Column(Float, nullable=True)  # Temporal stability
    
    # Overall Vision Score
    vision_deepfake_score = Column(Float, nullable=False)  # 0-1 final vision score
    
    # Metadata
    processed_at = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(50))  # For reproducibility
    
    def __repr__(self):
        return f"<VisionAnalysis(video_id={self.video_id}, score={self.vision_deepfake_score})>"


# ============================================================================
# AUDIO ANALYSIS RESULTS TABLE (Module 2)
# ============================================================================
class AudioAnalysis(Base):
    """Acoustic & Voice Cloning Detection Results"""
    __tablename__ = "audio_analysis"
    
    id = Column(String(36), primary_key=True)
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False)
    video = relationship("Video", back_populates="audio_result")
    
    # AASIST Anti-Spoofing Score (Language-Agnostic)
    aasist_spoofing_score = Column(Float, nullable=True)  # 0-1 synthetic likelihood
    aasist_confidence = Column(Float, nullable=True)
    
    # Mel-Spectrogram Analysis
    mel_spectrogram_anomalies = Column(JSON, nullable=True)  # Anomaly regions
    spectral_cutoff_presence = Column(Float, nullable=True)  # 0-1 likelihood of cutoff
    
    # CQCC (Constant-Q Cepstral Coefficients) - Language Agnostic
    cqcc_consistency = Column(Float, nullable=True)  # 0-1 naturalness
    phase_discontinuities = Column(Float, nullable=True)  # 0-1 artifact likelihood
    
    # Acoustic Features
    breath_pause_detection = Column(Float, nullable=True)  # 0-1 naturalness of pauses
    voice_pitch_consistency = Column(Float, nullable=True)  # 0-1 consistency
    formant_stability = Column(Float, nullable=True)  # 0-1 stability
    
    # Language Detection (Informational Only)
    detected_languages = Column(JSON, nullable=True)  # e.g., ["hi", "en", "ta"]
    language_confidence = Column(JSON, nullable=True)
    
    # Overall Audio Score
    audio_deepfake_score = Column(Float, nullable=False)  # 0-1 final audio score
    
    # Metadata
    audio_duration_seconds = Column(Float, nullable=True)
    sample_rate = Column(Integer, nullable=True)
    processed_at = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(50))
    
    def __repr__(self):
        return f"<AudioAnalysis(video_id={self.video_id}, score={self.audio_deepfake_score})>"


# ============================================================================
# SYNC ANALYSIS RESULTS TABLE (Module 3)
# ============================================================================
class SyncAnalysis(Base):
    """Lip-Sync & Regional Phoneme-Viseme Synchronization Results"""
    __tablename__ = "sync_analysis"
    
    id = Column(String(36), primary_key=True)
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False)
    video = relationship("Video", back_populates="sync_result")
    
    # Speech Recognition (Whisper - Multilingual)
    transcription = Column(Text, nullable=True)
    detected_language = Column(String(10), nullable=True)  # ISO 639-1 code
    language_confidence = Column(Float, nullable=True)
    phoneme_timestamps = Column(JSON, nullable=True)  # Phoneme-level timing
    
    # Lip-Sync Metrics
    viseme_extraction_confidence = Column(Float, nullable=True)
    mouth_region_stability = Column(Float, nullable=True)  # Temporal consistency
    
    # Cross-Modal Synchronization (Modified SyncNet)
    phoneme_viseme_sync_score = Column(Float, nullable=True)  # 0-1 cosine similarity
    sync_match_percentage = Column(Float, nullable=True)  # % of phonemes matching visemes
    lip_sync_mismatches = Column(JSON, nullable=True)  # Timestamp of mismatches
    
    # Voice Dubbing Detection
    voice_dubbing_likelihood = Column(Float, nullable=True)  # 0-1 dubbing score
    dubbed_regions = Column(JSON, nullable=True)  # Timestamp ranges
    
    # Overall Sync Score
    sync_deepfake_score = Column(Float, nullable=False)  # 0-1 final sync score
    
    # Metadata
    processed_at = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(50))
    
    def __repr__(self):
        return f"<SyncAnalysis(video_id={self.video_id}, score={self.sync_deepfake_score})>"


# ============================================================================
# FINAL MULTIMODAL FUSION RESULTS TABLE (Module 5)
# ============================================================================
class FinalResult(Base):
    """Multimodal Risk Score & Final Verdict"""
    __tablename__ = "final_results"
    
    id = Column(String(36), primary_key=True)
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False, unique=True)
    video = relationship("Video", back_populates="final_result")
    
    # Risk Score Calculation
    risk_score = Column(Float, nullable=False)  # 0-100%
    verdict = Column(String(50), nullable=False)  # "AUTHENTIC", "INCONCLUSIVE", "DEEPFAKE"
    confidence_level = Column(String(20), nullable=False)  # "LOW", "MEDIUM", "HIGH"
    
    # Component Weights (for explainability)
    vision_score = Column(Float, nullable=False)
    audio_score = Column(Float, nullable=False)
    sync_score = Column(Float, nullable=False)
    
    # Final Weights Used
    vision_weight = Column(Float, default=0.35)
    audio_weight = Column(Float, default=0.35)
    sync_weight = Column(Float, default=0.30)
    
    # Inconsistency Bonus
    cross_modal_inconsistency_bonus = Column(Float, nullable=True)
    
    # Summary & Key Findings
    summary = Column(Text, nullable=True)
    key_findings = Column(JSON, nullable=True)  # List of top anomalies
    disclaimer = Column(Text, nullable=True)
    
    # Temporal Heatmap (for dashboard visualization)
    frame_anomaly_timeline = Column(JSON, nullable=True)  # Frame-by-frame risk
    
    # Metadata
    generated_at = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(50))
    
    def __repr__(self):
        return f"<FinalResult(video_id={self.video_id}, verdict={self.verdict}, risk={self.risk_score})>"


# ============================================================================
# AUDIT LOG TABLE
# ============================================================================
class AuditLog(Base):
    """Complete audit trail for compliance & transparency"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(36), ForeignKey("videos.id"), nullable=False)
    video = relationship("Video", back_populates="audit_logs")
    
    # Log Details
    action = Column(String(100), nullable=False)  # e.g., "video_uploaded", "processing_started", "result_generated"
    actor = Column(String(255), nullable=True)  # User/system that triggered action
    status = Column(String(50), nullable=False)  # "success", "failure"
    details = Column(JSON, nullable=True)  # Additional context
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<AuditLog(video_id={self.video_id}, action={self.action}, timestamp={self.timestamp})>"


# ============================================================================
# BATCH JOB TABLE (For WhatsApp & API batch processing)
# ============================================================================
class BatchJob(Base):
    """Track batch processing jobs for high-volume submissions"""
    __tablename__ = "batch_jobs"
    
    id = Column(String(36), primary_key=True)
    job_name = Column(String(255), nullable=False)
    submission_source = Column(String(50))  # "whatsapp", "api", "web"
    
    # Status
    status = Column(String(50), default="queued")  # queued, processing, completed, failed
    total_videos = Column(Integer, nullable=False)
    processed_videos = Column(Integer, default=0)
    progress_percent = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    memberships = relationship("BatchVideoMembership", back_populates="batch", cascade="all, delete-orphan")


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================
def init_db(database_url: str):
    """Initialize database with all tables"""
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, SessionLocal


if __name__ == "__main__":
    # For manual initialization
    from config import DATABASE_URL
    engine, SessionLocal = init_db(DATABASE_URL)
    print("✅ Database initialized successfully!")
