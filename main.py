"""
FastAPI Backend for Deepfake Detection Suite
Handles video uploads, async processing, and results delivery
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

from config import (
    API_HOST, API_PORT, API_TITLE, API_VERSION, API_DESCRIPTION,
    VIDEO_UPLOAD_DIR, RESULTS_DIR, DATABASE_URL,
    ModelConfig, ProcessingConfig, SecurityConfig
)
from models import Video, VisionAnalysis, AudioAnalysis, SyncAnalysis, FinalResult, AuditLog, init_db
from pipeline import DeepfakeDetectionPipeline

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION
)

# CORS
if SecurityConfig.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=SecurityConfig.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Initialize database
engine, SessionLocal = init_db(DATABASE_URL)
logger.info("✅ Database initialized")

# Initialize detection pipeline
try:
    detection_pipeline = DeepfakeDetectionPipeline(device=ProcessingConfig.DEVICE)
    logger.info("✅ Detection pipeline initialized")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}")
    detection_pipeline = None


# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": API_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "pipeline_ready": detection_pipeline is not None
    }


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "endpoints": {
            "health": "/health",
            "analyze": "/api/v1/analyze",
            "status": "/api/v1/status/{video_id}",
            "results": "/api/v1/results/{video_id}",
            "whatsapp_webhook": "/webhooks/whatsapp"
        }
    }


# ============================================================================
# VIDEO ANALYSIS ENDPOINTS
# ============================================================================

@app.post("/api/v1/analyze")
async def upload_and_analyze(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    async_mode: bool = True
):
    """
    Upload video for deepfake detection analysis
    
    Parameters:
    - file: Video file (MP4, AVI, MOV, MKV, etc.)
    - async_mode: If True, return immediately and process in background
    
    Returns:
    - video_id for tracking
    - analysis results (if async_mode=False)
    """
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SecurityConfig.ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported format. Allowed: {SecurityConfig.ALLOWED_VIDEO_EXTENSIONS}"
            )
        
        # Generate video ID
        video_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path = VIDEO_UPLOAD_DIR / f"{video_id}{file_ext}"
        
        content = await file.read()
        if len(content) > SecurityConfig.MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="File too large")
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Video uploaded: {video_id} ({len(content) / 1024 / 1024:.1f} MB)")
        
        # Create database entry
        session = SessionLocal()
        try:
            db_video = Video(
                id=video_id,
                filename=file.filename,
                file_path=str(file_path),
                file_size_mb=len(content) / 1024 / 1024,
                status="pending",
                submission_platform="api",
                submitted_at=datetime.utcnow()
            )
            session.add(db_video)
            session.commit()
            
            # Log audit
            audit_log = AuditLog(
                video_id=video_id,
                action="video_uploaded",
                actor="api",
                status="success",
                details={"filename": file.filename}
            )
            session.add(audit_log)
            session.commit()
        finally:
            session.close()
        
        # Process async or sync
        if async_mode:
            background_tasks.add_task(process_video_background, video_id, str(file_path))
            return {
                "status": "queued",
                "video_id": video_id,
                "message": "Video queued for processing",
                "check_status_url": f"/api/v1/status/{video_id}"
            }
        else:
            # Synchronous processing
            result = process_video_sync(video_id, str(file_path))
            return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/v1/status/{video_id}")
async def get_status(video_id: str):
    """Get analysis status for a video"""
    
    session = SessionLocal()
    try:
        video = session.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        return {
            "video_id": video_id,
            "status": video.status,
            "progress": video.progress_percent,
            "submitted_at": video.submitted_at.isoformat(),
            "processing_started_at": video.processing_started_at.isoformat() if video.processing_started_at else None,
            "error": video.error_message
        }
    finally:
        session.close()


@app.get("/api/v1/results/{video_id}")
async def get_results(video_id: str):
    """Get analysis results for a video"""
    
    session = SessionLocal()
    try:
        video = session.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if video.status != "completed":
            return {
                "status": video.status,
                "message": f"Analysis not complete. Current status: {video.status}",
                "progress": video.progress_percent
            }
        
        final_result = session.query(FinalResult).filter(FinalResult.video_id == video_id).first()
        if not final_result:
            raise HTTPException(status_code=500, detail="Results not found")
        
        return {
            "video_id": video_id,
            "status": "completed",
            "verdict": final_result.verdict,
            "risk_score": final_result.risk_score,
            "confidence": final_result.confidence_level,
            "summary": final_result.summary,
            "key_findings": final_result.key_findings,
            "component_scores": {
                "vision": final_result.vision_score,
                "audio": final_result.audio_score,
                "sync": final_result.sync_score
            },
            "generated_at": final_result.generated_at.isoformat()
        }
    finally:
        session.close()


# ============================================================================
# BACKGROUND PROCESSING
# ============================================================================

def process_video_background(video_id: str, file_path: str):
    """Background task for video processing"""
    logger.info(f"[BG] Starting processing for {video_id}")
    
    session = SessionLocal()
    try:
        video = session.query(Video).filter(Video.id == video_id).first()
        video.status = "processing"
        video.progress_percent = 5
        video.processing_started_at = datetime.utcnow()
        session.commit()
        
        # Run detection
        if detection_pipeline is None:
            raise Exception("Pipeline not initialized")
        
        result = detection_pipeline.process_video(file_path)
        
        if "error" in result:
            video.status = "failed"
            video.error_message = result["error"]
            session.commit()
            return
        
        # Save results to database
        save_results_to_db(session, video_id, result)
        
        video.status = "completed"
        video.progress_percent = 100
        video.processing_completed_at = datetime.utcnow()
        session.commit()
        
        logger.info(f"[BG] Processing complete for {video_id}")
    
    except Exception as e:
        logger.error(f"[BG] Processing failed: {e}", exc_info=True)
        try:
            video.status = "failed"
            video.error_message = str(e)
            session.commit()
        except:
            pass
    
    finally:
        session.close()


def process_video_sync(video_id: str, file_path: str) -> dict:
    """Synchronous video processing"""
    
    session = SessionLocal()
    try:
        video = session.query(Video).filter(Video.id == video_id).first()
        video.status = "processing"
        video.processing_started_at = datetime.utcnow()
        session.commit()
        
        if detection_pipeline is None:
            raise Exception("Pipeline not initialized")
        
        result = detection_pipeline.process_video(file_path)
        
        if "error" in result:
            video.status = "failed"
            video.error_message = result["error"]
            session.commit()
            return result
        
        # Save results
        save_results_to_db(session, video_id, result)
        
        video.status = "completed"
        video.processing_completed_at = datetime.utcnow()
        session.commit()
        
        return result
    
    finally:
        session.close()


def save_results_to_db(session, video_id: str, result: dict):
    """Save analysis results to database"""
    
    final_result_data = result.get("final_result", {})
    
    # Vision results
    vision_data = result.get("vision_analysis", {})
    vision_analysis = VisionAnalysis(
        id=str(uuid.uuid4()),
        video_id=video_id,
        boundary_warping_score=vision_data.get("boundary_warping_score"),
        eye_blinking_regularity=vision_data.get("eye_blinking_regularity"),
        skin_texture_consistency=vision_data.get("skin_texture_consistency"),
        gan_artifact_score=vision_data.get("gan_artifact_score"),
        dct_compression_artifacts=vision_data.get("dct_compression_artifacts"),
        num_faces_detected=vision_data.get("num_faces_detected"),
        face_location_stability=vision_data.get("face_location_stability"),
        vision_deepfake_score=vision_data.get("vision_deepfake_score", 0.5),
        model_version=vision_data.get("model_version")
    )
    session.add(vision_analysis)
    
    # Audio results
    audio_data = result.get("audio_analysis", {})
    audio_analysis = AudioAnalysis(
        id=str(uuid.uuid4()),
        video_id=video_id,
        aasist_spoofing_score=audio_data.get("aasist_spoofing_score"),
        aasist_confidence=audio_data.get("aasist_confidence"),
        spectral_cutoff_presence=audio_data.get("spectral_cutoff_presence"),
        cqcc_consistency=audio_data.get("cqcc_consistency"),
        phase_discontinuities=audio_data.get("phase_discontinuities"),
        breath_pause_detection=audio_data.get("breath_pause_detection"),
        voice_pitch_consistency=audio_data.get("voice_pitch_consistency"),
        formant_stability=audio_data.get("formant_stability"),
        detected_languages=audio_data.get("detected_languages"),
        audio_deepfake_score=audio_data.get("audio_deepfake_score", 0.5),
        audio_duration_seconds=audio_data.get("audio_duration_seconds"),
        sample_rate=audio_data.get("sample_rate"),
        model_version=audio_data.get("model_version")
    )
    session.add(audio_analysis)
    
    # Sync results
    sync_data = result.get("sync_analysis", {})
    sync_analysis = SyncAnalysis(
        id=str(uuid.uuid4()),
        video_id=video_id,
        transcription=sync_data.get("transcription"),
        detected_language=sync_data.get("detected_language"),
        language_confidence=sync_data.get("language_confidence"),
        viseme_extraction_confidence=sync_data.get("viseme_extraction_confidence"),
        mouth_region_stability=sync_data.get("mouth_region_stability"),
        phoneme_viseme_sync_score=sync_data.get("phoneme_viseme_sync_score"),
        sync_match_percentage=sync_data.get("sync_match_percentage"),
        lip_sync_mismatches=sync_data.get("lip_sync_mismatches"),
        voice_dubbing_likelihood=sync_data.get("voice_dubbing_likelihood"),
        sync_deepfake_score=sync_data.get("sync_deepfake_score", 0.5),
        model_version=sync_data.get("model_version")
    )
    session.add(sync_analysis)
    
    # Final result
    final_result = FinalResult(
        id=str(uuid.uuid4()),
        video_id=video_id,
        risk_score=final_result_data.get("risk_score", 0),
        verdict=final_result_data.get("verdict", "INCONCLUSIVE"),
        confidence_level=final_result_data.get("confidence_level", "LOW"),
        vision_score=final_result_data.get("vision_score", 0.5),
        audio_score=final_result_data.get("audio_score", 0.5),
        sync_score=final_result_data.get("sync_score", 0.5),
        summary=final_result_data.get("summary", ""),
        key_findings=final_result_data.get("key_findings", []),
        frame_anomaly_timeline=final_result_data.get("frame_anomaly_timeline", []),
        model_version=final_result_data.get("model_version")
    )
    session.add(final_result)
    
    session.commit()


# ============================================================================
# WHATSAPP WEBHOOK (Placeholder)
# ============================================================================

@app.post("/webhooks/whatsapp")
async def whatsapp_webhook(
    background_tasks: BackgroundTasks,
    body: dict
):
    """
    WhatsApp Cloud API webhook for receiving messages
    Will be implemented in Phase 5
    """
    logger.info(f"WhatsApp webhook received: {body}")
    
    return {
        "status": "received",
        "message": "WhatsApp bot integration coming in Phase 5"
    }


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup():
    logger.info("🚀 Deepfake Detection API starting...")
    logger.info(f"Device: {ProcessingConfig.DEVICE}")
    logger.info(f"Max upload: {SecurityConfig.MAX_UPLOAD_SIZE_BYTES / 1024 / 1024:.0f} MB")


@app.on_event("shutdown")
async def shutdown():
    logger.info("🛑 Deepfake Detection API shutting down...")


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info"
    )
