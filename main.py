"""
FastAPI Backend for Deepfake Detection Suite
Handles video uploads, async processing, and results delivery
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query, Request
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional
import json
import time
from collections import defaultdict

from config import (
    API_HOST, API_PORT, API_TITLE, API_VERSION, API_DESCRIPTION,
    VIDEO_UPLOAD_DIR, RESULTS_DIR, DATABASE_URL,
    ModelConfig, ProcessingConfig, SecurityConfig
)
from models import Video, VisionAnalysis, AudioAnalysis, SyncAnalysis, FinalResult, AuditLog, BatchJob, BatchVideoMembership, init_db
from pipeline import DeepfakeDetectionPipeline
from pydantic import BaseModel
from typing import List


# ============================================================================
# REQUEST MODELS (Pydantic)
# ============================================================================

class BatchSubmitRequest(BaseModel):
    job_name: str
    video_ids: List[str]
    priority: str = "normal"  # low, normal, high
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_name": "election_batch_phase_1",
                "video_ids": ["vid_001", "vid_002", "vid_003"],
                "priority": "high"
            }
        }


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        })


logging.basicConfig(level=logging.INFO, format="%(message)s")
for handler in logging.getLogger().handlers:
    handler.setFormatter(JsonFormatter())
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION
)
_rate_windows = defaultdict(list)
_request_count = 0

@app.middleware("http")
async def request_metrics(request: Request, call_next):
    global _request_count
    _request_count += 1
    return await call_next(request)

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
    logger.critical("Failed to initialize detection pipeline: %s", e, exc_info=True)
    raise


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
        ,"models_validated": bool(getattr(detection_pipeline, "models_validated", False))
        ,"mode": (
            "production" if getattr(detection_pipeline, "models_validated", False)
            else "unvalidated"
        )
        ,"checkpoints": getattr(detection_pipeline, "checkpoints", {})
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
            "results": "/api/v1/results/{video_id}"
        }
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return f"# TYPE http_requests_total counter\nhttp_requests_total {_request_count}\n"


# ============================================================================
# VIDEO ANALYSIS ENDPOINTS
# ============================================================================

@app.post("/api/v1/analyze")
async def upload_and_analyze(
    request: Request,
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
    
    now = time.monotonic()
    client = request.client.host if request.client else "unknown"
    _rate_windows[client] = [t for t in _rate_windows[client] if now - t < 60]
    if len(_rate_windows[client]) >= 10:
        raise HTTPException(status_code=429, detail="Analysis rate limit exceeded")
    _rate_windows[client].append(now)
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
async def get_results(video_id: str, format: str = Query("json", pattern="^(json|pdf)$")):
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

        if format == "pdf":
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            pdf_path = RESULTS_DIR / f"{video_id}.pdf"
            report = canvas.Canvas(str(pdf_path), pagesize=letter)
            report.setFont("Helvetica", 11)
            y = 750
            for line in (
                f"Deepfake Detection Report: {video_id}",
                f"Verdict: {final_result.verdict}",
                f"Risk: {final_result.risk_score}",
                f"Findings: {final_result.key_findings or []}",
                f"Timeline: generated {final_result.generated_at.isoformat()}",
                f"Disclaimer: {final_result.disclaimer or 'None'}",
            ):
                report.drawString(50, y, line[:110])
                y -= 24
            report.save()
            return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path.name)
        
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
            ,"disclaimer": final_result.disclaimer
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
        ,disclaimer=result.get("disclaimer")
    )
    session.add(final_result)
    
    session.commit()


# ============================================================================
# BATCH PROCESSING (Phase 6)
# ============================================================================

@app.post("/api/v1/batch/submit")
async def submit_batch_job(
    request: BatchSubmitRequest
):
    """
    Submit multiple videos for batch processing (Phase 6)
    
    Queues videos to Celery worker pool for parallel analysis
    
    Request body:
    {
        "job_name": "election_videos_batch_1",
        "video_ids": ["vid1", "vid2", "vid3"],
        "priority": "high"
    }
    
    Returns:
    {
        "batch_id": "batch_uuid",
        "total_videos": 3,
        "status": "queued",
        "task_ids": ["task1", "task2", "task3"]
    }
    """
    
    from celery_tasks import batch_analyze_videos

    if not request.video_ids:
        raise HTTPException(status_code=400, detail="At least one video_id is required")
    if request.priority not in {"low", "normal", "high"}:
        raise HTTPException(status_code=400, detail="priority must be low, normal, or high")

    session = SessionLocal()
    try:
        found = session.query(Video.id).filter(Video.id.in_(request.video_ids)).all()
        found_ids = {video_id for (video_id,) in found}
        missing_ids = [video_id for video_id in request.video_ids if video_id not in found_ids]
        if missing_ids:
            raise HTTPException(status_code=404, detail=f"Videos not found: {missing_ids}")
        batch_id = str(uuid.uuid4())
        batch = BatchJob(
            id=batch_id, job_name=request.job_name, submission_source="api",
            status="queued", total_videos=len(request.video_ids)
        )
        session.add(batch)
        session.flush()
        session.add_all([
            BatchVideoMembership(batch_id=batch_id, video_id=video_id)
            for video_id in request.video_ids
        ])
        session.commit()
    finally:
        session.close()
    
    # Queue batch job
    task = batch_analyze_videos.apply_async(
        args=(batch_id, request.video_ids),
        task_id=batch_id,
        queue='default'
    )
    
    logger.info(f"📦 Batch {batch_id} submitted: {len(request.video_ids)} videos")
    
    return {
        "batch_id": batch_id,
        "total_videos": len(request.video_ids),
        "status": "queued",
        "task_ids": [task.id],
        "message": f"Batch submitted to queue. Check status with /api/v1/batch/{batch_id}/status"
    }


@app.get("/api/v1/batch/{batch_id}/status")
async def get_batch_status(batch_id: str):
    """
    Get status of a batch job (Phase 6)
    
    Returns:
    {
        "batch_id": "batch_uuid",
        "total": 10,
        "completed": 7,
        "failed": 1,
        "processing": 2,
        "progress": 80,
        "status": "processing"
    }
    """
    
    from celery_tasks import check_batch_status
    
    result = check_batch_status.run(batch_id)
    
    return result


@app.get("/api/v1/videos/{video_id}/batches")
async def get_video_batches(video_id: str):
    session = SessionLocal()
    try:
        if not session.query(Video.id).filter(Video.id == video_id).first():
            raise HTTPException(status_code=404, detail="Video not found")
        memberships = session.query(BatchVideoMembership).filter_by(video_id=video_id).order_by(
            BatchVideoMembership.added_at.desc()
        ).all()
        return {
            "video_id": video_id,
            "batches": [
                {
                    "batch_id": membership.batch_id,
                    "added_at": membership.added_at.isoformat(),
                    "status": membership.status,
                }
                for membership in memberships
            ],
        }
    finally:
        session.close()


@app.get("/api/v1/batch/{batch_id}/results")
async def get_batch_results(batch_id: str):
    """
    Get consolidated results for a batch (Phase 6)
    
    Returns summary statistics:
    {
        "batch_id": "batch_uuid",
        "total_analyzed": 10,
        "deepfakes": 3,
        "authentic": 5,
        "inconclusive": 2,
        "deepfake_percentage": 30.0,
        "average_risk_score": 42.5,
        "flagged_for_review": [...]
    }
    """
    
    from celery_tasks import get_batch_results
    
    result = get_batch_results.run(batch_id)
    
    return result


# ============================================================================
# WHATSAPP WEBHOOK
# ============================================================================

@app.get("/webhooks/whatsapp")
async def verify_whatsapp_webhook(
    hub_mode: str = Query("", alias="hub.mode"),
    hub_verify_token: str = Query("", alias="hub.verify_token"),
    hub_challenge: str = Query("", alias="hub.challenge"),
):
    raise HTTPException(status_code=410, detail="WhatsApp integration is disabled; use Telegram or the REST API")


async def _whatsapp_download_media(media_id: str, extension: str) -> tuple[bytes, str]:
    if not WHATSAPP_API_TOKEN:
        raise RuntimeError("WHATSAPP_API_TOKEN is not configured")
    headers = {"Authorization": f"Bearer {WHATSAPP_API_TOKEN}"}
    async with aiohttp.ClientSession(headers=headers) as client:
        async with client.get(f"https://graph.facebook.com/{WHATSAPP_GRAPH_API_VERSION}/{media_id}") as response:
            response.raise_for_status()
            media = await response.json()
        async with client.get(media["url"]) as response:
            response.raise_for_status()
            return await response.read(), extension


async def _whatsapp_send_result(recipient: str, text: str):
    if not WHATSAPP_PHONE_NUMBER_ID or not WHATSAPP_API_TOKEN:
        raise RuntimeError("WhatsApp credentials are not configured")
    url = f"https://graph.facebook.com/{WHATSAPP_GRAPH_API_VERSION}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_API_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": recipient, "type": "text",
               "text": {"body": text[:4096]}}
    async with aiohttp.ClientSession(headers=headers) as client:
        async with client.post(url, json=payload) as response:
            response.raise_for_status()


async def _process_whatsapp_video(video_id: str, file_path: str, recipient: str):
    result = process_video_sync(video_id, file_path)
    if result.get("status") == "completed":
        text = f"Verdict: {result.get('verdict')}\nRisk: {result.get('risk_score')}\n{result.get('summary', '')}"
    else:
        text = f"Analysis status: {result.get('status', 'failed')}"
    await _whatsapp_send_result(recipient, text)


@app.post("/webhooks/whatsapp")
async def whatsapp_webhook(
    background_tasks: BackgroundTasks,
    body: dict
):
    raise HTTPException(status_code=410, detail="WhatsApp integration is disabled; use Telegram or the REST API")
    """
    try:
        value = body["entry"][0]["changes"][0]["value"]
        message = value["messages"][0]
        if message.get("type") != "video":
            return {"status": "ignored"}
        sender = message["from"]
        media = message["video"]
        extension = Path(media.get("filename", "upload.mp4")).suffix.lower() or ".mp4"
        content, extension = await _whatsapp_download_media(media["id"], extension)
        if len(content) > SecurityConfig.MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="File too large")
        video_id = str(uuid.uuid4())
        file_path = VIDEO_UPLOAD_DIR / f"{video_id}{extension}"
        file_path.write_bytes(content)
        session = SessionLocal()
        try:
            session.add(Video(
                id=video_id, filename=media.get("filename", f"{video_id}{extension}"),
                file_path=str(file_path), file_size_mb=len(content) / 1024 / 1024,
                status="pending", submission_platform="whatsapp", submitted_at=datetime.utcnow(),
            ))
            session.commit()
        finally:
            session.close()
        background_tasks.add_task(_process_whatsapp_video, video_id, str(file_path), sender)
        return {"status": "queued", "video_id": video_id}
    except KeyError:
        return {"status": "ignored"}
    """


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
