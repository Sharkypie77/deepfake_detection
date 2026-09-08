"""
Phase 6: Batch Processing & Celery Task Queue
High-volume video analysis with distributed workers
"""

import logging
from pathlib import Path
from celery import Celery, Task
from celery.result import AsyncResult
from datetime import datetime
from typing import Dict, List
import json

from config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND, DATABASE_URL, RETENTION_DAYS
from models import Video, BatchJob, BatchVideoMembership, FinalResult, AuditLog, init_db
from pipeline import DeepfakeDetectionPipeline
import uuid

_, SessionLocal = init_db(DATABASE_URL)

logger = logging.getLogger(__name__)

# Celery app
app = Celery(
    'deepfake_detection',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    result_expires=3600,  # 1 hour
    beat_schedule={"retention-cleanup": {
        "task": "celery_tasks.cleanup_retention",
        "schedule": 86400.0,
    }},
)


@app.task
def cleanup_retention():
    """Delete expired uploads and their database rows."""
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(days=RETENTION_DAYS)
    session = SessionLocal()
    deleted = 0
    try:
        expired = session.query(Video).filter(Video.submitted_at < cutoff).all()
        for video in expired:
            path = Path(video.file_path) if video.file_path else None
            if path and path.exists():
                path.unlink()
            session.delete(video)
            deleted += 1
        session.commit()
        return {"deleted": deleted}
    finally:
        session.close()


class CallbackTask(Task):
    """Celery task with callbacks"""
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        logger.warning(f"Task {task_id} retrying: {exc}")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed: {exc}")
        session = SessionLocal()
        try:
            video_id = args[0] if args else None
            if video_id:
                video = session.query(Video).filter(Video.id == video_id).first()
                if video:
                    video.status = "failed"
                    video.error_message = str(exc)
                    session.commit()
        finally:
            session.close()
    
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Task {task_id} completed successfully")


app.Task = CallbackTask


@app.task(bind=True, max_retries=3)
def analyze_video_task(self, video_id: str, file_path: str, batch_id: str | None = None):
    """
    Celery task for analyzing a single video
    
    Retries up to 3 times on failure
    """
    
    session = SessionLocal()
    try:
        logger.info(f"[CELERY] Starting analysis for {video_id}")
        
        # Update status
        video = session.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise ValueError(f"Video {video_id} not found")
        
        video.status = "processing"
        video.processing_started_at = datetime.utcnow()
        session.commit()
        
        # Initialize pipeline
        pipeline = DeepfakeDetectionPipeline(device="cuda")
        
        # Analyze
        result = pipeline.process_video(file_path)
        
        if "error" in result:
            raise Exception(result["error"])
        
        # Save results to DB
        _save_results_to_db(session, video_id, result)
        
        # Update video
        video.status = "completed"
        video.progress_percent = 100
        video.processing_completed_at = datetime.utcnow()
        session.commit()
        if batch_id:
            membership = session.query(BatchVideoMembership).filter_by(
                batch_id=batch_id, video_id=video_id
            ).first()
            if membership:
                membership.status = "completed"
                session.commit()
        
        logger.info(f"[CELERY] Analysis complete for {video_id}")
        return {
            "video_id": video_id,
            "status": "completed",
            "verdict": result.get("final_result", {}).get("verdict")
        }
    
    except Exception as exc:
        logger.error(f"[CELERY] Analysis failed for {video_id}: {exc}")
        if batch_id and self.request.retries >= self.max_retries:
            membership = session.query(BatchVideoMembership).filter_by(
                batch_id=batch_id, video_id=video_id
            ).first()
            if membership:
                membership.status = "failed"
                session.commit()
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
    
    finally:
        session.close()


@app.task
def batch_analyze_videos(batch_id: str, video_ids: List[str]):
    """
    Celery task for batch processing multiple videos
    
    Creates individual tasks for each video
    """
    
    logger.info(f"[BATCH] Starting batch {batch_id} with {len(video_ids)} videos")
    
    session = SessionLocal()
    try:
        batch_job = session.query(BatchJob).filter(BatchJob.id == batch_id).first()
        if not batch_job:
            batch_job = BatchJob(
                id=batch_id, job_name=f"batch_{batch_id[:8]}", submission_source="api",
                status="processing", total_videos=len(video_ids), started_at=datetime.utcnow()
            )
            session.add(batch_job)
            session.flush()
            for video_id in video_ids:
                session.add(BatchVideoMembership(batch_id=batch_id, video_id=video_id))
            session.commit()
        
        # Queue individual video tasks
        task_ids = []
        for video_id in video_ids:
            video = session.query(Video).filter(Video.id == video_id).first()
            if video:
                membership = session.query(BatchVideoMembership).filter_by(
                    batch_id=batch_id, video_id=video_id
                ).one()
                task = analyze_video_task.apply_async(
                    args=(video_id, str(video.file_path), batch_id),
                    task_id=f"{batch_id}_{video_id}",
                    queue='default'
                )
                membership.status = "processing"
                session.commit()
                task_ids.append(task.id)
                logger.info(f"[BATCH] Queued {video_id} (Task: {task.id})")
        
        return {
            "batch_id": batch_id,
            "total_tasks": len(task_ids),
            "task_ids": task_ids
        }
    
    finally:
        session.close()


@app.task
def check_batch_status(batch_id: str):
    """
    Check status of all videos in a batch
    
    Returns summary statistics
    """
    
    session = SessionLocal()
    try:
        batch = session.query(BatchJob).filter(BatchJob.id == batch_id).first()
        if not batch:
            return {"error": "Batch not found"}
        
        memberships = session.query(BatchVideoMembership).filter_by(batch_id=batch_id).all()
        
        completed = sum(1 for m in memberships if m.status == "completed")
        failed = sum(1 for m in memberships if m.status == "failed")
        processing = sum(1 for m in memberships if m.status == "processing")
        
        progress = int((completed + failed) / len(memberships) * 100) if memberships else 0
        
        # Update batch
        batch.progress_percent = progress
        batch.processed_videos = completed + failed
        
        if progress == 100:
            batch.status = "completed"
            batch.completed_at = datetime.utcnow()
        
        session.commit()
        
        return {
            "batch_id": batch_id,
            "total": len(memberships),
            "completed": completed,
            "failed": failed,
            "processing": processing,
            "progress": progress,
            "status": batch.status
        }
    
    finally:
        session.close()


@app.task
def get_batch_results(batch_id: str) -> Dict:
    """
    Get consolidated results for a batch
    
    Returns verdict summary and statistics
    """
    
    session = SessionLocal()
    try:
        batch = session.query(BatchJob).filter(BatchJob.id == batch_id).first()
        if not batch:
            return {"error": "Batch not found"}

        # Query final results for the videos explicitly assigned to this batch.
        results = session.query(FinalResult).join(
            BatchVideoMembership, BatchVideoMembership.video_id == FinalResult.video_id
        ).filter(BatchVideoMembership.batch_id == batch_id).all()
        
        if not results:
            return {"error": "No results found"}
        
        # Aggregate statistics
        deepfake_count = sum(1 for r in results if r.verdict == "DEEPFAKE")
        authentic_count = sum(1 for r in results if r.verdict == "AUTHENTIC")
        inconclusive_count = sum(1 for r in results if r.verdict == "INCONCLUSIVE")
        
        avg_risk = sum(r.risk_score for r in results) / len(results) if results else 0
        
        return {
            "batch_id": batch_id,
            "total_analyzed": len(results),
            "deepfakes": deepfake_count,
            "authentic": authentic_count,
            "inconclusive": inconclusive_count,
            "deepfake_percentage": (deepfake_count / len(results) * 100) if results else 0,
            "average_risk_score": avg_risk,
            "flagged_for_review": [
                {
                    "video_id": r.video_id,
                    "verdict": r.verdict,
                    "risk_score": r.risk_score
                }
                for r in results if r.verdict == "DEEPFAKE"
            ]
        }
    
    finally:
        session.close()


def _save_results_to_db(session, video_id: str, result: dict):
    """Helper to save results to database"""
    
    from models import VisionAnalysis, AudioAnalysis, SyncAnalysis, FinalResult
    
    final_result_data = result.get("final_result", {})
    
    # Vision
    vision_data = result.get("vision_analysis", {})
    vision = VisionAnalysis(
        id=str(uuid.uuid4()),
        video_id=video_id,
        vision_deepfake_score=vision_data.get("vision_deepfake_score", 0.5),
        model_version=vision_data.get("model_version")
    )
    session.add(vision)
    
    # Audio
    audio_data = result.get("audio_analysis", {})
    audio = AudioAnalysis(
        id=str(uuid.uuid4()),
        video_id=video_id,
        audio_deepfake_score=audio_data.get("audio_deepfake_score", 0.5),
        model_version=audio_data.get("model_version")
    )
    session.add(audio)
    
    # Sync
    sync_data = result.get("sync_analysis", {})
    sync = SyncAnalysis(
        id=str(uuid.uuid4()),
        video_id=video_id,
        sync_deepfake_score=sync_data.get("sync_deepfake_score", 0.5),
        model_version=sync_data.get("model_version")
    )
    session.add(sync)
    
    # Final result
    final = FinalResult(
        id=str(uuid.uuid4()),
        video_id=video_id,
        risk_score=final_result_data.get("risk_score", 0),
        verdict=final_result_data.get("verdict", "INCONCLUSIVE"),
        confidence_level=final_result_data.get("confidence_level", "LOW"),
        vision_score=final_result_data.get("vision_score", 0.5),
        audio_score=final_result_data.get("audio_score", 0.5),
        sync_score=final_result_data.get("sync_score", 0.5),
        model_version=final_result_data.get("model_version")
    )
    session.add(final)
    
    session.commit()


if __name__ == "__main__":
    logger.info("🚀 Starting Celery worker...")
    app.worker_main([
        'worker',
        '--loglevel=info',
        '--concurrency=4',
        '-Q', 'default'
    ])
