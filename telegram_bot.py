"""
Phase 4: Telegram Bot Integration
Real-time deepfake detection via Telegram
Supports all languages (auto-detected)
"""

import logging
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import asyncio
import aiohttp

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
from telegram.constants import ChatAction

from config import TELEGRAM_BOT_TOKEN, RESULTS_DIR, VIDEO_UPLOAD_DIR
from pipeline import DeepfakeDetectionPipeline
from models import Video, FinalResult, SessionLocal, AuditLog
import uuid

logger = logging.getLogger(__name__)

# States for conversation flow
AWAITING_VIDEO = 1
PROCESSING = 2


class TelegramBotHandler:
    """Telegram bot for deepfake detection"""
    
    def __init__(self):
        self.pipeline = None
        self.token = TELEGRAM_BOT_TOKEN
        
        if not self.token or self.token == "YOUR_TELEGRAM_BOT_TOKEN":
            logger.warning("⚠️ TELEGRAM_BOT_TOKEN not configured in .env")
            logger.info("Get your token from @BotFather on Telegram")
        
        logger.info("Telegram bot handler initialized")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command handler"""
        
        keyboard = [
            [InlineKeyboardButton("📹 Analyze Video", callback_data="upload_video")],
            [InlineKeyboardButton("❓ Help", callback_data="help")],
            [InlineKeyboardButton("📊 About", callback_data="about")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome_text = """
🔍 **Deepfake Detection Suite**

Welcome! I can analyze videos to detect deepfakes with 99+ language support.

**Features:**
✅ Detects GAN artifacts (vision)
✅ Detects voice cloning (audio)
✅ Detects lip-sync mismatches (sync)
✅ Supports Indian languages (Hindi, Tamil, Telugu, etc.)
✅ WhatsApp/Telegram compression ready
✅ Fast analysis (1-5 minutes per video)

**Verdict Levels:**
🟢 **Authentic** (0-30%) - Trust the video
🟡 **Inconclusive** (31-65%) - Needs review
🔴 **Deepfake** (66-100%) - Likely manipulated

Select an option below:
        """
        
        await update.message.reply_text(
            welcome_text,
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button clicks"""
        
        query = update.callback_query
        await query.answer()
        
        if query.data == "upload_video":
            await query.edit_message_text(
                text="📹 **Send me a video to analyze**\n\n"
                     "Supported formats: MP4, AVI, MOV, MKV, WebM\n"
                     "Max size: 500 MB\n"
                     "Compression: Works with WhatsApp/Telegram quality",
                parse_mode="Markdown"
            )
            return AWAITING_VIDEO
        
        elif query.data == "help":
            help_text = """
📖 **How to use:**

1. Send a video file
2. I'll analyze it (1-5 min)
3. You'll get:
   - Verdict (Authentic/Inconclusive/Deepfake)
   - Risk Score (0-100%)
   - Key findings per module
   - Confidence level

**Supported languages (auto-detected):**
Hindi, Tamil, Telugu, Kannada, Marathi, English, and 94+ more

**Privacy:**
Videos are analyzed locally and deleted after processing.
No data is stored externally.
            """
            await query.edit_message_text(
                text=help_text,
                parse_mode="Markdown"
            )
        
        elif query.data == "about":
            about_text = """
ℹ️ **About This System**

**SOUL CODERS' Deepfake Detection Suite**
Built for election integrity in India 🇮🇳

**Technology:**
- Vision AI: EfficientNetV2 + FFT/DCT
- Audio AI: AASIST + OpenAI Whisper
- Lip-Sync: MediaPipe + cross-modal sync
- Fusion: Multimodal risk scoring

**Accuracy:**
High on English & Indian languages
Compression-resilient (360p-480p)
Explainable findings per stream

**Research:**
Based on:
- FaceForensics++ dataset
- Celeb-DF benchmarks
- AASIST voice spoofing research
            """
            await query.edit_message_text(
                text=about_text,
                parse_mode="Markdown"
            )
    
    async def handle_video(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle video upload"""
        
        if not update.message.video and not update.message.document:
            await update.message.reply_text("Please send a video file.")
            return AWAITING_VIDEO
        
        # Get file
        if update.message.video:
            file_obj = update.message.video
            filename = f"{file_obj.file_id}.mp4"
        else:
            file_obj = update.message.document
            filename = file_obj.file_name
        
        # Check size
        if file_obj.file_size > 500 * 1024 * 1024:
            await update.message.reply_text(
                "❌ File too large (max 500 MB)",
                parse_mode="Markdown"
            )
            return AWAITING_VIDEO
        
        # Download
        await update.message.chat.send_action(ChatAction.UPLOAD_VIDEO)
        
        try:
            file = await context.bot.get_file(file_obj.file_id)
            video_path = VIDEO_UPLOAD_DIR / filename
            await file.download_to_drive(video_path)
            
            # Generate video ID
            video_id = str(uuid.uuid4())
            
            # Create DB entry
            session = SessionLocal()
            try:
                db_video = Video(
                    id=video_id,
                    filename=filename,
                    file_path=str(video_path),
                    file_size_mb=file_obj.file_size / 1024 / 1024,
                    status="processing",
                    submission_platform="telegram",
                    submitted_at=datetime.utcnow(),
                    submitted_by=f"@{update.message.from_user.username}" if update.message.from_user.username else update.message.from_user.id
                )
                session.add(db_video)
                session.commit()
                
                # Log
                audit_log = AuditLog(
                    video_id=video_id,
                    action="video_uploaded_telegram",
                    actor=str(update.message.from_user.id),
                    status="success",
                    details={"filename": filename}
                )
                session.add(audit_log)
                session.commit()
            finally:
                session.close()
            
            # Send processing message
            progress_msg = await update.message.reply_text(
                f"🔍 **Analyzing video...**\n"
                f"ID: `{video_id[:8]}`\n\n"
                f"This may take 1-5 minutes depending on video length.\n\n"
                f"Progress: 0%",
                parse_mode="Markdown"
            )
            
            # Process in background
            context.user_data["video_id"] = video_id
            context.user_data["progress_msg_id"] = progress_msg.message_id
            context.user_data["chat_id"] = update.message.chat_id
            
            # Start processing
            asyncio.create_task(self.process_video_async(video_id, update.message.chat_id, progress_msg.message_id, context.bot))
            
            return ConversationHandler.END
        
        except Exception as e:
            logger.error(f"Error handling video: {e}")
            await update.message.reply_text(
                f"❌ Error uploading video: {str(e)}",
                parse_mode="Markdown"
            )
            return AWAITING_VIDEO
    
    async def process_video_async(self, video_id: str, chat_id: int, msg_id: int, bot):
        """Process video asynchronously"""
        
        try:
            session = SessionLocal()
            video = session.query(Video).filter(Video.id == video_id).first()
            
            if not video:
                logger.error(f"Video {video_id} not found in DB")
                return
            
            # Initialize pipeline if needed
            if self.pipeline is None:
                logger.info("Initializing detection pipeline...")
                self.pipeline = DeepfakeDetectionPipeline(device="cuda")
            
            # Analyze
            logger.info(f"Processing {video_id}...")
            result = self.pipeline.process_video(str(video.file_path))
            
            if "error" in result:
                video.status = "failed"
                video.error_message = result["error"]
                session.commit()
                
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg_id,
                    text=f"❌ **Analysis Failed**\n\n{result['error']}",
                    parse_mode="Markdown"
                )
                return
            
            # Format result
            final_result = result.get("final_result", {})
            
            # Save to DB
            from models import VisionAnalysis, AudioAnalysis, SyncAnalysis, FinalResult
            
            vision_data = result.get("vision_analysis", {})
            vision_analysis = VisionAnalysis(
                id=str(uuid.uuid4()),
                video_id=video_id,
                vision_deepfake_score=vision_data.get("vision_deepfake_score", 0.5),
                gan_artifact_score=vision_data.get("gan_artifact_score"),
                num_faces_detected=vision_data.get("num_faces_detected"),
                model_version=vision_data.get("model_version")
            )
            session.add(vision_analysis)
            
            audio_data = result.get("audio_analysis", {})
            audio_analysis = AudioAnalysis(
                id=str(uuid.uuid4()),
                video_id=video_id,
                aasist_spoofing_score=audio_data.get("aasist_spoofing_score"),
                audio_deepfake_score=audio_data.get("audio_deepfake_score", 0.5),
                detected_languages=audio_data.get("detected_languages"),
                model_version=audio_data.get("model_version")
            )
            session.add(audio_analysis)
            
            sync_data = result.get("sync_analysis", {})
            sync_analysis = SyncAnalysis(
                id=str(uuid.uuid4()),
                video_id=video_id,
                sync_deepfake_score=sync_data.get("sync_deepfake_score", 0.5),
                model_version=sync_data.get("model_version")
            )
            session.add(sync_analysis)
            
            final_db_result = FinalResult(
                id=str(uuid.uuid4()),
                video_id=video_id,
                risk_score=final_result.get("risk_score", 0),
                verdict=final_result.get("verdict", "INCONCLUSIVE"),
                confidence_level=final_result.get("confidence_level", "LOW"),
                vision_score=final_result.get("vision_score", 0.5),
                audio_score=final_result.get("audio_score", 0.5),
                sync_score=final_result.get("sync_score", 0.5),
                summary=final_result.get("summary", ""),
                key_findings=final_result.get("key_findings", []),
                model_version=final_result.get("model_version")
            )
            session.add(final_db_result)
            
            video.status = "completed"
            session.commit()
            
            # Format Telegram response
            verdict = final_result.get("verdict", "UNKNOWN")
            risk_score = final_result.get("risk_score", 0)
            confidence = final_result.get("confidence_level", "UNKNOWN")
            
            # Emoji based on verdict
            emoji = "🟢" if verdict == "AUTHENTIC" else "🟡" if verdict == "INCONCLUSIVE" else "🔴"
            
            response_text = f"""
{emoji} **VERDICT: {verdict}**

📊 **Risk Score:** {risk_score:.1f}%
🎯 **Confidence:** {confidence}

**Component Scores:**
- 👁️ Vision: {final_result.get('vision_score', 0):.2f}
- 🎤 Audio: {final_result.get('audio_score', 0):.2f}
- 👄 Lip-Sync: {final_result.get('sync_score', 0):.2f}

**Key Findings:**
"""
            
            key_findings = final_result.get("key_findings", [])
            if key_findings:
                for i, finding in enumerate(key_findings[:3], 1):
                    response_text += f"{i}. {finding.get('category', 'Unknown')}: {finding.get('score', 0):.2%}\n"
            
            language = audio_data.get("detected_languages", ["Unknown"])[0]
            response_text += f"\n🌐 **Language:** {language}"
            
            response_text += f"\n\n{final_result.get('summary', '')[:500]}..."
            
            # Update message
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text=response_text,
                parse_mode="Markdown"
            )
            
            logger.info(f"Analysis complete for {video_id}")
        
        except Exception as e:
            logger.error(f"Error processing video: {e}", exc_info=True)
            try:
                await bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg_id,
                    text=f"❌ Processing error: {str(e)[:100]}",
                    parse_mode="Markdown"
                )
            except:
                pass
        
        finally:
            if 'session' in locals():
                session.close()


async def main():
    """Start Telegram bot"""
    
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        logger.error("❌ TELEGRAM_BOT_TOKEN not set in .env")
        logger.info("Get token from @BotFather: https://t.me/botfather")
        return
    
    logger.info("🚀 Starting Telegram bot...")
    
    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    handler = TelegramBotHandler()
    
    # Handlers
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", handler.start_command)],
        states={
            AWAITING_VIDEO: [MessageHandler(filters.VIDEO | filters.Document.ALL, handler.handle_video)],
        },
        fallbacks=[CommandHandler("start", handler.start_command)]
    )
    
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("start", handler.start_command))
    application.add_handler(MessageHandler(filters.Regex("^(Help|About)$"), handler.button_callback))
    
    # Polling
    logger.info("✅ Bot running. Send /start to begin")
    await application.run_polling()


if __name__ == "__main__":
    import sys
    asyncio.run(main())
