"""
Main Detection Pipeline Orchestrator
Coordinates all modules (vision, audio, sync, fusion)
"""

import logging
import uuid
from pathlib import Path
from typing import Dict
from datetime import datetime

from modules.vision_ai import VisionAIModule
from modules.audio_analysis import AudioAnalysisModule
from modules.sync_analysis import SyncAnalysisModule
from modules.fusion import MultimodalFusionModule

logger = logging.getLogger(__name__)


class DeepfakeDetectionPipeline:
    """
    Complete end-to-end deepfake detection pipeline
    Processes video through all 5 modules and produces explainable results
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize all detection modules"""
        logger.info("Initializing Deepfake Detection Pipeline...")
        
        self.device = device
        self.vision_module = VisionAIModule(device=device)
        self.audio_module = AudioAnalysisModule(device=device)
        self.sync_module = SyncAnalysisModule()
        self.fusion_module = MultimodalFusionModule(
            vision_weight=0.35,
            audio_weight=0.35,
            sync_weight=0.30
        )
        
        logger.info("✅ All modules loaded successfully")
    
    def process_video(self, video_path: str, save_results: bool = False) -> Dict:
        """
        Process video through complete pipeline
        
        Args:
            video_path: Path to video file
            save_results: Whether to save results to file
            
        Returns:
            Complete analysis result with verdict
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {Path(video_path).name}")
        logger.info(f"{'='*60}\n")
        
        try:
            result_id = str(uuid.uuid4())[:8]
            
            # MODULE 1: Vision Analysis
            logger.info("[1/5] Running Vision AI Analysis...")
            vision_result = self.vision_module.analyze_video(video_path)
            logger.info(f"✓ Vision Score: {vision_result['vision_deepfake_score']:.3f}")
            
            # MODULE 2: Audio Analysis
            logger.info("[2/5] Running Audio Analysis...")
            audio_result = self.audio_module.analyze_video_audio(video_path)
            logger.info(f"✓ Audio Score: {audio_result['audio_deepfake_score']:.3f}")
            logger.info(f"✓ Language: {audio_result.get('detected_languages', ['Unknown'])[0]}")
            
            # Extract audio for sync module
            import torchaudio
            try:
                audio_waveform, sr = torchaudio.load(video_path)
                if audio_waveform.shape[0] > 1:
                    audio_waveform = audio_waveform.mean(dim=0)
                audio_np = audio_waveform.numpy()
            except:
                audio_np = None
            
            # MODULE 3: Sync Analysis
            logger.info("[3/5] Running Lip-Sync & Phoneme-Viseme Analysis...")
            if audio_np is not None:
                sync_result = self.sync_module.analyze_sync(video_path, audio_np, sr=sr)
            else:
                logger.warning("Could not extract audio for sync analysis")
                sync_result = self.sync_module._create_empty_result()
            logger.info(f"✓ Sync Score: {sync_result['sync_deepfake_score']:.3f}")
            
            # MODULE 4: Multimodal Fusion
            logger.info("[4/5] Fusing Multimodal Results...")
            total_frames = vision_result.get("num_faces_detected", 100) * 5  # Rough estimate
            fusion_result = self.fusion_module.fuse_results(
                vision_result, audio_result, sync_result, total_frames
            )
            
            logger.info(f"✓ Final Verdict: {fusion_result['verdict']}")
            logger.info(f"✓ Risk Score: {fusion_result['risk_score']:.1f}%")
            logger.info(f"✓ Confidence: {fusion_result['confidence_level']}")
            
            # Compile final output
            final_output = {
                "analysis_id": result_id,
                "timestamp": datetime.utcnow().isoformat(),
                "video_path": str(video_path),
                
                # Component results
                "vision_analysis": vision_result,
                "audio_analysis": audio_result,
                "sync_analysis": sync_result,
                
                # Final result
                "final_result": fusion_result,
                
                # Summary
                "summary": {
                    "verdict": fusion_result["verdict"],
                    "risk_score": fusion_result["risk_score"],
                    "confidence": fusion_result["confidence_level"],
                    "component_scores": {
                        "vision": fusion_result["vision_score"],
                        "audio": fusion_result["audio_score"],
                        "sync": fusion_result["sync_score"]
                    }
                }
            }
            
            logger.info(f"\n✅ Analysis complete (ID: {result_id})")
            return final_output
        
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Utility function for quick analysis
def analyze(video_path: str) -> Dict:
    """Quick analysis function"""
    pipeline = DeepfakeDetectionPipeline()
    return pipeline.process_video(video_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    result = analyze(video_path)
    
    # Print summary
    if "final_result" in result:
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(result["final_result"]["summary"])
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
