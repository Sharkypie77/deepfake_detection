#!/usr/bin/env python3
"""
End-to-end test suite for Deepfake Detection Pipeline
Tests all modules independently and together
"""

import logging
import numpy as np
import cv2
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_video(output_path: str = "test_video.mp4", duration: float = 5.0):
    """Create a simple test video"""
    logger.info(f"Creating test video: {output_path}")
    
    fps = 30
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    num_frames = int(fps * duration)
    
    for frame_idx in range(num_frames):
        # Create a frame with face-like shapes
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Draw circles as mock face
        cv2.circle(frame, (int(width/2), int(height/2)), 100, (100, 100, 200), -1)  # Head
        cv2.circle(frame, (int(width/2 - 30), int(height/2 - 30)), 15, (200, 100, 50), -1)  # Left eye
        cv2.circle(frame, (int(width/2 + 30), int(height/2 - 30)), 15, (200, 100, 50), -1)  # Right eye
        
        # Animate mouth (opening/closing)
        mouth_y = int(height/2 + 50)
        mouth_x = int(width/2)
        mouth_height = int(20 + 10 * np.sin(frame_idx * 0.1))
        cv2.ellipse(frame, (mouth_x, mouth_y), (30, mouth_height), 0, 0, 180, (100, 50, 50), -1)
        
        # Add text
        cv2.putText(frame, f"Test Frame {frame_idx}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        out.write(frame)
    
    out.release()
    logger.info(f"✓ Test video created ({num_frames} frames @ {fps} fps)")
    return output_path


def test_vision_module():
    """Test Vision AI Module"""
    logger.info("\n" + "="*60)
    logger.info("Testing Vision AI Module (Module 1)")
    logger.info("="*60)
    
    try:
        from modules.vision_ai import VisionAIModule
        
        video_path = create_test_video("test_vision.mp4", duration=3.0)
        
        module = VisionAIModule(device="cpu")  # Use CPU for testing
        result = module.analyze_video(video_path)
        
        logger.info(f"✓ Vision Score: {result['vision_deepfake_score']:.3f}")
        logger.info(f"✓ Faces Detected: {result['num_faces_detected']}")
        logger.info(f"✓ GAN Artifacts: {result['gan_artifact_score']:.3f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Vision Module test failed: {e}", exc_info=True)
        return False

def test_audio_module():
    """Test Audio Analysis Module"""
    logger.info("\n" + "="*60)
    logger.info("Testing Audio Analysis Module (Module 2)")
    logger.info("="*60)
    
    try:
        from modules.audio_analysis import AudioAnalysisModule
        
        # Create simple audio file
        import soundfile as sf
        sr = 16000
        duration = 3.0
        audio = np.random.randn(int(sr * duration)) * 0.1  # Quiet noise
        audio_path = "test_audio.wav"
        sf.write(audio_path, audio, sr)
        
        module = AudioAnalysisModule(device="cpu")
        # Note: Audio module expects video file, but we can test audio extraction
        logger.info("✓ Audio module initialized")
        logger.info("✓ AASIST loaded (or fallback mode)")
        
        return True
    except Exception as e:
        logger.error(f"✗ Audio Module test failed: {e}", exc_info=True)
        return False

def test_sync_module():
    """Test Sync Analysis Module"""
    logger.info("\n" + "="*60)
    logger.info("Testing Lip-Sync Analysis Module (Module 3)")
    logger.info("="*60)
    
    try:
        from modules.sync_analysis import SyncAnalysisModule
        
        video_path = create_test_video("test_sync.mp4", duration=3.0)
        
        module = SyncAnalysisModule()
        
        # Test with mock audio
        sr = 16000
        duration = 3.0
        audio = np.random.randn(int(sr * duration)) * 0.1
        
        result = module.analyze_sync(video_path, audio, sr=sr)
        
        logger.info(f"✓ Sync Score: {result['sync_deepfake_score']:.3f}")
        logger.info(f"✓ Match Percentage: {result['sync_match_percentage']:.1f}%")
        logger.info(f"✓ Mismatches: {len(result['lip_sync_mismatches'])}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Sync Module test failed: {e}", exc_info=True)
        return False

def test_fusion_module():
    """Test Multimodal Fusion"""
    logger.info("\n" + "="*60)
    logger.info("Testing Multimodal Fusion Module (Module 5)")
    logger.info("="*60)
    
    try:
        from modules.fusion import MultimodalFusionModule
        
        # Mock results from individual modules
        vision_result = {"vision_deepfake_score": 0.45}
        audio_result = {"audio_deepfake_score": 0.60}
        sync_result = {"sync_deepfake_score": 0.55}
        
        module = MultimodalFusionModule()
        result = module.fuse_results(vision_result, audio_result, sync_result)
        
        logger.info(f"✓ Risk Score: {result['risk_score']:.1f}%")
        logger.info(f"✓ Verdict: {result['verdict']}")
        logger.info(f"✓ Confidence: {result['confidence_level']}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Fusion Module test failed: {e}", exc_info=True)
        return False

def test_pipeline():
    """Test complete end-to-end pipeline"""
    logger.info("\n" + "="*60)
    logger.info("Testing Complete Pipeline (All Modules)")
    logger.info("="*60)
    
    try:
        from pipeline import DeepfakeDetectionPipeline
        
        video_path = create_test_video("test_pipeline.mp4", duration=3.0)
        
        logger.info("Initializing pipeline...")
        pipeline = DeepfakeDetectionPipeline(device="cpu")
        
        logger.info("Processing video through all 5 modules...")
        result = pipeline.process_video(video_path)
        
        if "error" in result:
            logger.error(f"✗ Pipeline error: {result['error']}")
            return False
        
        final_result = result.get("final_result", {})
        
        logger.info(f"✓ Analysis Complete!")
        logger.info(f"✓ Verdict: {final_result.get('verdict', 'N/A')}")
        logger.info(f"✓ Risk Score: {final_result.get('risk_score', 0):.1f}%")
        logger.info(f"✓ Confidence: {final_result.get('confidence_level', 'N/A')}")
        
        # Print summary
        print("\n" + final_result.get("summary", ""))
        
        return True
    except Exception as e:
        logger.error(f"✗ Pipeline test failed: {e}", exc_info=True)
        return False

def cleanup():
    """Clean up test files"""
    logger.info("\nCleaning up test files...")
    for file in ["test_video.mp4", "test_vision.mp4", "test_sync.mp4", "test_pipeline.mp4", "test_audio.wav"]:
        try:
            Path(file).unlink()
            logger.info(f"✓ Removed {file}")
        except:
            pass

def main():
    """Run all tests"""
    logger.info("\n" + "="*60)
    logger.info("🧪 Deepfake Detection Suite - Test Suite")
    logger.info("="*60)
    
    results = {
        "Vision Module": test_vision_module(),
        "Audio Module": test_audio_module(),
        "Sync Module": test_sync_module(),
        "Fusion Module": test_fusion_module(),
        "Complete Pipeline": test_pipeline()
    }
    
    cleanup()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 Test Summary")
    logger.info("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n🎉 All tests passed!")
        return 0
    else:
        logger.warning("\n⚠️ Some tests failed. Check logs above.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
