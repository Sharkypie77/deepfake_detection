"""
Module 1: Spatial-Frequency Vision AI
Detects facial boundaries, eye irregularities, compression artifacts
Language-agnostic (works on visual features only)
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import librosa
from scipy import signal
from pathlib import Path
import logging
from typing import Dict, Tuple, List
from facenet_pytorch import MTCNN
import timm
from config import VISION_CHECKPOINT_PATH
from checkpoint_provenance import checkpoint_status

logger = logging.getLogger(__name__)


class FaceDetector:
    """PyTorch MTCNN face detection with landmark extraction."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.detector = MTCNN(device=device, keep_all=False)
        logger.info(f"Face detector initialized on {device}")
    
    def extract_faces_and_landmarks(self, video_path: str, sample_every_n_frames: int = 5) -> Dict:
        """
        Extract face bounding boxes and 68 landmarks from video
        
        Args:
            video_path: Path to video file
            sample_every_n_frames: Extract face from every Nth frame
            
        Returns:
            Dict with face locations, landmarks, and temporal stability
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        face_boxes = []
        landmarks_list = []
        frame_indices = []
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_every_n_frames == 0:
                # Convert BGR to RGB for MTCNN
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect face
                boxes, probs, landmarks = self.detector.detect(frame_rgb, landmarks=True)
                
                if boxes is not None and len(boxes) > 0:
                    # Store first detected face
                    box = boxes[0]
                    landmark = landmarks[0] if landmarks is not None else None
                    
                    face_boxes.append(box)
                    landmarks_list.append(landmark)
                    frame_indices.append(frame_idx)
            
            frame_idx += 1
        
        cap.release()
        
        # Calculate temporal stability (variance in face location)
        if len(face_boxes) > 0:
            face_boxes = np.array(face_boxes)
            location_variance = np.var(face_boxes, axis=0).mean()
            stability_score = 1.0 / (1.0 + location_variance)  # Normalize to 0-1
        else:
            stability_score = 0.0
        
        return {
            "face_boxes": face_boxes,
            "landmarks": landmarks_list,
            "frame_indices": frame_indices,
            "num_faces_detected": len(face_boxes),
            "face_location_stability": float(stability_score),
            "fps": fps,
            "total_frames": total_frames
        }


class FrequencyAnalyzer:
    """FFT & DCT analysis for detecting GAN artifacts in spatial-frequency domain"""
    
    @staticmethod
    def compute_fft_spectrum(frame: np.ndarray, num_bands: int = 16) -> np.ndarray:
        """
        Compute FFT spectrum and divide into frequency bands
        Captures high-frequency GAN artifacts (unnatural checkerboard patterns)
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute 2D FFT
        f_transform = np.fft.fft2(frame)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Log scale for better visualization
        magnitude_spectrum = np.log1p(magnitude_spectrum)
        
        # Divide into frequency bands (center to outer)
        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2
        
        bands = []
        for i in range(1, num_bands + 1):
            radius = int((max(h, w) / 2) * (i / num_bands))
            y, x = np.ogrid[:h, :w]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            band_energy = magnitude_spectrum[mask].mean()
            bands.append(band_energy)
        
        return np.array(bands)
    
    @staticmethod
    def compute_dct_artifacts(frame: np.ndarray, block_size: int = 8) -> float:
        """
        Detect compression artifacts via DCT analysis
        H.264/VP9 compression creates block boundaries detectable in DCT domain
        """
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame = frame.astype(np.float32)
        
        # Analyze block boundaries (common in compressed videos)
        block_discontinuities = []
        h, w = frame.shape
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = frame[y:y+block_size, x:x+block_size]
                # Compute edge variance at block boundaries
                top_edge = block[0, :].std()
                left_edge = block[:, 0].std()
                block_discontinuities.append((top_edge + left_edge) / 2)
        
        if len(block_discontinuities) == 0:
            return 0.0
        
        # Higher variance = more compression artifacts
        artifact_score = np.mean(block_discontinuities) / 255.0
        return float(np.clip(artifact_score, 0, 1))


class SpatialAnalyzer:
    """Detects facial manipulation in spatial domain"""
    
    @staticmethod
    def compute_skin_texture_consistency(frame: np.ndarray, face_box: np.ndarray) -> float:
        """
        Analyze skin texture continuity
        AI-generated faces often have unnatural, blurry, or overly smooth textures
        """
        x1, y1, x2, y2 = face_box.astype(int)
        face_region = frame[max(0, y1):min(frame.shape[0], y2), 
                           max(0, x1):min(frame.shape[1], x2)]
        
        if face_region.size == 0:
            return 0.5
        
        # Convert to LAB color space for skin analysis
        face_lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
        
        # Compute Laplacian (texture detail)
        laplacian = cv2.Laplacian(face_lab[:,:,0], cv2.CV_64F)
        texture_variance = laplacian.var()
        
        # Normalize: very low variance = unnatural smoothness
        texture_score = min(1.0, texture_variance / 100.0)
        return float(texture_score)
    
    @staticmethod
    def compute_eye_blink_regularity(landmarks: List[np.ndarray]) -> float:
        """
        Track eye aspect ratio (EAR) across frames
        AI faces often have unnatural blinking patterns
        """
        if len(landmarks) < 10:
            return 0.5  # Not enough frames to analyze
        
        eye_aspect_ratios = []
        
        for landmark in landmarks:
            if landmark is None:
                continue
            
            # Eye landmarks in MTCNN: indices depend on detector output
            # Approximate left eye and right eye using facial landmarks
            try:
                # Left eye (rough approximation)
                left_eye = landmark[0:3]  # 3 points
                # Right eye
                right_eye = landmark[3:6]  # 3 points
                
                # Compute aspect ratio
                left_ear = np.linalg.norm(left_eye[1] - left_eye[2]) / np.linalg.norm(left_eye[0] - left_eye[1])
                right_ear = np.linalg.norm(right_eye[1] - right_eye[2]) / np.linalg.norm(right_eye[0] - right_eye[1])
                
                eye_aspect_ratios.append((left_ear + right_ear) / 2)
            except:
                continue
        
        if len(eye_aspect_ratios) == 0:
            return 0.5
        
        # Natural blinking should have some variance
        ear_variance = np.var(eye_aspect_ratios)
        blink_score = min(1.0, ear_variance / 0.01)  # Normalize
        
        return float(blink_score)


class VisionAIModule:
    """
    Multimodal Vision AI Module
    Combines spatial domain, frequency domain, and face detection
    Language-agnostic: Works on any video regardless of language
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.face_detector = FaceDetector(device="cpu")  # MTCNN prefers CPU
        self.freq_analyzer = FrequencyAnalyzer()
        self.spatial_analyzer = SpatialAnalyzer()
        
        self.backbone = timm.create_model('efficientnetv2_rw_s', pretrained=True, num_classes=2)
        if VISION_CHECKPOINT_PATH:
            checkpoint_path = Path(VISION_CHECKPOINT_PATH).expanduser()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Vision checkpoint not found: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            self.checkpoint_status = checkpoint_status("vision", str(checkpoint_path), checkpoint)
            if not self.checkpoint_status["validated"]:
                logger.warning("Vision checkpoint is unvalidated: %s", self.checkpoint_status["reason"])
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            state_dict = {key.removeprefix("backbone."): value for key, value in state_dict.items()}
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Invalid vision checkpoint {checkpoint_path}; missing={missing}, unexpected={unexpected}"
                )
            logger.info("Fine-tuned vision checkpoint loaded from %s", checkpoint_path)
        else:
            self.checkpoint_status = {
                "validated": False,
                "reason": "imagenet-only, no fine-tuned checkpoint",
            }
            logger.warning("Vision AI is using ImageNet weights only; no deepfake-tuned checkpoint configured")
        self.backbone = self.backbone.to(device)
        self.backbone.eval()
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Vision AI Module initialized on {device}")
    
    def analyze_video(self, video_path: str, sample_rate: int = 5) -> Dict:
        """
        Complete vision analysis pipeline
        
        Returns:
            Dict with all vision scores and metrics
        """
        logger.info(f"Starting vision analysis for {video_path}")
        
        # Step 1: Face detection and landmark extraction
        face_data = self.face_detector.extract_faces_and_landmarks(video_path, sample_rate)
        
        if face_data["num_faces_detected"] == 0:
            logger.warning("No faces detected in video")
            return self._create_empty_result()
        
        # Step 2: Frame-by-frame analysis
        cap = cv2.VideoCapture(video_path)
        fft_bands_list = []
        dct_artifacts = []
        texture_scores = []
        
        frame_idx = 0
        for sample_frame_idx in face_data["frame_indices"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # FFT analysis
            fft_bands = self.freq_analyzer.compute_fft_spectrum(frame)
            fft_bands_list.append(fft_bands)
            
            # DCT compression artifact detection
            dct_score = self.freq_analyzer.compute_dct_artifacts(frame)
            dct_artifacts.append(dct_score)
            
            # Spatial analysis
            if frame_idx < len(face_data["face_boxes"]):
                face_box = face_data["face_boxes"][frame_idx]
                texture_score = self.spatial_analyzer.compute_skin_texture_consistency(frame, face_box)
                texture_scores.append(texture_score)
            
            frame_idx += 1
        
        cap.release()
        
        # Step 3: Eye blink analysis
        blink_score = self.spatial_analyzer.compute_eye_blink_regularity(face_data["landmarks"])
        
        # Step 4: Aggregate scores
        fft_bands_array = np.array(fft_bands_list)
        gan_artifact_score = 1.0 - (np.mean(fft_bands_array) / np.max(fft_bands_array + 1e-6))
        
        dct_mean = np.mean(dct_artifacts) if len(dct_artifacts) > 0 else 0.0
        
        texture_mean = np.mean(texture_scores) if len(texture_scores) > 0 else 0.5
        
        # Boundary warping detection (via face box stability variance)
        boundary_warping = 1.0 - face_data["face_location_stability"]
        
        # Final vision score (weighted combination)
        vision_score = (
            gan_artifact_score * 0.30 +
            dct_mean * 0.20 +
            boundary_warping * 0.20 +
            (1.0 - blink_score) * 0.15 +
            (1.0 - texture_mean) * 0.15
        )
        
        result = {
            "boundary_warping_score": float(boundary_warping),
            "eye_blinking_regularity": float(blink_score),
            "skin_texture_consistency": float(texture_mean),
            "gan_artifact_score": float(gan_artifact_score),
            "dct_compression_artifacts": float(dct_mean),
            "num_faces_detected": face_data["num_faces_detected"],
            "face_location_stability": float(face_data["face_location_stability"]),
            "vision_deepfake_score": float(vision_score),
            "fft_spectrum_analysis": fft_bands_list,
            "model_version": "vision-v1.0"
        }
        
        logger.info(f"Vision analysis complete. Score: {vision_score:.3f}")
        return result
    
    def _create_empty_result(self) -> Dict:
        """Return neutral scores when analysis fails"""
        return {
            "boundary_warping_score": 0.5,
            "eye_blinking_regularity": 0.5,
            "skin_texture_consistency": 0.5,
            "gan_artifact_score": 0.5,
            "dct_compression_artifacts": 0.5,
            "num_faces_detected": 0,
            "face_location_stability": 0.0,
            "vision_deepfake_score": 0.5,
            "fft_spectrum_analysis": [],
            "model_version": "vision-v1.0"
        }


# Export
__all__ = ["VisionAIModule", "FaceDetector", "FrequencyAnalyzer", "SpatialAnalyzer"]
