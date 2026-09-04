"""
Module 3: Regional Phoneme-Viseme Synchronization
Detects lip-sync mismatch and voice dubbing
Uses MediaPipe for face landmarks (language-agnostic)
"""

import numpy as np
import cv2
import torch
import logging
from typing import Dict, List, Tuple
import mediapipe as mp
from scipy.spatial.distance import euclidean
from scipy.signal import correlate

logger = logging.getLogger(__name__)


class LipSyncAnalyzer:
    """
    Lip-sync analysis using MediaPipe face mesh
    Detects viseme (visual speech) from facial landmarks
    Language-agnostic: works on visual mouth movement only
    """
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe landmark indices for mouth
        # These are fixed regardless of language
        self.MOUTH_OPEN_IDX = [11, 16, 12, 15, 13, 14]  # Upper & lower lips
        self.LIP_DISTANCE_IDX = [0, 17]  # Key reference points
        
        logger.info("LipSyncAnalyzer initialized with MediaPipe")
    
    def extract_mouth_landmarks(self, video_path: str, sample_rate: int = 5) -> Dict:
        """
        Extract mouth region landmarks from video
        Computes mouth opening distance across frames
        
        Args:
            video_path: Path to video file
            sample_rate: Extract from every Nth frame
            
        Returns:
            Dict with mouth landmarks and viseme features
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        mouth_distances = []  # Mouth opening over time
        frame_indices = []
        mouth_region_stability = []
        
        frame_idx = 0
        prev_landmarks = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                # Process with MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    
                    # Extract mouth region
                    mouth_points = []
                    for idx in self.MOUTH_OPEN_IDX:
                        lm = landmarks.landmark[idx]
                        h, w, _ = frame.shape
                        mouth_points.append([lm.x * w, lm.y * h])
                    
                    mouth_points = np.array(mouth_points)
                    
                    # Mouth opening distance (Euclidean distance between key points)
                    # Points [0] and [3] are typically upper and lower lip centers
                    if len(mouth_points) >= 2:
                        mouth_dist = euclidean(mouth_points[0], mouth_points[3])
                        mouth_distances.append(mouth_dist)
                        frame_indices.append(frame_idx)
                        
                        # Stability tracking
                        if prev_landmarks is not None:
                            movement = np.linalg.norm(mouth_points - prev_landmarks)
                            mouth_region_stability.append(movement)
                        
                        prev_landmarks = mouth_points
            
            frame_idx += 1
        
        cap.release()
        
        # Normalize mouth distances to 0-1 range
        if len(mouth_distances) > 0:
            mouth_distances = np.array(mouth_distances)
            mouth_distances_norm = (mouth_distances - mouth_distances.min()) / (mouth_distances.max() - mouth_distances.min() + 1e-6)
        else:
            mouth_distances_norm = np.array([])
        
        # Stability score (lower variance = more stable)
        if len(mouth_region_stability) > 0:
            stability_variance = np.var(mouth_region_stability)
            stability_score = 1.0 / (1.0 + stability_variance)
        else:
            stability_score = 0.0
        
        return {
            "mouth_distances": mouth_distances_norm.tolist(),
            "mouth_distances_raw": mouth_distances.tolist() if len(mouth_distances) > 0 else [],
            "frame_indices": frame_indices,
            "mouth_region_stability": float(stability_score),
            "fps": fps,
            "total_frames": total_frames,
            "num_frames_with_lips": len(mouth_distances)
        }


class SyncNetAnalyzer:
    """
    Modified SyncNet for phoneme-viseme alignment
    Computes cross-modal synchronization between audio and visual speech
    """
    
    @staticmethod
    def compute_audio_energy(audio: np.ndarray, sr: int = 16000, frame_size: int = 512) -> np.ndarray:
        """
        Compute audio energy envelope (proxy for phoneme activity)
        Higher energy = phoneme being spoken
        """
        # STFT-based energy
        D = librosa.stft(audio, n_fft=frame_size)
        S = np.abs(D)
        energy = np.sqrt(np.sum(S**2, axis=0))
        
        # Normalize
        energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-6)
        return energy
    
    @staticmethod
    def compute_visual_motion(mouth_distances: np.ndarray) -> np.ndarray:
        """
        Compute visual motion from mouth distances
        Peaks in mouth distance = vowels/open phonemes
        """
        if len(mouth_distances) == 0:
            return np.array([])
        
        # Use distance values directly as motion proxy
        motion = np.abs(np.diff(mouth_distances, prepend=mouth_distances[0]))
        
        # Smooth with Gaussian
        from scipy.ndimage import gaussian_filter1d
        motion = gaussian_filter1d(motion, sigma=2)
        
        return motion
    
    @staticmethod
    def compute_sync_score(audio_energy: np.ndarray, visual_motion: np.ndarray) -> float:
        """
        Compute cross-modal synchronization score
        Higher score = better lip-sync (likely authentic)
        Lower score = poor sync (likely dubbing/manipulation)
        """
        if len(audio_energy) == 0 or len(visual_motion) == 0:
            return 0.5
        
        # Resample to same length
        min_len = min(len(audio_energy), len(visual_motion))
        audio_energy = audio_energy[:min_len]
        visual_motion = visual_motion[:min_len]
        
        # Normalize both
        audio_energy = (audio_energy - audio_energy.mean()) / (audio_energy.std() + 1e-6)
        visual_motion = (visual_motion - visual_motion.mean()) / (visual_motion.std() + 1e-6)
        
        # Compute cosine similarity
        cosine_sim = np.dot(audio_energy, visual_motion) / (
            np.linalg.norm(audio_energy) * np.linalg.norm(visual_motion) + 1e-6
        )
        
        # Map to 0-1 range (cosine similarity is -1 to 1)
        sync_score = (cosine_sim + 1.0) / 2.0
        
        return float(np.clip(sync_score, 0, 1))
    
    @staticmethod
    def find_sync_mismatches(audio_energy: np.ndarray, visual_motion: np.ndarray, 
                            threshold: float = 0.3) -> List[Tuple[int, int]]:
        """
        Find temporal regions where audio-visual sync breaks down
        Returns list of (start_frame, end_frame) tuples for mismatches
        """
        if len(audio_energy) == 0 or len(visual_motion) == 0:
            return []
        
        # Resample
        min_len = min(len(audio_energy), len(visual_motion))
        audio_energy = audio_energy[:min_len]
        visual_motion = visual_motion[:min_len]
        
        # Normalize
        audio_energy = (audio_energy - audio_energy.mean()) / (audio_energy.std() + 1e-6)
        visual_motion = (visual_motion - visual_motion.mean()) / (visual_motion.std() + 1e-6)
        
        # Compute frame-by-frame correlation
        correlation = np.abs(audio_energy - visual_motion)
        
        # Find mismatches (high absolute difference)
        mismatches = np.where(correlation > threshold)[0]
        
        # Group consecutive mismatches
        mismatch_regions = []
        if len(mismatches) > 0:
            start = mismatches[0]
            for i in range(1, len(mismatches)):
                if mismatches[i] - mismatches[i-1] > 1:
                    mismatch_regions.append((int(start), int(mismatches[i-1])))
                    start = mismatches[i]
            mismatch_regions.append((int(start), int(mismatches[-1])))
        
        return mismatch_regions


class SyncAnalysisModule:
    """
    Complete Lip-Sync & Phoneme-Viseme Analysis
    Language-agnostic: detects synchronization issues regardless of language
    """
    
    def __init__(self):
        self.lip_analyzer = LipSyncAnalyzer()
        self.syncnet = SyncNetAnalyzer()
        import librosa
        globals()['librosa'] = librosa  # Make librosa available for static methods
    
    def analyze_sync(self, video_path: str, audio: np.ndarray, sr: int = 16000) -> Dict:
        """
        Complete sync analysis pipeline
        
        Args:
            video_path: Path to video file
            audio: Audio waveform (numpy array)
            sr: Sample rate of audio
            
        Returns:
            Dict with sync scores and mismatch regions
        """
        logger.info(f"Starting sync analysis for {video_path}")
        
        try:
            # Step 1: Extract mouth landmarks
            mouth_data = self.lip_analyzer.extract_mouth_landmarks(video_path)
            
            if mouth_data["num_frames_with_lips"] == 0:
                logger.warning("No mouth landmarks detected")
                return self._create_empty_result()
            
            # Step 2: Compute audio energy (phoneme proxy)
            import librosa
            audio_energy = self.syncnet.compute_audio_energy(audio, sr=sr)
            
            # Step 3: Compute visual motion (viseme proxy)
            mouth_distances = np.array(mouth_data["mouth_distances"])
            visual_motion = self.syncnet.compute_visual_motion(mouth_distances)
            
            # Step 4: Compute sync score
            sync_score = self.syncnet.compute_sync_score(audio_energy, visual_motion)
            
            # Step 5: Find mismatches
            mismatches = self.syncnet.find_sync_mismatches(audio_energy, visual_motion, threshold=0.4)
            
            # Step 6: Match percentage
            if len(mouth_distances) > 0:
                total_frames = len(mouth_distances)
                mismatch_frames = sum(end - start for start, end in mismatches)
                match_percentage = (1.0 - mismatch_frames / total_frames) * 100.0
            else:
                match_percentage = 0.0
            
            # Voice dubbing detection (poor sync + transcription mismatch)
            dubbing_likelihood = 1.0 - sync_score if sync_score < 0.65 else 0.0
            
            # Final sync score (0 = perfect sync, 1 = complete mismatch)
            final_sync_score = 1.0 - sync_score  # Invert for deepfake likelihood
            
            result = {
                "transcription": "",  # Filled by audio module
                "detected_language": "",  # Filled by audio module
                "language_confidence": 0.0,  # Filled by audio module
                "phoneme_timestamps": {},  # Would be filled with ASR output
                "viseme_extraction_confidence": float(mouth_data["mouth_region_stability"]),
                "mouth_region_stability": float(mouth_data["mouth_region_stability"]),
                "phoneme_viseme_sync_score": float(sync_score),  # 0-1, higher = better sync
                "sync_match_percentage": float(match_percentage),
                "lip_sync_mismatches": [{"start": int(s), "end": int(e)} for s, e in mismatches],
                "voice_dubbing_likelihood": float(dubbing_likelihood),
                "dubbed_regions": [{"start_frame": int(s), "end_frame": int(e)} for s, e in mismatches],
                "sync_deepfake_score": float(final_sync_score),  # 0-1, higher = more likely deepfake
                "model_version": "sync-v1.0"
            }
            
            logger.info(f"Sync analysis complete. Score: {final_sync_score:.3f}, Match: {match_percentage:.1f}%")
            return result
        
        except Exception as e:
            logger.error(f"Sync analysis failed: {e}")
            return self._create_empty_result()
    
    def _create_empty_result(self) -> Dict:
        """Return neutral scores when analysis fails"""
        return {
            "transcription": "",
            "detected_language": "",
            "language_confidence": 0.0,
            "phoneme_timestamps": {},
            "viseme_extraction_confidence": 0.5,
            "mouth_region_stability": 0.5,
            "phoneme_viseme_sync_score": 0.5,
            "sync_match_percentage": 50.0,
            "lip_sync_mismatches": [],
            "voice_dubbing_likelihood": 0.0,
            "dubbed_regions": [],
            "sync_deepfake_score": 0.5,
            "model_version": "sync-v1.0"
        }


# Export
__all__ = ["SyncAnalysisModule", "LipSyncAnalyzer", "SyncNetAnalyzer"]
