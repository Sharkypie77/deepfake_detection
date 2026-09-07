"""
Module 2: Acoustic & Voice Cloning Detection
AASIST (Audio Anti-Spoofing using Integrated Spectral-Temporal GNN)
Supports all languages via pretrained models
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa
import logging
from pathlib import Path
from typing import Dict, Tuple
import whisper
from config import AASIST_CHECKPOINT_PATH
from checkpoint_provenance import checkpoint_status
from modules.aasist import AASIST

logger = logging.getLogger(__name__)


class AudioExtractor:
    """Extract audio from video and prepare features"""
    
    @staticmethod
    def extract_audio_from_video(video_path: str, sr: int = 16000) -> np.ndarray:
        """
        Extract mono audio at 16kHz from video file
        16kHz is standard for speech processing and AASIST
        """
        # Try using torchaudio first (efficient)
        try:
            waveform, sample_rate = torchaudio.load(video_path)
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            # Resample to 16kHz
            if sample_rate != sr:
                resampler = torchaudio.transforms.Resample(sample_rate, sr)
                waveform = resampler(waveform)
            return waveform.squeeze().numpy()
        except Exception as e:
            logger.warning(f"torchaudio failed: {e}, trying librosa")
            # Fallback to librosa (slower but more compatible)
            audio, _ = librosa.load(video_path, sr=sr, mono=True)
            return audio
    
    @staticmethod
    def compute_mel_spectrogram(audio: np.ndarray, sr: int = 16000, 
                               n_mels: int = 128, n_fft: int = 512) -> np.ndarray:
        """
        Compute mel-spectrogram
        Captures acoustic features language-agnostically
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=160
        )
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    @staticmethod
    def compute_mfcc(audio: np.ndarray, sr: int = 16000, n_mfcc: int = 13) -> np.ndarray:
        """
        Compute MFCC (Mel-Frequency Cepstral Coefficients)
        Acoustic features robust across languages
        """
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return mfcc
    
    @staticmethod
    def compute_cqcc(audio: np.ndarray, sr: int = 16000, n_cqcc: int = 19) -> np.ndarray:
        """
        Compute CQCC (Constant Q Cepstral Coefficients)
        Better at capturing voice characteristics across languages
        """
        # Constant-Q transform
        cqt = librosa.feature.chroma_cqt(y=audio, sr=sr, n_chroma=n_cqcc)
        return cqt


class AAISSTModule:
    """
    Audio Anti-Spoofing using Integrated Spectral-Temporal GNN
    Detects synthetic audio (voice cloning, text-to-speech)
    Language-agnostic: works on raw acoustic features
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.sr = 16000
        logger.info(f"AASIST Module initialized on {device}")
        
        self.model = self.load_aasist_model()

    def load_aasist_model(self):
        """Load and validate a real AASIST checkpoint."""
        if not AASIST_CHECKPOINT_PATH:
            raise RuntimeError("AASIST_CHECKPOINT_PATH is required; no audio fallback is permitted")
        checkpoint_path = Path(AASIST_CHECKPOINT_PATH).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"AASIST checkpoint not found: {checkpoint_path}")
        model = AASIST().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.checkpoint_status = checkpoint_status("aasist", str(checkpoint_path), checkpoint)
        if not self.checkpoint_status["validated"]:
            logger.warning("AASIST checkpoint is unvalidated: %s", self.checkpoint_status["reason"])
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        state_dict = {key.removeprefix("model."): value for key, value in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Invalid AASIST checkpoint {checkpoint_path}; missing={missing}, unexpected={unexpected}"
            )
        model.eval()
        logger.info("AASIST checkpoint loaded from %s (provenance=%s)",
                    checkpoint_path, self.checkpoint_status["validated"])
        return model
    
    def detect_spoofing(self, audio: np.ndarray) -> Dict:
        """
        Detect synthetic/cloned audio
        
        Returns:
            Dict with spoofing probability and confidence
        """
        # Extract mel-spectrogram
        samples = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        aasist_samples = 64600
        samples = samples[:aasist_samples]
        if samples.numel() < aasist_samples:
            samples = F.pad(samples, (0, aasist_samples - samples.numel()))
        samples = samples.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(samples)
            probabilities = torch.softmax(output, dim=1)
            fake_prob = float(probabilities[0, 1].cpu())
            confidence = float(probabilities[0].max().cpu())
        
        return {
            "aasist_spoofing_score": fake_prob,
            "aasist_confidence": confidence,
            "method": "aasist"
        }
    
    @staticmethod
    def _fallback_spoofing_detection(audio: np.ndarray) -> Dict:
        """
        Fallback statistical approach for spoofing detection
        Analyzes spectral characteristics
        """
        # Compute features
        mel_spec = AudioExtractor.compute_mel_spectrogram(audio)
        mfcc = AudioExtractor.compute_mfcc(audio)
        
        # Spectral statistics
        mel_mean = np.mean(mel_spec)
        mel_std = np.std(mel_spec)
        mel_entropy = -np.sum((mel_spec / mel_spec.sum()) * np.log(mel_spec / mel_spec.sum() + 1e-10))
        
        # MFCC statistics
        mfcc_delta = np.diff(mfcc, axis=1)
        mfcc_delta_delta = np.diff(mfcc_delta, axis=1)
        
        mfcc_variance = np.var(mfcc_delta_delta)
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(S=mel_spec)
        spectral_centroid_variance = np.var(spectral_centroids)
        
        # Combination: synthetic audio often has lower variance and entropy
        spoofing_score = max(0.0, min(1.0, 
            (1.0 - mel_std / 100.0) * 0.3 +
            (1.0 - mel_entropy / 10.0) * 0.3 +
            (1.0 - mfcc_variance / 100.0) * 0.2 +
            (1.0 - spectral_centroid_variance / 10000.0) * 0.2
        ))
        
        return {
            "aasist_spoofing_score": float(spoofing_score),
            "aasist_confidence": 0.7,  # Lower confidence for fallback
            "method": "statistical"
        }


class WhisperTranscription:
    """
    OpenAI Whisper for multilingual speech-to-text
    Detects language automatically and provides phoneme alignment
    """
    
    def __init__(self, model_name: str = "base"):
        self.model = whisper.load_model(model_name)
        self.sr = 16000
        logger.info(f"Whisper model '{model_name}' loaded")
    
    def transcribe_and_detect_language(self, audio: np.ndarray) -> Dict:
        """
        Transcribe audio and detect language
        Works for 99+ languages including Hindi, Tamil, Telugu, English
        """
        # Resample audio to 16kHz if needed
        if len(audio.shape) == 1:
            audio_mono = audio
        else:
            audio_mono = audio.mean(axis=0)
        
        # Normalize audio
        audio_mono = audio_mono / (np.max(np.abs(audio_mono)) + 1e-6)
        
        try:
            # Whisper expects PyTorch tensor
            audio_tensor = torch.from_numpy(audio_mono).float()
            
            # Use mel spectrogram padding (Whisper's default)
            mel = whisper.log_mel_spectrogram(audio_tensor)
            
            # Detect language
            _, probs = self.model.detect_language(mel)
            detected_language = max(probs, key=probs.get)
            language_confidence = probs[detected_language]
            
            # Transcribe with detected language
            options = whisper.DecodingOptions(language=detected_language)
            result = self.model.decode(mel, options)
            
            return {
                "transcription": result.text,
                "detected_language": detected_language,
                "language_confidence": float(language_confidence),
                "language_probabilities": {k: float(v) for k, v in list(probs.items())[:5]}  # Top 5
            }
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return {
                "transcription": "",
                "detected_language": "unknown",
                "language_confidence": 0.0,
                "language_probabilities": {}
            }


class AudioAnalysisModule:
    """
    Complete Audio Analysis Pipeline
    Combines AASIST spoofing detection, feature extraction, and Whisper transcription
    Language-agnostic: works on acoustic properties and automatically detects language
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.aasist = AAISSTModule(device)
        self.whisper = WhisperTranscription(model_name="base")
        self.extractor = AudioExtractor()
        self.sr = 16000
    
    def analyze_video_audio(self, video_path: str, audio_duration: int = 10) -> Dict:
        """
        Complete audio analysis pipeline
        
        Args:
            video_path: Path to video file
            audio_duration: Duration of audio to analyze in seconds
            
        Returns:
            Dict with all audio scores and transcription
        """
        logger.info(f"Starting audio analysis for {video_path}")
        
        try:
            # Extract audio
            audio = self.extractor.extract_audio_from_video(video_path, sr=self.sr)
            
            # Limit duration
            max_samples = min(len(audio), self.sr * audio_duration)
            audio = audio[:max_samples]
            
            if len(audio) == 0:
                logger.warning("No audio extracted from video")
                return self._create_empty_result()
            
            # AASIST spoofing detection
            spoofing_result = self.aasist.detect_spoofing(audio)
            
            # Speech recognition and language detection
            transcription_result = self.whisper.transcribe_and_detect_language(audio)
            
            # Feature extraction
            mel_spec = self.extractor.compute_mel_spectrogram(audio, self.sr)
            mfcc = self.extractor.compute_mfcc(audio, self.sr)
            cqcc = self.extractor.compute_cqcc(audio, self.sr)
            
            # Spectral analysis
            spectral_centroids = librosa.feature.spectral_centroid(S=mel_spec)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(S=mel_spec)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Check for spectral cutoffs (voice cloning artifact)
            spectral_cutoff_score = self._detect_spectral_cutoff(mel_spec)
            
            # Phase analysis
            phase_discontinuities = self._detect_phase_discontinuities(audio)
            
            # Breath/pause detection (natural speech indicator)
            breath_pause_score = self._detect_breathing_pauses(audio)
            
            # Pitch consistency
            f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=80, fmax=400, sr=self.sr)
            pitch_consistency = self._compute_pitch_consistency(f0)
            
            # Formant stability
            formant_stability = self._compute_formant_stability(mfcc)
            
            # Final audio score (weighted combination)
            audio_score = (
                spoofing_result["aasist_spoofing_score"] * 0.35 +  # AASIST is most reliable
                spectral_cutoff_score * 0.20 +
                phase_discontinuities * 0.15 +
                (1.0 - breath_pause_score) * 0.15 +
                (1.0 - pitch_consistency) * 0.15
            )
            
            result = {
                "aasist_spoofing_score": spoofing_result["aasist_spoofing_score"],
                "aasist_confidence": spoofing_result["aasist_confidence"],
                "aasist_method": spoofing_result["method"],
                "mel_spectrogram_anomalies": {},  # Can be extended with anomaly regions
                "spectral_cutoff_presence": float(spectral_cutoff_score),
                "cqcc_consistency": float(formant_stability),
                "phase_discontinuities": float(phase_discontinuities),
                "breath_pause_detection": float(breath_pause_score),
                "voice_pitch_consistency": float(pitch_consistency),
                "formant_stability": float(formant_stability),
                "detected_languages": [transcription_result["detected_language"]],
                "language_confidence": float(transcription_result["language_confidence"]),
                "transcription": transcription_result["transcription"],
                "audio_deepfake_score": float(audio_score),
                "audio_duration_seconds": float(len(audio) / self.sr),
                "sample_rate": self.sr,
                "model_version": "audio-v1.0"
            }
            
            logger.info(f"Audio analysis complete. Score: {audio_score:.3f}, Language: {transcription_result['detected_language']}")
            return result
        
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return self._create_empty_result()
    
    @staticmethod
    def _detect_spectral_cutoff(mel_spec: np.ndarray) -> float:
        """
        Detect abrupt spectral cutoff
        Voice cloning often has unnatural frequency boundaries
        """
        # Analyze high-frequency content
        high_freq_energy = mel_spec[-10:, :].mean()  # Last 10 mel bins
        mid_freq_energy = mel_spec[40:80, :].mean()  # Middle bins
        
        # Natural voice has more energy in higher frequencies
        cutoff_score = max(0.0, 1.0 - (high_freq_energy / (mid_freq_energy + 1e-6)))
        return float(np.clip(cutoff_score, 0, 1))
    
    @staticmethod
    def _detect_phase_discontinuities(audio: np.ndarray) -> float:
        """
        Detect phase anomalies (voice cloning artifact)
        """
        # Short-time Fourier transform
        D = librosa.stft(audio)
        phase = np.angle(D)
        
        # Phase differences between consecutive frames
        phase_diff = np.diff(phase, axis=1)
        
        # Unwrap phase to detect discontinuities
        phase_diff_unwrapped = np.abs(phase_diff)
        
        # High discontinuities indicate artifacts
        discontinuity_score = np.clip(phase_diff_unwrapped.mean() / np.pi, 0, 1)
        return float(discontinuity_score)
    
    @staticmethod
    def _detect_breathing_pauses(audio: np.ndarray, sr: int = 16000) -> float:
        """
        Detect presence of natural breathing and pauses
        Synthetic audio often lacks realistic breath patterns
        """
        # Detect silence periods (< -60dB)
        S = librosa.feature.melspectrogram(y=audio, sr=sr)
        S_db = librosa.power_to_db(S)
        
        silence_mask = S_db < -60
        silence_ratio = silence_mask.sum() / silence_mask.size
        
        # Natural speech has ~30-40% silent/low-energy frames
        breath_score = 1.0 - abs(silence_ratio - 0.35)
        return float(np.clip(breath_score, 0, 1))
    
    @staticmethod
    def _compute_pitch_consistency(f0: np.ndarray) -> float:
        """
        Measure pitch consistency
        Voiced speech should have relatively smooth pitch contour
        """
        # Remove unvoiced frames (f0 = 0)
        voiced_f0 = f0[f0 > 0]
        
        if len(voiced_f0) < 10:
            return 0.5
        
        # Compute pitch variation (standard deviation normalized)
        f0_mean = voiced_f0.mean()
        f0_std = voiced_f0.std()
        f0_cv = f0_std / (f0_mean + 1e-6)  # Coefficient of variation
        
        # Natural speech has CV around 0.1-0.3
        consistency = 1.0 - min(1.0, f0_cv / 0.5)
        return float(consistency)
    
    @staticmethod
    def _compute_formant_stability(mfcc: np.ndarray) -> float:
        """
        Measure MFCC (formant) stability
        Synthetic audio has less smooth formant transitions
        """
        # Compute MFCC delta (rate of change)
        mfcc_delta = np.diff(mfcc, axis=1)
        mfcc_delta_delta = np.diff(mfcc_delta, axis=1)
        
        # Natural speech has smooth formant transitions
        smoothness = 1.0 / (1.0 + np.mean(np.abs(mfcc_delta_delta)))
        return float(np.clip(smoothness, 0, 1))
    
    def _create_empty_result(self) -> Dict:
        """Return neutral scores when analysis fails"""
        return {
            "aasist_spoofing_score": 0.5,
            "aasist_confidence": 0.0,
            "aasist_method": "error",
            "mel_spectrogram_anomalies": {},
            "spectral_cutoff_presence": 0.5,
            "cqcc_consistency": 0.5,
            "phase_discontinuities": 0.5,
            "breath_pause_detection": 0.5,
            "voice_pitch_consistency": 0.5,
            "formant_stability": 0.5,
            "detected_languages": [],
            "language_confidence": 0.0,
            "transcription": "",
            "audio_deepfake_score": 0.5,
            "audio_duration_seconds": 0.0,
            "sample_rate": 16000,
            "model_version": "audio-v1.0"
        }


# Export
__all__ = ["AudioAnalysisModule", "AAISSTModule", "WhisperTranscription", "AudioExtractor"]
