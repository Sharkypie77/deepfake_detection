"""
Module 5: Multimodal Fusion & Risk Score Calculator
Combines vision, audio, and sync streams with explainability
"""

import numpy as np
import torch
import logging
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureFusion:
    """Fuses multimodal features into risk score"""
    
    def __init__(self, vision_weight: float = 0.35, audio_weight: float = 0.35, 
                 sync_weight: float = 0.30):
        self.vision_weight = vision_weight
        self.audio_weight = audio_weight
        self.sync_weight = sync_weight
        
        # Verify weights sum to 1.0
        total = vision_weight + audio_weight + sync_weight
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1.0: {total}. Normalizing...")
            self.vision_weight = vision_weight / total
            self.audio_weight = audio_weight / total
            self.sync_weight = sync_weight / total
    
    def compute_risk_score(self, vision_score: float, audio_score: float, 
                          sync_score: float) -> Dict:
        """
        Compute final risk score with attention weighting
        
        Args:
            vision_score: 0-1 (0=authentic, 1=deepfake)
            audio_score: 0-1 (0=authentic, 1=deepfake)
            sync_score: 0-1 (0=authentic, 1=deepfake/dubbed)
            
        Returns:
            Dict with risk score (0-100%), verdict, confidence
        """
        # Validate inputs
        vision_score = np.clip(float(vision_score), 0, 1)
        audio_score = np.clip(float(audio_score), 0, 1)
        sync_score = np.clip(float(sync_score), 0, 1)
        
        # Cross-modal consistency check
        inconsistency_bonus = self._compute_inconsistency_bonus(
            vision_score, audio_score, sync_score
        )
        
        # Weighted fusion
        raw_risk_score = (
            self.vision_weight * vision_score +
            self.audio_weight * audio_score +
            self.sync_weight * sync_score
        )
        
        # Apply inconsistency bonus
        # If multiple streams disagree strongly, raise suspicion
        final_risk_score = np.clip(raw_risk_score + inconsistency_bonus * 0.15, 0, 1)
        
        # Scale to 0-100%
        risk_percentage = final_risk_score * 100.0
        
        # Determine verdict and confidence
        verdict, confidence = self._determine_verdict(final_risk_score, inconsistency_bonus)
        
        return {
            "risk_score": float(risk_percentage),
            "risk_score_normalized": float(final_risk_score),
            "verdict": verdict,
            "confidence_level": confidence,
            "vision_score": float(vision_score),
            "audio_score": float(audio_score),
            "sync_score": float(sync_score),
            "vision_weight": float(self.vision_weight),
            "audio_weight": float(self.audio_weight),
            "sync_weight": float(self.sync_weight),
            "cross_modal_inconsistency_bonus": float(inconsistency_bonus)
        }
    
    @staticmethod
    def _compute_inconsistency_bonus(vision_score: float, audio_score: float, 
                                     sync_score: float) -> float:
        """
        If multiple streams strongly disagree, increase deepfake likelihood
        
        Example:
        - Vision: 0.1 (authentic)
        - Audio: 0.9 (deepfake)
        - Sync: 0.9 (dubbed)
        → Inconsistency between vision and audio/sync suggests manipulation
        """
        # Compute pairwise differences
        vision_audio_diff = abs(vision_score - audio_score)
        audio_sync_diff = abs(audio_score - sync_score)
        vision_sync_diff = abs(vision_score - sync_score)
        
        # If two streams are strongly one-sided and third disagrees, raise suspicion
        max_diff = max(vision_audio_diff, audio_sync_diff, vision_sync_diff)
        
        # Bonus proportional to inconsistency
        # Max inconsistency = 1.0, gives bonus of 0.2
        inconsistency_bonus = min(0.2, max_diff * 0.2)
        
        return float(inconsistency_bonus)
    
    @staticmethod
    def _determine_verdict(risk_score: float, inconsistency_bonus: float) -> tuple:
        """
        Determine verdict and confidence level
        
        Risk Scale:
        - 0-30%: AUTHENTIC (Green)
        - 31-65%: INCONCLUSIVE (Amber) 
        - 66-100%: DEEPFAKE (Red)
        """
        if risk_score < 0.30:
            verdict = "AUTHENTIC"
            # Higher confidence if near boundary
            confidence = "HIGH" if risk_score < 0.15 else "MEDIUM"
        elif risk_score < 0.65:
            verdict = "INCONCLUSIVE"
            confidence = "LOW"  # Always low confidence for ambiguous cases
        else:
            verdict = "DEEPFAKE"
            # Higher confidence if inconsistency is low and score is high
            confidence = "HIGH" if risk_score > 0.80 and inconsistency_bonus < 0.1 else "MEDIUM"
        
        return verdict, confidence


class ExplainabilityModule:
    """Generate explainable outputs and key findings"""
    
    @staticmethod
    def generate_summary(vision_result: Dict, audio_result: Dict, sync_result: Dict, 
                        risk_score_info: Dict) -> str:
        """Generate human-readable summary of findings"""
        
        verdict = risk_score_info["verdict"]
        risk = risk_score_info["risk_score"]
        
        # Start with verdict
        summary = f"**VERDICT: {verdict}** (Risk Score: {risk:.1f}%)\n\n"
        
        # Key findings
        summary += "**Key Findings:**\n"
        
        # Vision findings
        if vision_result.get("gan_artifact_score", 0) > 0.6:
            summary += f"• Visual: Possible GAN artifacts detected ({vision_result['gan_artifact_score']*100:.0f}% likelihood)\n"
        if vision_result.get("boundary_warping_score", 0) > 0.6:
            summary += f"• Visual: Facial boundary warping detected\n"
        if vision_result.get("dct_compression_artifacts", 0) > 0.5:
            summary += f"• Visual: Compression anomalies present\n"
        
        # Audio findings
        if audio_result.get("aasist_spoofing_score", 0) > 0.6:
            summary += f"• Audio: Voice cloning indicators ({audio_result['aasist_spoofing_score']*100:.0f}% likelihood)\n"
        if audio_result.get("spectral_cutoff_presence", 0) > 0.6:
            summary += f"• Audio: Unnatural spectral cutoff detected\n"
        lang = audio_result.get("detected_languages", ["Unknown"])[0]
        summary += f"• Detected Language: {lang}\n"
        
        # Sync findings
        if sync_result.get("voice_dubbing_likelihood", 0) > 0.6:
            summary += f"• Sync: Voice dubbing suspected ({sync_result.get('sync_match_percentage', 0):.0f}% match)\n"
        if len(sync_result.get("lip_sync_mismatches", [])) > 0:
            summary += f"• Sync: {len(sync_result['lip_sync_mismatches'])} lip-sync mismatch regions detected\n"
        
        # Overall assessment
        summary += f"\n**Confidence Level:** {risk_score_info['confidence_level']}\n"
        
        if verdict == "DEEPFAKE":
            summary += f"\nThe video shows multiple indicators of manipulation across visual, audio, and synchronization analyses."
        elif verdict == "INCONCLUSIVE":
            summary += f"\nThe video presents mixed signals. Additional manual review recommended."
        else:  # AUTHENTIC
            summary += f"\nThe video passes most authenticity checks. Manipulation indicators are minimal."
        
        return summary


class TimelineHeatmapGenerator:
    """Generate frame-by-frame risk timeline for visualization"""
    
    @staticmethod
    def generate_frame_timeline(vision_result: Dict, audio_result: Dict, 
                               sync_result: Dict, total_frames: int) -> List[Dict]:
        """
        Generate frame-by-frame risk scores for timeline visualization
        
        Returns:
            List of {"frame": int, "risk": float} dicts
        """
        timeline = []
        
        # Get per-frame data if available
        fft_spectrum = vision_result.get("fft_spectrum_analysis", [])
        
        if len(fft_spectrum) > 0:
            # Use FFT spectrum as frame-level risk
            for i, spectrum in enumerate(fft_spectrum):
                spectrum = np.array(spectrum)
                frame_risk = float(spectrum.mean() / (spectrum.max() + 1e-6))
                timeline.append({
                    "frame": i,
                    "risk": frame_risk,
                    "source": "vision_fft"
                })
        else:
            # Fallback: uniform risk across frames
            overall_risk = vision_result.get("vision_deepfake_score", 0.5)
            for frame_idx in range(total_frames):
                timeline.append({
                    "frame": frame_idx,
                    "risk": overall_risk,
                    "source": "overall"
                })
        
        return timeline


class MultimodalFusionModule:
    """
    Complete Multimodal Fusion Pipeline
    Combines all detection streams into final verdict
    """
    
    def __init__(self, vision_weight: float = 0.35, audio_weight: float = 0.35, 
                 sync_weight: float = 0.30):
        self.fusion = FeatureFusion(vision_weight, audio_weight, sync_weight)
        self.explainability = ExplainabilityModule()
        self.timeline = TimelineHeatmapGenerator()
    
    def fuse_results(self, vision_result: Dict, audio_result: Dict, 
                    sync_result: Dict, total_frames: int = 300) -> Dict:
        """
        Fuse all module outputs into final risk assessment
        
        Args:
            vision_result: Output from VisionAIModule
            audio_result: Output from AudioAnalysisModule
            sync_result: Output from SyncAnalysisModule
            total_frames: Total frames in video (for timeline)
            
        Returns:
            Complete multimodal fusion result with risk score, verdict, and explainability
        """
        logger.info("Starting multimodal fusion...")
        
        # Extract individual scores
        vision_score = float(vision_result.get("vision_deepfake_score", 0.5))
        audio_score = float(audio_result.get("audio_deepfake_score", 0.5))
        sync_score = float(sync_result.get("sync_deepfake_score", 0.5))
        
        # Compute risk score
        risk_info = self.fusion.compute_risk_score(vision_score, audio_score, sync_score)
        
        # Generate summary
        summary = self.explainability.generate_summary(
            vision_result, audio_result, sync_result, risk_info
        )
        
        # Generate timeline
        frame_timeline = self.timeline.generate_frame_timeline(
            vision_result, audio_result, sync_result, total_frames
        )
        
        # Compile key findings
        key_findings = self._extract_key_findings(
            vision_result, audio_result, sync_result, risk_info
        )
        
        # Final result
        final_result = {
            "risk_score": float(risk_info["risk_score"]),
            "verdict": risk_info["verdict"],
            "confidence_level": risk_info["confidence_level"],
            
            # Component scores
            "vision_score": float(vision_score),
            "audio_score": float(audio_score),
            "sync_score": float(sync_score),
            
            # Weights used
            "vision_weight": float(self.fusion.vision_weight),
            "audio_weight": float(self.fusion.audio_weight),
            "sync_weight": float(self.fusion.sync_weight),
            "cross_modal_inconsistency_bonus": float(risk_info["cross_modal_inconsistency_bonus"]),
            
            # Explainability
            "summary": summary,
            "key_findings": key_findings,
            "frame_anomaly_timeline": frame_timeline,
            
            # Metadata
            "generated_at": datetime.utcnow().isoformat(),
            "model_version": "fusion-v1.0"
        }
        
        logger.info(f"Fusion complete. Verdict: {risk_info['verdict']}, Score: {risk_info['risk_score']:.1f}%")
        return final_result
    
    @staticmethod
    def _extract_key_findings(vision_result: Dict, audio_result: Dict, 
                             sync_result: Dict, risk_info: Dict) -> List[Dict]:
        """Extract top anomalies for explainability"""
        
        findings = []
        
        # Vision anomalies
        if vision_result.get("gan_artifact_score", 0) > 0.5:
            findings.append({
                "type": "vision",
                "category": "GAN Artifacts",
                "score": float(vision_result["gan_artifact_score"]),
                "severity": "high" if vision_result["gan_artifact_score"] > 0.75 else "medium"
            })
        
        if vision_result.get("boundary_warping_score", 0) > 0.5:
            findings.append({
                "type": "vision",
                "category": "Boundary Warping",
                "score": float(vision_result["boundary_warping_score"]),
                "severity": "high" if vision_result["boundary_warping_score"] > 0.7 else "medium"
            })
        
        # Audio anomalies
        if audio_result.get("aasist_spoofing_score", 0) > 0.5:
            findings.append({
                "type": "audio",
                "category": "Voice Cloning",
                "score": float(audio_result["aasist_spoofing_score"]),
                "severity": "high" if audio_result["aasist_spoofing_score"] > 0.75 else "medium"
            })
        
        if audio_result.get("spectral_cutoff_presence", 0) > 0.5:
            findings.append({
                "type": "audio",
                "category": "Spectral Anomaly",
                "score": float(audio_result["spectral_cutoff_presence"]),
                "severity": "medium"
            })
        
        # Sync anomalies
        if sync_result.get("voice_dubbing_likelihood", 0) > 0.5:
            findings.append({
                "type": "sync",
                "category": "Voice Dubbing",
                "score": float(sync_result["voice_dubbing_likelihood"]),
                "severity": "high" if sync_result["voice_dubbing_likelihood"] > 0.75 else "medium"
            })
        
        if len(sync_result.get("lip_sync_mismatches", [])) > 0:
            findings.append({
                "type": "sync",
                "category": "Lip-Sync Mismatch",
                "score": float(1.0 - sync_result.get("sync_match_percentage", 50.0) / 100.0),
                "severity": "high" if len(sync_result["lip_sync_mismatches"]) > 5 else "medium"
            })
        
        # Sort by severity and score
        severity_order = {"high": 0, "medium": 1, "low": 2}
        findings.sort(key=lambda x: (severity_order[x["severity"]], -x["score"]))
        
        return findings[:5]  # Top 5 findings


# Export
__all__ = ["MultimodalFusionModule", "FeatureFusion", "ExplainabilityModule", "TimelineHeatmapGenerator"]
