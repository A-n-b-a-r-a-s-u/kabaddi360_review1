"""
Stage 7: Injury Risk Fusion
Multi-factor fusion of fall, impact, motion, and injury history to compute final risk score.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from collections import deque
from pathlib import Path
from loguru import logger

from config.config import RISK_FUSION_CONFIG, INJURY_HISTORY_PATH, INJURY_HISTORY_SCHEMA


class RiskFusionEngine:
    """Fuse multiple risk factors into final injury risk score."""
    
    def __init__(self):
        """Initialize risk fusion engine."""
        self.weights = RISK_FUSION_CONFIG["weights"]
        self.risk_levels = RISK_FUSION_CONFIG["risk_levels"]
        
        # Temporal smoothing
        self.smoothing_enabled = RISK_FUSION_CONFIG["temporal_smoothing"]
        self.smoothing_window = RISK_FUSION_CONFIG["smoothing_window"]
        self.risk_score_history = deque(maxlen=self.smoothing_window)
        
        # Injury history database
        self.injury_history_df = self._load_injury_history()
        
        # Risk tracking
        self.risk_timeline: List[Dict] = []
        
        logger.info("Risk Fusion Engine initialized")
    
    def _load_injury_history(self) -> Optional[pd.DataFrame]:
        """Load injury history from CSV if available."""
        try:
            if Path(INJURY_HISTORY_PATH).exists():
                # Try multiple encodings
                for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
                    try:
                        df = pd.read_csv(
                            INJURY_HISTORY_PATH,
                            dtype=INJURY_HISTORY_SCHEMA["dtypes"],
                            encoding=encoding,
                            on_bad_lines='skip'
                        )
                        logger.info(f"Loaded injury history for {len(df)} players with encoding: {encoding}")
                        return df
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                
                # If all encodings fail, return empty dataframe
                logger.warning(f"Could not decode injury history file with any encoding")
                return pd.DataFrame(columns=INJURY_HISTORY_SCHEMA["columns"])
            else:
                logger.warning(f"Injury history file not found: {INJURY_HISTORY_PATH}")
                # Create empty dataframe with correct schema
                df = pd.DataFrame(columns=INJURY_HISTORY_SCHEMA["columns"])
                return df
        except Exception as e:
            logger.error(f"Failed to load injury history: {e}")
            return pd.DataFrame(columns=INJURY_HISTORY_SCHEMA["columns"])
    
    def get_injury_history_modifier(self, player_id: int) -> float:
        """
        Get injury history risk modifier for a player (0-100).
        Returns 0 if no history available.
        """
        if self.injury_history_df is None or len(self.injury_history_df) == 0:
            return 0.0
        
        # Check if player_id column exists
        if "player_id" not in self.injury_history_df.columns:
            return 0.0
        
        player_id_str = str(player_id)
        player_record = self.injury_history_df[
            self.injury_history_df["player_id"] == player_id_str
        ]
        
        if len(player_record) == 0:
            return 0.0
        
        # Use risk_modifier if available
        if "risk_modifier" in player_record.columns:
            modifier = player_record.iloc[0]["risk_modifier"]
            return float(modifier) if not pd.isna(modifier) else 0.0
        
        # Calculate from injury count and severity
        injury_count = player_record.iloc[0].get("injury_count", 0)
        severity_avg = player_record.iloc[0].get("severity_avg", 0)
        
        # Simple formula: base on count and severity
        modifier = min(injury_count * 10 + severity_avg * 0.5, 100.0)
        return modifier
    
    def calculate_risk_score(
        self,
        fall_severity: float = 0.0,
        impact_severity: float = 0.0,
        motion_abnormality: float = 0.0,
        player_id: Optional[int] = None,
        frame_num: int = 0
    ) -> Dict:
        """
        Calculate final injury risk score using weighted fusion.
        
        Returns: Dictionary with risk score, level, and breakdown
        """
        logger.debug(f"[STAGE 7 - RISK FUSION] Calculating risk score at frame {frame_num}")
        logger.debug(f"[STAGE 7 - RISK FUSION] Fall: {fall_severity:.3f}, Impact: {impact_severity:.3f}, Motion: {motion_abnormality:.3f}")
        
        # Get injury history modifier
        injury_history_score = 0.0
        if player_id is not None:
            injury_history_score = self.get_injury_history_modifier(player_id)
            logger.debug(f"[STAGE 7 - RISK FUSION] Player {player_id} history score: {injury_history_score:.3f}")
        
        # Weighted fusion
        alpha = self.weights["fall_severity"]
        beta = self.weights["impact_severity"]
        gamma = self.weights["motion_abnormality"]
        delta = self.weights["injury_history"]
        
        raw_score = (
            alpha * fall_severity +
            beta * impact_severity +
            gamma * motion_abnormality +
            delta * injury_history_score
        )
        
        # Normalize to 0-100
        risk_score = np.clip(raw_score, 0.0, 100.0)
        logger.debug(f"[STAGE 7 - RISK FUSION] Raw score: {raw_score:.3f}, Clipped: {risk_score:.3f}")
        
        # Apply temporal smoothing
        if self.smoothing_enabled:
            self.risk_score_history.append(risk_score)
            smoothed_score = np.mean(list(self.risk_score_history))
            logger.debug(f"[STAGE 7 - RISK FUSION] Smoothed score: {smoothed_score:.3f}")
        else:
            smoothed_score = risk_score
        
        # Determine risk level
        risk_level = self._get_risk_level(smoothed_score)
        logger.debug(f"[STAGE 7 - RISK FUSION] Risk level: {risk_level}")
        
        # Create breakdown for explainability
        breakdown = {
            "fall_contribution": alpha * fall_severity,
            "impact_contribution": beta * impact_severity,
            "motion_contribution": gamma * motion_abnormality,
            "history_contribution": delta * injury_history_score,
            "weights": {
                "fall": alpha,
                "impact": beta,
                "motion": gamma,
                "history": delta
            }
        }
        
        # Record in timeline
        self.risk_timeline.append({
            "frame": frame_num,
            "risk_score": smoothed_score,
            "risk_level": risk_level,
            "fall_severity": fall_severity,
            "impact_severity": impact_severity,
            "motion_abnormality": motion_abnormality,
            "injury_history_score": injury_history_score
        })
        
        return {
            "risk_score": smoothed_score,
            "raw_score": risk_score,
            "risk_level": risk_level,
            "breakdown": breakdown,
            "components": {
                "fall_severity": fall_severity,
                "impact_severity": impact_severity,
                "motion_abnormality": motion_abnormality,
                "injury_history": injury_history_score
            }
        }
    
    def _get_risk_level(self, score: float) -> str:
        """Determine risk level from score."""
        for level, (min_val, max_val) in self.risk_levels.items():
            if min_val <= score <= max_val:
                return level.upper()
        return "UNKNOWN"
    
    def update_injury_history(self, player_id: int, injury_event: Dict):
        """
        Update injury history database with new injury event.
        
        Args:
            player_id: Player track ID
            injury_event: Dictionary with injury details (severity, frame, etc.)
        """
        if self.injury_history_df is None:
            logger.warning("Cannot update injury history: database not initialized")
            return
        
        player_id_str = str(player_id)
        
        # Check if player exists
        existing = self.injury_history_df[
            self.injury_history_df["player_id"] == player_id_str
        ]
        
        import datetime
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        severity = injury_event.get("severity", 0.0)
        
        if len(existing) > 0:
            # Update existing record
            idx = existing.index[0]
            current_count = self.injury_history_df.at[idx, "injury_count"]
            current_avg = self.injury_history_df.at[idx, "severity_avg"]
            
            # Update count and average severity
            new_count = current_count + 1
            new_avg = (current_avg * current_count + severity) / new_count
            
            self.injury_history_df.at[idx, "injury_count"] = new_count
            self.injury_history_df.at[idx, "severity_avg"] = new_avg
            self.injury_history_df.at[idx, "last_injury_date"] = current_date
            self.injury_history_df.at[idx, "risk_modifier"] = min(new_count * 10 + new_avg * 0.5, 100.0)
            
            logger.info(f"Updated injury history for player {player_id}: count={new_count}, avg_severity={new_avg:.1f}")
        else:
            # Add new record
            new_record = {
                "player_id": player_id_str,
                "injury_count": 1,
                "last_injury_date": current_date,
                "severity_avg": severity,
                "risk_modifier": min(10 + severity * 0.5, 100.0)
            }
            self.injury_history_df = pd.concat([
                self.injury_history_df,
                pd.DataFrame([new_record])
            ], ignore_index=True)
            
            logger.info(f"Added new injury record for player {player_id}")
        
        # Save updated history
        self._save_injury_history()
    
    def _save_injury_history(self):
        """Save injury history database to CSV."""
        try:
            Path(INJURY_HISTORY_PATH).parent.mkdir(parents=True, exist_ok=True)
            self.injury_history_df.to_csv(INJURY_HISTORY_PATH, index=False)
            logger.info(f"Injury history saved to {INJURY_HISTORY_PATH}")
        except Exception as e:
            logger.error(f"Failed to save injury history: {e}")
    
    def should_alert(self, risk_score: float) -> Tuple[bool, str]:
        """
        Determine if alert should be triggered based on risk score.
        
        Returns: (should_alert, alert_level)
        """
        if risk_score >= RISK_FUSION_CONFIG["critical_threshold"]:
            return True, "CRITICAL"
        elif risk_score >= RISK_FUSION_CONFIG["alert_threshold"]:
            return True, "WARNING"
        return False, "NONE"
    
    def get_risk_summary(self) -> Dict:
        """Get summary statistics of risk over time."""
        if not self.risk_timeline:
            return {
                "avg_risk": 0.0,
                "max_risk": 0.0,
                "min_risk": 0.0,
                "total_frames": 0,
                "high_risk_frames": 0,
                "medium_risk_frames": 0,
                "low_risk_frames": 0
            }
        
        scores = [r["risk_score"] for r in self.risk_timeline]
        
        high_risk = sum(1 for r in self.risk_timeline if r["risk_level"] == "HIGH")
        medium_risk = sum(1 for r in self.risk_timeline if r["risk_level"] == "MEDIUM")
        low_risk = sum(1 for r in self.risk_timeline if r["risk_level"] == "LOW")
        
        return {
            "avg_risk": np.mean(scores),
            "max_risk": max(scores),
            "min_risk": min(scores),
            "total_frames": len(self.risk_timeline),
            "high_risk_frames": high_risk,
            "medium_risk_frames": medium_risk,
            "low_risk_frames": low_risk,
            "timeline": self.risk_timeline
        }
    
    def reset(self):
        """Reset risk fusion state (but keep injury history)."""
        self.risk_score_history.clear()
        self.risk_timeline.clear()
        logger.info("Risk fusion engine reset")


def annotate_risk_score(
    frame: np.ndarray,
    risk_data: Dict,
    show_breakdown: bool = True,
    risk_engine: Optional[RiskFusionEngine] = None
) -> np.ndarray:
    """Annotate frame with final risk score and breakdown."""
    import cv2
    from utils.visualization import VideoAnnotator
    
    annotator = VideoAnnotator(frame.shape[1], frame.shape[0])
    
    # Draw risk meter - ensure risk_score is scalar float
    risk_score = risk_data["risk_score"]
    if isinstance(risk_score, (list, tuple, np.ndarray)):
        risk_score = float(np.mean(risk_score))
    else:
        risk_score = float(risk_score) if risk_score is not None else 0.0
    
    frame = annotator.draw_risk_meter(frame, risk_score, position=(50, 50))
    
    # Show breakdown if requested
    if show_breakdown and "breakdown" in risk_data:
        breakdown = risk_data["breakdown"]
        y_offset = 150
        
        cv2.putText(frame, "Risk Breakdown:", (50, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        components = [
            (f"Fall: {breakdown['fall_contribution']:.1f}", (0, 0, 255)),
            (f"Impact: {breakdown['impact_contribution']:.1f}", (0, 165, 255)),
            (f"Motion: {breakdown['motion_contribution']:.1f}", (255, 165, 0)),
            (f"History: {breakdown['history_contribution']:.1f}", (128, 0, 128))
        ]
        
        for text, color in components:
            cv2.putText(frame, text, (50, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
    
    # Alert if necessary (use provided engine instead of creating new one)
    if risk_engine is not None:
        should_alert, alert_level = risk_engine.should_alert(risk_score)
        if should_alert:
            alert_text = f"{alert_level} RISK ALERT"
            cv2.putText(frame, alert_text, (frame.shape[1] // 2 - 150, frame.shape[0] - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    return frame


if __name__ == "__main__":
    logger.info("Risk Fusion Engine module loaded")