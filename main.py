"""
Main Pipeline: Kabaddi Injury Prediction System
Orchestrates all 7 stages and produces final annotated output.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from loguru import logger
from tqdm import tqdm
import sys
from datetime import datetime
import json

# Import all modules
from config.config import (
    STAGE_DIRS, OUTPUT_DIR, VIDEO_CONFIG, LOG_CONFIG, PERFORMANCE_CONFIG
)

from utils.video_utils import VideoReader, VideoWriter
from utils.pipeline_status import PipelineStatus, MetricsLogger
from utils.visualization import VideoAnnotator

from models.player_detector import PlayerDetector
from models.raider_identifier import RaiderIdentifier, annotate_raider
from models.pose_estimator import PoseEstimator, annotate_pose
from models.fall_detector import FallDetector, annotate_fall_detection
from models.motion_analyzer import MotionAnalyzer, annotate_motion_analysis
from models.impact_detector import ImpactDetector, annotate_impact_detection
from models.risk_fusion import RiskFusionEngine, annotate_risk_score


# Configure logging with timestamped files
def configure_logging(output_dir: Path):
    """Configure loguru to write to timestamped log files and console."""
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"pipeline_{timestamp}.log"
    
    # Remove default handler
    logger.remove()
    
    # Add console handler (INFO level only - no debug spam)
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | {message}",
        level="INFO"
    )
    
    # Add file handler with ALL debug messages
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation=None,  # No rotation - keep entire log in one file per run
        retention=None  # Keep all logs forever
    )
    
    # Also create a summary index file
    index_file = logs_dir / "logs_index.json"
    index_data = {}
    if index_file.exists():
        try:
            with open(index_file, 'r') as f:
                index_data = json.load(f)
        except:
            pass
    
    index_data[timestamp] = {
        "log_file": str(log_file),
        "start_time": datetime.now().isoformat(),
        "status": "running"
    }
    
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    logger.info(f"Logging configured - File: {log_file}")
    return log_file


class KabaddiInjuryPipeline:
    """Main pipeline orchestrator for injury prediction."""
    
    def __init__(self, video_path: str, output_dir: Optional[str] = None):
        """
        Initialize pipeline with input video.
        
        Args:
            video_path: Path to input Kabaddi match video
            output_dir: Directory for outputs (default: config.OUTPUT_DIR)
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video not found: {video_path}. Please check the file path and try again.")
        
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging with timestamped files
        configure_logging(self.output_dir)
        
        logger.info(f"Pipeline initialized - Input: {self.video_path}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Initialize status tracker
        self.status = PipelineStatus(self.output_dir)
        self.metrics = MetricsLogger(self.output_dir)
        
        # Initialize all stage modules
        logger.info("Initializing pipeline modules...")
        self.player_detector = None
        self.raider_identifier = None
        self.pose_estimator = None
        self.fall_detector = None
        self.motion_analyzer = None
        self.impact_detector = None
        self.risk_fusion = None
        
        # Video properties (set during run)
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        
        logger.info(f"Pipeline initialized for video: {video_path}")
    
    def _initialize_modules(self, frame_width: int, frame_height: int):
        """Initialize all processing modules."""
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self.player_detector = PlayerDetector()
        self.raider_identifier = RaiderIdentifier(frame_width, frame_height)
        self.pose_estimator = PoseEstimator()
        self.fall_detector = FallDetector(frame_height)
        self.motion_analyzer = MotionAnalyzer()
        self.impact_detector = ImpactDetector()
        self.risk_fusion = RiskFusionEngine()
        
        logger.info("All modules initialized")
    
    def run(self, max_frames: Optional[int] = None, save_intermediate: bool = True):
        """
        Run the complete 7-stage pipeline.
        
        Args:
            max_frames: Maximum frames to process (None = all)
            save_intermediate: Whether to save intermediate stage outputs
        """
        logger.info("="*60)
        logger.info("STARTING KABADDI INJURY PREDICTION PIPELINE")
        logger.info("="*60)
        
        # Open video
        reader = VideoReader(str(self.video_path))
        self.fps = reader.fps
        
        # Initialize modules
        self._initialize_modules(reader.width, reader.height)
        
        # Prepare output paths
        final_output_path = self.output_dir / "final_output.mp4"
        
        # Create output writer
        writer = VideoWriter(
            str(final_output_path),
            self.fps,
            reader.width,
            reader.height,
            codec=VIDEO_CONFIG["codec"]
        )
        
        # Process frames
        total_frames = min(reader.total_frames, max_frames) if max_frames else reader.total_frames
        
        try:
            with tqdm(total=total_frames, desc="Processing Pipeline") as pbar:
                for frame_num, frame in reader.read_frames(max_frames):
                    # Process all stages
                    annotated_frame = self._process_frame(frame, frame_num, save_intermediate)
                    
                    # Write output
                    writer.write_frame(annotated_frame)
                    
                    # Periodic GPU memory cleanup (every 100 frames)
                    if frame_num % 100 == 0:
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except:
                            pass
                    
                    pbar.update(1)
            
            # Complete pipeline
            self._finalize_pipeline()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
        finally:
            reader.release()
            writer.release()
            # Cleanup GPU memory if using CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("GPU memory cleared")
            except Exception as e:
                logger.debug(f"Could not clear GPU memory: {e}")
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Final output: {final_output_path}")
        logger.info("="*60)
        
        # Print status report
        self.status.print_report()
        
        return {
            "output_video": str(final_output_path),
            "status": self.status.get_all_statuses(),
            "metrics": self.metrics.get_summary_stats()
        }
    
    def _process_frame(self, frame: np.ndarray, frame_num: int, save_intermediate: bool) -> np.ndarray:
        """Process single frame through all 7 stages."""
        import time
        start_time = time.time()
        
        try:
            # Stage 1: Player Detection & Tracking
            if frame_num == 0:
                self.status.start_stage(1)
            
            logger.debug(f"[MAIN PIPELINE] ============ FRAME {frame_num} START ============")
            logger.debug(f"[MAIN PIPELINE - STAGE 1] Starting player detection...")
            annotated_frame, players = self.player_detector.process_frame(frame, frame_num)
            logger.debug(f"[MAIN PIPELINE - STAGE 1] Detected {len(players)} players")
            
            if frame_num == 0:
                self.status.complete_stage(1, f"Detected {len(players)} players")
            
            # Save Stage 1 output
            if save_intermediate and frame_num % 30 == 0:
                stage1_path = STAGE_DIRS[1] / f"frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(stage1_path), annotated_frame)
            
            # Stage 2: Raider Identification
            if frame_num == 0:
                self.status.start_stage(2)
            
            logger.debug(f"[MAIN PIPELINE - STAGE 2] Starting raider identification...")
            raider_info = self.raider_identifier.get_raider_info(players)
            raider_id = raider_info["track_id"] if raider_info else None
            logger.debug(f"[MAIN PIPELINE - STAGE 2] Raider identified: {raider_id}")
            
            frame_with_raider = annotate_raider(frame.copy(), players, raider_info)
            
            if frame_num == 0:
                status_msg = f"Raider ID: {raider_id}" if raider_id else "Raider not detected"
                self.status.complete_stage(2, status_msg)
            
            # Save Stage 2 output
            if save_intermediate and frame_num % 30 == 0:
                stage2_path = STAGE_DIRS[2] / f"frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(stage2_path), frame_with_raider)
        
            # If no raider, skip pose-dependent stages
            if raider_id is None:
                logger.debug(f"[MAIN PIPELINE] No raider detected, skipping pose-dependent stages")
                # Log metrics
                processing_time = time.time() - start_time
                self.metrics.log_frame_metrics(
                    frame_num=frame_num,
                    detection_count=len(players),
                    raider_detected=False,
                    processing_time=processing_time
                )
                return frame_with_raider
            
            # Get raider bbox
            raider_bbox = raider_info["bbox"]
            
            # Stage 3: Pose Estimation (Raider Only)
            if frame_num == 0:
                self.status.start_stage(3)
            
            logger.debug(f"[MAIN PIPELINE - STAGE 3] Starting pose estimation...")
            joints = self.pose_estimator.extract_pose(frame, raider_bbox)
            angles = self.pose_estimator.calculate_joint_angles(joints) if joints else None
            logger.debug(f"[MAIN PIPELINE - STAGE 3] Pose extraction: {'SUCCESS' if joints else 'FAILED'}")
            
            frame_with_pose = annotate_pose(frame_with_raider.copy(), joints, angles)
            
            if frame_num == 0:
                status_msg = "Pose detected" if joints else "Pose detection failed"
                self.status.complete_stage(3, status_msg)
            
            # Save Stage 3 output
            if save_intermediate and frame_num % 30 == 0:
                stage3_path = STAGE_DIRS[3] / f"frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(stage3_path), frame_with_pose)
            
            # Stage 4: Fall Detection
            if frame_num == 0:
                self.status.start_stage(4)
            
            logger.debug(f"[MAIN PIPELINE - STAGE 4] Starting fall detection...")
            com = self.pose_estimator.get_center_of_mass(joints) if joints else None
            is_falling, fall_info = self.fall_detector.detect_fall(joints, frame_num, com)
            logger.debug(f"[MAIN PIPELINE - STAGE 4] Fall detected: {is_falling}")
            
            frame_with_fall = annotate_fall_detection(frame_with_pose.copy(), fall_info, frame_num)
            
            if frame_num == 0:
                self.status.complete_stage(4, "Fall detection active")
            
            # Save Stage 4 output
            if save_intermediate and is_falling:
                stage4_path = STAGE_DIRS[4] / f"fall_frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(stage4_path), frame_with_fall)
            
            # Stage 5: Motion Analysis
            if frame_num == 0:
                self.status.start_stage(5)
            
            logger.debug(f"[MAIN PIPELINE - STAGE 5] Starting motion analysis...")
            motion_metrics = self.motion_analyzer.calculate_motion_metrics(joints)
            motion_abnormality = self.motion_analyzer.calculate_abnormality_score(motion_metrics)
            logger.debug(f"[MAIN PIPELINE - STAGE 5] Motion abnormality: {motion_abnormality:.3f}")
            
            # Ensure motion_abnormality is a scalar float
            if motion_abnormality is None:
                motion_abnormality = 0.0
            elif isinstance(motion_abnormality, (np.ndarray, list, tuple)):
                motion_abnormality = float(np.mean(motion_abnormality))
            else:
                motion_abnormality = float(motion_abnormality)
            
            frame_with_motion = annotate_motion_analysis(frame_with_fall.copy(), motion_metrics, motion_abnormality)
            
            if frame_num == 0:
                self.status.complete_stage(5, f"Motion abnormality: {motion_abnormality:.1f}%")
            
            # Save Stage 5 output
            if save_intermediate and frame_num % 30 == 0:
                stage5_path = STAGE_DIRS[5] / f"frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(stage5_path), frame_with_motion)
            
            # Stage 6: Impact Detection
            if frame_num == 0:
                self.status.start_stage(6)
            
            logger.debug(f"[MAIN PIPELINE - STAGE 6] Starting impact detection...")
            impact_detected, impact_info = self.impact_detector.detect_impacts(players, raider_id, frame_num)
            logger.debug(f"[MAIN PIPELINE - STAGE 6] Impact detected: {impact_detected}")
            
            frame_with_impact = annotate_impact_detection(
                frame_with_motion.copy(), players, raider_id, impact_info
            )
            
            if frame_num == 0:
                self.status.complete_stage(6, "Impact detection active")
            
            # Save Stage 6 output
            if save_intermediate and impact_detected:
                stage6_path = STAGE_DIRS[6] / f"impact_frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(stage6_path), frame_with_impact)
            
            # Stage 7: Risk Fusion
            if frame_num == 0:
                self.status.start_stage(7)
            
            logger.debug(f"[MAIN PIPELINE - STAGE 7] Starting risk fusion...")
            fall_severity = fall_info.get("severity", 0.0)
            impact_severity = impact_info.get("severity", 0.0)
            
            # Ensure all severity values are scalar floats
            if fall_severity is None:
                fall_severity = 0.0
            else:
                fall_severity = float(fall_severity) if not isinstance(fall_severity, (list, tuple, np.ndarray)) else float(np.mean(fall_severity))
            
            if impact_severity is None:
                impact_severity = 0.0
            else:
                impact_severity = float(impact_severity) if not isinstance(impact_severity, (list, tuple, np.ndarray)) else float(np.mean(impact_severity))
            
            risk_data = self.risk_fusion.calculate_risk_score(
                fall_severity=fall_severity,
                impact_severity=impact_severity,
                motion_abnormality=motion_abnormality,
                player_id=raider_id,
                frame_num=frame_num
            )
            
            # Ensure risk_score is scalar float for all operations
            risk_score_val = risk_data["risk_score"]
            if isinstance(risk_score_val, (list, tuple, np.ndarray)):
                risk_score_val = float(np.mean(risk_score_val))
            else:
                risk_score_val = float(risk_score_val) if risk_score_val is not None else 0.0
            
            risk_data["risk_score"] = risk_score_val  # Update dict with scalar value
            logger.debug(f"[MAIN PIPELINE - STAGE 7] Risk score: {risk_score_val:.1f}%, Level: {risk_data['risk_level']}")
            
            # Auto-update injury history for high-risk events
            from config.config import RISK_FUSION_CONFIG
            if risk_score_val >= RISK_FUSION_CONFIG["critical_threshold"]:
                injury_event = {
                    "severity": risk_score_val,
                    "frame": frame_num,
                    "fall_severity": fall_severity,
                    "impact_severity": impact_severity,
                    "motion_abnormality": motion_abnormality
                }
                self.risk_fusion.update_injury_history(raider_id, injury_event)
                logger.warning(f"[MAIN PIPELINE - CRITICAL] High-risk event recorded for player {raider_id} at frame {frame_num} (risk: {risk_score_val:.1f})")
            
            final_frame = annotate_risk_score(frame_with_impact.copy(), risk_data, show_breakdown=True, risk_engine=self.risk_fusion)
            
            if frame_num == 0:
                self.status.complete_stage(7, f"Risk: {risk_data['risk_level']}")
            
            # Save Stage 7 output
            if save_intermediate and frame_num % 30 == 0:
                stage7_path = STAGE_DIRS[7] / f"frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(stage7_path), final_frame)
            
            # Log metrics
            processing_time = time.time() - start_time
            logger.debug(f"[MAIN PIPELINE - METRICS] Frame {frame_num} - Detections: {len(players)}, Risk Score: {risk_score_val:.1f}, Time: {processing_time:.3f}s")
            
            self.metrics.log_frame_metrics(
                frame_num=frame_num,
                detection_count=len(players),
                raider_detected=True,
                fall_detected=is_falling,
                risk_score=risk_score_val,
                processing_time=processing_time
            )
            
            logger.debug(f"[MAIN PIPELINE] ============ FRAME {frame_num} END (Time: {processing_time:.3f}s) ============")
            
            return final_frame
        
        except Exception as e:
            import traceback
            logger.error(f"[MAIN PIPELINE - ERROR] Frame {frame_num}: {str(e)}")
            logger.error(f"[MAIN PIPELINE - ERROR TRACEBACK]\n{traceback.format_exc()}")
            # Return original frame on error
            return frame
    
    def _finalize_pipeline(self):
        """Finalize pipeline and save all outputs."""
        logger.info("[MAIN PIPELINE] Starting pipeline finalization...")
        
        # Mark all stages as completed (finalize any that are still processing or yet to start)
        for stage_num in range(1, 8):
            stage = self.status.stages[stage_num]
            current_status = stage["status"]
            
            if current_status == "Processing":
                # If still processing, mark as completed
                self.status.complete_stage(stage_num, f"{stage['name']} completed")
            elif current_status == "Yet to start":
                # If never started (e.g., raider not detected), still mark as completed
                # since the pipeline ran but that stage was not needed
                self.status.complete_stage(stage_num, f"{stage['name']} skipped - no raider detected")
                logger.debug(f"Stage {stage_num} marked as completed (skipped - raider not detected)")
        
        # Save metrics
        self.metrics.save_metrics()
        logger.debug("[MAIN PIPELINE] Metrics saved")
        
        # Export status cards
        self.status.export_ui_cards(self.output_dir / "status_cards")
        logger.debug("[MAIN PIPELINE] Status cards exported")
        
        # Save summaries
        fall_summary = self.fall_detector.get_fall_summary()
        impact_summary = self.impact_detector.get_impact_summary()
        risk_summary = self.risk_fusion.get_risk_summary()
        motion_summary = self.motion_analyzer.get_summary_statistics()
        logger.debug("[MAIN PIPELINE] Summaries generated")
        
        import json
        
        summary = {
            "falls": fall_summary,
            "impacts": impact_summary,
            "risk": risk_summary,
            "motion": motion_summary,
            "metrics": self.metrics.get_summary_stats()
        }
        
        summary_path = self.output_dir / "pipeline_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Pipeline summary saved to {summary_path}")


def main():
    """Main entry point."""
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format=LOG_CONFIG["format"],
        level=LOG_CONFIG["level"]
    )
    logger.add(
        LOG_CONFIG["log_file"],
        format=LOG_CONFIG["format"],
        level=LOG_CONFIG["level"]
    )
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_video> [output_dir] [max_frames]")
        print("\nExample:")
        print("  python main.py kabaddi_match.mp4")
        print("  python main.py kabaddi_match.mp4 outputs/test1")
        print("  python main.py kabaddi_match.mp4 outputs/test1 300")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else PERFORMANCE_CONFIG["max_frames"]
    
    try:
        # Run pipeline
        pipeline = KabaddiInjuryPipeline(input_video, output_dir)
        results = pipeline.run(
            max_frames=max_frames,
            save_intermediate=PERFORMANCE_CONFIG["save_intermediate"]
        )
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*60)
        print(f"Output video: {results['output_video']}")
        print(f"Average risk score: {results['metrics']['avg_risk_score']:.1f}")
        print(f"Total fall events: {results['metrics']['total_fall_events']}")
        print(f"Processing time: {results['metrics']['avg_processing_time']:.3f}s per frame")
        print("="*60)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
