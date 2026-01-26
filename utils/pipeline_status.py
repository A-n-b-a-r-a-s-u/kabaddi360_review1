"""
Pipeline status tracking for UI-ready status reporting.
Manages stage progression and outputs status cards.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger


class PipelineStatus:
    """Track and report status of each pipeline stage."""
    
    STATUSES = ["Yet to start", "Processing", "Completed", "Failed"]
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.stages = {
            1: {
                "name": "Player Detection & Tracking",
                "status": "Yet to start",
                "details": "",
                "output_path": None,
                "start_time": None,
                "end_time": None,
            },
            2: {
                "name": "Raider Identification",
                "status": "Yet to start",
                "details": "",
                "output_path": None,
                "start_time": None,
                "end_time": None,
            },
            3: {
                "name": "Pose Estimation",
                "status": "Yet to start",
                "details": "",
                "output_path": None,
                "start_time": None,
                "end_time": None,
            },
            4: {
                "name": "Fall Detection",
                "status": "Yet to start",
                "details": "",
                "output_path": None,
                "start_time": None,
                "end_time": None,
            },
            5: {
                "name": "Motion Analysis",
                "status": "Yet to start",
                "details": "",
                "output_path": None,
                "start_time": None,
                "end_time": None,
            },
            6: {
                "name": "Impact Detection",
                "status": "Yet to start",
                "details": "",
                "output_path": None,
                "start_time": None,
                "end_time": None,
            },
            7: {
                "name": "Risk Fusion",
                "status": "Yet to start",
                "details": "",
                "output_path": None,
                "start_time": None,
                "end_time": None,
            },
        }
    
    def start_stage(self, stage_num: int):
        """Mark stage as processing."""
        if stage_num in self.stages:
            self.stages[stage_num]["status"] = "Processing"
            self.stages[stage_num]["start_time"] = datetime.now().isoformat()
            logger.info(f"Stage {stage_num} started: {self.stages[stage_num]['name']}")
            self._save_status()
    
    def complete_stage(self, stage_num: int, details: str = "", output_path: Optional[str] = None):
        """Mark stage as completed."""
        if stage_num in self.stages:
            self.stages[stage_num]["status"] = "Completed"
            self.stages[stage_num]["details"] = details
            self.stages[stage_num]["output_path"] = output_path
            self.stages[stage_num]["end_time"] = datetime.now().isoformat()
            
            # Calculate duration
            if self.stages[stage_num]["start_time"]:
                start = datetime.fromisoformat(self.stages[stage_num]["start_time"])
                end = datetime.fromisoformat(self.stages[stage_num]["end_time"])
                duration = (end - start).total_seconds()
                logger.info(f"Stage {stage_num} completed in {duration:.2f}s: {self.stages[stage_num]['name']}")
            
            self._save_status()
    
    def fail_stage(self, stage_num: int, error: str):
        """Mark stage as failed."""
        if stage_num in self.stages:
            self.stages[stage_num]["status"] = "Failed"
            self.stages[stage_num]["details"] = f"Error: {error}"
            self.stages[stage_num]["end_time"] = datetime.now().isoformat()
            logger.error(f"Stage {stage_num} failed: {error}")
            self._save_status()
    
    def get_stage_status(self, stage_num: int) -> Dict[str, Any]:
        """Get status of a specific stage."""
        return self.stages.get(stage_num, {})
    
    def get_all_statuses(self) -> Dict[int, Dict[str, Any]]:
        """Get all stage statuses."""
        return self.stages
    
    def is_completed(self) -> bool:
        """Check if all stages are completed."""
        return all(stage["status"] == "Completed" for stage in self.stages.values())
    
    def get_progress_percentage(self) -> float:
        """Get overall progress percentage."""
        completed = sum(1 for stage in self.stages.values() if stage["status"] == "Completed")
        return (completed / len(self.stages)) * 100
    
    def _save_status(self):
        """Save status to JSON file."""
        status_file = self.output_dir / "pipeline_status.json"
        with open(status_file, 'w') as f:
            json.dump(self.stages, f, indent=2)
    
    def load_status(self):
        """Load status from JSON file if exists."""
        status_file = self.output_dir / "pipeline_status.json"
        if status_file.exists():
            with open(status_file, 'r') as f:
                self.stages = json.load(f)
            logger.info("Pipeline status loaded from file")
    
    def generate_report(self) -> str:
        """Generate human-readable status report."""
        report = ["=" * 60]
        report.append("KABADDI INJURY PREDICTION PIPELINE STATUS")
        report.append("=" * 60)
        report.append(f"Overall Progress: {self.get_progress_percentage():.1f}%")
        report.append("-" * 60)
        
        for stage_num, stage in self.stages.items():
            report.append(f"\nStage {stage_num}: {stage['name']}")
            report.append(f"  Status: {stage['status']}")
            if stage['details']:
                report.append(f"  Details: {stage['details']}")
            if stage['output_path']:
                report.append(f"  Output: {stage['output_path']}")
            
            if stage['start_time'] and stage['end_time']:
                start = datetime.fromisoformat(stage['start_time'])
                end = datetime.fromisoformat(stage['end_time'])
                duration = (end - start).total_seconds()
                report.append(f"  Duration: {duration:.2f}s")
        
        report.append("=" * 60)
        return "\n".join(report)
    
    def print_report(self):
        """Print status report to console."""
        print(self.generate_report())
    
    def export_ui_cards(self, output_dir: Path):
        """Export status as UI card images."""
        try:
            from utils.visualization import VideoAnnotator
            import cv2
            
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            
            annotator = VideoAnnotator(420, 160)
            
            for stage_num in sorted(self.stages.keys()):
                stage = self.stages[stage_num]
                try:
                    card = annotator.create_status_card(
                        stage_name=f"Stage {stage_num}: {stage['name']}",
                        status=stage['status'],
                        details=stage.get('details', '')
                    )
                    
                    card_path = output_dir / f"stage_{stage_num:02d}_status.png"
                    success = cv2.imwrite(str(card_path), card)
                    
                    if success:
                        logger.debug(f"Status card exported: {card_path}")
                    else:
                        logger.error(f"Failed to write status card: {card_path}")
                except Exception as e:
                    logger.error(f"Error creating status card for stage {stage_num}: {e}")
            
            logger.info(f"UI status cards exported to {output_dir}")
        except Exception as e:
            logger.error(f"Error exporting UI cards: {e}")


class MetricsLogger:
    """Log and track various metrics during pipeline execution."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.metrics = {
            "frame_times": [],
            "detection_counts": [],
            "raider_detections": [],
            "fall_events": [],
            "risk_scores": [],
            "timestamps": [],
        }
    
    def log_frame_metrics(
        self, 
        frame_num: int, 
        detection_count: int = 0,
        raider_detected: bool = False,
        fall_detected: bool = False,
        risk_score: float = 0.0,
        processing_time: float = 0.0
    ):
        """Log metrics for a single frame."""
        self.metrics["timestamps"].append(frame_num)
        self.metrics["detection_counts"].append(detection_count)
        self.metrics["raider_detections"].append(1 if raider_detected else 0)
        self.metrics["fall_events"].append(1 if fall_detected else 0)
        self.metrics["risk_scores"].append(risk_score)
        self.metrics["frame_times"].append(processing_time)
    
    def save_metrics(self):
        """Save metrics to JSON."""
        metrics_file = self.output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        import numpy as np
        
        return {
            "total_frames": len(self.metrics["timestamps"]),
            "avg_detections": np.mean(self.metrics["detection_counts"]) if self.metrics["detection_counts"] else 0,
            "raider_detection_rate": np.mean(self.metrics["raider_detections"]) if self.metrics["raider_detections"] else 0,
            "total_fall_events": sum(self.metrics["fall_events"]),
            "avg_risk_score": np.mean(self.metrics["risk_scores"]) if self.metrics["risk_scores"] else 0,
            "max_risk_score": max(self.metrics["risk_scores"]) if self.metrics["risk_scores"] else 0,
            "avg_processing_time": np.mean(self.metrics["frame_times"]) if self.metrics["frame_times"] else 0,
        }
