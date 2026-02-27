"""
Main Pipeline: Kabaddi Injury Prediction System
Orchestrates all 7 stages and produces final annotated output.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from loguru import logger
from tqdm import tqdm
import sys
from datetime import datetime
import json
import threading
import time
import requests

# Import all modules
from config.config import (
    STAGE_DIRS, OUTPUT_DIR, VIDEO_CONFIG, LOG_CONFIG, PERFORMANCE_CONFIG
)

from utils.video_utils import VideoReader, VideoWriter
from utils.pipeline_status import PipelineStatus, MetricsLogger
from utils.visualization import VideoAnnotator

from models.court_line_detector import CourtLineDetector
from models.player_detector import PlayerDetector
from models.raider_identifier import RaiderIdentifier, annotate_raider_crossing
from models.pose_estimator import PoseEstimator, annotate_pose
from models.fall_detector import FallDetector, annotate_fall_detection
from models.motion_analyzer import MotionAnalyzer, annotate_motion_analysis
from models.impact_detector import ImpactDetector, annotate_impact_detection
from models.risk_fusion import RiskFusionEngine, annotate_risk_score
from utils.value_logger import ValueLogger
from utils.event_logger import EventLogger
from utils.raider_status_tracker import RaiderStatusTracker

# Server imports
try:
    import uvicorn
    from server import app as server_app
    SERVER_AVAILABLE = True
except ImportError:
    logger.warning("Server dependencies not available. Install with: pip install uvicorn fastapi")
    SERVER_AVAILABLE = False


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


# ============================================================================
# SERVER INITIALIZATION AND MANAGEMENT
# ============================================================================

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
server_thread = None
server_running = False

def start_server():
    """Start FastAPI server in background thread."""
    global server_thread, server_running
    
    if not SERVER_AVAILABLE:
        logger.warning("Server not available. Install uvicorn and fastapi dependencies.")
        return False
    
    if server_running:
        logger.info("Server already running.")
        return True
    
    def run_server():
        """Run server in thread."""
        try:
            print("\n" + "="*70)
            print("          KABADDI INJURY PREDICTION SERVER STARTING")
            print("="*70)
            print(f"Server starting on: {SERVER_URL}")
            print(f"WebSocket endpoint: ws://{SERVER_HOST}:{SERVER_PORT}/ws")
            print(f"API documentation: {SERVER_URL}/docs")
            print("="*70 + "\n")
            
            logger.info(f"[SERVER] Starting server on {SERVER_URL}")
            
            uvicorn.run(
                server_app,
                host=SERVER_HOST,
                port=SERVER_PORT,
                log_level="info",
                access_log=True
            )
        except Exception as e:
            logger.error(f"[SERVER] Failed to start server: {e}")
    
    # Start server in daemon thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    server_running = True
    
    # Wait for server to be ready
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(f"{SERVER_URL}/health", timeout=2)
            if response.status_code == 200:
                logger.info(f"[SERVER] Server is ready and healthy")
                print("\n✓ Server is READY and HEALTHY")
                print(f"✓ Listening on {SERVER_URL}")
                print("✓ Pipeline can now send real-time events\n")
                return True
        except:
            if i < max_retries - 1:
                time.sleep(0.5)
    
    logger.warning("[SERVER] Server started but health check failed - may still be initializing")
    return True

def send_raider_identified_event(raider_id: int, frame: int, timestamp: float, confidence: float = 1.0):
    """Send raider identification event to server."""
    if not server_running:
        return
    
    try:
        event_data = {
            "raider_id": raider_id,
            "frame": frame,
            "timestamp": timestamp,
            "confidence": confidence
        }
        
        response = requests.post(
            f"{SERVER_URL}/event/raider-identified",
            json=event_data,
            timeout=2
        )
        
        if response.status_code == 200:
            logger.info(f"[EVENT] Raider Identified event sent (ID: {raider_id}, Frame: {frame})")
        else:
            logger.debug(f"[EVENT] Server returned {response.status_code}")
    except Exception as e:
        logger.debug(f"[EVENT] Could not send raider event: {e}")

def send_injury_risk_event(raider_id: int, frame: int, timestamp: float, risk_score: float, 
                          risk_level: str, components: Dict):
    """Send injury risk update event to server."""
    if not server_running:
        return
    
    try:
        event_data = {
            "raider_id": raider_id,
            "frame": frame,
            "timestamp": timestamp,
            "risk_score": float(risk_score),
            "risk_level": risk_level,
            "components": {
                "fall_severity": float(components.get("fall_severity", 0)),
                "impact_severity": float(components.get("impact_severity", 0)),
                "motion_abnormality": float(components.get("motion_abnormality", 0)),
                "injury_history": float(components.get("injury_history", 0))
            }
        }
        
        response = requests.post(
            f"{SERVER_URL}/event/injury-risk",
            json=event_data,
            timeout=2
        )
        
        if response.status_code == 200:
            logger.info(f"[EVENT] Injury Risk event sent (Score: {risk_score:.1f}, Level: {risk_level})")
        else:
            logger.debug(f"[EVENT] Server returned {response.status_code}")
    except Exception as e:
        logger.debug(f"[EVENT] Could not send injury risk event: {e}")

def send_collision_event(raider_id: int, frame: int, timestamp: float, 
                        defender_ids: List[int], severity: float):
    """Send collision/touch event to server."""
    if not server_running:
        return
    
    try:
        event_data = {
            "raider_id": raider_id,
            "frame": frame,
            "timestamp": timestamp,
            "defender_ids": defender_ids,
            "collision_severity": float(severity)
        }
        
        response = requests.post(
            f"{SERVER_URL}/event/collision",
            json=event_data,
            timeout=2
        )
        
        if response.status_code == 200:
            logger.info(f"[EVENT] Collision event sent (Defenders: {defender_ids})")
        else:
            logger.debug(f"[EVENT] Server returned {response.status_code}")
    except Exception as e:
        logger.debug(f"[EVENT] Could not send collision event: {e}")

def send_fall_event(raider_id: int, frame: int, timestamp: float, 
                   fall_severity: float, indicators: List[str]):
    """Send fall detection event to server."""
    if not server_running:
        return
    
    try:
        event_data = {
            "raider_id": raider_id,
            "frame": frame,
            "timestamp": timestamp,
            "fall_severity": float(fall_severity),
            "indicators": indicators
        }
        
        response = requests.post(
            f"{SERVER_URL}/event/fall",
            json=event_data,
            timeout=2
        )
        
        if response.status_code == 200:
            logger.critical(f"[EVENT] FALL event sent (Severity: {fall_severity:.1f})")
        else:
            logger.debug(f"[EVENT] Server returned {response.status_code}")
    except Exception as e:
        logger.debug(f"[EVENT] Could not send fall event: {e}")

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
        
        # Initialize value logger (simple, non-intrusive)
        self.logger = ValueLogger(self.output_dir, max_frames=1000)
        
        # Initialize event logger for real-time dashboard updates
        self.event_logger = EventLogger(self.output_dir)
        self.raider_status = RaiderStatusTracker()
        self.last_flush_frame = 0  # Track frame count for periodic flush
        
        # Initialize all stage modules
        logger.info("Initializing pipeline modules...")
        self.court_line_detector = None
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
        
        # Collision tracking for display (show collision text for 20 frames)
        self.last_collision_frame = -100  # Frame of last collision
        self.collision_text = ""  # Text to display
        self.collision_history = []  # List of all collisions
        
        # Server event tracking
        self.raider_identified_sent = False  # Flag to send raider event only once
        self.last_risk_event_frame = -1  # Track last risk event frame for throttling
        self.risk_event_throttle = 5  # Send risk events every N frames max
        
        logger.info(f"Pipeline initialized for video: {video_path}")
    
    def _initialize_modules(self, frame_width: int, frame_height: int):
        """Initialize all processing modules."""
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self.court_line_detector = CourtLineDetector()
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
        
        # Start server for real-time event streaming
        print("\n" + "="*70)
        print("                    INITIALIZING SERVER")
        print("="*70)
        if not start_server():
            logger.warning("Could not start server. Continuing without real-time events.")
        else:
            logger.info("[MAIN] Server started successfully - real-time events enabled")
        print("="*70 + "\n")
        
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
        self.frame_count = total_frames  # Store for access in _process_frame
        
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
            
            # Finalize event logger - move to results
            self.event_logger.finalize_events()
            
            # Export logged values (non-intrusive, after all processing)
            self.logger.export()
            
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
        logger.info("Intermediate step values computed successfully")
        logger.info("="*60)
        
        # Save collision data to JSON
        collision_data_path = self.output_dir / "collision_data.json"
        collision_data = {
            "total_collisions": len(self.collision_history),
            "collisions": self.collision_history
        }
        with open(collision_data_path, 'w') as f:
            json.dump(collision_data, f, indent=2)
        logger.info(f"Collision data saved: {collision_data_path}")
        logger.info(f"Total collision events recorded: {len(self.collision_history)}")
        
        # Print success message to CLI
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print(f"✓ Final output: {final_output_path}")
        print("✓ Intermediate step values computed successfully")
        if self.collision_history:
            print(f"✓ Collision events recorded: {len(self.collision_history)}")
        print("="*60 + "\n")
        
        # Print status report
        self.status.print_report()
        
        return {
            "output_video": str(final_output_path),
            "status": self.status.get_all_statuses(),
            "metrics": self.metrics.get_summary_stats(),
            "collision_data": collision_data
        }
    
    def classify_players_by_team(self, players: List[Dict], raider_id: Optional[int], detection_line_x: int) -> Dict[str, List[Dict]]:
        """
        Classify players into raider, attackers (left side), and defenders (right side).
        
        Args:
            players: List of detected players
            raider_id: Track ID of raider (if detected)
            detection_line_x: x-coordinate of 25% detection line
            
        Returns:
            Dict with keys: 'raider', 'attackers', 'defenders'
        """
        raider = None
        attackers = []
        defenders = []
        
        for player in players:
            track_id = player["track_id"]
            player_x = player["center"][0]
            
            if track_id == raider_id:
                raider = player
            elif player_x < detection_line_x:
                # Left side = raider's team (attackers)
                attackers.append(player)
            else:
                # Right side = defending team
                defenders.append(player)
        
        return {
            "raider": raider,
            "attackers": attackers,
            "defenders": defenders
        }
    
    def detect_collisions(self, raider: Optional[Dict], defenders: List[Dict], collision_distance: int = 100) -> tuple:
        """
        Detect collisions between raider and defenders.
        
        Args:
            raider: Raider player data
            defenders: List of defender player data
            collision_distance: Distance threshold for collision (pixels)
            
        Returns:
            (colliding_defenders_ids, impact_score)
        """
        if raider is None or not defenders:
            return [], 0.0
        
        raider_x, raider_y = raider["center"]
        colliding_defenders = []
        impact_score = 0.0
        
        for defender in defenders:
            def_x, def_y = defender["center"]
            
            # Euclidean distance
            distance = np.sqrt((raider_x - def_x)**2 + (raider_y - def_y)**2)
            
            if distance < collision_distance:
                colliding_defenders.append(defender["track_id"])
                impact_score = min(impact_score + 0.3, 1.0)  # Cap at 1.0
                logger.debug(f"[COLLISION] Raider {raider['track_id']} hit by Defender {defender['track_id']} at distance {distance:.1f}px")
        
        return colliding_defenders, impact_score
    
    def draw_all_players(self, frame: np.ndarray, players: List[Dict], raider_id: Optional[int], 
                        detection_line_x: int, colliding_defenders: List[int]) -> np.ndarray:
        """
        Draw all players with appropriate colors and labels:
        - Raider: RED box, label "ID"
        - Attackers (left side): GREEN box, label "ID"
        - Defenders (right side): BLUE box, label "DEFENDER ID"
        
        Args:
            frame: Frame to draw on
            players: List of players
            raider_id: Track ID of raider
            detection_line_x: x-coordinate of 25% line
            colliding_defenders: List of defender track IDs colliding with raider
            
        Returns:
            Annotated frame
        """
        for player in players:
            track_id = player["track_id"]
            bbox = player["bbox"]
            x1, y1, x2, y2 = bbox
            
            if track_id == raider_id:
                # RAIDER - RED
                color = (0, 0, 255)  # RED in BGR
                label = f"RAIDER {track_id}"
                thickness = 3
            elif player["center"][0] < detection_line_x:
                # ATTACKER (left side) - GREEN
                color = (0, 255, 0)  # GREEN
                label = f"{track_id}"
                thickness = 2
            else:
                # DEFENDER (right side) - BLUE
                color = (255, 0, 0)  # BLUE
                label = f"DEFENDER {track_id}"
                thickness = 2
                
                # Highlight if colliding with raider
                if track_id in colliding_defenders:
                    color = (0, 165, 255)  # ORANGE (collision highlight)
                    thickness = 3
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # Draw label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_color = (255, 255, 255)  # White text
            text_thickness = 1
            
            text_size = cv2.getTextSize(label, font, font_scale, text_thickness)[0]
            text_x = int(x1)
            text_y = max(int(y1) - 5, 20)
            
            # Text background
            cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), 
                         (text_x + text_size[0] + 5, text_y + 5), color, -1)
            cv2.putText(frame, label, (text_x + 2, text_y - 2), font, font_scale, text_color, text_thickness)
        
        return frame
    
    def save_collision_data(self, frame_num: int, raider_id: int, colliding_defenders: List[int]):
        """Save collision data to history."""
        if colliding_defenders:
            collision_record = {
                "frame": frame_num,
                "raider_id": raider_id,
                "defender_ids": colliding_defenders,
                "defender_count": len(colliding_defenders)
            }
            self.collision_history.append(collision_record)
            logger.info(f"[COLLISION] Frame {frame_num}: Raider {raider_id} hit by Defenders {colliding_defenders}")
    
    def _process_frame(self, frame: np.ndarray, frame_num: int, save_intermediate: bool) -> np.ndarray:
        """Process single frame through all 8 stages (Stage 0-7)."""
        import time
        start_time = time.time()
        
        # Update frame count for event logger
        self.event_logger.update_frame(frame_num)
        
        # Calculate timestamp
        timestamp = frame_num / self.fps if self.fps > 0 else 0.0
        
        try:
            # Stage 0: Court Line Detection
            if frame_num == 0:
                self.status.start_stage(0)
            
            logger.debug(f"[MAIN PIPELINE] ============ FRAME {frame_num} START ============")
            logger.debug(f"[MAIN PIPELINE - STAGE 0] Starting court line detection...")
            
            # For court line intermediate saves, use output_dir if available
            court_line_save_dir = self.output_dir / "stage0_court_lines" if self.output_dir else STAGE_DIRS[0]
            if save_intermediate and frame_num == 0:
                court_line_save_dir.mkdir(parents=True, exist_ok=True)
            
            court_lines_frame, line_mapping = self.court_line_detector.process_frame(frame, frame_num)
            logger.debug(f"[MAIN PIPELINE - STAGE 0] Detected {len(line_mapping)} court lines")
            
            if frame_num == 0:
                self.status.complete_stage(0, f"Detected {len(line_mapping)} court lines")
                # Save court line coordinates on first frame
                self.court_line_detector.save_line_coordinates(line_mapping, STAGE_DIRS[0])
                if len(line_mapping) == 0:
                    logger.warning("[MAIN PIPELINE - STAGE 0] ⚠️ NO COURT LINES DETECTED - Check HSV thresholding parameters")
            
            # Save Stage 0 output to session directory (court line processing steps)
            if save_intermediate and frame_num in [0, int(self.frame_count // 3), int(self.frame_count // 1.5)]:
                # Only save the final court line frame, not intermediate steps (those are in process_frame)
                stage0_path = court_line_save_dir / f"frame_{frame_num:06d}_final.jpg"
                cv2.imwrite(str(stage0_path), court_lines_frame)
            
            # Use court_lines_frame as base, all subsequent annotations build on top
            annotated_frame = court_lines_frame.copy()  # Use .copy() to ensure clean slate
            
            # Stage 1: Player Detection & Tracking
            if frame_num == 0:
                self.status.start_stage(1)
            
            logger.debug(f"[MAIN PIPELINE - STAGE 1] Starting player detection...")
            # Process frame to get detections (don't use its annotations to preserve court lines)
            _, players = self.player_detector.process_frame(frame, frame_num)
            logger.debug(f"[MAIN PIPELINE - STAGE 1] Detected {len(players)} players")
            
            if frame_num == 0:
                self.status.complete_stage(1, f"Detected {len(players)} players")
            
            # Stage 2: Raider Identification - NEW Simple 25% Line Crossing Method
            if frame_num == 0:
                self.status.start_stage(2)
            
            logger.debug(f"[MAIN PIPELINE - STAGE 2] Starting raider detection (25% line crossing)...")
            raider_id = self.raider_identifier.detect_raider_by_line_crossing(players)
            logger.debug(f"[MAIN PIPELINE - STAGE 2] Raider detected: {raider_id}")
            
            # Send raider identification event to server (only once)
            if raider_id is not None and not self.raider_identified_sent:
                self.raider_identified_sent = True
                send_raider_identified_event(
                    raider_id=raider_id,
                    frame=frame_num,
                    timestamp=timestamp,
                    confidence=1.0
                )
                print(f"\n{'='*70}")
                print(f"✓ RAIDER IDENTIFIED: Track ID = {raider_id}, Frame = {frame_num}")
                print(f"  Event sent to server")
                print(f"{'='*70}\n")
            
            # Save Stage 1 output only when both defenders and raider detected
            if save_intermediate and len(players) > 0 and raider_id is not None:
                stage1_dir = self.output_dir / "stage1_detection" if self.output_dir else STAGE_DIRS[1]
                if frame_num == 0:
                    stage1_dir.mkdir(parents=True, exist_ok=True)
                stage1_path = stage1_dir / f"frame_{frame_num:06d}_players.jpg"
                cv2.imwrite(str(stage1_path), annotated_frame)
            
            # Log raider detection event
            if raider_id is not None and self.raider_status.raider_id is None:
                self.raider_status.set_raider_detected(
                    raider_id,
                    confidence=100.0,
                    detection_time=timestamp,
                    event_logger=self.event_logger
                )
            
            # Get detection line x position for team classification
            detection_line_x = self.raider_identifier.raider_detection_line_x
            
            # Classify all players by team
            team_data = self.classify_players_by_team(players, raider_id, detection_line_x)
            raider_player = team_data["raider"]
            attackers = team_data["attackers"]
            defenders = team_data["defenders"]
            
            logger.debug(f"[MAIN PIPELINE - STAGE 2] Teams: Raider={raider_id}, Attackers={len(attackers)}, Defenders={len(defenders)}")
            
            # Detect collisions between raider and defenders
            colliding_defenders = []
            collision_impact_score = 0.0
            if raider_player:
                colliding_defenders, collision_impact_score = self.detect_collisions(raider_player, defenders, collision_distance=100)
                if colliding_defenders:
                    self.last_collision_frame = frame_num
                    self.collision_text = f"COLLISION: Raider {raider_id} hit by Defender {','.join(map(str, colliding_defenders))}"
                    self.save_collision_data(frame_num, raider_id, colliding_defenders)
                    
                    # Send collision event to server
                    send_collision_event(
                        raider_id=raider_id,
                        frame=frame_num,
                        timestamp=timestamp,
                        defender_ids=colliding_defenders,
                        severity=collision_impact_score * 100  # Convert to 0-100 scale
                    )
                    
                    # Log touch event for each defender
                    for defender_id in colliding_defenders:
                        self.raider_status.raider_touched(
                            defender_id,
                            frame_num,
                            timestamp,
                            event_logger=self.event_logger
                        )
                    
                    logger.info(f"[MAIN PIPELINE - STAGE 2] {self.collision_text}")
            
            # Draw all players on frame
            annotated_frame = self.draw_all_players(annotated_frame.copy(), players, raider_id, 
                                                   detection_line_x, colliding_defenders)
            
            # Display collision text if within 20 frames of collision
            if frame_num - self.last_collision_frame < 20:
                # Draw collision text at top of frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                color = (0, 0, 255)  # RED
                thickness = 2
                text_size = cv2.getTextSize(self.collision_text, font, font_scale, thickness)[0]
                
                # Background box
                text_y = 50
                cv2.rectangle(annotated_frame, (10, text_y - text_size[1] - 10),
                            (20 + text_size[0], text_y + 10), color, -1)
                cv2.putText(annotated_frame, self.collision_text, (15, text_y), font, font_scale, 
                           (255, 255, 255), thickness)
            
            if frame_num == 0:
                status_msg = f"Raider ID: {raider_id}" if raider_id else "Waiting for raider crossing..."
                self.status.complete_stage(2, status_msg)
            
            # Save Stage 2 output only when raider detected
            if save_intermediate and raider_id is not None:
                stage2_dir = self.output_dir / "stage2_raider" if self.output_dir else STAGE_DIRS[2]
                if frame_num == 0:
                    stage2_dir.mkdir(parents=True, exist_ok=True)
                stage2_path = stage2_dir / f"frame_{frame_num:06d}_raider.jpg"
                cv2.imwrite(str(stage2_path), annotated_frame)
        
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
                return annotated_frame
            
            # Get raider bbox from players list
            raider_player = next((p for p in players if p["track_id"] == raider_id), None)
            if raider_player is None:
                logger.debug(f"[MAIN PIPELINE] Raider {raider_id} not found in players list")
                return annotated_frame
            
            raider_bbox = raider_player["bbox"]
            
            # Stage 3: Pose Estimation (Raider Only)
            if frame_num == 0:
                self.status.start_stage(3)
            
            logger.debug(f"[MAIN PIPELINE - STAGE 3] Starting pose estimation...")
            joints = self.pose_estimator.extract_pose(frame, raider_bbox)
            angles = self.pose_estimator.calculate_joint_angles(joints) if joints else None
            logger.debug(f"[MAIN PIPELINE - STAGE 3] Pose extraction: {'SUCCESS' if joints else 'FAILED'}")
            annotated_frame = annotate_pose(annotated_frame.copy(), joints, angles)
            
            if frame_num == 0:
                status_msg = "Pose detected" if joints else "Pose detection failed"
                self.status.complete_stage(3, status_msg)
            

            # Stage 4: Fall Detection
            if frame_num == 0:
                self.status.start_stage(4)
            
            logger.debug(f"[MAIN PIPELINE - STAGE 4] Starting fall detection...")
            com = self.pose_estimator.get_center_of_mass(joints) if joints else None
            is_falling, fall_info = self.fall_detector.detect_fall(joints, frame_num, com)
            logger.debug(f"[MAIN PIPELINE - STAGE 4] Fall detected: {is_falling}")
            
            # Log fall event
            if is_falling and raider_id is not None:
                severity = fall_info.get("severity", "Unknown")
                indicators = fall_info.get("indicators", [])
                
                # Send fall event to server
                send_fall_event(
                    raider_id=raider_id,
                    frame=frame_num,
                    timestamp=timestamp,
                    fall_severity=float(severity) if isinstance(severity, (int, float)) else 0.0,
                    indicators=indicators
                )
                
                self.raider_status.fall_detected(
                    frame_num,
                    timestamp,
                    severity,
                    event_logger=self.event_logger
                )
            
            annotated_frame = annotate_fall_detection(annotated_frame.copy(), fall_info, frame_num)
            
            if frame_num == 0:
                self.status.complete_stage(4, "Fall detection active")
            
            # Save Stage 4 output to session directory
            if save_intermediate and is_falling:
                stage4_dir = self.output_dir / "stage4_falls" if self.output_dir else STAGE_DIRS[4]
                if save_intermediate and frame_num == 0:
                    stage4_dir.mkdir(parents=True, exist_ok=True)
                stage4_path = stage4_dir / f"fall_frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(stage4_path), annotated_frame)
            
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
            
            annotated_frame = annotate_motion_analysis(annotated_frame.copy(), motion_metrics, motion_abnormality)
            
            if frame_num == 0:
                self.status.complete_stage(5, f"Motion abnormality: {motion_abnormality:.1f}%")
            

            # Stage 6: Impact Detection
            if frame_num == 0:
                self.status.start_stage(6)
            
            logger.debug(f"[MAIN PIPELINE - STAGE 6] Starting impact detection...")
            impact_detected, impact_info = self.impact_detector.detect_impacts(
                players, raider_id, frame_num, collision_impact_score
            )
            logger.debug(f"[MAIN PIPELINE - STAGE 6] Impact detected: {impact_detected}")
            
            annotated_frame = annotate_impact_detection(
                annotated_frame.copy(), players, raider_id, impact_info
            )
            
            if frame_num == 0:
                self.status.complete_stage(6, "Impact detection active")
            
            # Save Stage 6 output to session directory
            if save_intermediate and impact_detected:
                stage6_dir = self.output_dir / "stage6_impact" if self.output_dir else STAGE_DIRS[6]
                if save_intermediate and frame_num == 0:
                    stage6_dir.mkdir(parents=True, exist_ok=True)
                stage6_path = stage6_dir / f"impact_frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(stage6_path), annotated_frame)
            
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
            
            annotated_frame = annotate_risk_score(annotated_frame.copy(), risk_data, show_breakdown=True, risk_engine=self.risk_fusion)
            
            if frame_num == 0:
                self.status.complete_stage(7, f"Risk: {risk_data['risk_level']}")
            
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
            
            # Send injury risk update event to server (throttled to every N frames)
            if raider_id is not None and (frame_num - self.last_risk_event_frame) >= self.risk_event_throttle:
                self.last_risk_event_frame = frame_num
                send_injury_risk_event(
                    raider_id=raider_id,
                    frame=frame_num,
                    timestamp=timestamp,
                    risk_score=risk_score_val,
                    risk_level=risk_data.get("risk_level", "UNKNOWN"),
                    components={
                        "fall_severity": fall_severity,
                        "impact_severity": impact_severity,
                        "motion_abnormality": motion_abnormality,
                        "injury_history": risk_data.get("components", {}).get("injury_history", 0.0)
                    }
                )
            
            logger.debug(f"[MAIN PIPELINE] ============ FRAME {frame_num} END (Time: {processing_time:.3f}s) ============")
            
            # Log calculation values after all processing is complete (non-intrusive)
            try:
                raider_info = {"track_id": raider_id} if raider_id else None
                self.logger.log_frame(frame_num, {
                    "players": players,
                    "raider_info": raider_info,
                    "joints": joints,
                    "is_falling": is_falling,
                    "fall_info": fall_info,
                    "motion_abnormality": motion_abnormality,
                    "impact_detected": impact_detected,
                    "impact_info": impact_info,
                    "risk_data": risk_data
                })
            except Exception as e:
                logger.debug(f"[VALUE LOGGER] Could not log frame values: {e}")
            
            # Flush events every 10 frames for dashboard updates
            if (frame_num - self.last_flush_frame) >= 10:
                self.event_logger.flush_events()
                self.last_flush_frame = frame_num
            
            return annotated_frame
        
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
        for stage_num in range(0, 8):
            stage = self.status.stages[stage_num]
            current_status = stage["status"]
            
            if current_status == "Processing":
                # If still processing, mark as completed
                self.status.complete_stage(stage_num, f"{stage['name']} completed")
            elif current_status == "Yet to start":
                # If never started (e.g., raider not detected), still mark as completed
                # since the pipeline ran but that stage was not needed
                if stage_num == 0:
                    self.status.complete_stage(stage_num, f"{stage['name']} skipped")
                else:
                    self.status.complete_stage(stage_num, f"{stage['name']} skipped - no raider detected")
                logger.debug(f"Stage {stage_num} marked as completed (skipped)")
        
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
        
        if server_running:
            print("\n" + "-"*60)
            print("SERVER STATUS")
            print("-"*60)
            print(f"✓ Server is running on {SERVER_URL}")
            print(f"✓ WebSocket: ws://{SERVER_HOST}:{SERVER_PORT}/ws")
            print(f"✓ API Docs: {SERVER_URL}/docs")
            print(f"✓ Real-time events were sent during processing")
            print("-"*60)
        
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
