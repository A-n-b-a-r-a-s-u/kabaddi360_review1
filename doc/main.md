# main.py - Documentation

## 1. Overview
`main.py` is the central orchestrator of the Kabaddi Injury Prediction System. It serves as the entry point for the video processing pipeline, managing the flow of data through all 7 stages of analysis. It handles video input, initializes all analysis modules (player detection, raider identification, pose estimation, etc.), processes each frame sequentially, and produces the final annotated video and injury risk reports.

## 2. Technical Overview
- **Type**: Entry Point / Orchestrator Script
- **Dependencies**: 
    - `opencv` (cv2): Video reading/writing and image manipulation.
    - `numpy`: Array operations.
    - `loguru`: robust logging.
    - `tqdm`: Progress bar.
    - Internal Modules: `models.*`, `utils.*`, `config.*`.
- **Input**: A video file (MP4, AVI, etc.) containing a Kabaddi match.
- **Output**: 
    - Annotated Video (`final_output.mp4`)
    - JSON Reports (`pipeline_summary.json`, `metrics.json`)
    - Status Cards (Images in `status_cards/`)
    - Logs (`logs/pipeline_*.log`)

## 3. Detailed Technical
The script defines the `KabaddiInjuryPipeline` class, which encapsulates the entire processing logic.
- **Initialization**: Sets up logging, creates output directories, and instantiates all model classes (PlayerDetector, RaiderIdentifier, etc.).
- **Execution Loop**:
    1.  Opens the video file.
    2.  Iterates through frames using a loop.
    3.  Passes each frame through a cascade of 7 stages.
    4.  Collects metrics and intermediate results.
    5.  Writes the processed frame to the output video.
    6.  Periodically clears GPU memory to prevent OOM errors.
- **Error Handling**: Implements a try-catch block for the entire pipeline and per-frame processing to ensure robustness. If a frame fails, it logs the error and continues/skips ensuring the pipeline doesn't crash entirely.
- **Resource Management**: Uses `VideoReader` and `VideoWriter` context managers (or manual release) and explicit CUDA cache clearing.

## 4. Workflow and Function Details

### Workflow
1.  **Stage 1: Player Detection**: Detects all persons in the frame.
2.  **Stage 2: Raider Identification**: Identifies the active raider among detected players.
3.  **Stage 3: Pose Estimation**: Extracts 33 skeletal keypoints for the raider.
4.  **Stage 4: Fall Detection**: Analyzes pose changes to detect falls.
5.  **Stage 5: Motion Analysis**: Computes velocity, acceleration, and specific motion abnormalities.
6.  **Stage 6: Impact Detection**: Detects collisions between the raider and defenders.
7.  **Stage 7: Risk Fusion**: Combines all previous signals into a single injury risk score (0-100%).

### Functions

#### `configure_logging(output_dir: Path)`
- **Purpose**: Sets up the logging configuration.
- **Details**: Creates a `logs` directory. configured `loguru` to output INFO to console and DEBUG to a timestamped file. Also updates a `logs_index.json` for tracking multiple runs.

#### `KabaddiInjuryPipeline` Class

##### `__init__(self, video_path: str, output_dir: Optional[str])`
- **Purpose**: Constructor.
- **Details**: Validates input video path. Creates output directory. Initializes the `PipelineStatus` and `MetricsLogger`. Sets module placeholders to None (lazy/delayed init).

##### `_initialize_modules(self, frame_width: int, frame_height: int)`
- **Purpose**: Instantiates all 7 model classes.
- **Details**: Some models (like RaiderIdentifier) need frame dimensions for logic (e.g., defining court boundaries), so this is called after opening the video.

##### `run(self, max_frames: Optional[int], save_intermediate: bool)`
- **Purpose**: Main execution method.
- **Details**: 
    - Opens video reader/writer.
    - Loops over frames with `tqdm`.
    - Calls `_process_frame` for each.
    - Handles GPU memory cleanup every 100 frames.
    - Finalizes pipeline (saves reports) at the end.
    - Returns a dictionary of results.

##### `_process_frame(self, frame, frame_num, save_intermediate)`
- **Purpose**: Runs all 7 stages on a single frame.
- **Details**: 
    - **Logic**: Sequential if-checks. If Stage 2 fails (no raider), subsequent stages (3-7) are skipped for efficiency.
    - **Data Flow**: `frame` -> `players` -> `raider_id` -> `pose` -> `fall/motion/impact` -> `risk`.
    - **Visualization**: Each stage function (e.g., `annotate_pose`) returns an annotated frame which is passed to the next stage.

##### `_finalize_pipeline(self)`
- **Purpose**: Cleanup and reporting.
- **Details**: Marks any unfinished stages as completed/skipped in the status tracker. Saves all JSON metrics and summaries. Generates UI status cards.

#### `main()`
- **Purpose**: CLI entry point.
- **Details**: Parses command line arguments (`input_video`, `output_dir`, `max_frames`). Instantiates and runs the pipeline. Prints a summary to stdout.
