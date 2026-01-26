# pipeline_status.py - Documentation

## 1. Overview
`utils/pipeline_status.py` is responsible for tracking the real-time progress of the analysis pipeline. Since video processing can take minutes, this module ensures the user is kept informed about exactly which stage is running (e.g., "Processing Stage 3: Pose Estimation"). It generates JSON status files and visual "Status Cards" which are displayed in the Streamlit UI.

## 2. Technical Overview
- **Type**: State Management & Logging Utility
- **Dependencies**: 
    - `json`: Persistence.
    - `cv2`: Generating image-based status cards.
    - `loguru`: Logging events.
- **Data Structure**: Deep nested dictionary tracking `status`, `start_time`, `end_time`, and `details` for all 7 stages.

## 3. Detailed Technical
- **Status Persistence**: Every time a stage starts or finishes, the state is saved to `pipeline_status.json`. This allows the frontend (Streamlit) to poll this file and update its progress bars asynchronously without blocking the backend.
- **Metrics Logging**: The `MetricsLogger` class buffers frame-by-frame stats (e.g., "Frame 45: Risk 12%, Fall: No") to memory and dumps them to `metrics.json` at the end.
- **UI Card Export**: It renders small PNG images representing the status of each stage (Green=Done, Orange=Running, Grey=Pending). This is a clever workaround to display rich status in strict UI frameworks by simply loading an image.

## 4. Functions

### `PipelineStatus` Class

##### `__init__(self, output_dir)`
- **Purpose**: Setup.
- **Details**: Creates the 7-stage dictionary template.

##### `start_stage(self, stage_num)`
- **Purpose**: Mark start.
- **Details**: Sets status to "Processing", records timestamp.

##### `complete_stage(self, stage_num, details)`
- **Purpose**: Mark done.
- **Details**: Sets status to "Completed", calculates duration.

##### `export_ui_cards(self, output_dir)`
- **Purpose**: Rendering.
- **Details**: Calls `visualization.create_status_card` for each stage and saves the result as a PNG file.

### `MetricsLogger` Class

##### `log_frame_metrics(self, ...)`
- **Purpose**: Data collection.
- **Details**: Appends scalar values to internal lists.

##### `get_summary_stats(self)`
- **Purpose**: Aggregation.
- **Details**: Computes global averages (Avg Risk, Total Falls) for the final report.
