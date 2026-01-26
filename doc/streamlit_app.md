# streamlit_app.py - Documentation

## 1. Overview
`streamlit_app.py` provides a user-friendly Web UI for the Kabaddi Injury Prediction System. It allows users to upload video files, run the analysis pipeline interactively, view real-time progress, and explore the results through interactive dashboards. It serves as the primary frontend for end-users.

## 2. Technical Overview
- **Type**: Web Application (Frontend + Controller)
- **Dependencies**: 
    - `streamlit`: Web framework.
    - `plotly`: Interactive charting (Risk timelines, pie charts).
    - `pandas`: Data manipulation for charts.
    - `cv2`, `PIL`: Image handling for status cards.
- **Integration**: Imports `KabaddiInjuryPipeline` from `main.py` directly to run processing in the backend.

## 3. Detailed Technical
The app uses Streamlit's session state to manage the workflow across re-runs.
- **Session State**: Tracks `processing_complete`, `results`, and `output_dir` to persist data between user interactions.
- **Tab Structure**:
    1.  **Upload & Process**: File uploader, progress bar simulation, and trigger for pipeline execution.
    2.  **Results Dashboard**: Loads JSON data (`pipeline_summary.json`) and renders Plotly charts.
    3.  **Downloads**: Provides download buttons for the final video, reports, and logs.
- **Simulation**: Uses a dummy progress loop (updating text) while the actual heavy processing runs in `pipeline.run()`. *Note: Streamlit runs synchronously, so the UI blocks during processing unless threaded, but here it simply updates status before/after the blocking call.*

## 4. Workflow and Function Details

### Workflow
1.  **Upload**: User selects a video.
2.  **Process**: User clicks "Start Analysis". System creates a new session timestamp folder.
3.  **Run Pipeline**: Calls `KabaddiInjuryPipeline.run()`.
4.  **Visualize**: Once done, reads the generated JSON files from the output session directory and displays charts.

### Functions

#### `process_video(uploaded_file, ...)`
- **Purpose**: Wrapper to run the pipeline on an uploaded file.
- **Details**: Saves the uploaded bytes to a temporary file, initializes the pipeline, runs it, and returns the results dict and output directory path.

#### Visualization Functions
- `create_risk_timeline_chart(summary)`: Plots Risk Score vs Frame Number using Plotly. Adds threshold lines (Low/Medium/High).
- `create_component_breakdown_chart(summary)`: Pie chart showing contribution of Fall, Impact, Motion, History to the total risk.
- `create_event_timeline(summary)`: Bubble chart showing when falls/impacts occurred and their severity.

#### `display_stage_status(status)`
- **Purpose**: Renders the 7-stage status grid.
- **Details**: Iterates through the status dictionary and creates Streamlit columns with success/info/error boxes for each stage (Player Detection, Raider ID, etc.).

#### `main()`
- **Purpose**: App entry point.
- **Details**: Sets up the page config (`st.set_page_config`), CSS styles, and tabs. Handles the control flow based on session state.
