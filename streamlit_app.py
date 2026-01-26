"""
Streamlit Web Dashboard for Kabaddi Injury Prediction System
Upload videos, process them, and view results in real-time.
"""

import streamlit as st
import sys
import os
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import cv2
import base64
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from main import KabaddiInjuryPipeline
from config.config import OUTPUT_DIR, RISK_FUSION_CONFIG, INJURY_HISTORY_PATH


# Page configuration
st.set_page_config(
    page_title="Kabaddi Injury Prediction System",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #4CAF50;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FF9800;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #F44336;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'output_dir' not in st.session_state:
        st.session_state.output_dir = None
    if 'results' not in st.session_state:
        st.session_state.results = None


def get_video_info(video_file):
    """Extract basic video information."""
    return {
        "name": video_file.name,
        "size": f"{video_file.size / (1024*1024):.2f} MB",
        "type": video_file.type
    }


def process_video(uploaded_file, max_frames=None, save_intermediate=True):
    """Process uploaded video through pipeline."""
    
    logger.info(f"[STREAMLIT APP] Processing video: {uploaded_file.name}")
    
    # Create temporary directory for this session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_DIR / f"session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"[STREAMLIT APP] Session directory: {session_dir}")
    
    # Save uploaded file
    input_path = session_dir / uploaded_file.name
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.debug(f"[STREAMLIT APP] Video saved to: {input_path}")
    
    # Initialize pipeline
    logger.debug(f"[STREAMLIT APP] Initializing pipeline...")
    pipeline = KabaddiInjuryPipeline(str(input_path), str(session_dir))
    
    # Process video
    logger.info(f"[STREAMLIT APP] Starting video processing...")
    results = pipeline.run(
        max_frames=max_frames,
        save_intermediate=save_intermediate
    )
    logger.info(f"[STREAMLIT APP] Video processing completed")
    
    return results, session_dir


def load_pipeline_status(output_dir):
    """Load pipeline status from JSON."""
    status_file = Path(output_dir) / "pipeline_status.json"
    if status_file.exists():
        with open(status_file, 'r') as f:
            return json.load(f)
    return None


def load_pipeline_summary(output_dir):
    """Load pipeline summary from JSON."""
    summary_file = Path(output_dir) / "pipeline_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return json.load(f)
    return None


def load_metrics(output_dir):
    """Load metrics from JSON."""
    metrics_file = Path(output_dir) / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None


def create_risk_timeline_chart(summary):
    """Create plotly chart for risk timeline."""
    if not summary or "risk" not in summary or "timeline" not in summary["risk"]:
        return None
    
    timeline = summary["risk"]["timeline"]
    if not timeline:
        return None
    
    df = pd.DataFrame(timeline)
    
    fig = go.Figure()
    
    # Risk score line
    fig.add_trace(go.Scatter(
        x=df['frame'],
        y=df['risk_score'],
        mode='lines',
        name='Risk Score',
        line=dict(color='#E53935', width=3),
        fill='tozeroy',
        fillcolor='rgba(229, 57, 53, 0.2)'
    ))
    
    # Add threshold lines
    fig.add_hline(y=70, line_dash="dash", line_color="orange", 
                  annotation_text="Medium Risk", annotation_position="right")
    fig.add_hline(y=30, line_dash="dash", line_color="green",
                  annotation_text="Low Risk", annotation_position="right")
    
    fig.update_layout(
        title="Injury Risk Over Time",
        xaxis_title="Frame Number",
        yaxis_title="Risk Score (0-100)",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_component_breakdown_chart(summary):
    """Create plotly chart for risk component breakdown."""
    if not summary or "risk" not in summary or "timeline" not in summary["risk"]:
        return None
    
    timeline = summary["risk"]["timeline"]
    if not timeline:
        return None
    
    # Get average contributions
    avg_fall = sum(t.get('fall_severity', 0) for t in timeline) / len(timeline)
    avg_impact = sum(t.get('impact_severity', 0) for t in timeline) / len(timeline)
    avg_motion = sum(t.get('motion_abnormality', 0) for t in timeline) / len(timeline)
    avg_history = sum(t.get('injury_history_score', 0) for t in timeline) / len(timeline)
    
    # Create weighted contributions
    weights = RISK_FUSION_CONFIG["weights"]
    contributions = {
        'Fall Events': avg_fall * weights['fall_severity'],
        'Impact Collisions': avg_impact * weights['impact_severity'],
        'Motion Abnormality': avg_motion * weights['motion_abnormality'],
        'Injury History': avg_history * weights['injury_history']
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=list(contributions.keys()),
        values=list(contributions.values()),
        hole=.4,
        marker_colors=['#E53935', '#FB8C00', '#FDD835', '#7B1FA2']
    )])
    
    fig.update_layout(
        title="Average Risk Component Breakdown",
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_event_timeline(summary):
    """Create event timeline visualization."""
    if not summary:
        return None
    
    events = []
    
    # Fall events
    if "falls" in summary and "events" in summary["falls"]:
        for fall in summary["falls"]["events"]:
            events.append({
                'frame': fall['frame'],
                'type': 'Fall',
                'severity': fall['severity']
            })
    
    # Impact events
    if "impacts" in summary and "events" in summary["impacts"]:
        for impact in summary["impacts"]["events"]:
            events.append({
                'frame': impact['frame'],
                'type': 'Impact',
                'severity': impact['severity']
            })
    
    if not events:
        return None
    
    df = pd.DataFrame(events)
    
    fig = px.scatter(
        df,
        x='frame',
        y='severity',
        color='type',
        size='severity',
        title='Detected Events Timeline',
        labels={'frame': 'Frame Number', 'severity': 'Severity (0-100)'},
        color_discrete_map={'Fall': '#E53935', 'Impact': '#FB8C00'},
        height=400
    )
    
    fig.update_layout(template='plotly_white')
    
    return fig


def display_stage_status(status):
    """Display status cards for each stage."""
    if not status:
        st.warning("Status information not available")
        return
    
    st.markdown("### 📊 Pipeline Stage Status")
    
    # Create 3 columns for stages
    for row in range(0, 7, 3):
        cols = st.columns(3)
        for idx, col in enumerate(cols):
            stage_num = row + idx + 1
            if stage_num > 7:
                break
            
            # Handle both string and int keys
            stage = status.get(str(stage_num)) or status.get(stage_num) or {}
            stage_name = stage.get('name', f'Stage {stage_num}')
            stage_status = stage.get('status', 'Yet to start')
            stage_details = stage.get('details', '')
            
            with col:
                if stage_status == "Completed":
                    st.success(f"✅ **{stage_name}**")
                elif stage_status == "Processing":
                    st.info(f"⏳ **{stage_name}**")
                elif stage_status == "Failed":
                    st.error(f"❌ **{stage_name}**")
                else:
                    st.warning(f"⏸️ **{stage_name}**")
                
                if stage_details:
                    st.caption(stage_details)


def download_button(file_path, label, file_name):
    """Create download button for files."""
    if not Path(file_path).exists():
        return False
    
    with open(file_path, "rb") as f:
        bytes_data = f.read()
    
    st.download_button(
        label=label,
        data=bytes_data,
        file_name=file_name,
        mime="application/octet-stream"
    )
    return True


def main():
    """Main Streamlit application."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header"> Kabaddi Injury Aware System</div>', 
                unsafe_allow_html=True)
    st.markdown("**Done by Anbarasu with the guidance of Dr. T. Mala Mam (Professor)**")
    
    # Processing parameters
    # Always enable intermediate steps
    max_frames = None
    save_intermediate = True
    
    # Main content tabs
    tab1, tab2 = st.tabs([" Upload & Process", " Results Dashboard"])
    
    with tab1:
        st.markdown("###  Upload Kabaddi Match Video")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a Kabaddi match video for analysis"
        )
        
        if uploaded_file is not None:
            # Display video info
            video_info = get_video_info(uploaded_file)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Name", video_info['name'])
            with col2:
                st.metric("File Size", video_info['size'])
            with col3:
                st.metric("File Type", video_info['type'])
            
            st.markdown("---")
            
            # Auto-start analysis when video is uploaded
            if not st.session_state.processing_complete:
                with st.spinner("🔄 Processing video... This may take several minutes."):
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Simulate progress stages
                        stages = [
                            "Initializing pipeline...",
                            "Detecting players...",
                            "Identifying raider...",
                            "Analyzing pose...",
                            "Detecting falls...",
                            "Analyzing motion...",
                            "Detecting impacts...",
                            "Calculating risk scores...",
                            "Generating outputs..."
                        ]
                        
                        logger.info("[STREAMLIT APP] Starting video processing...")
                        
                        for i, stage in enumerate(stages):
                            status_text.text(stage)
                            progress_bar.progress((i + 1) / len(stages))
                            
                            if i == 2:  # Actually start processing after initialization
                                logger.debug("[STREAMLIT APP] Calling process_video function...")
                                results, output_dir = process_video(
                                    uploaded_file,
                                    max_frames=max_frames if max_frames else None,
                                    save_intermediate=save_intermediate
                                )
                                logger.debug(f"[STREAMLIT APP] process_video returned with output_dir: {output_dir}")
                                st.session_state.results = results
                                st.session_state.output_dir = output_dir
                                logger.debug("[STREAMLIT APP] Results stored in session state")
                            
                            time.sleep(0.3)
                        
                        st.session_state.processing_complete = True
                        progress_bar.progress(1.0)
                        status_text.text("✅ Processing complete!")
                        logger.info("[STREAMLIT APP] Processing marked as complete")
                        
                        st.success("🎉 Video processing completed successfully!")
                        
                        # Display video playback
                        st.markdown("---")
                        st.markdown("### 🎬 Processed Video")
                        final_video = Path(st.session_state.output_dir) / "final_output.mp4"
                        if final_video.exists():
                            st.video(str(final_video))
                            st.success("✅ Video processed and ready to view!")
                            
                            # Add download button
                            with open(final_video, "rb") as f:
                                st.download_button(
                                    label="⬇️ Download Annotated Video",
                                    data=f.read(),
                                    file_name="kabaddi_analysis.mp4",
                                    mime="video/mp4"
                                )
                        else:
                            st.warning("⚠️ Output video not found. Check processing logs.")
                        
                        st.markdown("---")
                        
                        # Display quick summary
                        st.markdown("### 📊 Quick Summary")
                        if st.session_state.results:
                            logger.debug("[STREAMLIT APP] Displaying results summary...")
                            metrics = st.session_state.results.get('metrics', {})
                            logger.debug(f"[STREAMLIT APP] Metrics: {metrics}")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(
                                    "Avg Risk Score",
                                    f"{metrics.get('avg_risk_score', 0):.1f}",
                                    delta=None
                                )
                            with col2:
                                st.metric(
                                    "Fall Events",
                                    metrics.get('total_fall_events', 0)
                                )
                            with col3:
                                st.metric(
                                    "Processing Time",
                                    f"{metrics.get('avg_processing_time', 0):.2f}s/frame"
                                )
                            with col4:
                                st.metric(
                                    "Impact Events",
                                    metrics.get('total_impact_events', 0)
                                )
                        else:
                            logger.warning("[STREAMLIT APP] No results found in session state")
                        
                        st.info("👉 Go to the **Results Dashboard** tab to view detailed analysis")

                        # Intermediate Outputs (visual debug)
                        st.markdown("---")
                        st.markdown("### 🧭 Intermediate Outputs & Pipeline Analysis")

                        # Load metrics and summary (if present)
                        metrics = load_metrics(st.session_state.output_dir)
                        summary = load_pipeline_summary(st.session_state.output_dir)

                        # ============================================================
                        # SECTION 1: Court Line Detection Pipeline
                        # ============================================================
                        st.markdown("#### 🎬 Court Line Detection Pipeline")
                        try:
                            video_path = Path(st.session_state.output_dir) / "final_output.mp4"
                            if video_path.exists():
                                cap = cv2.VideoCapture(str(video_path))
                                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                                
                                # Select a frame within first 20 frames
                                if total > 0:
                                    fidx = min(15, total - 1)  # Frame 15 or last frame if video is shorter
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
                                    ret, frame = cap.read()
                                    
                                    if ret:
                                        # Step 1: Convert to grayscale
                                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                        
                                        # Step 2: Apply Gaussian Blur
                                        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
                                        
                                        # Step 3: Canny Edge Detection
                                        edges = cv2.Canny(blurred, 50, 150)
                                        
                                        # Step 4: Hough Line Transform
                                        lines_img = edges.copy()
                                        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
                                        hough_lines_data = []
                                        if lines is not None:
                                            lines_img_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                                            for line in lines:
                                                x1, y1, x2, y2 = line[0]
                                                cv2.line(lines_img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                                hough_lines_data.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})
                                            lines_img = cv2.cvtColor(lines_img_color, cv2.COLOR_BGR2RGB)
                                        else:
                                            lines_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                                        
                                        # Save intermediate outputs to files
                                        intermediate_dir = Path(st.session_state.output_dir) / "intermediate_outputs"
                                        intermediate_dir.mkdir(exist_ok=True)
                                        
                                        # Save images
                                        cv2.imwrite(str(intermediate_dir / "1_grayscale.png"), gray)
                                        cv2.imwrite(str(intermediate_dir / "2_blurred.png"), blurred)
                                        cv2.imwrite(str(intermediate_dir / "3_canny_edges.png"), edges)
                                        cv2.imwrite(str(intermediate_dir / "4_hough_lines.png"), cv2.cvtColor(lines_img, cv2.COLOR_RGB2BGR))
                                        
                                        # Save Hough lines as JSON
                                        hough_json = {
                                            "frame_index": int(fidx),
                                            "total_lines_detected": len(hough_lines_data),
                                            "lines": hough_lines_data,
                                            "hough_parameters": {
                                                "minLineLength": 50,
                                                "maxLineGap": 10,
                                                "threshold": 50
                                            }
                                        }
                                        with open(intermediate_dir / "hough_lines.json", 'w') as f:
                                            json.dump(hough_json, f, indent=2)
                                        
                                        # Save edge detection matrix
                                        edges_matrix = {
                                            "shape": list(edges.shape),
                                            "dtype": str(edges.dtype),
                                            "canny_threshold_low": 50,
                                            "canny_threshold_high": 150,
                                            "edges_detected_count": int(np.count_nonzero(edges))
                                        }
                                        with open(intermediate_dir / "edge_detection_info.json", 'w') as f:
                                            json.dump(edges_matrix, f, indent=2)
                                        
                                        # Save court line detection summary
                                        summary_data = {
                                            "pipeline_stage": "Court Line Detection",
                                            "frame_processed": int(fidx),
                                            "processing_steps": [
                                                {"step": 1, "name": "Grayscale Conversion", "output": "1_grayscale.png"},
                                                {"step": 2, "name": "Gaussian Blur (5x5, sigma=1.5)", "output": "2_blurred.png"},
                                                {"step": 3, "name": "Canny Edge Detection (50-150)", "output": "3_canny_edges.png", "edges_count": int(np.count_nonzero(edges))},
                                                {"step": 4, "name": "Hough Line Transform", "output": "4_hough_lines.png", "lines_detected": len(hough_lines_data)}
                                            ]
                                        }
                                        with open(intermediate_dir / "court_line_summary.json", 'w') as f:
                                            json.dump(summary_data, f, indent=2)
                                        
                                        # Display all 4 stages in 2x2 grid
                                        cols = st.columns(2)
                                        
                                        # Step 1: Grayscale
                                        with cols[0]:
                                            _, png = cv2.imencode('.png', gray)
                                            st.image(png.tobytes(), caption="1️⃣ Grayscale", use_column_width=True)
                                        
                                        # Step 2: Gaussian Blur
                                        with cols[1]:
                                            _, png = cv2.imencode('.png', blurred)
                                            st.image(png.tobytes(), caption="2️⃣ Gaussian Blur", use_column_width=True)
                                        
                                        # Step 3: Canny Edge Detection
                                        with cols[0]:
                                            _, png = cv2.imencode('.png', edges)
                                            st.image(png.tobytes(), caption=f"3️⃣ Canny Edge Detection ({np.count_nonzero(edges)} edges)", use_column_width=True)
                                        
                                        # Step 4: Hough Line Transform
                                        with cols[1]:
                                            _, png = cv2.imencode('.png', lines_img)
                                            st.image(png.tobytes(), caption=f"4️⃣ Hough Lines ({len(hough_lines_data)} detected)", use_column_width=True)
                                    
                                    cap.release()
                            else:
                                st.info("Output video not available for frame extraction")
                        except Exception as e:
                            st.warning(f"Could not process court line detection: {e}")

                        st.markdown("---")

                        # ============================================================
                        # SECTION 2: Raider Detected Frames (when raider is detected)
                        # ============================================================
                        st.markdown("#### 🎯 Raider Detection & Tracking")
                        try:
                            video_path = Path(st.session_state.output_dir) / "final_output.mp4"
                            if metrics and "raider_detections" in metrics and video_path.exists():
                                raiders = metrics.get('raider_detections', [])
                                # Find ALL frames where raider is detected (not just first 30)
                                raider_frames = [i for i, r in enumerate(raiders) if r == 1]
                                
                                if raider_frames:
                                    # Show first 3-4 raider detected frames
                                    cap = cv2.VideoCapture(str(video_path))
                                    cols = st.columns(2)
                                    
                                    for idx, fidx in enumerate(raider_frames[:4]):  # First 4 raider frames
                                        col = cols[idx % 2]
                                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
                                        ret, frame = cap.read()
                                        if not ret:
                                            continue
                                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        with col:
                                            _, png = cv2.imencode('.png', rgb)
                                            st.image(png.tobytes(), caption=f"🏏 Raider detected at frame {fidx}", use_column_width=True)
                                    
                                    cap.release()
                                else:
                                    st.info('❌ No raider detected throughout the video')
                            else:
                                st.info('⚠️ Raider detection data not available')
                        except Exception as e:
                            st.warning(f"Could not load raider frames: {e}")

                        st.markdown("---")

                        # Joint Skeleton Trajectory: show raider skeleton for fall frames in 2-column layout
                        st.markdown("#### Raider Joint Skeleton Trajectory")
                        try:
                            fall_frames = []
                            if summary and 'falls' in summary:
                                fall_frames = summary['falls'].get('fall_frames', [])[:4]
                            
                            if fall_frames:
                                video_path = Path(st.session_state.output_dir) / "final_output.mp4"
                                if video_path.exists():
                                    cap = cv2.VideoCapture(str(video_path))
                                    skeleton_frames = fall_frames[:4]  # Max 4 frames
                                    cols = st.columns(2)  # 2-column layout
                                    
                                    for idx, fidx in enumerate(skeleton_frames):
                                        col = cols[idx % 2]
                                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
                                        ret, frame = cap.read()
                                        if not ret:
                                            continue
                                        
                                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        with col:
                                            _, png = cv2.imencode('.png', rgb)
                                            st.image(png.tobytes(), caption=f"Skeleton at frame {fidx}", use_column_width=True)
                                    
                                    cap.release()
                                else:
                                    st.info("Could not load video for joint visualization")
                            else:
                                st.info('No fall frames detected - skeleton trajectory not available')
                        except Exception as e:
                            st.warning(f"Could not load joint skeleton frames: {e}")

                        st.markdown("---")

                        # Injury Score & Result Summary
                        st.markdown("#### 🏥 Injury Assessment Score & Result")
                        try:
                            if summary:
                                # Calculate injury score from risk metrics
                                avg_risk = summary.get('risk', {}).get('avg_risk', 0)
                                total_falls = summary.get('falls', {}).get('total_falls', 0)
                                total_impacts = summary.get('impacts', {}).get('total_impacts', 0)
                                high_risk_frames = summary.get('risk', {}).get('high_risk_frames', 0)
                                
                                # Injury score calculation: weighted combination
                                injury_score = (avg_risk * 0.4) + (total_falls * 10) + (total_impacts * 5) + (high_risk_frames * 0.1)
                                
                                # Determine injury result/severity
                                if injury_score < 20:
                                    injury_result = "🟢 LOW RISK - Minimal Injury Likelihood"
                                    result_color = "green"
                                elif injury_score < 50:
                                    injury_result = "🟡 MODERATE RISK - Caution Advised"
                                    result_color = "orange"
                                elif injury_score < 80:
                                    injury_result = "🟠 HIGH RISK - Injury Probable"
                                    result_color = "orange"
                                else:
                                    injury_result = "🔴 CRITICAL RISK - Severe Injury Likely"
                                    result_color = "red"
                                
                                # Display injury metrics
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("💯 Injury Score", f"{injury_score:.1f}")
                                with col2:
                                    st.markdown(f"<div style='padding: 20px; border-radius: 10px; background-color: {result_color}20; border-left: 5px solid {result_color}'>{injury_result}</div>", unsafe_allow_html=True)
                            else:
                                st.info('⚠️ Injury assessment data not available')
                        except Exception as e:
                            st.warning(f"Could not load injury assessment: {e}")
                        
                    except Exception as e:
                        logger.error(f"[STREAMLIT APP - ERROR] {str(e)}", exc_info=True)
                        st.error(f"❌ Error during processing: {str(e)}")
                        st.exception(e)
    
    with tab2:
        st.markdown("### 📊 Analysis Results")
        logger.debug(f"[STREAMLIT APP] Results tab loaded. Processing complete: {st.session_state.processing_complete}")
        
        if not st.session_state.processing_complete:
            st.info("👈 Please upload and process a video first")
            logger.debug("[STREAMLIT APP] Waiting for video processing")
        else:
            logger.debug("[STREAMLIT APP] Processing complete, loading results...")
            output_dir = st.session_state.output_dir
            logger.debug(f"[STREAMLIT APP] Loading data from: {output_dir}")
            
            # Load data
            status = load_pipeline_status(output_dir)
            summary = load_pipeline_summary(output_dir)
            metrics = load_metrics(output_dir)
            
            logger.debug(f"[STREAMLIT APP] Status loaded: {status is not None}")
            logger.debug(f"[STREAMLIT APP] Summary loaded: {summary is not None}")
            logger.debug(f"[STREAMLIT APP] Metrics loaded: {metrics is not None}")
            
            # Display stage status
            display_stage_status(status)
            
            st.markdown("---")
            
            # Key Metrics
            if summary:
                logger.debug("[STREAMLIT APP] Displaying key metrics from summary")
                st.markdown("### 📈 Key Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_risk = summary.get('risk', {}).get('avg_risk', 0)
                    risk_color = "🟢" if avg_risk < 30 else "🟠" if avg_risk < 70 else "🔴"
                    st.metric(
                        f"{risk_color} Average Risk",
                        f"{avg_risk:.1f}%"
                    )
                
                with col2:
                    total_falls = summary.get('falls', {}).get('total_falls', 0)
                    st.metric("🤸 Total Falls", total_falls)
                
                with col3:
                    total_impacts = summary.get('impacts', {}).get('total_impacts', 0)
                    st.metric("💥 Total Impacts", total_impacts)
                
                with col4:
                    high_risk_frames = summary.get('risk', {}).get('high_risk_frames', 0)
                    st.metric("⚠️ High Risk Frames", high_risk_frames)
                
                st.markdown("---")
                
                # Risk Timeline Chart
                st.markdown("### 📉 Risk Score Timeline")
                risk_chart = create_risk_timeline_chart(summary)
                if risk_chart:
                    st.plotly_chart(risk_chart, use_column_width=True)
                    logger.debug("[STREAMLIT APP] Risk timeline chart displayed")
                else:
                    st.warning("No timeline data available")
                    logger.warning("[STREAMLIT APP] Risk timeline chart not available")
                
                # Two column layout for additional charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🥧 Risk Component Breakdown")
                    component_chart = create_component_breakdown_chart(summary)
                    if component_chart:
                        st.plotly_chart(component_chart, use_column_width=True)
                        logger.debug("[STREAMLIT APP] Component breakdown chart displayed")
                
                with col2:
                    st.markdown("### ⚡ Event Timeline")
                    event_chart = create_event_timeline(summary)
                    if event_chart:
                        st.plotly_chart(event_chart, use_column_width=True)
                    else:
                        st.info("No events detected")
                
                st.markdown("---")
                
                # Detailed Statistics
                with st.expander("📊 Detailed Statistics", expanded=False):
                    st.json(summary)


if __name__ == "__main__":
    main()
