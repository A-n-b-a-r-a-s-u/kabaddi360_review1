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
    st.markdown('<div class="main-header">🏏 Kabaddi Injury Prediction System</div>', 
                unsafe_allow_html=True)
    st.markdown("**Done by Anbarasu with the guidance of Dr. T. Mala Mam (Professor)**")
    st.markdown("---")
    
    # Processing parameters
    # Always enable intermediate steps
    max_frames = None
    save_intermediate = True
    
    st.markdown("---")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Results Dashboard", "📁 Downloads"])
    
    with tab1:
        st.markdown("### 📤 Upload Kabaddi Match Video")
        
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
            
            # Process button
            if st.button("🚀 Start Analysis", type="primary"):
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
                        
                        # Display quick summary
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
                                    "Raider Detection",
                                    f"{metrics.get('raider_detection_rate', 0)*100:.1f}%"
                                )
                            with col4:
                                st.metric(
                                    "Processing Time",
                                    f"{metrics.get('avg_processing_time', 0):.2f}s/frame"
                                )
                        else:
                            logger.warning("[STREAMLIT APP] No results found in session state")
                        
                        st.info("👉 Go to the **Results Dashboard** tab to view detailed analysis")
                        
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
    
    with tab3:
        st.markdown("### 📁 Download Results")
        
        if not st.session_state.processing_complete:
            st.info("👈 Please upload and process a video first")
        else:
            output_dir = st.session_state.output_dir
            
            st.markdown("#### 🎬 Output Video")
            final_video = Path(output_dir) / "final_output.mp4"
            if download_button(final_video, "⬇️ Download Annotated Video", "kabaddi_analysis.mp4"):
                st.success("Video ready for download!")
            
            st.markdown("---")
            st.markdown("#### 📄 Analysis Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                summary_file = Path(output_dir) / "pipeline_summary.json"
                download_button(summary_file, "📊 Download Summary (JSON)", "summary.json")
                
                metrics_file = Path(output_dir) / "metrics.json"
                download_button(metrics_file, "📈 Download Metrics (JSON)", "metrics.json")
            
            with col2:
                status_file = Path(output_dir) / "pipeline_status.json"
                download_button(status_file, "✅ Download Status (JSON)", "status.json")
                
                log_file = Path(output_dir) / "pipeline.log"
                download_button(log_file, "📝 Download Logs", "pipeline.log")
            
            st.markdown("---")
            st.markdown("#### 🖼️ Pipeline Status Cards")
            
            status_cards_dir = Path(output_dir) / "status_cards"
            if status_cards_dir.exists():
                card_files = sorted(list(status_cards_dir.glob("*.png")))
                if card_files:
                    try:
                        st.write("All pipeline stages and their current status:")
                        
                        cols = st.columns(3)
                        for idx, card_file in enumerate(card_files):
                            col_idx = idx % 3
                            with cols[col_idx]:
                                try:
                                    from PIL import Image
                                    img = Image.open(card_file)
                                    st.image(img, use_column_width=True, caption=card_file.stem)
                                except Exception as e:
                                    st.warning(f"Could not display: {card_file.name}")
                    except Exception as e:
                        st.error(f"Error displaying status cards: {str(e)}")
                else:
                    st.info("Status cards will appear after processing completes.")
            else:
                st.info("Status cards folder not found yet.")
            
            st.markdown("---")
            st.info(f"💡 All outputs: `{output_dir}`")


if __name__ == "__main__":
    main()
