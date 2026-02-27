"""
Streamlit Web Dashboard for Kabaddi Injury Prediction System
Upload videos, process them, and view results in real-time.
"""

import streamlit as st
import sys
import os
import json
import time
import threading
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import cv2
import base64
from loguru import logger
from PIL import Image
import subprocess
import socket

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from main import KabaddiInjuryPipeline
from config.config import OUTPUT_DIR, RISK_FUSION_CONFIG, INJURY_HISTORY_PATH


# ==================== SERVER AUTO-START ====================
def start_server_if_not_running():
    """Start the server in background if not already running."""
    try:
        import requests
        # Check if server is already running
        response = requests.get("http://127.0.0.1:8000/health", timeout=2)
        if response.status_code == 200:
            return True, "ALREADY_RUNNING"
    except:
        pass
    
    # Server not running, start it
    try:
        import uvicorn
        from server import app as server_app
        
        def run_server():
            """Run the FastAPI server in background."""
            try:
                uvicorn.run(
                    server_app,
                    host="0.0.0.0",
                    port=8000,
                    log_level="info"
                )
            except Exception as e:
                logger.error(f"[STREAMLIT] Server error: {e}")
        
        # Start server in daemon thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to be ready
        import time
        for i in range(10):
            try:
                import requests
                response = requests.get("http://127.0.0.1:8000/health", timeout=2)
                if response.status_code == 200:
                    logger.info("[STREAMLIT] Server started successfully!")
                    return True, "STARTED"
            except:
                time.sleep(1)
        
        return False, "TIMEOUT"
    except Exception as e:
        logger.error(f"[STREAMLIT] Could not start server: {e}")
        return False, "ERROR"


# Initialize server on app load (ONLY ONCE)
if 'server_checked' not in st.session_state:
    st.session_state.server_checked = True
    st.session_state.server_started = False
    
    # Try to start server
    server_running, status = start_server_if_not_running()
    st.session_state.server_started = server_running
    
    if server_running:
        logger.info("[STREAMLIT] Server is running and ready for connections")
    else:
        logger.warning(f"[STREAMLIT] Server startup status: {status}")


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
    if 'processing_thread' not in st.session_state:
        st.session_state.processing_thread = None
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'live_events' not in st.session_state:
        st.session_state.live_events = None
    if 'raider_status' not in st.session_state:
        st.session_state.raider_status = None
    if 'session_events_archive' not in st.session_state:
        st.session_state.session_events_archive = []  # Store archived events for Results tab
    if 'uploaded_video_name' not in st.session_state:
        st.session_state.uploaded_video_name = None


def get_video_info(video_file):
    """Extract basic video information."""
    return {
        "name": video_file.name,
        "size": f"{video_file.size / (1024*1024):.2f} MB",
        "type": video_file.type
    }


def check_server_connection():
    """Check if server is running and return connection information."""
    import socket
    import requests
    
    try:
        # Check localhost
        response = requests.get("http://127.0.0.1:8000/health", timeout=2)
        localhost_running = response.status_code == 200
    except:
        localhost_running = False
    
    # Get network IP address
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        network_ip = s.getsockname()[0]
        s.close()
    except:
        network_ip = "127.0.0.1"
    
    return {
        "localhost_running": localhost_running,
        "network_ip": network_ip,
        "localhost_url": "http://127.0.0.1:8000",
        "network_url": f"http://{network_ip}:8000",
        "websocket_url": f"ws://{network_ip}:8000/ws",
        "api_docs": f"http://{network_ip}:8000/docs"
    }


def display_server_connection_info():
    """Display server connection status and URLs."""
    connection_info = check_server_connection()
    
    col1, col2, col3 = st.columns(3)
    
    if connection_info["localhost_running"]:
        status = "🟢 READY"
        with col1:
            st.metric("Server Status", status, delta="ACTIVE")
    else:
        status = "🔴 ERROR"
        with col1:
            st.metric("Server Status", status, delta="Check server")
    
    with col2:
        st.metric("Laptop IP", connection_info["network_ip"])
    
    with col3:
        st.metric("Server Port", "8000")
    
    # Display connection URLs
    st.markdown("### 📱 Connection Information for Expo App")
    
    connection_code = f"""
**Your Laptop Details:**
- Laptop IP Address: `{connection_info['network_ip']}`
- Server Port: `8000`
- Dashboard Port: `8501`

**For Expo App Connection:**
- API Base URL: `{connection_info['network_url']}`
- WebSocket URL: `{connection_info['websocket_url']}`

**For Browser:**
- Streamlit Dashboard: `http://localhost:8501`
- FastAPI Docs: `{connection_info['api_docs']}`
- Server Health Check: `{connection_info['network_url']}/health`
    """
    
    st.markdown(connection_code)
    
    # Status message
    st.markdown("---")
    if connection_info["localhost_running"]:
        st.success(f"""
        ✅ **Server is READY and WAITING FOR CONNECTIONS!**
        
        Your Expo app can now connect to: **`{connection_info['network_url']}`**
        
        **Next Steps:**
        1. Your Expo app connects to the server (using WebSocket)
        2. Upload a video in the "Upload & Process" tab below
        3. Server will stream events as video is processed:
           - 🏃 Raider identification
           - ⚠️ Collision detection
           - 💥 Fall detection
           - 📊 Injury risk updates
        """)
    else:
        st.error(f"""
        ❌ **Server is NOT RUNNING**
        
        Something went wrong starting the server. Please:
        1. Refresh the page (F5)
        2. If still not working, check the terminal logs
        3. You can manually start the server in terminal: `python server.py`
        """)


def load_live_events(output_dir):
    """Load live events from JSON file. Returns None if file doesn't exist."""
    try:
        live_events_file = Path(output_dir) / "live_events.json"
        if live_events_file.exists():
            with open(live_events_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.debug(f"[STREAMLIT] Could not load live events: {e}")
    return None


def display_event_log(events_data, max_messages=4):
    """Display event log with latest events. Max 4 recent messages."""
    if not events_data or 'events' not in events_data:
        st.info("Waiting for raider detection...")
        return
    
    events = events_data.get('events', [])
    if not events:
        st.info("Waiting for raider detection...")
        return
    
    # Show only latest max_messages events
    latest_events = events[-max_messages:]
    
    for event in latest_events:
        message = event.get('message', 'Unknown Event')
        time_str = event.get('time_str', '00:00')
        
        # Color-code by event type
        event_type = event.get('type', '')
        if 'FALL' in event_type:
            st.warning(f"**{time_str}** - {message}")
        elif 'TOUCH' in event_type:
            st.info(f"**{time_str}** - {message}")
        elif 'DETECTED' in event_type:
            st.success(f"**{time_str}** - {message}")
        else:
            st.write(f"**{time_str}** - {message}")


def display_raider_status_card(events_data):
    """Display raider status card with current information."""
    if not events_data:
        st.markdown("""
        <div style="background-color: #E3F2FD; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #1976D2;">Waiting for Raider Detection</h3>
            <p>Processing video... Raider will be detected once they cross the court line.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    raider_status = events_data.get('raider_status', {})
    
    if raider_status.get('id') is None:
        st.markdown("""
        <div style="background-color: #E3F2FD; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #1976D2;">Waiting for Raider Detection</h3>
            <p>Processing video... Raider will be detected once they cross the court line.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Raider has been detected - show full status
        raider_id = raider_status.get('id')
        confidence = raider_status.get('confidence', 0.0)
        state = raider_status.get('state', 'Unknown')
        touch_count = raider_status.get('touch_count', 0)
        detected_at = raider_status.get('detected_at')
        
        # Format the detection time
        if detected_at:
            minutes = int(detected_at) // 60
            seconds = int(detected_at) % 60
            time_str = f"{minutes:02d}:{seconds:02d}"
        else:
            time_str = "N/A"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Raider ID", f"#{raider_id}")
            st.metric("Detected At", time_str)
        
        with col2:
            st.metric("Confidence", f"{confidence:.1f}%")
            st.metric("Touches by Defenders", touch_count)
        
        # State indicator
        state_color = "#FFC107"  # Yellow for moving
        
        if "Fall" in str(state):
            state_color = "#F44336"  # Red for fallen
        elif "Touch" in str(state):
            state_color = "#FF9100"  # Orange for touched
        
        st.markdown(f"""
        <div style="background-color: {state_color}; color: white; padding: 15px; border-radius: 10px; text-align: center; font-size: 18px; font-weight: bold;">
            {state}
        </div>
        """, unsafe_allow_html=True)


def process_video_background(uploaded_file, output_dir, max_frames=None, save_intermediate=True):
    """
    Process video in background thread (NO STREAMLIT CALLS!).
    Only saves results to files - doesn't touch st.session_state.
    """
    try:
        logger.info(f"[BACKGROUND THREAD] Starting video processing: {uploaded_file.name}")
        
        # Save uploaded file
        input_path = output_dir / uploaded_file.name
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"[BACKGROUND THREAD] Video saved to: {input_path}")
        
        # Initialize and run pipeline
        pipeline = KabaddiInjuryPipeline(str(input_path), str(output_dir))
        results = pipeline.run(
            max_frames=max_frames,
            save_intermediate=save_intermediate
        )
        
        # Save results to JSON file for main thread to read
        results_file = output_dir / "processing_results.json"
        with open(results_file, 'w') as f:
            # Convert Path objects to strings for JSON serialization
            json_results = {
                "status": "completed",
                "results": {k: v for k, v in results.items() if k != "output_video"},
                "output_video": str(results.get("output_video", "")),
                "completed_at": datetime.now().isoformat()
            }
            json.dump(json_results, f, indent=2)
        
        # Create a marker file to signal completion
        completion_marker = output_dir / ".processing_complete"
        completion_marker.write_text("done")
        
        logger.info("[BACKGROUND THREAD] Video processing completed successfully")
        
    except Exception as e:
        logger.error(f"[BACKGROUND THREAD] Error during processing: {e}", exc_info=True)
        
        # Save error to file for main thread to read
        error_file = output_dir / "processing_error.json"
        with open(error_file, 'w') as f:
            json.dump({
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.now().isoformat()
            }, f, indent=2)
        
        # Create error marker
        error_marker = output_dir / ".processing_error"
        error_marker.write_text(str(e))


def display_live_processing_panel():
    """Display live processing panel with event updates and raider status."""
    
    with st.container():
        # Create columns for layout - SWAPPED: Status on left, Events on right
        col_status, col_events = st.columns([1, 2])
        
        with col_status:
            st.markdown("### Raider Status")
            status_placeholder = st.empty()
        
        with col_events:
            st.markdown("### Live Event Log")
            event_placeholder = st.empty()
        
        # Polling loop - wait for completion marker file
        output_dir = st.session_state.output_dir
        completion_marker = output_dir / ".processing_complete"
        error_marker = output_dir / ".processing_error"
        
        max_wait_time = 3600  # 1 hour max wait
        start_time = datetime.now()
        
        while True:
            # Check for completion marker
            if completion_marker.exists():
                st.session_state.is_processing = False
                st.session_state.processing_complete = True
                break
            
            # Check for error marker
            if error_marker.exists():
                st.session_state.is_processing = False
                st.session_state.processing_complete = False
                with event_placeholder.container():
                    st.error("Processing failed!")
                break
            
            # Check timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > max_wait_time:
                st.session_state.is_processing = False
                with event_placeholder.container():
                    st.error("Processing timeout (>1 hour)")
                break
            
            # Load live events from file
            live_events = load_live_events(output_dir)
            
            with event_placeholder.container():
                display_event_log(live_events, max_messages=4)
            
            with status_placeholder.container():
                display_raider_status_card(live_events)
            
            # Refresh every 500ms
            time.sleep(0.5)
        
        # Final update after processing
        live_events = load_live_events(output_dir)
        with event_placeholder.container():
            # Store events for Results tab
            if live_events or completion_marker.exists():
                final_events_file = output_dir / "events_timeline.json"
                if final_events_file.exists():
                    with open(final_events_file, 'r') as f:
                        final_data = json.load(f)
                        st.session_state.session_events_archive = final_data.get('timeline', [])
            
            if completion_marker.exists():
                st.success("Processing Complete!")
            else:
                st.warning("Processing ended unexpectedly")
        
        with status_placeholder.container():
            display_raider_status_card(live_events)


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
    
    st.markdown("### Pipeline Stage Status")
    
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
                    st.success(f"**{stage_name}**")
                elif stage_status == "Processing":
                    st.info(f"**{stage_name}**")
                elif stage_status == "Failed":
                    st.error(f"**{stage_name}**")
                else:
                    st.warning(f"**{stage_name}**")
                
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
    
    # Display server connection info
    st.markdown("---")
    display_server_connection_info()
    st.markdown("---")
    
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
            
            # Check if already processing another video
            if st.session_state.is_processing:
                st.error("**ANOTHER VIDEO IS BEING PROCESSED**. Please wait for it to complete before uploading a new video.")
                st.info("Current processing will continue in the background. Check back soon!")
                
                # Show live updates while processing
                display_live_processing_panel()
            
            elif not st.session_state.processing_complete:
                # Start new processing
                st.info("Starting video analysis... This may take several minutes.")
                st.write("Video will be processed in the background. Live updates will appear below:")
                
                # Initialize output directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = OUTPUT_DIR / f"session_{timestamp}"
                output_dir.mkdir(parents=True, exist_ok=True)
                st.session_state.output_dir = output_dir
                st.session_state.uploaded_video_name = uploaded_file.name
                
                # Start background processing thread
                st.session_state.is_processing = True
                st.session_state.processing_complete = False
                
                thread = threading.Thread(
                    target=process_video_background,
                    args=(uploaded_file, output_dir, None, True),
                    daemon=True
                )
                st.session_state.processing_thread = thread
                thread.start()
                
                logger.info(f"[STREAMLIT APP] Started background processing thread for {uploaded_file.name}")
                
                # Display live processing panel with polling
                display_live_processing_panel()
                
                # After processing completes, show results
                if st.session_state.processing_complete:
                    st.markdown("---")
                    st.success("**Processing Completed Successfully!**")
                    
                    # Display video playback
                    st.markdown("### Processed Video")
                    final_video = Path(st.session_state.output_dir) / "final_output.mp4"
                    if final_video.exists():
                        st.video(str(final_video))
                        
                        # Add download button
                        with open(final_video, "rb") as f:
                            st.download_button(
                                label="Download Annotated Video",
                                data=f.read(),
                                file_name="kabaddi_analysis.mp4",
                                mime="video/mp4"
                            )
                    
                    st.markdown("---")
                    
                    # Display Intermediate Outputs with Dropdowns
                    st.markdown("### Processing Results")
                    
                    session_dir = st.session_state.output_dir
                    
                    # Court Line Detection
                    court_line_dir = session_dir / "stage0_court_lines"
                    if court_line_dir.exists():
                        court_images = sorted(court_line_dir.glob("*.jpg"))
                        if court_images:
                            with st.expander("🏐 Court Line Detection"):
                                for img_path in court_images[:3]:
                                    st.image(Image.open(img_path), caption=img_path.name, use_column_width=True)
                    
                    # Player Detection
                    player_dir = session_dir / "stage1_detection"
                    if player_dir.exists():
                        player_images = sorted(player_dir.glob("*.jpg"))
                        if player_images:
                            with st.expander("👥 Player Detection"):
                                cols = st.columns(2)
                                for idx, img_path in enumerate(player_images[:10]):
                                    with cols[idx % 2]:
                                        st.image(Image.open(img_path), caption=img_path.name, use_column_width=True)
                    
                    # Raider Detection
                    raider_dir = session_dir / "stage2_raider"
                    if raider_dir.exists():
                        raider_images = sorted(raider_dir.glob("*.jpg"))
                        if raider_images:
                            with st.expander("Raider Detection"):
                                cols = st.columns(2)
                                for idx, img_path in enumerate(raider_images[:10]):
                                    with cols[idx % 2]:
                                        st.image(Image.open(img_path), caption=img_path.name, use_column_width=True)
                    
                    # Impact Detection
                    impact_dir = session_dir / "stage6_impact"
                    if impact_dir.exists():
                        impact_images = sorted(impact_dir.glob("*.jpg"))
                        if impact_images:
                            with st.expander("Impact Detection"):
                                cols = st.columns(2)
                                for idx, img_path in enumerate(impact_images[:10]):
                                    with cols[idx % 2]:
                                        st.image(Image.open(img_path), caption=img_path.name, use_column_width=True)
                    
                    # Fall Detection
                    fall_dir = session_dir / "stage4_falls"
                    if fall_dir.exists():
                        fall_images = sorted(fall_dir.glob("*.jpg"))
                        if fall_images:
                            with st.expander("Fall Detection"):
                                cols = st.columns(2)
                                for idx, img_path in enumerate(fall_images[:10]):
                                    with cols[idx % 2]:
                                        st.image(Image.open(img_path), caption=img_path.name, use_column_width=True)
                    
                    st.markdown("---")
                    
                    # Display Key Metrics and Scores
                    st.markdown("### Key Performance Metrics")
                    
                    output_dir = st.session_state.output_dir
                    summary = load_pipeline_summary(output_dir)
                    
                    if summary:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            avg_risk = summary.get('risk', {}).get('avg_risk', 0)
                            st.metric(
                                f"Average Risk",
                                f"{avg_risk:.1f}%"
                            )
                        
                        with col2:
                            total_falls = summary.get('falls', {}).get('total_falls', 0)
                            st.metric("Total Falls", total_falls)
                        
                        with col3:
                            total_impacts = summary.get('impacts', {}).get('total_impacts', 0)
                            st.metric("Total Impacts", total_impacts)
                        
                        with col4:
                            high_risk_frames = summary.get('risk', {}).get('high_risk_frames', 0)
                            st.metric("High Risk Frames", high_risk_frames)
                    
    with tab2:
        st.markdown("### Analysis Results")
        logger.debug(f"[STREAMLIT APP] Results tab loaded. Processing complete: {st.session_state.processing_complete}")
        
        if not st.session_state.processing_complete:
            st.info("Please upload and process a video first")
            logger.debug("[STREAMLIT APP] Waiting for video processing")
        else:
            # Safety check - ensure output_dir is valid
            if not st.session_state.output_dir or st.session_state.output_dir is None:
                st.error("Output directory not found. Please upload and process a video first.")
                logger.debug("[STREAMLIT APP] Processing complete, loading results...")
            else:
                output_dir = st.session_state.output_dir
                logger.debug(f"[STREAMLIT APP] Loading data from: {output_dir}")
                
                # Load data with error handling
                try:
                    status = load_pipeline_status(output_dir)
                    summary = load_pipeline_summary(output_dir)
                    metrics = load_metrics(output_dir)
                    
                    logger.debug(f"[STREAMLIT APP] Status loaded: {status is not None}")
                    logger.debug(f"[STREAMLIT APP] Summary loaded: {summary is not None}")
                    logger.debug(f"[STREAMLIT APP] Metrics loaded: {metrics is not None}")
                    
                    # Display stage status
                    display_stage_status(status)
                    
                    # Key Metrics
                    if summary:
                        logger.debug("[STREAMLIT APP] Displaying key metrics from summary")
                        st.markdown("### Key Performance Metrics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            avg_risk = summary.get('risk', {}).get('avg_risk', 0)
                            st.metric(
                                f"Average Risk",
                                f"{avg_risk:.1f}%"
                            )
                        
                        with col2:
                            total_falls = summary.get('falls', {}).get('total_falls', 0)
                            st.metric("Total Falls", total_falls)
                        
                        with col3:
                            total_impacts = summary.get('impacts', {}).get('total_impacts', 0)
                            st.metric("Total Impacts", total_impacts)
                        
                        with col4:
                            high_risk_frames = summary.get('risk', {}).get('high_risk_frames', 0)
                            st.metric("High Risk Frames", high_risk_frames)
                        
                        st.markdown("---")
                        
                        # Risk Timeline Chart
                        st.markdown("### Risk Score Timeline")
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
                        with st.expander("Detailed Statistics", expanded=False):
                            st.json(summary)
                except Exception as e:
                    logger.error(f"[STREAMLIT APP] Error loading results: {str(e)}")
                    st.error(f"Error loading results: {str(e)}")


if __name__ == "__main__":
    # Print connection info to CLI
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        network_ip = s.getsockname()[0]
        s.close()
    except:
        network_ip = "127.0.0.1"
    
    print("\n" + "="*90)
    print(" "*20 + "KABADDI INJURY PREDICTION SYSTEM")
    print("="*90)
    print("\n✅ STREAMLIT DASHBOARD STARTED!")
    print(f"   📍 Dashboard URL: http://localhost:8501")
    print()
    print("✅ SERVER STARTED AUTOMATICALLY!")
    print(f"   🌐 Server URL:     http://{network_ip}:8000")
    print(f"   🔌 WebSocket URL:  ws://{network_ip}:8000/ws")
    print()
    print("📱 YOUR EXPO APP CAN NOW CONNECT:")
    print(f"   💻 Laptop IP:      {network_ip}")
    print(f"   🔗 Connection URL: http://{network_ip}:8000")
    print()
    print("⚙️ WORKFLOW:")
    print("   1. Expo app connects to WebSocket: ws://{network_ip}:8000/ws")
    print("   2. Go to Dashboard: http://localhost:8501")
    print("   3. Upload Kabaddi video in 'Upload & Process' tab")
    print("   4. Main pipeline starts processing video")
    print("   5. Server broadcasts events to Expo app in real-time:")
    print("      - 🏃 Raider identification")
    print("      - ⚠️  Collision detection")
    print("      - 💥 Fall detection")
    print("      - 📊 Injury risk updates")
    print()
    print("📚 API DOCUMENTATION:")
    print(f"   📖 FastAPI Docs: http://{network_ip}:8000/docs")
    print(f"   💚 Health Check:  http://{network_ip}:8000/health")
    print()
    print("="*90)
    print()
    
    main()
