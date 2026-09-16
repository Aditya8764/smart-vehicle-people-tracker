import streamlit as st
import cv2
import tempfile
import pandas as pd
import time
from datetime import datetime
import supervision as sv
from detector import Detector
from tracker import Tracker
from counter import Counter
from utils import Annotator
import os

st.set_page_config(page_title="Real-Time Object Counter", layout="wide")
st.title("Real-Time Object Counter & Tracker")

# Session state initialization
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'classes_names' not in st.session_state:
    st.session_state.classes_names = {}
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'crossing_log' not in st.session_state:
    st.session_state.crossing_log = []

# Sidebar Controls
st.sidebar.header("Settings")
input_source = st.sidebar.radio("Input Source", ["Upload Video", "Webcam"])

video_file = None
if input_source == "Upload Video":
    video_file = st.sidebar.file_uploader("Upload a video (mp4/mov)", type=["mp4", "mov"])

# Load model button to avoid loading it multiple times
if st.session_state.detector is None:
    with st.spinner("Loading YOLOv8 model..."):
        st.session_state.detector = Detector("yolov8n.pt")
        st.session_state.classes_names = st.session_state.detector.model.names
    st.sidebar.success("Model loaded!")

# Class Selection
available_classes = st.session_state.classes_names
# Default to person (0) and car (2) if they exist
default_classes = []
for k, v in available_classes.items():
    if v in ["person", "car"]:
        default_classes.append(v)

selected_class_names = st.sidebar.multiselect(
    "Select classes to track", 
    options=list(available_classes.values()), 
    default=default_classes
)

# Convert selected names back to IDs
name_to_id = {v: k for k, v in available_classes.items()}
selected_class_ids = [name_to_id[name] for name in selected_class_names]

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.35, 0.05)

# Line Placement (using percentage of frame width/height to be resolution independent)
st.sidebar.subheader("Line Placement")
line_orientation = st.sidebar.radio("Orientation", ["Horizontal", "Vertical"])
line_position = st.sidebar.slider("Position (%)", 10, 90, 50)

# Controls
col1, col2, col3 = st.sidebar.columns(3)
if col1.button("Start"):
    st.session_state.is_running = True
if col2.button("Stop"):
    st.session_state.is_running = False
if col3.button("Reset"):
    st.session_state.crossing_log = []

# Display Area
main_video_placeholder = st.empty()
counts_placeholder = st.empty()

# Processing Loop
if st.session_state.is_running:
    if input_source == "Upload Video" and video_file is None:
        st.warning("Please upload a video file first.")
        st.session_state.is_running = False
    else:
        # Open video source
        if input_source == "Upload Video":
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            cap = cv2.VideoCapture(0) # Webcam

        if not cap.isOpened():
            st.error("Error opening video source.")
            st.session_state.is_running = False
        else:
            # Initialize tracker and annotator per run
            tracker = Tracker()
            annotator = Annotator()
            
            # Read first frame to get dimensions
            ret, frame = cap.read()
            if ret:
                h, w, _ = frame.shape
                
                # Setup line based on UI
                if line_orientation == "Horizontal":
                    y_pos = int(h * (line_position / 100))
                    start_pt = (0, y_pos)
                    end_pt = (w, y_pos)
                else:
                    x_pos = int(w * (line_position / 100))
                    start_pt = (x_pos, 0)
                    end_pt = (x_pos, h)
                
                counter = Counter(start_pt, end_pt)
                
                # We need to track previous counts to log events
                prev_in_count = 0
                prev_out_count = 0
                
                # Rewind video to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # Stop button placeholder inside the loop to allow breaking
                stop_placeholder = st.empty()
                if stop_placeholder.button("Stop Video"):
                    st.session_state.is_running = False
                
                while cap.isOpened() and st.session_state.is_running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # 1. Detect
                    result = st.session_state.detector.detect(frame, classes=selected_class_ids, conf=conf_threshold)
                    detections = sv.Detections.from_ultralytics(result)
                    
                    # 2. Track
                    detections = tracker.update(detections)
                    
                    # 3. Count
                    in_count, out_count = counter.trigger(detections)
                    
                    # Log events if count increased
                    if in_count > prev_in_count:
                        st.session_state.crossing_log.append({"timestamp": datetime.now(), "direction": "in"})
                        prev_in_count = in_count
                    if out_count > prev_out_count:
                        st.session_state.crossing_log.append({"timestamp": datetime.now(), "direction": "out"})
                        prev_out_count = out_count
                    
                    # 4. Annotate
                    annotated_frame = annotator.annotate_frame(frame, detections, counter.line_zone, st.session_state.classes_names)
                    
                    # 5. Display
                    # Convert BGR to RGB for Streamlit
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    main_video_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                    
                    # Update counts display
                    counts_placeholder.markdown(f"### 📈 Counts - In: {in_count} | Out: {out_count}")
                    
                    # Small delay to prevent Streamlit from choking on high FPS
                    time.sleep(0.01)
                    
            cap.release()
            if input_source == "Upload Video" and 'tfile' in locals():
                try:
                    os.unlink(tfile.name)
                except:
                    pass
            st.session_state.is_running = False

# Export Log
if len(st.session_state.crossing_log) > 0:
    st.subheader("Data Logging & Export")
    df = pd.DataFrame(st.session_state.crossing_log)
    st.dataframe(df.tail(5)) # Show last 5 entries
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download log as CSV",
        data=csv,
        file_name='crossing_log.csv',
        mime='text/csv',
    )
