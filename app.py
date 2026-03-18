import streamlit as st
import cv2
from deepface import DeepFace
import pandas as pd
import time

# 1. Enhanced UI Configuration
st.set_page_config(page_title="Pro AI Emotion Analytics", layout="wide")

# Sidebar for Project Info (Makes it look like a professional tool)
st.sidebar.title("System Controls")
st.sidebar.info("Model: VGG-Face via DeepFace\nBackend: TensorFlow")
run = st.sidebar.checkbox('Activate System', value=True)
clear_history = st.sidebar.button("Clear Sentiment History")

st.title("🎭 Real-Time AI Emotion Analytics Dashboard")
st.markdown("---")

# Initialize Session State for the "History" feature
if 'history' not in st.session_state or clear_history:
    st.session_state['history'] = []

# Layout: 3 Columns (Video, Bar Chart, Sentiment Trend)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📡 Live AI Monitoring")
    FRAME_WINDOW = st.image([])

with col2:
    st.subheader("📊 Live Confidence")
    chart_placeholder = st.empty()
    st.subheader("📈 Sentiment Trend")
    trend_placeholder = st.empty()

# 2. Optimized Capture Logic
camera = cv2.VideoCapture(1) # Change to 1 if "Camera not found"
count = 0 

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Accessing Camera... (If stuck, check Privacy Settings)")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process every 5th frame for performance
    if count % 5 == 0:
        try:
            results = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
            emotions = results[0]['emotion']
            dominant = results[0]['dominant_emotion']
            
            # Save history for the Trend Line
            st.session_state['history'].append(emotions[dominant])
            if len(st.session_state['history']) > 20: # Keep only last 20 data points
                st.session_state['history'].pop(0)

            # Update Bar Chart
            df_bar = pd.DataFrame(list(emotions.items()), columns=['Emotion', 'Score'])
            chart_placeholder.bar_chart(df_bar.set_index('Emotion'))
            
            # Update Trend Line (The "Something Extra")
            trend_placeholder.line_chart(st.session_state['history'])

        except Exception as e:
            pass
    
    # Update video
    FRAME_WINDOW.image(rgb_frame)
    count += 1

camera.release()