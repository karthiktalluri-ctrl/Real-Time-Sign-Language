import streamlit as st
import cv2
import numpy as np
import os
import time
from core import SignLanguageConfig, FeatureExtractor, DataCollector, SignModel

# Page Config
st.set_page_config(page_title="Sign Language Detector", layout="wide")

# Initialize Config & Core Classes
if 'config' not in st.session_state:
    st.session_state.config = SignLanguageConfig()
    st.session_state.extractor = FeatureExtractor()
    st.session_state.collector = DataCollector(st.session_state.config)
    st.session_state.model_handler = SignModel(st.session_state.config)

# Sidebar
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Home", "Data Collection", "Train Model", "Real-time Inference"])

# Global Styles
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def home_page():
    st.title("Sign Language Detection System")
    st.markdown("""
    Welcome to the **Sign Language Detection System**. This application uses LSTM (Long Short-Term Memory) 
    neural networks to detect sign language gestures in real-time.
    
    ### How to use:
    1. **Data Collection**: Record your own sign language gestures.
    2. **Train Model**: Train the LSTM model on your collected data.
    3. **Inference**: Test the model in real-time.
    """)
    
    st.info("Navigate using the sidebar to start.")

def data_collection_page():
    st.title("Data Collection")
    
    # Input for actions
    actions_input = st.text_input("Enter actions to collect (comma separated)", "hello,thanks")
    actions = [a.strip() for a in actions_input.split(",")]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(f"**Config:** {st.session_state.config.sequences_per_sign} sequences x {st.session_state.config.sequence_length} frames")
        start_btn = st.button("Start Collection")
        stop_btn = st.button("Stop")
        
        frame_placeholder = st.empty()
        status_text = st.empty()
        
    if start_btn:
        st.session_state.collector.setup_directories(actions)
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened
        if not cap.isOpened():
            st.error("Could not open webcam.")
            return

        with st.session_state.extractor.get_model() as holistic:
            for action in actions:
                for sequence in range(st.session_state.config.sequences_per_sign):
                    for frame_num in range(st.session_state.config.sequence_length):
                        
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Camera read failed.")
                            break
                            
                        image, results = st.session_state.extractor.process_frame(frame, holistic)
                        st.session_state.extractor.draw_styled_landmarks(image, results)
                        
                        # Wait logic for first frame
                        if frame_num == 0:
                            status_text.warning(f"Get Ready! Collecting: {action} - Sequence {sequence}")
                            cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            
                            # Show wait frame
                            frame_placeholder.image(image, channels="BGR")
                            cv2.waitKey(2000)
                        else:
                            cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,12), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            frame_placeholder.image(image, channels="BGR")
                        
                        # Save keypoints
                        keypoints = st.session_state.extractor.extract_keypoints(results)
                        st.session_state.collector.save_frame(action, sequence, frame_num, keypoints)
                        
                        # Allow stopping
                        # (Streamlit doesn't handle interrupts well in loops without rerun, but we can check a file or something. 
                        # For now, simple loop)
                        
        cap.release()
        status_text.success("Data Collection Complete!")

def training_page():
    st.title("Model Training")
    
    actions_input = st.text_input("Enter actions to train on (must match collection)", "hello,thanks")
    actions = [a.strip() for a in actions_input.split(",")]
    
    if st.button("Train Model"):
        with st.spinner("Loading data..."):
            try:
                X, y = st.session_state.collector.load_data(actions)
            except Exception as e:
                st.error(f"Error loading data: {e}. Did you collect data for these actions?")
                return

        with st.spinner("Building and Training Model... (This may take a while)"):
            model = st.session_state.model_handler.build(len(actions))
            
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Custom callback to update streamlit
            # Since we can't easily pass callbacks to model.fit in this structure without modifying core,
            # we'll just run it. In a real app, we'd use a Keras callback.
            
            st.session_state.model_handler.train(X, y)
            st.session_state.model_handler.save()
            
        st.success("Training Complete! Model Saved.")

def inference_page():
    st.title("Real-time Inference")
    
    actions_input = st.text_input("Enter actions (must match training)", "hello,thanks")
    actions = [a.strip() for a in actions_input.split(",")]
    
    run_inference = st.checkbox("Start Camera")
    frame_placeholder = st.empty()
    probs_placeholder = st.empty()
    
    if run_inference:
        # Load model
        if not st.session_state.model_handler.load():
            st.error("Model not found. Please train the model first.")
            return

        cap = cv2.VideoCapture(0)
        sequence = []
        sentence = []
        threshold = 0.5
        
        with st.session_state.extractor.get_model() as holistic:
            while cap.isOpened() and run_inference:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                image, results = st.session_state.extractor.process_frame(frame, holistic)
                st.session_state.extractor.draw_styled_landmarks(image, results)
                
                # Prediction logic
                keypoints = st.session_state.extractor.extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                if len(sequence) == 30:
                    res = st.session_state.model_handler.predict(np.array(sequence))
                    
                    # Visualization
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])
                            
                    if len(sentence) > 5:
                        sentence = sentence[-5:]
                        
                    # Visualize probabilities
                    # (Simple text for now)
                    probs_text = " | ".join([f"{actions[i]}: {res[i]:.2f}" for i in range(len(actions))])
                    probs_placeholder.text(f"Probabilities: {probs_text}")
                    
                    cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(image, ' '.join(sentence), (3,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                frame_placeholder.image(image, channels="BGR")
                
                # Stop if checkbox unchecked (Streamlit re-runs script on interaction, so this loop might just break naturally)
                
        cap.release()

# Routing
if app_mode == "Home":
    home_page()
elif app_mode == "Data Collection":
    data_collection_page()
elif app_mode == "Train Model":
    training_page()
elif app_mode == "Real-time Inference":
    inference_page()
