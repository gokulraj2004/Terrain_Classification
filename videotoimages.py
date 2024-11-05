import streamlit as st
import cv2
from PIL import Image
import numpy as np
import io

def extract_frames(video_path, interval=5):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = frame_count // fps
    frame_interval = interval * fps  # interval in frames

    for sec in range(0, total_seconds, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, sec * fps)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

# Streamlit App
st.title("Video Snapshot Extractor")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save uploaded video to a temporary location
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    interval = st.number_input("Snapshot Interval (seconds)", min_value=1, max_value=10, value=5)
    st.write(f"Extracting snapshots every {interval} seconds...")

    frames = extract_frames(video_path, interval=interval)
    
    st.write("Snapshots:")
    for i, frame in enumerate(frames):
        st.image(frame, caption=f"Snapshot at {i*interval} seconds", use_column_width=True)

    # Optional: allow users to download snapshots as a ZIP file
    if st.button("Download snapshots as ZIP"):
        import zipfile
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for i, frame in enumerate(frames):
                img_byte_arr = io.BytesIO()
                frame.save(img_byte_arr, format="PNG")
                zf.writestr(f"snapshot_{i*interval}.png", img_byte_arr.getvalue())
        
        st.download_button(
            label="Download ZIP",
            data=zip_buffer.getvalue(),
            file_name="snapshots.zip",
            mime="application/zip"
        )
