from ultralytics import YOLO
import cv2
import streamlit as st
import tempfile
import numpy as np

def load_model(model_path):
    model = YOLO(model_path)
    return model

# 视频处理函数
def play_stored_video(video_path, confidence, model):
    # 创建临时文件
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(video_path.read_bytes())
    
    # 打开视频文件
    cap = cv2.VideoCapture(tfile.name)
    
    st_frame = st.empty()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # 执行目标检测
        results = model.predict(frame, conf=confidence)
        annotated_frame = results[0].plot()
        
        # 转换BGR到RGB
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st_frame.image(annotated_frame, caption="Detected Video", channels="RGB", use_container_width=True)
    
    cap.release()
    tfile.close()