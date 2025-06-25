from ultralytics import YOLO
import pandas as pd
import numpy as np
import streamlit as st
import cv2
import os


# 侧边栏配置
with st.sidebar:
    st.header("图像/视频配置")
    
    # 模型选择
    model_type = st.selectbox(
        "选择模型类型",
        ["YOLOv8n", "YOLOv8s"],
        index=0
    )
    
    # 置信度滑块 
    confidence = float(st.slider(
        "选择模型置信度", 25, 100, 40)) / 100
    
    # 检测模式选择
    detection_mode = st.radio(
        "选择检测模式",
        ["图像检测", "视频检测"]
    )

# 模型路径映射
model_paths = {
    "YOLOv8n": "weights/yolov8n.pt",
    "YOLOv8s": "weights/yolov8s.pt",
    
}
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        st.sidebar.success("模型加载成功！")
        return model
    except Exception as ex:
        st.sidebar.error(f"无法加载模型: {str(ex)}")
        return None

# 加载选择的模型
model = load_model(model_paths[model_type])

# 图像检测功能 
if detection_mode == "图像检测":
    st.header("YOLOv8图像目标检测")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("上传图像")
        uploaded_image = st.file_uploader(
            "选择图像文件", 
            type=["jpg", "jpeg", "png", "bmp"]
        )
        
        if uploaded_image is not None:
            # 显示上传的图像
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(image, caption="原始图像", use_container_width=True)
    
    with col2:
        st.subheader("检测结果")
        
        if st.button("检测目标", key="detect_image"):
            if uploaded_image is None:
                st.warning("请先上传图像")
            elif model is None:
                st.error("模型未加载成功")
            else:
                # 进行目标检测
                results = model.predict(image, conf=confidence)
                
                # 处理检测结果
                if len(results) > 0:
                    res_plotted = results[0].plot()
                    st.image(res_plotted, caption="检测结果", use_container_width=True)
                    
                    # 显示检测框坐标 
                    with st.expander("检测结果详情"):
                        boxes = results[0].boxes
                        if len(boxes) > 0:
                            for i, box in enumerate(boxes):
                                st.write(f"目标 {i+1}: {box.xywh}")
                        else:
                            st.info("未检测到目标")
                else:
                    st.warning("未检测到目标")
# 视频检测功能 
elif detection_mode == "视频检测":
    st.header("YOLOv8视频目标检测")
    
    # 视频选择 
    video_files = [f for f in os.listdir("videos") if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        st.warning("视频目录中没有找到视频文件")
    else:
        selected_video = st.selectbox("选择视频文件", video_files)
        video_path = os.path.join("videos", selected_video)
        
        # 显示原始视频 
        st.subheader("原始视频")
        st.video(video_path)
        
        # 视频检测按钮
        if st.button("检测视频目标", key="detect_video"):
            if model is None:
                st.error("模型未加载成功")
            else:
                # 使用OpenCV处理视频
                st.subheader("视频处理中...")
                
                # 打开视频文件
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # 创建进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 创建用于显示结果的容器
                result_container = st.empty()
                
                # 逐帧处理视频 
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 进行目标检测
                    results = model.predict(frame, conf=confidence)
                    
                    # 绘制检测结果
                    if len(results) > 0:
                        frame = results[0].plot()
                    
                    # 显示处理后的帧
                    result_container.image(frame, channels="BGR", use_container_width=True)
                    
                    # 更新进度
                    frame_count += 1
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"处理中: {frame_count}/{total_frames} 帧 ({progress:.1%})")
                
                # 释放资源
                cap.release()
                
                # 显示处理完成
                progress_bar.empty()
                status_text.text("视频处理完成！")