from pathlib import Path
from PIL import Image
import streamlit as st
import settings
import helper
import cv2
import numpy as np

# 设置页面布局
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 主页面标题
st.title("Object Detection using YOLOv8")

# 侧边栏配置
st.sidebar.header("ML Model Config")

# 模型选项
model_type = st.sidebar.radio("Select Task", ["Detection"])
confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100

# 模型选择功能
MODEL_OPTIONS = {
    "YOLOv8n (nano)": "yolov8n.pt",
    "YOLOv8s (small)": "yolov8s.pt",
    "YOLOv8m (medium)": "yolov8m.pt",
    "YOLOv8l (large)": "yolov8l.pt",
    "YOLOv8x (extra large)": "yolov8x.pt"
}

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_OPTIONS.keys()),
    index=0
)

model_filename = MODEL_OPTIONS[selected_model_name]
model_path = Path(settings.MODEL_DIR) / model_filename

# 加载预训练模型
try:
    if model_type == 'Detection':
        model = helper.load_model(model_path)
        st.sidebar.success(f"Loaded model: {selected_model_name}")
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
    st.stop()

# 数据源配置
st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

# 图像检测功能
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", 
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            if source_img is None:
                default_image = Image.open(settings.DEFAULT_IMAGE)
                st.image(default_image, caption="Default Image", use_container_width=True)
            else:
                uploaded_image = Image.open(source_img)
                st.image(source_img, caption="Uploaded Image", use_container_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is not None and st.sidebar.button('Detect Objects'):
            try:
                # 将PIL图像转换为numpy数组
                uploaded_image_np = np.array(uploaded_image)
                
                # 使用模型进行预测
                res = model.predict(uploaded_image_np, conf=confidence)
                res_plotted = res[0].plot()[:, :, ::-1]  # BGR转RGB
                
                # 显示检测结果
                st.image(res_plotted, caption='Detected Image', use_container_width=True)
                
                # 显示检测结果详情
                with st.expander("Detection Results"):
                    for box in res[0].boxes:
                        st.write(f"Class: {model.names[int(box.cls)]}, Confidence: {box.conf.item():.2f}")
                        st.write(f"Coordinates: {box.xyxy.tolist()}")
            except Exception as ex:
                st.error("Error during detection.")
                st.error(ex)

# 视频检测功能
elif source_radio == settings.VIDEO:
    source_video = st.sidebar.selectbox(
        "Choose a video...", 
        list(settings.VIDEOS_DICT.keys())
    )
    
    # 执行视频目标检测
    if st.sidebar.button('Detect Video Objects'):
        try:
            video_path = settings.VIDEOS_DICT[source_video]
            helper.play_stored_video(video_path, confidence, model)
        except Exception as ex:
            st.error(f"Error loading video: {str(ex)}")

# 无效数据源处理
else:
    st.error("Please select a valid source type!")

# 显示当前使用的模型信息
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.write(f"**Selected Model:** {selected_model_name}")
st.sidebar.write(f"**Model File:** {model_filename}")
st.sidebar.write(f"**Confidence Threshold:** {confidence:.2f}")