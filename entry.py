from pathlib import Path
from PIL import Image
import streamlit as st
import settings
import helper

 # Setting page layout
st.set_page_config(
page_title="Object Detection using YOLOv8",
page_icon="�",
layout="wide",
initial_sidebar_state="expanded")