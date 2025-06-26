import pandas as pd
import numpy as np
import streamlit as st


st.write("Here's our first attempt at using data to create a table")
st.write(pd.DataFrame({
 'first column': [1, 2, 3, 4],
 'second column': [10, 20, 30, 40]
 }))


dataframe = pd.DataFrame(
np.random.randn(10, 20),
 columns=[f'column_{i}' for i in range(20)]
 )
st.dataframe(dataframe.style.highlight_max(axis=0))

chart_data = pd.DataFrame(
 np.random.randn(20, 3),
 columns=['a', 'b', 'c']
 )
st.line_chart(chart_data)

map_data = pd.DataFrame(
 np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
 columns=['lat', 'lon']
 )
st.map(map_data)

x = st.slider('x', 0, 100, 50)  
st.write(x, 'squared is', x * x)

user_name = st.text_input("Your name", key="name")
st.write("Hello,", user_name)
 
df = pd.DataFrame({
 'first column': [1, 2, 3, 4],
 'second column': [10, 20, 30, 40]
 })
option = st.selectbox(
 'Which number do you like best?',
df['first column']
 )
st.write('You selected:', option)

with st.sidebar:
 st.header("Image/Video Config")
 source = st.selectbox(
 "Choose a source",
 ["Image", "Video"]
 )
 confidence = st.slider(
 "Confidence", 0.0, 1.0, 0.5
 )

with st.sidebar:
 st.title("YOLOv8 Config")
 model_size = st.radio(
 "Model Size",
 ["nano", "small", "medium"]
 )
 use_gpu = st.checkbox("Use GPU Acceleration")
 st.write(f"Selected: {model_size}, GPU: {use_gpu}")

col1, col2 = st.columns(2)
with col1:
 st.header("Original Image")
 st.image("input.jpg")
with col2:
 st.header("Detected Image")
 st.image("output.jpg")

left, right = st.columns([1, 2])  
with left:
 st.button("Generate Random Data")
with right:
 data = np.random.randn(10, 5)
 st.dataframe(data)

add_selectbox = st.sidebar.selectbox(
 'How would you like to be contacted?',  
('Email', 'Home phone', 'Mobile phone')  
)
add_slider = st.sidebar.slider(
 'Select a range of values',  
0.0, 100.0, 
(25.0, 75.0)              
)

left_column, right_column = st.columns(2)
left_column.button('Press me!')
with right_column:
 dog_breed = st.radio(
 'Choose Dog Breed:',
 options=['Husky', 'Corgi', 'Chihuahua', 'Spotty'],
 index=0
 )
 st.write(f"You selected: {dog_breed}")
