import streamlit as st
import joblib as jb
import numpy as np
import sys
import os

st.set_page_config(page_title="Hello Streamlit", page_icon=":wave:")
st.title("Welcome to Streamlit!")
# load a image of a compurter using image url
st.image("https://png.pngtree.com/png-clipart/20250515/original/pngtree-modern-desktop-computer-png-image_20978094.png", caption="Computer", use_container_width=True)
# use this cell phone model for predictions
model_path = os.path.join(os.path.dirname(__file__), 'linear_regression_model.pkl')
model = jb.load(model_path)
# create a Ui for user inputs where all parameter users will provide and depending on that the price will be predicted
st.header("Cell Phone Price Prediction")
# define input fields for user to enter the features
def user_input_features():
    battery_power = st.number_input('Battery Power (mAh)', min_value=500, max_value=10000, value=2000)
    ram = st.number_input('RAM (MB)', min_value=256, max_value=16384, value=2048)
    px_height = st.number_input('Pixel Height', min_value=200, max_value=4000, value=800)
    px_width = st.number_input('Pixel Width', min_value=200, max_value=4000, value=600)
    mobile_wt = st.number_input('Mobile Weight (grams)', min_value=80, max_value=300, value=150)
    data = {
        'battery_power': battery_power,
        'ram': ram,
        'px_height': px_height,
        'px_width': px_width,
        'mobile_wt': mobile_wt
    }
    features = np.array([battery_power, ram, px_height, px_width, mobile_wt]).reshape(1, -1)
    return features
input_features = user_input_features()
# predict the price using the model
if st.button('Predict Price'):
    prediction = model.predict(input_features)
    st.success(f'The predicted price of the cell phone is: ${prediction[0]:.2f}')
