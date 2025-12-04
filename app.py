import streamlit as st
import joblib as jb
import numpy as np
import sys
import os

st.set_page_config(page_title="Hello Streamlit", page_icon=":wave:")
st.title("Welcome to Streamlit!")
# load a image of a compurter using image url
st.image("https://img.freepik.com/premium-psd/mobile-phone-ads-super-sale-discount-offer-instagram-post-design_534178-188.jpg", caption="Computer", use_container_width=True)
# use this cell phone model for predictions
model_path = os.path.join(os.path.dirname(__file__), 'linear_regression_model.pkl')
model = jb.load(model_path)
# create a Ui for user inputs where all parameter users will provide and depending on that the price will be predicted
st.header("Cell Phone Price Prediction")
# define input fields for user to enter the features
def user_input_features():
    # Collect all 13 features expected by the model (in order)
    col1, col2 = st.columns(2)
    
    with col1:
        product_id = st.number_input('Product ID', min_value=1, max_value=9999, value=100)
        sale = st.number_input('Sale (%)', min_value=0, max_value=100, value=10)
        weight = st.number_input('Weight (grams)', min_value=80.0, max_value=300.0, value=135.0)
        resolution = st.number_input('Resolution (ppi)', min_value=200, max_value=600, value=400)
        ppi = st.number_input('PPI', min_value=200.0, max_value=600.0, value=400.0)
        cpu_core = st.selectbox('CPU Core', [1, 2, 4, 6, 8])
        cpu_freq = st.number_input('CPU Frequency (GHz)', min_value=1.0, max_value=5.0, value=2.5)
    
    with col2:
        internal_mem = st.number_input('Internal Memory (GB)', min_value=4, max_value=512, value=64)
        ram = st.number_input('RAM (MB)', min_value=256, max_value=16384, value=2048)
        rear_cam = st.number_input('Rear Camera (MP)', min_value=0.5, max_value=108.0, value=13.0)
        front_cam = st.number_input('Front Camera (MP)', min_value=0.0, max_value=48.0, value=8.0)
        battery = st.number_input('Battery (mAh)', min_value=500, max_value=10000, value=2610)
        thickness = st.number_input('Thickness (mm)', min_value=5.0, max_value=20.0, value=7.4)
    
    # Create feature array in the exact order the model expects
    # Order: Product_id, Sale, weight, resoloution, ppi, cpu_core, cpu_freq, internal_mem, ram, RearCam, Front_Cam, battery, thickness
    features = np.array([[
        product_id, sale, weight, resolution, ppi, cpu_core, cpu_freq, 
        internal_mem, ram, rear_cam, front_cam, battery, thickness
    ]])
    return features
input_features = user_input_features()
# predict the price using the model
if st.button('Predict Price'):
    prediction = model.predict(input_features)
    st.success(f'The predicted price of the cell phone is: ${prediction[0]:.2f}')
