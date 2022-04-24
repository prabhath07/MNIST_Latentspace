import streamlit as st
import numpy as np 
from tensorflow.keras.models import load_model

st.set_page_config(layout="wide")

padding = 3
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        
    }} </style> """, unsafe_allow_html=True)
st.header('   Latent_space Interpretation')

model = load_model('./decoder_latent.h5')

col1, col2, col3,col4,col5 = st.columns([1,1,1,1,2])

with col1:
    l0 = st.slider('l0',1.6090515e-07, 0.9999213, ,value =0.9999213,step= 1.6090515e-07)    
    l1 = st.slider('l1',1.6090515e-07, 0.9999213,1.6090515e-07)
    l2 = st.slider('l2',1.6090515e-07, 0.9999213, 1.6090515e-07)
    l3 = st.slider('l3',1.6090515e-07, 0.9999213, 1.6090515e-07)
    l4 = st.slider('l4',1.6090515e-07, 0.9999213, 1.6090515e-07)
with col3:
    l5 = st.slider('l5',1.6090515e-07, 0.9999213, 1.6090515e-07)
    l6 = st.slider('l6',1.6090515e-07, 0.9999213, 1.6090515e-07)
    l7 = st.slider('l7',1.6090515e-07, 0.9999213, 1.6090515e-07)
    l8 = st.slider('l8',1.6090515e-07, 0.9999213, 1.6090515e-07)
    l9 = st.slider('l9',1.6090515e-07, 0.9999213, 1.6090515e-07)
    

arr = [l0,l1,l2,l3,l4,l5,l6,l7,l8,l9]
arr = np.array(arr)

pred =  model.predict(np.expand_dims(arr,axis=0))

with col5:
    st.image(pred[0],width = 224)

