import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# title
st.title("Transport classification")

# uploading a file
file = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
if file:
    st.image(file)

    # PIL convert
    img = PILImage.create(file)

    # model
    model = load_learner('transports_model.pkl')

    #prediction
    pred, pred_id, prob = model.predict(img)
    st.success(f'Prediction: {pred}')
    st.info(f'Probability: {prob[pred_id]*100: .1f}%')

    # plot
    fig = px.bar(x=prob*100, y=model.dls.vocab)
    st.plotly_chart(fig)
