import os
import streamlit as st
import numpy as np
from PIL import  Image
import warnings
warnings.filterwarnings("ignore")

# Custom imports 
from multipage import MultiPage
from pages import Bigbull_News_Sentiments, home, Time_Series_Forecasting, Algorithmic_Trading, aboutus

# Create an instance of the app 
app = MultiPage()
# st.title("Stock Market Analysis")
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
padding = 0.5
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

# Title of the main page
# display = Image.open('Images/Background_2.png')
# display = np.array(display)
# st.image(display)
title = '<p style="background-color:#ED7D31; color:white;font-weight:bold; font-size: 28px; height:60px; width:100%; padding-top:8px; text-align:center;">Algorithmic Trading & Analyzing impact of News Sentiments on Stock Markets</p>'
# s1,a1,s2 = st.columns([4,2,4])
st.markdown(title, unsafe_allow_html=True)
st.write(" ")

# st.title("Data Storyteller Application")
# col1, col2 = st.columns(2)
# col1.image(display, width = 200)
# col2.header("Algorithmic Trading & Analyzing Impact of News Sentiment on Stock Markets")

# Add all your application here
app.add_page("Home", home.app)
app.add_page("News Sentiments", Bigbull_News_Sentiments.app)
app.add_page("Time Series Forecasting", Time_Series_Forecasting.app)
app.add_page("Algorithmic Trading", Algorithmic_Trading.app)
app.add_page("About Us", aboutus.app)

# The main app
app.run()