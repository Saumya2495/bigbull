#Importing necessary libraries
import pandas as pd
import numpy as np
import re
import json
import warnings
import csv
import streamlit as st
import base64
import plotly.graph_objects as go
from PIL import  Image
from itertools import cycle
from datetime import datetime
import pandas_datareader as pdr
import plotly.express as px
import base64
import streamlit.components.v1 as components
warnings.filterwarnings("ignore")

def app():
    about_us = '<p style="color:#ED7D31; font-size: 42px;">About Us</p>'
    s1,a1,s2 = st.columns([4,2,4])
    a1.markdown(about_us, unsafe_allow_html=True)
    st.write("")
    text = '<p style="text-align:center;">Bigbull is a fintech consulting team which provide traders a one stop solution to empower their trading strategies, \
    analyze the fluctuation in financial markets thereby maximizing their profits. \
    The main aim of this project is to perform algorithmic trading using machine learning with a concept of time series forecasting. \
    Our approach is similar to “Paper Trading” where we integrate the concept of stock market price prediction with algorithmic trading.\
     Moreover, we are evaluating the impact of news sentiments of stock market data using Natural language Processing.<p>'

    s1,t1,s2 = st.columns([1,8,1])
    t1.markdown(text, unsafe_allow_html=True)

    with t1.expander('Univariate Time Series Forecasting'):
        st.write('We will be utilizing only the Close Price feature and Date Index for performing the forecasting using statsmodels, Recurrent Neural Network(Stacked LSTM) and FB Prophet models. After evaluation of all models, We have used Stacked LSTM as our finalized model.')

    with t1.expander('Multivariate Time Series Forecasting'):
        st.write('For this method, we will be using Independent features like Open, High, Low and Volume to forecast Dependent features such as Close Price using the Deep Learning Recurrent Neural Network models (Bidirectional LSTM).\n Moreover, we have integrated the Sentiment Polarity of News Data of Individual Stocks as an independent features to see how external factors affect the stock price predictions as well as forecasting.')
    
    with t1.expander('News Sentiment Analysis'):
        st.write('We have implemented Twitter Search API for fetching the tweets related to a particular stock or that company. We have fetched around 6000 tweets per day for a particular stock that contains the mentioned search keyword. Later, we applied VADER sentiment analysis on a sample dataset to analyze the sentiments using several data visualizations.')
    
    with t1.expander('Algorithmic Trading'):
        st.write('We implemented various trading strategies that uses our machine learning model predictions as input and outputs actual buy/sell orders. \
        Here, I used various market indicators such as EMA, MFI, MACD and Candlestick Chart to generate the Buy or Sell signals for paper trading purposes on historical data. \
        After evaluating the generated signals using trad- ing strategies on historical data, we can derive the most accurate strategy to implement my future scope of Low Frequency Semi Algorithmic Trading on the real market by building a Trade Bot based on the efficient trading strategies.\n \
        Later, we have calculated the Profit and Loss Percentage on the basis of Buy and Sell Signals Generated by Individual Trading Strategies on various stock market data')
    

    st.write("")

    our_team = '<p style="color:#ED7D31; font-size: 42px;">Our Team</p>'
    s1,a1,s2 = st.columns([4,2,4])
    a1.markdown(our_team, unsafe_allow_html=True)
    st.write("")

    s1,c1,c2,c3,c4,s2 = st.columns([1.5,2,2,2,2,1])

    box1 = '<h5>Vishal Kapadia</h5> \
    Email: vkapadia@usc.edu \
    <p><button class="button" style="background-color: #ED7D31;  border: none; color: white;  padding-left: 15px; padding-right:15px; padding-top:5px; padding-bottom:5px; text-align: center;  text-decoration: none;  display: inline-block;  font-size: 16px;  margin: 4px 2px;  cursor: pointer;  border-radius: 12px; font-weight:bold;">Contact</button></p>'

    file_ = open("Images/vishal.jpeg", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    c1.markdown(f'<img src="data:image/gif;base64,{data_url}" height=137 width=120 style = "border-radius: 50%;">',unsafe_allow_html=True)
    c1.markdown(box1, unsafe_allow_html=True)

    box2 = '<h5>Tirth Patel</h5> \
    Email: tirthsha@usc.edu \
    <p><button class="button" style="background-color: #ED7D31;  border: none; color: white;  padding-left: 15px; padding-right:15px; padding-top:5px; padding-bottom:5px; text-align: center;  text-decoration: none;  display: inline-block;  font-size: 16px;  margin: 4px 2px;  cursor: pointer;  border-radius: 12px; font-weight:bold;">Contact</button></p>'
    
    file_ = open("Images/tirth.jpeg", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    c2.markdown(f'<img src="data:image/gif;base64,{data_url}" height=137 width=150 style = "border-radius: 50%;">',unsafe_allow_html=True)
    c2.markdown(box2, unsafe_allow_html=True)

    box3 = '<h5>Saumya Shah</h5> \
    Email: saumyass@usc.edu \
    <p><button class="button" style="background-color: #ED7D31;  border: none; color: white;  padding-left: 15px; padding-right:15px; padding-top:5px; padding-bottom:5px; text-align: center;  text-decoration: none;  display: inline-block;  font-size: 16px;  margin: 4px 2px;  cursor: pointer;  border-radius: 12px; font-weight:bold;">Contact</button></p>'
    
    file_ = open("Images/saumya.jpg", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    c3.markdown(f'<img src="data:image/gif;base64,{data_url}" height=137 width=150 style = "border-radius: 50%;">',unsafe_allow_html=True)
    c3.markdown(box3, unsafe_allow_html=True)

    box4 = '<h5>Yash Naik</h5> \
    Email: ynaik@usc.edu \
    <p><button class="button" style="background-color: #ED7D31;  border: none; color: white;  padding-left: 15px; padding-right:15px; padding-top:5px; padding-bottom:5px; text-align: center;  text-decoration: none;  display: inline-block;  font-size: 16px;  margin: 4px 2px;  cursor: pointer;  border-radius: 12px; font-weight:bold;">Contact</button></p>'
    
    file_ = open("Images/yash.jpeg", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    c4.markdown(f'<img src="data:image/gif;base64,{data_url}" height=137 width=150 style = "border-radius: 50%;">',unsafe_allow_html=True)
    c4.markdown(box4, unsafe_allow_html=True)

