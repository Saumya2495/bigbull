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
warnings.filterwarnings("ignore")

def app():
    col1, col2, col3, col4, col5 = st.columns(5)    

    #Fetching the Stock Data
    def GetStockInfo(stock_name, stock_ticker, gif):
        file_ = open("Images/"+gif, "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        col1, col2, col3 = st.columns([2,1,2])

        with col1:
            st.write("")

        with col2:
            st.markdown(f'<img src="data:image/gif;base64,{data_url}" height=150 width=200 alt="cat gif">',unsafe_allow_html=True)

        with col3:
            st.write("")

        end = datetime.now()
        start = datetime(end.year - 5, end.month, end.day)
        stock_data = pdr.DataReader(stock_ticker, data_source="yahoo",start=start,end=end)
        st.subheader(stock_name)
        c1,c2,c3,c4,c5 = st.columns((1,1,1,1,1))
        c1.metric(label="High", value=round(stock_data.iloc[-1]["High"],3), delta=round(stock_data.iloc[-1]["High"] - stock_data.iloc[-2]["High"],3))
        c2.metric(label="Low", value=round(stock_data.iloc[-1]["Low"],3), delta=round(stock_data.iloc[-1]["Low"] - stock_data.iloc[-2]["Low"],3))
        c3.metric(label="Open Price", value=round(stock_data.iloc[-1]["Open"],3), delta=round(stock_data.iloc[-1]["Open"] - stock_data.iloc[-2]["Open"],3))
        c4.metric(label="Close Price", value=round(stock_data.iloc[-1]["Close"],3), delta=round(stock_data.iloc[-1]["Close"] - stock_data.iloc[-2]["Close"],3))        
        c5.metric(label="Volume", value=round(stock_data.iloc[-1]["Volume"],3))

        def GetClosePriceChart(stock_data):
            title ='<p style="color:#ED7D31; font-size: 18px; font-weight:bold;">Close Price for Stock</p>'
            fig = px.line(stock_data, x=stock_data.index, y="Close", width=1150, height=600)
            # fig.update_layout(paper_bgcolor="black")
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            st.markdown(title,unsafe_allow_html=True)
            st.plotly_chart(fig)
        
        def GetCandleStickChart(stock_data):
            candle_title ='<p style="color:#ED7D31; font-size: 18px; font-weight:bold;">Candlestick Chart for Stock</p>'
            figure=go.Figure(
                data=[
                    go.Candlestick(
                        x= stock_data.index,
                        low=stock_data["Low"],
                        high=stock_data["High"],
                        close=stock_data["Close"],
                        open=stock_data["Open"],
                        increasing_line_color="green",
                        decreasing_line_color="red"
                        )
                    ]
                )
            figure.update_layout(
                yaxis_title="Stock Price ($)",
                xaxis_title="Date",
                width=1150, height=600
            )
            figure.update_xaxes(showgrid=False)
            figure.update_yaxes(showgrid=False)
            st.markdown(candle_title,unsafe_allow_html=True)
            st.plotly_chart(figure)

        with st.expander('Close Price Analysis'):
            GetClosePriceChart(stock_data)
        with st.expander('Candlestick Depiction'):
            GetCandleStickChart(stock_data)
    flag = True
    if (col1.button("Amazon")):
        flag = False
        GetStockInfo("Amazon",'AMZN', "amazon_gif.gif")
    if (col2.button("Apple")):
        flag = False
        GetStockInfo("Apple",'AAPL', "apple_gif.gif")
    if (col3.button("Tesla")):
        flag = False
        GetStockInfo("Tesla",'TSLA', "tesla_gif.gif")
    if (col4.button("Ulta Beauty")):
        flag = False
        GetStockInfo("Ulta Beauty",'ULTA', "ulta_gif.gif")
    if (col5.button("CocaCola")):
        flag = False
        GetStockInfo("CocaCola",'KO', "cocacola_gif.gif")
    if flag:
        GetStockInfo("Amazon",'AMZN', "amazon_gif.gif")
        # st.write("Please click on a Stock to see the Info.")

    # about_us = '<p style="color:#ED7D31; font-size: 42px;">About Us</p>'
    # st.markdown(about_us, unsafe_allow_html=True)
    # st.write('Bigbull is a fintech consulting team which provide traders a one stop solution to empower their trading strategies, \
    # analyze the fluctuation in financial markets thereby maximizing their profits. \
    # The main aim of this project is to perform algorithmic trading using machine learning with a concept of time series forecasting. \
    # Our approach is similar to “Paper Trading” where we integrate the concept of stock market price prediction with algorithmic trading.\
    #  Moreover, we are evaluating the impact of news sentiments of stock market data using Natural language Processing.')
