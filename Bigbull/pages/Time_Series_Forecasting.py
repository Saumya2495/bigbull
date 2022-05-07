import os
import streamlit as st
import numpy as np
from PIL import  Image
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import datetime as dt
from datetime import datetime
import base64
warnings.filterwarnings("ignore")

from pages.univariate_forecast import univariate 
from pages.multivariate_forecast import multivariate 

def app():
    def getgif(gif):
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

    title = '<p style="color:#ED7D31; font-size: 42px;">Time Series Forecasting</p>'
    st.markdown(title, unsafe_allow_html=True)

    c1, col1, col2, col3, c2 = st.columns([1,2,2,2,1])
    company = col1.selectbox('Select the company:',('Apple', 'Amazon', 'Tesla', 'Ulta Beauty', 'Coca Cola'))
    days = col2.radio('Select the days:',('7 days', '15 days', '30 days'))
    method = col3.selectbox('Select the method:',('Univariate Forecast','Multivariate Forecast'))
    
    cc_title = '<p style="color:#ED7D31; font-size: 24px;">'+ method + ' - ' + company + ":" + '</p>'
    uni = univariate()
    multi = multivariate()

    if (company == 'Amazon'):
        getgif("amazon_gif.gif")
        t1, t2 = st.columns([1,7])
        t2.markdown(cc_title, unsafe_allow_html=True)

        if (method == 'Univariate Forecast'):
            uni.app(days.split(" ")[0],"AMZN", "uni_model_AMZN.json", "uni_model_AMZN.h5")
        else:
            var_day = days.split(" ")[0]
            var_model = "multivariate_models/model_multi_amzn_" + var_day + ".json"
            var_weight = "multivariate_models/model_multi_amzn_" + var_day + ".h5"
            # print(var_model)
            # print(var_weight)
            multi.app(var_day, "AMZN", var_model, var_weight)

    
    if (company == 'Apple'):
        getgif("apple_gif.gif")
        t1, t2 = st.columns([1,7])
        t2.markdown(cc_title, unsafe_allow_html=True)

        if (method == 'Univariate Forecast'):
            uni.app(days.split(" ")[0],"AAPL", "uni_model_AAPL.json", "uni_model_AAPL.h5")
            # uni.app(days.split(" ")[0],"AAPL", "model_AAPL.json", "model_AAPL.h5")
        else:
            var_day = days.split(" ")[0]
            var_model = "multivariate_models/model_multi_aapl_" + var_day + ".json"
            var_weight = "multivariate_models/model_multi_aapl_" + var_day + ".h5"
            multi.app(var_day, "AAPL", var_model, var_weight)

    elif (company == 'Tesla'):
        getgif("tesla_gif.gif")
        t1, t2 = st.columns([1,7])
        t2.markdown(cc_title, unsafe_allow_html=True)

        if (method == 'Univariate Forecast'):
            uni.app(days.split(" ")[0],"TSLA", "uni_model_TSLA.json", "uni_model_TSLA.h5")
            # uni.app(days.split(" ")[0],"TSLA", "model_TSLA.json", "model_TSLA.h5")
        else:
            var_day = days.split(" ")[0]
            var_model = "multivariate_models/model_multi_tsla_" + var_day + ".json"
            var_weight = "multivariate_models/model_multi_tsla_" + var_day + ".h5"
            multi.app(var_day, "TSLA", var_model, var_weight)

    elif (company == 'Ulta Beauty'):
        getgif("ulta_gif.gif")
        t1, t2 = st.columns([1,7])
        t2.markdown(cc_title, unsafe_allow_html=True)

        if (method == 'Univariate Forecast'):
            uni.app(days.split(" ")[0],"ULTA", "uni_model_ULTA.json", "uni_model_ULTA.h5")
            # uni.app(days.split(" ")[0],"ULTA", "model_ULTA.json", "model_ULTA.h5")
        else:
            var_day = days.split(" ")[0]
            var_model = "multivariate_models/model_multi_ulta_" + var_day + ".json"
            var_weight = "multivariate_models/model_multi_ulta_" + var_day + ".h5"
            multi.app(var_day, "ULTA", var_model, var_weight)

    elif (company == 'Coca Cola'):
        getgif("cocacola_gif.gif")
        t1, t2 = st.columns([1,7])
        t2.markdown(cc_title, unsafe_allow_html=True)

        if (method == 'Univariate Forecast'):
            uni.app(days.split(" ")[0],"KO", "uni_model_KO.json", "uni_model_KO.h5")
            # uni.app(days.split(" ")[0],"KO", "model_KO.json", "model_KO.h5")
        else:
            var_day = days.split(" ")[0]
            var_model = "multivariate_models/model_multi_ulta_" + var_day + ".json"
            var_weight = "multivariate_models/model_multi_ulta_" + var_day + ".h5"
            multi.app(var_day, "KO", var_model, var_weight)