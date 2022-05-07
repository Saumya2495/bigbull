#Importing Necessary Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use("fivethirtyeight")
import pandas_datareader as pdr
import datetime as dt
from datetime import datetime
import yfinance as yf
import datetime 
import base64
import plotly.express as px
import streamlit as st

def app():
  title = '<p style="color:#ED7D31; font-size: 42px;">Algorithmic Trading</p>'
  st.markdown(title, unsafe_allow_html=True)

  choices = {"AMZN": ["Amazon","amazon_gif.gif"], "AAPL" : ["Apple","apple_gif.gif"], "TSLA" : ["Tesla", "tesla_gif.gif"], "ULTA" : ["Ulta Beauty", "ulta_gif.gif"]}

  def format_func(option):
    return choices[option][0]

  company = st.selectbox('Select the company:', options = list(choices.keys()), format_func = format_func)

  col2, col3, col4 = st.columns([2,3,2])

  stock_data = yf.download(company, start="2020-03-01", end="2022-03-01",group_by="ticker") 

  if(col2.button("Moving Average")):
    
    # Moving Average
    shortEMA= stock_data.Close.ewm(span=5, adjust=False).mean() #Moving Window 
    medEMA= stock_data.Close.ewm(span=15, adjust=False).mean() #Moving Window 
    longEMA= stock_data.Close.ewm(span=30, adjust=False).mean() #Moving Window

    #Visualize the Close Price and Exponential Moving Average
    # plt.figure(figsize=(20,8))
    # plt.title("Close Price with MA",fontsize=16)
    # plt.plot(stock_data["Close"], label="Close Price", color="Blue")
    # plt.plot(shortEMA, label="Short EMA", color="red")
    # plt.plot(medEMA, label="Medium EMA", color="orange")
    # plt.plot(longEMA, label="Long EMA", color="green")

    # plt.xlabel("Date",fontsize=14)
    # plt.ylabel("Close Price USD($)",fontsize=14)
    # plt.show()

    stock_data["Short EMA"]= shortEMA
    stock_data["Medium EMA"]= medEMA
    stock_data["Long EMA"]= longEMA
   
    def buy_sell(data):
      buy=[]
      sell=[]
      flag_long=False
      flag_short=False

      for i in range(0, len(data)):
        if data["Medium EMA"][i] < data["Long EMA"][i] and data["Short EMA"][i] < data["Medium EMA"][i] and flag_long==False:
          buy.append(data["Close"][i])
          sell.append(np.nan)
          flag_short=True
        elif flag_short==True and data["Short EMA"][i] > data["Medium EMA"][i]:
          sell.append(data["Close"][i])
          buy.append(np.nan)
          flag_short=False
        elif data["Medium EMA"][i] > data["Long EMA"][i] and data["Short EMA"][i] > data["Medium EMA"][i] and flag_long==False:
          buy.append(data["Close"][i])
          sell.append(np.nan)
          flag_long=True
        elif flag_long==True and data["Short EMA"][i] < data["Medium EMA"][i]:
          sell.append(data["Close"][i])
          buy.append(np.nan)
          flag_long=False
        else:
          buy.append(np.nan)
          sell.append(np.nan)
      return (buy,sell)

    stock_data["Buy"]= buy_sell(stock_data)[0]
    stock_data["Sell"]= buy_sell(stock_data)[1]
    c1,c2,c3,c4 = st.columns(4)

    file_ = open("Images/" + choices[company][1], "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    c2.markdown(f'<img src="data:image/gif;base64,{data_url}" height=150 width=200 alt="cat gif">',unsafe_allow_html=True)
    st.write("")
    st.write("")
    stock_data_buy = stock_data['Buy'].replace(np.nan, 0)
    stock_data_sell = stock_data['Sell'].replace(np.nan, 0)
    profit_loss = ((sum(stock_data_sell) - sum(stock_data_buy)) / sum(stock_data_buy)) * 100

    if(profit_loss >= 0):
      c3.metric(label="Profit %", value=round(profit_loss,3), delta=round(profit_loss,3))
    else:
      c3.metric(label="Loss %", value=round(profit_loss,3), delta=round(profit_loss,3))

    with st.expander('Visualize the Close Price with Moving Average'):
      fig = px.line(stock_data, x=stock_data.index, y=["Close","Short EMA", "Medium EMA", "Long EMA"], width=1150, height=600)
      # fig.update_layout(plot_bgcolor="rgb(0,0,0)")
      fig.update_xaxes(showgrid=False)
      fig.update_yaxes(showgrid=False)
      st.plotly_chart(fig)

    with st.expander('Buy-Sell Signals using Moving Average'):
      fig, ax = plt.subplots(figsize=(20,8))
      # ax.figure(figsize=(20,8))
      ax.set_title("Buy-Sell Signals",fontsize=12)
      ax.plot(stock_data["Close"], label="Close Price", color="orange")

      ax.scatter(stock_data.index, stock_data["Buy"], label='Buy it',color="green", marker="^", alpha=1)
      ax.scatter(stock_data.index, stock_data["Sell"], label='Sell it',color="red", marker="v", alpha=1)
      ax.set_xlabel("Date",fontsize=11)
      ax.set_ylabel("Close Price USD($)",fontsize=11)
      ax.legend(loc='upper right')
      # ax.grid()
      st.pyplot(fig)
      
  #Calculate the MACD and Signal Line Indicators
  if(col3.button("Moving Average Convergence/Divergence")):
    #SHORT EXPONENTIAL MOVING AVERAGE
    shortEMACD= stock_data.Close.ewm(span=7, adjust=False).mean() #Moving Window 
    #MEDIUM EXPONENTIAL MOVING AVERAGE
    longEMACD= stock_data.Close.ewm(span=15, adjust=False).mean() #Moving Window 

    #Calculate MACD
    MACD=shortEMACD-longEMACD
    #Calculate Signal Indicator
    si=MACD.ewm(span=9, adjust=False).mean()


    # In[36]:


    #Visualize the MACD and Signal Indicator
    # plt.figure(figsize=(20,8))
    # plt.title("Buy-Sell Signals",fontsize=16)

    # plt.plot(stock_data.index, MACD, color="purple", label="MACD" )
    # plt.plot(stock_data.index, si, label="Signal Indicator" ,color="green")
    # plt.xlabel("Date",fontsize=14)
    # plt.ylabel("Close Price USD($)",fontsize=14)

    # plt.legend(loc='upper right')
    # plt.show()


    # In[37]:


    #Adding MACD and Signal Indicators in original dataset
    stock_data["MACD"]=MACD
    stock_data["Signal Indicators"]=si


    #Create a function to indicate Buy or Sell Signals based on MACD and Signal Line
    def buy_sell_macd(signal):
      Buy=[]
      Sell=[]
      flag=-1

      for i in range(0, len(signal)):
        if signal["MACD"][i] > signal["Signal Indicators"][i]:
          Sell.append(np.nan)
          if flag!=1:
            Buy.append(signal["Close"][i])
            flag=1
          else:
            Buy.append(np.nan)
        elif signal["MACD"][i] < signal["Signal Indicators"][i]:
          Buy.append(np.nan)
          if flag!=0:
            Sell.append(signal["Close"][i])
            flag=0
          else:
            Sell.append(np.nan)
        else:
          Buy.append(np.nan)
          Sell.append(np.nan)
      return(Buy,Sell)


    # In[43]:


    #Appending Buy Sell result in original dataset
    macd_bs= buy_sell_macd(stock_data)
    stock_data["Buy_MACD"]=macd_bs[0]
    stock_data["Sell_MACD"]=macd_bs[1]

    a1,a2,a3,a4 = st.columns(4)

    file_ = open("Images/" + choices[company][1], "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    a2.markdown(f'<img src="data:image/gif;base64,{data_url}" height=150 width=200 alt="cat gif">',unsafe_allow_html=True)

    stock_data_buy = stock_data['Buy_MACD'].replace(np.nan, 0)
    stock_data_sell = stock_data['Sell_MACD'].replace(np.nan, 0)
    profit_loss = ((sum(stock_data_sell) - sum(stock_data_buy)) / sum(stock_data_buy)) * 100

    st.write("")
    st.write("")

    if(profit_loss >= 0):
      a3.metric(label="Profit %", value=round(profit_loss,3), delta=round(profit_loss,3))
    else:
      a3.metric(label="Loss %", value=round(profit_loss,3), delta=round(profit_loss,3))
    
    with st.expander('Visualize the Close Price with MACD'):
      fig = px.line(stock_data, x=stock_data.index, y=["MACD","Signal Indicators"], width=1150, height=600)
      # fig.update_layout(plot_bgcolor="rgb(0,0,0)")
      fig.update_xaxes(showgrid=False)
      fig.update_yaxes(showgrid=False)
      st.plotly_chart(fig)

    with st.expander('Buy-Sell Signals using MACD'):
      fig, ax = plt.subplots(figsize=(20,8))
      ax.set_title("Buy-Sell Signals using MACD",fontsize=12)
      ax.plot(stock_data["Close"], label="Close Price", color="black", alpha=0.4)

      ax.scatter(stock_data.index,  stock_data["Buy_MACD"], label='Buy it',color="green", marker="^", alpha=1 )
      ax.scatter(stock_data.index, stock_data["Sell_MACD"], label='Sell it', color="red", marker="v", alpha=1)

      ax.set_xlabel("Date",fontsize=11)
      ax.set_ylabel("Close Price USD($)",fontsize=11)
      ax.legend(loc='upper right')
      st.pyplot(fig)

  if(col4.button("Money Flow Index")):
    #Calculate the Typical Price of Stocks
    typical_price= (stock_data["High"] + stock_data["Low"] + stock_data["Close"])/3
    #Setting Period of 14 days as MFI generally uses 14 days of period.
    period=14

    #Calculate MFI
    moneyflow= typical_price*stock_data["Volume"]

    #Positive and Negative MFI
    pos_mfi=[]
    neg_mfi=[]

    for i in range(1, len(typical_price)):
      if typical_price[i] > typical_price[i-1]:
        pos_mfi.append(moneyflow[i-1])
        neg_mfi.append(0)
      elif typical_price[i] < typical_price[i-1]:
        neg_mfi.append(moneyflow[i-1])
        pos_mfi.append(0)
      else:
        pos_mfi.append(0)
        neg_mfi.append(0)
    
    pos_mf=[]
    neg_mf=[]

    for i in range(period-1, len(pos_mfi)):
      pos_mf.append(sum(pos_mfi[i+1-period : i+1]))
    for i in range(period-1, len(neg_mfi)):
      neg_mf.append(sum(neg_mfi[i+1-period : i+1]))
    
    MFI= 100 * (np.array(pos_mf) / (np.array(pos_mf) + np.array(neg_mf)))

    #Visualization of MFI
    mfi_df=pd.DataFrame()
    mfi_df["MFI"]=MFI

    #Create the plot
    # plt.figure(figsize=(20,8))
    # plt.title("Money Flow Index",fontsize=16)
    # plt.plot(mfi_df["MFI"], label="MFI", color="orange")

    # plt.axhline(10, linestyle="--", color="red")
    # plt.axhline(20, linestyle="--", color="green")
    # plt.axhline(80, linestyle="--", color="green")
    # plt.axhline(90, linestyle="--", color="red")

    # plt.ylabel("MFI Values",fontsize=14)
    # plt.show()

    stock_data1=pd.DataFrame()
    stock_data1=stock_data[period:]
    stock_data1["MFI"]=MFI

    #Create a function to generate Buy or Sell Signals
    def get_buysell(data,high,low):
      buy_MFI=[]
      sell_MFI=[]

      for i in range(0, len(data["MFI"])):
        if data["MFI"][i] > high:
          buy_MFI.append(np.nan)
          sell_MFI.append(data["Close"][i])
        elif data["MFI"][i] < low:
          buy_MFI.append(data["Close"][i])
          sell_MFI.append(np.nan)
        else:
          buy_MFI.append(np.nan)
          sell_MFI.append(np.nan)
      return (buy_MFI, sell_MFI)
    #Adding Buy or Sell Signal to original data
    stock_data1["Buy_MFI"]=get_buysell(stock_data1,80,20)[0] #Overbought Values
    stock_data1["Sell_MFI"]=get_buysell(stock_data1,80,20)[1] #Oversold Values

    c1,c2,c3,c4 = st.columns(4)

    file_ = open("Images/" + choices[company][1], "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    c2.markdown(f'<img src="data:image/gif;base64,{data_url}" height=150 width=200 alt="cat gif">',unsafe_allow_html=True)
    st.write("")
    st.write("")
    
    # print(stock_data1['Buy_MFI'])

    stock_data_buy = stock_data1['Buy_MFI'].replace(np.nan, 0)
    
    stock_data_sell = stock_data1['Sell_MFI'].replace(np.nan, 0)
    profit_loss = ((sum(stock_data_sell) - sum(stock_data_buy)) / sum(stock_data_buy)) * 100

    if(profit_loss >= 0):
      c3.metric(label="Profit %", value=round(profit_loss,3), delta=round(profit_loss,3))
    else:
      c3.metric(label="Loss %", value=round(profit_loss,3), delta=round(profit_loss,3))

    with st.expander('Visualize the Close Price with Money Flow Index'):
      fig, ax = plt.subplots(figsize=(20,8))
      ax.set_title("Money Flow Index",fontsize=12)
      ax.plot(mfi_df["MFI"], label="MFI", color="orange")

      ax.axhline(10, linestyle="--", color="red")
      ax.axhline(20, linestyle="--", color="green")
      ax.axhline(80, linestyle="--", color="green")
      ax.axhline(90, linestyle="--", color="red")

      ax.set_ylabel("MFI Values",fontsize=11)
      st.pyplot(fig)

    #Visualize the Buy Sell Signals
    with st.expander('Buy-Sell Signals using Money Flow Index'):
      fig, ax = plt.subplots(figsize=(20,8))
      ax.set_title("Buy-Sell Signals using MFI",fontsize=12)
      ax.plot(stock_data1["Close"], label="Close Price", color="brown", alpha=0.4)

      ax.scatter(stock_data1.index,  stock_data1["Buy_MFI"], label='Buy',color="green", marker="^", alpha=1 )
      ax.scatter(stock_data1.index, stock_data1["Sell_MFI"], label='Sell', color="red", marker="v", alpha=1)

      ax.set_xlabel("Date",fontsize=11)
      ax.set_ylabel("Close Price USD($)",fontsize=11)
      ax.legend(loc='upper right')
      st.pyplot(fig)




