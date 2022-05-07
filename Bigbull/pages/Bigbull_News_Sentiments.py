#Importing necessary libraries
import pandas as pd
import numpy as np
import re
import json
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.dates as mdates
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem.porter import *
import warnings
import csv
import streamlit as st
import base64
import plotly.graph_objects as go
from PIL import  Image
from itertools import cycle
from collections import Counter
import plotly.express as px
warnings.filterwarnings("ignore")

def app():
  title = '<p style="color:#ED7D31; font-size: 42px;">Sentiment Analysis</p>'
  st.markdown(title, unsafe_allow_html=True)

  #Reading the CSV File
  #To Display All Text
  def GetSentiments(data2, stock_label):
    pd.set_option('display.max_colwidth',None)
    #Drop Unnamed Column
    data2.drop(columns='Unnamed: 0', inplace=True)
    #Quick Look at the Data
    with st.expander("Raw Data"):
        st.write(data2.head())
    
    nltk.download('stopwords')
    #Load English Stop Words
    stopword = stopwords.words('english')
    data2['Tweets']=data2['Tweets'].str.lstrip('RT')
    data2['Tweets']=data2['Tweets'].str.replace( ":",'')
    data2['Tweets']=data2['Tweets'].str.replace( ";",'')
    data2['Tweets']=data2['Tweets'].str.replace( ".",'')
    data2['Tweets']=data2['Tweets'].str.replace( ",",'')
    data2['Tweets']=data2['Tweets'].str.replace( "!",'')
    data2['Tweets']=data2['Tweets'].str.replace( "&",'')
    data2['Tweets']=data2['Tweets'].str.replace( "-",'')
    data2['Tweets']=data2['Tweets'].str.replace( "_",'')
    data2['Tweets']=data2['Tweets'].str.replace( "$",'')
    data2['Tweets']=data2['Tweets'].str.replace( "/",'')
    data2['Tweets']=data2['Tweets'].str.replace( "?",'')
    data2['Tweets']=data2['Tweets'].str.replace( "''",'')
    data2['Tweets']=data2['Tweets'].str.replace( "httpstco","")
    #Lowercase
    data2['Tweets']=data2['Tweets'].str.lower()
    #Tweet Clean Function
    def tweet_clean(twee):
        #Remove URL
        twee = re.sub(r'https?://\S+|www\.\S+', " ", twee)
        #Remove Mentions
        twee = re.sub(r'@\w+',' ',twee)
        #Remove Digits
        twee = re.sub(r'\d+', ' ', twee)
        #Remove HTML tags
        twee = re.sub('r<.*?>',' ', twee)
        #Remove HTML tags
        twee = re.sub('r<.*?>',' ', twee)
        #Remove Stop Words 
        twee = twee.split()
        
        twee = " ".join([word for word in twee if word not in stopword])
        
        #Remove words with length less than equal to 2
        twee = ' '.join([w for w in twee.split() if len(w) > 2])

        return twee
    #Applying Tweet Clean Function
    data2['Clean Tweet'] = data2['Tweets'].astype(str).apply(lambda x: tweet_clean(x))
    # Tokenize Data
    tokenize_tweets = data2['Clean Tweet'].apply(lambda x: x.split()) 
    tokenize_tweets.head()
    # **Tokenization**
    #Tokenize the Tweets
    for i in range(len(tokenize_tweets)):
      tokenize_tweets[i] = ' '.join(tokenize_tweets[i])
    data2['Clean Tweet'] = tokenize_tweets
    data2.head()
    
    from collections import Counter
    all_words = [text for text in data2['Clean Tweet']]
    all_word = []
    for w in all_words:
        all_word.extend(w.split(" "))
    count_words = Counter(all_word)
    count_words = dict(sorted(count_words.items(),key=lambda item: -item[1])[:25])
    df_bubble = pd.DataFrame()
    df_bubble['word'] = count_words.keys()
    df_bubble['count'] = count_words.values()
    df_bubble = df_bubble.sample(frac=1).reset_index(drop=True)
    with st.expander('Bubble Chart'):
      fig = px.scatter(df_bubble, x=df_bubble.index, y="count",size="count", color="count",hover_name="word", size_max=60, width=1150, height=600)
      fig.update_layout(hovermode="y")
      fig.update_xaxes(showgrid=False)
      fig.update_yaxes(showgrid=False)
      st.plotly_chart(fig)
    
    analyser = SentimentIntensityAnalyzer()

    scores = []
    for sentence in data2['Clean Tweet']:
      score = analyser.polarity_scores(sentence)
      scores.append(score)
      
    scores = pd.DataFrame(scores)
    data2['Compound'] = scores['compound']
    data2['Negative'] = scores['neg']
    data2['Neutral'] = scores['neu']
    data2['Positive'] = scores['pos']
    data2.head()
    data2=data2.drop(columns=['User_ID','Tweet_ID',"Favorite Count"])
    # Set type of polarity
    polarity = []

    for i in range(len(data2)):
      if (data2['Compound'][i] < 0):
        polarity.append("Negative")
      elif (data2['Compound'][i] > 0):
        polarity.append("Positive")
      else:
        polarity.append("Neutral")
        
    polarity = pd.DataFrame(polarity)
    data2['Polarity'] = polarity
    data2.to_csv('NIFTYSentiments_20Sept.csv')
    #Creating a new column for Sentiment Labels
    data2['Sentiment Labels']=np.nan
    #Converting the Sentiments to the Numerical Labels for Visualization
    for i in data2.index:
        if data2['Polarity'][i]=='Negative':
            data2['Sentiment Labels'][i]=-1
        elif data2['Polarity'][i]=='Positive':
            data2['Sentiment Labels'][i]=1
        else:
            data2['Sentiment Labels'][i]=0

    # from matplotlib import colors
    # plt.figure(figsize=(10,6))
    #Histogram for Success Ratio of 1 month
    # N, bins, patches = plt.hist(data2['Compound'], bins=20)
    # fracs = N / N.max()
    # norm = colors.Normalize(fracs.min(), fracs.max())

    # for thisfrac, thispatch in zip(fracs, patches):
    #     color = plt.cm.viridis(norm(thisfrac))
    #     thispatch.set_facecolor(color)
        
    # plt.hist(data2['Compound'],density=True)
    # plt.title("Histogram for Sentiment Polarity",fontsize=12, weight='bold')
    # plt.grid()

    #Class Distribution
    class_df = data2.groupby('Polarity').count()['Sentiment Labels'].reset_index().sort_values(by='Sentiment Labels',ascending=False)
    class_df.style.background_gradient(cmap='PiYG')
    class_df.style.background_gradient(cmap='PiYG')

    with st.expander("Gauge Chart:"):
      fig = go.Figure(go.Indicator(
          mode = "gauge+number",
          value = float(data2.iloc[-1,-5]),
          domain = {'x': [0, 1], 'y': [0, 1]},
          title = {'text': "Today's Sentiment Score:", 'font': {'size': 24}},
          gauge = {
              'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
              'bar': {'color': "darkblue"},
              'borderwidth': 2,
              'bordercolor': "gray",
              'steps': [
                  {'range': [-1,0 ], 'color': 'crimson'},
                  {'range': [0,0], 'color': 'yellow'},
                  {'range': [0, 1], 'color': 'green'}],
              'threshold': {
                  'line': {'color': "red", 'width': 4},
                  'thickness': 0.75,
                  'value': float(data2.iloc[-1,-5])}}))

      fig.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})

      st.plotly_chart(fig)

    #Pie Chart
    with st.expander("Sentiment Labels - "+ stock_label):
      percent_class=class_df['Sentiment Labels']
      labels= class_df.Polarity

      fig = go.Figure(
        go.Pie(
        labels = labels,
        values = percent_class,
        hoverinfo = "label+percent",
        textinfo = "value",
        hole = .5
      ))

      st.plotly_chart(fig)
      # **Bar Plot: Sentiment Distribution**
      #Visualization of Sentiment Classes
    # with st.expander("Counter Plot for Sentiments Lables - "+ stock_label):
    #   st.bar_chart(data2['Sentiment Labels'].value_counts())

  # st.write("Stocks:")
  # amazon_logo = Image.open('Images/amazon_icon.png')
  # apple_logo = Image.open('Images/apple_icon.png')
  # tesla_logo = Image.open('Images/tesla_icon.png')
  # ulta_logo = Image.open('Images/ulta_icon.png')
  # coca_logo = Image.open('Images/cocacola_icon.png')

  e1,co1, co2, co3, co4, co5,e2 = st.columns(7)

  file_ = open("Images/amazon_icon.png", "rb")
  contents = file_.read()
  data_url = base64.b64encode(contents).decode("utf-8")
  file_.close()
  co1.markdown(f'<img src="data:image/gif;base64,{data_url}" height=90 width=100>',unsafe_allow_html=True)

  file_ = open("Images/apple_icon.png", "rb")
  contents = file_.read()
  data_url = base64.b64encode(contents).decode("utf-8")
  file_.close()
  co2.markdown(f'<img src="data:image/gif;base64,{data_url}" height=80 width=80>',unsafe_allow_html=True)

  file_ = open("Images/tesla_icon.png", "rb")
  contents = file_.read()
  data_url = base64.b64encode(contents).decode("utf-8")
  file_.close()
  co3.markdown(f'<img src="data:image/gif;base64,{data_url}" height=90 width=100>',unsafe_allow_html=True)

  file_ = open("Images/ulta_icon.png", "rb")
  contents = file_.read()
  data_url = base64.b64encode(contents).decode("utf-8")
  file_.close()
  co4.markdown(f'<img src="data:image/gif;base64,{data_url}" height=90 width=100>',unsafe_allow_html=True)

  file_ = open("Images/cocacola_icon.png", "rb")
  contents = file_.read()
  data_url = base64.b64encode(contents).decode("utf-8")
  file_.close()
  co5.markdown(f'<img src="data:image/gif;base64,{data_url}" height=90 width=100>',unsafe_allow_html=True)

  # col1.image(amazon_logo)
  # col2.image(apple_logo)
  # col3.image(tesla_logo)
  # col4.image(ulta_logo)
  # col5.image(coca_logo)
  e1,col1, col2, col3, col4, col5,e2 = st.columns(7)

  if (col1.button("Amazon")):
    data2= pd.read_csv("amzn_28.csv")
    GetSentiments(data2, "Amazon")

  if (col2.button("Apple")):
    data2= pd.read_csv("aapl_28.csv")
    GetSentiments(data2, "Apple")

  if (col3.button("Tesla")):
    data2= pd.read_csv("tsla_28.csv")
    GetSentiments(data2,"Tesla")

  if (col4.button("Ulta Beauty")):
    data2= pd.read_csv("ulta_28.csv")
    GetSentiments(data2,"Ulta Beauty")

  if (col5.button("Coca Cola")):
    data2= pd.read_csv("cocacola_28.csv")
    GetSentiments(data2, "Coca Cola")

  # st.write("Please click on a Stock to see the Sentiments.")