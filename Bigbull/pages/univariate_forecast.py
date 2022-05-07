
#Importing Necessary Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import datetime as dt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
#plt.style.use('fivethirtyeight')


#Fetching the Stock Data
class univariate:
  def app(self, day, stock_tick, st_model, st_weight):
    day = int(day)
    from datetime import datetime
    import streamlit as st
    end=datetime.now()
    start=datetime(end.year - 5, end.month, end.day)
    stock_data=pdr.DataReader(stock_tick, data_source="yahoo",start=start,end=end)
    #Reset the Index
    stock_data1=stock_data.reset_index()['Close']
    #Feature Scaling
    from sklearn.preprocessing import StandardScaler
    scale=StandardScaler()
    stock_data1=scale.fit_transform(np.array(stock_data1).reshape(-1,1))
    train_size=int(len(stock_data1)*0.80)
    test_size=len(stock_data1)-train_size
    train_data,test_data=stock_data1[0:train_size,:],stock_data1[train_size:len(stock_data1),:1]

    #Size of Train & Test Data
    train_size,test_size

    import numpy
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_stamp=1):
      X, y = [], []
      for i in range(len(dataset)-time_stamp-1):
        a = dataset[i:(i+time_stamp), 0]    
        X.append(a)
        y.append(dataset[i + time_stamp, 0])
      return numpy.array(X), numpy.array(y)

    """Here, I took Time Stamp of 90 past days and Prediction Stamp of next 30 days to forecast the close price"""

    #Creating the Time_Stamp of 90 days for Splitting the dataset into Training and Testing Data
    if (stock_tick == 'TSLA'):
      time_stamp = 5
      # time_stamp = 90
    else:
      time_stamp = 30  #Number of Past days to make our model train 
      # time_stamp=90

    X_train, y_train = create_dataset(train_data, time_stamp)
    X_test, y_test = create_dataset(test_data, time_stamp)

    # reshape input to be [samples, time stamp, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    #Importing Necessary Libraries of Tensorflow for training LSTM Model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
    from keras.layers.core import Dense, Activation, Dropout,Flatten
    from keras.preprocessing import sequence
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    import tensorflow_addons as tfa
    import tensorflow as tf

    # #For Choosing Right number of Hidden Nodes
    # time_dim=1
    # hidden_nodes = int(2/3 * (time_stamp * time_dim))
    
    # load json and create model
    from keras.models import model_from_json
    json_file = open('pages/'+st_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("pages/"+st_weight)
    # print("Loaded model from disk")
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt,
                  loss="mean_squared_error",
                  metrics=[
                          tf.keras.metrics.MeanAbsoluteError(),
                          ])
    # Perform predictions
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)

    #Perfoming Inverse Transformation to get original data
    train_predict=scale.inverse_transform(train_predict)
    test_predict=scale.inverse_transform(test_predict)

    y_test=y_test.reshape(-1, 1)
    y_train=y_train.reshape(-1, 1)

    #Performing Inverse Transformation of Close Price
    y_test=scale.inverse_transform(y_test)
    y_train=scale.inverse_transform(y_train)

    #Calculating Mean Absolute Error on Test Data
    from sklearn.metrics import mean_absolute_error
    # MAE = mean_absolute_error(test_predict, y_test)
    # print("Stock:",stock_tick)
    # print("Day:",day)
    # print("Mean Absolute Error on Test Data:",MAE)

    from sklearn.metrics import mean_squared_error

    RMSE = np.sqrt(mean_squared_error(test_predict, y_test))
    print('='*60)
    print("Uni:")
    print("Ticker: ", stock_tick)
    print("Days: ", day)
    print("Timestamp: ", time_stamp)
    print("RMSE: ", RMSE)

    # MAE2 = mean_absolute_error(train_predict, y_train)
    # print("Mean Absolute Error on Train Data:",MAE2)

    # Shifting the Training predictions to past 90 days for plotting
    # look_back=90
    # plt.figure(figsize=(18,8))
    # trainPredictPlot = numpy.empty_like(stock_data1)
    # trainPredictPlot[:, :] = np.nan
    # trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

    # # Shifting test predictions for plotting
    # testPredictPlot = numpy.empty_like(stock_data1)
    # testPredictPlot[:, :] = numpy.nan
    # testPredictPlot[len(train_predict)+(look_back*2)+1:len(stock_data1)-1, :] = test_predict

    # # plot baseline and predictions
    # plt.plot(scale.inverse_transform(stock_data1))
    # plt.plot(trainPredictPlot)
    # plt.plot(testPredictPlot)
    # plt.xlabel("Days")
    # plt.ylabel("Close Price")
    # plt.title("Amazon: Predicted Close Price on Training and Testing Data of")
    # plt.show()

    """Here, The Blue Line Depicts original close price,
    Orange Line is the Prediction on Training Data and
    Green line is the prediction on Testing Data

    """

    #Taking the Past 90 days for Predicting Future ( Size of Test Data -90)

    a = len(test_data)
    x_input=test_data[a - time_stamp:].reshape(1,-1)
    # x_input.shape

    #Creating a Temp List to list the past input days
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    """Taking Moving Window of Past 90 Days and Predicting for next day and then appending the predicted output 
    in test data and then moving the window forward by 1 day for predicting the next day. Hence, this cycle will 
    repeat till next 30 days."""

    #Predicting for the Next 30 days
    from numpy import array

    next_output=[]
    n_steps=time_stamp 
    i=0
    while(i<day):
        
        if(len(temp_input)>time_stamp):
          
            x_input=np.array(temp_input[1:])
            # print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            
            yhat = model.predict(x_input, verbose=0)
            # print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
          
            next_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            # print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            # print(len(temp_input))
            next_output.extend(yhat.tolist())
            i=i+1

    #Length of Predicted Data
    # len(next_output)

    #Taking Past 90 days lookback to predict next 30 days stock market trends
    day_new=np.arange(1,time_stamp + 1)
    day_pred=np.arange(time_stamp + 1,time_stamp + 1 + day)

    # len(stock_data1)

    #Perfoming Inverse Transformation to get original data
    normal_output=scale.inverse_transform(next_output)

    #Converting to Dataframe
    normal_output=pd.DataFrame(normal_output).fillna('')

    # normal_output

    """Here, I am Creating a list of Next 30 Days Dates excluding the weekends on which stock market is closed."""

    #Creating the List of Next 30 days Dates
    import datetime 
    i=0
    weekdays=[]
    for i in range(60):
      if len(weekdays)<day:
        NextDay_Date = datetime.datetime.today() + datetime.timedelta(days=i+1)
        if NextDay_Date.isoweekday()!=6 and NextDay_Date.isoweekday()!=7:
          weekdays.append(NextDay_Date)

    #Strip the date in Y-M-D Format
    dates=[]
    for i in weekdays:
      a= i.strftime("%Y-%m-%d")
      dates.append(a)

    #Converting to Dataframe
    dates=pd.DataFrame(dates).fillna('')

    # dates

    final=pd.concat([dates,normal_output], axis=1, ignore_index=True)

    #Rename the Dataframe Name
    final.rename(columns = {0:"Date",1:"Close"}, inplace = True)

    final.reset_index(drop=True)

    final.set_index("Date",inplace=True)
    # print(final)
    # if stock_tick == "KO":
    #   final["Close"] = final["Close"]*1.2

    #For Visualization Purpose3
    a=stock_data['Close'][-90:]
    a=pd.DataFrame(a)

    #Historical & Forecasted Close Price
    category=[]
    for i in range(len(a)-1):
      category.append("History")
    for i in range(len(final)+1):
      category.append("Forecast")

    #Appending Both Data
    graph=a.append(final)
    graph['Close Price']=category
    # graph

    # Using plotly.express
    import plotly.express as px
    fig = px.line(graph, x=graph.index, y="Close", color='Close Price', width=1150, height=600)
    fig.update_layout(plot_bgcolor="rgb(0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig)

    # print("""Next 30 Days Forecasted Close Price""")
    #Creating a Table to store the above output.

if __name__ == '__main__':
    bx = app()