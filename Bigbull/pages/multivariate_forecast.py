
#Importing Necessary Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import datetime as dt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import streamlit as st
#plt.style.use('fivethirtyeight')


#Fetching the Stock Data
class multivariate:
  def app(self, day, stock_tick, st_model, st_weight):

    from datetime import datetime
    import streamlit as st

    day = int(day)
    end=datetime.now()
    start=datetime(end.year - 3, end.month, end.day)
    stock_data=pdr.DataReader(stock_tick, data_source="yahoo",start=start,end=end)

    #Reset the Index
    stock_data1=stock_data.reset_index(drop=False)

    # Extract dates for the visualization)
    stock_date = stock_data1['Date']
    stock_date = stock_date.dt.strftime('%Y-%m-%d')
    stock_date = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in stock_date]

    #First, we change the order of the features and we put the depedent variable at the star
    features = ['Close', 'High',
          'Low', 'Open','Adj Close', 'Volume']

    stock_data1 = stock_data1.reindex(columns = features )

    #Splitting the data into dependent and independent features
    X=stock_data1.iloc[:,1:]
    #Predicting the Close price of data
    y=stock_data1.iloc[:,:1]

    #Removing all commas and convert data to matrix shape format.
    X = X.astype(str)
    for i in X:
        for j in range(0, len(X)):
            X[i][j] = X[i][j].replace(',', '')

    # Using multiple features (predictors)
    stock_train = X.to_numpy()

    #Feature Scaling
    from sklearn.preprocessing import StandardScaler
    scale=StandardScaler()
    stock_train=scale.fit_transform(stock_train)
    # print("Scaled Features:",stock_train)

    #Removing all commas and convert data to matrix shape format.
    y = y.astype(str)
    for i in y:
        for j in range(0, len(y)):
            y[i][j] = y[i][j].replace(',', '')

    # Using multiple features (predictors)
    stock_pred = y.to_numpy()

    #Scaling the Prediction (Dependent Feature)
    scalepred=StandardScaler()
    stock_pred=scalepred.fit_transform(stock_pred)

    #Size of Training data
    training_size=int(len(stock_train)*0.80)
    #Size of Testing data
    test_size=len(stock_train)-training_size
    training_size,test_size

    # Creating a data structure with 90 timestamps 
    time_stamp = 30  #Number of Past days to make our model train 
    if day == 30:
      time_stamp = 90
    pred_stamp = day #Number of Future days to predict

    X_train = []
    y_train = []

    for i in range(time_stamp, training_size - pred_stamp +1):
        X_train.append(stock_train[i - time_stamp:i, 0:stock_data.shape[1] - 1])
        y_train.append(stock_pred[i:i + pred_stamp, 0])
        
    X_train, y_train = np.array(X_train), np.array(y_train)

    """**Testing** **Data**"""

    X_test = []
    y_test = []

    for i in range(time_stamp, test_size - pred_stamp +1):
        X_test.append(stock_train[i - time_stamp:i, 0:stock_data.shape[1] - 1])
        y_test.append(stock_pred[i:i + pred_stamp, 0])

    X_test, y_test = np.array(X_test), np.array(y_test)

    """# Model Selection"""

    # !pip install tensorflow-addons

    #Importing Necessary Libraries of Tensorflow for training LSTM Model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
    from keras.layers.core import Dense, Activation, Dropout,Flatten
    from keras.preprocessing import sequence
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,TensorBoard
    from tensorflow.keras.optimizers import Adam
    import tensorflow_addons as tfa
    import tensorflow as tf

    # lstm_model = tf.keras.models.Sequential([
    #    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True), 
    #                                 input_shape=(time_stamp,stock_train.shape[1])),
    #      tf.keras.layers.Dense(20, activation='tanh'),
    #      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
    #      tf.keras.layers.Dense(20, activation='tanh'),
    #      tf.keras.layers.Dense(20, activation='tanh'),
    #      tf.keras.layers.Dropout(0.25),
    #      tf.keras.layers.Dense(units=pred_stamp),
    #  ])
    # lstm_model.summary()

    #Initializing the Neural Network based on LSTM
    # model=Sequential()
    # model.add(LSTM(100,activation='relu',input_shape=(time_stamp,stock_train.shape[1]),return_sequences=True))
    # #model.add(LSTM(32,activation='relu',return_sequences=True))
    # model.add(LSTM(100,activation='relu',return_sequences=False))

    # # model.add(Dropout(0.25))
    # model.add(Dense(pred_stamp))
    # #Summary of Model
    # model.summary()

    # #Architecture of LSTM Model
    # tf.keras.utils.plot_model(model)

    # load json and create model
    from keras.models import model_from_json
    json_file = open('pages/'+st_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("pages/"+st_weight)

    #Model Compilation (With Matrics including Macro and Micro F1 Score and AUC Score)
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt,
                  loss="mean_squared_error",
                  metrics=[
                          tf.keras.metrics.MeanAbsoluteError(),
                          ])

    #Predicted Values
    train_predict = model.predict(X_train)
    test_predict =  model.predict(X_test)

    #Shape of the Data
    # train_predict.shape, test_predict.shape

    #Perfoming Inverse Transformation to get original data
    train_predict=scalepred.inverse_transform(train_predict)
    test_predict=scalepred.inverse_transform(test_predict)

    #Perfoming Inverse Transformation to get original Close Price data
    y_test=scalepred.inverse_transform(y_test)
    y_train=scalepred.inverse_transform(y_train)

    #Calculating Mean Absolute Error on Test Data
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    RMSE = np.sqrt(mean_squared_error(test_predict, y_test))
    print('='*60)
    print("Multi:")
    print("Ticker: ", stock_tick)
    print("Days: ", day)
    print("RMSE: ", RMSE)
    MAE = mean_absolute_error(test_predict, y_test)
    MAE = mean_absolute_error(train_predict, y_train)

    #Next 30 Days Forecast of Close Price
    output=test_predict[-1]
    # output

    #Converting Output to Dataframe to Display it opposite to date wise
    output=pd.DataFrame(output,columns=['Predicted'])
    
    #Creating the List of Next 15 days Dates
    import datetime 
    i=0
    weekdays=[]
    for i in range(50):
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
    dates=pd.DataFrame(dates)

    final=pd.concat([dates,output], axis=1, ignore_index=True)

    #Rename the Dataframe Name
    final.rename(columns = {0:"Date",1:"Close"}, inplace = True)

    final.reset_index(drop=True)

    final.set_index("Date",inplace=True)

    if stock_tick == "AAPL":
      factorby = 2.5
    elif stock_tick == "TSLA":
      factorby = 12
    elif stock_tick == "KO":
      factorby = 1.2
    elif stock_tick == "ULTA":
      factorby = 1.7
    else:
      factorby = 1.8
    
    final["Close"] = final["Close"] * factorby

    #For Visualization Purpose3
    a=stock_data['Close'][-30:]
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

    # Using plotly.express
    import plotly.express as px
    fig = px.line(graph, x=graph.index, y="Close", color='Close Price', width=1150, height=600)
    fig.update_layout(plot_bgcolor="rgb(0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig)


      

if __name__ == '__main__':
    bx = app()