#%%
#imports
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
Scaler = MinMaxScaler(feature_range=(0,1))



#imports for model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout



class API:
    
    def __init__(self,symbol):
        self.symbol = symbol
        

    def fetchFromYahoo(self):
        yobj = yf.Ticker(self.symbol)
        tickerDict = yobj.info
        #print(yobj.info.keys())
        df = yobj.history(period="max")
        df = df.drop(['Dividends','Stock Splits'],axis=1)
        df.index =  pd.to_datetime(df.index)
        #print('\n'+tickerDict['longBusinessSummary'])
        print(df.tail())
        plt.plot(df['Close'])
        return df,tickerDict
       
    
    def get_train_test_dataset(self,df):
        try:
            print('this will return a training and test data')
            print('\n'+'Recent Data' + '\n',df.tail())
            print('MEAN CLOSE: ',df['Close'].mean())
            print('MAX CLOSE: ',df['Close'].max())
            print('MIN CLOSE: ',df['Close'].min())
            close_price = df.reset_index()['Close']
            close_price = Scaler.fit_transform(np.array(close_price).reshape(-1,1))
            train_size = int(len(close_price)*0.70)
            test_size = int(len(close_price*0.30))
            train_data = close_price[0:train_size,:]
            test_data = close_price[train_size:len(close_price),:1]
            return train_data,test_data
        except ValueError:
            print('Try a different Scrip')


    def prepare_data_for_LSTM(self,dataset,timestep=1):
        dataX, dataY = [], []
        for i in range(len(dataset)- timestep-1):
            record = dataset[i:(i+timestep),0]
            dataX.append(record)
            dataY.append(dataset[i + timestep, 0])
        return np.array(dataX), np.array(dataY)

    def reshape_for_LSTM(self,train_data, test_data):
        train_data = train_data.reshape(train_data.shape[0],train_data.shape[1],1)
        test_data = test_data.reshape(test_data.shape[0],test_data.shape[1],1)
        return train_data, test_data


    def create_LSTM_model(self,lstm_layers_after_main=0,lstm_units=1,shape=(),loss='mean_squared_error',optimizer='adam'):
        dropout = 0.0
        model = Sequential()
        model.add(LSTM(lstm_units,return_sequences=True,input_shape=shape))
        if lstm_layers_after_main > 2 and lstm_layers_after_main < 5:
            dropout = 0.4
        elif lstm_layers_after_main < 2:
            dropout = 0.
        for i in range(lstm_layers_after_main):
            model.add(LSTM(lstm_units,return_sequences=True))
            if i % 2 == 0:
                continue
            model.add(Dropout(dropout))
        
        model.add(LSTM(lstm_units))
        model.add(Dense(1))
        print('Dropping out ' + str(dropout*100) + '%')
        model.summary()
        model.compile(loss=loss,optimizer=optimizer)
        return model



symbol = str(input('ENTER SYMBOL: '))+'.NS'
obj = API('TCS.NS')
df,dictionary = obj.fetchFromYahoo()
train_data, test_data = obj.get_train_test_dataset(df)
#prepare seperate for train and test
xtrain, ytrain = obj.prepare_data_for_LSTM(train_data, timestep = 100)
xtest, ytest = obj.prepare_data_for_LSTM(test_data, timestep = 100)

#shaping data for LSTM
xtrain, xtest = obj.reshape_for_LSTM(xtrain,xtest)

#create model
model = obj.create_LSTM_model(lstm_layers_after_main=4,
                                lstm_units=100,
                                shape=(xtrain.shape[1],xtrain.shape[2])
                                )

history = model.fit(xtrain,ytrain,batch_size=32,epochs=10,validation_data=(xtest,ytest))



# %%
