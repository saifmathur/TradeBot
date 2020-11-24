# %%

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
import pandas as pd
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
import numpy as np


def fetchFromYahoo(symbol,**kwargs):
        yobj = yf.Ticker(symbol)
        tickerDict = yobj.info
        #print(yobj.info.keys())
        df = yobj.history(period="max")
        df = df.drop(['Dividends','Stock Splits'],axis=1)
        df.index =  pd.to_datetime(df.index)
        #print('\n'+tickerDict['longBusinessSummary'])
        return df,tickerDict



def get_train_test_dataset(df):
    try:
        print('this will return a training and test data')
        print('\n DATA INSIGHT')
        print('\n'+'Recent Data' + '\n',df.tail())
        print('Shape of dataframe',df.shape)
        print('MEAN CLOSE: ',df['Close'].mean())
        print('MAX CLOSE: ',df['Close'].max())
        print('MIN CLOSE: ',df['Close'].min())
        print('\n')
        print('PREPROCESSING...')
        close_price = df.reset_index()['Close']
        close_price = scaler.fit_transform(np.array(close_price).reshape(-1,1))
        train_size = int(len(close_price)*0.70)
        test_size = int(len(close_price*0.30))
        train_data = close_price[0:train_size,:]
        test_data = close_price[train_size:len(close_price),:1]
        return train_data,test_data
    except ValueError:
        print('Try a different Scrip')




def create_dataset(dataset,timestep=1):
    #code 




df,info = fetchFromYahoo('EICHERMOT.NS')
train_data,test_data = get_train_test_dataset(df)



# %%

df.columns





# %%
