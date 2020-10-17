# %%
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding
from alpha_vantage.timeseries import TimeSeries


KEY = 'RM2J9YHI19A162UW'
SYMBOL = 'AAPL'
INTERVAL = '1min'
OUTPUT_FORMAT = 'pandas' #or json
 
def fetch_data_alphav(key, symbol, interval, output_format):
    ts = TimeSeries(key,output_format= output_format)
    data, metadata = ts.get_intraday(symbol,interval,outputsize='full')
    data = data.rename(columns={
        '1. open':'Open',
        '2. high':'High',
        '3. low':'Low',
        '4. close':'Close',
        '5. volume':'Volume'
    })
    return data, metadata


df, metadata = fetch_data_alphav(KEY,SYMBOL, INTERVAL,OUTPUT_FORMAT)

# %%
