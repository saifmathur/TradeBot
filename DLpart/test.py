# %%
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding
key = 'RM2J9YHI19A162UW'

#fetch_df = pdr.get_data_alphavantage('SBI',key)
from alpha_vantage.timeseries import TimeSeries
# %%
ts = TimeSeries(key=key, output_format='pandas')
# ts.get
data, metadata = ts.get_intraday(symbol='BSE:SBIN', interval='1min', outputsize='full')
# %%
#using nsetools for top gainers,losers,most active
from nsetools import Nse
nse = Nse()
infy = nse.get_quote('INFY')

# %%
#fetching by request
import requests
import alpha_vantage
import pandas as pd
import pandas_datareader as pdr

KEY = 'RM2J9YHI19A162UW'
API_URL = 'https://www.alphavantage.co/query/'
SYMBOL = 'INFY'


import json
import sys
# response = requests.get(API_URL,data).json()



def intra_fetch(url,key,symbol):
    struct = {}
    data = {
        'function':'TIME_SERIES_INTRADAY_EXTENDED',
        'symbol': symbol,
        'interval':'1min',
        'slice':'year1month1', #most recent
        'apikey': key,
        'datatype':'csv'
    }
    return requests.get(url,data).json()
    #print(json.loads(requests.get(url,data).json()))
    # try:
    #     dataform = str(requests.get(url,data)).strip("'<>()").replace('\'','\"')
    #     struct = json.loads(dataform)
    #     return struct
    # except:
    #     print(repr(dataform))
    #     print(sys.exc_info())
    

json_var = intra_fetch(API_URL,KEY,SYMBOL)


# %%

data = pd.read_csv(json_var)



# %%
# data = pd.DataFrame.from_dict(response_json['Time Series Intraday'], orient='index').sort_index(axis=1)
# data = data.rename(columns={
#     '1. open':'Open',
#     '2. high':'High',
#     '3. low':'Low',
#     '4. close':'Close',
#     '5. volume':'Volume'
# })

# data = data[['Open','High','Low','Close','Volume']]


# %%
import yfinance as yf
nsei = yf.Ticker("BAJAJFINSV.NS")
nsei.info
# %%
hist = nsei.history(period="max")
# %%
nsei.dividends
# %%
