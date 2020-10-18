# %%
import requests
import pandas as pd
import pandas_datareader as pdr

#requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=BSE:SBIN&interval=15min&slice=year1month1&apikey=RM2J9YHI19A162UW').json()

KEY = 'RM2J9YHI19A162UW'
SYMBOL = 'BSE:SBIN'
BASE_API_URL = 'https://www.alphavantage.co/query?'
FUNCTION = 'TIME_SERIES_WEEKLY'
INTERVAL = '1min'

#url = BASE_API_URL + 'function='+FUNCTION+'&symbol='+SYMBOL+'&interval='+INTERVAL+'&apikey='+KEY
# url = BASE_API_URL + 'function='+FUNCTION+'&symbol='+SYMBOL+'&apikey='+KEY
# response_json = requests.get(url).json()
# response_json

# %%

def fetch(function,symbol,interval,key):
    url = BASE_API_URL + 'function='+function+'&symbol='+symbol+'&interval='+interval+'&apikey='+key
    response_json = requests.get(url).json()
    #print(response_json.keys())
    dataframe = pd.DataFrame.from_dict(response_json['Weekly Time Series'], orient='index').sort_index(axis=1)
    # dataframe.head()
    dataframe = dataframe.rename(columns={
        '1. open':'Open',
        '2. high':'High',
        '3. low':'Low',
        '4. close':'Close',
        '5. volume':'Volume'
    })
    dataframe = dataframe[['Open','High','Low','Close','Volume']]
    return dataframe


print(fetch(FUNCTION,SYMBOL,INTERVAL,KEY))
# %%
