# %%

########################## DATA FETCHING ####################################
import requests
import pandas as pd
import pandas_datareader as pdr

#requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=BSE:SBIN&interval=15min&slice=year1month1&apikey=RM2J9YHI19A162UW').json()

KEY = 'RM2J9YHI19A162UW'
SYMBOL = 'BSE:SBIN'
BASE_API_URL = 'https://www.alphavantage.co/query?'
FUNCTION = 'TIME_SERIES_WEEKLY'
INTERVAL = '1min'
SLICE = 'year1month1'

def fetchIntraday(symbol,interval,key):
    url = BASE_API_URL + 'function=TIME_SERIES_INTRADAY'+'&symbol='+symbol+'&interval='+interval+'&apikey='+key
    response_json = requests.get(url).json()
    # print(response_json)
    # print(response_json.keys())
    dataframe = pd.DataFrame.from_dict(response_json['Time Series (1min)'], orient='index').sort_index(axis=1)
    dataframe = dataframe.rename(columns={
        '1. open':'Open',
        '2. high':'High',
        '3. low':'Low',
        '4. close':'Close',
        '5. volume':'Volume'
    })
    dataframe = dataframe[['Open','High','Low','Close','Volume']]
    return dataframe

# df = fetchIntraday(SYMBOL,INTERVAL,KEY)

def fetchWeekly(symbol,interval,key):
    url = BASE_API_URL + 'function=TIME_SERIES_WEEKLY'+'&symbol='+symbol+'&apikey='+key
    response_json = requests.get(url).json()
    print(response_json.keys())
    dataframe = pd.DataFrame.from_dict(response_json['Weekly Time Series'], orient='index').sort_index(axis=1)
    dataframe = dataframe.rename(columns={
        '1. open':'Open',
        '2. high':'High',
        '3. low':'Low',
        '4. close':'Close',
        '5. volume':'Volume'
    })
    dataframe = dataframe[['Open','High','Low','Close','Volume']]
    return dataframe


def fetchIntradayExtended(symbol,interval,Slice,key):
    url = BASE_API_URL + 'function=TIME_SERIES_INTRADAY_EXTENDED'+'&symbol='+symbol+'&interval='+interval+'&slice='+Slice+'&apikey='+key
    response_json = requests.get(url).json()
    #print(response_json.keys())
    dataframe = pd.DataFrame.from_dict(response_json['Time Series (1min)'], orient='index').sort_index(axis=1)
    dataframe = dataframe.rename(columns={
        '1. open':'Open',
        '2. high':'High',
        '3. low':'Low',
        '4. close':'Close',
        '5. volume':'Volume'
    })
    dataframe = dataframe[['Open','High','Low','Close','Volume']]
    return dataframe

def fetchDaily(symbol,key):
    url = BASE_API_URL + 'function=TIME_SERIES_DAILY'+'&symbol='+symbol+'&outputsize=full'+'&apikey='+key
    response_json = requests.get(url).json()
    print(response_json.keys())
    print('NOTE: The full-length time series of 20+ years of historical data is being returned')
    dataframe = pd.DataFrame.from_dict(response_json['Time Series (Daily)'], orient='index').sort_index(axis=1)
    dataframe = dataframe.rename(columns={
        '1. open':'Open',
        '2. high':'High',
        '3. low':'Low',
        '4. close':'Close',
        '5. volume':'Volume'
    })
    dataframe = dataframe[['Open','High','Low','Close','Volume']]
    return dataframe

def fetchDailyAdjusted(symbol,key):
    url = BASE_API_URL + 'function=TIME_SERIES_DAILY_ADJUSTED'+'&symbol='+symbol+'&outputsize=full'+'&apikey='+key
    response_json = requests.get(url).json()
    print(response_json.keys())
    print('NOTE: The full-length time series of 20+ years of historical data is being returned')
    dataframe = pd.DataFrame.from_dict(response_json['Time Series (Daily)'], orient='index').sort_index(axis=1)
    dataframe = dataframe.rename(columns={
        '1. open':'Open',
        '2. high':'High',
        '3. low':'Low',
        '4. close':'Close',
        '5. volume':'Volume'
    })
    dataframe = dataframe[['Open','High','Low','Close','Volume']]
    return dataframe


def getQuoteEndpoint(symbol):
    url = BASE_API_URL + 'function=GLOBAL_QUOTE'+'&symbol='+symbol+'&outputsize=full'+'&apikey='+KEY
    response_json = requests.get(url).json()
    dataframe = pd.DataFrame.from_dict(response_json['Global Quote'], orient='index').sort_index(axis=1)
    return dataframe

########################## DATA FETCHING ####################################

