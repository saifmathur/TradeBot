# %%
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt

KEY = 'RM2J9YHI19A162UW'
SYMBOL = 'TTM'
INTERVAL = '1min'
OUTPUT_FORMAT = 'pandas' #or json
 
def fetch_data_for_one(key, symbol, interval, output_format):
    ts = TimeSeries(key,output_format= output_format)
    data, metadata = ts.get_intraday('NSE:'+symbol,interval,outputsize='full')
    data = data.rename(columns={
        '1. open':'Open',
        '2. high':'High',
        '3. low':'Low',
        '4. close':'Close',
        '5. volume':'Volume'
    })
    print('data for ' + metadata['2. Symbol'])
    print(data.head() + '\n')
    plt.plot(data['Open'])
    plt.show()
    return data, metadata


df, metadata = fetch_data_for_one(KEY,SYMBOL, INTERVAL,OUTPUT_FORMAT)


# %%
from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
KEY = 'RM2J9YHI19A162UW'
SYMBOL = ['AAPL','IBM','AMD','AMZN','TSLA']
INTERVAL = '1min'
OUTPUT_FORMAT = 'pandas' #or json

def fetch_data_alphav(key, symbol, interval, output_format):
    dfArray = []
    ts = TimeSeries(key,output_format= output_format)
    for i in range(len(symbol)):
        data, metadata = ts.get_intraday(symbol[i],interval,outputsize='full')
        data = data.rename(columns={
        '1. open':'Open',
        '2. high':'High',
        '3. low':'Low',
        '4. close':'Close',
        '5. volume':'Volume'
        })
        plt.title('Plot for ' +symbol[i])
        plt.plot(data['Close'][i])
        plt.show()    

    print('\n')
    print('Displaying data for the following: ')
    for i in range(len(symbol)):
        print(symbol[i])
    print('\n data for ' + metadata['2. Symbol'])
    print(data.head())
    print('\n')

    

fetch_data_alphav(KEY,SYMBOL, INTERVAL,OUTPUT_FORMAT)

# %%
