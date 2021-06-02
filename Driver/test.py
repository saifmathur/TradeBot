#%%

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import yfinance as yf
class Trade:
    def __init__(self):
        self.HOME_PAGE = 'https://kite.zerodha.com/'
        self.NSE_PAGE = 'https://www.nseindia.com/market-data/live-equity-market?symbol=NIFTY%2050'
        self.DRIVER_PATH = 'chromedriver.exe'
    def StockSelectionForLongTerm(self,exchange = 'NSE'):
        options = webdriver.ChromeOptions()
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--incognito")
        options.add_argument('user_agent={0}'.format(user_agent))
        #options.add_argument("headless")
        
        driver = webdriver.Chrome(self.DRIVER_PATH, options=options)
        driver.get(self.NSE_PAGE)
        driver.find_element_by_xpath('//*[@id="equity-stock"]/div[2]/div/div[3]/div/ul/li/a').click()
        

        try:
            if exchange == 'NSE':
                print()
                
            else:
                raise Exception('TradeException')
        except:
            print('Trades in BSE are not recommended by this software, due to volatility.')


    

obj = Trade()
obj.StockSelectionForLongTerm()
# %%
