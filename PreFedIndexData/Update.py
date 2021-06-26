#%%
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

class Update:
    def fetchAndUpdate(self):    
        URL = "https://www.nseindia.com/market-data/live-equity-market?symbol=NIFTY%20100"
        driver = webdriver.Chrome(executable_path='../Driver/chromedriver.exe')
        driver.get(URL)
        driver.delete_all_cookies()
        table = driver.find_element_by_xpath('//*[@id="equityStockTable"]')
        for i in table:
            print(i.text)
        table_id = driver.find_element_by_xpath('//*[@id="app"]/div[2]/div[2]/div/div/section/div/div/table')
        rows = table_id.find_elements(By.TAG_NAME, "tr")
a = Update()
a.fetchAndUpdate()



# %%
