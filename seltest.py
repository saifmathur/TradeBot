#%%
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

class Login:
    def __init__(self):
        self.HOME_PAGE = 'https://kite.zerodha.com/'
        self.DRIVER_PATH = 'chromedriver.exe'

    def checkPL(self,string):
        for i in string:
            if i == '-':
                return 'Overall loss'
                break
            else:
                return 'Overall profit'
                break


    def _init_driver(self, DRIVER_PATH):
        options = webdriver.ChromeOptions()
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--incognito")
        #options.add_argument("headless")
        driver = webdriver.Chrome(DRIVER_PATH, options=options)
        return driver

    def process_information(self):
        driver = self._init_driver(self.DRIVER_PATH)
        driver.get(self.HOME_PAGE)
        #getting elements
        username = driver.find_element_by_id('userid')
        password = driver.find_element_by_id('password')
        #sending keys
        # username.send_keys('WP6817')
        # password.send_keys('Lockdown2020')
        username.send_keys()
        password.send_keys()
        #button press
        login_button = driver.find_element_by_xpath('//*[@id="container"]/div/div/div/form/div[4]/button')
        action = ActionChains(driver)
        action.click(on_element=login_button)
        action.perform()
        #entering pin 
        driver.implicitly_wait(1)
        pin = driver.find_element_by_css_selector('#pin') 
        pin.send_keys('361967')
        driver.implicitly_wait(1)
        cont_but = driver.find_element_by_xpath('//*[@id="container"]/div/div/div/form/div[3]/button')
        action.click(on_element=cont_but)
        action.perform()
        #getting all data from the dashboard
        name = driver.find_element_by_xpath('//*[@id="app"]/div[2]/div/div/div/h1/span')
        PL = driver.find_element_by_xpath('//*[@id="app"]/div[2]/div/div/div/div[2]/div[1]/div[2]/div[1]/div[1]/span[1]')
        print(name.text,"'s Account")
        print(self.checkPL(PL.text))
        #click on holdings
        WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="app"]/div[1]/div/div[2]/div[1]/a[3]/span'))).click()
        driver.implicitly_wait(1)
        total_investment_value = float(str(driver.find_element_by_xpath('//*[@id="app"]/div[2]/div/div/div/div[1]/div[1]/h1').text).replace(',',''))
        current_investment_value = float(str(driver.find_element_by_xpath('//*[@id="app"]/div[2]/div[2]/div/div/div[1]/div[2]/h1').text).replace(',',''))
        print('Total Investment: ',total_investment_value)
        print('Current value of Investment: ', current_investment_value)
        #P&L calculation
        if current_investment_value > total_investment_value:
            print('Total Profit: ' , '+',float(current_investment_value - total_investment_value))
        else:
            print('Total Loss: ' , '-',float(total_investment_value - current_investment_value))
        
        #getting holdings
        holdings = []
        table_id = driver.find_element_by_xpath('//*[@id="app"]/div[2]/div[2]/div/div/section/div/div/table')
        rows = table_id.find_elements(By.TAG_NAME, "tr")
        for i in rows:
            holdings.append(str(i.text).replace('\n',' ').split(sep=' '))
        
        #listing out pre-trade qty and tradeable qty
        holdings.pop(0)
        for i in range(len(holdings)):
            for j in range(len(holdings[i])):
                j += 1
            if j > 8:
                holdings[i].pop(1)
                print("\n"+holdings[i][0] +"\n"+ "Pre-Trade qty:" + holdings[i][1])
                print("Tradeable qty:" + holdings[i][2] + "\n")

            #print(holdings[i])

        
            
            
            
        





        

        #driver.close()

obj = Login()
obj.process_information()


#%%
total = 89
current = 120
print(total-current)




# %%
