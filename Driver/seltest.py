#%%
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.common.keys import Keys
import datetime
from datetime import time
import Driver.keys as keys
#import keys
saif = keys.Keys_Saif
kanika = keys.Keys_Kanika


class Login:
    def __init__(self, username):
        self.HOME_PAGE = 'https://kite.zerodha.com/'
        self.DRIVER_PATH = 'Driver/chromedriver.exe'
        #self.DRIVER_PATH = 'chromedriver.exe'
        self.USERNAME = username

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
        options.add_experimental_option("detach", True)
        #options.add_argument("headless")
        #options.add_argument('--disable-gpu')
        driver = webdriver.Chrome(DRIVER_PATH, options=options)
        return driver

    


    def process_information(self):
        driver = self._init_driver(self.DRIVER_PATH)
        driver.get(self.HOME_PAGE)
        #getting elements
        username = driver.find_element_by_id('userid')
        password = driver.find_element_by_id('password')
        #sending keys
        if self.USERNAME == 'saif':
            username.send_keys(saif.saif_user)
            password.send_keys(saif.saif_password)  
        if self.USERNAME == 'kanika':
            username.send_keys(kanika.kanika_user)
            password.send_keys(kanika.kanika_password)
        #button press
        login_button = driver.find_element_by_xpath('//*[@id="container"]/div/div/div/form/div[4]/button')
        action = ActionChains(driver)
        action.click(on_element=login_button)
        action.perform()
        #entering pin 
        driver.implicitly_wait(1)
        pin = driver.find_element_by_css_selector('#pin') 
        
        if self.USERNAME == 'saif':
            pin.send_keys(saif.saif_pin)
        if self.USERNAME == 'kanika':
            pin.send_keys(kanika.kanika_pin)
        
        driver.implicitly_wait(1)
        driver.find_element_by_xpath('//*[@id="container"]/div/div/div/form/div[3]/button').click()
        #getting all data from the dashboard
        name = driver.find_element_by_xpath('//*[@id="app"]/div[2]/div/div/div/h1/span')
        PL = driver.find_element_by_xpath('//*[@id="app"]/div[2]/div/div/div/div[2]/div[1]/div[2]/div[1]/div[1]/span[1]')
        print(name.text,"'s Account")
        print(self.checkPL(PL.text))
        #click on holdings
        driver.find_element_by_xpath('//*[@id="app"]/div[1]/div/div[2]/div[1]/a[3]/span').click()
        total_investment_value = float(str(driver.find_element_by_xpath('//*[@id="app"]/div[2]/div[2]/div/div/div/div[1]/div[4]/h1/span[1]').text).replace(',',''))
        current_investment_value = float(str(driver.find_element_by_xpath('//*[@id="app"]/div[2]/div[2]/div/div/div/div[1]/div[2]/h1').text).replace(',',''))
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
            try:
                if j > 8:
                    holdings[i].pop(1)
                    print("\n"+holdings[i][0] +"\n"+ "Pre-Trade qty:" + holdings[i][1])
                    print("Tradeable qty:" + holdings[i][2])
                    print('Average Buying Price: ' + holdings[i][3])
                    print("Current value invested: " , float(str(holdings[i][5]).replace(',','')))
                if j == 8:
                    print("\n" + holdings[i][0] + "\n" + "Tradeable qty: " + holdings[i][1])
                    print('Average Buying Price: ' + holdings[i][2])
                    print("Current value invested: " , float(str(holdings[i][4]).replace(',','')))
            except ValueError:
                continue
        #Click on funds, get funds
        WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="app"]/div[1]/div/div[2]/div[1]/a[5]/span'))).click()
        available_funds = driver.find_element_by_xpath('/html/body/div[1]/div[2]/div[2]/div/div/div[2]/div[1]/div/table/tbody/tr[1]/td[2]/h1')
        print("\nAvailable balance: " , available_funds.text) 
        
        url = driver.command_executor._url
        session_id = driver.session_id

        return url, session_id, float(str(available_funds.text).replace(',',''))

    

    def AddFunds(self,session_id='',url=''):
        driver = webdriver.Remote(command_executor = url, desired_capabilities={})
        driver.close()
        driver.session_id = session_id
        driver.find_element_by_xpath('//*[@id="app"]/div[1]/div/div[2]/div[1]/a[5]').click()


class Trade:
    def __init__(self, symbol, session_id, url):
        self.symbol = symbol
        self.session_id = session_id
        self.url = url
    

    def PlaceBuyOrder(self, qty = 0,limit_price = 0 , funds = 0, market_hours = False, stop_loss=-5, target = 10, longTerm = False, swingTrade = False):
        driver = webdriver.Remote(command_executor = self.url)
        driver.close()
        driver.session_id = self.session_id
        actions = ActionChains(driver)
        #1.Search for the stock
        driver.implicitly_wait(1)
        driver.find_element_by_xpath('/html/body/div[1]/div[2]/div[1]/div/div[1]/div/div/input').clear()
        driver.find_element_by_xpath('/html/body/div[1]/div[2]/div[1]/div/div[1]/div/div/input').send_keys(self.symbol)

        #2.get the result list
        search_result = driver.find_element_by_xpath('//*[@id="app"]/div[2]/div[1]/div/div[1]/ul')

        #3.searching the list and selecting the 0th element
        rows = search_result.find_elements(By.TAG_NAME, "li")
        hover_over_first_element_of_search = actions.move_to_element(rows[0])
        hover_over_first_element_of_search.perform()

        #4. get market depth
        driver.find_element_by_xpath('//*[@id="app"]/div[2]/div[1]/div/div[1]/ul/div/li[1]/span[3]/button[4]').click()
        if swingTrade:
            #5. check market hours
            if market_hours:
                print('\nmarket hours, limit order will be placed.')
                
                #click buy on the market depth
                driver.find_element_by_xpath('//*[@id="app"]/div[5]/div/div/div[3]/div/div/div[2]/button[1]').click()
                
                #select CNC order type
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[1]/div/div[2]/label').click()

                #enter qty
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[2]/div[1]/div[1]/div/input').clear()
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[2]/div[1]/div[1]/div/input').send_keys(str(qty))

                #enter limit price
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[2]/div[2]/div[2]/div/div[2]/label').click() #click on limit radio
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[2]/div[1]/div[2]/div/input').clear()
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[2]/div[1]/div[2]/div/input').send_keys(str(limit_price)) #enter price

                #set stop loss and target
                #stop loss
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[3]/div/div[2]/div[1]/label/span[1]').click()
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[3]/div/div[2]/div[2]/div/input').clear()
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[3]/div/div[2]/div[2]/div/input').send_keys(str(stop_loss))
                #target
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[3]/div/div[3]/div[1]/label/span[2]').click()
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[3]/div/div[3]/div[2]/div/input').clear()
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[3]/div/div[3]/div[2]/div/input').send_keys(str(target))

                #final buy click
                driver.find_element_by_xpath('//*[@id="app"]/form/section/footer/div/div[2]/button[1]').click()

                #close
                driver.find_element_by_xpath('//*[@id="app"]/div[5]/div/div/div[3]/div/div/div[2]/button[3]').click()

                

            else:
                print('\nNon market hours, an AMO(After market Order) will be placed.')
                #click on buy on market depth
                driver.find_element_by_xpath('//*[@id="app"]/div[5]/div/div/div[3]/div/div/div[2]/button[1]').click()

                #click on AMO label
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[1]/div/div[3]/label').click()

                #click on CNC product type
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[1]/div/div[2]/label').click()

                #set QTY
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[2]/div[1]/div[1]/div/input').clear()
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[2]/div[1]/div[1]/div/input').send_keys(str(qty))

                #set price to MKT
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[2]/div[2]/div[2]/div/div[1]/label').click()

                print('\nOrder will be placed on the next trading day, you can modify the order from the "Order" tab.')

                #final click 
                driver.find_element_by_xpath('//*[@id="app"]/form/section/footer/div/div[2]/button[1]').click()

                #closing the market depth
                driver.find_element_by_xpath('//*[@id="app"]/div[5]/div/div/div[3]/div/div/div[2]/button[3]').click()

                #going to order tab
                driver.find_element_by_xpath('//*[@id="app"]/div[1]/div/div[2]/div[1]/a[2]').click()
                
                #getting pending order table
                order_table = driver.find_element_by_xpath('//*[@id="app"]/div[2]/div[2]/div[2]/div/section[1]/div/div/table')
                rows = order_table.find_elements(By.TAG_NAME,"tr")
                #print(rows[1].text)
                action_2 = ActionChains(driver)
                action_2.move_to_element(rows[1]).perform()
                driver.implicitly_wait(2)
                
                #click on context menu of the first order
                driver.find_element_by_xpath('/html/body/div[1]/div[2]/div[2]/div[2]/div/section[1]/div/div/table/tbody/tr/td[4]/div/div/span').click()
                driver.implicitly_wait(2)

                #get menu
                list_of_order_options = driver.find_element_by_xpath('/html/body/div[1]/div[2]/div[2]/div[2]/div/section[1]/div/div/table/tbody/tr/td[4]/div/ul')
                items = list_of_order_options.find_elements(By.TAG_NAME,"li")
                items[6].click() #creating a stop loss and target GTT

                #GTT menu
                driver.find_element_by_xpath('//*[@id="app"]/div[4]/div/div/div/div[2]/div/div[1]/div/div[1]/div[2]/div/div[2]/label').click() #click on sell
                driver.find_element_by_xpath('//*[@id="app"]/div[4]/div/div/div/div[2]/div/div[1]/div/div[2]/div[2]/div[1]/div[2]/label').click() #click on OCO

                #setting stop loss
                driver.find_element_by_xpath('//*[@id="app"]/div[4]/div/div/div/div[2]/div/div[2]/form/div[1]/div[1]/div[3]/div/input').clear()
                driver.find_element_by_xpath('//*[@id="app"]/div[4]/div/div/div/div[2]/div/div[2]/form/div[1]/div[1]/div[3]/div/input').send_keys(str(stop_loss)) #sending stop_loss percentage
                driver.implicitly_wait(1)
                price = str(driver.find_element_by_xpath('//*[@id="app"]/div[4]/div/div/div/div[2]/div/div[2]/form/div[1]/div[1]/div[2]/div/input').text) #getting the stop loss price
                driver.find_element_by_xpath('//*[@id="app"]/div[4]/div/div/div/div[2]/div/div[2]/form/div[1]/div[3]/div[2]/div[1]/div/input').clear()
                driver.find_element_by_xpath('//*[@id="app"]/div[4]/div/div/div/div[2]/div/div[2]/form/div[1]/div[3]/div[2]/div[1]/div/input').send_keys(str(qty)) #sending qty
                driver.find_element_by_xpath('//*[@id="app"]/div[4]/div/div/div/div[2]/div/div[2]/form/div[1]/div[3]/div[2]/div[2]/div/input').send_keys(price)

                #setting target
                driver.find_element_by_xpath('//*[@id="app"]/div[4]/div/div/div/div[2]/div/div[2]/form/div[2]/div[1]/div[3]/div/input').clear()
                driver.find_element_by_xpath('//*[@id="app"]/div[4]/div/div/div/div[2]/div/div[2]/form/div[2]/div[1]/div[3]/div/input').send_keys(str(target))
                LTP = str(driver.find_element_by_xpath('//*[@id="app"]/div[4]/div/div/div/div[1]/div/div/div[1]/span[3]').text).replace(',','' )
                target_price = round(float(LTP)*(target/100) + float(LTP),ndigits=1)
                driver.find_element_by_xpath('//*[@id="app"]/div[4]/div/div/div/div[2]/div/div[2]/form/div[2]/div[3]/div[2]/div[1]/div/input').clear()
                driver.find_element_by_xpath('//*[@id="app"]/div[4]/div/div/div/div[2]/div/div[2]/form/div[2]/div[3]/div[2]/div[1]/div/input').send_keys(str(qty))
                driver.find_element_by_xpath('/html/body/div[1]/div[4]/div/div/div/div[2]/div/div[2]/form/div[2]/div[3]/div[2]/div[2]/div/input').clear()
                driver.find_element_by_xpath('/html/body/div[1]/div[4]/div/div/div/div[2]/div/div[2]/form/div[2]/div[3]/div[2]/div[2]/div/input').send_keys(str(target_price))

                #placing final GTT order 
                print('\nStop loss and Target orders set via GTT, you will have to authorize the orders if either of them are hit.')
                driver.find_element_by_xpath('//*[@id="app"]/div[4]/div/div/div/div[3]/div/div/div[2]/button[1]').click()

                #get back to dash
                #driver.find_element_by_xpath('//*[@id="app"]/div[1]/div/div[2]/div[1]/a[1]').click()
        elif longTerm:
            if market_hours:
                print('\nmarket hours, limit order will be placed.')
                #click buy on the market depth
                driver.find_element_by_xpath('//*[@id="app"]/div[5]/div/div/div[3]/div/div/div[2]/button[1]').click()
                #select CNC order type
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[1]/div/div[2]/label').click()
                #enter qty
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[2]/div[1]/div[1]/div/input').clear()
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[2]/div[1]/div[1]/div/input').send_keys(str(qty))
                #enter limit price
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[2]/div[2]/div[2]/div/div[2]/label').click() #click on limit radio
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[2]/div[1]/div[2]/div/input').clear()
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[2]/div[1]/div[2]/div/input').send_keys(str(limit_price)) #enter price
                #final buy click
                driver.find_element_by_xpath('//*[@id="app"]/form/section/footer/div/div[2]/button[1]').click()
                #close
                driver.find_element_by_xpath('//*[@id="app"]/div[5]/div/div/div[3]/div/div/div[2]/button[3]').click()
            else:
                print('\nNon market hours, an AMO(After market Order) will be placed.')
                #click on buy on market depth
                driver.find_element_by_xpath('//*[@id="app"]/div[5]/div/div/div[3]/div/div/div[2]/button[1]').click()
                #click on AMO label
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[1]/div/div[3]/label').click()
                #click on CNC product type
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[1]/div/div[2]/label').click()
                #set QTY
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[2]/div[1]/div[1]/div/input').clear()
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[2]/div[1]/div[1]/div/input').send_keys(str(qty))
                #set price to MKT
                driver.find_element_by_xpath('//*[@id="app"]/form/section/div[2]/div[2]/div[2]/div[2]/div/div[1]/label').click()
                print('\nOrder will be placed on the next trading day, you can modify the order from the "Order" tab.')
                #final click 
                driver.find_element_by_xpath('//*[@id="app"]/form/section/footer/div/div[2]/button[1]').click()
                #closing the market depth
                driver.find_element_by_xpath('//*[@id="app"]/div[5]/div/div/div[3]/div/div/div[2]/button[3]').click()
        else:
            print('No trade type selected.')
            driver.close()



    

        
                
        

class HandlePositions:
    def __init__(self, session_id, url):
        self.session_id = session_id
        self.url = url

    def GetPosition(self):
        driver = webdriver.Remote(command_executor = self.url, desired_capabilities={})
        driver.close()
        driver.session_id = self.session_id
        actions = ActionChains(driver)

        #click position
        driver.find_element_by_xpath('//*[@id="app"]/div[1]/div/div[2]/div[1]/a[4]').click()

        #accessing positions table
        positions = []
        position_table = driver.find_element_by_xpath('//*[@id="app"]/div[2]/div[2]/div/div/section[1]/div/div/table')
        rows = position_table.find_elements(By.TAG_NAME,"tr")
        for i in range(1,len(rows)):
            #print(rows[i].text + "\n")
            positions.append(str(rows[i].text).replace('\n',' ').split(sep=' '))

        #preparing positions data
        for i in range(len(positions)):
            if len(positions[i]) > 8:
                positions[i].remove("HOLDING")
                #print(len(positions[i]))
            for j in range(len(positions[i])):                      
                if j == 7:
                    positions[i][j] = str(positions[i][j]).replace('%','')
        

        #omitting the last since it shows only total
        positions.pop()

        #adding a status of trade based 
        for i in positions:
            for j in range(len(i)):
                if j == 3:
                    if i[j]=='0':
                        i.append('Inactive')
                        break
                    else:
                        i.append('Active')
                        break
                
        #adding a status of trade based 
        for i in positions:
            for j in range(len(i)):
                if j == 6:
                    if str(i[j]).startswith('+'):
                        i.append('Profit')
                        break
                    elif str(i[j]).startswith('-'):
                        i.append('Loss')
                        break
                    elif i[j] == '0.0':
                        i.append('NetLoss')  #since tax not counted
                        break
        
        columns=['Order Type','Instrument','Exchange','Qty','Avg','LTP','P&L','Change','Status','Outcome']
        positions_dataframe = pd.DataFrame(positions,columns=columns)
        
        
        print(positions_dataframe.head())
        return positions_dataframe


class HandleOrders:
    def __init__(self,session_id,url):
        self.session_id = session_id
        self.url = url
    def GetOrders(self):
        driver = webdriver.Remote(command_executor = self.url, desired_capabilities={})
        driver.close()
        driver.session_id = self.session_id
        actions = ActionChains(driver)

        #click on orders
        driver.find_element_by_xpath('//*[@id="app"]/div[1]/div/div[2]/div[1]/a[2]').click()
        #get open Orders
        open_orders = []
        #try:
        #get executed orders
        #finding the div with class pending orders
        
        pending_exists = 0
        try:
            driver.find_element_by_xpath('//*[@id="app"]/div[2]/div[2]/div[2]/div/section[1]/div')
            pending_exists = 1
        except:
            print('No pending order')
            pending_exists = 0

        
        if pending_exists == 1:
            #print('Orders Exists')
            pending_order_table = driver.find_element_by_xpath('//*[@id="app"]/div[2]/div[2]/div[2]/div/section[1]/div/div/table')
            rows = pending_order_table.find_elements(By.TAG_NAME,"tr")
            for i in range(1,len(rows)):
                #print(rows[i].text)
                row = str(rows[i].text).replace('\n', ' ').split(sep=' ')
                row[5] = row[5]+row[6]+row[7]
                row.pop(6)
                row.pop(6)
                open_orders.append(row)

        PENDING_ORDERS_DATAFRAME = pd.DataFrame(open_orders,columns=['Time','Type','Instrument','Exchange','Product','Qty','Avg.Price','Status'])
        print(PENDING_ORDERS_DATAFRAME.head())
        
               
            


class OptionTrading:
    def __init__(self):  
        self.HOME_PAGE = 'https://www.nseindia.com/option-chain'
        self.DRIVER_PATH = 'chromedriver.exe'
    
    def _init_driver(self, DRIVER_PATH):
        options = webdriver.ChromeOptions()
        options.add_argument("--ignore-certificate-errors")
        #options.add_experimental_option("detach", True) #remove when done testing
        #options.add_argument("--incognito")
        #options.add_argument("headless")
        driver = webdriver.Chrome(DRIVER_PATH, options=options)
        return driver
    
    def getOptionChain(self,view_options_contract_for='NIFTY',symbol='',expiry_date='',strike_price=0): #,view_options_contract_for,symbol,expiry_date,strike_price
        #view_options_contract_for = 'NIFTY' or 'BANKNIFTY' or 'FINNIFTY'
        driver = self._init_driver(self.DRIVER_PATH)
        driver.get(self.HOME_PAGE)

        

    
#login = Login(str(input("Enter user's name: ")))
#url , session_id, funds = login.process_information()

#trade = Trade('TATAMOTORS',session_id=session_id,url=url)
#symbol = ['NMDC','TATAMOTORS','TATAPOWER','PNB']
#for i in range(0,len(symbol)):
#    Trade(symbol[i],session_id=session_id,url=url).PlaceBuyOrder(qty=1, funds=funds, market_hours=False, swingTrade=True)
# trade.BuyStock(session_id = session_id, url = url,qty=100,funds=funds,market_hours=False)
#handleOrders = HandleOrders(session_id = session_id, url = url)
#handleOrders.GetOrders()

#handlePositions = HandlePositions(session_id = session_id, url = url)
#df = handlePositions.GetPosition()



# print('Press ENTER to exit...')
# input()





# %%



# %%
