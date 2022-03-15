#%%
#TradeBot
import pandas as pd
import datetime
import os
clear = lambda: os.system('cls')
import threading
from time import sleep
from Driver import seltest
import getpass
from DLpart.PredictStock import StockListAnalysis,Technicals
from datetime import time
import yfinance as yf


def GiveSwingSuggestions():
    sla = StockListAnalysis()
    df = sla.SwingTrade()
    print(df)
    df = df.sort_values('PRICE',ascending=True)
    df.reset_index()
    return df

def GiveLongTermSuggestions():
    sla = StockListAnalysis()
    df = sla.LongTerm()
    df = df.sort_values("PRICE",ascending=True)
    df.reset_index()
    print(df)
    return df


def GreetingsAndDisclosure():
    hours = datetime.datetime.now().time().hour
    if hours < 12:
        print('\nGood Morning, Welcome to TradeBot\n')
        sleep(4)
        clear()
    elif hours == 12:
        print('\nGood Noon, Welcome to TradeBot\n')
        sleep(4)
        clear()
    elif hours > 12 and hours < 16:
        print('\nGood Afternoon, Welcome to TradeBot\n')
        sleep(4)
        clear()
    elif hours >= 16:
        print('\nGood Evening, Welcome to TradeBot\n')
        sleep(4)
        clear()
    else:
        print('\nWelcome to TradeBot')
        sleep(4)
        clear()

    #clear()
    print('\n----------------- * RISK DISCLOSURE *-----------------')
    print(' Investment/Trading in securities Market is subject to market risk,\n past performance is not a guarantee of future performance.\n The risk of loss in trading and investment in Securities markets\n including Equites, Derivatives, commodity and Currency can be substantial.\n These are leveraged products that carry a substantial risk of\n loss up to your invested capital and may not be suitable for everyone.\n You should therefore carefully consider whether such trading is suitable\n for you in light of your financial condition.\n Please ensure that you understand fully the risks involved and do invest\n money according to your risk bearing capacity.\n')
    print(' TradeBot does not guarantee any returns in any of its products or services.\n Investment in markets is subject to market risk.')
    print(' Hence, TradeBot is not liable for any losses in any case.\n All our services are nonrefundable.')
    print('------------------------------------------------------\n')
    response_to_risk_disclosure = str(input('Do you agree to the above terms and conditions? [y/n]: '))
    if response_to_risk_disclosure == 'y' or response_to_risk_disclosure == 'Y':
        clear()
        return True
    else:
        print('Please accept the terms and conditions in order to proceed')
        sleep(3)
        clear()
        exit(1)

def SwingOrLong():
    UpdateDisclaimer()
    sleep(3)
    #clear()
    Flag = str(input('\nDo you wish to invest for long term or do a swing trade? [long/swing]: '))  
    if (Flag == 'long' or Flag == 'LONG' or Flag == 'Long'):
        #get long term suggestions
        long_df = GiveLongTermSuggestions()
        return long_df, Flag
    elif (Flag == 'swing' or Flag == 'SWING' or Flag == 'Swing'):
        #get swing trade suggestions
        swing_df = GiveSwingSuggestions()        
        return swing_df, Flag

def Invest():
    df, decision = SwingOrLong()
    url , session_id, funds, login_obj = login()
    funds = funds-500 #keep minimum funds    
    if funds > 100:
        if decision == 'long':
            print('Investing the available balance of ' + '₹'+str(funds))
            #clear()
            UpdateDisclaimer()
            final_order = decideQty(df,funds = funds)
            print('\nThis is the order the system is going to place:\n',final_order)
            RiskToRewardDisclaimer()
            #place order
            if MarketHours():
                placeOrder(final_order=final_order,session_id=session_id,url=url,funds=funds,market_hours=True, longORSwing=decision)    
            else: #not market hour order
                placeOrder(final_order=final_order,session_id=session_id,url=url,funds=funds,market_hours=False, longORSwing=decision)
        elif decision == 'swing':
            print('Investing the available balance of ' + '₹'+str(funds))
            #clear()
            UpdateDisclaimer()
            final_order = decideQty(df,funds = funds)
            print('\nThis is the order the system is going to place:\n',final_order)
            RiskToRewardDisclaimer()
            #place order
            if MarketHours():
                placeOrder(final_order=final_order,session_id=session_id,url=url,funds=funds,market_hours=True, longORSwing=decision)    
            else: #not market hour order
                placeOrder(final_order=final_order,session_id=session_id,url=url,funds=funds,market_hours=False, longORSwing=decision)

    else:
        clear()
        print('\nFunds too low. Redirecting...')
        print('Restart TradeBot once funds are added.')
        login_obj.AddFunds(session_id = session_id, url = url)
        

def decideQty(suggestions, weightage = 0.5, funds=0):
    suggestions = suggestions.sort_values('PRICE', ascending = True)
    
    print('\n*** \nCurrent weightage for each stock is 50% of your funds. \nWe do not advise you to change this weightage.\n***')
    qtys = []
    try:
        for i in range(0,2):
            qty = int(round((funds*weightage)/suggestions['PRICE'][i]))
            qtys.append([suggestions['SYMBOL'][i],suggestions['PRICE'][i],str(qty),suggestions['LIMIT_PRICE'][i]])
    
    except ValueError:
        print('There is only one good suggestions at the moment, proceeding to place the order...')
        exit(0)
    except KeyError:
        print('There are no good suggestions at the moment, please try again after the next trading session.')
        exit(0)
    return pd.DataFrame(qtys, columns=['SYMBOL','PRICE','QTY','LIMIT_PRICE']) 

def MarketHours(): #returns true if order placed during market hours
    days = ['Monday','Tueday','Wednesday','Thursday','Friday','Saturday','Sunday']
    today = days[datetime.datetime.now().weekday()]
    start = time(hour=9,minute=15,second=0)
    end = time(hour=15,minute=30,second=0)
    if (datetime.datetime.now().hour > start.hour) and (datetime.datetime.now().hour < end.hour) and not((today == 'Saturday') or (today == 'Sunday')):
        return True
    else:
        return False



def RiskToRewardDisclaimer():
    clear()
    print('\n\n ****** \nThe Risk to Reward ratio for you is set to 1:2 by default\nthe order placed still has to be authorized by you\nif you wish to hold the shares longer\nyou can simply cancel the\norder placed by the system.\nAlthough, this is not advisable.\n******')
    
        
def placeOrder(final_order,session_id='',url='',funds=0,market_hours=False, longORSwing = ''):
    if longORSwing == 'long':
        try:
            for i in range(0,2):    
                trade = seltest.Trade(final_order['SYMBOL'][i],session_id=session_id,url=url).PlaceBuyOrder(qty=final_order['QTY'][i],limit_price=final_order['LIMIT_PRICE'][i],funds=funds,market_hours=market_hours,longTerm=True)
        except ValueError:
            print('There is only one good suggestions at the moment, proceeding to place the order...')
            exit(0)
        except KeyError:
            print('There are no good suggestions at the moment, please try again after the next trading session.')
            exit(0)
    elif longORSwing == 'swing':
        try:
            for i in range(0,2):
                try:    
                    trade = seltest.Trade(final_order['SYMBOL'][i],session_id=session_id,url=url).PlaceBuyOrder(qty=final_order['QTY'][i],limit_price=final_order['LIMIT_PRICE'][i],funds=funds,market_hours=market_hours,swingTrade=True)
                except IndexError:
                    tech = Technicals('^NSEI')
                    EMA_50 = tech.EMA(timeframe=50,plot=True,interval='1h')
                    if yf.Ticker('^NSEI').history(period='1d')['Close'][-1] < EMA_50:
                        print('No suggestions, market is bearish for now')
                break

                    
        except ValueError:
            print('There is only one good suggestions at the moment, proceeding to place the order...')
            exit(0)
        except KeyError:
            print('There are no good suggestions at the moment, please try again after the next trading session.')
            exit(0)        
        
    
            
    #clear()
   
    #print('Pending Orders: \n')
    #print(seltest.HandleOrders(session_id = session_id, url=url).GetOrders())



def UpdateDisclaimer():
    print('\nBefore we suggest we would like you to update the data, steps to update are written in README.txt')

def login():
    login = seltest.Login(str(input("Enter user's name: ")))
    url , session_id, funds = login.process_information()
    return url , session_id, funds, login

def main():
    clear()
    #checking if the user accepted risk disclosure
    if(GreetingsAndDisclosure()): #if accepted
        Invest() #start
    
        


 
if __name__ == '__main__':
    main()






# %%
