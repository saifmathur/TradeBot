
# def calculateRiskCapacity(amount=0):
#     max_loss = 5
#     target = 10
#     print('\nBased on your investment amount...')
#     print('\nMonthly investment: ', amount, 'INR')
#     print('Yearly investment: ', amount*12, 'INR')
#     print('Risk to reward ratio for everyone is set to 1:2, that is for every investment\nyou can accept a 5% loss and a 10% profit.')
#     response = input('Do you wish to modify these levels? [y/n]: ')
#     if response == 'y' or response == 'Y':
#         modified_max_loss = float(input('\nEnter loss percentage: '))
#         modified_max_profit = float(input('\nEnter target percentage: '))
#         return modified_max_loss,modified_max_profit
#     else:   
#         return max_loss,target


# def Questionare():
#     #print menu
#     clear()
#     print('\t \t \t \t \t \t \t \t', 'Date: ',datetime.datetime.now().date())
#     #print('\nGreat, Welcome to TradeBot!\nPlease enter the following to the best of your knowledge.\n') 
#     print('\nGreat, Welcome to TradeBot!\nPlease enter the following details in order to proceed.\n') 
#     NAME = input('\nPlease enter a local username(this username is to recognise you on your system locally): ')
#     USER_ID = getpass.getpass('\nPlease Enter user Id for zerodha: ')
#     USER_ID = USER_ID.upper()
#     PASSWORD = getpass.getpass('\nEnter password for zerodha account: ')
#     PIN = getpass.getpass('\nEnter your 6 digit PIN: ')
#     try:
#         if len(PIN)!=6:
#             raise Exception('PIN length not of 6 digits. Please try again')
#     except Exception as e:
#         print(e)
#         sleep(2)
#         exit(1)

#     monthly_investment = float(input('\nEnter the amount you would like to invest every month: '))
#     clear()
#     print('\n------- *Risk Capacity* -------')
#     risk, reward = calculateRiskCapacity(monthly_investment)
#     print('-------------------------------')
#     try:
#         f = open('./Driver/Keys.py',"a+")
#         name = NAME
#         user = USER_ID
#         password = PASSWORD
#         pin = PIN
#         #f = open('fileTest.py',"a+")
#         f.write("class Keys_"+name+":"+"\n\t"+name+"_user = "+"'"+user+"'"+"\n\t"+name+"_password = "+"'"+password+"'"+"\n\t"+name+"_pin = "+"'"+pin+"'"+"\n\t"+name+"_investment_monthly = "+str(monthly_investment))
#         f.close()
#         print('Local profile created')
#     except Exception as e:
#         print('Local profile not created ', e)
#         sleep(1)
#         exit(1)
#     #creating a local user profile

# def CheckUser():
#     clear()
#     Flag = input('\nAre you using TradeBot for the first time? [y/n]: ')
#     if Flag == 'y' or Flag == 'Y':
#         Questionare()
#     elif Flag == 'n' or Flag == 'N':
#         existingUser()


# login = Login(str(input("Enter user's name: ")))
# url , session_id, funds = login.process_information()

# trade = Trade('TATAMOTORS')
# trade.BuyStock(session_id = session_id, url = url,qty=0,funds=funds, weightage_of_trade=10)

#handleOrders = HandleOrders(session_id = session_id, url = url)
#handleOrders.GetOrders()

#handlePositions = HandlePositions(session_id = session_id, url = url)
#df = handlePositions.GetPosition()



# print('Press ENTER to exit...')
# input()


