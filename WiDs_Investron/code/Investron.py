from tvDatafeed import TvDatafeed, Interval
import pandas as pd
def zero_list() :
    a = []
    for i in range(1000):
        a.append(0)
    return a

    
username = 'Hriske'
password = 'Mavily123*4'
tv = TvDatafeed(username, password)

symbol ="HINDALCO" 
nifty_index_data = tv.get_hist(symbol,exchange='NSE',interval=Interval.in_daily,n_bars=1000)
l = {'MACD' : zero_list() , 'RSI' : zero_list() ,'positions' : zero_list()}
a =nifty_index_data.assign(**l)
# MACD Calculation
b = a
b.reset_index(drop=True, inplace=True)
short_ema = b['close'].ewm(span=12, adjust=False).mean()  # 12-day EMA # here the first EMA is set to first data point so the first row is 0
long_ema = b['close'].ewm(span=26, adjust=False).mean()  # 26-day EMA
a['MACD'] = short_ema - long_ema

# RSI calculation
data = { 'Change' : zero_list(), 'Gain' : zero_list(), 'Loss' : zero_list(), 'Avg Gain': zero_list(), 'Avg Loss' : zero_list() }
df = pd.DataFrame(data)
df['Change'] = a['close'].diff()

# Separate gains and losses
df['Gain'] = df['Change'].where(df['Change'] > 0, 0)
df['Loss'] = -df['Change'].where(df['Change'] < 0, 0)

# Calculate average gain and average loss for a 3-day period
period = 14
df['Avg Gain'] = df['Gain'].rolling(window=period, min_periods=1).mean()
df['Avg Loss'] = df['Loss'].rolling(window=period, min_periods=1).mean()
a['RSI'] = 100*(1/((df['Avg Loss'])/df['Avg Gain']+1))

# positions
flag = False
buy = 0
sell = 0 
profit = 0
for i in range(len(a)):
    if ((a.loc[i, 'MACD'] > 0)) and flag == False:
        a.loc[i,'positions'] = 1
        buy = a.loc[i,'open']
        flag = True 
    elif ((a.loc[i, 'MACD'] < 0) and flag == True) :
        a.loc[i,'positions'] = -1
        sell = a.loc[i,'close']
        profit = profit + (sell - buy)
        flag = False
    else:
        a.loc[i,'positions'] = 0
a.loc[1,'profit'] = profit
a.to_csv( symbol + ".csv",index = False)
print(profit)
print(a)



