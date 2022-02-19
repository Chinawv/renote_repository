import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as mpf
import numpy as np
import talib
# 五日 十日 三十日均线
gold=pd.read_csv(r'.\data\LBMA-GOLD.csv')
gold.columns=['Date','data']
gold=gold.set_index(['Date'])
gold_train=gold.iloc[:1000,:]
gold_test=gold.iloc[1000:,:]
bitcoin=pd.read_csv(r'.\data\BCHAIN-MKPRU.csv')
bitcoin.columns=['Date','data']
#bitcoin=bitcoin.drop(['Date'],axis=1)
bitcoin.iloc[:,0]=pd.to_datetime(bitcoin.iloc[:,0])
bitcoin=bitcoin.set_index(['Date'])
bitcoin['sam5']=bitcoin['data'].rolling(5).mean()

bitcoin['sam10']=bitcoin['data'].rolling(10).mean()

bitcoin['sam30']=bitcoin['data'].rolling(30).mean()
bitcoin['dif']=bitcoin['data'].diff()
plt.figure()
plt.plot(bitcoin['data'].iloc[0:300],label='data')
plt.plot(bitcoin['sam5'].iloc[0:300],label='san5')
plt.plot(bitcoin['sam10'].iloc[0:300],label='sam10')
plt.plot(bitcoin['sam30'].iloc[0:300],label='sam30')


plt.legend()
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()
