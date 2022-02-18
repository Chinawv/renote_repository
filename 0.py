import pandas as pd
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
gold=pd.read_csv(r'C:\Users\Lenovo\Desktop\2022_Problem_C_DATA\LBMA-GOLD.csv')
bitcoin=pd.read_csv(r'C:\Users\Lenovo\Desktop\2022_Problem_C_DATA\BCHAIN-MKPRU.csv')
gold_date=gold['Date'].str.split('/',expand=True)
gold_all=pd.concat([gold,gold_date],axis=1)

#gold_all['date']=gold_all
bitcoin_date=bitcoin['Date'].str.split('/',expand=True)
bitcoin_all=pd.concat([bitcoin,bitcoin_date],axis=1)

gold_all['Date']='20'
gold_all['Date']=gold_all['Date'].str.cat(gold_all.iloc[:,4].astype('str'))
gold_all['Date']=gold_all['Date'].str.cat(gold_all.iloc[:,2].astype('str'),sep='-')
gold_all['Date']=gold_all['Date'].str.cat(gold_all.iloc[:,3].astype('str'),sep='-')
gold_all.columns=['Date','data','month','date','year']
gold_all=gold_all.drop(['month','date','year'],axis=1)
gold=gold.set_index(['Date'])
gold_all.to_csv(r"C:\Users\Lenovo\Desktop\2022_Problem_C_DATA\gold.csv")
#print(gold_all)
bitcoin_all['Date']='20'
bitcoin_all['Date']=bitcoin_all['Date'].str.cat(bitcoin_all.iloc[:,4].astype('str'))
bitcoin_all['Date']=bitcoin_all['Date'].str.cat(bitcoin_all.iloc[:,2].astype('str'),sep='-')
bitcoin_all['Date']=bitcoin_all['Date'].str.cat(bitcoin_all.iloc[:,3].astype('str'),sep='-')
bitcoin_all.columns=['Date','data','month','date','year']
bitcoin_all=bitcoin_all.drop(['month','date','year'],axis=1)
bitcoin_all.to_csv(r"C:\Users\Lenovo\Desktop\2022_Problem_C_DATA\bitcoin.csv")
#print(bitcoin_all)
gold=pd.read_csv(r'C:\Users\Lenovo\Desktop\2022_Problem_C_DATA\gold.csv')
print(gold)