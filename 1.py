#arima
import pandas as pd
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
gold=pd.read_csv(r'C:\Users\Lenovo\Desktop\2022_Problem_C_DATA\gold.csv')
gold.columns=['1','Date','data']
gold=gold.drop(['1'],axis=1)
gold=gold.set_index(['Date'])
gold_train=gold.iloc[:1000,:]
gold_test=gold.iloc[1000:,:]
bitcoin=pd.read_csv(r'C:\Users\Lenovo\Desktop\2022_Problem_C_DATA\bitcoin.csv')
bitcoin.columns=['1','Date','data']
bitcoin=bitcoin.drop(['1'],axis=1)
bitcoin.iloc[:,0]=pd.to_datetime(bitcoin.iloc[:,0])
bitcoin=bitcoin.set_index(['Date'])
bitcoin_train=bitcoin.iloc[:1000,:]
bitcoin_test=bitcoin.iloc[1000:,:]
print(bitcoin_train)
print(bitcoin_test)
alpha_gold=0.01
alpha_bitcoin=0.02
gold_diff=gold.diff()
bitcoin_diff=bitcoin.diff()
gold_diff.columns=['gold_diff']
bitcoin_diff.columns=['bitcoin_diff']
gold_diff.plot()
bitcoin_diff.plot()
plt.show()
p=d=q=range(0,5)
pdq=list(itertools.product(p,d,q))
#print(p,d,q)
seasonal_pdq=[(x[0],x[1],x[2])for x in list(itertools.product(p,d,q))]
#print(seasonal_pdq)
minAIC=100000
"""
for param in pdq:
        try:
            model = sm.tsa.statespace.SARIMAX(bitcoin_train, order=param,  enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit()
            print('ARIMA{}x - AIC:{}'.format(param, results.aic))
            #print('ARIMA{}x - AIC:{}'.format(param, results.aic))
            if(results.aic<minAIC):
                minAIC=results.aic
                bestparam=param
                #bestparam_seasonal=param_seasonal
        except:
            continue
#print('The best parameters of ARIMA{}x{} - AIC:{}'.format(bestparam,bestparam_seasonal,minAIC))
print('The best parameters of ARIMA{}x - AIC:{}'.format(bestparam,minAIC))
"""
bestparam=[4,1,4]
#bestparam_seasonal=[1,1,1,12]
model = sm.tsa.statespace.SARIMAX(bitcoin_train, order=bestparam,enforce_stationarity=False, enforce_invertibility=False)
results = model.fit()
#print(results.summary().tables[1])
#results.plot_diagnostics(figsize=(12, 12))
#plt.show()
pred = results.get_prediction(start=pd.to_datetime('2016-9-11'),end=pd.to_datetime('2021-9-10'), dynamic=False)
pred_ci = pred.conf_int()
ax = bitcoin_test['2019-6-8':'2021-9-10'].plot(label='Observed',figsize=(12, 6))
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
print(pred_ci)
ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('data')
plt.legend()
plt.show()
