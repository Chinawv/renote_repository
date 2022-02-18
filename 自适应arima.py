#arima
import pandas as pd
import itertools
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
gold=pd.read_csv(r'C:\Users\Lenovo\Desktop\2022_Problem_C_DATA\gold.csv')
gold.columns=['Date','data']
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
bitcoin_log=bitcoin
bitcoin_log['data']=bitcoin_log['data'].apply(np.log1p)
print(bitcoin_train)
print(bitcoin_test)
alpha_gold=0.01
alpha_bitcoin=0.02
gold_diff=gold.diff()
bitcoin_diff=bitcoin.diff()
gold_diff.columns=['gold_diff']
bitcoin_diff.columns=['bitcoin_diff']
#gold_diff.plot()
#bitcoin_diff.plot()
#plt.show()

#print(seasonal_pdq)

def train_model(data):
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    minAIC = 100000
    # print(p,d,q)
    #seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]
    for param in pdq:
       # for param_seasonal in seasonal_pdq:
            try:
                #model = sm.tsa.statespace.SARIMAX(data, order=param, seasonal_order=param_seasonal,
                                                #  enforce_stationarity=False, enforce_invertibility=False)
                model = sm.tsa.statespace.SARIMAX(data, order=param,
                enforce_stationarity = False, enforce_invertibility = False)
                results = model.fit()
                print('ARIMA{}x - AIC:{}'.format(param, results.aic))
               # print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
                if (results.aic < minAIC):
                    minAIC = results.aic
                    bestparam = param
                    #bestparam_seasonal = param_seasonal
            except:
                print('sb')
                continue
    return bestparam

#print('The best parameters of ARIMA{}x{} - AIC:{}'.format(bestparam,bestparam_seasonal,minAIC))
#print('The best parameters of ARIMA{}x - AIC:{}'.format(bestparam,minAIC))
result_pre=bitcoin_train
result_pre=bitcoin_train.apply(np.expm1)
for i in range(1001,1800,100):
    bestpa = train_model(bitcoin_log.iloc[i-100:i-1,:])
    model = sm.tsa.statespace.SARIMAX(bitcoin_log.iloc[i-100:i-1,:], order=bestpa,  enforce_stationarity=False,
                                      enforce_invertibility=False)
    results = model.fit()
    #pred = results.get_prediction(start=pd.to_datetime('2016-9-11'), end=pd.to_datetime('2021-9-10'), dynamic=False)
    forc=results.forecast(10)
    forc=forc.to_frame()
    print(forc)
    forc.columns=['data']
    forc['data']=forc['data'].apply(np.expm1)
    print(type(forc))
    print(type(result_pre))
    #result_pre=result_pre.append(forc, )
    result_pre=pd.concat([result_pre,forc],axis=0)
    print(result_pre)
plt.plot(result_pre)
plt.plot(bitcoin_test.apply(np.expm1))
plt.show()
    #pred_ci = pred.conf_int()
#print(results.summary().tables[1])
#results.plot_diagnostics(figsize=(12, 12))
#plt.show()

#ax = bitcoin_test['2019-6-8':'2021-9-10'].plot(label='Observed',figsize=(12, 6))
#pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
#print(pred_ci)
#ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='k', alpha=.2)
#ax.set_xlabel('Date')
#ax.set_ylabel('data')
#plt.legend()
#plt.show()
#cancha=pred.predicted_mean[:1000,]
