import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
bitcoin=pd.read_csv(r'./data/BCHAIN-MKPRU.csv')
bitcoin.columns=['Date','data']
#bitcoin=bitcoin.drop(['1'],axis=1)
bitcoin.iloc[:,0]=pd.to_datetime(bitcoin.iloc[:,0])
bitcoin=bitcoin.set_index(['Date'])
gold=pd.read_csv(r'./data/gold.csv')
gold.columns=['Date','data']
gold.iloc[:,0]=pd.to_datetime(gold.iloc[:,0])
gold=gold.set_index(['Date'])
data=pd.concat([gold,bitcoin],axis=1)
data.columns=['gold_real','bitcoin_real']
data['dollar']=[1]*len(data)
#data['dollar_real']=[0]*len(data)
data.iloc[0,0]=1324.60
re=data.pct_change()
re.columns=['gold_returns','bitcoin_returns','dollar_returns']
re.iloc[0,:]=0

print(re)
bitcoin_pre=pd.read_csv(r'./bitcoin_pre.csv')
gold_pre=pd.read_csv(r'./gold_pre.csv')
gold_pre=gold_pre.drop(['data','pre+4','pre+5'],axis=1)
bitcoin_pre=bitcoin_pre.drop(['data','pre+4','pre+5'],axis=1)
gold_pre.columns=['Date','goldpre+1','goldpre+2','goldpre+3']
bitcoin_pre.iloc[:,0]=pd.to_datetime(bitcoin_pre.iloc[:,0])
bitcoin_pre.columns=['Date','bitcoinpre+1','bitcoinpre+2','bitcoinpre+3']
gold_pre.iloc[:,0]=pd.to_datetime(gold_pre.iloc[:,0])
gold_pre=gold_pre.set_index(['Date'])
bitcoin_pre=bitcoin_pre.set_index(['Date'])
all=pd.concat([data.iloc[:-2,:],gold_pre,bitcoin_pre,re],axis=1)
all=all.drop(all.tail(1).index)
#all=all.drop(all.head(1).index)

all['gg']=[0]*len(all)
all['bb']=[0]*len(all)
all['dd']=[1]*len(all)
all['zichan']=[1000]*len(all)
all['cov']=[0]*len(all)

print('all\n',all)
print(all[all.isnull().values==True])
data=data.pct_change()

cost_gold=0.01
cost_bit=0.02
#data=data.iloc[1300:1500,:]
mean_returns = data.mean()
cov_matrix = data.cov()
num_portfolios = 5000
risk_free_rate = 0
def compute_returns_and_cov():
    for i in range(20,len(all)):
        all['cov'].iloc[i]=all.iloc[i-20:i,:3].cov()
        if i>=19:
            all['gold_returns'].iloc[i]=(0.99/1.01)*(all['goldpre+2'].iloc[i]-all['gold_real'].iloc[i])/all['gold_real'].iloc[i]-2*0.01/1.01
            all['bitcoin_returns'].iloc[i]=(0.98/1.02)*(all['bitcoinpre+2'].iloc[i]-all['bitcoin_real'].iloc[i])/all['bitcoin_real'].iloc[i]-2*0.01/1.01
            #all['dollar_returns'].iloc[i]=(all['dollar_pre'].iloc[i]-all['dollar_real'])/all['dollar_real']

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
   returns = np.sum(mean_returns*weights ) *365
   std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(365)
   return std, returns

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
   results = np.zeros((3,num_portfolios))
   weights_record = []
   for i in range(num_portfolios):
       weights = np.random.random(3)
       weights /= (np.sum(weights))
       weights_record.append(weights)
       portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
       results[0,i] = portfolio_std_dev
       results[1,i] = portfolio_return
       results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
   return results, weights_record


def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0, max_sharpe_idx], results[1, max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx], index=data.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0, min_vol_idx], results[1, min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx], index=data.columns, columns=['allocation'])
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    print("-" * 80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp, 2))
    print("Annualised Volatility:", round(sdp, 2))
    print("\n")
    print(max_sharpe_allocation)
    print("-" * 80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min, 2))
    print("Annualised Volatility:", round(sdp_min, 2))
    print("\n")
    print(min_vol_allocation)
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp, rp, marker='*', color='r', s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min, rp_min, marker='*', color='g', s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)
    plt.show()
    """
    return max_sharpe_allocation
compute_returns_and_cov()
alphagold=0.01
alphabitcoin=0.02
print("12345\n")
gold_trade=0
#display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)
for i in range(30,1800):
    #第i天进行调仓
    if(all.iloc[i:i+1,:1].values==all.iloc[i-1:i,:1].values):
        gold_trade=0;
    else:
        gold_trade=1
    www_before=[all.loc[:,'gg'].iloc[i],all.loc[:,'bb'].iloc[i],all.loc[:,'dd'].iloc[i]]#调仓前资产比重
    zichan_before=all.loc[:,'zichan'].iloc[i-1]
    gold_change=all.loc[:,'gold_real'].iloc[i]/all.loc[:,'gold_real'].iloc[i-1]#黄金真实变化率
    bitcoin_change= all.loc[:,'bitcoin_real'].iloc[i]/all.loc[:,'bitcoin_real'].iloc[i-1]#比特币真实变化率
    zichan=www_before[0]*zichan_before*gold_change+www_before[1]*zichan_before*bitcoin_change+zichan_before*www_before[2]

    www=display_simulated_ef_with_random(all.loc[:,'gold_returns':'dollar_returns'].iloc[i],
                                         all['cov'].iloc[i], num_portfolios, risk_free_rate).values
    #调仓后的资产比重
    print(www)

    if(gold_trade==0):
        all.loc[:, 'gg'].iloc[i + 1] =all.loc[:,'gg'].iloc[i]
        gold_c=1-all.loc[:,'gg'].iloc[i]
        all.loc[:, 'bb'].iloc[i + 1] = gold_c*(www[0][1]/(www[0][1]+www[0][2])) / 100
        all.loc[:, 'dd'].iloc[i + 1] = gold_c*(www[0][2]/(www[0][1]+www[0][2])) / 100

    all.loc[:,'gg'].iloc[i+1]=www[0][0]/100
    all.loc[:,'bb'].iloc[i+1]=www[0][1]/100
    all.loc[:,'dd'].iloc[i+1]=www[0][2]/100
    zichan=zichan*(1-abs(all.loc[:,'gg'].iloc[i+1]-all.loc[:,'gg'].iloc[i])*alphagold\
           -abs(all.loc[:,'bb'].iloc[i+1]-all.loc[:,'bb'].iloc[i+1])*alphabitcoin)
    all.loc[:, 'zichan'].iloc[i] = zichan  # 今天实际资产
    print('第',i,'天的','资产：',zichan)
all.to_csv(r".\all3.csv")
print(1)
