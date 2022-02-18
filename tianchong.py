import pandas as pd
import itertools
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

gold = pd.read_csv(r'C:\Users\Lenovo\Desktop\2022_Problem_C_DATA\gold.csv')
gold.columns = ['1', 'Date', 'data']
gold = gold.drop(['1'], axis=1)
gold.iloc[:, 0] = pd.to_datetime(gold.iloc[:, 0])
gold = gold.set_index(['Date'])
t = pd.DataFrame(index=pd.date_range(gold.index[0], gold.index[-1]))
print(t)
gold = t.join(gold).fillna(method='pad')
print(gold)
gold.to_csv(r"C:\Users\Lenovo\Desktop\2022_Problem_C_DATA\gold.csv")