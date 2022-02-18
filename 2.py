#talib
import pandas as pd
import talib as ta
gold=pd.read_csv(r'C:\Users\Lenovo\Desktop\2022_Problem_C_DATA\gold.csv')

gold.columns=['1','Date','data']
gold=gold.drop(['1'],axis=1)

bitcoin=pd.read_csv(r'C:\Users\Lenovo\Desktop\2022_Problem_C_DATA\bitcoin.csv')
bitcoin.columns=['1','Date','data']
bitcoin=bitcoin.drop(['1'],axis=1)

print(ta.get_functions())

print(ta.get_function_groups())

ta_fun=ta.get_function_groups()

ta_fun.keys()