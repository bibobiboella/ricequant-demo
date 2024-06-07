import rqdatac
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
rqdatac.init()
"""
start = '20120101'
end = '20240606'
start_date = pd.to_datetime(start, format='%Y%m%d')
end_date = pd.to_datetime(end, format='%Y%m%d')
factor =  'pcf_ratio_total_lyr'
ticker = rqdatac.all_instruments('CS').order_book_id
#dates = pd.date_range(start=start, end=end, freq='B') 

#print(rqdatac.get_all_factor_names(type='eod_indicator'))
industry_data = pd.DataFrame()
neutralized_factor_data = pd.DataFrame()

#get数据,get完后可以comment掉
all_stocks = rqdatac.all_instruments(type='CS')
all_stock_symbols = all_stocks['order_book_id'].tolist()
df = rqdatac.get_factor(all_stock_symbols, factor, start_date=start, end_date=end).reset_index()
dates = pd.to_datetime(df['date'].unique()).strftime('%Y-%m-%d')
today = datetime.now().strftime('%Y-%m-%d')


csv_file_path = '/Users/ella/factor.csv' 
df.to_csv(csv_file_path)
print(f"数据已保存为 {csv_file_path}")
"""


"""
#————————————————第一步————————————————
#找到每天st，停牌，上市不满一年的股票, 并替换为na
#因子
csv_file_path_fac = '/Users/ella/factor.csv' 
#ST的股票
csv_file_path_st = '/Users/ella/st_stocks.csv' 
#停牌
csv_file_path_sus = '/Users/ella/suspended_stocks.csv'
#上市不满一年
#1: 退市日期 - 上市日期  < 1年 #248
#2: 上市日期 > now - 1年 #179
csv_file_path_all_1 = '/Users/ella/all_stocks.csv' 
csv_file_path_new_2 = '/Users/ella/newly_listed_stocks.csv'
#读数据
factor = pd.read_csv(csv_file_path_fac)
st_stocks = pd.read_csv(csv_file_path_st)
suspended_stocks = pd.read_csv(csv_file_path_sus)
newly_listed_stocks_1 = pd.read_csv(csv_file_path_all_1)
newly_listed_stocks_2 = pd.read_csv(csv_file_path_new_2)

#切成小数据
#获取一下所有的dates
dates = factor['date'][-10:]
#print(dates)

#简简单单清洗一下
#factor.drop('Unnamed: 0', axis=1)
factor = factor.drop('Unnamed: 0', axis=1)
factor.sort_values(by=['date', 'order_book_id'], inplace=True)
factor.set_index(['date', 'order_book_id'], inplace=True)
st_stocks.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
st_stocks.set_index('date', inplace=True)
suspended_stocks.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
suspended_stocks.set_index('date', inplace=True)
newly_listed_stocks_2 = newly_listed_stocks_2.drop('Unnamed: 0', axis=1)
newly_listed_stocks_2.rename(columns={'0': 'order_book_id'}, inplace=True)

#处理上市不满一年的情况1: 退市日期 - 上市日期  < 1年 
newly_listed_stocks_1['listed_date'] = pd.to_datetime(newly_listed_stocks_1['listed_date'],  errors='coerce')
newly_listed_stocks_1['de_listed_date'] = pd.to_datetime(newly_listed_stocks_1['de_listed_date'],  errors='coerce')
newly_listed_stocks_1['is_one_year'] = newly_listed_stocks_1['de_listed_date'] - newly_listed_stocks_1['listed_date'] > pd.Timedelta(days=365)
#print(newly_listed_stocks_1['is_one_year'].sum()) #248
newly_listed_stocks_1 = newly_listed_stocks_1[newly_listed_stocks_1['is_one_year'] == True]


#看一下因子index的形式(date, ticker)

#print("因子: \n", factor.head())
#print("ST股票: \n",st_stocks.head())
#print("停牌: \n", suspended_stocks.head())
#print("上市不满一年_1: \n", newly_listed_stocks_1) #248, 情况1
#print("上市不满一年_2: \n", newly_listed_stocks_2) #179, 情况2


date = ['2024-05-31']
for specific_date in date:
#filter一下所有为True的股票
#ONLY ticker~
#specific_date = '2024-05-31'
#是ST的
    st = st_stocks.loc[specific_date]
    st = st[st == True].index.tolist()
    #print(stocks_st)

    #是停牌的
    suspended = suspended_stocks.loc[specific_date]
    suspended = suspended[suspended == True].index.tolist()
    #print(suspended)

    #是上市不满一年的
    smaller_1yr = newly_listed_stocks_1['order_book_id'].tolist() + newly_listed_stocks_2['order_book_id'].tolist()
    #print(smaller_1yr)



    # 整合所有条件
    filtered_stocks = set(st) | set(suspended) | set(smaller_1yr)
    filtered_stocks = list(filtered_stocks)
    #print("条件三合一: \n  ST+停牌+上市不满一年: \n", filtered_stocks)

    #开始替换为Nan
    mask = (factor.index.get_level_values('date') == specific_date) & (factor.index.get_level_values('order_book_id').isin(filtered_stocks))
    factor.loc[mask, 'pcf_ratio_total_lyr'] = np.nan

# 打印修改后的 DataFrame
print("\n修改后的 DataFrame:")
print(factor)


csv_file_path_clean_fac = '/Users/ella/factor_clean.csv'
factor.to_csv(csv_file_path_clean_fac)
print(f"数据已保存为 {csv_file_path_clean_fac}")


"""


#————————————————第二步————————————————
#第二步，因子去极值和标准化（注意都是在同一个时间截面上，不要在时序上去操作，会导致因子包含有未来信息）
#读第一步替换好na的数据
csv_file_path_clean_fac = '/Users/ella/factor_clean.csv'
df = pd.read_csv(csv_file_path_clean_fac)
print("第一步替换过na后的表: \n", df.head())

