import rqdatac
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
rqdatac.init()

start = '20120101'
end = '20240606'
start_date = pd.to_datetime(start, format='%Y%m%d')
end_date = pd.to_datetime(end, format='%Y%m%d')
factor =  'pe_ratio'
ticker = rqdatac.all_instruments('CS').order_book_id
#dates = pd.date_range(start=start, end=end, freq='B') 
industry_data = pd.DataFrame()
neutralized_factor_data = pd.DataFrame()



#get数据,get完后可以comment掉
all_stocks = rqdatac.all_instruments(type='CS')
all_stock_symbols = all_stocks['order_book_id'].tolist()
df = rqdatac.get_factor(all_stock_symbols, factor, start_date=start, end_date=end).reset_index()
#print(df['date'].unique())
dates = pd.to_datetime(df['date'].unique()).strftime('%Y-%m-%d')
today = datetime.now().strftime('%Y-%m-%d')
    #date_str = today.strftime('%Y-%m-%d')
# 获取每天ST的股票
st_stocks = rqdatac.is_st_stock(ticker, start_date=start, end_date=end)
#st_stocks = st_stocks[st_stocks.loc['2024-05-31'] == True]#.index.tolist()

csv_file_path = '/Users/ella/factor.csv' 
df.to_csv(csv_file_path)
print(f"数据已保存为 {csv_file_path}")

csv_file_path = '/Users/ella/st_stocks.csv' 
st_stocks.to_csv(csv_file_path)
print(f"数据已保存为 {csv_file_path}")
#df = pd.read_csv(csv_file_path)



    # 获取每天停牌的股票
suspended_stocks = rqdatac.is_suspended(ticker, start_date=start, end_date=end)
#print(suspended_stocks)
#suspended_stocks = suspended_stocks[suspended_stocks[end] == True].index.tolist()
csv_file_path = '/Users/ella/suspended_stocks.csv'
suspended_stocks.to_csv(csv_file_path)
print(f"数据已保存为 {csv_file_path}")

#print(suspended_stocks)

    # 获取上市不满一年的股票
one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
newly_listed_stocks = all_stocks[all_stocks['listed_date'] > one_year_ago]['order_book_id'].tolist()
newly_listed_stocks = pd.DataFrame(newly_listed_stocks)
csv_file_path = '/Users/ella/newly_listed_stocks.csv'
newly_listed_stocks.to_csv(csv_file_path)
print(f"数据已保存为 {csv_file_path}")

"""
    # 整合所有条件
filtered_stocks = set(st_stocks.columns) | set(suspended_stocks.columns) | set(newly_listed_stocks)


#factor_data = rqdatac.get_factor(ticker, factor, start_date=start, end_date=end)
#factor_data.loc[all_filtered_stocks, :] = None

print("\n", st_stocks)
print("\n", suspended_stocks)
print("\n", newly_listed_stocks)
print("\n", len(filtered_stocks))
# 获取因子数据并替换为NA
#factor_data = get_factor_data()  # 获取因子数据的函数，需要自行定义或根据实际情况调整
#factor_data.loc[filtered_stocks] = np.nan

# 保存结果
#factor_data.to_csv('filtered_factor_data.csv')
"""

