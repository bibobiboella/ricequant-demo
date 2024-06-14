import rqdatac
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os
import pickle
rqdatac.init()

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
start = '20120101'
end = '20240606'

ticker = rqdatac.all_instruments('CS').order_book_id
factor_single =  'pe_ratio'

#get数据,get完后可以comment掉
all_stocks = rqdatac.all_instruments(type='CS')
all_stock_symbols = all_stocks['order_book_id'].tolist()
df = rqdatac.get_factor(all_stock_symbols, factor_single, start_date=start, end_date=end).reset_index()
dates = pd.to_datetime(df['date'].unique()).strftime('%Y-%m-%d')
today = datetime.now().strftime('%Y-%m-%d')
# 获取每天ST的股票
st_stocks = rqdatac.is_st_stock(ticker, start_date=start, end_date=end)

csv_file_path = '/Users/ella/factor.csv' 
df.to_csv(csv_file_path)
print(f"数据已保存为 {csv_file_path}")

csv_file_path = '/Users/ella/st_stocks.csv' 
st_stocks.to_csv(csv_file_path)
print(f"数据已保存为 {csv_file_path}")



# 获取每天停牌的股票
suspended_stocks = rqdatac.is_suspended(ticker, start_date=start, end_date=end)
csv_file_path = '/Users/ella/suspended_stocks.csv'
suspended_stocks.to_csv(csv_file_path)
print(f"数据已保存为 {csv_file_path}")

# 获取上市不满一年的股票
one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
newly_listed_stocks = all_stocks[all_stocks['listed_date'] > one_year_ago]['order_book_id'].tolist()

newly_listed_stocks = pd.DataFrame(newly_listed_stocks)
csv_file_path = '/Users/ella/newly_listed_stocks.csv'
newly_listed_stocks.to_csv(csv_file_path)
print(f"数据已保存为 {csv_file_path}")

#下载流通市值数据
#股票总市值
ticker = rqdatac.all_instruments('CS').order_book_id
market_cap_data = rqdatac.get_factor(ticker, 'market_cap_2', start, end)
market_cap_df = pd.DataFrame(market_cap_data).reset_index()
#print(market_cap_data)
#log(市值)
market_cap_df['log_market_cap'] = np.log(market_cap_df['market_cap_2'])
csv_file_path_ind = '/Users/ella/market_cap_df.csv'
market_cap_df.to_csv(csv_file_path_ind)
print(f"数据已保存为 {csv_file_path_ind}")

start = '20120101'
end = '20240606'
ticker = rqdatac.all_instruments('CS').order_book_id
dates = pd.date_range(start=start, end=end, freq='B') 
#print(dates)

root = '/Users/ella/'
csv_file_path_fac = os.path.join(root, 'factor.csv')
factor = pd.read_csv(csv_file_path_fac, index_col = 0)
dates = factor['date'].unique()

#下载每日的行业数据
ticker = rqdatac.all_instruments('CS').order_book_id
all_industry_dummies = {}
for current in dates:
    df_sw_ind = rqdatac.shenwan_instrument_industry(ticker, date = current)
    industry_dummies = pd.get_dummies(df_sw_ind['index_name'], prefix='industry')
    all_industry_dummies[current] = industry_dummies
#将字典形式保存
with open('industry_dummies_dict.pickle', 'wb') as file:
    pickle.dump(all_industry_dummies, file)

#读字典
with open('industry_dummies_dict.pickle', 'rb') as file:
    # 使用pickle.load()方法从文件中读取对象
    loaded_data = pickle.load(file)

print("打开!打开!: \n",loaded_data['2012-01-04'])
