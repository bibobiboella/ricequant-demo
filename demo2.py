import rqdatac
import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
rqdatac.init()

start = '20240103'
end = '20240601'
period = 20
start_date = pd.to_datetime(start, format='%Y%m%d')
end_date = pd.to_datetime(end, format='%Y%m%d')
new_start_date = start_date + pd.tseries.offsets.BDay(period)
new_end_date = end_date + pd.tseries.offsets.BDay(period)
new_start = new_start_date.strftime('%Y%m%d')
new_end = new_end_date.strftime('%Y%m%d')
factor =  'pe_ratio'
ticker = rqdatac.all_instruments('CS').order_book_id
dates = pd.date_range(start=start, end=end, freq='B') 
industry_data = pd.DataFrame()

for date in dates:
    date_str = date.strftime('%Y-%m-%d')
    industry_info = rqdatac.get_instrument_industry(ticker, date=date_str)
    industry_info['date'] = date_str
    industry_data = pd.concat([industry_data, industry_info])

industry_data.reset_index(drop=True, inplace=True)
#print("动态 industry data: \n", industry_data)


#因子
df = rqdatac.get_factor(ticker, factor, start_date=start, end_date=end)
unstack_df = df[factor].unstack(level='order_book_id')
nan_columns = unstack_df.isna().any()
#print("Columns with all values as NaN:", nan_columns[nan_columns == True])
#unstack_df.drop(columns=nan_columns[nan_columns == True].index)
#申万一级行业分类数据
df_sw_ind = rqdatac.shenwan_instrument_industry(ticker, level=1)
#股票总市值
market_cap_data = rqdatac.get_factor(ticker, 'market_cap', start_date, end_date)
market_cap_df = pd.DataFrame(market_cap_data).reset_index()
#log(市值)
market_cap_df['log_market_cap'] = np.log(market_cap_df['market_cap'])
industry_dummies = pd.get_dummies(df_sw_ind['index_name'], prefix='industry')
df = df.reset_index()


#result df
neutralized_factor_data = pd.DataFrame()
# 逐日回归
for date in unstack_df.index:
    date_str = date.strftime('%Y-%m-%d')
    #因子值
    factor_data_daily = df[df['date'] == date_str].reset_index()
    #市值数据
    market_cap_daily = market_cap_df[market_cap_df['date'] == date_str].reset_index()
    #动态行业分类数据
    industry_daily = industry_data[industry_data['date'] == date_str].reset_index()
    #合并
    merged_df = factor_data_daily.merge(market_cap_daily[['order_book_id', 'date', 'log_market_cap']], on=['order_book_id', 'date'])#merge行
    merged_df = merged_df.merge(industry_dummies.reset_index(), on='order_book_id', how='left').drop('index', axis=1).dropna()#merge列
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()#避免inf
    exclude_columns = ['date', 'order_book_id', 'log_market_cap', 'pe_ratio']
    bool_columns = [col for col in merged_df.select_dtypes(include='bool').columns if col not in exclude_columns]
    merged_df[bool_columns] = merged_df[bool_columns].astype(int)
    merged_df = merged_df.set_index(['date', 'order_book_id']).dropna()
    #分X和y
    y = merged_df.pop(factor)
    X = merged_df.copy()
    X = sm.add_constant(X) 
    #print('X: \n', X)
    #print('y: \n', y)
    #线性回归
    model = sm.OLS(y, X.astype(float)).fit()
    #get残差
    merged_df['neutralized_factor_value'] = model.resid
    merged_df['date_'] = date_str
    merged_df = merged_df.reset_index() #reset一下方便后面加date columns
    #save result to a new df
    neutralized_factor_data = pd.concat([neutralized_factor_data, merged_df[['date_', 'order_book_id', 'neutralized_factor_value']]])


neutralized_factor_data.reset_index(drop=True, inplace=True)
neutralized_factor_data = neutralized_factor_data.set_index(['date_', 'order_book_id'])
# 打印中性化后的因子值
print(neutralized_factor_data)

