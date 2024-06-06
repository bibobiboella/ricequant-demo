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
print("动态?industry data: \n", industry_data)


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
market_cap_df = pd.DataFrame(market_cap_data)

#log(市值)
market_cap_df['log_market_cap'] = np.log(market_cap_df['market_cap'])
industry_dummies = pd.get_dummies(df_sw_ind['index_name'], prefix='industry')

#merge因子值/市值/行业哑变量
factor_df = unstack_df.stack().reset_index()
factor_df.columns = ['date', 'order_book_id', 'factor_value']
market_cap_df = market_cap_df.reset_index()
merged_df = factor_df.merge(market_cap_df[['order_book_id', 'date', 'log_market_cap']], on=['order_book_id', 'date'])#merge行
merged_df = merged_df.merge(industry_dummies.reset_index(), on='order_book_id', how='left')#merge列
merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()#避免inf
exclude_columns = ['date', 'order_book_id', 'factor_value', 'log_market_cap']
bool_columns = [col for col in merged_df.select_dtypes(include='bool').columns if col not in exclude_columns]
merged_df[bool_columns] = merged_df[bool_columns].astype(int)
merged_df = merged_df.set_index(['date', 'order_book_id']).dropna()

#分X和y
y = merged_df.pop("factor_value")
X = merged_df.copy()
X = sm.add_constant(X) 
#print('X: \n', X)
#print('y: \n', y)
#线性回归
model = sm.OLS(y, X.astype(float)).fit()
#残差 = 中性化后的因子值
print(model.resid)

