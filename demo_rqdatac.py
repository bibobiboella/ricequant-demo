import rqdatac
from datetime import datetime, timedelta
import statsmodels.api as sm
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



#——————————————————————————————————第一步————————————————————————————————————————————
#找到每天st，停牌，上市不满一年的股票, 并替换为na
factor_single =  'pe_ratio'
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

#get所有的日期
dates = factor['date'].unique()
print("日期: \n",dates)

#简简单单清洗一下
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



###dates
for specific_date in dates:
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
    factor.loc[mask, factor_single] = np.nan

# 打印修改后的 DataFrame
print("\n修改后的 DataFrame:")
#print(factor)


csv_file_path_clean_fac = '/Users/ella/factor_clean.csv'
factor.to_csv(csv_file_path_clean_fac)
print(f"数据已保存为 {csv_file_path_clean_fac}")





#——————————————————————————————————第二步————————————————————————————————————————————
#第二步，因子去极值和标准化（注意都是在同一个时间截面上，不要在时序上去操作，会导致因子包含有未来信息）
#读第一步替换好na的数据

"""
csv_file_path_clean_fac = '/Users/ella/factor_clean.csv'
df = pd.read_csv(csv_file_path_clean_fac)
#print("第一步替换过na后的表: \n", df.head())

def remove_extreme_and_std(series, n=3): #中位数去极值 --by研报
    median = series.median()
    mad = np.median(np.abs(series - median))  # 计算中位数绝对偏差
    lower_limit = median - n * mad
    upper_limit = median + n * mad
    result = series.clip(lower_limit, upper_limit)
    return (result - result.mean()) / result.std()

df['factor_remove_extreme_std'] = df.groupby('date')['pcf_ratio_total_lyr'].transform(remove_extreme_and_std)
#print("clean factor data: \n", df)
"""

#——————————————————————————————————第三步————————————————————————————————————————————
#第三步,行业市值中性化
#dates = ['2024-05-31']
"""
csv_file_path_ind = '/Users/ella/industry_dummies_1.csv'
industry_data = pd.read_csv(csv_file_path_ind)
print("行业数值: \n", industry_data)

csv_file_path_ind = '/Users/ella/market_cap_df.csv'
market_cap_df = pd.read_csv(csv_file_path_ind)
print("流动市值数据: \n", market_cap_df)

#下载流通市值数据
start = '20120101'
end = '20240606'
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
"""

"""
start = '20120101'
end = '20240606'
#先get一下动态的行业分类数据, 完成后可以删掉
#start_date = pd.to_datetime(start, format='%Y%m%d')
#end_date = pd.to_datetime(end, format='%Y%m%d')
#factor =  'pe_ratio'
ticker = rqdatac.all_instruments('CS').order_book_id
dates = pd.date_range(start=start, end=end, freq='B') 
#print(dates)

all_industry_dummies = pd.DataFrame()
for current in dates:
    df_sw_ind = rqdatac.shenwan_instrument_industry(ticker, date = current)
    industry_dummies = pd.get_dummies(df_sw_ind['index_name'], prefix='industry')
    industry_dummies['date'] = current
    industry_dummies['date'] = pd.to_datetime(industry_dummies['date'])
    industry_dummies.set_index('date', append=True, inplace=True)
    industry_dummies = industry_dummies.reorder_levels(['date', 'order_book_id'])
    all_industry_dummies = pd.concat([all_industry_dummies, industry_dummies], axis=0)


print(all_industry_dummies)

csv_file_path_ind = '/Users/ella/industry_dummies_1.csv'
all_industry_dummies.to_csv(csv_file_path_ind)
print(f"数据已保存为 {csv_file_path_ind}")

"""

"""
#date = '2024-06-06'
neutralized_factor_data = pd.DataFrame()
#dates = ['2024-06-06']
dates = df['date'].unique()
for date in dates:
    #因子值
    factor_data_daily = df[df['date'] == date].reset_index()
    #print("factor data daily: \n", factor_data_daily)
    #市值数据
    market_cap_daily = market_cap_df[market_cap_df['date'] == date].reset_index()
    #print("market cap data daily: \n", market_cap_daily)
    #动态行业分类数据
    industry_daily = industry_data[industry_data['date'] == date].reset_index()
    industry_daily = industry_daily.fillna(False)
    #print("industry data daily: \n", industry_daily)
    #合并
    merged_df = factor_data_daily.merge(market_cap_daily[['order_book_id', 'date', 'log_market_cap']], on=['order_book_id', 'date'])#merge行
    #print("合并行: \n",merged_df)
    merged_df = merged_df.merge(industry_daily.reset_index(), on='order_book_id', how='left').drop('pcf_ratio_total_lyr', axis=1).dropna()#merge列
    #merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()#避免inf
    #print("合并过后的df: \n",merged_df)
    #exclude_columns = ['date', 'order_book_id', 'log_market_cap']
    #bool_columns = [col for col in merged_df.select_dtypes(include='bool').columns if col not in exclude_columns]
    #merged_df[bool_columns] = merged_df[bool_columns].astype(int)
    #merged_df = merged_df.set_index(['date', 'order_book_id']).dropna()
    #分X和y
    y = merged_df.pop('factor_remove_extreme_std')
    X = merged_df.copy().drop(['index_x', 'index_y', 'date_x', 'order_book_id','date_y'], axis=1)
    X = sm.add_constant(X) 
    #print('X: \n', X)
    #print('y: \n', y)
    #线性回归
    model = sm.OLS(y, X.astype(float)).fit()
    #print(model.resid)
    #get残差
    merged_df['neutralized_factor_value'] = model.resid
    merged_df['date_'] = date
    merged_df = merged_df.reset_index() #reset一下方便后面加date columns
    #save result to a new df
    neutralized_factor_data = pd.concat([neutralized_factor_data, merged_df[['date_', 'order_book_id', 'neutralized_factor_value']]])
    neutralized_factor_data.reset_index(drop=True, inplace=True)

neutralized_factor_data = neutralized_factor_data.set_index(['date_', 'order_book_id'])
# 打印中性化后的因子值
print("行业数值中性化结果: \n", neutralized_factor_data)

csv_file_path_neu = '/Users/ella/neutralized_factor_data.csv'
neutralized_factor_data.to_csv(csv_file_path_neu)
print(f"数据已保存为 {csv_file_path_neu}")

csv_file_path_merge = '/Users/ella/merged_df_4.csv'
merged_df.to_csv(csv_file_path_merge)
print(f"数据已保存为 {csv_file_path_merge}")
"""






#——————————————————————————————————第四步————————————————————————————————————————————
#第四步，再次标准化
csv_file_path_neu = '/Users/ella/neutralized_factor_data.csv'
result_3 = pd.read_csv(csv_file_path_neu)
#print("行业数值中性化逐日回归后的数据: \n", result_3)

#再次标准化
result_3['result_std'] = (result_3['neutralized_factor_value'] - result_3['neutralized_factor_value'].mean()) / result_3['neutralized_factor_value'].std()
print("标准化后的因子残差:  \n",result_3)