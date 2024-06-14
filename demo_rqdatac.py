import rqdatac
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
import pandas as pd
import numpy as np
import pickle
from scipy.stats import spearmanr
import os
rqdatac.init()

#——————————————————————————————————第一步————————————————————————————————————————————
#找到每天st，停牌，上市不满一年的股票, 并替换为na
root = '/Users/ella/'
factor_single =  'pcf_ratio_total_lyr'
#因子
csv_file_path_fac = os.path.join(root, 'factor.csv')
#ST的股票
csv_file_path_st =  os.path.join(root, 'st_stocks.csv')
#停牌
csv_file_path_sus = os.path.join(root, 'suspended_stocks.csv')
#上市不满一年
#1: 退市日期 - 上市日期  < 1年
#2: 上市日期 > now - 1年 
csv_file_path_all_1 = os.path.join(root, 'all_stocks.csv')

#csv_file_path_new_2 = os.path.join(root, 'newly_listed_stocks.csv') #截止到当前日期上市不满一年的股票
#读数据
factor = pd.read_csv(csv_file_path_fac, index_col = 0)
st_stocks = pd.read_csv(csv_file_path_st, index_col = 0)
suspended_stocks = pd.read_csv(csv_file_path_sus, index_col = 0)
all_stocks = pd.read_csv(csv_file_path_all_1, index_col = 0, parse_dates=True)


#get所有的日期
dates = factor['date'].unique() #取全部因子数据中的date index
#先设置上市不满一年的s x t TF表格
less_1_yr = pd.DataFrame(False, index=dates, columns=all_stocks['order_book_id'])

#在每一个时间点上,截止到当前日期上市不满一年
for date in dates:
    one_year_ago = (datetime.strptime(date, '%Y-%m-%d') - relativedelta(years=1)).strftime('%Y-%m-%d')
    newly_listed_stocks_daily = all_stocks[all_stocks['listed_date'] > one_year_ago]['order_book_id'].tolist()
    less_1_yr.loc[date][newly_listed_stocks_daily] = True
   



#转换为t为列的index, s为行的index的格式
factor = factor.pivot(index='date', columns='order_book_id', values='pcf_ratio_total_lyr')

#处理上市不满一年的情况1: 退市日期 - 上市日期  < 1年 
all_stocks['listed_date'] = pd.to_datetime(all_stocks['listed_date'],  errors='coerce')
all_stocks['de_listed_date'] = pd.to_datetime(all_stocks['de_listed_date'],  errors='coerce')
all_stocks['is_one_year'] = all_stocks['de_listed_date'] - all_stocks['listed_date'] < pd.Timedelta(days=365)
duration_less_1yr = all_stocks[all_stocks['is_one_year'] == True]

#先合并ST和suspended,size一样可以直接or
combined = st_stocks | suspended_stocks | less_1_yr
print("所有条件集合: \n",combined)
#去除上市不满一年的股票
duration_less_1yr = duration_less_1yr['order_book_id'].tolist()
combined[duration_less_1yr] = True
#替换na表格
factor = factor.mask(combined)
print("替换na后的因子表: \n",factor)

#——————————————————————————————————第二步————————————————————————————————————————————
#第二步，因子去极值和标准化（注意都是在同一个时间截面上，不要在时序上去操作，会导致因子包含有未来信息）
def remove_extreme_and_standardize(df, n = 3):
    #去极值
    median = df.median(axis=1) #每行: axis=1
    mad = np.abs(df.subtract(median, axis=0)).median(axis=1)
    lower_limit = median - n * 1.4826 * mad
    upper_limit = median + n * 1.4826 * mad
    df_clipped = df.clip(lower=lower_limit, upper=upper_limit, axis=0)
    # 标准化
    mean = df_clipped.mean(axis=1)
    std = df_clipped.std(axis=1)
    df_standardized = df_clipped.sub(mean, axis=0).div(std, axis=0)
    
    return df_standardized

# 对每个时间截面进行去极值和标准化
factor_standardized = remove_extreme_and_standardize(factor)

#——————————————————————————————————第三步————————————————————————————————————————————
#第三步,行业市值中性化
#读数据 - 浮动市值
csv_file_path_market = os.path.join(root, 'market_cap_df.csv')
market_cap_df = pd.read_csv(csv_file_path_market, index_col=0)
#把market_cap_df 也转换成列=t 行=s的格式
market_cap_df.drop('market_cap_2', axis=1)
market_cap_df = market_cap_df.pivot(index='date', columns='order_book_id', values='log_market_cap')

#读industry dummies,为字典形式
with open('industry_dummies_dict.pickle', 'rb') as file:
    industry_dict = pickle.load(file)

#进行逐日回归
resid_df = pd.DataFrame()
for date in dates:
    #获取每日数据
    daily_ind = industry_dict[date]
    daily_ind['mark_value'] = market_cap_df.loc[date]
    daily_fac = factor.loc[date]
    #合并每日所需X=: 行业+市值
    merged = daily_ind.merge(daily_fac, on='order_book_id', how='inner').dropna()
    #y = 当日因子值
    y = merged.pop(date)
    X = merged.copy()
    model = sm.OLS(y, X.astype(float)).fit()
    #获取因子残差作为结果, 并设置列名称 = 日期
    resid_series = pd.Series(model.resid, name = date)
    # 将残差添加到resid_df中，按列合并，并自动对齐索引
    resid_df = pd.concat([resid_df, resid_series], axis=1)

# 打印结果 DataFrame
print("逐日回归结果: \n", resid_df)
csv_file_path_merge = '/Users/ella/resid_result.csv'
resid_df.to_csv(csv_file_path_merge)
print(f"数据已保存为 {csv_file_path_merge}")


#——————————————————————————————————第四步————————————————————————————————————————————
#第四步，再次标准化
csv_file_path = os.path.join(root, 'resid_result.csv')
#互换一下行列的index
df = pd.read_csv(csv_file_path)
df.set_index('Unnamed: 0', inplace=True)
df.index.name = 'date'
df = df.T
df.index.name = 'order_book_id'
#再次标准化
df = remove_extreme_and_standardize(df)
print("再次标准化后的数据: \n", df)

