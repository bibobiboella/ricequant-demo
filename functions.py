#一些常用的functions
import rqdatac
from datetime import datetime, timedelta
import statsmodels.api as sm
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from scipy.stats import spearmanr
import os
import pickle
import time
rqdatac.init()
#——————————————————————————————————因子清洗————————————————————————————————————————————
#下载数据类:
def get_factor(factors, start, end, file_name, ticker = rqdatac.all_instruments(type='CS')['order_book_id'].tolist(), root = '/Users/ella/rqdata/'):
    factor = rqdatac.get_factor(ticker, factors, start_date=start, end_date=end).reset_index()
    #global dates
    #dates = factor['date'].unique() 
    factor = factor.pivot(index='date', columns='order_book_id', values=factors)
    csv_file_path = os.path.join(root, file_name)
    factor.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")
    return factor

def get_market_cap( start, end, file_name, ticker = rqdatac.all_instruments(type='CS')['order_book_id'].tolist(), market_factor = 'market_cap_2', root = '/Users/ella/rqdata/'):
    market_cap_df = rqdatac.get_factor(ticker, market_factor, start_date=start, end_date=end).reset_index()
    #log(市值)
    market_cap_df['log_market_cap'] = np.log(market_cap_df['market_cap_2'])
    market_cap_df.drop('market_cap_2', axis=1)
    market_cap_df = market_cap_df.pivot(index='date', columns='order_book_id', values='log_market_cap')
    csv_file_path = os.path.join(root, file_name)
    market_cap_df.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")
    return market_cap_df

def get_st(start, end, file_name, ticker = rqdatac.all_instruments(type='CS')['order_book_id'].tolist(), root = '/Users/ella/rqdata/'):
    st_stocks = rqdatac.is_st_stock(ticker, start_date=start, end_date=end)
    csv_file_path = os.path.join(root, file_name)
    st_stocks.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")
    return st_stocks

def get_suspended(start, end, file_name, ticker = rqdatac.all_instruments(type='CS')['order_book_id'].tolist(), root = '/Users/ella/rqdata/'):
    suspended_stocks = rqdatac.is_suspended(ticker, start_date=start, end_date=end)
    csv_file_path = os.path.join(root, file_name)
    suspended_stocks.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")
    return suspended_stocks
    

def get_less_1yr(dates, file_name, root = '/Users/ella/rqdata/'):
    all_stocks = rqdatac.all_instruments(type='CS')
    #设置空t x s表格
    less_1_yr = pd.DataFrame(False, index=dates, columns=all_stocks['order_book_id'])
    for date in dates:
        one_year_ago = (datetime.strptime(date, '%Y-%m-%d') - relativedelta(years=1)).strftime('%Y-%m-%d')
        newly_listed_stocks_daily = all_stocks[all_stocks['listed_date'] > one_year_ago]['order_book_id'].tolist()
        less_1_yr.loc[date][newly_listed_stocks_daily] = True
    #处理上市不满一年的情况1: 退市日期 - 上市日期  < 1年 
    all_stocks['listed_date'] = pd.to_datetime(all_stocks['listed_date'],  errors='coerce')
    all_stocks['de_listed_date'] = pd.to_datetime(all_stocks['de_listed_date'],  errors='coerce')
    all_stocks['is_one_year'] = all_stocks['de_listed_date'] - all_stocks['listed_date'] < pd.Timedelta(days=365)
    duration_less_1yr = all_stocks[all_stocks['is_one_year'] == True]
    duration_less_1yr = duration_less_1yr['order_book_id'].tolist()
    csv_file_path = os.path.join(root, file_name)
    less_1_yr.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")
    less_1_yr[duration_less_1yr] = True
    return less_1_yr

def get_industry_dummies(dates, ticker = rqdatac.all_instruments('CS').order_book_id):
    #下载每日的行业数据
    all_industry_dummies = {}
    for current in dates:
        df_sw_ind = rqdatac.shenwan_instrument_industry(ticker, date = current)
        industry_dummies = pd.get_dummies(df_sw_ind['index_name'], prefix='industry')
        all_industry_dummies[current] = industry_dummies
    #将字典形式保存
    with open('industry_dummies_dict.pickle', 'wb') as file:
        pickle.dump(all_industry_dummies, file)

    return all_industry_dummies
    

#获取时间类
def get_dates(start, end):
    return pd.date_range(start=start, end=end, freq='B').strftime('%Y-%m-%d').tolist()

#读数据
def read(file_name, root = '/Users/ella/rqdata/'):
    csv_file_path = os.path.join(root, file_name)
    df = pd.read_csv(csv_file_path)
    return df

#整合条件类
def replace_na(factor, st, suspended, less_1yr):
    filter = st | suspended | less_1yr
    factor = factor.mask(filter)
    return factor

#标准化类
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

#行业市值中性化
def clean(dates, industry_dict, market_cap_df, factor_standardized, file_name, root = '/Users/ella/rqdata/'):
    resid_df = pd.DataFrame()
    for date in dates:
        #获取每日数据
        daily_ind = industry_dict[date]
        daily_ind['mark_value'] = market_cap_df.loc[date]
        daily_fac = factor_standardized.loc[date]
        daily_fac.name = daily_fac.name.strftime('%Y-%m-%d')
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
    #再次标准化
    resid_df = remove_extreme_and_standardize(resid_df)
    csv_file_path = os.path.join(root, file_name)
    resid_df.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")
    return resid_df

 






start = '20240601'
end = '20240606'
"""
factor = 'pcf_ratio_total_lyr'
dates = get_dates(start, end)
factor = get_factor(factor, start, end, 'fac')

df1 =  get_st(start, end, 'df1')
df2 =  get_suspended(start, end, 'df2')
df3 =  get_less_1yr(dates,'df3')
df4 = get_market_cap(start, end, 'df4')
ind = get_industry_dummies(dates)

factor = replace_na(factor, df1, df2, df3)
factor_standardized = remove_extreme_and_standardize(factor)

result = clean(dates, ind, df4,factor_standardized, 'result')
print(result)
"""


#——————————————————————————————————因子评价————————————————————————————————————————————
#下载数据类:

def get_return(start, end,period, file_name, ticker = rqdatac.all_instruments(type='CS')['order_book_id'].tolist(), root = '/Users/ella/rqdata/', field = 'close'):
    end_date = pd.to_datetime(end, format='%Y%m%d')
    new_end_date = end_date + pd.tseries.offsets.BDay(period)
    new_end = new_end_date.strftime('%Y%m%d')
    price = rqdatac.get_price(ticker, start_date=start, fields=field, end_date=new_end)[field].unstack(level='order_book_id')
    returns_df = price.pct_change(period)
    csv_file_path = os.path.join(root, file_name)
    returns_df.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")

    returns_df = pd.read_csv(csv_file_path, index_col=0)
    return returns_df
#def align(df1, df2):


#计算因子评价
def calc_icir(period, returns_df, result, option, ic_values = []):
    dates = pd.to_datetime(returns_df.index)
    for date in dates:
        future_date = date + pd.tseries.offsets.BDay(period)
        date = date.strftime('%Y-%m-%d')
        future_date = future_date.strftime('%Y-%m-%d')
        if future_date in returns_df.index:
            factor_ranks = result.loc[date].rank()
            #factor_ranks.fillna(0, inplace=True)
            returns_ranks = returns_df.loc[future_date].rank()
            ic, _ = spearmanr(factor_ranks, returns_ranks) #_ = p_val
            ic_values.append({'date': date, 'IC': ic})
    if option == '1':
        ic_df = pd.DataFrame(ic_values)
        print("IC DF: \n", ic_df)
        return ic_df
        

    elif option == '2':
        ic_df = pd.DataFrame(ic_values)
        ir_df = ic_df['IC'].mean() / ic_df['IC'].std()
        print("IR = : \n", ir_df)
        return ir_df
    
    elif option == '3':
        ranked_df = result.apply(quantile_rank, axis=0)
        ranked_df = ranked_df.stack().reset_index()
        ranked_df.columns = ['date', 'order_book_id', 'gorup']
        ranked_df.set_index(['date','order_book_id'])
        print("RANKED TABLE: \n", ranked_df)
        
def quantile_rank(row):
    quantiles = pd.qcut(row, 5, labels=[1, 2, 3, 4, 5])
    return quantiles








def main():
    while True:
        print("\n Select an option for calculation:")
        print("1. Calculation ic")
        print("2. Calculation ir")
        print("3. 分层收益")
        print("0. Exit")

        
        user_input = input("Enter your choice (1/2/3/4): ")
        if user_input == '0':
            print("Exiting the program.")
            break
        #returns_df = get_return(start, end, 2, 'returns_df')
        returns_df = pd.read_csv('/Users/ella/rqdata/returns_df',index_col=0)
        #print(returns_df)
        result = pd.read_csv('/Users/ella/rqdata/result',index_col=0)
        result = result.T
        result.index.name = 'date'
        result.dropna(axis = 1, inplace = True)
        #print("result: \n", result)
        returns_df, result = returns_df.align(result, join='inner', axis=0)  # 对齐行
        returns_df, result = returns_df.align(result, join='inner', axis=1)  # 对齐列
        #print("Factor: ", result)
        #print("Return: ", returns_df)
        


        result = calc_icir(2, returns_df, result, user_input)
        print(result)

if __name__ == "__main__":
    main()















