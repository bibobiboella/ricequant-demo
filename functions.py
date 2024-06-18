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
import matplotlib.pyplot as plt
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
def remove_extreme_and_standardize(df, n = 3, root = '/Users/ella/rqdata/'):
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

    csv_file_path = os.path.join(root, 'standard')
    df_standardized.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")
    return df_standardized

#行业市值中性化
def clean(dates, industry_dict, market_cap_df, factor_standardized, file_name, root = '/Users/ella/rqdata/'):
    resid_df = pd.DataFrame()
    for date in dates:
        #获取每日数据
        daily_ind = industry_dict[date]
        daily_ind['mark_value'] = market_cap_df.loc[date]
        daily_fac = factor_standardized.loc[date]
        #合并每日所需X=: 行业+市值
        merged = daily_ind.merge(daily_fac, left_index=True, right_index=True, how='inner').dropna()
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

 


"""
with open('industry_dummies_dict.pickle', 'rb') as file:
    ind = pickle.load(file)
df1 = pd.read_csv('/Users/ella/rqdata/df1', index_col=0)
#print(df1)
df2 = pd.read_csv('/Users/ella/rqdata/df2', index_col=0)
#print(df2)
df3 = pd.read_csv('/Users/ella/rqdata/df3', index_col=0)
#print(df3)
df4 = pd.read_csv('/Users/ella/rqdata/df4', index_col=0)
#print(df4)
factor = pd.read_csv('/Users/ella/rqdata/fac', index_col=0)
factor = replace_na(factor, df1, df2, df3)
#print(factor)
factor_standardized = pd.read_csv('/Users/ella/rqdata/standard', index_col=0)

factor_standardized = factor_standardized.T
factor_standardized.index.name = 'date'
#factor_standardized.dropna(axis = 1, inplace = True)
#print(factor_standardized)
dates = list(factor.index)[:-1]#.strftime('%Y-%m-%d')
#print("dates: \n", dates)
result = clean(dates, ind, df4,factor_standardized, 'result')
print(result)
"""
"""



start = '20120101'
end = '20240617'


factor = 'pe_ratio'
dates = get_dates(start, end)
#print(dates)
factor = get_factor(factor, start, end, 'fac')

df1 =  get_st(start, end, 'df1')
df2 =  get_suspended(start, end, 'df2')
df3 =  get_less_1yr(dates,'df3')
df4 = get_market_cap(start, end, 'df4')
ind = get_industry_dummies(dates)
factor = replace_na(factor, df1, df2, df3)
factor_standardized = remove_extreme_and_standardize(factor)
dates = list(factor_standardized.index.strftime('%Y-%m-%d'))
#print(dates)
#print(factor_standardized)
result = clean(dates, ind, df4,factor_standardized, 'result')
print(result)
"""





#——————————————————————————————————因子评价————————————————————————————————————————————
#下载数据类:
def get_price(start, end,period, file_name = 'get_price', ticker = rqdatac.all_instruments(type='CS')['order_book_id'].tolist(), root = '/Users/ella/rqdata/'):
    end_date = pd.to_datetime(end, format='%Y%m%d')
    new_end_date = end_date + pd.tseries.offsets.BDay(period)
    new_end = new_end_date.strftime('%Y%m%d')

    price = rqdatac.get_price(ticker, start_date=start, fields='close', end_date=new_end)['close'].unstack(level='order_book_id')
    csv_file_path = os.path.join(root, file_name)
    price.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")


def get_return(start, end,period, file_name, ticker = rqdatac.all_instruments(type='CS')['order_book_id'].tolist(), root = '/Users/ella/rqdata/', field = 'close'):
    end_date = pd.to_datetime(end, format='%Y%m%d')
    new_end_date = end_date + pd.tseries.offsets.BDay(period)
    new_end = new_end_date.strftime('%Y%m%d')
    price = rqdatac.get_price(ticker, start_date=start, fields=field, end_date=new_end)[field].unstack(level='order_book_id')
    csv_file_path = os.path.join(root, 'get_price')
    price.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")


    returns_df = price.pct_change(period)
    csv_file_path = os.path.join(root, file_name)
    returns_df.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")

    #returns_df = pd.read_csv(csv_file_path, index_col=0)
    #return returns_df
#def align(df1, df2):


#计算因子评价
def calc_icir(period, returns_df, result, option, ic_values = []):
    dates = pd.to_datetime(returns_df.index)
    for date in dates:
        future_date = date + pd.tseries.offsets.BDay(period)
        #print(future_date)
        date = date.strftime('%Y-%m-%d')
        future_date = future_date.strftime('%Y-%m-%d')
        if future_date in returns_df.index:
            factor_ranks = result.loc[date].rank()
            returns_ranks = returns_df.loc[future_date].rank()
            ic, _ = spearmanr(factor_ranks, returns_ranks, nan_policy='omit') #_ = p_val, omit means ignoring nan but still performs calculation
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
        #result = pd.read_csv('/Users/ella/rqdata/result',index_col=0)
        #print("result_3: \n", result)
        ranked_df = result.apply(quantile_rank, axis=1)
        print("RANKED TABLE: \n", ranked_df)
        layer_ind = result.apply(lambda x: pd.qcut(x.rank(method="first"), q=5, labels=range(5)), axis=1)
        print("layer_ind: \n",layer_ind)
        
        price = pd.read_csv('/Users/ella/rqdata/get_price',index_col=0)
        #print("price: ", price)
        days_after = 2
        dates = list(price.index)
        # 创建一个空的 DataFrame 来存储所有日期的结果
        all_average_returns_df = pd.DataFrame()
        for date in dates:
            # 计算结束日期
            end_date = (pd.to_datetime(date) + pd.offsets.BDay(days_after)).strftime('%Y-%m-%d')
            if end_date in dates:
                # 计算收益率
                price_start = price.loc[date]
                price_end = price.loc[end_date]
                returns = (price_end - price_start) / price_start
                # 将标签和收益率合并
                returns_df = returns.to_frame(name='returns')
                returns_df['label'] = layer_ind.loc[date]
                # 按标签计算平均收益率
                average_returns = returns_df.groupby('label')['returns'].mean()
                # 将结果转为 DataFrame，并设置索引为日期
                average_returns_df = average_returns.to_frame().T
                average_returns_df.index = [date]
                all_average_returns_df = pd.concat([all_average_returns_df, average_returns_df])
        csv_file_path = '/Users/ella/rqdata/group_avg_returns'
        all_average_returns_df.to_csv(csv_file_path)
        print(f"数据已保存为 {csv_file_path}")
        print("\n 分层收益率结果: \n",all_average_returns_df)
        




    elif option == '4':
        file_path_input = input("Enter filepath: ")
        df = pd.read_csv(os.path.join(file_path_input, 'group_avg_returns'), index_col=0)
        df.index = pd.to_datetime(df.index)
        # 画出折线图
        plt.figure(figsize=(12, 6))

        # 为每个列标签画出折线图
        for column in df.columns:
            plt.plot(df.index, df[column], label=column, alpha = 0.4)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Line Graph for Each Label')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(file_path_input, 'line_graph.png'))
    elif option == '9':
        with open('industry_dummies_dict.pickle', 'rb') as file:
            ind = pickle.load(file)
        file_path_input = input("Enter filepath: ")
        df1 = pd.read_csv(os.path.join(file_path_input, 'df1'), index_col=0) #st
        df2 = pd.read_csv(os.path.join(file_path_input, 'df2'), index_col=0) #停牌
        df3 = pd.read_csv(os.path.join(file_path_input, 'df3'), index_col=0) #不满一年
        df4 = pd.read_csv(os.path.join(file_path_input, 'df4'), index_col=0) #市值
        factor = pd.read_csv(os.path.join(file_path_input, 'fac'), index_col=0) #因子
        factor = replace_na(factor, df1, df2, df3)
        factor_standardized = remove_extreme_and_standardize(factor)
        factor_standardized = pd.read_csv(os.path.join(file_path_input, 'standard'), index_col=0)
        #print(factor_standardized)
        dates = list(factor.index)
        #print("dates: \n", dates)
        result = clean(dates, ind, df4,factor_standardized, 'result')
        #print("result; \n", result)
        



        
def quantile_rank(row):
    non_nan_row = row.dropna()
    quantiles = np.unique(np.quantile(non_nan_row, [0.2, 0.4, 0.6, 0.8]))
    bins = [-np.inf, *quantiles, np.inf]
    labels = pd.cut(non_nan_row, bins=bins, labels=[1, 2, 3, 4, 5])
    return labels







def main():
    while True:
        print("\n Select an option for calculation:")
        print("1. 计算 ic")
        print("2. 计算 ir")
        print("3. 分层收益")
        print("4. 画图")
        print("9. 读数据&因子清洗")
        print("0. Exit")

        
        user_input = input("Enter your choice (1/2/3/4/9/0) : ")
        if user_input == '0':
            print("Exiting the program.")
            break
        #get_return(start, end, 2, 'returns_df')
        returns_df = pd.read_csv('/Users/ella/rqdata/returns_df',index_col=0)
        #print(returns_df)
        result = pd.read_csv('/Users/ella/rqdata/result',index_col=0)
        result = result.T
        result.index.name = 'date'
        #print(result)
        #result.dropna(axis = 1, inplace = True)
        returns_df, result = returns_df.align(result, join='inner', axis=0)  # 对齐行
        returns_df, result = returns_df.align(result, join='inner', axis=1)  # 对齐列

        #price = pd.read_csv('/Users/ella/rqdata/get_price',index_col=0)
        #print("price: ", price)
        #print("Return: ", returns_df)
        


        result = calc_icir(2, returns_df, result, user_input)
        #return result

if __name__ == "__main__":
    main()











