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
import matplotlib.dates as mdates
import time
rqdatac.init()
root = '/Users/ella/test/'
start = '20120101'
end = '20240630'
period = 5
ticker_name_type = 'order_book_id'
#——————————————————————————————————因子清洗————————————————————————————————————————————
#下载数据类:
def get_factor(factors, start, end, file_name, ticker = rqdatac.all_instruments(type='CS')[ticker_name_type].tolist(), root = root):
    """
    [input]
    factor = 因子字段
    起始和结束日期
    file_name = 想要保存为csv的文件名
    [output]
    csv文件
    获取每日制定因子值并保存为csv
    """
    factor = rqdatac.get_factor(ticker, factors, start_date=start, end_date=end).reset_index()
    factor = factor.pivot(index='date', columns=ticker_name_type, values=factors)
    csv_file_path = os.path.join(root, file_name)
    factor.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")


def get_turnover_rate(start, end, file_name = 'turnover', ticker = rqdatac.all_instruments(type='CS')[ticker_name_type].tolist(), root = root):
    """
    [input]
    起始和结束日期
    file_name = 想要保存为csv的文件名
    [output]
    csv文件
    获取每日换手率因子值并保存为csv
    """
    turnover = rqdatac.get_turnover_rate(ticker, start, end, fields='today')
    turnover = turnover.reset_index().pivot(index='tradedate', columns=ticker_name_type, values='today')
    turnover.index.name = 'date'
    csv_file_path = os.path.join(root, file_name)
    turnover.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")

def get_market_cap(start, end, file_name, ticker = rqdatac.all_instruments(type='CS')[ticker_name_type].tolist(), market_factor = 'market_cap_2', root = root):
    """
    [input]
    起始和结束日期
    file_name = 想要保存为csv的文件名
    [output]
    csv文件
    获取每日股票log市值并保存为csv
    """
    market_cap_df = rqdatac.get_factor(ticker, market_factor, start_date=start, end_date=end).reset_index()
    #log(市值)
    market_cap_df['log_market_cap'] = np.log(market_cap_df['market_cap_2'])
    #market_cap_df.drop('market_cap_2', axis=1)
    market_cap_df = market_cap_df.pivot(index='date', columns=ticker_name_type, values='log_market_cap')
    csv_file_path = os.path.join(root, file_name)
    market_cap_df.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")

def get_st(start, end, file_name, ticker = rqdatac.all_instruments(type='CS')[ticker_name_type].tolist(), root = root):
    """
    [input]
    起始和结束日期
    file_name = 想要保存为csv的文件名
    [output]
    csv文件
    获取每日的ST股并保存为csv
    """
    st_stocks = rqdatac.is_st_stock(ticker, start_date=start, end_date=end)
    csv_file_path = os.path.join(root, file_name)
    st_stocks.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")

def get_suspended(start, end, file_name, ticker = rqdatac.all_instruments(type='CS')[ticker_name_type].tolist(), root = root):
    """
    [input]
    起始和结束日期
    file_name = 想要保存为csv的文件名
    [output]
    csv文件
    获取每日被停牌的股票并保存为csv
    """
    suspended_stocks = rqdatac.is_suspended(ticker, start_date=start, end_date=end)
    csv_file_path = os.path.join(root, file_name)
    suspended_stocks.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")
    

def get_less_1yr(dates, file_name, root = root):
    """
    [input]
    dates = 一个时间范围的list
    file_name = 想要保存为csv的文件名
    [output]
    csv文件
    获取动态的上市少于一年的股票并保存为csv
    """
    all_stocks = rqdatac.all_instruments(type='CS')
    #设置空t x s表格
    less_1_yr = pd.DataFrame(False, index=dates, columns=all_stocks[ticker_name_type])
    for date in dates:
        #查看每一天的时间点上上市少于一年的股票
        one_year_ago = (datetime.strptime(date, '%Y-%m-%d') - relativedelta(years=1)).strftime('%Y-%m-%d')
        newly_listed_stocks_daily = all_stocks[all_stocks['listed_date'] > one_year_ago][ticker_name_type].tolist()
        less_1_yr.loc[date][newly_listed_stocks_daily] = True
    #处理上市不满一年的情况: 退市日期 - 上市日期  < 1年 
    all_stocks['listed_date'] = pd.to_datetime(all_stocks['listed_date'],  errors='coerce')
    all_stocks['de_listed_date'] = pd.to_datetime(all_stocks['de_listed_date'],  errors='coerce')
    all_stocks['is_one_year'] = all_stocks['de_listed_date'] - all_stocks['listed_date'] < pd.Timedelta(days=365)
    duration_less_1yr = all_stocks[all_stocks['is_one_year'] == True]
    duration_less_1yr = duration_less_1yr[ticker_name_type].tolist()
    less_1_yr[duration_less_1yr] = True
    csv_file_path = os.path.join(root, file_name)
    less_1_yr.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")

def get_industry_dummies(dates = rqdatac.get_trading_dates(start, end, market = 'cn'), ticker = rqdatac.all_instruments('CS').order_book_id):
    """
    [input]
    dates = 一个时间范围的list
    [output]
    pickle
    获取动态行业分类数据并以dictionary的形式保存本地
    """
    #下载每日的行业分类数据
    all_industry_dummies = {}
    for current in dates:
        df_sw_ind = rqdatac.shenwan_instrument_industry(ticker, date = current)
        industry_dummies = pd.get_dummies(df_sw_ind['index_name'], prefix='industry')
        all_industry_dummies[current] = industry_dummies
    #将字典形式保存
    with open(os.path.join(root, 'industry_dummies_dict.pickle'), 'wb') as file:
        pickle.dump(all_industry_dummies, file)
    print("已成功保存industry_dummies_dict")
    

#获取时间类
def get_dates(start, end):
    """
    获取日期(虽然好像没用上)
    """
    return pd.date_range(start=start, end=end, freq='B').strftime('%Y-%m-%d').tolist()

#读数据
def read(file_name, root = root):
    """
    读数据(虽然好像也没用上)
    """
    csv_file_path = os.path.join(root, file_name)
    df = pd.read_csv(csv_file_path)
    return df

#整合条件类
def replace_na(factor, st, suspended, less_1yr):
    """
    [input]
    因子值, st股票,停牌股票, 小于一年的股票
    [output]
    因子值df
    将不满足条件整合并替换掉符合筛选条件的股票因子值为na
    """
    filter = st | suspended | less_1yr
    factor = factor.mask(filter)
    return factor

#标准化类
def remove_extreme_and_standardize(df, file_name, n = 3, root = root):
    """
    [input]
    df = 因子值
    [output]
    标准化过后的因子值df
    MAD去极值并标准化因子值
    """
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

    csv_file_path = os.path.join(root, f'{file_name}')
    df_standardized.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")
    return df_standardized

#行业市值中性化,并取残差作为新的因子值
def clean(dates, industry_dict, market_cap_df, factor_standardized, file_name, root = root):
    """
    [input]
    dates = 一个时间范围的list
    标准化后的因子值, st股票,停牌股票, 小于一年的股票
    [output]
    因子的残差值df
    对标准化后的因子进行逐日回归,并取残差值作为新的因子值
    """
    resid_df = pd.DataFrame()
    #price = price[factor_standardized.index[0]:factor_standardized.index[-1]]
    market_cap_df = market_cap_df[factor_standardized.index[0]:factor_standardized.index[-1]]
    industry_dict = {key: value for key, value in industry_dict.items() if key in dates}
    factor_standardized = factor_standardized.T[list(industry_dict.keys())].T
    dates = list(factor_standardized.index)
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
        X = sm.add_constant(X) 
        model = sm.OLS(y, X.astype(float)).fit()
        #获取因子残差作为结果, 并设置列名称 = 日期
        resid_series = pd.Series(model.resid, name = date)
        # 将残差添加到resid_df中，按列合并，并自动对齐索引
        resid_df = pd.concat([resid_df, resid_series], axis=1)

    print("逐日回归结果: \n", resid_df)
    #将残差值结果再次标准化
    resid_df = remove_extreme_and_standardize(resid_df, f'{file_name}_standard')
    csv_file_path = os.path.join(root, f'{file_name}_regress_result')
    resid_df.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")
    return resid_df

#检查类
def check_file_exists(file_path):
    """
    检查一个路径下是否存在某个文件
    """
    return os.path.isfile(file_path)



#——————————————————————————————————因子评价————————————————————————————————————————————
#下载数据类:
def get_price(start, end, file_name = 'get_price', ticker = rqdatac.all_instruments(type='CS')[ticker_name_type].tolist(), root = root):
    """
    [input]
    起始和结束日期
    period = 需要计算return时的period
    [output]
    csv文件
    获取每日股票价格并保存为csv
    """
    #end_date = pd.to_datetime(end, format='%Y%m%d')
    #new_end_date = end_date + pd.tseries.offsets.BDay(period)
    #new_end = new_end_date.strftime('%Y%m%d')
    price = rqdatac.get_price(ticker, start_date=start, fields='open', end_date=end)['open'].reset_index().pivot(index='date', columns='order_book_id', values='open')
    csv_file_path = os.path.join(root, file_name)
    price.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")

#待删一个, get_price重复
def get_return(start, end, file_name = 'returns_df', period = period, ticker = rqdatac.all_instruments(type='CS')[ticker_name_type].tolist(), root = root, field = 'close'):
    """
    [input]
    起始和结束日期
    [output]
    csv文件
    计算period长度的百分比变化为return并保存为csv
    """
    #end_date = pd.to_datetime(end, format='%Y%m%d')
    #new_end_date = end_date + pd.tseries.offsets.BDay(period)
    #new_end = new_end_date.strftime('%Y%m%d')
    price = rqdatac.get_price(ticker, start_date=start, fields=field, end_date=end)[field].unstack(level=ticker_name_type)
    csv_file_path = os.path.join(root, 'get_price')
    price.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")


    returns_df = price.pct_change(period)
    csv_file_path = os.path.join(root, file_name)
    returns_df.to_csv(csv_file_path)
    print(f"数据已保存为 {csv_file_path}")

#筛选股票
def filter_stocks():
    """
    可以输入指数成分的字段来筛选只属于那个指数成分的股票
    """
    user_input = input("请输入想要筛选的Index (不要带字符引号, 例: 000300.XSHG 沪深300): ")
    print(f"{user_input}已存在") if check_file_exists(os.path.join(root, f'{user_input}.pickle')) else get_filter_stocks(user_input , start, end)

    with open(os.path.join(root, f'{user_input}.pickle'), 'rb') as file:
        filter = pickle.load(file)
    result = pd.read_csv(os.path.join(root, 'result'),index_col=0)
    result = result.T
    result.index.name = 'date'
    filtered_df = pd.DataFrame()
    # 利用获取的指数成分股字典中的日期和股票列表进行过滤
    for date, stocks in filter.items():
        if date in result.index:
            try:
                # 选择特定日期的行并重新索引以仅保留指定股票列
                filtered_data = result.loc[date, stocks]
                # 将过滤后的数据添加到新的dataframe中
                filtered_df = pd.concat([filtered_df, filtered_data.to_frame().T], axis=0)
            except KeyError:
                # 跳过不存在的股票
                available_stocks = [stock for stock in stocks if stock in result.columns]
                if available_stocks:
                    filtered_data = result.loc[date, available_stocks]
                    filtered_df = pd.concat([filtered_df, filtered_data.to_frame().T], axis=0)
    return filtered_df


def download_data():
    """
    下载数据,检查如果已存在就不用重新下载
    """
    user_input_factor_1 = input("是否想查看换手率(y/N): ")
    if user_input_factor_1 == 'y':
        if not check_file_exists(os.path.join(root, 'turnover')):
            get_turnover_rate(start, end)
        else:
            print("turnover已存在")
        factor = pd.read_csv(os.path.join(root, 'turnover'), index_col=0)
    elif user_input_factor_1 == 'N':
        user_input_factor = input("Enter a factor: ")
        if not check_file_exists(os.path.join(root, f'{user_input_factor}')):
            get_factor(user_input_factor, start, end, f'{user_input_factor}')
        else:
            print(f'{user_input_factor}已存在')
        factor = pd.read_csv(os.path.join(root, f'{user_input_factor}'), index_col=0)
    print("st已存在") if check_file_exists(os.path.join(root, 'df1')) else get_st(start, end, 'df1')
    print("停牌已存在") if check_file_exists(os.path.join(root, 'df2')) else get_suspended(start, end, 'df2')
    dates = list(factor.index)
    print("上市小于一年已存在") if check_file_exists(os.path.join(root, 'df3')) else get_less_1yr(dates,'df3')
    print("市值已存在") if check_file_exists(os.path.join(root, 'df4')) else get_market_cap(start, end, 'df4')
    print("industry dummies已存在") if check_file_exists(os.path.join(root, 'industry_dummies_dict.pickle')) else get_industry_dummies(dates)
    with open(os.path.join(root, 'industry_dummies_dict.pickle'), 'rb') as file:
        ind = pickle.load(file)
    print(list(ind.keys())[-5:])
    print("price已存在") if check_file_exists(os.path.join(root, 'get_price')) else get_price(start, end)
    print("returns已存在") if check_file_exists(os.path.join(root, 'returns_df')) else get_return(start, end)

def read_data():
    """
    读取数据并进行因子清洗
    """
    user_input_factor_1 = input("是否想查看换手率(y/N): ")
    if user_input_factor_1 == 'N':
        user_input_factor = input("Enter a factor: ")
        if check_file_exists(os.path.join(root, f'{user_input_factor}')):
            factor = pd.read_csv(os.path.join(root, f'{user_input_factor}'), index_col=0) #因子
        else:
            print(f'{user_input_factor}不存在,请先在1下载')
        #print("因子: \n", factor)
    elif user_input_factor_1 == 'y':
        factor = pd.read_csv(os.path.join(root, 'turnover'), index_col=0) #因子
        #print("换手率因子: \n", factor)
    with open(os.path.join(root, 'industry_dummies_dict.pickle'), 'rb') as file:
        ind = pickle.load(file)
    #print("ind: \n", len(ind))
    df1 = pd.read_csv(os.path.join(root, 'df1'), index_col=0) #st
    df1 = df1[factor.index[0]:factor.index[-1]]
    #print("st: \n", df1)
    df2 = pd.read_csv(os.path.join(root, 'df2'), index_col=0) #停牌
    df2 = df2[factor.index[0]:factor.index[-1]]
    #print("停牌: \n", df2)
    df3 = pd.read_csv(os.path.join(root, 'df3'), index_col=0) #不满一年
    df3 = df3[factor.index[0]:factor.index[-1]]
    #print("不满一年: \n", df3)
    df4 = pd.read_csv(os.path.join(root, 'df4'), index_col=0) #市值
    df4 = df4[factor.index[0]:factor.index[-1]]
    #print("市值: \n", df4)
    #factor = replace_na(factor, df1, df2, df3)
    #print("替换na: \n", factor)
    if user_input_factor_1 == 'N':
        if 'ths' in user_input_factor:
            with open(os.path.join(root, 'tickers.pickle'), 'rb') as file:
                tickers = pickle.load(file)
            factor = factor.rename(columns=tickers)
            factor = replace_na(factor, df1, df2, df3)
            print("替换na: \n", factor)
        #factor = np.sin(factor) * np.log(abs(factor) + 1)
        factor_standardized = remove_extreme_and_standardize(factor, f'{user_input_factor}_standard') 
        factor_standardized = pd.read_csv(os.path.join(root, f'{user_input_factor}_standard'), index_col=0)
        #print("标准化因子: \n", factor_standardized)
        dates = list(factor.index)
        df4 = df4[dates[0]:dates[-1]]
        ind = {key: value for key, value in ind.items() if key in dates}
        clean(dates, ind, df4, factor, f'{user_input_factor}')
    if user_input_factor_1 == 'y':
        factor_standardized = remove_extreme_and_standardize(factor, 'turnover_standard')
        factor_standardized = pd.read_csv(os.path.join(root, 'turnover_standard'), index_col=0)
        print("标准化因子: \n", factor_standardized)
        dates = list(factor.index)
        clean(dates, ind, df4,factor_standardized, 'turnover')

#计算因子评价
def calc_icir(period, returns_df, result, factor_name,  option, ic_values = []):
    """
    进行因子评价
    """
    if option == '3':
        #dates = pd.to_datetime(returns_df.index)
        price = pd.read_csv(os.path.join('/Users/ella/Desktop/hjx因子1/hjx因子1/因子', 'labels'), index_col=0)
        print(price)
        #pd.read_csv(os.path.join(root, 'get_price'),index_col=0)
        
        dates = list(result.index)
        #对齐日期
        price = price[result.index[0]:result.index[-1]]
        print("price: \n", price)
        #with open(os.path.join(root, 'tickers.pickle'), 'rb') as file:
            #tickers_dict = pickle.load(file)
        #result = result.rename(columns=tickers_dict) #thsindex1_2022
        print("result: \n", result)
        stock_r = price.pct_change(period).shift(-(period + 1))
        index_r = pd.read_csv(os.path.join('/Users/ella/test/', 'index_r'),index_col=0)[stock_r.index[0]: stock_r.index[-1]]['000905.XSHG']
        excess_r = stock_r.sub(index_r, axis = 0)
        IC_series = result.corrwith(excess_r, axis=1, method="spearman").dropna()
        print("IC Mean: \n", IC_series.mean())
        ir = IC_series.mean() / IC_series.std()
        print("IR = : \n", ir)
        with open('/Users/ella/test/ths_icir_result.txt', 'a', encoding='utf-8') as file:
            file.write(f"{factor_name}: \nIC Mean: {IC_series.mean()} \nIR = : {ir} \n")
        print("数据已追加到文件 /Users/ella/test/ths_icir_result.txt")
        return IC_series
        
    
    elif option == '5':
        r = []
        #price = pd.read_csv(os.path.join(root, 'get_price'),index_col=0)
        price = pd.read_csv(os.path.join('/Users/ella/Desktop/hjx因子1/hjx因子1/因子', 'labels'), index_col=0)
        price.rename(columns=lambda x: x[:6], inplace=True)
        price = price[result.index[0]:result.index[-1]]
        user_input = input("是否需要筛选特定指数成分股 (y/N): ")
        if user_input == 'y':
            print("指数成分股列表已存在") if check_file_exists(os.path.join(root, f'{user_input}.pickle')) else get_filter_stocks(input("Enter Index (不要带字符引号, 例: '000300.XSHG' 沪深300): ") , start, end)
            result = filter_stocks()
        #对齐columns
        price = price[list(set(price.columns) & set(result.columns))]
        result = result[list(set(price.columns) & set(result.columns))]
        layer_ind = result.apply(lambda x: pd.qcut(x.rank(method="first"), q=10, labels=[10,9,8,7,6,5,4,3,2,1]), axis=1)#按理说不需要drop
        dates = pd.Series(layer_ind.index) #原来是price的index
        index_r = pd.read_csv(os.path.join('/Users/ella/test/', 'index_r_daily'),index_col=0)[layer_ind.index[0]: layer_ind.index[-1]]['000905.XSHG']
        for i in range(0, len(dates), period):
            fac_date = dates[0]
            buy_date = dates[1]
            dates = dates.shift(-period)
            facend_date = dates[0]
            buyend_date = dates[1]
            if facend_date == None:
                break
            
            #计算每日收益率: period日期区间的diff除以t0的价格
            period_daily_return = (price.loc[buy_date: buyend_date].diff()/price.loc[buy_date])[1:] 
            #获取t0的所有股票以及分组label
            period_label = layer_ind.loc[fac_date]
            #筛选出每组的每日股票平均收益率
            group_1_returns = pd.concat([period_daily_return[period_label[period_label == 1].index].mean(axis=1)]).to_frame(name=1)
            group_2_returns = pd.concat([period_daily_return[period_label[period_label == 2].index].mean(axis=1)]).to_frame(name=2)
            group_3_returns = pd.concat([period_daily_return[period_label[period_label == 3].index].mean(axis=1)]).to_frame(name=3)
            group_4_returns = pd.concat([period_daily_return[period_label[period_label == 4].index].mean(axis=1)]).to_frame(name=4)
            group_5_returns = pd.concat([period_daily_return[period_label[period_label == 5].index].mean(axis=1)]).to_frame(name=5)
            group_6_returns = pd.concat([period_daily_return[period_label[period_label == 6].index].mean(axis=1)]).to_frame(name=6)
            group_7_returns = pd.concat([period_daily_return[period_label[period_label == 7].index].mean(axis=1)]).to_frame(name=7)
            group_8_returns = pd.concat([period_daily_return[period_label[period_label == 8].index].mean(axis=1)]).to_frame(name=8)
            group_9_returns = pd.concat([period_daily_return[period_label[period_label == 9].index].mean(axis=1)]).to_frame(name=9)
            group_10_returns = pd.concat([period_daily_return[period_label[period_label == 10].index].mean(axis=1)]).to_frame(name=10)
            r.append(pd.concat([group_1_returns, group_2_returns, group_3_returns, group_4_returns, group_5_returns, group_6_returns, group_7_returns, group_8_returns, group_9_returns, group_10_returns], axis=1))
        
        all_average_returns_df = pd.concat(r)
        all_average_returns_df = all_average_returns_df.cumsum()
        
        rank = all_average_returns_df.iloc[-1].rank(method='min').astype(int)
        correlation, p_value = spearmanr(rank.index, rank.values)
        with open('/Users/ella/test/ths_icir_result.txt', 'a', encoding='utf-8') as file:
            file.write(f"{factor_name}: \n分层单调性: {correlation} \n")
        print("数据已追加到文件 /Users/ella/test/ths_icir_result.txt")
    










        csv_file_path = os.path.join(root, f'{factor_name}_group_avg_returns')
        all_average_returns_df.to_csv(csv_file_path)
        print(f"数据已保存为 {csv_file_path}")
        print("\n 分层收益率结果: \n",all_average_returns_df)
        user_input = input("enter a time range (all/custom): ")
        all_average_returns_df.index = pd.to_datetime(all_average_returns_df.index)
        if user_input == 'custom':
            start_date = input("start: ")
            end_date = input("end: ")
            all_average_returns_df = all_average_returns_df[(all_average_returns_df.index >= start_date) & (all_average_returns_df.index <= end_date)]
            #df = df.cumsum()
        #elif user_input == 'all':
            #df = df.cumsum()
        #df.index = pd.to_datetime(df.index)
        plt.figure(figsize=(12, 6))
        for column in all_average_returns_df.columns:
            plt.plot(all_average_returns_df.index, all_average_returns_df[column], label=column, alpha=0.7)
        plt.xlabel('Date')
        plt.ylabel('Group Return')
        plt.title(f'Line Graph for Each Label {period} days return')
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(root, f'{factor_name}.png'))
        print(f"图片已保存为 {os.path.join(root,  f'{factor_name}.png')}")
        


        
def quantile_rank(row):
    non_nan_row = row.dropna()
    quantiles = np.unique(np.quantile(non_nan_row, [0.2, 0.4, 0.6, 0.8]))
    bins = [-np.inf, *quantiles, np.inf]
    labels = pd.cut(non_nan_row, bins=bins, labels=[1, 2, 3, 4, 5])
    return labels

#获取每日指数成分列表
def get_filter_stocks(index_code,start, end):
    #'000300.XSHG'  # 沪深300指数代码
    constituents = rqdatac.index_components(index_code,  start_date = start,end_date =end)
    constituents = pd.Series(constituents).rename(lambda x: x.strftime('%Y-%m-%d')).to_dict()
    with open(os.path.join(root, f'{index_code}.pickle'), 'wb') as file:
        pickle.dump(constituents, file)
    print("数据已保存成功")



def main():
    while True:
        print("\n Select an option for calculation:")
        print("1. 下载需要的数据")
        print("2. 读数据&因子清洗")
        print("3. 计算 ic/ir")
        print("5. 分层收益&画图")
        print("0. Exit")

        
        user_input = input("Enter your choice (1/2/3/5/0) : ")
        if user_input == '0':
            print("Exiting the program.")
            break
        elif user_input == '1':
            
            download_data()
        elif user_input == '2':
            read_data()

        elif user_input != '2' or user_input != '1':
            returns_df = pd.read_csv(os.path.join(root, 'returns_df'),index_col=0)
            user_input_factor = input("Enter a factor: ")
            result = pd.read_csv(os.path.join(root, f'{user_input_factor}'),index_col=0) #regress_result
            with open(os.path.join(root, 'tickers.pickle'), 'rb') as file:
                tickers = pickle.load(file)
            result = result.rename(columns=tickers)
            result = result#.T
            print("result: ", result)
            #result.index.name = 'date'
            
            #returns_df, result = returns_df.align(result, join='inner', axis=0)  # 对齐行
            #returns_df, result = returns_df.align(result, join='inner', axis=1)  # 对齐列
            #result = calc_icir(period, returns_df, result, user_input_factor,  user_input)

if __name__ == "__main__":
    main()






