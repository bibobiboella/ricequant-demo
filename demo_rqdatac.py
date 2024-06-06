import rqdatac
import datetime
import pandas as pd
from scipy.stats import spearmanr
rqdatac.init()
#print(rqdatac.get_price('000002.XSHE', start_date=pd.Timestamp("20240101"), end_date=datetime.datetime(2024,2,1)))
#rqdatac.user.get_quota()
#value = rqdatac.get_price('000001.XSHE', start_date=20240101, end_date=20240201)
#value.to_csv('test.csv') #下载成csv格式


#current_performance
#fields = 财务指标数据字典 - 快报数据
#print(rqdatac.get_pit_financials_ex(fields=['revenue','net_profit'], start_quarter='2018q2', end_quarter='2018q3',order_book_ids=['000001.XSHE','000048.XSHE']))
#print(rqdatac.current_performance('000004.XSHE',quarter='2017q4',fields = ['basic_eps'], interval='2q'))

#performance_forecast
#fields = 财务指标数据字典 - 业绩预告数据
#业绩预告主要用来调取公司对即将到来的财务季度的业绩预期的信息。有时同一个财务季度会有多条记录，分别是季度预期和累计预期（即本年至今）。
#print(rqdatac.performance_forecast(['000001.XSHE','000006.XSHE'],fields=['forecast_description','forecast_earning_floor']))

#get_factor
#基础财务数据
#print(rqdatac.get_factor(['000001.XSHE','000002.XSHE'],'debt_to_equity_ratio',start_date='20180102',end_date='20180103'))
start = '20240103'
end = '20240601'
period = 20
start_date = pd.to_datetime(start, format='%Y%m%d')
end_date = pd.to_datetime(end, format='%Y%m%d')
new_start_date = start_date + pd.tseries.offsets.BDay(period)
new_end_date = end_date + pd.tseries.offsets.BDay(period)
new_start = new_start_date.strftime('%Y%m%d')
new_end = new_end_date.strftime('%Y%m%d')
factor =  'debt_to_equity_ratio' #'pe_ratio'


#ticker = list(rqdatac.all_instruments(market = 'cn')['order_book_id'])[:100]
#print(ticker)
#ticker = rqdatac.get_industry('银行')
ticker = rqdatac.all_instruments('CS').order_book_id

df = rqdatac.get_factor(ticker, factor, start_date=start, end_date=end)
unstack_df = df[factor].unstack(level='order_book_id')
nan_columns = unstack_df.isna().any()
#print("Columns with all values as NaN:", nan_columns[nan_columns == True])
#unstack_df.drop(columns=nan_columns[nan_columns == True].index)



#get price
price = rqdatac.get_price(ticker, start_date=start, fields='close', end_date=new_end)['close'].unstack(level='order_book_id')
returns_df = price.pct_change(period)
#print("Price: ", price)


factor_df = rqdatac.get_factor(ticker, factor, start_date=start, end_date=end)
factor_unstack_df = factor_df[factor].unstack(level='order_book_id').drop(columns=nan_columns[nan_columns == True].index)

ic_values = []
dates = unstack_df.index


#对齐!!
returns_df, factor_unstack_df = returns_df.align(factor_unstack_df, join='inner', axis=0)  # 对齐行
returns_df, factor_unstack_df = returns_df.align(factor_unstack_df, join='inner', axis=1)  # 对齐列
#print("Factor: ", factor_unstack_df)
#print("Return: ", returns_df)

for date in dates:
    future_date = date + pd.tseries.offsets.BDay(period)
    if future_date in returns_df.index:
        factor_ranks = factor_unstack_df.loc[date].rank()
        returns_ranks = returns_df.loc[future_date].rank()
        ic, _ = spearmanr(factor_ranks, returns_ranks) #_ = p_val
        ic_values.append({'date': date, 'IC': ic})
ic_df = pd.DataFrame(ic_values)
print("IC DF: \n", ic_df)

ir_df = ic_df['IC'].mean() / ic_df['IC'].std()
print("IR = : \n", ir_df)


#---------- sorting table------------

def quantile_rank(row):
    quantiles = pd.qcut(row, 5, labels=[1, 2, 3, 4, 5])
    return quantiles
ranked_df = factor_unstack_df.apply(quantile_rank, axis=1)
ranked_df = ranked_df.stack().reset_index()
ranked_df.columns = ['date', 'order_book_id', 'gorup']
ranked_df.set_index(['date','order_book_id'])
print("RANKED TABLE: \n", ranked_df)
ranked_df.hist(figsize=(12,6),bins=20)
#import alphalens
#alphalens.utils.get_clean_factor_and_forward_returns()
