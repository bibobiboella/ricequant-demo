#读数据/预处理
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import os
import pickle
import calc_return
import filter
import windus
import get_stock_rsk
import get_excess_result




root = '/Users/ella/test/画图/optimizer/'
combine = pd.read_parquet('/Users/ella/test/画图/optimizer/combine.parquet')
combine.rename(columns=lambda x: x[:6], inplace=True)
new_columns = ['000979', '600270']
combine[new_columns] = np.nan
combine = combine[dr.columns]

df = pd.read_feather(os.path.join(root, 'predict_alphaseeker_model1.feather'))
dr = df.pivot(index='dt', columns='code', values='predict')
dr.rename(columns=lambda x: x[:6], inplace=True)

bench_weight = pd.read_csv(os.path.join(root, 'zz500_weight'), index_col=0)

price = pd.read_csv('/Users/ella/test/画图/price', index_col=0)#['600967.XSHG']
price.rename(columns=lambda x: x[:6], inplace=True)
zz500_price = pd.read_csv(os.path.join(root, 'zz500_price'), index_col=0)
zz500_price.set_index('date', inplace=True)
zz500_pct = zz500_price.pct_change()[:-3]['open']
#改一下,不要使用dr.columns
dum_in_bench_df = pd.DataFrame(0, index=combine.index[:-3], columns=combine.columns)
for i in range(len(dum_in_bench_df)):
    # 获取当天的成分股列表
    components = list(bench_weight.iloc[i].dropna().index)
    # 更新当天的成分股状态
    dum_in_bench_df.iloc[i, dum_in_bench_df.columns.isin(components)] = 1

with open(os.path.join(root, 'bench_ind.pickle'), 'rb') as file:
        bench_ind = pickle.load(file)
        
with open(os.path.join('/Users/ella/test/画图/optimizer', 'dumindus_fullmark.pickle'), 'rb') as file:
    dumindus_fullmark = pickle.load(file)
#获取benchmark行业分类的weight
windus_weight = windus.get_windus(bench_ind, bench_weight)
bench_weight.rename(columns=lambda x: x[:6], inplace=True)

stock_rsk = pd.read_feather(os.path.join(root,'stock_rsk.feather'))
#2019-01-02 ~ 2024-05-22
#combine = pd.concat([combine.T, pd.DataFrame([{'dt': date, 'code': '001286'} for date in combine.index])], ignore_index=True)




#优化
import numpy as np
import pandas as pd
import cvxpy as cp
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#超额收益率
excess_returns_nonconstraint = {}
for year, returns in grouped_nonconstraint:
    excess_return = returns.sub(bench_r.get_group(year).values, axis = 0).cumsum()[-1]
    excess_returns_nonconstraint[str(year)] = excess_return
excess_returns_nonconstraint

#最大超额回撤率
#超额回撤
max_drawdown_nonocnstraint = {}
for year, returns in grouped_nonconstraint:
    excess_r = (1 + (returns - bench_r.get_group(year))).cumprod()
    peak = excess_r.cummax()
    drawdown = (peak - excess_r) / peak
    max_drawdown_nonocnstraint[year] = drawdown.max()
max_drawdown_nonocnstraint
#夏普比例
sharpe_ratios_nonconstraint = {}
for year, returns in grouped_nonconstraint:
    #excess_returns = returns - bench_r.get_group(year)
    excess_returns = returns.cumsum()[-1] - bench_r.get_group(year).cumsum()[-1]
    #average_excess_return = excess_returns.mean() * len(bench_r.get_group(year))
    standard_deviation = returns.std()
    sharpe_ratio = excess_returns / standard_deviation
    sharpe_ratios_nonconstraint[year] = sharpe_ratio
sharpe_ratios_nonconstraint

calmar_ratios_nonconstraint = {}
for year, returns in grouped_nonconstraint:
    return_year = returns.cumsum()[-1]
    down_year = max_drawdown_nonocnstraint[year]
    calmar_ratios_nonconstraint[year] = return_year / down_year
calmar_ratios_nonconstraint

stock_rsk = get_stock_rsk.get_stock_rsk()
f_expose = get_stock_rsk.get_f_expose(stock_rsk)

#combine = combine[:-3]
w_min = 0.0  # 权重最小值
w_max = 0.01  # 权重最大值
delta = 0.005  # 换手率限制
w_mininbench = 0.4 #成分股比例限制
e = 0.1 #行业偏离
rsk_e = 0.1
#lmabda = 40
n_max = 100
previous_weights = pd.Series([0] * len(combine.columns), index=combine.columns)
optimal_weights_df = pd.DataFrame(index=combine.index, columns=combine.columns)
#保证行业分类的顺序一致
windus_weight.sort_index(axis=1, inplace=True)
for t in combine.index.strftime('%Y-%m-%d'):
    #try:
    print(f"Processing date: {t}")
    r = combine.loc[t]#.fillna(0).values 
    non_nan_indices = pd.Series(False, index=r.index)
    non_nan_indices.loc[r[~r.isna()].index.intersection(stock_rsk.loc[t].fillna(0).index)] = True

    r_non_nan = r[non_nan_indices[non_nan_indices == True].index].values#r[non_nan_indices].values  #只对非 NaN 值进行优化
    #initial要求的解的变量
    weights = cp.Variable(len(r_non_nan))
    # 定义约束条件
    dum_in_bench = dum_in_bench_df.loc[t][non_nan_indices].values
    #port_ben = dum_in_bench_df.loc[t][non_nan_indices][dum_in_bench_df.loc[t][non_nan_indices] == 1]
    #windus_t = windus_weight.loc[t].dropna().values 
    windus_t = windus_weight.loc[t].fillna(0).values 
    dumindus_fullmark_t = dumindus_fullmark[t].fillna(0).values
    stock_rsk_fullmark = stock_rsk.loc[t].fillna(0).loc[non_nan_indices[non_nan_indices].index]
    stock_rsk_fullmark = stock_rsk_fullmark[~stock_rsk_fullmark.index.duplicated(keep='first')].values.T
    # 定义目标函数，只考虑预期收益
    objective = cp.Maximize(weights.T @ r_non_nan)#lmabda * weights.T @ stock_rsk_fullmark @ weights
    constraints = [
        cp.sum(weights) == 1,  # 权重总和为1
        weights >= w_min,  # 个股权重最小值
        weights <= w_max,  # 个股权重最大值
        cp.abs(weights - previous_weights[non_nan_indices]) <= delta, # 换手率限制
        cp.sum(cp.multiply(dum_in_bench, weights)) >= w_mininbench, #成分股权重限制 check
        #cp.sum(cp.abs(cp.multiply(port_ben.values, weights[np.where(dum_in_bench != 0)[0]]) - bench_weight.loc[t][port_ben.index])) <= 0.2,
        cp.matmul(dumindus_fullmark_t[:, non_nan_indices], weights) <= windus_t + e, #行业权重最大值 check
        cp.matmul(dumindus_fullmark_t[:, non_nan_indices], weights) >= windus_t - e, #行业权重最小值 check
        cp.matmul(stock_rsk_fullmark, weights) <= f_expose.loc[t] + rsk_e, #check
        cp.matmul(stock_rsk_fullmark, weights) >= f_expose.loc[t] - rsk_e, #check
        cp.sum(cp.pos(weights)) <= n_max
    ]
    #daily = stock_rsk_fullmark * (pd.Series([1/len(r_non_nan)] * len(r_non_nan)).values).sum()
#daily_sum = daily.sum(axis=1)
    problem = cp.Problem(objective, constraints)
    # 求解问题
    problem.solve(warm_start=True, solver=cp.ECOS, max_iters=500)
    if problem.status == cp.OPTIMAL or problem.status == cp.OPTIMAL_INACCURATE:
            optimal_weights = weights.value
            optimal_weights_df.loc[t, non_nan_indices] = optimal_weights
            previous_weights[non_nan_indices] = optimal_weights
    else:
        print("Optimization failed at time step", t)
        break
optimal_weights_df.index = optimal_weights_df.index.strftime('%Y-%m-%d')
csv_file_path = os.path.join(root, 'optimal_weights_df_40%_rsk0.1_e0.1')
optimal_weights_df.to_csv(csv_file_path)
print(f"数据已保存为 {csv_file_path}")
optimal_weights_df = optimal_weights_df[:-3]
result = calc_return.run_backtest(optimal_weights_df, price,  tax = 0.001, fee = 0.00012, slip = 0.0005, holding_period=5)[0]#.cumsum()
csv_file_path = os.path.join(root, 'result_return_40%_rsk0.1_e0.1')
result.cumsum().to_csv(csv_file_path)
print(f"数据已保存为 {csv_file_path}")
result.cumsum().plot()
result.index = pd.to_datetime(result.index)
grouped = result.groupby(result.index.year)
zz500_pct.index = pd.to_datetime(zz500_pct.index)
zz500_pct = zz500_pct[:returns.index[-1]]
zz500_grouped = zz500_pct.groupby(zz500_pct.index.year)
#超额收益率
excess_returns = {}
for year, returns in grouped:
    excess_return = returns.sub(zz500_grouped.get_group(year).cumsum(), axis = 0)[returns.index[-1.strftime('%Y-%m-%d')]
    excess_returns[str(year)] = excess_return
excess_returns

#画图
import matplotlib.pyplot as plt
df1 = pd.read_csv(os.path.join(root, 'optimal_weights_df_20%_rsk1_e0.05'), index_col=0)
df1 = calc_return.run_backtest(df1, price,  tax = 0.001, fee = 0.00012, slip = 0.0005, holding_period=5)[0]
df1.index = pd.to_datetime(df1.index)
grouped = df1.groupby(df1.index.year)
bench_r = zz500_pct.groupby(zz500_pct.index.year)
max_drawdown1 = {}
for year, returns in grouped:
    excess_r = (1 + (returns - bench_r.get_group(year))).cumprod()
    peak[peak <= 0] = 1e-9
    peak = excess_r.cummax()
    drawdown = (peak - excess_r) / peak
    max_drawdown1[year] = drawdown.max()
max_drawdown1








result = pd.read_csv(os.path.join(root, 'result_return_20%_rsk0.5_e0.05'), index_col=0)
#result.index = result.index.strftime('%Y-%m-%d')
strategy_return = result
benchmark_return = zz500_price.pct_change()[:-3]['open']
strategy_return.index = pd.to_datetime(strategy_return.index)
benchmark_return.index = pd.to_datetime(benchmark_return.index)
# Calculate cumulative returns
strategy_cum_return = strategy_return#.cumsum()#(1 + strategy_return).cumprod() - 1
benchmark_cum_return = benchmark_return.cumsum()#(1 + benchmark_return).cumprod() - 1

# Calculate excess return
excess_return = strategy_cum_return.sub(benchmark_cum_return[strategy_cum_return.index[0]: strategy_cum_return.index[-1]], axis=0)
excess_return.index = excess_return.index.strftime('%Y-%m-%d')
# Calculate drawdown
def calculate_drawdown(return_series):
    wealth_index = (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return drawdowns

strategy_drawdown = calculate_drawdown(strategy_return)

# Find the max drawdown point and its duration
max_drawdown = strategy_drawdown.min()
max_drawdown_point = strategy_drawdown.idxmin()
drawdown_start = excess_return[:max_drawdown_point].idxmax()

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(strategy_cum_return, label='Portfolio')
plt.plot(benchmark_cum_return, label='Benchmark(ZZ500)')
plt.plot(excess_return, label='Excess')
plt.scatter(max_drawdown_point, strategy_cum_return.loc[max_drawdown_point], color='green', marker='v', s=100, label='Biggest Drawdown')
plt.scatter(drawdown_start, strategy_cum_return.loc[drawdown_start], color='blue', marker='D', s=100, label='Longest Drawdown Period')

# Annotations
plt.annotate('Biggest Drawdown', xy=(max_drawdown_point, strategy_cum_return.loc[max_drawdown_point]), 
             xytext=(max_drawdown_point, strategy_cum_return.loc[max_drawdown_point]+0.1),
             arrowprops=dict(facecolor='yellow', shrink=0.05))
plt.annotate('Longest Drawdown Period', xy=(drawdown_start, strategy_cum_return.loc[drawdown_start]), 
             xytext=(drawdown_start, strategy_cum_return.loc[drawdown_start]-0.1),
             arrowprops=dict(facecolor='orange', shrink=0.05))
plt.legend()
plt.xlabel('Date')
plt.ylabel('Return')
plt.title('Portfolio vs Benchmark vs Excess')
plt.show()









optimal_weights
            
            
            
optimal_weights = pd.read_csv(os.path.join(root, 'optimal_weights_df_20%_rsk0.5_e0.05'), index_col=0)
optimal_weights = calc_return.run_backtest(optimal_weights, price,  tax = 0.001, fee = 0.00012, slip = 0.0005, holding_period=5)[0]#.cumsum()
optimal_weights.index = pd.to_datetime(optimal_weights.index)
grouped = optimal_weights.groupby(optimal_weights.index.year)
#夏普
sharpe_ratios = {}
for year, returns in grouped:
    #excess_returns = (1+returns).cumprod()[-1] - (1+bench_r.get_group(year)).cumprod()[-1]
    excess_returns = returns.cumsum()[-1] - bench_r.get_group(year).cumsum()[-1]
    #print("excess_returns \n", excess_returns)
    #average_excess_return = excess_returns.mean()
    standard_deviation = (returns.cumsum() - bench_r.get_group(year).cumsum()).std()
    sharpe_ratio = excess_returns / standard_deviation
    sharpe_ratios[year] = sharpe_ratio
sharpe_ratios
#超额回撤
max_drawdown = {}
for year, returns in grouped:
    excess_r = (1 + (returns - bench_r.get_group(year))).cumprod()
    peak = excess_r.cummax()
    drawdown = (peak - excess_r) / peak
    max_drawdown[year] = drawdown.max()
max_drawdown

#卡玛比
calmar_ratios = {}
for year, returns in grouped:
    return_year = returns.cumsum()[-1]
    down_year = max_drawdown[year]
    calmar_ratios[year] = return_year / down_year
calmar_ratios


#对比
sharp_excess = {}
for year in sharpe_ratios_nonconstraint:
    sharp_excess[year] = "{:.4f}%".format((sharpe_ratios.get(year) - sharpe_ratios_nonconstraint.get(year)) * 100)
sharp_excess

down_excess = {}
for year in max_drawdown:
    down_excess[year] = "{:.4f}%".format((max_drawdown_nonocnstraint.get(year) - max_drawdown.get(year)) * 100)
down_excess


"""

