# %% [markdown]
# # ETF轮动之拥挤度策略
# 
# 本方案分为五大模块：**成交额**、**换手率**、**波动率**、**动量**和**量价相关性**。每个模块包含不同的指标类型。
# 
# **成交额 & 成交额市值比**模块包括以下指标：滚动平均、滚动分位数、回归标准化和回归分位数。**换手率**模块的指标有滚动平均、滚动分位数、滚动加权平均和滚动乖离度。**波动率**模块则包括收盘价波动率和跳空价差。**动量**模块的指标为超额平均、超额乖离和超额波动率。**量价相关性**模块中，我们使用量价相关系数和离散相关系数（-1, 0, 1）。
# 
# 每个滚动指标的窗口期设置为5天、10天、15天、20天、25天、30天、60天和90天。每个细分指标（按类型和窗口）应用后望20天最大回撤的相关性进行检验，越负越有效。
# 
# 根据前文的指标构建与测试，选出以下关键指标：`amount_rollq_60`、`proportion_rollq_15`、`turnover_rollq_90`、`turnover_deviation_30`、`volatility_20`和`excess_volatility_30`。
# 
# 因子构建步骤如下：首先对所有指标进行归一化处理，然后等权组合这些指标，最后设置调仓信号，选择因子分位数大于threshold=0.9的日期进行操作。
# 

# %% [markdown]
# ## 环境配置

# %%
import numpy as np
import pandas as pd
import datetime
import pandas_market_calendars as mcal
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import vectorbt as vbt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# %%
from WindPy import w
w.start()

# %%
from jqdatasdk import *
account = '15706838901'
password = 'Gtjas025660'
auth(account,password)

# %%
import rqdatac as rq
rq.init(username='17623895046',password='17623895046')

# %% [markdown]
# ## 候选etf池和设置参数

# %% [markdown]
# 起止日期和窗口频率

# %%
start_date = "2021-06-30"
end_date = "2024-06-30"
freq = 30

# %% [markdown]
# **候选etf池** ： 来自GetMyETFPool.ipynb

# %%
etf_candidates = pd.read_csv('etf_data/etf_candidates20240807.csv')
ticker = etf_candidates['ticker']
etf_candidates

# %% [markdown]
# ticker ——> windticker ： 调用wind_suffix函数

# %%
def wind_suffix(sec_code):
    exchange_suffix = ''
    if type(sec_code) != str:
        return None

    if len(sec_code) == 6:

        first_two = sec_code[:2]
        if first_two.isdigit():
            # 权益类证券
            first = sec_code[0]
            if first == '6':  # 上交所、股票
                sec_type = '股票'
                exchange_suffix = '.SH'
            elif first == '5':  # 上交所、ETF
                sec_type = 'ETF基金'
                exchange_suffix = '.SH'
            elif first in ['3', '0']:  # 深交所、股票
                sec_type = '股票'
                exchange_suffix = '.SZ'
            elif first == '1':  # 深交所、ETF
                sec_type = 'ETF基金'
                exchange_suffix = '.SZ'
            elif first in ['8','4']:
                sec_type = '股票'
                exchange_suffix = 'BJ'
        else:
            # 股指期货
            sec_type = '指数期货'
            exchange_suffix = '.CFE'

    elif len(sec_code) == 8:
        sec_type = '期权'
        first = sec_code[0]
        if first == '1':
            exchange_suffix = '.SH'
        elif first == '9':
            exchange_suffix = '.SZ'
        else:
            exchange_suffix = '.CFE'

    else:
        raise ValueError('错误的匹配方式')
    if exchange_suffix:
        return sec_code+exchange_suffix
    else:
        return None
    
windticker = [wind_suffix(t[:6]) for t in ticker]
windticker

# %% [markdown]
# 获取全市场etf列表：数据来源为"ETF_info.pkl"

# %%
all_etf_info = pd.read_pickle('ETF_info.pkl')
all_etfs = all_etf_info['基金代码']
all_etfs

# %% [markdown]
# 获取全市场etf的量价数据：数据来源为"bar_data.pkl"

# %%
market_df = pd.read_pickle('bar_data.pkl')
market_df

# %% [markdown]
# **etf池的收盘价** 
# \
# 用于构建一系列量价指标，和指标测试时计算最大回撤

# %%
def get_etf_close(ticker, start_date, end_date, market_df):
    """
    从 market_df 中提取 ETF 的收盘价数据。

    参数:
    ticker: list, ETF 的 ticker 列表。
    start_date: str, 起始日期，格式为 'YYYY-MM-DD'。
    end_date: str, 结束日期，格式为 'YYYY-MM-DD'。
    market_df: DataFrame, 包含市场数据的 DataFrame。

    返回:
    DataFrame, ETF 的收盘价数据，索引为日期，列为不同的 ETF。
    """
    # 筛选出指定日期范围内的数据
    mask = (market_df['date'] >= start_date) & (market_df['date'] <= end_date)
    filtered_df = market_df.loc[mask]

    # 创建一个空的 DataFrame 用于存储结果
    etf_close_df = pd.DataFrame()

    # 遍历每个 ticker 获取收盘价数据
    for t in ticker:
        temp_df = filtered_df[filtered_df['order_book_id'] == t][['date', 'close']]
        temp_df.set_index('date', inplace=True)
        temp_df.columns = [t]
        if etf_close_df.empty:
            etf_close_df = temp_df
        else:
            etf_close_df = pd.concat([etf_close_df, temp_df], axis=1)
    
    etf_close_df.columns.name = 'ticker'

    return etf_close_df

etf_close = get_etf_close(ticker, start_date, end_date, market_df)
etf_close


# %% [markdown]
# **etf池开盘价**

# %%
def get_etf_open(ticker, start_date, end_date, market_df):
    """
    从 market_df 中提取 ETF 的开盘价数据。

    参数:
    ticker: list, ETF 的 ticker 列表。
    start_date: str, 起始日期，格式为 'YYYY-MM-DD'。
    end_date: str, 结束日期，格式为 'YYYY-MM-DD'。
    market_df: DataFrame, 包含市场数据的 DataFrame。

    返回:
    DataFrame, ETF 的开盘价数据，索引为日期，列为不同的 ETF。
    """
    # 筛选出指定日期范围内的数据
    mask = (market_df['date'] >= start_date) & (market_df['date'] <= end_date)
    filtered_df = market_df.loc[mask]

    # 创建一个空的 DataFrame 用于存储结果
    etf_open_df = pd.DataFrame()

    # 遍历每个 ticker 获取开盘价数据
    for t in ticker:
        temp_df = filtered_df[filtered_df['order_book_id'] == t][['date', 'open']]
        temp_df.set_index('date', inplace=True)
        temp_df.columns = [t]
        if etf_open_df.empty:
            etf_open_df = temp_df
        else:
            etf_open_df = pd.concat([etf_open_df, temp_df], axis=1)
    
    etf_open_df.columns.name = 'ticker'

    return etf_open_df


etf_open = get_etf_open(ticker, start_date, end_date, market_df)
etf_open

# %% [markdown]
# 从米筐获取全市场etf量价数据并降维

# %%
# all_etf_info = rq.all_instruments(type='ETF',market='cn',date = end_date)
# all_etf_info = all_etf_info[all_etf_info['status']=='Active']

# all_etf_price = rq.get_price(all_etf_info.order_book_id.to_list(),
#                          start_date=start_date,end_date=end_date,
#                         fields=['open','high','low','close','total_turnover','volume'])
# all_etf_price = all_etf_price.reset_index()
# all_etf_price.rename(columns={'order_book_id': 'ticker'}, inplace=True)
# all_etf_price.set_index(['date', 'ticker'], inplace=True)
# all_etf_price

# %%
# all_close = all_etf_price['close'].unstack(level=-1)
# all_high = all_etf_price['high'].unstack(level=-1)
# all_open = all_etf_price['open'].unstack(level=-1)
# all_low = all_etf_price['low'].unstack(level=-1)

# %% [markdown]
# ## 最大回撤相关度检验 
# 计算后望20天的最大回撤，相关性检测各一维指标

# %%
def calculate_max_drawdown(prices, window=20):
    # 创建一个空的 DataFrame 来存储结果
    max_drawdown = pd.DataFrame(index=prices.index, columns=prices.columns)
    
    # 逐列计算最大回撤
    for column in prices.columns:
        for i in range(len(prices) - window + 1):
            window_data = prices[column].iloc[i:i + window]
            roll_max = window_data.max()
            daily_drawdown = window_data / roll_max - 1.0
            max_drawdown[column].iloc[i] = daily_drawdown.min()
    
    return max_drawdown.dropna(how='all')

max_drawdown = calculate_max_drawdown(etf_close)
max_drawdown

# %%
def correlation_exam(base_df=max_drawdown, exam_df=etf_close, exam_df_name=''):
    """
    计算两个 DataFrame 之间的相关系数，按列计算。
    
    参数:
    base_df: DataFrame, 最大回撤数据，索引为日期，列为不同的 ETF。
    exam_df: DataFrame, 其他需要比较的数据，索引为日期，列为不同的 ETF。
    
    返回:
    DataFrame, 每列的相关系数。
    """
    
    # 确保两个 DataFrame 的日期格式一致
    base_df.index = pd.to_datetime(base_df.index)
    exam_df.index = pd.to_datetime(exam_df.index)

    # 提取两个 DataFrame 的共同日期
    common_dates = base_df.index.intersection(exam_df.index)

    # 创建两个新的 DataFrame，包含共同日期的数据
    max_drawdown_common = base_df.loc[common_dates].copy()
    other_common = exam_df.loc[common_dates].copy()

    # 初始化结果 DataFrame
    correlation_df = pd.DataFrame(index=max_drawdown_common.columns, columns=["correlation"])

    # 计算每列的相关系数
    for column in max_drawdown_common.columns:
        correlation_df.loc[column, "correlation"] = max_drawdown_common[column].corr(other_common[column])

    # 确保返回的相关系数是数值类型并且没有 NaN 值
    correlation_df = correlation_df.astype(float).dropna()

    # 绘制热力图
    plt.figure(figsize=(30, 3))
    sns.heatmap(correlation_df.T, cmap='coolwarm', annot=True, center=0, cbar=True)
    plt.title(f'Correlation Heatmap {exam_df_name}')
    plt.show()

    return None


closeprice_exam = correlation_exam() #示例测试


# %% [markdown]
# ## 指标一：成交额及其占比
# 
# 成交额时序分位数，成交额占比时序分位数
# \
# 获取成交额 ——> 计算日度市场占比 ——> 回望窗口期的分位数

# %%

def calculate_etf_amount_proportion(ticker, start_date, end_date, market_df):
    # 将start_date和end_date转换为日期格式
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # 筛选出需要的日期范围的数据
    market_df['date'] = pd.to_datetime(market_df['date'])
    market_df = market_df[(market_df['date'] >= start_date) & (market_df['date'] <= end_date)]
    
    # 删除重复的date和order_book_id组合
    market_df = market_df.drop_duplicates(subset=['date', 'order_book_id'])
    
    # 提取所有ETF的volume数据
    all_etf_volume = market_df.pivot(index='date', columns='order_book_id', values='total_turnover')

    # 重命名列为ticker
    all_etf_volume.columns.name = 'ticker'

    # 计算每日总成交额
    market_total_volume = all_etf_volume.sum(axis=1)

    # 计算每个ETF每日成交额占市场成交额的比例
    etf_amount_proportion = all_etf_volume.div(market_total_volume, axis=0)

    # 提取用户选定的ETF对应的成交额比例数据
    proportion_df = etf_amount_proportion[ticker]
    
    # 提取用户选定的ETF对应的成交额数据
    amount_df = all_etf_volume[ticker]

    return proportion_df, amount_df


etf_proportion, etf_amount = calculate_etf_amount_proportion(ticker, start_date, end_date, market_df)

etf_proportion.to_csv('etf_data/etf_proportion.csv', index=True)
etf_amount.to_csv('etf_data/etf_amount.csv', index=True)


# %%
etf_proportion

# %%
etf_amount

# %% [markdown]
# ##### 成交额市场占比的滚动平均

# %%
def calculate_rolling_average(proportion_df, window=30):
    """
    计算换手率净值滚动平均值。

    参数:
    proportion_df: DataFrame, 比例数据，索引为日期，列为不同的 ETF。
    window: int, 回望窗口大小。

    返回:
    DataFrame, 换手率净值滚动平均值。
    """
    # 计算滚动平均值
    rolling_average = proportion_df.rolling(window=window).mean()

    return rolling_average.dropna()

# 计算滚动平均值
proportion_rollavg = calculate_rolling_average(etf_proportion, window=freq)
proportion_rollavg

# %%
proportion_rollavg_exam = correlation_exam(exam_df=proportion_rollavg) #按照初始的freq

# %% [markdown]
# 5/10/15/20/25/30/60/90日均成交市场比

# %%
windows = [5, 10, 15, 20, 25, 30, 50, 60, 90, 120, 150, 180, 240]  # 多个窗口大小
rolling_avgs = {}
for window in windows:
    rolling_avgs[window] = calculate_rolling_average(etf_proportion, window=window)
    correlation_exam(exam_df=rolling_avgs[window], exam_df_name=f'proportion_rollavg_{window}')
    vars()[f'proportion_rollavg_{window}'] = rolling_avgs[window]

# %% [markdown]
# 可见，回望的窗口期越长，成交市场比和最大回撤的相关性表现越不显著，指标越无效。

# %% [markdown]
# ##### 成交额的滚动分位数

# %%
def calculate_etf_rolling_quantile(df, window=freq):
    # 创建一个新的DataFrame用于存储结果
    rollq = pd.DataFrame(index=df.index, columns=df.columns)
    
    # 计算滚动分位数
    for ticker in tqdm(df.columns):
        rollq[ticker] = df[ticker].rolling(window=window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # 删除前 window - 1 天没有数据的行
    rollq = rollq.iloc[window-1:]
    
    return rollq

# %%
amount_rollq = calculate_etf_rolling_quantile(etf_amount, window=freq)
amount_rollq

# %%
amount_rollq_exam = correlation_exam(exam_df=amount_rollq)

# %%
windows = [5, 10, 15, 20, 25, 30, 60, 90, 120, 150, 180]
amount_rollq_windows = {}
for window in windows:
    amount_rollq_windows[window] = calculate_etf_rolling_quantile(etf_amount, window=window)
    correlation_exam(exam_df=amount_rollq_windows[window], exam_df_name=f'amount_rollq_{window}')
    vars()[f'amount_rollq_{window}'] = amount_rollq_windows[window]

# %% [markdown]
# 60日成交额分位数相对较优

# %% [markdown]
# ##### 成交额市场占比的滚动分位数

# %%
proportion_rollq = calculate_etf_rolling_quantile(etf_proportion, freq)
proportion_rollq

# %%
proportion_rollq_exam = correlation_exam(exam_df=proportion_rollq)

# %%
windows = [5, 10, 15, 20, 25, 30, 50, 60, 90, 120, 150, 180, 240]  # 多个窗口大小
proportion_rolling_quants = {}
for window in windows:
    proportion_rolling_quants[window] = calculate_etf_rolling_quantile(etf_proportion, window=window)
    correlation_exam(exam_df=proportion_rolling_quants[window], exam_df_name=f'proportion_rollq_{window}')
    vars()[f'proportion_rollq_{window}'] = proportion_rolling_quants[window]

# %% [markdown]
# 可见，回望时间在15-25期间，成交额市场比分位数这个指标比较有效，在两端趋于无效。

# %% [markdown]
# ##### 成交额市场占比回归成交额

# %%
def calculate_volume_proportion_regression(volume, proportion):
    # 创建一个新的DataFrame用于存储回归拟合值
    volume_proportion_reg = pd.DataFrame(index=volume.index, columns=volume.columns)
    
    # 对每个ticker进行回归分析
    for ticker in volume.columns:
        # 去除 NaN 数据
        valid_data = volume[ticker].notna() & proportion[ticker].notna()
        X = volume[ticker][valid_data].values.reshape(-1, 1)
        Y = proportion[ticker][valid_data].values.reshape(-1, 1)

        if len(X) == 0 or len(Y) == 0:
            continue
        
        # 进行线性回归
        model = LinearRegression().fit(X, Y)
        
        # 生成拟合值
        # 填充 NaN 值
        filled_volume = volume[ticker].fillna(method='ffill').fillna(method='bfill')
        volume_proportion_reg[ticker] = model.predict(filled_volume.values.reshape(-1, 1))
    
    return volume_proportion_reg

# %% [markdown]
# **先回归，再取分位数**
# \
# 颗粒度更细，指标更清晰

# %%
proportion_reg= calculate_volume_proportion_regression(etf_amount, etf_proportion)
proportion_regq = calculate_etf_rolling_quantile(proportion_reg, freq)
proportion_regq

# %%
proportion_regq_exam = correlation_exam(exam_df=proportion_regq)

# %% [markdown]
# **用分位数回归**
# \
# 颗粒度更大，阻断断点和突变的影响

# %%
proportion_qreg = calculate_volume_proportion_regression(amount_rollq, proportion_rollq)
proportion_qreg

# %%
proportion_qreg_exam = correlation_exam(exam_df=proportion_qreg)

# %% [markdown]
# ## 指标二：换手率
# 换用Wind API中的turn，表示成交量/流通份额

# %%
def get_etf_turnover(ticker, start_date, end_date):

    # 转换ticker为windticker
    windticker = [wind_suffix(t[:6]) for t in ticker]
    
    # 创建一个空的DataFrame用于存储结果
    etf_turnover_df = pd.DataFrame()

    # 遍历每个ticker获取数据
    for t, wt in tqdm(zip(ticker, windticker)):
        # 提取换手率数据
        data = w.wsd(wt, 'turn', start_date, end_date, "")
        
        if data.ErrorCode != 0:
            print(f"Error fetching data for {t}: {data.ErrorCode}")
            continue
        
        # 创建临时DataFrame
        temp_df = pd.DataFrame(data.Data[0], index=data.Times, columns=[t])
        
        # 合并到结果DataFrame中
        if etf_turnover_df.empty:
            etf_turnover_df = temp_df
        else:
            etf_turnover_df = pd.concat([etf_turnover_df, temp_df], axis=1)
    
    # 重命名行列
    etf_turnover_df.columns.name = 'ticker'
    etf_turnover_df.index.name = 'date'
    
    return etf_turnover_df

# 提取ETF的换手率数据
etf_turnover = get_etf_turnover(ticker, start_date, end_date)

# 将结果保存为CSV文件
etf_turnover.to_csv('etf_data/etf_turnover.csv', index=True)

etf_turnover


# %%
# 计算相关系数
correlation_matrix = etf_turnover.corrwith(etf_amount, axis=0)

# 转换为DataFrame
turnover_amount_corr = pd.DataFrame(correlation_matrix, columns=['correlation'])

# 绘制热力图
plt.figure(figsize=(30, 1.5))
sns.heatmap(turnover_amount_corr.T, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation between ETF Turnover and Amount')
plt.show()

# %% [markdown]
# 由上图可见，换手率指标和成交量指标高度相关，少数etf存在较大偏离，原因是流通份额在统计期间发生大变动。

# %% [markdown]
# ##### 换手率滚动平均
# **对换手率的处理**的操作一，按照一定的日频回望取平均

# %%
def calculate_rolling_mean(etf_turnover_df, window=freq):
    # 计算滚动均值
    rolling_mean_df = etf_turnover_df.rolling(window=window).mean()
    
    # 删除含有NaN值的行
    rolling_mean_df.dropna(inplace=True)
    
    return rolling_mean_df

turnover_rollavg = calculate_rolling_mean(etf_turnover, window=freq)

# # 将结果保存为CSV文件
# turnover_rollmean.to_csv('etf_rolling_mean.csv', index=True)

turnover_rollavg

# %%
turnover_rollavg_exam = correlation_exam(exam_df=turnover_rollavg)

# %% [markdown]
# ##### 换手率滚动分位数
# 操作二，按同样的频率回望取分位数

# %%
turnover_rollq = calculate_etf_rolling_quantile(etf_turnover,window = freq)
turnover_rollq

# %%
turnover_rollq_exam = correlation_exam(exam_df=turnover_rollq)

# %% [markdown]
# 换手率的分位数与最大回撤显著相关，指标非常有效。改变窗口期进行下一步测试。

# %%
windows = [5, 10, 15, 20, 25, 30, 50, 60, 90, 120, 150, 180, 240]  # 多个窗口大小
turnover_rolling_quants = {}
for window in windows:
    turnover_rolling_quants[window] = calculate_etf_rolling_quantile(etf_turnover, window=window)
    correlation_exam(exam_df=turnover_rolling_quants[window], exam_df_name=f'turnover_rollq_{window}')
    vars()[f'turnover_rollq_{window}'] = turnover_rolling_quants[window]

# %% [markdown]
# 总体而言， 对于换手率分位数，窗口期越长指标越显著

# %% [markdown]
# 
# ##### 换手率滚动加权平均
# 操作三，根据距离当前日期的远近设置权重，离当前日期越远权重越小，并标准化

# %%
def calculate_weighted_average(df, freq=freq):
    # 创建权重数组，距离当日越近权重越大
    weights = np.arange(1, freq + 1)

    def weighted_average(x):
        return np.dot(x, weights) / weights.sum()

    # 复制数据用于操作
    df_copy = df.copy()

    # 对每个列进行滚动窗口的加权平均
    turnover_wa = df_copy.rolling(window=freq).apply(weighted_average, raw=True)

    # 去除空值
    turnover_wa.dropna(inplace=True)

    return turnover_wa

turnover_wa = calculate_weighted_average(etf_turnover, freq)
turnover_wa

# %%
def standardize_and_weighted_average(df, freq=freq):
    def standardize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def weighted_average(x, weights):
        return np.dot(x, weights) / weights.sum()
    
    # 创建权重数组，距离当日越近权重越大
    weights = np.arange(1, freq + 1)

    # 复制数据用于操作
    df_copy = df.copy()

    # 创建一个空的 DataFrame 来存储结果
    result_df = pd.DataFrame(index=df_copy.index)

    # 对每个列进行滚动窗口的标准化和加权平均
    for column in df_copy.columns:
        standardized_col_list = []
        for i in range(len(df_copy[column])):
            if i < freq - 1:
                standardized_col_list.append(np.nan)
            else:
                window_data = df_copy[column].iloc[i-freq+1:i+1]
                standardized_window = standardize(window_data)
                weighted_avg = weighted_average(standardized_window, weights)
                standardized_col_list.append(weighted_avg)
        
        result_df[column] = standardized_col_list

    # 去除空值
    result_df.dropna(inplace=True)

    return result_df

turnover_wastd = standardize_and_weighted_average(etf_turnover, freq)
turnover_wastd


# %%
turnover_wastd_exam = correlation_exam(exam_df=turnover_wastd)

# %% [markdown]
# 换手率的回望加权平均值与最大回撤无显著关联

# %% [markdown]
# ##### 换手率滚动乖离度

# %%
def calculate_turnover_deviation(turnover_df, window):
    """
    计算换手率净值乖离度。

    参数:
    turnover_df: DataFrame, 换手率数据，索引为日期，列为不同的 ETF。
    window: int, 回望窗口大小。

    返回:
    DataFrame, 换手率净值乖离度。
    """
    # 计算历史平均换手率
    historical_mean = turnover_df.rolling(window=window).mean()

    # 计算当前换手率与历史平均换手率的差异
    deviation = turnover_df - historical_mean

    # 计算乖离度的标准差
    deviation_std = deviation.rolling(window=window).std()

    # 计算乖离度（标准化差异）
    deviation_score = deviation / deviation_std

    return deviation_score.dropna()

# 计算换手率乖离度
turnover_deviation = calculate_turnover_deviation(etf_turnover, window=freq)
turnover_deviation


# %%
turnover_deviation_exam = correlation_exam(exam_df=turnover_deviation)

# %% [markdown]
# 换手率乖离度指标有效性较强，可以用更改窗口期对其进行二次检验

# %%
windows = [5, 10, 15, 20, 25, 30, 60, 90, 120, 150, 180]  # 多个窗口大小
turnover_deviations = {}
for window in windows:
    turnover_deviations[window] = calculate_turnover_deviation(etf_turnover, window=window)
    correlation_exam(exam_df=turnover_deviations[window], exam_df_name=f'turnover_deviation_{window}')
    vars()[f'turnover_deviation_{window}'] = turnover_deviations[window]

# %% [markdown]
# 选取30日换手率乖离度，偏离最小

# %% [markdown]
# ## 指标三：波动率

# %% [markdown]
# ##### 收盘价滚动波动率

# %%
def calculate_volatility(df, window=freq, year=252):
    """
    计算回望周期内的波动率（volatility）。
    
    参数：
    df - 包含收盘价数据的 DataFrame，索引为日期，列为ticker
    window - 回望周期长度
    
    返回：
    包含波动率数据的 DataFrame
    """
    volatility_df = pd.DataFrame(index=df.index)

    # 计算每个列的波动率
    for column in df.columns:
        # 计算对数收益率
        log_returns = np.log(df[column] / df[column].shift(1))
        # 计算滚动窗口内的波动率
        rolling_volatility = log_returns.rolling(window=window).std() * np.sqrt(window)
        # 波动率年化
        annualized_volatility = rolling_volatility * np.sqrt(year/ window) 
        volatility_df[column] = annualized_volatility

    # 去除空值
    volatility_df.dropna(inplace=True)
    
    return volatility_df

volatility = calculate_volatility(etf_close, window=freq)
volatility


# %%
volatility_exam = correlation_exam(exam_df=volatility)

# %% [markdown]
# 更改窗口期，测试波动率指标

# %%
windows = [5, 10, 15, 20, 25, 30, 50, 60, 90, 120, 150, 180, 240]  
volatility_windows = {}
for window in windows:
    volatility_windows[window] = calculate_volatility(etf_close, window=window)
    correlation_exam(exam_df=volatility_windows[window], exam_df_name=f'volatility_{window}')
    vars()[f'volatility_{window}'] = volatility_windows[window]

# %% [markdown]
# 20日波动率最为有效

# %% [markdown]
# ##### 跳空价差

# %%
def calculate_gap(etf_open, etf_close):
    # 对齐两个DataFrame的日期
    etf_open_aligned = etf_open.align(etf_close, join='inner', axis=0)[0]
    etf_close_aligned = etf_close.align(etf_open, join='inner', axis=0)[0]
    
    # 计算前一天的收盘价
    prev_close = etf_close_aligned.shift(1)
    
    # 计算跳空价差
    gap = etf_open_aligned - prev_close
    
    # 舍去空值
    gap.dropna(inplace=True)
    
    return gap

# 计算跳空价差
closeopen_gap = calculate_gap(etf_open, etf_close)
closeopen_gap


# %%
closeopen_gap_exam = correlation_exam(exam_df=closeopen_gap)

# %% [markdown]
# **根据跳空价差回望freq的时间序列，计算超过n*sigma的断点数量占比**

# %%
def calculate_jumpiness(gap_df, freq, sigma):
    # 创建一个空的DataFrame来存储结果
    jumpiness_df = pd.DataFrame(index=gap_df.index, columns=gap_df.columns)
    
    # 定义一个函数来计算跳跃性指标
    def calculate_jumpiness_for_window(window):
        mean = np.mean(window)
        std = np.std(window)
        threshold_upper = mean + sigma * std
        threshold_lower = mean - sigma * std
        jumps = np.sum((window > threshold_upper) | (window < threshold_lower))
        return jumps / freq
    
    # 对每个列进行滚动窗口计算跳跃性指标
    for column in gap_df.columns:
        jumpiness_col = []
        for i in range(len(gap_df[column])):
            if i < freq - 1:
                jumpiness_col.append(np.nan)
            else:
                window = gap_df[column].iloc[i-freq+1:i+1]
                jumpiness_col.append(calculate_jumpiness_for_window(window))
        
        jumpiness_df[column] = jumpiness_col
    
    # 去除空值
    jumpiness_df.dropna(inplace=True)
    
    return jumpiness_df


jumpiness = calculate_jumpiness(closeopen_gap, freq=freq, sigma=2)
jumpiness


# %%
jumpiness_exam = correlation_exam(exam_df=jumpiness)

# %% [markdown]
# **根据跳空价差回望freq的时间序列，计算当日价差距离均值（0）小于n*sigma（n=1）的概率**

# %%
from scipy.stats import norm

def calculate_prob_within_sigma(gap_df, freq, sigma=1):
    # 创建一个空的 DataFrame 来存储结果
    prob_df = pd.DataFrame(index=gap_df.index, columns=gap_df.columns)
    
    # 对每个列进行滚动窗口计算价差在 n_sigma 内的概率
    for column in gap_df.columns:
        prob_col = []
        for i in range(len(gap_df[column])):
            if i < freq - 1:
                prob_col.append(np.nan)
            else:
                window = gap_df[column].iloc[i-freq+1:i+1]
                mean = np.mean(window)
                std = np.std(window)
                if std == 0:  # 避免标准差为 0 的情况
                    prob = 1.0 if mean == 0 else 0.0
                else:
                    prob = norm.cdf(sigma * std, loc=mean, scale=std) - norm.cdf(-sigma * std, loc=mean, scale=std)
                prob_col.append(prob)
        
        prob_df[column] = prob_col
    
    # 去除空值
    prob_df.dropna(inplace=True)
    
    return prob_df


stability = calculate_prob_within_sigma(closeopen_gap, freq=freq, sigma=1)
stability


# %%
stability_exam = correlation_exam(exam_df=stability)

# %% [markdown]
# ## 指标四：动量
# 

# %% [markdown]
# 计算相对基准指数的超额收益

# %%

def get_etf_excess_returns(etf_close, benchmark_index, start_date, end_date):
    """
    计算 ETF 相对于基准指数的超额收益。

    参数:
    etf_close: DataFrame, ETF 的收盘价数据，索引为日期，列为不同的 ETF。
    benchmark_index: str, 用于计算超额收益的基准指数。
    start_date: str, 起始日期，格式为 'YYYY-MM-DD'。
    end_date: str, 结束日期，格式为 'YYYY-MM-DD'。

    返回:
    DataFrame, ETF 相对于基准指数的超额收益。
    """
    
    # 获取基准指数的日度收盘价数据
    data = w.wsd(benchmark_index, "close", start_date, end_date, "")
    if data.ErrorCode != 0:
        print(f"Error fetching data for {benchmark_index}: {data.ErrorCode}")
        return pd.DataFrame()
    
    benchmark_close = pd.DataFrame(data.Data[0], index=data.Times, columns=[benchmark_index])

    # 计算收益率
    etf_returns = etf_close.pct_change().dropna()
    benchmark_returns = benchmark_close.pct_change().dropna()
    
    # 调整基准指数 DataFrame 的格式以匹配 ETF
    benchmark_returns = benchmark_returns.reindex(etf_returns.index).ffill()

    # 计算超额收益
    etf_excess_return = etf_returns.subtract(benchmark_returns[benchmark_index], axis=0)
    
    # 调整超额收益 DataFrame 的格式
    etf_excess_return.index.name = 'date'
    etf_excess_return.columns.name = 'ticker'
    etf_excess_return.dropna()

    return etf_excess_return



benchmark = "000300.SH"  # 选择沪深300作为基准
etf_excess_return = get_etf_excess_returns(etf_close, benchmark, start_date, end_date)
etf_excess_return.to_csv('etf_data/etf_excess_return.csv', index=True)

etf_excess_return


# %% [markdown]
# ##### 动量：回望期间的平均超额

# %%
momentum = etf_excess_return.rolling(window=freq).sum().dropna()
momentum

# %%
momentum_exam = correlation_exam(exam_df=momentum)

# %% [markdown]
# ##### 超额收益乖离度

# %%
def calculate_excess_return_deviation(excess_return_df, window):
    """
    计算超额收益净值乖离度。

    参数:
    excess_return_df: DataFrame, 超额收益数据，索引为日期，列为不同的 ETF。
    window: int, 回望窗口大小。

    返回:
    DataFrame, 超额收益净值乖离度。
    """
    # 计算历史平均超额收益
    historical_mean = excess_return_df.rolling(window=window).mean()

    # 计算当前超额收益与历史平均超额收益的差异
    deviation = excess_return_df - historical_mean

    # 计算乖离度的标准差
    deviation_std = deviation.rolling(window=window).std()

    # 计算乖离度（标准化差异）
    deviation_score = deviation / deviation_std

    return deviation_score.dropna()

excess_deviation = calculate_excess_return_deviation(etf_excess_return, window = freq)
excess_deviation


# %%
excess_deviation_exam = correlation_exam(exam_df=excess_deviation)

# %% [markdown]
# ##### 超额收益波动率

# %%
excess_volatility = etf_excess_return.rolling(window=freq).std().dropna() * np.sqrt(252 / freq)
excess_volatility

# %%
excess_volatility_exam = correlation_exam(exam_df=excess_volatility)

# %%
windows = [5, 10, 15, 20, 25, 30, 60, 90, 120, 150, 180]  
excess_volatility_windows = {}
for window in windows:
    excess_volatility_windows[window] = etf_excess_return.rolling(window=window).std().dropna() * np.sqrt(252 / window)
    correlation_exam(exam_df=excess_volatility_windows[window], exam_df_name=f'excess_volatility_{window}')
    vars()[f'excess_volatility_{window}'] = excess_volatility_windows[window]


# %% [markdown]
# 超额波动率选择30日比较好

# %% [markdown]
# ## 指标五：量价相关性
# 监控动量和反转

# %%
def calculate_volume_price_correlation(etf_amount, etf_close, freq=freq):
    """
    计算量价相关性。

    参数:
    etf_amount: DataFrame, ETF 的成交量数据，索引为日期，列为不同的 ETF。
    etf_close: DataFrame, ETF 的收盘价数据，索引为日期，列为不同的 ETF。
    freq: int, 回望窗口大小。

    返回:
    DataFrame, 量价相关性。
    """
    correlation_df = pd.DataFrame(index=etf_amount.index)

    for column in etf_amount.columns:
        # 计算滚动窗口的量价相关系数
        rolling_corr = etf_amount[column].rolling(window=freq).corr(etf_close[column])
        correlation_df[column] = rolling_corr

    return correlation_df.dropna()


amountprice_corr = calculate_volume_price_correlation(etf_amount, etf_close, freq=freq)
amountprice_corr


# %%
# 绘制图表
plt.figure(figsize=(40, 20))

for ticker in amountprice_corr.columns:
    plt.plot(amountprice_corr.index, amountprice_corr[ticker], label=ticker)

plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
plt.xlabel('Date')
plt.ylabel('Volume-Price Correlation')
plt.title('Volume-Price Correlation for All ETFs')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# 上面的线太杂乱看不清
# \
# 归到 -1 0 1 上，得到新指标“integral_corr”

# %%

integral_corr = pd.DataFrame(index=amountprice_corr.index)

# 将量价相关性数据量化为 -1, 0, 1
for ticker in amountprice_corr.columns:
    integral_corr[ticker] = pd.cut(amountprice_corr[ticker], bins=[-1, -1/3, 1/3, 1], labels=[-1, 0, 1])
integral_corr = integral_corr.astype(int)

integral_corr

# %%
# 复制原始的 DataFrame
integral_corr_plt = integral_corr.copy()

# 修改日期格式
integral_corr_plt.index = pd.to_datetime(integral_corr_plt.index).strftime('%Y-%m-%d')

# 生成热力图
plt.figure(figsize=(14, 10))
sns.heatmap(integral_corr_plt.T, cmap='coolwarm', cbar=True, annot=False, center=0)
plt.show()

# %%
amountprice_corr_exam = correlation_exam(exam_df=amountprice_corr)

# %%
integral_corr_exam = correlation_exam(exam_df=integral_corr)

# %%
windows = [5, 10, 15, 20, 25, 30, 60, 90, 120, 150, 180]  
amountprice_corr_windows = {}
for window in windows:
    amountprice_corr_windows[window] = calculate_volume_price_correlation(etf_amount, etf_close, freq=window)
    correlation_exam(exam_df=amountprice_corr_windows[window], exam_df_name=f'amountprice_corr_{window}')
    vars()[f'amountprice_corr_{window}'] = amountprice_corr_windows[window]

# %% [markdown]
# ## 构建因子
# 
# 根据前文的指标构建与测试，选出以下最优指标，分别为：

# %% [markdown]
# amount_rollq_60 \
# proportion_rollq_15 \
# turnover_rollq_90 \
# turnover_deviation_30 \
# volatility_20 \
# excess_volatility_30
# 

# %% [markdown]
# 对于某一个表现较好的因子指标，选择邻近的滚动周期，进行赋权组合/提取主成分，以增强因子的鲁棒性
# 1. amount_rollq: 30 60 90
# 2. proportion_rollq: 15 20 25
# 3. turnover_rollq: 30 60 90
# 4. turnover_deviation: 30
# 5. volatility: 10 15 20
# 6. excess_volatility: 20 25 30

# %%
# 构建指标字典
indicators_selection = {
    'amount_rollq': [amount_rollq_30, amount_rollq_60, amount_rollq_90],
    'proportion_rollq': [proportion_rollq_15, proportion_rollq_20, proportion_rollq_25],
    'turnover_rollq': [turnover_rollq_30, turnover_rollq_60, turnover_rollq_90],
    'turnover_deviation': [turnover_deviation_30],
    'volatility': [volatility_10, volatility_15, volatility_20],
    'excess_volatility': [excess_volatility_20, excess_volatility_25, excess_volatility_30]
}

indicators_all = {
    'amount_rollq': [amount_rollq_5, amount_rollq_10, amount_rollq_15, amount_rollq_20, amount_rollq_25, amount_rollq_30, amount_rollq_60, amount_rollq_90],
    'proportion_rollq': [proportion_rollq_5, proportion_rollq_10, proportion_rollq_15, proportion_rollq_20, proportion_rollq_25, proportion_rollq_30, proportion_rollq_60, proportion_rollq_90],
    'turnover_rollq': [turnover_rollq_5, turnover_rollq_10, turnover_rollq_15, turnover_rollq_20, turnover_rollq_25, turnover_rollq_30, turnover_rollq_60, turnover_rollq_90],
    'turnover_deviation': [turnover_deviation_5, turnover_deviation_10, turnover_deviation_15, turnover_deviation_20, turnover_deviation_25, turnover_deviation_30, turnover_deviation_60, turnover_deviation_90],
    'volatility': [volatility_5, volatility_10, volatility_15, volatility_20, volatility_25, volatility_30, volatility_60, volatility_90],
    'excess_volatility': [excess_volatility_5, excess_volatility_10, excess_volatility_15, excess_volatility_20, excess_volatility_25, excess_volatility_30, excess_volatility_60, excess_volatility_90]
}

indicators_simple = {
    'amount_rollq_60': amount_rollq_60,
    'proportion_rollq_15': proportion_rollq_15,
    'turnover_rollq_90': turnover_rollq_90,
    'turnover_deviation_30': turnover_deviation_30,
    'volatility_20': volatility_20,
    'excess_volatility_30': excess_volatility_30
}

# %% [markdown]
# 针对indicators_selection和indicators_all，采用PCA方法构建因子（目前看来表现不佳，解释度不好，舍弃）

# %%
def pca_on_indicators(indicator_dict):
    """
    对每个指标组进行主成分分析（PCA），提取指定数量的主成分并生成一个新的 DataFrame。

    参数:
    indicator_dict: dict, 每个键对应一个指标组的 DataFrame 列表。

    返回:
    dict, 每个键对应一个提取的主成分构成的 DataFrame。
    """
    pca_results = {}

    for key, df_list in indicator_dict.items():
        # 初始化一个字典来存储每个 ETF 的主成分
        pca_result = {}

        # 获取所有的 ETF 列名（假设所有 DataFrame 的列名相同）
        etf_columns = df_list[0].columns

        # 对每个 ETF 列进行 PCA
        for etf in etf_columns:
            # 提取该 ETF 在所有 DataFrame 中的列
            etf_data = pd.concat([df[etf] for df in df_list], axis=1)

            # 去除包含 NaN 的行
            etf_data = etf_data.dropna()

            # 如果去除 NaN 后数据为空，跳过该 ETF
            if etf_data.empty:
                continue

            # 进行主成分分析
            pca = PCA(n_components=1)
            principal_component = pca.fit_transform(etf_data)

            # 存储主成分，并使用原始的日期索引
            pca_result[etf] = pd.Series(principal_component.flatten(), index=etf_data.index)

        # 将结果转换为 DataFrame 并存储在结果字典中
        pca_results[key] = pd.DataFrame(pca_result)

    return pca_results



# 执行主成分分析
indicators_combine = pca_on_indicators(indicators_selection)
indicators_combine


# %%
def plot_indicators(indicators):
    """
    绘制每个指标的图表。

    参数:
    indicators: dict, 包含多个 DataFrame 的字典。
    """
    for name, df in indicators.items():
        plt.figure(figsize=(60, 6))
        for column in df.columns:
            plt.plot(df.index, df[column], label=column)
        plt.title(f'{name}')
        plt.xlabel('date')
        plt.ylabel(name)
        plt.show()

plot_indicators(indicators_simple)

# %%
def normalize_and_merge_indicators(indicators):
    """
    归一化并合并多个 DataFrame，根据日期交集进行合并，计算等权平均值。

    参数:
    indicators: dict, 包含多个 DataFrame 的字典。

    返回:
    DataFrame, 归一化并等权平均后的 DataFrame。
    """
    # 提取所有 DataFrame 的日期交集
    common_dates = None
    for key, df in indicators.items():
        if common_dates is None:
            common_dates = df.index
        else:
            common_dates = common_dates.intersection(df.index)

    # 过滤每个 DataFrame 到交集日期范围，并归一化
    normalized_dfs = {}
    for key, df in indicators.items():
        filtered_df = df.loc[common_dates]
        normalized_df = (filtered_df - filtered_df.min()) / (filtered_df.max() - filtered_df.min())
        normalized_dfs[key] = normalized_df

    # 合并所有归一化后的 DataFrame
    combined_df = pd.concat(normalized_dfs.values(), axis=1)

    # 计算等权平均值
    mean_df = combined_df.groupby(combined_df.columns, axis=1).mean()

    return mean_df

# 调用函数进行处理
factor_df = normalize_and_merge_indicators(indicators_simple)
factor_df

# %%
def generate_signal(factor_df, quantile=0.9):
    """
    根据因子 DataFrame 生成信号 DataFrame。

    参数:
    factor_df: DataFrame, 因子 DataFrame。
    quantile: float, 分位数阈值。

    返回:
    DataFrame, 信号 DataFrame。
    """
    # 计算分位数阈值
    threshold = factor_df.quantile(quantile, axis=0)
    
    # 生成信号 DataFrame
    signal_df = (factor_df > threshold).astype(int)
    
    return signal_df

# 调用函数生成信号 DataFrame
signal = generate_signal(factor_df, quantile=0.9) #可调整门槛值
signal

# %% [markdown]
# ## 回测方法
# 运用锟哥的函数

# %%
def convert_signal_to_weight(signal):
    """
    将信号转换为权重，使用 equal_weight 方法。

    参数:
    signal: DataFrame, 含有信号值的 DataFrame。

    返回:
    DataFrame, 含有权重值的 DataFrame。
    """
    # 计算每天信号为1的ticker数量
    active_tickers_count = signal.sum(axis=1)

    # 避免除以零，将零值替换为 NaN，然后将 NaN 替换为零
    weights = 1 / active_tickers_count.replace(0, np.nan)

    # 通过信号 DataFrame 乘以权重 Series 构建权重 DataFrame
    weights_df = signal.multiply(weights, axis=0).fillna(0)

    return weights_df

# %%
# def vbt_backtest(portfolio_weight, price_data = etf_close, open_data = etf_open.shift(-1), initial_capital=10000, freq='days', year_freq='252 days'):
    
#     """
#     使用 vectorbt 对基于规则的因子进行回测

#     输入:
#         price_data: 收盘价数据的 DataFrame
#         open_data: 开盘价数据的 DataFrame（向后移位1）
#         freq: 投资组合再平衡的频率
#         year_freq: 区分年份的频率
#     """
    
#     # 设置
#     vbt.settings.array_wrapper['freq'] = freq
#     vbt.settings.returns['year_freq'] = year_freq
#     vbt.settings.portfolio.stats['incl_unrealized'] = True

#     # 对齐日期范围
#     open_data = open_data.dropna()
#     price_data = price_data.dropna()
    
#     start_date = max(open_data.index.min(), price_data.index.min(), portfolio_weight.index.min())
#     end_date = min(open_data.index.max(), price_data.index.max(), portfolio_weight.index.max())
    
#     adj_close_prices = price_data[(price_data.index >= start_date) & (price_data.index <= end_date)]
#     adj_close_prices = adj_close_prices[adj_close_prices.columns[:]]

#     adj_open_prices = open_data[(open_data.index >= start_date) & (open_data.index <= end_date)]
#     adj_open_prices = adj_open_prices[adj_open_prices.columns[:]]

#     portfolio_weight = portfolio_weight[(portfolio_weight.index >= start_date) & (portfolio_weight.index <= end_date)]
    
#     # 输入规模
#     size = portfolio_weight.to_numpy(dtype='float32')
    
#     # 构建列层次结构，使一个权重对应一个价格序列
#     num_tests = 1
#     input_close = adj_close_prices.vbt.tile(num_tests, keys=pd.Index(np.arange(num_tests), name='symbol_group'))
#     input_open = adj_open_prices.vbt.tile(num_tests, keys=pd.Index(np.arange(num_tests), name='symbol_group'))

#     # 创建投资组合进行测试
#     portfolio = vbt.Portfolio.from_orders(
#         input_close,
#         size=size,
#         price=input_open,
#         size_type='targetpercent',
#         group_by='symbol_group',
#         direction=2,
#         cash_sharing=True,
#         fees=0.00,
#         slippage=0/10000,
#         init_cash=initial_capital,
#         freq='1D',
#         min_size=0.0,
#         size_granularity=1, 
#         log=True, 
#         call_seq='auto',
#         ffill_val_price=True
#     )
        
#     # 显示交易记录中规模小于等于0的记录
#     print(portfolio.orders.records_readable[portfolio.orders.records_readable['Size'] <= 0])
    
#     # 结果展示
#     print(portfolio.stats())
#     equity_data = portfolio.value()
#     drawdown_data = portfolio.drawdown() * 100

#     # 绘制净值曲线
#     equity_trace = go.Scatter(x=equity_data.index, y=equity_data, mode='lines', name='净值曲线')
#     combined_traces = [equity_trace]
    
#     equity_layout = go.Layout(title='净值曲线', xaxis_title='日期', yaxis_title='净值')
#     fig = go.Figure(data=combined_traces, layout=equity_layout)

#     fig.update_layout(
#         yaxis_title="净值"
#     )

#     fig.add_hline(y=0, line_width=0.5, line_color="gray", line_dash="dash")
#     fig.show()

#     # 绘制回撤曲线
#     drawdown_trace = go.Scatter(
#         x=drawdown_data.index,
#         y=drawdown_data,
#         mode='lines',
#         name='回撤曲线',
#         fill='tozeroy',
#         line=dict(color='grey')
#     )
#     drawdown_layout = go.Layout(
#         title='回撤曲线',
#         xaxis_title='日期',
#         yaxis_title='回撤 %',
#         template='plotly_white'
#     )
#     drawdown_fig = go.Figure(data=[drawdown_trace], layout=drawdown_layout)
#     drawdown_fig.show()
    
#     return equity_data


# %%
portfolio = convert_signal_to_weight(signal)
portfolio

# %%
# result=vbt_backtest(portfolio, etf_close, etf_open.shift(-1), initial_capital=10000)

# %% [markdown]
# ## 优化因子

# %% [markdown]
# #### 单因子贡献回归测试
# 从单因子提取信号设权重，得到的净值序列回归全因子策略的净值序列

# %%
result

# %%
amount_rollq_60

# %%
# from numpy import single
# from sklearn.metrics import r2_score
# from scipy.stats import ttest_rel

# def reg_test(single_indicator, price_data = etf_close, open_data = etf_open.shift(-1) , result = result , quantile=0.9, initial_capital=10000):

#     # 生成信号列
#     def generate_signal(factor_df, quantile=0.9):
#         threshold = factor_df.quantile(quantile, axis=0)
#         signal_df = (factor_df > threshold).astype(int)
#         return signal_df

#     single_signal = generate_signal(single_indicator, quantile=quantile)
    
#     # 将信号转换为权重
#     def convert_signal_to_weight(signal):
#         active_tickers_count = signal.sum(axis=1)
#         weights = 1 / active_tickers_count.replace(0, np.nan)
#         weights_df = signal.multiply(weights, axis=0).fillna(0)
#         return weights_df

#     portfolio_weight = convert_signal_to_weight(single_signal)
    
#     # 回测函数
#     def vbt_backtest(portfolio_weight, price_data, open_data, initial_capital, freq='days', year_freq='252 days'):
#         vbt.settings.array_wrapper['freq'] = freq
#         vbt.settings.returns['year_freq'] = year_freq
#         vbt.settings.portfolio.stats['incl_unrealized'] = True

#         open_data = open_data.dropna()
#         price_data = price_data.dropna()
        
#         start_date = max(open_data.index.min(), price_data.index.min(), portfolio_weight.index.min())
#         end_date = min(open_data.index.max(), price_data.index.max(), portfolio_weight.index.max())
        
#         adj_close_prices = price_data[(price_data.index >= start_date) & (price_data.index <= end_date)]
#         adj_open_prices = open_data[(open_data.index >= start_date) & (open_data.index <= end_date)]
#         portfolio_weight = portfolio_weight[(portfolio_weight.index >= start_date) & (portfolio_weight.index <= end_date)]
        
#         size = portfolio_weight.to_numpy(dtype='float32')
        
#         input_close = adj_close_prices.vbt.tile(1, keys=pd.Index(np.arange(1), name='symbol_group'))
#         input_open = adj_open_prices.vbt.tile(1, keys=pd.Index(np.arange(1), name='symbol_group'))

#         portfolio = vbt.Portfolio.from_orders(
#             input_close,
#             size=size,
#             price=input_open,
#             size_type='targetpercent',
#             group_by='symbol_group',
#             direction=2,
#             cash_sharing=True,
#             fees=0.00,
#             slippage=0/10000,
#             init_cash=initial_capital,
#             freq='1D',
#             min_size=0.0,
#             size_granularity=1, 
#             log=True, 
#             call_seq='auto',
#             ffill_val_price=True
#         )
        
#         equity_data = portfolio.value()
#         return equity_data
    
#     # 计算策略的 equity_data
#     equity_data = vbt_backtest(portfolio_weight, price_data, open_data, initial_capital)
    
#     # 对齐数据日期范围
#     equity_data = equity_data[equity_data.index.isin(result.index)]
#     result = result[result.index.isin(equity_data.index)]
    
#     # 回归分析
#     X = equity_data.values.reshape(-1, 1)  # 转换为二维数组
#     y = result.values
#     model = LinearRegression()
#     model.fit(X, y)
#     y_pred = model.predict(X)
    
#     # 输出回归结果
#     print(f'回归系数: {model.coef_[0]}')
#     print(f'R^2: {r2_score(y, y_pred)}')
    
#     # 进行t检验
#     t_stat, p_value = ttest_rel(equity_data, result)
#     print(f't统计量: {t_stat}')
#     print(f'p值: {p_value}')

# %%
# reg_test(amount_rollq_60)

# %% [markdown]
# #### 规则类判断
# ##### 获取门槛值
# 1、 \
# 分位数类型指标 触发门槛 = 0.5 \
# 乖离度指标 触发门槛 = 0 \
# 其他指标年化/标准化后选中位数 \
# \
# 2、\
# 都选小样本中位数，假设小样本具备总体的性质
# 
# 

# %%
indicators_all = {
    'amount_rollq': [amount_rollq_5, amount_rollq_10, amount_rollq_15, amount_rollq_20, amount_rollq_25, amount_rollq_30, amount_rollq_60, amount_rollq_90],
    'proportion_rollq': [proportion_rollq_5, proportion_rollq_10, proportion_rollq_15, proportion_rollq_20, proportion_rollq_25, proportion_rollq_30, proportion_rollq_60, proportion_rollq_90],
    'turnover_rollq': [turnover_rollq_5, turnover_rollq_10, turnover_rollq_15, turnover_rollq_20, turnover_rollq_25, turnover_rollq_30, turnover_rollq_60, turnover_rollq_90],
    'turnover_deviation': [turnover_deviation_5, turnover_deviation_10, turnover_deviation_15, turnover_deviation_20, turnover_deviation_25, turnover_deviation_30, turnover_deviation_60, turnover_deviation_90],
    'volatility': [volatility_5, volatility_10, volatility_15, volatility_20, volatility_25, volatility_30, volatility_60, volatility_90],
    'excess_volatility': [excess_volatility_5, excess_volatility_10, excess_volatility_15, excess_volatility_20, excess_volatility_25, excess_volatility_30, excess_volatility_60, excess_volatility_90]
}

# %% [markdown]
# 缩小范围

# %%
indicators_selection = {
    'amount_rollq': [amount_rollq_30, amount_rollq_60, amount_rollq_90],
    'proportion_rollq': [proportion_rollq_15, proportion_rollq_20, proportion_rollq_25],
    'turnover_rollq': [turnover_rollq_30, turnover_rollq_60, turnover_rollq_90],
    'turnover_deviation': [turnover_deviation_30],
    'volatility': [volatility_10, volatility_15, volatility_20],
    'excess_volatility': [excess_volatility_20, excess_volatility_25, excess_volatility_30]
}

# %%
for indicator_type, data_frames in indicators_all.items():
    combined_df = pd.concat(data_frames)
    median_value = combined_df.stack().median()
    print(f"{indicator_type} 的中位数:", median_value)

# %% [markdown]
# 根据indicators_all计算得到的样本中位数，保存一下： 
# - amount_rollq 的中位数: 0.52 
# - proportion_rollq 的中位数: 0.5 
# - turnover_rollq 的中位数: 0.5166666666666667 
# - turnover_deviation 的中位数: -0.26885789879070043 
# - volatility 的中位数: 0.9734314850290797 
# - excess_volatility 的中位数: 0.03248792400541715 

# %% [markdown]
# 根据样本中位数设立门槛值

# %%
def calculate_trigger_counts(data_frames, threshold=0.5):
    """
    对于每一个ticker和每一个日期，如果在多个DataFrame中的数字大于门槛值（如0.5）就计数1。
    最后输出一个DataFrame包含每个ticker每个日期的指标触发数量。
    
    参数：
    data_frames (list): 包含多个DataFrame的列表，每个DataFrame代表一个时间段的数据
    threshold (float): 阈值，默认值为0.5
    
    返回：
    pd.DataFrame: 包含每个日期和每个ticker的指标触发数量
    """
    all_tickers = data_frames[0].columns
    all_dates = data_frames[0].index
    
    result_df = pd.DataFrame(0, index=all_dates, columns=all_tickers)
    
    for df in data_frames:
        boolean_df = df > threshold
        int_df = boolean_df.astype(int)
        result_df = result_df.add(int_df, fill_value=0)
    
    return result_df


# %%
# amount_rollq_counts = calculate_trigger_counts(indicators_all['amount_rollq'], threshold=0.5)
# proportion_rollq_counts = calculate_trigger_counts(indicators_all['proportion_rollq'], threshold=0.5)
# turnover_rollq_counts = calculate_trigger_counts(indicators_all['turnover_rollq'], threshold=0.5)
# turnover_deviation_counts = calculate_trigger_counts(indicators_all['turnover_deviation'], threshold=0)
# volatility_counts = calculate_trigger_counts(indicators_all['volatility'], threshold=1)
# excess_volatility_counts = calculate_trigger_counts(indicators_all['excess_volatility'], threshold=0.04)

# %%
amount_rollq_counts = calculate_trigger_counts(indicators_selection['amount_rollq'], threshold=0.5)
proportion_rollq_counts = calculate_trigger_counts(indicators_selection['proportion_rollq'], threshold=0.5)
turnover_rollq_counts = calculate_trigger_counts(indicators_selection['turnover_rollq'], threshold=0.5)
turnover_deviation_counts = calculate_trigger_counts(indicators_selection['turnover_deviation'], threshold=0)
volatility_counts = calculate_trigger_counts(indicators_selection['volatility'], threshold=1)
excess_volatility_counts = calculate_trigger_counts(indicators_selection['excess_volatility'], threshold=0.04)

# %% [markdown]
# ##### **跨类组合设置信号** 
# \
# 结果为merged_trigger_XXX \
# threshold= num(indicators_all) / 2

# %%
def merge_trigger_counts(*trigger_counts_dfs):
    """
    合并多个触发计数的DataFrame，对应格子的数字进行加总。
    
    参数：
    trigger_counts_dfs (list): 包含多个触发计数的DataFrame
    
    返回：
    pd.DataFrame: 合并后的触发计数DataFrame
    """
    result_df = pd.DataFrame()
    
    for df in trigger_counts_dfs:
        if result_df.empty:
            result_df = df.copy()
        else:
            result_df = result_df.add(df, fill_value=0)
    
    return result_df

# %%
# 合并四个触发计数的df
merged_trigger_counts = merge_trigger_counts(amount_rollq_counts, proportion_rollq_counts, turnover_rollq_counts, turnover_deviation_counts, volatility_counts, excess_volatility_counts)
merged_threshold = 8 # 自定义合并指标触发门槛
merged_trigger_signal = (merged_trigger_counts > merged_threshold).astype(int)
merged_trigger_signal

# %%
# merged_trigger_result = vbt_backtest(merged_trigger_portfolio)
# merged_trigger_portfolio.to_csv('etf_indicators/merged_trigger_porfolio.csv')
# merged_trigger_result

# %% [markdown]
# ##### **分类指标设置信号** 
# \
# 触发值均为1/2 \
# 结果为merged_signal_XXX

# %% [markdown]
# 先看分类指标的回测结果
# 1. 成交额分位数

# %%
amount_rollq_signal = (amount_rollq_counts > 1).astype(int)
# amount_rollq_result = vbt_backtest(convert_signal_to_weight(amount_rollq_signal))

# %% [markdown]
# 2. 成交额市场比分位数

# %%
proportion_rollq_signal = (proportion_rollq_counts > 1).astype(int)
# proportion_rollq_result = vbt_backtest(convert_signal_to_weight(proportion_rollq_signal))

# %% [markdown]
# 3. 换手率分位数

# %%
turnover_rollq_signal = (turnover_rollq_counts > 1).astype(int)
# turnover_rollq_result = vbt_backtest(convert_signal_to_weight(turnover_rollq_signal))

# %% [markdown]
# 4. 换手率乖离度

# %%
turnover_deviation_signal = (turnover_deviation_counts > 0).astype(int)  #调整触发门槛
# turnover_deviation_result = vbt_backtest(convert_signal_to_weight(turnover_deviation_signal))

# %% [markdown]
# 5、年化波动率

# %%
volatility_signal = (volatility_counts > 1).astype(int)  #调整触发门槛
# volatility_result = vbt_backtest(convert_signal_to_weight(volatility_signal))

# %% [markdown]
# 6、年化超额波动率

# %%
excess_volatility_signal = (excess_volatility_counts > 1).astype(int)  #调整触发门槛
# exess_volatility_result = vbt_backtest(convert_signal_to_weight(excess_volatility_signal))

# %% [markdown]
# 将上述分类指标的信号合并，再设立门槛值生成新信号

# %%
merged_signal_counts = merge_trigger_counts(amount_rollq_signal, proportion_rollq_signal, turnover_rollq_signal, turnover_deviation_signal, volatility_signal, excess_volatility_signal)
merged_signal_counts

# %%
merged_signal_sign = (merged_signal_counts > 3).astype(int)
# merged_signal_result = vbt_backtest(convert_signal_to_weight(merged_signal_sign))

# %% [markdown]
# #### 修改回测代码
# 1. “因子——信号——权重” 的逻辑修改为 “因子——权重” 
# 2. 增加滞后性，前一天的拥挤度指标用于后一天的调仓
# 3. 从拥挤交易观察回撤到筛除拥挤观察收益，避险逻辑替换为因子逻辑

# %%
daily_returns = etf_open.pct_change().fillna(0)
daily_returns

# %% [markdown]
# 拿一个组合因子做一下测试

# %%
merged_trigger_signal = (merged_trigger_counts <= 8).astype(int)
merged_trigger_signal

# %%
weights = merged_trigger_signal.shift(1).fillna(0)
weights = weights.div(weights.sum(axis=1), axis=0).fillna(0)

# 计算组合的每日收益率
portfolio_daily_returns = (weights * daily_returns).sum(axis=1).fillna(0)
portfolio_daily_returns.replace([np.inf, -np.inf], 0, inplace=True)
portfolio_daily_returns = portfolio_daily_returns.reindex(pd.date_range(start_date, end_date), fill_value=0)

# 起止日期根据信号的起止日期
start_date = merged_trigger_signal.index.min()
end_date = merged_trigger_signal.index.max()
portfolio_daily_returns = portfolio_daily_returns[start_date:end_date]

portfolio_daily_returns


# %%
portfolio_value = (1 + portfolio_daily_returns).cumprod()
portfolio_value

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines', name='净值曲线'))
fig.update_layout(title='净值曲线', xaxis_title='日期', yaxis_title='净值')
fig.show()

# %% [markdown]
# 前面的bug出现在有一些值是无穷大 \
# 下面是**封装的新函数**

# %%
def calculate_portfolio_performance(signals_df, threshold, greater_than=False, open_prices=etf_open):
    """
    计算组合的净值曲线、年化收益率和最大回撤。
    
    参数：
    signals_df: DataFrame，包含买卖信号的DataFrame。
    threshold: float，门槛值。
    greater_than: bool，True表示信号值大于门槛值时持有，False表示信号值小于门槛值时持有。
    open_prices: DataFrame，每个ETF的开盘价。
    
    返回：
    portfolio_value: Series，组合的净值曲线。
    """
    
    # 根据门槛值提取买卖信号
    if greater_than:
        signals = (signals_df > threshold).astype(int)
    else:
        signals = (signals_df < threshold).astype(int)
    
    # 计算每日的 ETF 收益率
    daily_returns = open_prices.pct_change().fillna(0)
    
    # 将买卖信号转换为持仓权重，1 表示持有，0 表示不持有
    weights = signals.shift(1).fillna(0)
    weights = weights.div(weights.sum(axis=1), axis=0).fillna(0)
    
    # 计算组合的每日收益率
    portfolio_daily_returns = (weights * daily_returns).sum(axis=1).fillna(0)
    
    # 起止日期根据信号的起止日期
    start_date = signals.index.min()
    end_date = signals.index.max()
    portfolio_daily_returns = portfolio_daily_returns[start_date:end_date]
    
    # 填补缺失值为0，并将异常值（如 inf）替换为0
    portfolio_daily_returns.replace([np.inf, -np.inf], 0, inplace=True)
    portfolio_daily_returns = portfolio_daily_returns.reindex(pd.date_range(start_date, end_date), fill_value=0)
    
    # 计算净值变化
    portfolio_value = (1 + portfolio_daily_returns).cumprod()
    
    # 计算年化收益率
    total_days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
    annual_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (365 / total_days) - 1
    
    # 计算最大回撤
    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 计算最大涨幅
    running_min = portfolio_value.cummin()
    upward_movement = (portfolio_value - running_min) / running_min
    max_upward_movement = upward_movement.max()

    # 打印年化收益率和最大回撤
    print(f"年化收益率： {annual_return:.2%}")
    print(f"最大回撤： {max_drawdown:.2%}")
    print(f"最大涨幅： {max_upward_movement:.2%}")
    
    # 绘制净值曲线
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines', name='净值曲线'))
    fig.update_layout(title='净值曲线', xaxis_title='日期', yaxis_title='净值')
    fig.show()
    
    return portfolio_value



# %% [markdown]
# 以下仅为函数测试 \
# 理论上跑出来的结果应该和vbt_basket相同，但是好像反了？

# %%
calculate_portfolio_performance(signal, threshold=0.5, greater_than=True)

# %% [markdown]
# #### 进一步完善构建组合的逻辑

# %% [markdown]
# 通过观察单指标和未来20天最大回撤，第二次筛选出比较有效的指标 \
# （参考华泰金工：绝对收益型ETF轮动策略）

# %%
indicators_brief = {
    'proportion_rollq': [proportion_rollq_5, proportion_rollq_10, proportion_rollq_15, proportion_rollq_20, proportion_rollq_25],
    'turnover_deviation': [turnover_deviation_5, turnover_deviation_10, turnover_deviation_120, turnover_deviation_150, turnover_deviation_180],
    'volatility': [volatility_5, volatility_10, volatility_15, volatility_20, volatility_25],
}

# %%
proportion_rollq_counts = calculate_trigger_counts(indicators_brief['proportion_rollq'], threshold=0.95)
proportion_rollq_signal = (proportion_rollq_counts >= 3).astype(int)

# %%
turnover_deviation_counts = calculate_trigger_counts(indicators_brief['turnover_deviation'], threshold=0.5)
turnover_deviation_signal = (turnover_deviation_counts >= 3).astype(int)

# %%
volatility_counts = calculate_trigger_counts(indicators_brief['volatility'], threshold=1.5)
volatility_signal = (volatility_counts >= 3).astype(int)

# %%
huatai_signal = proportion_rollq_signal + turnover_deviation_signal + volatility_signal

# %%
calculate_portfolio_performance(huatai_signal, 2)

# %% [markdown]
# 在前边的尝试中，我发现在现有的ETF池中，筛选的量价指标同时存在动量和反转效应，即使对于单一指标而言，取得绝对收益的条件非常苛刻，指标组合的效果更差。 \
# 尽管这不妨碍，在极端拥挤的条件下损失同样会很大。 \
# 因此我们不妨更换思路，在现有的指标构建基础上：
# 1. 增加更多新指标的尝试
# 2. 放弃规则类判断组合的方式，选用连续因子+门槛值调参的办法，减小颗粒度
# 3. 通过计算ICIR的方法检验因子是否有效

# %% [markdown]
# ##### 计算ICIR的函数
# 输入构建完的因子即可，lag为滞后期

# %%
def calculate_icir(factor_data, close_prices=etf_close, lag=0):
    """
    计算每个ETF在时间序列上的IC（Information Coefficient）和ICIR。

    参数：
    - factor_data: DataFrame, 因子值数据，行索引为日期，列索引为ETF代码
    - close_prices: DataFrame, 收盘价数据，行索引为日期，列索引为ETF代码
    - lag: int, 因子滞后天数，默认为0

    返回：
    - icir: float, ICIR值
    """

    # 将因子数据进行滞后处理
    factor_data = factor_data.shift(lag)
    factor_data.dropna(inplace=True)
    # 计算未来一天的收益率
    future_returns = close_prices.pct_change().shift(-1)
    future_returns.dropna(inplace=True)

    # 确保因子数据和收盘价数据的时间序列对齐
    common_dates = factor_data.index.intersection(future_returns.index)
    factor_data = factor_data.loc[common_dates]
    future_returns = future_returns.loc[common_dates]

    # 计算每个日期的IC
    ic_series = []
    for date in common_dates:
        daily_factor = factor_data.loc[date]
        daily_return = future_returns.loc[date]
        ic, _ = spearmanr(daily_factor, daily_return)
        ic_series.append(ic)

    ic_series = pd.Series(ic_series, index=common_dates)

    # 根据IC计算ICIR
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    icir = ic_mean / ic_std

    print(ic_series)

    return icir

# %%
calculate_icir(amountprice_corr_120)  # 测试icir的函数

# %% [markdown]
# ##### 构建proportion因子 
# 在20天最大回撤相关性检验中，发现较短窗口期的成交额市场比分位数表现更优，因此我们选用window=10，20，30的proportion滚动分位数构建第一个因子

# %% [markdown]
# 1、等权相加

# %%
proportion_factor1 = 1 - (proportion_rollq_10 * 1/3 + proportion_rollq_20 * 1/3 + proportion_rollq_30 * 1/3)
proportion_factor1.dropna(inplace=True)
proportion_factor1

# %%
calculate_icir(proportion_factor1, lag=3)

# %%
proportion_factor1_lag3 = proportion_factor1.shift(3).dropna(inplace=False)
proportion_factor1_lag3 

# %%
calculate_portfolio_performance(proportion_factor1_lag3, 0.5, True)

# %%
calculate_icir(proportion_factor1, lag=2)

# %% [markdown]
# 由上，proportion_rollq（10、20、30）在滞后2-3期有效性显著上升，到lag=4衰减

# %%
proportion_factor1_lag2 = proportion_factor1.shift(2).dropna(inplace=False)
proportion_factor1_lag2 

# %%
calculate_portfolio_performance(proportion_factor1_lag2, 0.5, True)

# %% [markdown]
# 等权相加的构建逻辑下，滞后3期的策略收益和icir更优

# %% [markdown]
# 2、window越小，权重越大

# %%
proportion_factor2 = 1 - (proportion_rollq_10 * 1/2 + proportion_rollq_20 * 1/3 + proportion_rollq_30 * 1/6)
proportion_factor2.dropna(inplace=True)
proportion_factor2

# %%
calculate_icir(proportion_factor2, lag=3)

# %%
proportion_factor2_lag3 = proportion_factor2.shift(3).dropna(inplace=False)
calculate_portfolio_performance(proportion_factor2_lag3, 0.5)

# %% [markdown]
# 效果明显比等权相加要差很多

# %%
proportion_factor3 = 1 - (proportion_rollq_10 * 0.5 + proportion_rollq_20 * 0.3 + proportion_rollq_30 * 0.2)
proportion_factor3.dropna(inplace=True)
calculate_icir(proportion_factor3, lag=3)

# %% [markdown]
# 结论：proportion用等权相加的方式构建因子即可（可考虑滞后3期） \
# 下面开始构建短窗口期下的proportion_rollq_factor

# %%
proportion_rollq_factor_510 = 1 - (proportion_rollq_5 * 1/2 + proportion_rollq_10 * 1/2)   
proportion_rollq_factor_510.dropna(inplace=True)
calculate_portfolio_performance(proportion_rollq_factor_510, 0.5, True)

# %%
proportion_rollq_factor_120240 = 1 - (proportion_rollq_120 * 1/2 + proportion_rollq_240 * 1/2)
proportion_rollq_factor_120240.dropna(inplace=True)
calculate_portfolio_performance(proportion_rollq_factor_120240, 0.5, True)

# %% [markdown]
# ##### 构建turnover因子
# 
# **换手率分位数因子** \
# turnover_rollq在小窗口期有效，turnover_deviation在两端有效 \
# 参考proportion因子，用等权相加的方法

# %%
turnover_rollq_factor_51015 = 1 - (turnover_rollq_5 * 1/3 + turnover_rollq_10 * 1/3 + turnover_rollq_15 * 1/3)
turnover_rollq_factor_510 = 1 - (turnover_rollq_5 * 1/2 + turnover_rollq_10 * 1/2)
turnover_rollq_factor_51015.dropna(inplace=True)
turnover_rollq_factor_510.dropna(inplace=True)

# %%
turnrollq_weekfactor_result = calculate_portfolio_performance(turnover_rollq_factor_510, 0.6, True)

# %%
calculate_icir(turnover_rollq_factor_510) 

# %%
# turnover_rollq_factor_lag1 = turnover_rollq_factor_51015.shift(1).dropna(inplace=False)
# turnover_rollq_factor_lag2 = turnover_rollq_factor_51015.shift(2).dropna(inplace=False)
# turnover_rollq_factor_lag3 = turnover_rollq_factor_51015.shift(3).dropna(inplace=False)

# %%
# lags = [turnover_rollq_factor, turnover_rollq_factor_lag1, turnover_rollq_factor_lag2, turnover_rollq_factor_lag3]
# for i, lag in enumerate(lags, start=0):
#     print(f"测试滞后{i}天的因子表现")
#     calculate_portfolio_performance(lag, 0.7, True)

# %% [markdown]
# 前面是turnover_rollq在短窗口期（5d、10d）的组合因子 \
# 下面测试长周期（120d、240d），观察是否同样存在反转效应

# %%
turnover_rollq_factor_120240 = 1 - (turnover_rollq_120 * 1/2 + turnover_rollq_240 * 1/2)
turnover_rollq_factor_50120240 = 1 - (turnover_rollq_50 * 1/3 + turnover_rollq_120 * 1/3 + turnover_rollq_240 * 1/3)
turnover_rollq_factor_120240.dropna(inplace=True)
turnover_rollq_factor_50120240.dropna(inplace=True)

# %%
calculate_icir(turnover_rollq_factor_120240)

# %%
turnrollq_annfactor_result = calculate_portfolio_performance(turnover_rollq_factor_120240, 0.9, True)

# %%
turnrollq_annfactor_result.corr(turnrollq_weekfactor_result)

# %%
turnover_rollq_factor_120240

# %% [markdown]
# **换手率乖离度因子**

# %%
turnover_deviation_factor = (1 - turnover_deviation_15 * 1/3 - turnover_deviation_20 * 1/3 - turnover_deviation_30 * 1/3)
turnover_deviation_factor.dropna(inplace=True)
turnover_deviation_factor

# %%
calculate_icir(turnover_deviation_factor, lag=3)

# %%
turnover_deviation_factor_lag3 = turnover_deviation_factor.shift(3).dropna()

# %%
calculate_portfolio_performance(turnover_deviation_factor, 2, True)

# %% [markdown]
# ##### 构建volatility因子

# %% [markdown]
# 短期波动率和未来20d最大回撤相关性较大，选用5d和10d的volatility进行组合

# %%
for volatility_i in [volatility_5, volatility_10, volatility_120, volatility_240]:
    print(volatility_i.stack().median())

# %% [markdown]
# 不同窗口期的波动率差异显著，说明波动率忘记年化了，返回去修改一下（已修改）

# %%
volatility_factor_510 = volatility_5*1/2 + volatility_10*1/2
volatility_factor_510.dropna(inplace=True)

# %%
calculate_icir(volatility_factor_510)

# %%
volatility_factor_510

# %%
for i in range(10, 50, 5):
    var_name = f"volafactor510_thres{i}_result"
    print(f"测试短窗口volatility_rollq因子阈值{i/100}的收益表现")
    exec(f"{var_name} = calculate_portfolio_performance(volatility_factor_510, i/100)")

# %% [markdown]
# 波动率因子的中位数在0.2左右。低波动率预示着缺少支撑，价格具有下行压力。实际回测结果也说明，当我们筛选低波etf持仓时，收益显著为负。

# %% [markdown]
# 同样地，观察长窗口期下的波动率因子表现

# %%
volatility_factor_120240 = volatility_120 * 1/2 + volatility_240 * 1/2
volatility_factor_120240.dropna(inplace=True)
print(volatility_factor_120240.stack().median())
print(calculate_icir(volatility_factor_120240))
volatility_factor_120240

# %%
for i in range(0, 50, 5):
    var_name = f"volafactor120240_thres{i}_result"
    print(f"测试长窗口volatility因子阈值{i/100}的收益表现")
    exec(f"{var_name} = calculate_portfolio_performance(volatility_factor_120240, i/100, True)")

# %% [markdown]
# 长窗口期下的波动率因子表现一般，但icir值为正（短周期下icir<0）

# %% [markdown]
# ##### 组合上述因子

# %%
proturn = proportion_rollq_factor_510 * 1/2 + turnover_rollq_factor_510 * 1/2
print(proturn.stack().median(), proturn.stack().mean())
proturn.dropna(inplace=True)
print(calculate_icir(proturn))

# %%
for i in range(1, 10):
    var_name = f"proturn_thres{i}_result"
    print(f"测试proturn因子阈值{i/10}的收益表现")
    exec(f"{var_name} = calculate_portfolio_performance(proturn, i/10, True)")

# %%
proturnvol = proportion_rollq_factor_510 * 1/3 + turnover_rollq_factor_510 * 1/3 + volatility_factor_510 * 1/3
proturnvol.dropna(inplace=True)
print(proturnvol.stack().median(), proturnvol.stack().mean())
calculate_icir(proturnvol)

# %%
for i in range(1, 10):
    var_name = f"proturnvol_thres{i}_result"
    print(f"测试proturnvol因子阈值{i/10}的收益表现")
    exec(f"{var_name} = calculate_portfolio_performance(proturnvol, i/10, True)")

# %% [markdown]
# 测一下volatility对原先proportion和turnover指标组合的贡献

# %%
# 归一化
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
proturn_normalized = pd.DataFrame(scaler.fit_transform(proturn), columns=proturn.columns, index=proturn.index)
proturnvol_normalized = pd.DataFrame(scaler.fit_transform(proturnvol), columns=proturnvol.columns, index=proturnvol.index)

# %%
proturn_normalresult = calculate_portfolio_performance(proturn_normalized, 0.5, True)
proturnvol_normalresult = calculate_portfolio_performance(proturnvol_normalized, 0.5, True)
print(f'proturn的波动率为{proturn_normalresult.std()}, proturnvol的波动率为{proturnvol_normalresult.std()}')

# %% [markdown]
# 总体上看，添加波动率的因子组合对回测结果几乎无影响（贡献不大）

# %% [markdown]
# 发现2022年4月底到7月初连续上涨，查看股指表现

# %%
def fetch_and_plot_index(indices, start_date, end_date):
    """
    获取指定日期范围内的股票指数数据并绘制图表。

    参数:
    indices (list): 股票指数代码列表。
    start_date (str): 起始日期，格式为 'YYYY-MM-DD'。
    end_date (str): 结束日期，格式为 'YYYY-MM-DD'。
    """

    # 获取指数数据
    data = {}
    for index in indices:
        wsd_data = w.wsd(index, "close", start_date, end_date, "")
        if wsd_data.ErrorCode != 0:
            print(f"Error in fetching data for {index}: {wsd_data.ErrorCode}")
            continue
        data[index] = pd.DataFrame({'date': wsd_data.Times, 'close': wsd_data.Data[0]})

    # 绘制图表
    plt.figure(figsize=(10, 6))
    for index, df in data.items():
        plt.plot(df['date'], df['close'], label=index)

    plt.title(f'index from {start_date} to {end_date}')
    plt.legend()
    plt.grid(False)
    plt.show()

fetch_and_plot_index(['000300.SH', '000905.SH', '000906.SH', '000852.SH'], '2022-04-26', '2022-07-05')

# %% [markdown]
# etf跟着股指一同上涨，莫得问题
# 
# 不妨看一下起止日期之间的指数走势

# %%
fetch_and_plot_index(['000300.SH', '000905.SH'], '2021-06-30', '2024-06-30')

# %%



