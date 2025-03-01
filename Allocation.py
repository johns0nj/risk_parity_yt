import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import time  # 用于计算程序运行时间

# 获取ETF历史数据
def get_etf_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    data = data.fillna(method='ffill')
    return data

# 计算日收益率
def calculate_daily_returns(data):
    return data.pct_change().dropna()

# 计算年化波动率
def calculate_annualized_volatility(daily_returns):
    return daily_returns.std() * np.sqrt(252)

# 计算夏普比率
def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.0):
    excess_returns = daily_returns - risk_free_rate / 252
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return sharpe_ratio

# 计算最大回撤
def calculate_max_drawdown(portfolio_value):
    cumulative_max = portfolio_value.cummax()
    drawdown = (portfolio_value - cumulative_max) / cumulative_max
    return drawdown.min()

# 计算回撤数据
def calculate_drawdown(portfolio_value):
    cumulative_max = portfolio_value.cummax()
    drawdown = (portfolio_value - cumulative_max) / cumulative_max
    return drawdown

# 计算年化收益率
def calculate_annualized_return(portfolio_value):
    return (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (252 / len(portfolio_value)) - 1

# 计算总收益
def calculate_total_return(portfolio_value):
    return (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1

# 计算月度胜率
def calculate_monthly_win_rate(data):
    monthly_returns = data.resample('M').last().pct_change().dropna()
    win_rate = (monthly_returns > 0).mean()
    return win_rate

# 风险平价优化目标函数
def risk_parity_objective(weights, cov_matrix):
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    risk_contributions = weights * np.dot(cov_matrix, weights) / portfolio_volatility
    target_risk_contributions = portfolio_volatility / len(weights)
    return np.sum((risk_contributions - target_risk_contributions) ** 2)

# 风险平价优化
def risk_parity_optimization(cov_matrix):
    num_assets = cov_matrix.shape[0]
    initial_weights = np.ones(num_assets) / num_assets
    bounds = [(0, 1) for _ in range(num_assets)]
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    result = minimize(risk_parity_objective, initial_weights, args=(cov_matrix,),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# 最大化夏普比率的目标函数
def maximize_sharpe_objective(weights, daily_returns, risk_free_rate):
    portfolio_return = np.dot(daily_returns.mean(), weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov() * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio  # 负号是因为 minimize 是最小化目标函数

# 最大化夏普比率的优化（无权重约束）
def maximize_sharpe_optimization(daily_returns, risk_free_rate=0.0):
    num_assets = daily_returns.shape[1]
    initial_weights = np.ones(num_assets) / num_assets
    bounds = [(0, 1) for _ in range(num_assets)]
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    result = minimize(maximize_sharpe_objective, initial_weights, args=(daily_returns, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# 最大化夏普比率的优化（每类资产权重不小于5%）
def maximize_sharpe_optimization_min5(daily_returns, risk_free_rate=0.0):
    num_assets = daily_returns.shape[1]
    initial_weights = np.ones(num_assets) / num_assets
    bounds = [(0.05, 1) for _ in range(num_assets)]  # 每类资产权重不小于5%
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    result = minimize(maximize_sharpe_objective, initial_weights, args=(daily_returns, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# 最大化夏普比率的优化（允许做空）
def maximize_sharpe_optimization_unconstrained(daily_returns, risk_free_rate=0.0):
    num_assets = daily_returns.shape[1]
    initial_weights = np.ones(num_assets) / num_assets
    bounds = [(-1, 1) for _ in range(num_assets)]  # 允许权重在 -1 到 1 之间
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    result = minimize(maximize_sharpe_objective, initial_weights, args=(daily_returns, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# 固定比例组合的权重
def fixed_weight_portfolio(tickers):
    # 定义资产类别
    stock_tickers = ['IWB', 'IWM']  # 股票类
    bond_tickers = ['IEF', 'LQD']   # 债券类
    commodity_tickers = ['GLD', 'USO']  # 商品类
    other_tickers = ['IYR', 'VNQ']  # 其他类

    # 计算每类资产的权重
    stock_weight = 0.48 / len(stock_tickers)
    bond_weight = 0.32 / len(bond_tickers)
    commodity_weight = 0.10 / len(commodity_tickers)
    other_weight = 0.10 / len(other_tickers)

    # 构建权重字典
    weights = {}
    for ticker in tickers:
        if ticker in stock_tickers:
            weights[ticker] = stock_weight
        elif ticker in bond_tickers:
            weights[ticker] = bond_weight
        elif ticker in commodity_tickers:
            weights[ticker] = commodity_weight
        elif ticker in other_tickers:
            weights[ticker] = other_weight
        else:
            weights[ticker] = 0.0

    # 转换为权重数组
    return np.array([weights[ticker] for ticker in tickers])

# 回测函数
def backtest_strategies(tickers, start_date, end_date, look_back, risk_free_rate=0.0):
    print("开始获取ETF历史数据...")
    data_start_date = (datetime.strptime(start_date, '%Y-%m-%d') - relativedelta(months=look_back)).strftime('%Y-%m-%d')
    data = get_etf_data(tickers, data_start_date, end_date)
    data = data.loc[:, tickers]
    daily_returns = calculate_daily_returns(data)
    print("ETF历史数据获取完成，开始回测...")

    # 初始化组合净值和权重记录
    portfolio_value_rp = pd.Series(1.0, index=data.index[data.index.get_loc(pd.to_datetime(start_date)) - 1:])  # 风险平价组合
    portfolio_value_ms = pd.Series(1.0, index=data.index[data.index.get_loc(pd.to_datetime(start_date)) - 1:])  # 最大化夏普比率组合
    portfolio_value_ms_min5 = pd.Series(1.0, index=data.index[data.index.get_loc(pd.to_datetime(start_date)) - 1:])  # 最大化夏普比率组合（每类资产>=5%）
    portfolio_value_ms_unconstrained = pd.Series(1.0, index=data.index[data.index.get_loc(pd.to_datetime(start_date)) - 1:])  # 最大化夏普比率组合（允许做空）
    portfolio_value_fixed = pd.Series(1.0, index=data.index[data.index.get_loc(pd.to_datetime(start_date)) - 1:])  # 固定比例组合
    weights_rp = np.ones(len(tickers)) / len(tickers)  # 风险平价组合初始权重
    weights_ms = np.ones(len(tickers)) / len(tickers)  # 最大化夏普比率组合初始权重
    weights_ms_min5 = np.ones(len(tickers)) / len(tickers)  # 最大化夏普比率组合（每类资产>=5%）初始权重
    weights_ms_unconstrained = np.ones(len(tickers)) / len(tickers)  # 最大化夏普比率组合（允许做空）初始权重
    weights_fixed = fixed_weight_portfolio(tickers)  # 固定比例组合权重
    weight_history_rp = []  # 风险平价组合权重历史
    weight_history_ms = []  # 最大化夏普比率组合权重历史
    weight_history_ms_min5 = []  # 最大化夏普比率组合（每类资产>=5%）权重历史
    weight_history_ms_unconstrained = []  # 最大化夏普比率组合（允许做空）权重历史
    rebalance_dates = []
    portfolio_daily_returns_rp = []  # 风险平价组合每日收益
    portfolio_daily_returns_ms = []  # 最大化夏普比率组合每日收益
    portfolio_daily_returns_ms_min5 = []  # 最大化夏普比率组合（每类资产>=5%）每日收益
    portfolio_daily_returns_ms_unconstrained = []  # 最大化夏普比率组合（允许做空）每日收益
    portfolio_daily_returns_fixed = []  # 固定比例组合每日收益

    print("开始回测，预计时间较长，请耐心等待...")
    for i in range(data.index.get_loc(pd.to_datetime(start_date)), len(data)):
        current_date = data.index[i]
        if current_date.month != data.index[i - 1].month:
            # 每月重新平衡
            print(f"调仓日: {current_date.strftime('%Y-%m-%d')}，开始优化权重...")
            look_back_data = data.iloc[data.index < current_date]
            look_back_returns = calculate_daily_returns(look_back_data)
            cov_matrix = look_back_returns.cov() * 252

            # 风险平价组合优化
            weights_rp = risk_parity_optimization(cov_matrix)
            # 最大化夏普比率组合优化
            weights_ms = maximize_sharpe_optimization(look_back_returns, risk_free_rate)
            # 最大化夏普比率组合优化（每类资产>=5%）
            weights_ms_min5 = maximize_sharpe_optimization_min5(look_back_returns, risk_free_rate)
            # 最大化夏普比率组合优化（允许做空）
            weights_ms_unconstrained = maximize_sharpe_optimization_unconstrained(look_back_returns, risk_free_rate)

            rebalance_dates.append(current_date)
            weight_history_rp.append(weights_rp)
            weight_history_ms.append(weights_ms)
            weight_history_ms_min5.append(weights_ms_min5)
            weight_history_ms_unconstrained.append(weights_ms_unconstrained)
            print(f"调仓日: {current_date.strftime('%Y-%m-%d')}，权重优化完成。")

        # 计算每日收益
        portfolio_return_rp = np.dot(weights_rp, daily_returns.loc[current_date])
        portfolio_return_ms = np.dot(weights_ms, daily_returns.loc[current_date])
        portfolio_return_ms_min5 = np.dot(weights_ms_min5, daily_returns.loc[current_date])
        portfolio_return_ms_unconstrained = np.dot(weights_ms_unconstrained, daily_returns.loc[current_date])
        portfolio_return_fixed = np.dot(weights_fixed, daily_returns.loc[current_date])

        # 更新组合净值
        k = portfolio_value_rp.index.get_loc(current_date)
        portfolio_value_rp.iloc[k] = portfolio_value_rp.iloc[k - 1] * (1 + portfolio_return_rp)
        portfolio_value_ms.iloc[k] = portfolio_value_ms.iloc[k - 1] * (1 + portfolio_return_ms)
        portfolio_value_ms_min5.iloc[k] = portfolio_value_ms_min5.iloc[k - 1] * (1 + portfolio_return_ms_min5)
        portfolio_value_ms_unconstrained.iloc[k] = portfolio_value_ms_unconstrained.iloc[k - 1] * (1 + portfolio_return_ms_unconstrained)
        portfolio_value_fixed.iloc[k] = portfolio_value_fixed.iloc[k - 1] * (1 + portfolio_return_fixed)

        # 记录每日收益
        portfolio_daily_returns_rp.append([current_date, portfolio_return_rp])
        portfolio_daily_returns_ms.append([current_date, portfolio_return_ms])
        portfolio_daily_returns_ms_min5.append([current_date, portfolio_return_ms_min5])
        portfolio_daily_returns_ms_unconstrained.append([current_date, portfolio_return_ms_unconstrained])
        portfolio_daily_returns_fixed.append([current_date, portfolio_return_fixed])

        # 更新权重（非再平衡日）
        weights_rp = (daily_returns.loc[current_date] + 1) * weights_rp
        weights_rp = weights_rp / weights_rp.sum()
        weights_ms = (daily_returns.loc[current_date] + 1) * weights_ms
        weights_ms = weights_ms / weights_ms.sum()
        weights_ms_min5 = (daily_returns.loc[current_date] + 1) * weights_ms_min5
        weights_ms_min5 = weights_ms_min5 / weights_ms_min5.sum()
        weights_ms_unconstrained = (daily_returns.loc[current_date] + 1) * weights_ms_unconstrained
        weights_ms_unconstrained = weights_ms_unconstrained / weights_ms_unconstrained.sum()

    print("回测完成，开始生成结果...")

    # 将权重历史转换为 DataFrame
    weight_history_rp = pd.DataFrame(weight_history_rp, index=rebalance_dates, columns=tickers)
    weight_history_rp.index = pd.to_datetime(weight_history_rp.index)
    weight_history_ms = pd.DataFrame(weight_history_ms, index=rebalance_dates, columns=tickers)
    weight_history_ms.index = pd.to_datetime(weight_history_ms.index)
    weight_history_ms_min5 = pd.DataFrame(weight_history_ms_min5, index=rebalance_dates, columns=tickers)
    weight_history_ms_min5.index = pd.to_datetime(weight_history_ms_min5.index)
    weight_history_ms_unconstrained = pd.DataFrame(weight_history_ms_unconstrained, index=rebalance_dates, columns=tickers)
    weight_history_ms_unconstrained.index = pd.to_datetime(weight_history_ms_unconstrained.index)

    # 将每日收益转换为 DataFrame
    df_portfolio_daily_returns_rp = pd.DataFrame(portfolio_daily_returns_rp, columns=['date', 'dailyReturn'])
    df_portfolio_daily_returns_ms = pd.DataFrame(portfolio_daily_returns_ms, columns=['date', 'dailyReturn'])
    df_portfolio_daily_returns_ms_min5 = pd.DataFrame(portfolio_daily_returns_ms_min5, columns=['date', 'dailyReturn'])
    df_portfolio_daily_returns_ms_unconstrained = pd.DataFrame(portfolio_daily_returns_ms_unconstrained, columns=['date', 'dailyReturn'])
    df_portfolio_daily_returns_fixed = pd.DataFrame(portfolio_daily_returns_fixed, columns=['date', 'dailyReturn'])

    print("结果生成完成。")
    return (
        portfolio_value_rp, portfolio_value_ms, portfolio_value_ms_min5, portfolio_value_ms_unconstrained, portfolio_value_fixed,  # 五种策略的组合净值
        data.loc[portfolio_value_rp.index[0]:],  # 资产净值
        weight_history_rp, weight_history_ms, weight_history_ms_min5, weight_history_ms_unconstrained,  # 四种策略的权重历史
        df_portfolio_daily_returns_rp, df_portfolio_daily_returns_ms, df_portfolio_daily_returns_ms_min5, df_portfolio_daily_returns_ms_unconstrained, df_portfolio_daily_returns_fixed  # 五种策略的每日收益
    )

# 计算风险指标
def calculate_risk_metrics(portfolio_value, daily_returns, risk_free_rate=0.0):
    annualized_return = calculate_annualized_return(portfolio_value)
    total_return = calculate_total_return(portfolio_value)
    annualized_volatility = calculate_annualized_volatility(daily_returns)
    sharpe_ratio = calculate_sharpe_ratio(daily_returns, risk_free_rate)
    max_drawdown = calculate_max_drawdown(portfolio_value)
    monthly_win_rate = calculate_monthly_win_rate(portfolio_value)
    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Monthly Win Rate': monthly_win_rate
    }

# 计算年度收益率
def calculate_annual_returns(data):
    annual_returns = data.resample('Y').last().pct_change().dropna()
    annual_returns.index = annual_returns.index.year
    return annual_returns

# 创建 Dash 应用
def create_dash_app(portfolio_value_rp, portfolio_value_ms, portfolio_value_ms_min5, portfolio_value_ms_unconstrained, portfolio_value_fixed, asset_values, weight_history_rp, weight_history_ms, weight_history_ms_min5, weight_history_ms_unconstrained,
                    portfolio_metrics_rp, portfolio_metrics_ms, portfolio_metrics_ms_min5, portfolio_metrics_ms_unconstrained, portfolio_metrics_fixed, assets_metrics, portfolio_annual_returns_rp,
                    portfolio_annual_returns_ms, portfolio_annual_returns_ms_min5, portfolio_annual_returns_ms_unconstrained, portfolio_annual_returns_fixed, assets_annual_returns):
    app = dash.Dash(__name__)

    # 定义颜色映射
    colors = {
        'IWB': 'blue', 'IWM': 'green', 'IEF': 'orange', 'LQD': 'cyan', 'GLD': 'purple', 'USO': 'brown', 'IYR': 'gray', 'VNQ': 'pink',
        'Risk Parity Portfolio': 'darkblue',  # 风险平价组合
        'Max Sharpe Portfolio': 'red',  # 最大化夏普比率组合
        'Max Sharpe Portfolio (Min 5%)': 'darkred',  # 最大化夏普比率组合（每类资产>=5%）
        'Max Sharpe Portfolio (Unconstrained)': 'orange',  # 最大化夏普比率组合（允许做空）
        'Fixed Weight Portfolio': 'green',  # 固定比例组合
        'Equal Weighted Portfolio': 'gray'  # 等权重组合
    }

    # ETF 中文名映射
    ticker_names = {
        'IWB': '罗素1000指数 (IWB)',
        'IWM': '罗素2000指数 (IWM)',
        'IEF': '7-10年期国债 (IEF)',
        'LQD': '投资级公司债 (LQD)',
        'GLD': '黄金 (GLD)',
        'USO': '原油 (USO)',
        'IYR': '房地产 (IYR)',
        'VNQ': '房地产信托 (VNQ)',
        'Risk Parity Portfolio': '风险平价组合',
        'Max Sharpe Portfolio': '最大化夏普比率组合',
        'Max Sharpe Portfolio (Min 5%)': '最大化夏普比率组合（每类资产>=5%）',
        'Max Sharpe Portfolio (Unconstrained)': '最大化夏普比率组合（允许做空）',
        'Fixed Weight Portfolio': '固定比例组合（6:4:1:1）',
        'Equal Weighted Portfolio': '等权重组合'
    }

    # 净值曲线图
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=portfolio_value_rp.index, y=portfolio_value_rp, mode='lines',
                              name=ticker_names['Risk Parity Portfolio'], line=dict(color=colors['Risk Parity Portfolio'], width=3)))
    fig1.add_trace(go.Scatter(x=portfolio_value_ms.index, y=portfolio_value_ms, mode='lines',
                              name=ticker_names['Max Sharpe Portfolio'], line=dict(color=colors['Max Sharpe Portfolio'], width=3)))
    fig1.add_trace(go.Scatter(x=portfolio_value_ms_min5.index, y=portfolio_value_ms_min5, mode='lines',
                              name=ticker_names['Max Sharpe Portfolio (Min 5%)'], line=dict(color=colors['Max Sharpe Portfolio (Min 5%)'], width=3)))
    fig1.add_trace(go.Scatter(x=portfolio_value_ms_unconstrained.index, y=portfolio_value_ms_unconstrained, mode='lines',
                              name=ticker_names['Max Sharpe Portfolio (Unconstrained)'], line=dict(color=colors['Max Sharpe Portfolio (Unconstrained)'], width=3)))
    fig1.add_trace(go.Scatter(x=portfolio_value_fixed.index, y=portfolio_value_fixed, mode='lines',
                              name=ticker_names['Fixed Weight Portfolio'], line=dict(color=colors['Fixed Weight Portfolio'], width=3)))
    fig1.add_trace(go.Scatter(x=asset_values.index, y=asset_values['Equal Weighted Portfolio'] / asset_values['Equal Weighted Portfolio'].iloc[0],
                              mode='lines', name=ticker_names['Equal Weighted Portfolio'], line=dict(color=colors['Equal Weighted Portfolio'], width=3)))
    for ticker in asset_values.columns:
        if ticker != 'Equal Weighted Portfolio':
            fig1.add_trace(go.Scatter(x=asset_values.index, y=asset_values[ticker] / asset_values[ticker].iloc[0],
                                      mode='lines', name=ticker_names[ticker], line=dict(color=colors[ticker], width=1)))
    fig1.update_layout(
        title='1. 净值曲线',
        xaxis_title='日期',
        yaxis_title='归一化净值',
        margin=dict(l=50, r=50, b=50, t=50, pad=10),
        yaxis=dict(automargin=True, fixedrange=False),
        title_font=dict(size=18, color='black', family='Arial Black')
    )

    # 回撤图
    drawdown_rp = calculate_drawdown(portfolio_value_rp)
    drawdown_ms = calculate_drawdown(portfolio_value_ms)
    drawdown_ms_min5 = calculate_drawdown(portfolio_value_ms_min5)
    drawdown_ms_unconstrained = calculate_drawdown(portfolio_value_ms_unconstrained)
    drawdown_fixed = calculate_drawdown(portfolio_value_fixed)
    drawdown_equal_weighted = calculate_drawdown(asset_values['Equal Weighted Portfolio'])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=drawdown_rp.index, y=drawdown_rp, mode='lines',
                              name=ticker_names['Risk Parity Portfolio'], line=dict(color=colors['Risk Parity Portfolio'], width=2)))
    fig2.add_trace(go.Scatter(x=drawdown_ms.index, y=drawdown_ms, mode='lines',
                              name=ticker_names['Max Sharpe Portfolio'], line=dict(color=colors['Max Sharpe Portfolio'], width=2)))
    fig2.add_trace(go.Scatter(x=drawdown_ms_min5.index, y=drawdown_ms_min5, mode='lines',
                              name=ticker_names['Max Sharpe Portfolio (Min 5%)'], line=dict(color=colors['Max Sharpe Portfolio (Min 5%)'], width=2)))
    fig2.add_trace(go.Scatter(x=drawdown_ms_unconstrained.index, y=drawdown_ms_unconstrained, mode='lines',
                              name=ticker_names['Max Sharpe Portfolio (Unconstrained)'], line=dict(color=colors['Max Sharpe Portfolio (Unconstrained)'], width=2)))
    fig2.add_trace(go.Scatter(x=drawdown_fixed.index, y=drawdown_fixed, mode='lines',
                              name=ticker_names['Fixed Weight Portfolio'], line=dict(color=colors['Fixed Weight Portfolio'], width=2)))
    fig2.add_trace(go.Scatter(x=drawdown_equal_weighted.index, y=drawdown_equal_weighted, mode='lines',
                              name=ticker_names['Equal Weighted Portfolio'], line=dict(color=colors['Equal Weighted Portfolio'], width=2)))
    fig2.update_layout(
        title='2. 回撤图',
        xaxis_title='日期',
        yaxis_title='回撤',
        margin=dict(l=50, r=50, b=50, t=50, pad=10),
        yaxis=dict(automargin=True, fixedrange=False),
        title_font=dict(size=18, color='black', family='Arial Black')
    )

    # 年度收益率直方图
    fig3 = go.Figure()
    for ticker in asset_values.columns:
        fig3.add_trace(go.Bar(x=assets_annual_returns[ticker].index, y=assets_annual_returns[ticker],
                              name=ticker_names[ticker], marker_color=colors[ticker]))
    fig3.add_trace(go.Bar(x=portfolio_annual_returns_rp.index, y=portfolio_annual_returns_rp,
                          name=ticker_names['Risk Parity Portfolio'], marker_color=colors['Risk Parity Portfolio']))
    fig3.add_trace(go.Bar(x=portfolio_annual_returns_ms.index, y=portfolio_annual_returns_ms,
                          name=ticker_names['Max Sharpe Portfolio'], marker_color=colors['Max Sharpe Portfolio']))
    fig3.add_trace(go.Bar(x=portfolio_annual_returns_ms_min5.index, y=portfolio_annual_returns_ms_min5,
                          name=ticker_names['Max Sharpe Portfolio (Min 5%)'], marker_color=colors['Max Sharpe Portfolio (Min 5%)']))
    fig3.add_trace(go.Bar(x=portfolio_annual_returns_ms_unconstrained.index, y=portfolio_annual_returns_ms_unconstrained,
                          name=ticker_names['Max Sharpe Portfolio (Unconstrained)'], marker_color=colors['Max Sharpe Portfolio (Unconstrained)']))
    fig3.add_trace(go.Bar(x=portfolio_annual_returns_fixed.index, y=portfolio_annual_returns_fixed,
                          name=ticker_names['Fixed Weight Portfolio'], marker_color=colors['Fixed Weight Portfolio']))
    fig3.update_layout(
        title='3. 年度收益率直方图',
        xaxis_title='年份',
        yaxis_title='收益率',
        barmode='group',
        title_font=dict(size=18, color='black', family='Arial Black')
    )

    # 风险平价组合权重堆积图
    fig4 = go.Figure()
    for ticker in weight_history_rp.columns:
        fig4.add_trace(go.Bar(x=weight_history_rp.index, y=weight_history_rp[ticker],
                              name=ticker_names[ticker], marker_color=colors[ticker]))
    fig4.update_layout(
        title='4. 风险平价组合权重堆积图',
        xaxis_title='日期',
        yaxis_title='权重',
        barmode='stack',
        title_font=dict(size=18, color='black', family='Arial Black')
    )

    # 最大化夏普比率组合权重堆积图
    fig5 = go.Figure()
    for ticker in weight_history_ms.columns:
        fig5.add_trace(go.Bar(x=weight_history_ms.index, y=weight_history_ms[ticker],
                              name=ticker_names[ticker], marker_color=colors[ticker]))
    fig5.update_layout(
        title='5. 最大化夏普比率组合权重堆积图',
        xaxis_title='日期',
        yaxis_title='权重',
        barmode='stack',
        title_font=dict(size=18, color='black', family='Arial Black')
    )

    # 最大化夏普比率组合（每类资产>=5%）权重堆积图
    fig6 = go.Figure()
    for ticker in weight_history_ms_min5.columns:
        fig6.add_trace(go.Bar(x=weight_history_ms_min5.index, y=weight_history_ms_min5[ticker],
                              name=ticker_names[ticker], marker_color=colors[ticker]))
    fig6.update_layout(
        title='6. 最大化夏普比率组合（每类资产>=5%）权重堆积图',
        xaxis_title='日期',
        yaxis_title='权重',
        barmode='stack',
        title_font=dict(size=18, color='black', family='Arial Black')
    )

    # 风险指标表格
    metrics = ['Total Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Monthly Win Rate']
    metrics_table = []
    metrics_table.append(['指标'] + [ticker_names[ticker] for ticker in asset_values.columns] +
                         [ticker_names['Risk Parity Portfolio'], ticker_names['Max Sharpe Portfolio'], ticker_names['Max Sharpe Portfolio (Min 5%)'], ticker_names['Max Sharpe Portfolio (Unconstrained)'], ticker_names['Fixed Weight Portfolio']])
    for metric in metrics:
        row = [metric]
        for ticker in asset_values.columns:
            value = assets_metrics[ticker][metric]
            if metric != 'Sharpe Ratio':
                value = f"{value * 100:.2f}%"
            else:
                value = f"{value:.2f}"
            row.append(value)
        value_rp = portfolio_metrics_rp[metric]
        value_ms = portfolio_metrics_ms[metric]
        value_ms_min5 = portfolio_metrics_ms_min5[metric]
        value_ms_unconstrained = portfolio_metrics_ms_unconstrained[metric]
        value_fixed = portfolio_metrics_fixed[metric]
        if metric != 'Sharpe Ratio':
            value_rp = f"{value_rp * 100:.2f}%"
            value_ms = f"{value_ms * 100:.2f}%"
            value_ms_min5 = f"{value_ms_min5 * 100:.2f}%"
            value_ms_unconstrained = f"{value_ms_unconstrained * 100:.2f}%"
            value_fixed = f"{value_fixed * 100:.2f}%"
        else:
            value_rp = f"{value_rp:.2f}"
            value_ms = f"{value_ms:.2f}"
            value_ms_min5 = f"{value_ms_min5:.2f}"
            value_ms_unconstrained = f"{value_ms_unconstrained:.2f}"
            value_fixed = f"{value_fixed:.2f}"
        row.append(value_rp)
        row.append(value_ms)
        row.append(value_ms_min5)
        row.append(value_ms_unconstrained)
        row.append(value_fixed)
        metrics_table.append(row)

    # Dash 布局
    app.layout = html.Div([
        html.H1("组合策略分析", style={'text-align': 'center'}),
        html.Div([dcc.Graph(figure=fig1, style={'height': '600px'})]),  # 净值曲线图
        html.Div([dcc.Graph(figure=fig2, style={'height': '600px'})]),  # 回撤图
        html.Div([dcc.Graph(figure=fig3, style={'height': '600px'})]),  # 年度收益率直方图
        html.Div([dcc.Graph(figure=fig4, style={'height': '600px'})]),  # 风险平价组合权重堆积图
        html.Div([dcc.Graph(figure=fig5, style={'height': '600px'})]),  # 最大化夏普比率组合权重堆积图
        html.Div([dcc.Graph(figure=fig6, style={'height': '600px'})]),  # 最大化夏普比率组合（每类资产>=5%）权重堆积图
        html.H3("风险指标", style={'text-align': 'center'}),
        html.Div(
            style={'overflowX': 'auto', 'fontSize': '12px'},
            children=[
                html.Table(
                    [html.Tr([html.Th(col, style={
                        'border': '1px solid black',
                        'padding': '8px',
                        'background-color': '#f2f2f2',
                        'min-width': '100px',
                        'white-space': 'normal',
                        'word-wrap': 'break-word',
                    }) for col in metrics_table[0]])] +
                    [html.Tr([html.Td(cell, style={
                        'border': '1px solid black',
                        'padding': '8px',
                        'min-width': '100px',
                        'background-color': '#f2f2f2' if i == len(metrics_table[0]) - 6 or i == len(metrics_table[0]) - 5 or i == len(metrics_table[0]) - 4 or i == len(metrics_table[0]) - 3 or i == len(metrics_table[0]) - 2 or i == len(metrics_table[0]) - 1 else 'white'
                    }) for i, cell in enumerate(row)]) for row in metrics_table[1:]],
                    style={
                        'border': '1px solid black',
                        'border-collapse': 'collapse',
                        'width': '100%',
                        'margin-top': '20px',
                        'table-layout': 'fixed',
                    }
                )
            ]
        )
    ])
    return app

# 主函数
def main():
    start_time = time.time()  # 记录程序开始时间
    tickers = ['IWB', 'IWM', 'IEF', 'LQD', 'GLD', 'USO', 'IYR', 'VNQ']
    start_date = '2007-01-03'
    end_date = '2024-08-30'
    risk_free_rate = 0.02

    print("开始回测策略...")
    portfolio_value_rp, portfolio_value_ms, portfolio_value_ms_min5, portfolio_value_ms_unconstrained, portfolio_value_fixed, asset_values, weight_history_rp, weight_history_ms, weight_history_ms_min5, weight_history_ms_unconstrained, df_portfolio_daily_returns_rp, df_portfolio_daily_returns_ms, df_portfolio_daily_returns_ms_min5, df_portfolio_daily_returns_ms_unconstrained, df_portfolio_daily_returns_fixed = backtest_strategies(tickers, start_date, end_date, look_back=6, risk_free_rate=risk_free_rate)
    asset_values['Equal Weighted Portfolio'] = asset_values.mean(axis=1)

    print("计算风险指标...")
    portfolio_daily_returns_rp = calculate_daily_returns(portfolio_value_rp)
    portfolio_daily_returns_ms = calculate_daily_returns(portfolio_value_ms)
    portfolio_daily_returns_ms_min5 = calculate_daily_returns(portfolio_value_ms_min5)
    portfolio_daily_returns_ms_unconstrained = calculate_daily_returns(portfolio_value_ms_unconstrained)
    portfolio_daily_returns_fixed = calculate_daily_returns(portfolio_value_fixed)
    portfolio_metrics_rp = calculate_risk_metrics(portfolio_value_rp, portfolio_daily_returns_rp, risk_free_rate)
    portfolio_metrics_ms = calculate_risk_metrics(portfolio_value_ms, portfolio_daily_returns_ms, risk_free_rate)
    portfolio_metrics_ms_min5 = calculate_risk_metrics(portfolio_value_ms_min5, portfolio_daily_returns_ms_min5, risk_free_rate)
    portfolio_metrics_ms_unconstrained = calculate_risk_metrics(portfolio_value_ms_unconstrained, portfolio_daily_returns_ms_unconstrained, risk_free_rate)
    portfolio_metrics_fixed = calculate_risk_metrics(portfolio_value_fixed, portfolio_daily_returns_fixed, risk_free_rate)
    assets_metrics = {}
    for ticker in asset_values.columns:
        asset_daily_returns = calculate_daily_returns(asset_values[ticker])
        assets_metrics[ticker] = calculate_risk_metrics(asset_values[ticker], asset_daily_returns, risk_free_rate)

    print("计算年度收益率...")
    portfolio_annual_returns_rp = calculate_annual_returns(portfolio_value_rp)
    portfolio_annual_returns_ms = calculate_annual_returns(portfolio_value_ms)
    portfolio_annual_returns_ms_min5 = calculate_annual_returns(portfolio_value_ms_min5)
    portfolio_annual_returns_ms_unconstrained = calculate_annual_returns(portfolio_value_ms_unconstrained)
    portfolio_annual_returns_fixed = calculate_annual_returns(portfolio_value_fixed)
    assets_annual_returns = {ticker: calculate_annual_returns(asset_values[ticker]) for ticker in asset_values.columns}

    print("创建 Dash 应用...")
    app = create_dash_app(portfolio_value_rp, portfolio_value_ms, portfolio_value_ms_min5, portfolio_value_ms_unconstrained, portfolio_value_fixed, asset_values, weight_history_rp, weight_history_ms, weight_history_ms_min5, weight_history_ms_unconstrained,
                          portfolio_metrics_rp, portfolio_metrics_ms, portfolio_metrics_ms_min5, portfolio_metrics_ms_unconstrained, portfolio_metrics_fixed, assets_metrics, portfolio_annual_returns_rp,
                          portfolio_annual_returns_ms, portfolio_annual_returns_ms_min5, portfolio_annual_returns_ms_unconstrained, portfolio_annual_returns_fixed, assets_annual_returns)
    print("Dash 应用创建完成，启动服务器...")
    print("程序运行完成，总运行时间: {:.2f}秒".format(time.time() - start_time))  # 打印总运行时间
    app.run_server(debug=True, use_reloader=False)

    

if __name__ == "__main__":
    main()