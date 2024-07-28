import pandas as pd
import numpy as np

def max_drawdown(returns):
    """
    Calculate the maximum drawdown of a returns series.
    """
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    return max_drawdown

def annual_sharpe(returns, N=365):
    if isinstance(returns, pd.Series) and not returns.empty:
        returns = returns.resample('D').sum()
        return np.sqrt(N) * returns.mean() / returns.std()
    elif isinstance(returns, np.ndarray):
        return np.sqrt(N) * returns.mean() / returns.std()
    else:
        return None

def annual_returns(returns, N=365):
    returns = returns.resample('D').sum()
    return np.mean(returns) * N

def calmar_ratio(returns, risk_free_rate=0.0, N=365):
    """
    Calculate the Calmar ratio of a returns series.
    """
    annual_return = annual_returns(returns)
    max_dd = max_drawdown(returns)
    calmar = (annual_return - risk_free_rate) / abs(max_dd)
    return calmar

def sortino_ratio(returns, risk_free_rate=0.0, N=365):
    """
    Calculate the Sortino ratio of a returns series.
    """
    returns = returns.resample('D').sum()
    downside_risk = np.sqrt(np.mean(np.minimum(0, returns) ** 2)) * np.sqrt(N)
    annual_return = np.mean(returns) * N  # Assuming daily returns
    sortino = (annual_return - risk_free_rate) / downside_risk
    return sortino

def calculate_metrics(returns):
    metrics = {
        'sharpe': annual_sharpe(returns),
        'max_drawdown': max_drawdown(returns),
        'calmar_ratio': calmar_ratio(returns),
        'sortino_ratio': sortino_ratio(returns),
        'annual_returns': annual_returns(returns),
    }
    return metrics
