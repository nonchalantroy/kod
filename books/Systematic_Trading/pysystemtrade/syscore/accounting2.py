from copy import copy, deepcopy
import pandas as pd
import scipy.stats as stats, random, numpy as np
from syscore.algos import robust_vol_calc
from syscore.dateutils import BUSINESS_DAYS_IN_YEAR, ROOT_BDAYS_INYEAR, WEEKS_IN_YEAR, ROOT_WEEKS_IN_YEAR
from syscore.dateutils import MONTHS_IN_YEAR, ROOT_MONTHS_IN_YEAR

DEFAULT_CAPITAL = 10000000.0
DEFAULT_ANN_RISK_TARGET = 0.16
DEFAULT_DAILY_CAPITAL=DEFAULT_CAPITAL * DEFAULT_ANN_RISK_TARGET / ROOT_BDAYS_INYEAR

def sharpe(price, forecast):        
    base_capital = DEFAULT_CAPITAL
    daily_risk_capital = DEFAULT_CAPITAL * DEFAULT_ANN_RISK_TARGET / ROOT_BDAYS_INYEAR        
    ts_capital=pd.Series([DEFAULT_CAPITAL]*len(price), index=price.index)        
    ann_risk = ts_capital * DEFAULT_ANN_RISK_TARGET
    get_daily_returns_volatility = robust_vol_calc(price.diff())
    multiplier = daily_risk_capital * 1.0 * 1.0 / 10.0
    denominator = get_daily_returns_volatility
    numerator = forecast *  multiplier
    positions = numerator.ffill() /  denominator.ffill()
    use_positions = positions.shift(1)
    cum_trades = use_positions.ffill()
    trades_to_use=cum_trades.diff()
    price_returns = price.diff()
    instr_ccy_returns = cum_trades.shift(1)* price_returns 
    instr_ccy_returns=instr_ccy_returns.cumsum().ffill().reindex(price.index).diff()
    base_ccy_returns = instr_ccy_returns 
    mean_return = base_ccy_returns.mean() * BUSINESS_DAYS_IN_YEAR
    vol = base_ccy_returns.std() * ROOT_BDAYS_INYEAR
    print (mean_return / vol)

