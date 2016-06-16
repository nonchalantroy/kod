"""
Suite of things to work out p&l, and statistics thereof

"""

from copy import copy, deepcopy

import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
from scipy.stats import skew, ttest_rel, ttest_1samp
import scipy.stats as stats
import random

from syscore.algos import robust_vol_calc
from syscore.pdutils import  drawdown
from syscore.dateutils import BUSINESS_DAYS_IN_YEAR, ROOT_BDAYS_INYEAR, WEEKS_IN_YEAR, ROOT_WEEKS_IN_YEAR
from syscore.dateutils import MONTHS_IN_YEAR, ROOT_MONTHS_IN_YEAR

"""
some defaults
"""
DEFAULT_CAPITAL = 10000000.0
DEFAULT_ANN_RISK_TARGET = 0.16
DEFAULT_DAILY_CAPITAL=DEFAULT_CAPITAL * DEFAULT_ANN_RISK_TARGET / ROOT_BDAYS_INYEAR


class accountCurveSingleElementOneFreq(pd.Series):
    def __init__(self, returns_df, capital, weighted_flag=False, frequency="D"):
        super().__init__(returns_df)
        
        returns_scalar=dict(D=BUSINESS_DAYS_IN_YEAR, W=WEEKS_IN_YEAR,
                            M=MONTHS_IN_YEAR, Y=1)[frequency]

        vol_scalar=dict(D=ROOT_BDAYS_INYEAR, W=ROOT_WEEKS_IN_YEAR,
                            M=ROOT_MONTHS_IN_YEAR, Y=1)[frequency]
                    
        setattr(self, "frequency", frequency)
        setattr(self, "_returns_scalar", returns_scalar)
        setattr(self, "_vol_scalar", vol_scalar)
        setattr(self, "_returns_df", returns_df)
        setattr(self, "weighted_flag", weighted_flag)
        setattr(self, "capital", capital)

    def as_df(self):
        print("Deprecated accountCurve.as_df use .as_ts() please")
        ## backward compatibility
        return self.as_ts()


    def as_ts(self):
        return pd.Series(self._returns_df)

    def percent(self):
        perc_returns=self.as_percent()
        new_curve=accountCurveSingleElementOneFreq(perc_returns, 100.0, self.weighted_flag, self.frequency)        
        return new_curve

    def as_percent(self):
        return 100.0 * self.as_ts() / self.capital

    def mean(self):
        return float(self.as_ts().mean())
    
    def std(self):
        return float(self.as_ts().std())

    def sharpe(self):
        mean_return = self.mean() * self._returns_scalar
        vol = self.std() * self._vol_scalar
        return mean_return / vol


class accountCurveSingleElement(accountCurveSingleElementOneFreq):
    
    def __init__(self, returns_df, capital, weighted_flag=False):
        ## We often want to use  
        daily_returns = returns_df.resample("1B", how="sum")        
        super().__init__(daily_returns, capital, frequency="D",  weighted_flag=weighted_flag)
        setattr(self, "daily", accountCurveSingleElementOneFreq(daily_returns, capital, frequency="D", weighted_flag=weighted_flag))

    def __repr__(self):
        return super().__repr__()+ "\n Use object.freq.method() to access periods (freq=daily, weekly, monthly, annual) default: daily"

class accountCurveSingle(accountCurveSingleElement):
    
    def __init__(self, gross_returns, net_returns, capital, weighted_flag=False):
        super().__init__(net_returns,  capital, weighted_flag=weighted_flag)
        
        setattr(self, "net", accountCurveSingleElement(net_returns, capital, weighted_flag=weighted_flag))
        setattr(self, "gross", accountCurveSingleElement(gross_returns, capital, weighted_flag=weighted_flag))

class accountCurve(accountCurveSingle):

    def __init__(self, price=None,   cash_costs=None, SR_cost=None, 
                 capital=None, ann_risk_target=None, pre_calc_data=None,
                 weighted_flag = False, weighting=None, 
                apply_weight_to_costs_only=False,
                 **kwargs):
        (base_capital, ann_risk, daily_risk_capital)=resolve_capital(price, capital, ann_risk_target)

        returns_data=pandl_with_data(price, daily_risk_capital=daily_risk_capital,  **kwargs)

        (cum_trades, trades_to_use, instr_ccy_returns,
            base_ccy_returns, use_fx, value_of_price_point)=returns_data
        
        self._calc_and_set_returns(base_ccy_returns, base_capital, 
                                    weighted_flag=weighted_flag, weighting=weighting,
                                    apply_weight_to_costs_only=apply_weight_to_costs_only)
        

    def _calc_and_set_returns(self, base_ccy_returns, base_capital, 
                              weighted_flag=False, weighting=None, 
                              apply_weight_to_costs_only=False):
        
        use_weighting = None
        
        super().__init__(base_ccy_returns, base_ccy_returns, base_capital, weighted_flag=weighted_flag)

    def __repr__(self):
        return super().__repr__()+ "\n Use object.calc_data() to see calculation details"        

def resolve_capital(ts_to_scale_to, capital=None, ann_risk_target=None):
    if capital is None:
        base_capital=copy(DEFAULT_CAPITAL)
    else:
        base_capital = copy(capital)
        
    if ann_risk_target is None:
        ann_risk_target=DEFAULT_ANN_RISK_TARGET
        
    ## might be a float or a Series, depending on capital
    daily_risk_capital = base_capital * ann_risk_target / ROOT_BDAYS_INYEAR

    if type(base_capital) is float or type(base_capital) is int:
        ts_capital=pd.Series([base_capital]*len(ts_to_scale_to), index=ts_to_scale_to.index)
        base_capital = float(base_capital)
    else:
        ts_capital=copy(base_capital)
    
    ## always a time series
    ann_risk = ts_capital * ann_risk_target
    
    return (base_capital, ann_risk, daily_risk_capital)



def pandl_with_data(price, trades=None, marktomarket=True, positions=None,
          delayfill=True, roundpositions=False,
          get_daily_returns_volatility=None, forecast=None, fx=None,
          daily_risk_capital=None, 
          value_of_price_point=1.0):
    
    use_fx = pd.Series([1.0] * len(price.index),index=price.index)

    prices_to_use = copy(price)
    positions = get_positions_from_forecasts(price,
                                             get_daily_returns_volatility,
                                             forecast,
                                             use_fx,
                                             value_of_price_point,
                                             daily_risk_capital)
    use_positions = copy(positions)

    use_positions = use_positions.shift(1)

    cum_trades = use_positions.ffill()
    trades_to_use=cum_trades.diff()
        
    price_returns = prices_to_use.diff()

    instr_ccy_returns = cum_trades.shift(1)* price_returns * value_of_price_point
    
    instr_ccy_returns=instr_ccy_returns.cumsum().ffill().reindex(price.index).diff()
    base_ccy_returns = instr_ccy_returns * use_fx
    
    return (cum_trades, trades_to_use, instr_ccy_returns,
            base_ccy_returns, use_fx, value_of_price_point)




def get_positions_from_forecasts(price, get_daily_returns_volatility, forecast,
                                 use_fx, value_of_price_point, daily_risk_capital,
                                  **kwargs):

    get_daily_returns_volatility = robust_vol_calc(price.diff(), **kwargs)
        
    multiplier = daily_risk_capital * 1.0 * 1.0 / 10.0

    denominator = (value_of_price_point * get_daily_returns_volatility* use_fx)

    numerator = forecast *  multiplier

    positions = numerator.ffill() /  denominator.ffill()

    return positions

