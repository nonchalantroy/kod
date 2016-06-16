from copy import copy, deepcopy
import pandas as pd
from pandas.tseries.offsets import BDay
from scipy.stats import skew, ttest_rel, ttest_1samp
import scipy.stats as stats, random, numpy as np
from syscore.algos import robust_vol_calc
from syscore.pdutils import  drawdown
from syscore.dateutils import BUSINESS_DAYS_IN_YEAR, ROOT_BDAYS_INYEAR, WEEKS_IN_YEAR, ROOT_WEEKS_IN_YEAR
from syscore.dateutils import MONTHS_IN_YEAR, ROOT_MONTHS_IN_YEAR

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

class accountCurve(accountCurveSingleElementOneFreq):

    def __init__(self, price=None, capital=None, ann_risk_target=None, **kwargs):
        
        base_capital = DEFAULT_CAPITAL
        daily_risk_capital = DEFAULT_CAPITAL * DEFAULT_ANN_RISK_TARGET / ROOT_BDAYS_INYEAR        
        ts_capital=pd.Series([DEFAULT_CAPITAL]*len(price), index=price.index)        
        ann_risk = ts_capital * DEFAULT_ANN_RISK_TARGET
        
        (cum_trades,
         trades_to_use,
         instr_ccy_returns,
         base_ccy_returns,
         value_of_price_point)=pandl_with_data(price,
                                               daily_risk_capital=daily_risk_capital,
                                               **kwargs)
        
        super().__init__(base_ccy_returns, base_capital, frequency="D")        

def pandl_with_data(price, trades=None, marktomarket=True, positions=None,
          delayfill=True, roundpositions=False,
          get_daily_returns_volatility=None, forecast=None, fx=None,
          daily_risk_capital=None, 
          value_of_price_point=1.0):
    
    get_daily_returns_volatility = robust_vol_calc(price.diff())
        
    multiplier = daily_risk_capital * 1.0 * 1.0 / 10.0

    denominator = (value_of_price_point * get_daily_returns_volatility)

    numerator = forecast *  multiplier

    positions = numerator.ffill() /  denominator.ffill()

    use_positions = positions.shift(1)

    cum_trades = use_positions.ffill()

    trades_to_use=cum_trades.diff()
        
    price_returns = price.diff()

    instr_ccy_returns = cum_trades.shift(1)* price_returns * value_of_price_point
    
    instr_ccy_returns=instr_ccy_returns.cumsum().ffill().reindex(price.index).diff()
    base_ccy_returns = instr_ccy_returns 
    
    return (cum_trades, trades_to_use, instr_ccy_returns,
            base_ccy_returns, value_of_price_point)

