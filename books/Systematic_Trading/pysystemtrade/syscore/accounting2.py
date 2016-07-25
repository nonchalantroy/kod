import inspect
import pandas as pd, random, numpy as np
from copy import copy, deepcopy
from pandas.tseries.offsets import BDay
from scipy.stats import skew, ttest_rel, ttest_1samp
from syscore.algos import robust_vol_calc
from syscore.pdutils import  drawdown
from syscore.dateutils import BUSINESS_DAYS_IN_YEAR, ROOT_BDAYS_INYEAR, WEEKS_IN_YEAR, ROOT_WEEKS_IN_YEAR
from syscore.dateutils import MONTHS_IN_YEAR, ROOT_MONTHS_IN_YEAR
import scipy.stats as stats

DEFAULT_CAPITAL = 10000000.0
DEFAULT_ANN_RISK_TARGET = 0.16
DEFAULT_DAILY_CAPITAL=DEFAULT_CAPITAL * DEFAULT_ANN_RISK_TARGET / ROOT_BDAYS_INYEAR

def pandl_with_data(price,
                    trades=None,
                    marktomarket=True,
                    positions=None,
                    delayfill=True, roundpositions=False,
                    get_daily_returns_volatility=None, forecast=None, fx=None,
                    daily_risk_capital=None, 
                    value_of_price_point=1.0):
    
    use_fx = pd.Series([1.0] * len(price.index),index=price.index)
    get_daily_returns_volatility = robust_vol_calc(price.diff())
    multiplier = daily_risk_capital * 1.0 * 1.0 / 10.0
    denominator = (value_of_price_point * get_daily_returns_volatility* use_fx)
    numerator = forecast *  multiplier
    positions = numerator.ffill() /  denominator.ffill()
    cum_trades = positions.shift(1).ffill()
    trades_to_use=cum_trades.diff()        
    price_returns = price.diff()
    instr_ccy_returns = cum_trades.shift(1)* price_returns 
    instr_ccy_returns=instr_ccy_returns.cumsum().ffill().reindex(price.index).diff()
    base_ccy_returns = instr_ccy_returns * use_fx    
    return (cum_trades, trades_to_use, instr_ccy_returns,
            base_ccy_returns, use_fx, value_of_price_point)
    
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
        setattr(self, "capital", capital)

    def as_df(self):
        print("Deprecated accountCurve.as_df use .as_ts() please")
        return self.as_ts()

    def as_ts(self):
        return pd.Series(self._returns_df)

    def percent(self):        
        perc_returns=self.as_percent()
        new_curve=accountCurveSingleElementOneFreq(perc_returns, 100.0, self.weighted_flag, self.frequency)
        print("accounting percent" + str(new_curve.tail())) 
        return new_curve

    def cumulative(self):        
        cum_returns = self.as_cumulative()
        new_curve = accountCurveSingleElementOneFreq(cum_returns, self.capital, self.weighted_flag, self.frequency)        
        return new_curve

    def as_percent(self):
        return 100.0 * self.as_ts() / self.capital

    def skew(self):
        return skew(self.values[pd.isnull(self.values) == False])

class accountCurveSingleElement(accountCurveSingleElementOneFreq):
    def __init__(self, returns_df, capital, weighted_flag=False):
        daily_returns = returns_df.resample("1B", how="sum")
        super().__init__(daily_returns, capital, frequency="D",  weighted_flag=weighted_flag)

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

        
        base_capital = DEFAULT_CAPITAL
        daily_risk_capital = DEFAULT_CAPITAL * DEFAULT_ANN_RISK_TARGET / ROOT_BDAYS_INYEAR
        returns_data=pandl_with_data(price, daily_risk_capital=daily_risk_capital,  **kwargs)
        (cum_trades, trades_to_use, instr_ccy_returns,base_ccy_returns, use_fx, value_of_price_point)=returns_data
        self._calc_and_set_returns(base_ccy_returns,
                                   base_capital, 
                                    weighted_flag=weighted_flag,
                                   weighting=weighting)
        
        setattr(self, "cum_trades", cum_trades)
        setattr(self, "trades_to_use", trades_to_use)
        setattr(self, "capital", base_capital)
        setattr(self, "fx", use_fx)
        setattr(self, "value_of_price_point", value_of_price_point)

    def _calc_and_set_returns(self, base_ccy_returns,  base_capital, 
                              weighted_flag=False, weighting=None):
        use_weighting = None
        net_base_returns=base_ccy_returns         
        super().__init__(base_ccy_returns, net_base_returns, base_capital, weighted_flag=weighted_flag)  
        setattr(self, "weighted_flag", weighted_flag)
        setattr(self, "weighting", use_weighting)

