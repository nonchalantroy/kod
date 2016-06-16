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

    
def percent(accurve):
    """
    Takes any account curve object
    
    Returns accountCurveSingleElementOneFreq - anything else is lost
    """
    pass

class accountCurveSingleElementOneFreq(pd.Series):
    """
    A single account curve for one asset (instrument / trading rule variation, ...)
     and one part of it (gross, net, costs)
     and for one frequency (daily, weekly, monthly...)
    
    Inherits from series

    We never init these directly but only as part of accountCurveSingleElement
    
    """
    def __init__(self, returns_df, capital, weighted_flag=False, frequency="D"):
        super().__init__(returns_df)
        
        try:
            returns_scalar=dict(D=BUSINESS_DAYS_IN_YEAR, W=WEEKS_IN_YEAR,
                                M=MONTHS_IN_YEAR, Y=1)[frequency]
            print ("returns_scalar=" + str(returns_scalar))
                                
            vol_scalar=dict(D=ROOT_BDAYS_INYEAR, W=ROOT_WEEKS_IN_YEAR,
                                M=ROOT_MONTHS_IN_YEAR, Y=1)[frequency]
            
        except KeyError:
            raise Exception("Not a frequency %s" % frequency)
        
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
        print ("in percent self.as_ts=" + str(self.as_ts().tail(4)))
        return 100.0 * self.as_ts() / self.capital


    def mean(self):
        return float(self.as_ts().mean())
    
    def std(self):
        return float(self.as_ts().std())

    def ann_mean(self):
        avg = self.mean()
        print ("self=" + str(self.tail(4)))
        return avg * self._returns_scalar

    def ann_std(self):
        period_std = self.std()

        return period_std * self._vol_scalar


    def sharpe(self):
        mean_return = self.ann_mean()
        vol = self.ann_std()
        try:
            sharpe=mean_return / vol
        except ZeroDivisionError:
            sharpe=np.nan
        return sharpe

    def __repr__(self):
        if self.weighted_flag:
            weight_comment="Weighted"
        else:
            weight_comment="Unweighted"
        return super().__repr__()+"\n %s account curve; use object.stats() to see methods" % weight_comment
    


class accountCurveSingleElement(accountCurveSingleElementOneFreq):
    
    def __init__(self, returns_df, capital, weighted_flag=False):
        ## We often want to use  
        daily_returns = returns_df.resample("1B", how="sum")
        print ("daily_returns=" + str(daily_returns.tail(4)))
        weekly_returns=returns_df.resample("W", how="sum")
        monthly_returns=returns_df.resample("MS", how="sum")
        annual_returns=returns_df.resample("A", how="sum")
        
        super().__init__(daily_returns, capital, frequency="D",  weighted_flag=weighted_flag)

        setattr(self, "daily", accountCurveSingleElementOneFreq(daily_returns, capital, frequency="D", weighted_flag=weighted_flag))
        setattr(self, "weekly", accountCurveSingleElementOneFreq(weekly_returns, capital, frequency="W",  weighted_flag=weighted_flag))
        setattr(self, "monthly", accountCurveSingleElementOneFreq(monthly_returns, capital, frequency="M", weighted_flag=weighted_flag))
        setattr(self, "annual", accountCurveSingleElementOneFreq(annual_returns, capital, frequency="Y",  weighted_flag=weighted_flag))

    def __repr__(self):
        return super().__repr__()+ "\n Use object.freq.method() to access periods (freq=daily, weekly, monthly, annual) default: daily"

class accountCurveSingle(accountCurveSingleElement):
    
    def __init__(self, gross_returns, net_returns, costs, capital, weighted_flag=False):
        super().__init__(net_returns,  capital, weighted_flag=weighted_flag)
        
        setattr(self, "net", accountCurveSingleElement(net_returns, capital, weighted_flag=weighted_flag))
        setattr(self, "gross", accountCurveSingleElement(gross_returns, capital, weighted_flag=weighted_flag))
        setattr(self, "costs", accountCurveSingleElement(costs,  capital, weighted_flag=weighted_flag))

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

        print ("cum_trades=" + str(cum_trades.tail(4)))
        print ("base_ccy_returns=" + str(base_ccy_returns.tail(4)))

        ## always returns a time series
        (costs_base_ccy, costs_instr_ccy)=calc_costs(returns_data, cash_costs, SR_cost, ann_risk)

        ## keep track of this
        unweighted_instr_ccy_pandl=dict(gross=instr_ccy_returns, costs=costs_instr_ccy, 
                             net=instr_ccy_returns+costs_instr_ccy)


        ## Initially we have an unweighted version
        
        self._calc_and_set_returns(base_ccy_returns, costs_base_ccy, base_capital, 
                                    weighted_flag=weighted_flag, weighting=weighting,
                                    apply_weight_to_costs_only=apply_weight_to_costs_only)
        
        ## Save all kinds of useful statistics

        setattr(self, "unweighted_instr_ccy_pandl", unweighted_instr_ccy_pandl)
        setattr(self, "cum_trades", cum_trades)
        setattr(self, "trades_to_use", trades_to_use)
        setattr(self, "capital", base_capital)
        setattr(self, "fx", use_fx)
        setattr(self, "value_of_price_point", value_of_price_point)


    def _calc_and_set_returns(self, base_ccy_returns, costs_base_ccy, base_capital, 
                              weighted_flag=False, weighting=None, 
                              apply_weight_to_costs_only=False):
        
        use_weighting = None

        net_base_returns=base_ccy_returns + costs_base_ccy ## costs are negative returns
        print ("net_base_returns=" + str(net_base_returns.tail(4)))
        
        super().__init__(base_ccy_returns, net_base_returns, costs_base_ccy, base_capital, weighted_flag=weighted_flag)
            
        ## save useful stats
        ## have to do this after super() call
        setattr(self, "weighted_flag", weighted_flag)
        setattr(self, "weighting", use_weighting)

    def __repr__(self):
        return super().__repr__()+ "\n Use object.calc_data() to see calculation details"

        

    def calc_data(self):
        """
        Returns detailed calculation information
        
        :returns: dictionary of float
        """
        calc_items=["cum_trades",  "trades_to_use",    "unweighted_instr_ccy_pandl",
                     "capital",  "weighting", "fx","value_of_price_point"]
        
        calc_dict=dict([(calc_name, getattr(self, calc_name)) for calc_name in calc_items])
        
        return calc_dict

        
def calc_costs(returns_data, cash_costs, SR_cost, ann_risk):
    (cum_trades, trades_to_use, instr_ccy_returns,
        base_ccy_returns, use_fx, value_of_price_point)=returns_data

    ## set costs to zero
    costs_instr_ccy=pd.Series([0.0]*len(use_fx), index=use_fx.index)

    ## fx is on master (price timestamp)
    ## costs_instr_ccy needs downsampling
    costs_instr_ccy=costs_instr_ccy.cumsum().ffill().reindex(use_fx.index).diff()
    
    costs_base_ccy=costs_instr_ccy *  use_fx.ffill()
    costs_base_ccy[np.isnan(costs_base_ccy)]=0.0

    return (costs_base_ccy, costs_instr_ccy)

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

def acc_list_to_pd_frame(list_of_ac_curves, asset_columns):
    list_of_df=[acc.as_ts() for acc in list_of_ac_curves]
    ans=pd.concat(list_of_df, axis=1,  join="outer")
    
    ans.columns=asset_columns
    ans=ans.cumsum().ffill().diff()
    
    return ans


