import inspect
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

    print ("price=" + str(price.tail()))
       
    use_fx = pd.Series([1.0] * len(price.index),
                       index=price.index)
    prices_to_use = copy(price)
    get_daily_returns_volatility = robust_vol_calc(price.diff())
    multiplier = daily_risk_capital * 1.0 * 1.0 / 10.0
    denominator = (value_of_price_point * get_daily_returns_volatility* use_fx)
    numerator = forecast *  multiplier
    positions = numerator.ffill() /  denominator.ffill()
    
    use_positions = copy(positions)
    use_positions = use_positions.shift(1)                                
    cum_trades = use_positions.ffill()
    trades_to_use=cum_trades.diff()       
    cum_trades = trades_to_use.cumsum().ffill()

    price_returns = prices_to_use.diff()
    instr_ccy_returns = cum_trades.shift(1)* price_returns * value_of_price_point    
    instr_ccy_returns=instr_ccy_returns.cumsum().ffill().reindex(price.index).diff()
    base_ccy_returns = instr_ccy_returns * use_fx
    
    print (skew(base_ccy_returns.dropna()))
    exit()
    
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
   
class accountCurveSingleElementOneFreq(pd.Series):
    def __init__(self, returns_df, capital, weighted_flag=False, frequency="D"):
        super().__init__(returns_df)
        
        try:
            returns_scalar=dict(D=BUSINESS_DAYS_IN_YEAR, W=WEEKS_IN_YEAR,
                                M=MONTHS_IN_YEAR, Y=1)[frequency]
                                
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

    def cumulative(self):
        
        cum_returns = self.as_cumulative()
        new_curve = accountCurveSingleElementOneFreq(cum_returns, self.capital, self.weighted_flag, self.frequency)
        
        return new_curve

    def as_percent(self):
        print ("capital " + str(self.capital))
        print ("as ts " + str(self.as_ts().tail()))
        return 100.0 * self.as_ts() / self.capital
    
    def curve(self):
        # we cache this since it's used so much
        if hasattr(self, "_curve"):
            return self._curve
        else:
            curve = self.cumsum().ffill()
            setattr(self, "_curve", curve)
            return curve

    def mean(self):
        return float(self.as_ts().mean())
    
    def std(self):
        return float(self.as_ts().std())

    def ann_mean(self):
        avg = self.mean()

        return avg * self._returns_scalar

    def ann_std(self):
        period_std = self.std()

        return period_std * self._vol_scalar

    def vals(self):
        x = [z for z in self.values if not np.isnan(z)]
        return x

    def skew(self):
        return skew(self.vals())



class accountCurveSingleElement(accountCurveSingleElementOneFreq):
    """
    A single account curve for one asset (instrument / trading rule variation, ...)
     and one part of it (gross, net, costs)
    
    Inherits from data frame

    We never init these directly but only as part of accountCurveSingle
    
    """
    
    def __init__(self, returns_df, capital, weighted_flag=False):
        """
        :param returns_df: series of returns
        :type returns_df: Tx1 pd.Series

        :param weighted_flag: Is this account curve of weighted returns?
        :type weighted_flag: bool


        """
        ## We often want to use  
        daily_returns = returns_df.resample("1B", how="sum")
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
    """
    A single account curve for one asset (instrument / trading rule variation, ...)
    
    Inherits from data frame
    
    On the surface we see the 'net' but there's also a gross and cost part included
    
    """
    def __init__(self, gross_returns, net_returns, costs, capital, weighted_flag=False):
        """
        :param gross_returns: series of returns, no costs applied
        :type gross_returns: Tx1 pd.Series

        :param costs: series of costs (minus is a cost)
        :type costs: Tx1 pd.Series

        :param net_returns: series of costs (minus is a cost)
        :type net_returns: Tx1 pd.Series

        :param weighted_flag: Is this account curve of weighted returns?
        :type weighted_flag: bool

        :param capital: capital
        :type capital: Tx1 pd.Series of float

        
        """
        
        super().__init__(net_returns,  capital, weighted_flag=weighted_flag)
        
        setattr(self, "net", accountCurveSingleElement(net_returns, capital, weighted_flag=weighted_flag))
        setattr(self, "gross", accountCurveSingleElement(gross_returns, capital, weighted_flag=weighted_flag))
        setattr(self, "costs", accountCurveSingleElement(costs,  capital, weighted_flag=weighted_flag))

    def __repr__(self):
        return super().__repr__()+"\n Use object.curve_type.freq.method() (freq=net, gross, costs) default: net"
                
    def to_ncg_frame(self):
        """
        View net gross and costs together
        
        :returns: Tx3 pd.DataFrame
        """
        
        ans=pd.concat([self.net.as_ts(), self.gross.as_ts(), self.costs.as_ts()], axis=1)
        ans.columns=["net", "gross", "costs"]
        
        return ans


class accountCurve(accountCurveSingle):

    def __init__(self, price=None,   cash_costs=None, SR_cost=None, 
                 capital=None, ann_risk_target=None, pre_calc_data=None,
                 weighted_flag = False, weighting=None, 
                apply_weight_to_costs_only=False,
                 **kwargs):
        (base_capital, ann_risk, daily_risk_capital)=resolve_capital(price, capital, ann_risk_target)

        returns_data=pandl_with_data(price, daily_risk_capital=daily_risk_capital,  **kwargs)
    
        (cum_trades, trades_to_use, instr_ccy_returns,base_ccy_returns, use_fx, value_of_price_point)=returns_data
                
        ## always returns a time series
        (costs_base_ccy, costs_instr_ccy)=calc_costs(returns_data, cash_costs, SR_cost, ann_risk)
            
        ## keep track of this
        unweighted_instr_ccy_pandl=dict(gross=instr_ccy_returns, costs=costs_instr_ccy, 
                                        net=instr_ccy_returns+costs_instr_ccy)

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
        """
        This hidden method is called when we setup the account curve, to 
        
        Also called again if we get a weighted version of this account curve

        :param base_ccy_returns: Pre-cost returns in base currency terms (unweighted)
        :type base_ccy_returns: Tx1 pd.Series

        :param costs_base_ccy: Costs in base currency terms, aligned to base_ccy_returns (unweighted)
        :type costs_base_ccy: Tx1 pd.Series

        :param base_capital: base_capital in base currency terms
        :type base_capital: Tx1 pd.Series (aligned to base_ccy_returns) or float
        
        :param weighted_flag: Apply a weighting scheme, or not
        :type weighted_flag: bool

        :param weighting: Weighting scheme to apply to returns, MAY NOT BE aligned to base_ccy_returns
        :type weighting: Tx1 pd.Series

        :param apply_weight_to_costs_only: Apply weights only to costs, not gross returns
        :type apply_weight_to_costs_only: bool

        """

        
        if weighted_flag:
            use_weighting = weighting.reindex(base_ccy_returns.index).ffill()
            if not apply_weight_to_costs_only:
                ## only apply to gross returns if they aren't already weighted
                base_ccy_returns = base_ccy_returns* use_weighting
            
            ## Always apply to costs
            costs_base_ccy = costs_base_ccy* use_weighting
        else:
            use_weighting = None

        net_base_returns=base_ccy_returns + costs_base_ccy ## costs are negative returns
        
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





def weighted(account_curve,  weighting, apply_weight_to_costs_only=False, allow_reweighting=False):
        """
        Creates a copy of account curve with weights applied 

        :param account_curve: Curve to copy from
        :type account_curve: accountCurve
        
        :param weighting: Weighting scheme to apply to returns 
        :type weighting: Tx1 pd.Series

        :param apply_weight_to_costs_only: Apply weights only to costs, not gross returns
        :type apply_weight_to_costs_only: bool

        :param allow_reweighting: Apply weights only to costs, not gross returns
        :type allow_reweighting: bool

        :returns: accountCurve
        
        """
        if account_curve.weighted_flag:
            if allow_reweighting:
                pass
            else:
                raise Exception("You can't reweight weighted returns!")

        ## very clunky but I can't make copy, deepcopy or inheritance work for this use case...
        base_capital=copy(account_curve.capital)
        gross_returns=copy(account_curve.gross.as_ts())
        costs_base_ccy=copy(account_curve.costs.as_ts())
        unweighted_instr_ccy_pandl=copy(account_curve.unweighted_instr_ccy_pandl)

        returns_data=(account_curve.cum_trades, account_curve.trades_to_use, 
                      unweighted_instr_ccy_pandl["gross"],
                gross_returns, account_curve.fx, account_curve.value_of_price_point)

        pre_calc_data=(returns_data, base_capital, costs_base_ccy, unweighted_instr_ccy_pandl)
        
        ## Create a cloned account curve with the pre calculated data
        ## recalculate the returns with weighting applied
        new_account_curve=accountCurve(pre_calc_data=pre_calc_data,
                                        weighted_flag=True,
                                       weighting=weighting, 
                                       apply_weight_to_costs_only=apply_weight_to_costs_only)
        
        
        return new_account_curve

        
def calc_costs(returns_data, cash_costs, SR_cost, ann_risk):
    """
    Calculate costs
    
    :param returns_data: returns data
    :type returns_data: 5 tuple returned by pandl data function
    
    :param cash_costs: Cost in local currency units per instrument block 
    :type cash_costs: 3 tuple of floats; value_total_per_block, value_of_pertrade_commission, percentage_cost
    
    :param SR_cost: Cost in annualised Sharpe Ratio units (0.01 = 0.01 SR)
    :type SR_cost: float

    Set to None if not using. If both included use SR_cost

    :param ann_risk: Capital (capital * ann vol) at risk on annaulised basis. Used for SR calculations
    :type ann_risk: Tx1 pd.Series
    
    :returns : Tx1 pd.Series of costs. Minus numbers are losses
    
    """

    (cum_trades, trades_to_use, instr_ccy_returns,
        base_ccy_returns, use_fx, value_of_price_point)=returns_data

    if SR_cost is not None:
        ## use SR_cost
        ann_cost = -SR_cost*ann_risk
        
        costs_instr_ccy = ann_cost/BUSINESS_DAYS_IN_YEAR
    
    elif cash_costs is not None:
        ## use cost per blocks
        
        (value_total_per_block, value_of_pertrade_commission, percentage_cost)=cash_costs

        trades_in_blocks=trades_to_use.abs()
        costs_blocks = - trades_in_blocks*value_total_per_block

        value_of_trades=trades_in_blocks * value_of_price_point
        costs_percentage = percentage_cost * value_of_trades
        
        traded=trades_to_use[trades_to_use>0]
        
        if len(traded)==0:
            costs_pertrade = pd.Series([0.0]*len(cum_trades.index), cum_trades.index)
        else:
            costs_pertrade = pd.Series([value_of_pertrade_commission]*len(traded.index), traded.index)
            costs_pertrade = costs_pertrade.reindex(trades_to_use.index)

        ## everything on the trades index, so can do this:s        
        costs_instr_ccy = costs_blocks+costs_percentage+costs_pertrade

    else:
        ## set costs to zero
        costs_instr_ccy=pd.Series([0.0]*len(use_fx), index=use_fx.index)

    ## fx is on master (price timestamp)
    ## costs_instr_ccy needs downsampling
    costs_instr_ccy=costs_instr_ccy.cumsum().ffill().reindex(use_fx.index).diff()
    
    costs_base_ccy=costs_instr_ccy *  use_fx.ffill()
    costs_base_ccy[np.isnan(costs_base_ccy)]=0.0

    return (costs_base_ccy, costs_instr_ccy)

def resolve_capital(ts_to_scale_to, capital=None, ann_risk_target=None):
    """
    Resolve and setup capital
    We need capital for % returns and possibly for SR stuff

    Capital is used for:
    
      - going from forecast to position in profit and loss calculation (fixed or a time series): daily_risk_capital
      - calculating costs from SR costs (always a time series): ann_risk
      - calculating percentage returns (maybe fixed or variable time series): capital

    :param ts_to_scale_to: If capital is fixed, what time series to scale it to  
    :type capital: Tx1 pd.DataFrame
    
    :param capital: Capital at risk. Used for % returns, and calculating daily risk for SR costs  
    :type capital: None, int, float or Tx1 pd.DataFrame
    
    :param ann_risk_target: Annual risk target, as % of capital 0.10 is 10%. Used to calculate daily risk for SR costs
    :type ann_risk_target: None or float
    
    :returns tuple: 3 tuple of Tx1 pd.Series / float, pd.Series, pd.Series or float
    (capital, ann_risk, daily_risk_capital)

    """
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

    

    
        


if __name__ == '__main__':
    import doctest
    doctest.testmod()


