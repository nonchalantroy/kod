import inspect
from copy import copy
import sys; sys.path.append('..')
import pandas as pd, numpy as np, random
from sysdata.configdata import Config
from systems.account import Account
from systems.forecast_combine import ForecastCombine
from systems.forecast_scale_cap import ForecastScaleCap
from systems.basesystem import System
from sysdata.csvdata import csvFuturesData
from systems.forecasting import Rules
from systems.forecasting import TradingRule
from systems.positionsizing import PositionSizing
from systems.stage import SystemStage
from systems.basesystem import ALL_KEYNAME
from syscore.objects import update_recalc, resolve_function
from syscore.genutils import str2Bool
from syscore.genutils import str2Bool, group_dict_from_natural

from syslogdiag.log import logtoscreen
from syscore.pdutils import df_from_list, must_have_item
from scipy.optimize import minimize
import datetime

CALENDAR_DAYS_IN_YEAR = 365.25
BUSINESS_DAYS_IN_YEAR = 256.0
ROOT_BDAYS_INYEAR = BUSINESS_DAYS_IN_YEAR**.5
WEEKS_IN_YEAR = CALENDAR_DAYS_IN_YEAR / 7.0
ROOT_WEEKS_IN_YEAR = WEEKS_IN_YEAR**.5
MONTHS_IN_YEAR = 12.0
ROOT_MONTHS_IN_YEAR = MONTHS_IN_YEAR**.5
ARBITRARY_START=pd.datetime(1900,1,1)
TARGET_ANN_SR=0.5
FLAG_BAD_RETURN=-9999999.9

def get_avg_corr(sigma):
    new_sigma=copy(sigma)
    np.fill_diagonal(new_sigma,np.nan)
    if np.all(np.isnan(new_sigma)):
        return np.nan    
    avg_corr=np.nanmean(new_sigma)
    return avg_corr


def correlation_single_period(data_for_estimate, 
                              using_exponent=True, min_periods=20, ew_lookback=250,
                              floor_at_zero=True):

    using_exponent=str2Bool(using_exponent)
            
    if using_exponent:
        dindex=data_for_estimate.index
        dlenadj=float(len(dindex))/len(set(list(dindex)))
        corrmat=pd.ewmcorr(data_for_estimate, span=int(ew_lookback*dlenadj), min_periods=min_periods)
        corrmat=corrmat.values[-1]
    else:
        corrmat=data_for_estimate.corr(min_periods=min_periods)
        corrmat=corrmat.values
    if floor_at_zero:
        corrmat[corrmat<0]=0.0
    return corrmat

def clean_correlation(corrmat, corr_with_no_data, must_haves=None):
    if must_haves is None:
        must_haves=[True]*corrmat.shape[0]
    if not np.any(np.isnan(corrmat)):
        return corrmat
    if np.all(np.isnan(corrmat)):
        return corr_with_no_data
    size_range=range(corrmat.shape[0])
    avgcorr=get_avg_corr(corrmat)
    def _good_correlation(i,j,corrmat, avgcorr, must_haves, corr_with_no_data):
        value=corrmat[i][j]
        must_have_value=must_haves[i] and must_haves[j]
        
        if np.isnan(value):
            if must_have_value:
                return avgcorr
            else:
                return corr_with_no_data[i][j]
        else:
            return value

    corrmat=np.array([[_good_correlation(i,j, corrmat, avgcorr, must_haves,corr_with_no_data) 
                       for i in size_range] for j in size_range], ndmin=2)
    np.fill_diagonal(corrmat,1.0)    
    return corrmat

def boring_corr_matrix(size, offdiag=0.99, diag=1.0):
    size_index=range(size)
    def _od(offdag, i, j):
        if i==j:
            return diag
        else:
            return offdiag
    m= [[_od(offdiag, i,j) for i in size_index] for j in size_index]
    m=np.array(m)
    return m

class CorrelationList(object):
    def __init__(self, corr_list, column_names, fit_dates):
        setattr(self, "corr_list", corr_list)
        setattr(self, "columns", column_names)
        setattr(self, "fit_dates", fit_dates)
    def __repr__(self):
        return "%d correlation estimates for %s" % (len(self.corr_list), ",".join(self.columns))
    
class CorrelationEstimator(CorrelationList):

    def __init__(self, data, log=logtoscreen("optimiser"), frequency="W",
                 date_method="expanding", rollyears=20, 
                 dict_group=dict(), boring_offdiag=0.99, cleaning=True, **kwargs):
        cleaning=str2Bool(cleaning)
    
        ## grouping dictionary, convert to faster, algo friendly, form
        group_dict=group_dict_from_natural(dict_group)

        data=df_from_list(data)    
        column_names=list(data.columns)

        data=data.resample(frequency, how="last")
            
        ### Generate time periods
        fit_dates = generate_fitting_dates(data, date_method=date_method, rollyears=rollyears)
        size=len(column_names)
        corr_with_no_data=boring_corr_matrix(size, offdiag=boring_offdiag)        
        ## create a list of correlation matrices
        corr_list=[]        
        print(__file__ + ":" + str(inspect.getframeinfo(inspect.currentframe())[:3][1]) + ":" +"Correlation estimate")
        
        ## Now for each time period, estimate correlation
        for fit_period in fit_dates:
            print(__file__ + ":" + str(inspect.getframeinfo(inspect.currentframe())[:3][1]) + ":" +"Estimating from %s to %s" % (fit_period.period_start, fit_period.period_end))            
            if fit_period.no_data:
                ## no data to fit with
                corr_with_nan=boring_corr_matrix(size, offdiag=np.nan, diag=np.nan)
                corrmat=corr_with_nan                
            else:                
                data_for_estimate=data[fit_period.fit_start:fit_period.fit_end]  
                corrmat=correlation_single_period(data_for_estimate, **kwargs)

            if cleaning:
                current_period_data=data[fit_period.fit_start:fit_period.fit_end] 
                must_haves=must_have_item(current_period_data)

                # means we can use earlier correlations with sensible values
                corrmat=clean_correlation(corrmat, corr_with_no_data, must_haves) 

            corr_list.append(corrmat)
        
        setattr(self, "corr_list", corr_list)
        setattr(self, "columns", column_names)
        setattr(self, "fit_dates", fit_dates)

def generate_fitting_dates(data, date_method, rollyears=20):

    print ("date_method=" + str(date_method))
    if date_method not in ["in_sample","rolling", "expanding"]:
        raise Exception("don't recognise date_method %s should be one of in_sample, expanding, rolling" % date_method)
    
    if type(data) is list:
        start_date=min([dataitem.index[0] for dataitem in data])
        end_date=max([dataitem.index[-1] for dataitem in data])
    else:
        start_date=data.index[0]
        end_date=data.index[-1]

    if date_method=="in_sample":
        return [fit_dates_object(start_date, end_date, start_date, end_date)]

    yearstarts=list(pd.date_range(start_date, end_date, freq="12M"))+[end_date]

    periods=[]
    for tidx in range(len(yearstarts))[1:-1]:
        period_start=yearstarts[tidx]
        period_end=yearstarts[tidx+1]
        if date_method=="expanding":
            fit_start=start_date
        elif date_method=="rolling":
            yearidx_to_use=max(0, tidx-rollyears)
            fit_start=yearstarts[yearidx_to_use]
        else:
            raise Exception("don't recognise date_method %s should be one of in_sample, expanding, rolling" % date_method)
            
        if date_method in ['rolling', 'expanding']:
            fit_end=period_start
        else:
            raise Exception("don't recognise date_method %s " % date_method)        
        periods.append(fit_dates_object(fit_start, fit_end, period_start, period_end))
    if date_method in ['rolling', 'expanding']:
        periods=[fit_dates_object(start_date, start_date, start_date, yearstarts[1], no_data=True)]+periods

    return periods

def robust_vol_calc(x, days=35, min_periods=10, vol_abs_min=0.0000000001, vol_floor=True,
                    floor_min_quant=0.05, floor_min_periods=100,
                    floor_days=500):
    vol = pd.ewmstd(x, span=days, min_periods=min_periods)
    vol[vol < vol_abs_min] = vol_abs_min
    if vol_floor:
        vol_min = pd.rolling_quantile(
            vol, floor_days, floor_min_quant, floor_min_periods)
        vol_min.set_value(vol_min.index[0], 0.0)
        vol_min = vol_min.ffill()
        vol_with_min = pd.concat([vol, vol_min], axis=1)
        vol_floored = vol_with_min.max(axis=1, skipna=False)
    else:
        vol_floored = vol

    return vol_floored

def ewmac(price, Lfast=32, Lslow=128):
    fast_ewma = pd.ewma(price, span=Lfast)
    slow_ewma = pd.ewma(price, span=Lslow)
    raw_ewmac = fast_ewma - slow_ewma
    vol = robust_vol_calc(price.diff())
    return raw_ewmac / vol

def un_fix_weights(mean_list, weights):
    def _unfixit(xmean, xweight):
        if xmean==FLAG_BAD_RETURN:
            return np.nan
        else:
            return xweight    
    fixed_weights=[_unfixit(xmean, xweight) for (xmean, xweight) in zip(mean_list, weights)]    
    return fixed_weights


class fit_dates_object(object):
    def __init__(self, fit_start, fit_end, period_start, period_end, no_data=False):
        setattr(self, "fit_start", fit_start)
        setattr(self, "fit_end", fit_end)
        setattr(self, "period_start", period_start)
        setattr(self, "period_end", period_end)
        setattr(self, "no_data", no_data)        
    def __repr__(self):
        if self.no_data:
            return "Fit without data, use from %s to %s" % (self.period_start, self.period_end)
        else:
            return "Fit from %s to %s, use in %s to %s" % (self.fit_start, self.fit_end, self.period_start, self.period_end)
        
class PortfoliosEstimated(SystemStage):
    
    def __init__(self): setattr(self, "name", "portfolio")
        
    def get_instrument_correlation_matrix(self, system):
        #pandl=system.accounts.pandl_across_subsystems().to_frame()
        #pandl.to_csv("out.csv")
        dfs = []
        insts = ['EDOLLAR','US10','EUROSTX','V2X','MXP','CORN']
        for c in insts:
            df = pd.read_csv("c:/Users/burak/Documents/kod/books/Systematic_Trading/pysystemtrade/examples/out-%s.csv" % c,index_col=0,parse_dates=True)
            dfs.append(df)
        pandl = pd.concat(dfs,axis=1)        
        #frequency=corr_params['frequency']
        frequency="W"
        print ("frequency=" + str(frequency))
        pandl=pandl.cumsum().resample(frequency).diff()
        return CorrelationEstimator(pandl, frequency=frequency,
                                    ew_lookback=500, floor_at_zero=True,
                                    min_periods=20,
                                    cleaning=True,using_exponent=True,
                                    date_method='expanding', rollyears=20)
    
if __name__ == "__main__": 
     
    random.seed(0)
    np.random.seed(0)
    data = csvFuturesData()
    my_config = Config()
    insts = ['EDOLLAR','US10','EUROSTX','V2X','MXP','CORN']
    my_config.instruments=insts
    ewmac_8 = TradingRule((ewmac, [], dict(Lfast=8, Lslow=32)))
    ewmac_32 = TradingRule(dict(function=ewmac, other_args=dict(Lfast=32, Lslow=128)))
    my_rules = Rules(dict(ewmac8=ewmac_8, ewmac32=ewmac_32))

    my_system = System([Account(), PortfoliosEstimated(), PositionSizing(), ForecastScaleCap(), my_rules, ForecastCombine()], data, my_config)
    res = my_system.portfolio.get_instrument_correlation_matrix(my_system).corr_list[-1]
    res = pd.DataFrame(res, index=insts)
    res.columns = insts
    print (res)
