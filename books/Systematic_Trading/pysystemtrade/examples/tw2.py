import inspect
from copy import copy
import sys; sys.path.append('..')
import pandas as pd, numpy as np, random
from systems.provided.example.rules import ewmac_forecast_with_defaults as ewmac
from sysdata.configdata import Config
from systems.account import Account
from systems.forecast_combine import ForecastCombine
from systems.forecast_scale_cap import ForecastScaleCap
from systems.basesystem import System
from sysdata.csvdata import csvFuturesData
from systems.forecasting import Rules
from systems.forecasting import TradingRule
from systems.positionsizing import PositionSizing
from syscore.accounting import decompose_group_pandl
from systems.stage import SystemStage
from systems.basesystem import ALL_KEYNAME
from syscore.objects import update_recalc, resolve_function
from syscore.genutils import str2Bool
from syscore.correlations import CorrelationEstimator
from syslogdiag.log import logtoscreen
from syscore.pdutils import df_from_list, must_have_item
from syscore.dateutils import generate_fitting_dates
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

def un_fix_weights(mean_list, weights):
    def _unfixit(xmean, xweight):
        if xmean==FLAG_BAD_RETURN:
            return np.nan
        else:
            return xweight    
    fixed_weights=[_unfixit(xmean, xweight) for (xmean, xweight) in zip(mean_list, weights)]    
    return fixed_weights


def variance(weights, sigma):
    return (weights*sigma*weights.transpose())[0,0]

def neg_SR(weights, sigma, mus):
    weights=np.matrix(weights)
    estreturn=(weights*mus)[0,0]
    std_dev=(variance(weights,sigma)**.5)    
    return -estreturn/std_dev

def addem(weights):
    return 1.0 - sum(weights)

def fix_sigma(sigma):    
    def _fixit(x):
        if np.isnan(x):
            return 0.0
        else:
            return x    
    sigma=[[_fixit(x) for x in sigma_row] for sigma_row in sigma]    
    sigma=np.array(sigma)    
    return sigma

def fix_mus(mean_list):
    def _fixit(x):
        if np.isnan(x):
            return FLAG_BAD_RETURN
        else:
            return x    
    mean_list=[_fixit(x) for x in mean_list]    
    return mean_list

def optimise( sigma, mean_list):
    mean_list=fix_mus(mean_list)
    sigma=fix_sigma(sigma)    
    mus=np.array(mean_list, ndmin=2).transpose()
    number_assets=sigma.shape[1]
    start_weights=[1.0/number_assets]*number_assets
    bounds=[(0.0,1.0)]*number_assets
    cdict=[{'type':'eq', 'fun':addem}]
    ans=minimize(neg_SR, start_weights, (sigma, mus), method='SLSQP', bounds=bounds, constraints=cdict, tol=0.00001)
    weights=ans['x']
    weights=un_fix_weights(mean_list, weights)    
    return weights

def sigma_from_corr_and_std(stdev_list, corrmatrix):
    stdev=np.array(stdev_list, ndmin=2).transpose()
    sigma=stdev*corrmatrix*stdev
    return sigma

def SR_equaliser(stdev_list, target_SR):
    return [target_SR * asset_stdev for asset_stdev in stdev_list]

def vol_equaliser(mean_list, stdev_list):
    if np.all(np.isnan(stdev_list)):
        return (([np.nan]*len(mean_list), [np.nan]*len(stdev_list)))

    avg_stdev=np.nanmean(stdev_list)
    norm_factor=[asset_stdev/avg_stdev for asset_stdev in stdev_list]    
    norm_means=[mean_list[i]/norm_factor[i] for (i, notUsed) in enumerate(mean_list)]
    norm_stdev=[stdev_list[i]/norm_factor[i] for (i, notUsed) in enumerate(stdev_list)]    
    return (norm_means, norm_stdev)

def equal_weights(period_subset_data, moments_estimator,
                cleaning, must_haves, **other_opt_args_ignored):
    asset_count = period_subset_data.shape[1]
    weights= [1.0 / asset_count for i in range(asset_count)]    
    diag=dict(raw=None, sigma=None, mean_list=None, 
              unclean=weights, weights=weights)    
    return (weights, diag)

def bs_one_time(subset_data, moments_estimator, cleaning, must_haves,
                bootstrap_length,**other_opt_args):
    bs_idx=[int(random.uniform(0,1)*len(subset_data)) for notUsed in range(bootstrap_length)]    
    returns=subset_data.iloc[bs_idx,:]     
    (weights, diag)=markosolver(returns, moments_estimator, cleaning, must_haves, 
                       **other_opt_args)
    return (weights, diag)

        
def clean_weights(weights,  must_haves=None, fraction=0.5):
    if must_haves is None:
        must_haves=[True]*len(weights)    
    if not any(must_haves):
        return [0.0]*len(weights)    
    needs_replacing=[(np.isnan(x) or x==0.0) and must_haves[i] for (i,x) in enumerate(weights)]
    keep_empty=[(np.isnan(x) or x==0.0) and not must_haves[i] for (i,x) in enumerate(weights)]
    no_replacement_needed=[(not keep_empty[i]) and (not needs_replacing[i]) for (i,x) in enumerate(weights)]
    if not any(needs_replacing):
        return weights    
    missing_weights=sum(needs_replacing)
    total_for_missing_weights=fraction*missing_weights/(
        float(np.nansum(no_replacement_needed)+np.nansum(missing_weights)))
    
    adjustment_on_rest=(1.0-total_for_missing_weights)    
    each_missing_weight=total_for_missing_weights/missing_weights    
    def _good_weight(value, idx, needs_replacing, keep_empty, 
                     each_missing_weight, adjustment_on_rest):        
        if needs_replacing[idx]:
            return each_missing_weight
        if keep_empty[idx]:
            return 0.0
        else:
            return value*adjustment_on_rest

    weights=[_good_weight(value, idx, needs_replacing, keep_empty, 
                          each_missing_weight, adjustment_on_rest) 
             for (idx, value) in enumerate(weights)]    
    xsum=sum(weights)
    weights=[x/xsum for x in weights]    
    return weights
        
def work_out_net(data_gross, data_costs, annualisation=BUSINESS_DAYS_IN_YEAR,   
                 equalise_gross=False, cost_multiplier=1.0,
                 period_target_SR=TARGET_ANN_SR/(BUSINESS_DAYS_IN_YEAR**.5)):
    if equalise_gross:
        target_vol=np.mean(data_gross.std().values) ## assumes all have same vol
        target_mean=period_target_SR*(target_vol)
        actual_period_mean=data_gross.mean().values
        shifts = target_mean - actual_period_mean
        shifts = np.array([list(shifts)]*len(data_gross.index))
        shifts=pd.DataFrame(shifts, index=data_gross.index, columns=data_gross.columns)    
        use_gross = data_gross + shifts    
    else:
        use_gross = data_gross    

    use_costs = data_costs * cost_multiplier 
    net = use_gross + use_costs ## costs are negative    
    return net

def bootstrap_portfolio(subset_data, moments_estimator,
                cleaning, must_haves,
                  monte_runs=100, bootstrap_length=50,
                  **other_opt_args):
    print(__file__ + ":" + str(inspect.getframeinfo(inspect.currentframe())[:3][1]) + ":" + "bootstrap_length=" + str(bootstrap_length))
    print(__file__ + ":" + str(inspect.getframeinfo(inspect.currentframe())[:3][1]) + ":" + "bootstrap_length=" + str(type(moments_estimator)))

    all_results=[bs_one_time(subset_data, moments_estimator,
                            cleaning, must_haves, 
                            bootstrap_length,
                            **other_opt_args)
                                for unused_index in range(monte_runs)]
        
    weightlist=np.array([x[0] for x in all_results], ndmin=2)
    diaglist=[x[1] for x in all_results]
         
    theweights_mean=list(np.mean(weightlist, axis=0))
    
    diag=dict(bootstraps=diaglist)
    
    return (theweights_mean, diag)

def markosolver(period_subset_data,
                moments_estimator,
                cleaning,
                must_haves,
                equalise_SR=False ,
                equalise_vols=True,
                **ignored_args): 

    print ("equalise_SR=" + str(equalise_SR))
    print ("equalise_vols=" + str(equalise_vols))
    print(__file__ + ":" + str(inspect.getframeinfo(inspect.currentframe())[:3][1]) + ":" + "markosolver")
    
    rawmoments=moments_estimator.moments(period_subset_data)    
    (mean_list, corrmatrix, stdev_list)=copy(rawmoments)

    if equalise_vols:
        (mean_list, stdev_list)=vol_equaliser(mean_list, stdev_list)        
    if equalise_SR:
        ann_target_SR = moments_estimator.ann_target_SR
        mean_list=SR_equaliser(stdev_list, ann_target_SR)
    
    sigma=sigma_from_corr_and_std(stdev_list, corrmatrix)    
    unclean_weights=optimise( sigma, mean_list)
    weights=clean_weights(unclean_weights, must_haves)
    diag=dict(raw=rawmoments, sigma=sigma, mean_list=mean_list, 
              unclean=unclean_weights, weights=weights)
    return (weights, diag)

def fix_weights_vs_pdm(weights, pdm):
    pdm_ffill = pdm.ffill()
    adj_weights = weights.reindex(pdm_ffill.index, method='ffill')
    adj_weights = adj_weights[pdm.columns]
    adj_weights[np.isnan(pdm_ffill)] = 0.0
    def _sum_row_fix(weight_row):
        swr = sum(weight_row)
        if swr == 0.0: return weight_row
        new_weights = weight_row / swr
        return new_weights
    adj_weights = adj_weights.apply(_sum_row_fix, 1)
    return adj_weights

def diversification_mult_single_period(corrmatrix, weights, dm_max=2.5):
    if all([x==0.0 for x in list(weights)]) or np.all(np.isnan(weights)): return 1.0
    weights=np.array(weights, ndmin=2)    
    dm=np.min([1.0 / (float(np.dot(np.dot(weights, corrmatrix), weights.transpose())) **.5),dm_max])
    return dm

def diversification_multiplier_from_list(correlation_list_object, weight_df_raw, 
                                         ewma_span=125,  **kwargs):
    weight_df=weight_df_raw[correlation_list_object.columns]
    ref_periods=[fit_period.period_start for fit_period in correlation_list_object.fit_dates]
    div_mult_vector=[]
    for (corrmatrix, start_of_period) in zip(correlation_list_object.corr_list, ref_periods):    
        weight_slice=weight_df[:start_of_period]
        if weight_slice.shape[0]==0:
            div_mult_vector.append(1.0)
            continue
        weights=list(weight_slice.iloc[-1,:].values)
        div_multiplier=diversification_mult_single_period(corrmatrix, weights,  **kwargs)
        div_mult_vector.append(div_multiplier)

    div_mult_df=pd.Series(div_mult_vector,  index=ref_periods)
    div_mult_df=div_mult_df.reindex(weight_df.index, method="ffill")
    div_mult_df=pd.ewma(div_mult_df, span=ewma_span)
    return div_mult_df

class optSinglePeriod(object):
    def __init__(self, parent, data, fit_period, optimiser, cleaning):
        if cleaning:
            current_period_data=data[fit_period.period_start:fit_period.period_end] 
            must_haves=must_have_item(current_period_data)
        
        else:
            must_haves=None
        
        if fit_period.no_data:
            diag=None
            size=current_period_data.shape[1]
            weights_with_nan=[np.nan/size]*size
            weights=weights_with_nan
            if cleaning:
                weights=clean_weights(weights, must_haves)
        else:
            subset_fitting_data=data[fit_period.fit_start:fit_period.fit_end]    
            (weights, diag)=optimiser.call(subset_fitting_data, cleaning, must_haves)

        setattr(self, "diag", diag)
        setattr(self, "weights", weights)

class optimiserWithParams(object):
    def __init__(self, method, optimise_params, moments_estimator):
        print(__file__ + ":" + str(inspect.getframeinfo(inspect.currentframe())[:3][1]) + ":" + "optimiserWithParams")        
        opt_func=bootstrap_portfolio
        setattr(self, "opt_func", resolve_function(opt_func))        
        setattr(self, "params", optimise_params)        
        setattr(self, "moments_estimator", moments_estimator)
        
    def call(self, optimise_data, cleaning, must_haves):        
        params=self.params
        return self.opt_func(optimise_data, self.moments_estimator, cleaning, must_haves, **params)
   
class GenericOptimiser(object):

    def __init__(self,  log=logtoscreen("optimiser"), frequency="W", date_method="expanding", 
                         rollyears=20, method="bootstrap", cleaning=True, 
                         cost_multiplier=1.0, apply_cost_weight=True, 
                         ann_target_SR=TARGET_ANN_SR, equalise_gross=False,
                         **passed_params):
                
        cleaning=str2Bool(cleaning)
        optimise_params=copy(passed_params)

        ## annualisation
        ann_dict=dict(D=BUSINESS_DAYS_IN_YEAR, W=WEEKS_IN_YEAR, M=MONTHS_IN_YEAR, Y=1.0)
        annualisation=ann_dict.get(frequency, 1.0)

        period_target_SR=ann_target_SR/(annualisation**.5)
        
        ## A moments estimator works out the mean, vol, correlation
        ## Also stores annualisation factor and target SR (used for shrinkage and equalising)
        moments_estimator=momentsEstimator(optimise_params, annualisation,  ann_target_SR)

        ## The optimiser instance will do the optimation once we have the appropriate data
        optimiser=optimiserWithParams(method, optimise_params, moments_estimator)

        setattr(self, "optimiser", optimiser)
        setattr(self, "log", log)
        setattr(self, "frequency", frequency)
        setattr(self, "method", method)
        setattr(self, "equalise_gross", equalise_gross)
        setattr(self, "cost_multiplier", cost_multiplier)
        setattr(self, "annualisation", annualisation)
        setattr(self, "period_target_SR", period_target_SR)
        setattr(self, "date_method", date_method)
        setattr(self, "rollyears", rollyears)
        setattr(self, "cleaning", cleaning)
        setattr(self, "apply_cost_weight", apply_cost_weight)

    def need_data(self):
        if self.method=="equal_weights":
            return False
        else:
            return True

    def set_up_data(self, data_gross=None, data_costs=None, weight_matrix=None):
        if weight_matrix is not None:
            setattr(self, "data", weight_matrix.ffill())
            return None
        
        log=self.log
        frequency=self.frequency
        equalise_gross = self.equalise_gross
        cost_multiplier = self.cost_multiplier
        annualisation = self.annualisation
        period_target_SR = self.period_target_SR

        data_gross = [data_item.cumsum().resample(frequency, how="last").diff() for
                       data_item in data_gross]
        
        data_costs = [data_item.cumsum().resample(frequency, how="last").diff() for
                      data_item in data_costs]

        data_gross=df_from_list(data_gross)    
        data_costs=df_from_list(data_costs)    

        if equalise_gross:
            print(__file__ + ":" + str(inspect.getframeinfo(inspect.currentframe())[:3][1]) + ":" +"Setting all gross returns to be identical - optimisation driven only by costs")
        if cost_multiplier!=1.0:
            print(__file__ + ":" + str(inspect.getframeinfo(inspect.currentframe())[:3][1]) + ":" +"Using cost multiplier on optimisation of %.2f" % cost_multiplier)                
        data = work_out_net(data_gross, data_costs, annualisation=annualisation,
                            equalise_gross=equalise_gross, cost_multiplier=cost_multiplier,
                            period_target_SR=period_target_SR)                    
        setattr(self, "data", data)

    def optimise(self, ann_SR_costs=None):
        log=self.log
        date_method = self.date_method
        rollyears = self.rollyears
        optimiser = self.optimiser
        cleaning = self.cleaning
        apply_cost_weight = self.apply_cost_weight        
        data=getattr(self, "data", None)
        if data is None:
            log.critical("You need to run .set_up_data() before .optimise()")
        
        fit_dates = generate_fitting_dates(data, date_method=date_method, rollyears=rollyears)
        setattr(self, "fit_dates", fit_dates)    
        weight_list=[]
        opt_results=[]
        print(__file__ + ":" + str(inspect.getframeinfo(inspect.currentframe())[:3][1]) + ":" +"Optimising...")
        
        for fit_period in fit_dates:            
            print(__file__ + ":" + str(inspect.getframeinfo(inspect.currentframe())[:3][1]) + ":" +"Optimising for data from %s to %s" % (str(fit_period.period_start), str(fit_period.period_end)))

            results_this_period=optSinglePeriod(self, data, fit_period, optimiser, cleaning)

            opt_results.append(results_this_period)

            weights=results_this_period.weights

            dindex=[fit_period.period_start+datetime.timedelta(days=1), 
                    fit_period.period_end-datetime.timedelta(days=1)]            

            weight_row=pd.DataFrame([weights]*2, index=dindex, columns=data.columns)
            weight_list.append(weight_row)
        raw_weight_df=pd.concat(weight_list, axis=0)
        
        if apply_cost_weight:
            print(__file__ + ":" + str(inspect.getframeinfo(inspect.currentframe())[:3][1]) + ":" +"Applying cost weighting to optimisation results")
            weight_df = apply_cost_weighting(raw_weight_df, ann_SR_costs)
        else:
            weight_df =raw_weight_df 
        
        setattr(self, "results", opt_results)
        setattr(self, "weights", weight_df)
        setattr(self, "raw_weights", raw_weight_df)

class momentsEstimator(object):
    def __init__(self, optimise_params, annualisation=BUSINESS_DAYS_IN_YEAR, 
                 ann_target_SR=.5):        
        corr_estimate_params=copy(optimise_params["correlation_estimate"])
        mean_estimate_params=copy(optimise_params["mean_estimate"])
        vol_estimate_params=copy(optimise_params["vol_estimate"])
        corr_estimate_func=resolve_function(corr_estimate_params.pop("func"))
        mean_estimate_func=resolve_function(mean_estimate_params.pop("func"))
        vol_estimate_func=resolve_function(vol_estimate_params.pop("func"))
        setattr(self, "corr_estimate_params", corr_estimate_params)
        setattr(self, "mean_estimate_params", mean_estimate_params)
        setattr(self, "vol_estimate_params", vol_estimate_params)        
        setattr(self, "corr_estimate_func", corr_estimate_func)
        setattr(self, "mean_estimate_func", mean_estimate_func)
        setattr(self, "vol_estimate_func", vol_estimate_func)
        period_target_SR = ann_target_SR / (annualisation**.5)        
        setattr(self, "annualisation", annualisation)
        setattr(self, "period_target_SR", period_target_SR)
        setattr(self, "ann_target_SR", ann_target_SR)

    def correlation(self, data_for_estimate):
        params=self.corr_estimate_params
        corrmatrix=self.corr_estimate_func(data_for_estimate, **params)
        return corrmatrix
    
    def means(self, data_for_estimate):
        params=self.mean_estimate_params
        mean_list=self.mean_estimate_func(data_for_estimate, **params)        
        mean_list=list(np.array(mean_list)*self.annualisation)
        return mean_list
    
    def vol(self, data_for_estimate):
        params=self.vol_estimate_params
        stdev_list=self.vol_estimate_func(data_for_estimate, **params)
        stdev_list=list(np.array(stdev_list)*(self.annualisation**.5))
        return stdev_list

    def moments(self, data_for_estimate):
        ans=(self.means(data_for_estimate), self.correlation(data_for_estimate),  self.vol(data_for_estimate))
        return ans
        
class PortfoliosEstimated(SystemStage):
    
    def __init__(self): setattr(self, "name", "portfolio")
        
    def get_instrument_correlation_matrix(self, system):
        corr_params=copy(system.config.instrument_correlation_estimate)
        tmp = corr_params.pop("func") # pop the function, leave the args
        instrument_codes=system.get_instrument_list()
        pandl=system.accounts.pandl_across_subsystems().to_frame()            
        frequency=corr_params['frequency']
        pandl=pandl.cumsum().resample(frequency).diff()
        return CorrelationEstimator(pandl, log=self.log.setup(call="correlation"), **corr_params)

    def get_instrument_diversification_multiplier(self, system):
        div_mult_params=copy(system.config.instrument_div_mult_estimate)            
        tmp=div_mult_params.pop("func")
        correlation_list_object=self.get_instrument_correlation_matrix(system)
        weight_df=self.get_instrument_weights(system)
        print ("weight_df=" + str(weight_df))
        ts_idm=diversification_multiplier_from_list(correlation_list_object, weight_df, **div_mult_params)
        return ts_idm

    def get_instrument_weights(self, system):

        print(__file__ + ":" + str(inspect.getframeinfo(inspect.currentframe())[:3][1]) + ":" +"Calculating instrument weights")
        instrument_codes=system.get_instrument_list()
        weighting_params=copy(system.config.instrument_weight_estimate)
        tmp=weighting_params.pop("func")
        weight_func=GenericOptimiser(**weighting_params)
        pandl=system.accounts.pandl_across_subsystems()
        (pandl_gross, pandl_costs) = decompose_group_pandl([pandl])
        # zeros
        pandl_costs[0] = pd.DataFrame(0, index=pandl_costs[0].index,columns=pandl_costs[0].columns)
        weight_func.set_up_data(data_gross = pandl_gross, data_costs = pandl_costs)
        weight_func.optimise(ann_SR_costs=[0.0, 0.0])
        raw_instr_weights = weight_func.weights

        instrument_list = list(raw_instr_weights.columns)
        subsys_positions = [system.positionSize.get_subsystem_position(instrument_code)
                            for instrument_code in instrument_list]

        subsys_positions = pd.concat(subsys_positions, axis=1).ffill()
        subsys_positions.columns = instrument_list
        instrument_weights = fix_weights_vs_pdm(raw_instr_weights, subsys_positions)
        weighting=system.config.instrument_weight_ewma_span  
        instrument_weights = pd.ewma(instrument_weights, weighting) 
        return instrument_weights

if __name__ == "__main__": 
     
    random.seed(0)
    np.random.seed(0)

    data = csvFuturesData()
    my_config = Config()
    my_config.instruments=["US20", "SP500"]

    ewmac_8 = TradingRule((ewmac, [], dict(Lfast=8, Lslow=32)))
    ewmac_32 = TradingRule(dict(function=ewmac, other_args=dict(Lfast=32, Lslow=128)))
    my_rules = Rules(dict(ewmac8=ewmac_8, ewmac32=ewmac_32))

    my_system = System([Account(), PortfoliosEstimated(), PositionSizing(), ForecastScaleCap(), my_rules, ForecastCombine()], data, my_config)
    my_system.config.forecast_weight_estimate['method']="equal_weights"
    my_system.config.instrument_weight_estimate['method']="bootstrap"
    my_system.config.instrument_weight_estimate["monte_runs"]=1
    my_system.config.instrument_weight_estimate["bootstrap_length"]=250
    print(my_system.portfolio.get_instrument_diversification_multiplier(my_system))

    # 10,250 weights=0.75,0.25 idm=1.26
    # 30,250 weights=0.75,0.25 
