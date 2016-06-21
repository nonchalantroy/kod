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
from syscore.pdutils import  fix_weights_vs_pdm
from syscore.objects import update_recalc, resolve_function
from syscore.genutils import str2Bool

class Portfolios(SystemStage):    
    def __init__(self):
        setattr(self, "name", "portfolio")
        setattr(self, "description", "unswitched")
        
    def _system_init(self, system):
        if str2Bool(system.config.use_instrument_weight_estimates):
            fixed_flavour=False
        else:
            fixed_flavour=True    
        
        if fixed_flavour:
            self.__class__= PortfoliosFixed
            self.__init__()
            setattr(self, "parent", system)

        else:
            self.__class__= PortfoliosEstimated
            self.__init__()
            setattr(self, "parent", system)

class PortfoliosFixed(SystemStage):

    def __init__(self):
        setattr(self, "name", "portfolio")
        setattr(self, "description", "fixed")

    def get_subsystem_position(self, instrument_code):
        return self.parent.positionSize.get_subsystem_position(instrument_code)

    def get_volatility_scalar(self, instrument_code):
        return self.parent.positionSize.get_volatility_scalar(instrument_code)


    def get_raw_instrument_weights(self):
        def _get_raw_instrument_weights(system, an_ignored_variable, this_stage):
            print(__file__ + ":" + str(inspect.getframeinfo(inspect.currentframe())[:3][1]) + ":" +"Calculating raw instrument weights")

            instrument_weights = system.config.instrument_weights
            instrument_list = sorted(instrument_weights.keys())

            subsys_ts = [
                this_stage.get_subsystem_position(instrument_code).index
                for instrument_code in instrument_list]

            earliest_date = min([min(fts) for fts in subsys_ts])
            latest_date = max([max(fts) for fts in subsys_ts])

            weight_ts = pd.date_range(start=earliest_date, end=latest_date)

            instrument_weights_weights = dict([
                (instrument_code, pd.Series([instrument_weights[
                 instrument_code]] * len(weight_ts), index=weight_ts))
                for instrument_code in instrument_list])

            instrument_weights_weights = pd.concat(
                instrument_weights_weights, axis=1)
            instrument_weights_weights.columns = instrument_list

            return instrument_weights_weights

        instrument_weights = self.parent.calc_or_cache(
            "get_raw_instrument_weights", ALL_KEYNAME, _get_raw_instrument_weights, self)
        return instrument_weights

    def get_instrument_weights(self):
        def _get_instrument_weights(system, an_ignored_variable, this_stage):

            print(__file__ + ":" + str(inspect.getframeinfo(inspect.currentframe())[:3][1]) + ":" +"Calculating instrument weights")

            raw_instr_weights = this_stage.get_raw_instrument_weights()
            instrument_list = list(raw_instr_weights.columns)

            subsys_positions = [this_stage.get_subsystem_position(instrument_code)
                                for instrument_code in instrument_list]

            subsys_positions = pd.concat(subsys_positions, axis=1).ffill()
            subsys_positions.columns = instrument_list
            instrument_weights = fix_weights_vs_pdm(
                raw_instr_weights, subsys_positions)
            weighting=system.config.instrument_weight_ewma_span  
            instrument_weights = pd.ewma(instrument_weights, weighting) 
            return instrument_weights

        instrument_weights = self.parent.calc_or_cache(
            "get_instrument_weights", ALL_KEYNAME, _get_instrument_weights, self)
        return instrument_weights


    def get_instrument_diversification_multiplier(self):
        def _get_instrument_div_multiplier(
                system, an_ignored_variable, this_stage):

            print(__file__ + ":" + str(inspect.getframeinfo(inspect.currentframe())[:3][1]) + ":" +"Calculating diversification multiplier")

            div_mult=system.config.instrument_div_multiplier
            weight_ts = this_stage.get_instrument_weights().index
            ts_idm = pd.Series([div_mult] * len(weight_ts), index=weight_ts)
            return ts_idm

        instrument_div_multiplier = self.parent.calc_or_cache(
            "get_instrument_diversification_multiplier", ALL_KEYNAME, _get_instrument_div_multiplier, self)
        return instrument_div_multiplier

    def capital_multiplier(self):
        return self.parent.accounts.capital_multiplier()
        
class PortfoliosEstimated(PortfoliosFixed):
    def __init__(self):

        super(PortfoliosEstimated, self).__init__()
        protected = ['get_instrument_correlation_matrix']
        update_recalc(self, protected)
        setattr(self, "description", "Estimated")    
        nopickle=["calculation_of_raw_instrument_weights"]
        setattr(self, "_nopickle", nopickle)

    def get_instrument_subsystem_SR_cost(self, instrument_code):
        return self.parent.accounts.subsystem_SR_costs(instrument_code, roundpositions=False)
    
    def get_instrument_correlation_matrix(self):

        def _get_instrument_correlation_matrix(system, NotUsed,  this_stage, 
                                               corr_func, **corr_params):

            print(__file__ + ":" + str(inspect.getframeinfo(inspect.currentframe())[:3][1]) + ":" +"Calculating instrument correlations")

            instrument_codes=system.get_instrument_list()
            pandl=this_stage.pandl_across_subsystems().to_frame()            
            frequency=corr_params['frequency']
            pandl=pandl.cumsum().resample(frequency).diff()
            return corr_func(pandl,  log=this_stage.log.setup(call="correlation"), **corr_params)
                            
        corr_params=copy(self.parent.config.instrument_correlation_estimate)
        corr_func=resolve_function(corr_params.pop("func"))
        forecast_corr_list = self.parent.calc_or_cache(
            'get_instrument_correlation_matrix', ALL_KEYNAME,  
            _get_instrument_correlation_matrix,
             self,  corr_func, **corr_params)
        return forecast_corr_list

    def get_instrument_diversification_multiplier(self, system):

        div_mult_params=copy(system.config.instrument_div_mult_estimate)            
        idm_func=resolve_function(div_mult_params.pop("func"))            
        correlation_list_object=self.get_instrument_correlation_matrix()
        weight_df=self.get_instrument_weights()
        ts_idm=idm_func(correlation_list_object, weight_df, **div_mult_params)
        return ts_idm
        
    def get_raw_instrument_weights(self):

        def _get_raw_instrument_weights(system, notUsed, this_stage):
            print(__file__ + ":" + str(inspect.getframeinfo(inspect.currentframe())[:3][1]) + ":" +"Getting raw instrument weights")

            return this_stage.calculation_of_raw_instrument_weights().weights

        raw_instrument_weights = self.parent.calc_or_cache(
            'get_raw_instrument_weights',  ALL_KEYNAME,
            _get_raw_instrument_weights,
             self)
                
        return raw_instrument_weights

        instrument_weights = self.parent.calc_or_cache(
            'get_instrument_weights', ALL_KEYNAME, _get_instrument_weights, self)
        return instrument_weights

    def pandl_across_subsystems(self): 
        return self.parent.accounts.pandl_across_subsystems()


    def calculation_of_raw_instrument_weights(self):
        def _calculation_of_raw_instrument_weights(system, NotUsed1, this_stage, 
                                      weighting_func, **weighting_params):


            print(__file__ + ":" + str(inspect.getframeinfo(inspect.currentframe())[:3][1]) + ":" +"Calculating raw instrument weights")

            instrument_codes=system.get_instrument_list()

            weight_func=weighting_func(log=this_stage.log.setup(call="weighting"), **weighting_params)
            pandl=this_stage.pandl_across_subsystems()
            (pandl_gross, pandl_costs) = decompose_group_pandl([pandl]) 
            weight_func.set_up_data(data_gross = pandl_gross, data_costs = pandl_costs)

            SR_cost_list = [0.0, 0.0]

            weight_func.optimise(ann_SR_costs=SR_cost_list)
        
            return weight_func


        ## Get some useful stuff from the config
        weighting_params=copy(self.parent.config.instrument_weight_estimate)

        ## which function to use for calculation
        weighting_func=resolve_function(weighting_params.pop("func"))
        
        calcs_of_instrument_weights = self.parent.calc_or_cache(
            'calculation_of_raw_instrument_weights', ALL_KEYNAME, 
            _calculation_of_raw_instrument_weights,
             self, weighting_func, **weighting_params)
        
        return calcs_of_instrument_weights

random.seed(0)
np.random.seed(0)

data = csvFuturesData()
my_config = Config()
my_config.instruments=["US20", "SP500"]

ewmac_8 = TradingRule((ewmac, [], dict(Lfast=8, Lslow=32)))
ewmac_32 = TradingRule(
    dict(function=ewmac, other_args=dict(Lfast=32, Lslow=128)))
my_rules = Rules(dict(ewmac8=ewmac_8, ewmac32=ewmac_32))

my_system = System([Account(), PortfoliosEstimated(), PositionSizing(), ForecastScaleCap(), my_rules, ForecastCombine()], data, my_config)
my_system.config.forecast_weight_estimate['method']="equal_weights"
my_system.config.instrument_weight_estimate['method']="bootstrap"
my_system.config.instrument_weight_estimate["monte_runs"]=1
my_system.config.instrument_weight_estimate["bootstrap_length"]=250
print(my_system.portfolio.get_instrument_diversification_multiplier(my_system))
print (my_system.portfolio.get_instrument_weights())

# 10,250 weights=0.75,0.25 idm=1.26
# 30,250 weights=0.75,0.25 
