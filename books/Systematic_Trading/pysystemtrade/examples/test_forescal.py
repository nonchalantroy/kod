import sys; sys.path.append('..')
import inspect
from copy import copy
from systems.provided.futures_chapter15.estimatedsystem import futures_system

system=futures_system()
system.config.forecast_scalar_estimate['min_periods']=50
system.config.forecast_scalar_estimate['pool_instruments']=False
res = system.forecastScaleCap.get_forecast_scalar("EDOLLAR", "ewmac64_256") #1.68
#res = system.forecastScaleCap.get_forecast_scalar("EDOLLAR", "ewmac32_128") #2.57
#res = system.forecastScaleCap.get_forecast_scalar("EDOLLAR", "ewmac2_8") # 12.22
print (res.tail())
