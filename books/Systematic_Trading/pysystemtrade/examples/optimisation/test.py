import logging
import logging
import logging
import logging
import logging
import logging
import logging
import logging
import sys; sys.path.append('../..')
from matplotlib.pyplot import show, title

from systems.provided.futures_chapter15.estimatedsystem import futures_system

system=futures_system()

#system.forecastScaleCap.get_scaled_forecast("EDOLLAR", "ewmac64_256").plot()
res=system.rules.get_raw_forecast("EDOLLAR", "ewmac2_8").dropna().head(5)
print (res)

#            ewmac2_8
#1983-10-10  0.695929
#1983-10-11 -0.604704
#1983-10-12 -0.536305
#1983-10-13 -0.737899
#1983-10-14 -0.242641

#show()
