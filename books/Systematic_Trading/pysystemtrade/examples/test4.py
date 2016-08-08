import sys; sys.path.append('..')

from systems.provided.futures_chapter15.basesystem import futures_system
system=futures_system()
system.accounts.portfolio().stats() ## see some statistics
print (system.accounts.portfolio().stats()) ## see some statistics
print (system.accounts.pandl_for_instrument_forecast("EDOLLAR", "carry").skew())
