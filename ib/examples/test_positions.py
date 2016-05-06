#!/usr/bin/env python
# show positions in account

from time import sleep

# LOAD the ib.ext and ib.opt Libraries
from ib.ext.Contract import Contract
from ib.opt import ibConnection, message

# DEFINE a basic function to capture error messages
def error_handler(msg):
    print "Error", msg

# DEFINE a basic function to print the "raw" server replies
def replies_handler(msg):
    print "Server Reply:", msg

# DEFINE a basic function to print the "parsed" server replies for an
# IB Request of "Portfolio Update" to list an IB portfolio position
def print_portfolio_position(msg):
    print "Position:", msg.contract.m_symbol,
                       msg.position,
                       msg.marketPrice,
                       msg.contract.m_currency,
                       msg.contract.m_secType

# Main code - adding "if __name__ ==" is not necessary

# Create the connection to IBGW with client socket id=1234
ibgw_conChannel = ibConnection(port=4002,clientId=1)
ibgw_conChannel.connect()


# Map server replies for "Error" messages to the "error_handler" function
ibgw_conChannel.register(error_handler, 'Error')

# Map server replies to "print_portfolio_position" function for
# "UpdatePortfolio" client requests
ibgw_conChannel.register(print_portfolio_position, 'UpdatePortfolio')

# Map server "raw" replies to "replies_handler" function for "UpdateAccount"
# client requests 
ibgw_conChannel.register(replies_handler, 'UpdateAccountValue')

# Make client request for AccountUpdates (includes request for Portfolio positions)
ibgw_conChannel.reqAccountUpdates(1, 'DU401739')

sleep(5)

# Disconnect - optional
print 'disconnected', ibgw_conChannel.disconnect()
