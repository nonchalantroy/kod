import Quandl

auth = open('../.quandl').read()

#res = Quandl.search(query = 'China crude oil consumption',authtoken=auth)
#print len(res), type(res)

#mydata = Quandl.get("OFDP/FUTURE_RB1",
mydata = Quandl.get("ICE/T", 
                    trim_start="2008-01-01",
                    trim_end="2009-01-01",
                    returns="pandas",
                    authtoken=auth)

print mydata
#              Open    High     Low  Settle  Volume  Open Interest
#Date                                                             
