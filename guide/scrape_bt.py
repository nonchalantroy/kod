'''
Get Blood Type Diet ingredient information from the Dadamo's site
'''
import re, urllib, urllib2
import logging, sys, os
from datetime import datetime
from datetime import timedelta
from urllib import FancyURLopener

class MyOpener(FancyURLopener):
    version = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11'        

fout = open("%s/food.dat" % os.environ['TEMP'],"w")
fout.write("Dadamo_Site_Id,Food,A_S,A_NS,B_S,B_NS,AB_S,AB_NS,O_S,O_NS\n")

opener = MyOpener()
url = "http://www.dadamo.com/typebase4/typeindexer.htm"
h = opener.open(url)
content = h.read()

tmp = re.findall("depictor5.pl\?(\d*?)\">(.*?)<", content)
for g in tmp:
    print g
    h = opener.open("http://www.dadamo.com/typebase4/depictor5.pl?%d" % int(g[0]))
    content2 = h.read()
    res = re.findall("(AVOID|NEUTRAL|BENEFICIAL)", content2)    
    res = ",".join(res)
    line = g[0] + "," + g[1] + "," + res
    fout.write(line)
    fout.write("\n")
    fout.flush()        
    
fout.close()

