'''
Get Blood Type Diet ingredient information from the Dadamo's site
'''
import re, urllib, urllib2
import logging, sys, os, glob
from datetime import datetime
from datetime import timedelta
from urllib import FancyURLopener

def extract(content):
    res = re.findall("\s(AVOID|NEUTRAL|BENEFICIAL)", content)
    res = ";".join(res)
    return res
    
    
if __name__ == "__main__": 
 
    fout = open("%s/food.dat" % os.environ['TEMP'],"w")
    fout.write("Dadamo_Site_Id;Food;A_S;A_NS;B_S;B_NS;AB_S;AB_NS;O_S;O_NS\n")
    files = glob.glob("c:/Users/burak/www.dadamo.com/typebase4/*.pl*")
    
    for g in files:
        tmp = re.findall("pl@(\d*)", g)        
        id = tmp[0]
        content = open(g).read()
        tmp = re.findall("<title>.*?:\s(.*?)</title>", content)
        name = tmp[0]
        line = extract(content)
        print id, name
        line = id + ";" + name + ";" + line
        fout.write(line)
        fout.write("\n")
        fout.flush()

    fout.close()

