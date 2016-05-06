# coding=utf-8
from pymongo import MongoClient
import urllib, datetime, logging, sys
import re, os, foam, pandas as pd
from pandas_datareader import data, wb

def web_earnings(day):
    url = "http://biz.yahoo.com/research/earncal/%d.html" % day
    h = urllib.urlopen(url)
    content = h.read()
    regex = "q\?s=(.*?)\".*?<small>(.*?)</small>"
    res = re.findall(regex, content, re.DOTALL)
    return res

def get_earnings():
    
    connection = MongoClient()
    db = connection.foam
    earnings = db.earnings

    curr = foam.get_beginning_of_time()

    last_date_in_db = list(db.earnings.find().sort("_id", -1).limit(1))
    if len(last_date_in_db) > 0:
        curr = pd.to_datetime(str(last_date_in_db[0]['_id']), format='%Y%m%d')
        logging.debug('last record %s', repr(curr))

    today,today_i = foam.get_today()
    logging.debug('today %s', repr(today))

    while True:
        curr_i = int(curr.strftime('%Y%m%d') )
        if curr_i <= today_i:
            if curr.weekday() < 5:
                logging.debug('downloading %s', repr(curr))
                earn = web_earnings(curr_i)
                new_row = {"_id": curr_i, "c": earn}
                logging.debug(curr_i)
                earnings.save(new_row)            

            curr = curr + datetime.timedelta(days=1)
        else:
            break
            
if __name__ == "__main__": 
    logging.basicConfig(filename='/tmp/earnings.log',level=logging.DEBUG)
    if len(sys.argv) == 2:
        res = web_earnings(int(sys.argv[1]))
        print res
    elif len(sys.argv) == 1: 
        get_earnings()
    
    
