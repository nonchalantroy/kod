import os, fut2, pandas as pd, datetime
from pymongo import MongoClient

def fake_download_1(contract,start,end):
    f = contract.replace("/","-")
    f = "./test/%s.csv" % f
    if not os.path.isfile(f): raise Quandl.Quandl.DatasetNotFound()
    df = pd.read_csv(f)
    df = df.set_index("Date")
    return df

def fake_today_1():
    return datetime.datetime(2016, 5, 1) 

def init():
    c = MongoClient()
    c['futtest'].drop_collection('ticker')

def test_simple():
    fut2.download_data(downloader=fake_download_1,today=fake_today_1)
    
if __name__ == "__main__": 
    init()
    test_simple()

    
