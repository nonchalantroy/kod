import os, futures, pandas as pd, datetime
from pymongo import MongoClient
import Quandl

testdb = "fakedb"

def load_data(contract,subdir):
    f = contract.replace("/","-")
    f = "./test/%s/%s.csv" % (subdir,f)
    if not os.path.isfile(f): raise Quandl.Quandl.DatasetNotFound()
    df = pd.read_csv(f)
    df = df.set_index("Date")
    return df
    
def fake_download_1(contract,start,end):
    return load_data(contract, "data_1")

def fake_today_1():
    return datetime.datetime(2016, 5, 1) 

def init():
    c = MongoClient()
    c[testdb].drop_collection('ticker')

def test_simple():
    futures.download_data(downloader=fake_download_1,today=fake_today_1,db=testdb)
    res = futures.get(market="CME", sym="CL", month="F", year=1984, dt=19831205, db=testdb)
    assert res[0]['oi'] == 5027.0
    
if __name__ == "__main__": 
    init()
    test_simple()

    
