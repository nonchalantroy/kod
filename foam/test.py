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

def fake_download_2(contract,start,end):
    return load_data(contract, "data_2")

def fake_download_3(contract,start,end):
    return load_data(contract, "data_3")

def fake_today_1():
    return datetime.datetime(2016, 5, 1) 

def fake_today_2():
    return datetime.datetime(1984, 1, 1) 

def fake_today_3():
    return datetime.datetime(1983, 7, 26) 

def fake_today_4():
    return datetime.datetime(1983, 7, 27) 

def init():
    c = MongoClient()
    c[testdb].tickers.drop()
    return c[testdb]

def test_simple():
    db = init()
    futures.download_data(downloader=fake_download_1,today=fake_today_1,
                          db=testdb, years=(1984,1985))
    res = futures.get(market="CME", sym="CL", month="F", year=1984, dt=19831205, db=testdb)
    assert res[0]['oi'] == 5027.0
    res = futures.get(market="CME", sym="CL", month="G", year=1984, dt=19830624, db=testdb)
    assert res[0]['oi'] == 5.0
    res = futures.last_contract("CL","CME", db)
    assert res[0]['_id']['month'] == 'G'

    res = futures.existing_nonexpired_contracts("CL","CME", db,fake_today_1())
    assert len(res) == 0
    res = futures.existing_nonexpired_contracts("CL","CME", db,fake_today_2())
    assert len(res) > 0

def test_incremental():
    db = init()
    futures.download_data(downloader=fake_download_2,today=fake_today_3,
                          db=testdb, years=(1984,1985))
    assert futures.last_date_in_contract("CL","CME","F", 1984, db) == 19830726
    futures.download_data(downloader=fake_download_3,today=fake_today_4,
                          db=testdb, years=(1984,1985))
    
if __name__ == "__main__": 
    #test_simple()
    test_incremental()
    
