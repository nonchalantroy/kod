
from pymongo import MongoClient
import fut2

def init():
    c = MongoClient()
    c['futtest'].drop_collection('ticker')

def test_simple():
    fut2.download_data(downloader=fut2.fake_download)
    
if __name__ == "__main__": 
    init()
    test_simple()

    
