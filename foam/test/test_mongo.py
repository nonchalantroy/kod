
from pymongo import MongoClient

connection = MongoClient()

db = connection.foam

import datetime

ticker = {
    "_id" : {
        "sym" : "MSFT",
        "dt" : "2014-07-15"
    },
    "o" : 42.33,
    "h" : 42.47,
    "l" : 42.03,
    "c" : 42.45,
    "v" : 28748700,
    "a" : 42.45
}

tickers = db.tickers

print tickers.insert(ticker)

print tickers.find_one()

