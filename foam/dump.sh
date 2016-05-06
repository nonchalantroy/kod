cd $HOME/Dropbox/Public/data
mongodump
cd dump/foam
gzip -f tickers.bson
gzip -f earnings.bson
split -n 6 tickers.bson.gz tickers-
