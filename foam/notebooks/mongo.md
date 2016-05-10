
# Test notebook

```python
from pymongo import MongoClient
import pandas as pd
connection = MongoClient()
db = connection.fakedb
```


```python
q = {"$query" :{"_id.sym": "GOOG", "_id.dt": 20160205} }
res = list(db.tickers.find(q).limit(1))[0]
print res
```

```text
{u'a': 683.570007, u'c': 683.570007, u'h': 703.98999, u'l': 680.150024, u'o': 703.869995, u'v': 5070000.0, u'_id': {u'dt': 20160205, u'sym': u'GOOG'}}
```


```python
import sys; sys.path.append('..')
import foam
df = foam.get_multi(['SPY'])
df.SPY.plot()
plt.savefig('mongo_01.png')
```


```python
import foam
df1 = foam.get("MCD")
df2 = foam.get_multi(['MCD','CMG'])
```


```python
q = {"$query" :{"_id.sym": 'DDD'},"$orderby":{"_id.dt" : -1}}
ts = db.tickers.find(q).limit(1)
last_date_in_db = int(ts[0]['_id']['dt'])
print last_date_in_db
```

```text
20160125
```

```python
q = {"$query" :{"_id.sym": 'DDD'} }
ts = list(db.tickers.find(q).sort({"_id.dt": -1}).limit(1))
```

```python
#q = { "$query" : {"_id.sym": "DDD" }, "$orderby": {"_id.dt" : -1} }
q = { "$query" : {"_id.sym": "DDD" } }
#tmp = list(db.tickers.find( q ).sort("{$natural:-1}").limit(1))
tmp = list(db.tickers.find( q ).sort({"_id.dt":-1}).limit(1))
print tmp
```

```text
[{u'a': 5.08333, u'c': 15.249989999999999, u'h': 15.60999, u'l': 14.67999, 
u'o': 15.06, u'v': 231900.0, u'_id': {u'dt': 20080325, u'sym': u'DDD'}}]
```


```python
print db.tickers.count()
db.tickers.remove({"_id.sym": "AHP", "_id.dt": 20160205 })
print db.tickers.count()
```

```text
14037032
14037032
```

```python
#ts = db.tickers.find( {"_id.sym": "DDD", "_id.dt": {"$gt" : 20070101 }} ).limit(10)
ts = db.tickers.find( {"_id.sym": "OFDP/FUTURE_SI1", "_id.dt": {"$gt" : 20070101 }} ).limit(10)
for t in ts: print t
```

```text
{u'Volume': 3486.0, u'Open Interest': 60681.0, u'High': 13.285, u'Settle': 13.27, u'Low': 12.965, u'_id': {u'dt': 20070102, u'sym': u'OFDP/FUTURE_SI1'}, u'Open': 12.965}
{u'Volume': 289.0, u'Open Interest': 331.0, u'High': 12.555, u'Settle': 12.555, u'Low': 12.555, u'_id': {u'dt': 20070103, u'sym': u'OFDP/FUTURE_SI1'}, u'Open': 12.555}
{u'Volume': 48.0, u'Open Interest': 88.0, u'High': 12.73, u'Settle': 12.73, u'Low': 12.73, u'_id': {u'dt': 20070104, u'sym': u'OFDP/FUTURE_SI1'}, u'Open': 12.73}
{u'Volume': 32.0, u'Open Interest': 115.0, u'High': 12.13, u'Settle': 12.13, u'Low': 12.13, u'_id': {u'dt': 20070105, u'sym': u'OFDP/FUTURE_SI1'}, u'Open': 12.13}
{u'Volume': 116.0, u'Open Interest': 116.0, u'High': 12.26, u'Settle': 12.26, u'Low': 12.25, u'_id': {u'dt': 20070108, u'sym': u'OFDP/FUTURE_SI1'}, u'Open': 12.25}
{u'Volume': 2.0, u'Open Interest': 3.0, u'High': 12.5, u'Settle': 12.5, u'Low': 12.2, u'_id': {u'dt': 20070109, u'sym': u'OFDP/FUTURE_SI1'}, u'Open': 12.2}
{u'Volume': 2.0, u'Open Interest': 3.0, u'High': 12.35, u'Settle': 12.35, u'Low': 12.35, u'_id': {u'dt': 20070110, u'sym': u'OFDP/FUTURE_SI1'}, u'Open': 12.35}
{u'Volume': 80.0, u'Open Interest': 83.0, u'High': 12.375, u'Settle': 12.375, u'Low': 12.375, u'_id': {u'dt': 20070111, u'sym': u'OFDP/FUTURE_SI1'}, u'Open': 12.375}
{u'Volume': 15.0, u'Open Interest': 96.0, u'High': 12.8, u'Settle': 12.8, u'Low': 12.8, u'_id': {u'dt': 20070112, u'sym': u'OFDP/FUTURE_SI1'}, u'Open': 12.8}
{u'Volume': 1369.0, u'Open Interest': 57042.0, u'High': 13.035, u'Settle': 12.98, u'Low': 12.875, u'_id': {u'dt': 20070115, u'sym': u'OFDP/FUTURE_SI1'}, u'Open': 12.905}
```

```python
ts = list(db.earnings.find( {"_id": 20160211 } ))
print ts
```

```text
[{u'c': [[u'acor', u'Before Market Open'], [u'atvi', u'After Market Close'], [u'iots', u'Time Not Supplied'], [u'ads.de', u'Time Not Supplied'], [u'aap', u'06:30 am ET'], [u'aero', u'Time Not Supplied'], [u'afg.ol', u'Time Not Supplied'], [u'ayr', u'Before Market Open'], [u'alu.pa', u'Time Not Supplied'], [u'alr', u'Before Market Open'], [u'alle', u'Before Market Open'], [u'ab', u'Before Market Open'], [u'alny', u'After Market Close'], [u'aad.de', u'Time Not Supplied'], [u'ambr', u'Time Not Supplied'], [u'aig', u'After Market Close'], [u'apgi', u'Time Not Supplied'], [u'amkr', u'After Market Close'], [u'ahh', u'Before Market Open'], [u'amnf', u'Time Not Supplied'], [u'amnf', u'Time Not Supplied'], [u'ashm.l', u'Before Market Open'], [u'asx.ax', u'Time Not Supplied'], [u'avp', u'Before Market Open'], [u'awdr.ol', u'Before Market Open'], [u'axfo.st', u'Before Market Open'], [u'crg.mi', u'Time Not Supplied'], [u'bar.br', u'01:30 am ET'], [u'bebe', u'After Market Close'], [u'bfg.ax', u'Time Not Supplied'], [u'bng.to', u'After Market Close'], [u'gbf.de', u'01:30 am ET'], [u'binck.as', u'Time Not Supplied'], [u'bcor', u'After Market Close'], [u'nile', u'Before Market Open'], [u'bol.st', u'01:45 am ET'], [u'bwa', u'Before Market Open'], [u'blin', u'Time Not Supplied'], [u'bcov', u'After Market Close'], [u'blmt', u'Time Not Supplied'], [u'bsqr', u'Time Not Supplied'], [u'bts.bk', u'Time Not Supplied'], [u'bg', u'Before Market Open'], [u'bwp.ax', u'Time Not Supplied'], [u'ccg', u'Time Not Supplied'], [u'camt.ta', u'Time Not Supplied'], [u'cf.to', u'After Market Close'], [u'cpla', u'Before Market Open'], [u'cth.l', u'Time Not Supplied'], [u'ctre', u'Time Not Supplied'], [u'cbs', u'After Market Close'], [u'cetx', u'Time Not Supplied'], [u'cetx', u'After Market Close'], [u'cve.to', u'Before Market Open'], [u'kool', u'Time Not Supplied'], [u'cadc', u'Time Not Supplied'], [u'cdi.pa', u'After Market Close'], [u'cix.to', u'Before Market Open'], [u'cty1s.he', u'02:00 am ET'], [u'cst.v', u'Time Not Supplied'], [u'cobz', u'After Market Close'], [u'cce', u'Before Market Open'], [u'coh.ax', u'Time Not Supplied'], [u'cohu', u'After Market Close'], [u'cxp', u'After Market Close'], [u'colm', u'4:00 pm ET'], [u'cdco', u'Time Not Supplied'], [u'cdco', u'Time Not Supplied'], [u'cfms', u'Time Not Supplied'], [u'cor', u'Before Market Open'], [u'cray', u'After Market Close'], [u'cres.ba', u'Time Not Supplied'], [u'cyan', u'Time Not Supplied'], [u'cybr', u'After Market Close'], [u'dva', u'After Market Close'], [u'ddr', u'After Market Close'], [u'dbd', u'Before Market Open'], [u'dno.ol', u'02:00 am ET'], [u'drm.to', u'Time Not Supplied'], [u'dw', u'Before Market Open'], [u'dnb', u'After Market Close'], [u'dysl', u'Time Not Supplied'], [u'edig', u'Time Not Supplied'], [u'elon', u'After Market Close'], [u'eden.pa', u'01:00 am ET'], [u'elli', u'After Market Close'], [u'long', u'Time Not Supplied'], [u'egn', u'After Market Close'], [u'esoa', u'Time Not Supplied'], [u'etm', u'Before Market Open'], [u'esp', u'Time Not Supplied'], [u'eo.pa', u'02:00 am ET'], [u'ffg', u'After Market Close'], [u'feye', u'After Market Close'], [u'faf', u'06:45 am ET'], [u'flir', u'Before Market Open'], [u'fls.co', u'06:00 am ET'], [u'fet', u'After Market Close'], [u'gnca', u'Time Not Supplied'], [u'gxi.de', u'01:30 am ET'], [u'gla1v.he', u'06:00 am ET'], [u'gnc', u'Before Market Open'], [u'gmg.ax', u'Time Not Supplied'], [u'gwo.to', u'Time Not Supplied'], [u'gpi', u'Before Market Open'], [u'grpn', u'After Market Close'], [u'guid', u'After Market Close'], [u'he.to', u'Time Not Supplied'], [u'hdng', u'Time Not Supplied'], [u'he', u'After Market Close'], [u'hgg.l', u'02:00 am ET'], [u'hex.ol', u'01:00 am ET'], [u'huh1v.he', u'01:30 am ET'], [u'hun', u'Before Market Open'], [u'ipwr', u'Time Not Supplied'], [u'nk.pa', u'After Market Close'], [u'imsc', u'Time Not Supplied'], [u'incy', u'07:00 am ET'], [u'iag.to', u'Before Market Open'], [u'infn', u'After Market Close'], [u'inf.l', u'Before Market Open'], [u'ifp.to', u'Time Not Supplied'], [u'ivc', u'Before Market Open'], [u'irsa.ba', u'Before Market Open'], [u'iti', u'Time Not Supplied'], [u'iwg.v', u'Time Not Supplied'], [u'kat.to', u'Time Not Supplied'], [u'060250.kq', u'Time Not Supplied'], [u'k', u'Before Market Open'], [u'king', u'After Market Close'], [u'kkr', u'Before Market Open'], [u'knl', u'After Market Close'], [u'kn', u'After Market Close'], [u'kog.ol', u'Time Not Supplied'], [u'or.pa', u'After Market Close'], [u'lr.pa', u'01:30 am ET'], [u'lc', u'Before Market Open'], [u'liox', u'Time Not Supplied'], [u'logm', u'After Market Close'], [u'lpx', u'Before Market Open'], [u'lpla', u'After Market Close'], [u'lpn.bk', u'Time Not Supplied'], [u'lxft', u'After Market Close'], [u'manu', u'Before Market Open'], [u'mfc.to', u'Before Market Open'], [u'mdf.to', u'Time Not Supplied'], [u'merc', u'After Market Close'], [u'msl.to', u'Time Not Supplied'], [u'meo.de', u'01:30 am ET'], [u'mgr.ax', u'Time Not Supplied'], [u'mztf.ta', u'Time Not Supplied'], [u'tap', u'Before Market Open'], [u'mgi', u'Before Market Open'], [u'type', u'Before Market Open'], [u'mww', u'Before Market Open'], [u'mos', u'Before Market Open'], [u'mpsx', u'After Market Close'], [u'nnn', u'Before Market Open'], [u'nci', u'Before Market Open'], [u'ntwk', u'Before Market Open'], [u'nbix', u'After Market Close'], [u'nr', u'After Market Close'], [u'nice.ta', u'Before Market Open'], [u'nlsn', u'Before Market Open'], [u'nokia.he', u'01:00 am ET'], [u'northm.co', u'Time Not Supplied'], [u'nwe', u'Before Market Open'], [u'nas.ol', u'01:00 am ET'], [u'npro.ol', u'01:00 am ET'], [u'nus', u'After Market Close'], [u'nuva', u'After Market Close'], [u'ozm', u'Before Market Open'], [u'ork.ol', u'01:00 am ET'], [u'out1v.he', u'02:00 am ET'], [u'p', u'After Market Close'], [u'phm.v', u'Time Not Supplied'], [u'pbf', u'Before Market Open'], [u'pbfx', u'Before Market Open'], [u'pdfs', u'After Market Close'], [u'btu', u'Before Market Open'], [u'pag', u'Before Market Open'], [u'pep', u'Before Market Open'], [u'ri.pa', u'01:30 am ET'], [u'pnk', u'Before Market Open'], [u'pjt', u'Before Market Open'], [u'pkc1v.he', u'01:15 am ET'], [u'ptsx', u'Time Not Supplied'], [u'pd.to', u'Before Market Open'], [u'pdex', u'Time Not Supplied'], [u'pub.pa', u'Before Market Open'], [u'qlik', u'After Market Close'], [u'q', u'Before Market Open'], [u'quot', u'After Market Close'], [u'rmr1v.he', u'02:00 am ET'], [u'rec.mi', u'Time Not Supplied'], [u'rsg', u'After Market Close'], [u'rxl.pa', u'01:30 am ET'], [u'rai', u'Before Market Open'], [u'rio.l', u'Before Market Open'], [u'rcky', u'Time Not Supplied'], [u'rovi', u'After Market Close'], [u'sanw', u'Time Not Supplied'], [u'saja', u'Time Not Supplied'], [u'ssn.ax', u'Time Not Supplied'], [u'scss', u'After Market Close'], [u'semc.st', u'Time Not Supplied'], [u'shp.l', u'Time Not Supplied'], [u'5cp.si', u'Time Not Supplied'], [u'szmk', u'Time Not Supplied'], [u'sma.to', u'After Market Close'], [u'gle.pa', u'01:00 am ET'], [u'local.pa', u'Before Market Open'], [u'sofo', u'Time Not Supplied'], [u'son', u'Before Market Open'], [u'spdc', u'Time Not Supplied'], [u'sqd.v', u'Time Not Supplied'], [u'ssnc', u'After Market Close'], [u'stc', u'06:15 am ET'], [u'sum', u'Before Market Open'], [u'sun.ax', u'Time Not Supplied'], [u'syz.v', u'Time Not Supplied'], [u'syn.v', u'Time Not Supplied'], [u'symx', u'After Market Close'], [u'tbl.to', u'Time Not Supplied'], [u't.to', u'Before Market Open'], [u'teri3.sa', u'Time Not Supplied'], [u'teva.ta', u'07:00 am ET'], [u'tgh', u'09:00 am ET'], [u'tri.to', u'Before Market Open'], [u'time', u'Before Market Open'], [u'tmd.to', u'Time Not Supplied'], [u'x.to', u'After Market Close'], [u'top.co', u'06:00 am ET'], [u'fp.pa', u'02:00 am ET'], [u'towr', u'Time Not Supplied'], [u'rnw.to', u'Time Not Supplied'], [u'trp.to', u'Before Market Open'], [u'tfi.to', u'After Market Close'], [u'tcl.ax', u'Time Not Supplied'], [u'tzoo', u'Before Market Open'], [u'ths', u'Before Market Open'], [u'trip', u'Time Not Supplied'], [u'trup', u'After Market Close'], [u'tssi', u'Before Market Open'], [u'ttkom.is', u'After Market Close'], [u'vakbn.is', u'Time Not Supplied'], [u'ulbi', u'Before Market Open'], [u'uns.to', u'Time Not Supplied'], [u'unitech.ns', u'Time Not Supplied'], [u'vrns', u'After Market Close'], [u'woof', u'08:00 am ET'], [u'vcm.to', u'Time Not Supplied'], [u'vei.ol', u'01:00 am ET'], [u'vrsn', u'After Market Close'], [u'vah.ax', u'Time Not Supplied'], [u'vsto', u'Before Market Open'], [u'vcra', u'Time Not Supplied'], [u'vg', u'Before Market Open'], [u'gra', u'06:00 am ET'], [u'wbc', u'Before Market Open'], [u'wso', u'Before Market Open'], [u'web', u'After Market Close'], [u'wft.to', u'After Market Close'], [u'wwav', u'Before Market Open'], [u'int', u'Before Market Open'], [u'wwe', u'Before Market Open'], [u'wynn', u'After Market Close'], [u'xplr', u'Time Not Supplied'], [u'yoo.v', u'Time Not Supplied'], [u'yar.ol', u'02:00 am ET'], [u'y.to', u'Time Not Supplied'], [u'zg', u'After Market Close'], [u'zurn.vx', u'12:45 am ET']], u'_id': 20160211}]
```

```python
print db.earnings.count()
db.earnings.remove({"_id": 20160210 })
print db.earnings.count()
```

```text
2635
2635
```

```python
import sys; sys.path.append('..')
import foam
res = foam.get_hft("GOOG", 20160211)
print res.head()
```

```text
           close      high       low      open volume
153402  676.9150  679.0000  674.5500  675.2900      0
153905  674.8100  677.0620  673.5700  676.2000  73300
154407  678.2600  678.9700  673.4520  675.2140  46500
154903  674.1800  678.5000  674.1800  677.9500  46800
155400  675.9400  676.1790  672.9000  674.1100  43500
```
























