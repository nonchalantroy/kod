# -*- coding: utf-8 -*-
"""
Python replacement for newsbeuter, an RSS based news reader. 
"""
import feedparser, sys, codecs
import re, time, os

sys.stdout = codecs.getwriter('utf8')(sys.stdout)
sys.stderr = codecs.getwriter('utf8')(sys.stderr)

import base64
def encode(key, clear):
    enc = []
    for i in range(len(clear)):
        key_c = key[i % len(key)]
        enc_c = chr((ord(clear[i]) + ord(key_c)) % 256)
        enc.append(enc_c)
    return base64.urlsafe_b64encode("".join(enc))

def decode(key, enc):
    dec = []
    enc = base64.urlsafe_b64decode(enc)
    for i in range(len(enc)):
        key_c = key[i % len(key)]
        dec_c = chr((256 + ord(enc[i]) - ord(key_c)) % 256)
        dec.append(dec_c)
    return "".join(dec)

def show():

    feeds = [
        ("Reuters (UK World)",'http://feeds.reuters.com/reuters/UKWorldNews',-1),
        ("The Guardian","http://www.theguardian.com/world/rss",10),
        ("Diken","http://www.diken.com.tr/feed/",-1),
        ("Cumhuriyet","http://www.cumhuriyet.com.tr/rss/son_dakika.xml",10),
        ("Al-Jazeera","http://aljazeera.com.tr/rss.xml",-1),
        ("T24","https://twitrss.me/twitter_user_to_rss/?user=t24comtr",-1),
        ("Reuters (Top News)",'http://feeds.reuters.com/reuters/topNews',-1),
        ("Reuters (World)",'http://feeds.reuters.com/reuters/worldNews',-1),
        ("Independent, The", "http://www.independent.co.uk/news/world/rss", 10),
        ("Reuters (Business)", "http://feeds.reuters.com/reuters/businessNews",-1),        
        ("Bloomberg","https://twitrss.me/twitter_user_to_rss/?user=business",-1),
        ('Huffington Post','http://www.huffingtonpost.com/feeds/verticals/world/index.xml',-1),
        ('BBC','http://newsrss.bbc.co.uk/rss/newsonline_world_edition/front_page/rss.xml',20),
        ("Sputnik News","http://tr.sputniknews.com/export/rss2/archive/index.xml",15),
        ("The Atlantic", "http://www.theatlantic.com/feed/all/",-1),
        ("Dunya Finans","http://www.dunya.com/service/rss.php",10),
        ("EB",decode('1234', "maanpKRsYmOlqZyoo6WmYp6XYqiom6eolqSSqaSXpZOloZKmpKVic6almKZul5WVk5OblZ8="),-1),
        (u"Açık Gazete","https://www.acikgazete.com/feed",-1),
        ("O",decode('1234', "maanpGthYqOVk6eqX5WioWCkpqdfopuk"),10),
        ("A", decode('1234', "maanpGthYpWfmJSekqCmYpShoGOjpaY="),10)
    ]

    #feeds = [   ]

    for name,url,lim in feeds:
        print("\n")
        print("## " + name)
        print("\n")
        d = feedparser.parse(url)
        for i,post in enumerate(d.entries):
            if lim > 0 and i==int(lim): break
            link = post.link; title = post.title
            if len(re.findall(r"Erdo.an", title, re.IGNORECASE)) > 0: continue
            if len(re.findall(r"top.u k", title, re.IGNORECASE)) > 0: continue
            if len(re.findall(r"Engin Ard", title, re.IGNORECASE)) > 0: continue
            if len(re.findall(r" .ld.rd.", title, re.IGNORECASE)) > 0: continue
            print("[[%s][%s]]" % (link,unicode(title)))
show()
