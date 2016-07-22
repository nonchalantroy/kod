# -*- coding: utf-8 -*-
"""
Python replacement for newsbeuter, an RSS based news reader. 
"""
import feedparser, sys, codecs
import re, time, os

sys.stdout = codecs.getwriter('utf8')(sys.stdout)
sys.stderr = codecs.getwriter('utf8')(sys.stderr)

def show():

    feeds = [("The Guardian","http://www.theguardian.com/world/rss",10),
             ("Diken","http://www.diken.com.tr/feed/",-1),
             ("Cumhuriyet","http://www.cumhuriyet.com.tr/rss/son_dakika.xml",10),
             (u"Hürriyet", "http://www.hurriyet.com.tr/rss/gundem",10),
             ("Al-Jazeera","http://aljazeera.com.tr/rss.xml",-1),
             (u"Açık Gazete","https://www.acikgazete.com/feed",-1),
             ("T24","https://twitrss.me/twitter_user_to_rss/?user=t24comtr",-1),
             ("Reuters (Top News)",'http://feeds.reuters.com/reuters/topNews',-1),
             ("Reuters (UK World)",'http://feeds.reuters.com/reuters/UKWorldNews',-1),
             ("Reuters (World)",'http://feeds.reuters.com/reuters/worldNews',-1),
             ("Reuters (Business)", "http://feeds.reuters.com/reuters/businessNews",-1),
             ("Bloomberg","https://twitrss.me/twitter_user_to_rss/?user=business",-1),
             ('Huffington Post','http://www.huffingtonpost.com/feeds/verticals/world/index.xml',-1),
             ('BBC','http://newsrss.bbc.co.uk/rss/newsonline_world_edition/front_page/rss.xml',20),
             ("Sputnik News","http://tr.sputniknews.com/export/rss2/archive/index.xml",15),
             ("EB","https://twitrss.me/twitter_user_to_rss/?user=ebabahan",-1),
             ("Fuat Avni","https://twitrss.me/twitter_user_to_rss/?user=fuatavni_f",-1)
    ]

    for name,url,lim in feeds:
        print("\n")
        print("## " + name)
        print("\n")
        d = feedparser.parse(url)
        for i,post in enumerate(d.entries):
            link = post.link; title = post.title
            if len(re.findall(r"Erdo.an", title)) > 0: continue
            print("[[%s][%s]]" % (link,unicode(title)))
            if lim > 0 and i==lim: break

show()            
