# -*- coding: utf-8 -*-
"""
Python replacement for newsbeuter, an RSS based news reader. 
"""
import feedparser, sys, codecs
import re, time, os

feeds = [
    ("Reuters (Top News)",'http://feeds.reuters.com/reuters/topNews'),
    ("Reuters (World)",'http://feeds.reuters.com/reuters/worldNews'),
    ("Reuters (Business)", "http://feeds.reuters.com/reuters/businessNews"),
    ("Reuters (Economy)", "http://feeds.reuters.com/news/economy"),
    ('BBC','http://newsrss.bbc.co.uk/rss/newsonline_world_edition/front_page/rss.xml'),
    ('Huffington Post','http://www.huffingtonpost.com/feeds/verticals/world/index.xml'),
    ("The Guardian","http://www.theguardian.com/world/rss"),
    ("Cumhuriyet","http://www.cumhuriyet.com.tr/rss/son_dakika.xml"),
    ("Hurriyet", "http://www.hurriyet.com.tr/rss/gundem"),
    ("Al-Jazeera","http://aljazeera.com.tr/rss.xml"),
    ("Acik Gazete","https://www.acikgazete.com/feed/"),
    ("Diken","http://www.diken.com.tr/feed/"),
    ("T24","https://twitrss.me/twitter_user_to_rss/?user=t24comtr")
]

     
def show():
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)
    sys.stderr = codecs.getwriter('utf8')(sys.stderr)

    if len(sys.argv) == 2 and sys.argv[1] == "x":
        feeds.append(("Fuat Avni","https://twitrss.me/twitter_user_to_rss/?user=fuatavni_f"))
    
    for feed in feeds:
        print("\n")
        print("## " + feed[0])
        print("\n")
        d = feedparser.parse(feed[1])
        for post in d.entries:
            link = post.link; title = post.title
            if len(re.findall(r"Erdo.an", title)) > 0: continue
            print("[[%s][%s]]" % (link,title))


show()            
