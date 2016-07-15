# -*- coding: utf-8 -*-
import feedparser, sys, codecs
import re, time, os

feeds = [("Reuters (Top News)",'http://feeds.reuters.com/reuters/topNews'),
         ("Reuters (World)",'http://feeds.reuters.com/reuters/worldNews'),
         ("Reuters (Business)", "http://feeds.reuters.com/reuters/businessNews"),
         ("Reuters (Economy)", "http://feeds.reuters.com/news/economy"),
         ('BBC','http://newsrss.bbc.co.uk/rss/newsonline_world_edition/front_page/rss.xml'),
         ('Huffington Post','http://www.huffingtonpost.com/feeds/verticals/world/index.xml'),
         ("The Guardian","http://www.theguardian.com/world/rss")
]

     
def show():
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)
    sys.stderr = codecs.getwriter('utf8')(sys.stderr)
    
    for feed in feeds:
        print("\n")
        print("## " + feed[0])
        print("\n")
        d = feedparser.parse(feed[1])
        for post in d.entries:
            print("[[%s][%s]]" % (post.link,repr(post.title)))

show()            
