# -*- coding: utf-8 -*-
from urllib import FancyURLopener
import feedparser, sys, codecs

sys.stdout = codecs.getwriter('utf8')(sys.stdout)
sys.stderr = codecs.getwriter('utf8')(sys.stderr)

def get_feeds():

     feeds = [("Reuters (Top News)",'http://feeds.reuters.com/reuters/topNews'),
              ("Reuters (World)",'http://feeds.reuters.com/reuters/worldNews'),
              ("Reuters (Business)", "http://feeds.reuters.com/reuters/businessNews"),
              ("Reuters (Economy)", "http://feeds.reuters.com/news/economy"),
              ('BBC','http://newsrss.bbc.co.uk/rss/newsonline_world_edition/front_page/rss.xml'),
              ('Huffington Post','http://www.huffingtonpost.com/feeds/verticals/world/index.xml')              
    ]

                   
     sys.stdout = codecs.getwriter('utf8')(sys.stdout)
     sys.stderr = codecs.getwriter('utf8')(sys.stderr)
     
     for feed in feeds:
        print "\n#", feed[0], "\n"
        d = feedparser.parse(feed[1])
        for post in d.entries:
            print "[[%s][%s]]" % (post.link,post.title)

get_feeds()
