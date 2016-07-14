# -*- coding: utf-8 -*-
from Pymacs import lisp
import feedparser, sys, codecs
import re, time, os
        
interactions = {}


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

    lisp.switch_to_buffer("*news*")
    lisp.kill_buffer(lisp.get_buffer("*news*"))
    lisp.switch_to_buffer_other_window("*news*")
    for feed in feeds:
        lisp.insert("\n")
        lisp.insert("## " + feed[0])
        lisp.insert("\n\n")
        d = feedparser.parse(feed[1])
        for post in d.entries:
            lisp.insert("[[%s][%s]]" % (post.link,repr(post.title)))
            lisp.insert("\n")
            
    lisp.org_mode()
            
interactions[show] = ''
