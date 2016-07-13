from urllib import FancyURLopener
import feedparser

def get_feeds():

     feeds = [("Reuters (Top News)",'http://feeds.reuters.com/reuters/topNews'),
              ("Reuters (World)",'http://feeds.reuters.com/reuters/worldNews'),
              ("Reuters (Business)", "http://feeds.reuters.com/reuters/businessNews"),
              ("Reuters (Economy)", "http://feeds.reuters.com/news/economy"),
              ('BBC','http://newsrss.bbc.co.uk/rss/newsonline_world_edition/front_page/rss.xml'),
              ('Huffington Post','http://www.huffingtonpost.com/feeds/verticals/world/index.xml')
              
    ]

     for feed in feeds:
        print "\n#", feed[0], "\n"
        d = feedparser.parse(feed[1])
        for post in d.entries:
            if ".mp3" in post.link: continue
            print "[[%s][%s]]" % (post.link, post.title)

get_feeds()
