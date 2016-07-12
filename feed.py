from urllib import FancyURLopener
import feedparser

def get_feeds():

     feeds = [("Reuters",'http://feeds.reuters.com/reuters/topNews'),
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
