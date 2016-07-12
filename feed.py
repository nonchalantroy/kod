from urllib import FancyURLopener
import feedparser

def get_feeds():

    feeds = ['http://feeds.reuters.com/reuters/topNews',
             'http://newsrss.bbc.co.uk/rss/newsonline_world_edition/front_page/rss.xml',
             'http://www.huffingtonpost.com/feeds/verticals/world/index.xml'
    ]

    for feed in feeds:
        d = feedparser.parse(feed)
        print "==", feed
        for post in d.entries:
            if ".mp3" in post.link: continue
            p = post.title
            p = p.replace("Bloomberg Advantage: ","")
            p = p.replace("The Bloomberg Advantage: ","")
            print "[[%s][%s]]" % (post.link, p.decode("utf-8"))

get_feeds()
