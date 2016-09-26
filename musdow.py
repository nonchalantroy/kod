'''
Downloads 10 songs from a known site
'''
import re, requests, random, os
from urllib import urlretrieve

url = "http://www.sos-jukebox.net/music"
response = requests.get(url)
response_body = response.content
regex = "<li><a href=\"(.*?)\">.*?</a></li>"
songs = re.findall(regex, response_body, re.DOTALL)
print songs[:4]

for i in range(10):
    idx = random.choice(range(len(songs)))
    if ".mp3" in songs[idx]: os.system("wget %s/%s" % (url,songs[idx]))
    
