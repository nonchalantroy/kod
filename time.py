# Get time stamp information from any server with a Web Server running
# using HTTP protocol. Comes in handy if I connect to a Wifi
# connection that redirects me to a login webpage, even I do not have
# the credentials to get a real Internet connection, I can still get
# the time information from that server, through Wifi.
#
# Usage: time.py [IP ADDRESS]
#
# Ip address could be the Default Gateway address (see the output of ifconfig
# or ipconfig, or route print, or nslookup. 
#
import urllib2, sys, re
req = urllib2.Request('http://%s' % sys.argv[1])
res = urllib2.urlopen(req)
res.close()
content = str(res.info())
res = re.findall("Date: (.*?)\r*\n", content, re.DOTALL)
print "Current Time is", res[0]
