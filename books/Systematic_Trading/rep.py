# replace.py - search and replace through python, run it like this
# sh -c 'find ./test -type f -name "*" -exec python replace.py {} \;'
# the code to change stuff goes below
import os, re, sys
filename = sys.argv[1]
content = open(filename).read()
fout = open(filename,"w")

# you can insert stuff
# fout.write("import bla\n")

# or replace stuff through regex
content = content.replace("log.terse(","print(__file__ + \" \" + ")
content = content.replace("log.msg(","print(__file__ + \" \" + ")
content = content.replace("this_stage.print","print")
content = content.replace("self.print","print")
content = content.replace("rules_stage.print","print")
content = re.sub("print(.*?),\n(.*?)\n","print\\1)\n",content,re.DOTALL)
content = content.replace(", instrument_code=instrument_code","")

fout.write(content)
fout.close()
