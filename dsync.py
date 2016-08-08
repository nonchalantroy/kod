import os, sys, shutil, rsync, re

if len(sys.argv) < 2: print "Usage dsync.py [letter]"; exit()

if sys.argv[1] == "hd":
    os.system("python rsync.py c:\\Users\\burak\\Documents\\quant_at d:\\archive\\quant_at --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\Dropbox d:\\archive\\Dropbox --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\classnotes d:\\archive\\classnotes --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\kod d:\\archive\\kod --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\book_idx d:\\archive\\book_idx --delete")

if sys.argv[1] == "flash":
    os.system("python rsync.py c:\\Users\\burak\\Documents\\quant_at e:\\quant_at --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\classnotes e:\\classnotes --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\kod e:\\kod --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\book_idx e:\\book_idx --delete")
    
if sys.argv[1] == "de":
    os.system("python rsync.py D:\\kitaplar E:\\archive\\kitaplar")
    
if sys.argv[1] == "systematic":
    os.system("python rsync.py c:\\Users\\burak\\Downloads\\pysystemtrade c:\\Users\\burak\\Documents\\kod\\books\\Systematic_Trading\\pysystemtrade")
    rsync.deleteDir("c:\\Users\\burak\\Documents\\kod\\books\\Systematic_Trading\\pysystemtrade\\.git")
    rsync.purge("c:/Users/burak/Documents/kod/books/Systematic_Trading/pysystemtrade","\.pck")
    
