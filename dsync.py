import os, sys, shutil, rsync, re

if len(sys.argv) < 2: print "Usage dsync.py [letter]"; exit()

if sys.argv[1] == "d":
    os.system("python rsync.py c:\\Users\\burak\\Documents\\Dropbox e:\\archive\\Dropbox --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\classnotes e:\\archive\\classnotes --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\kod e:\\archive\\kod --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\book_idx e:\\archive\\book_idx --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\quant_at e:\\archive\\quant_at --delete")

if sys.argv[1] == "e":
    os.system("python rsync.py c:\\Users\\burak\\Documents\\classnotes d:\\classnotes --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\kod d:\\kod --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\book_idx d:\\book_idx --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\quant_at d:\\quant_at --delete")
    
if sys.argv[1] == "ed":
    os.system("python rsync.py E:\\kitaplar D:\\archive\\kitaplar")
    
if sys.argv[1] == "systematic":
    os.system("python rsync.py c:\\Users\\burak\\Downloads\\pysystemtrade c:\\Users\\burak\\Documents\\kod\\books\\Systematic_Trading\\pysystemtrade")
    rsync.deleteDir("c:\\Users\\burak\\Documents\\kod\\books\\Systematic_Trading\\pysystemtrade\\.git")
    rsync.purge("c:/Users/burak/Documents/kod/books/Systematic_Trading/pysystemtrade","\.pck")
    
