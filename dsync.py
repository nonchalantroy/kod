import os, sys, shutil, rsync, re

if len(sys.argv) < 2: print "Usage dsync.py [letter]"; exit()

if sys.argv[1] == "d":
    os.system("python rsync.py c:\\Users\\burak\\Documents\\Dropbox D:\\archive\\Dropbox --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\classnotes D:\\archive\\classnotes --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\kod D:\\archive\\kod --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\book_idx D:\\archive\\book_idx --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\quant_at D:\\archive\\quant_at --delete")

if sys.argv[1] == "e":
    os.system("python rsync.py c:\\Users\\burak\\Documents\\classnotes E:\\classnotes --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\kod E:\\kod --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\book_idx E:\\book_idx --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\quant_at E:\\quant_at --delete")
    
if sys.argv[1] == "ed":
    os.system("python rsync.py E:\\kitaplar D:\\archive\\kitaplar")
    
if sys.argv[1] == "systematic":
    #os.system("python rsync.py c:\\Users\\burak\\Downloads\\pysystemtrade c:\\Users\\burak\\Documents\\kod\\books\\Systematic_Trading\\pysystemtrade")
    #rsync.deleteDir("c:\\Users\\burak\\Documents\\kod\\books\\Systematic_Trading\\pysystemtrade\\.git")
    rsync.purge("c:/Users/burak/Documents/kod/books/Systematic_Trading/pysystemtrade","\.pck")
    
