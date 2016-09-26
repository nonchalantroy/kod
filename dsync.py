import os, sys, shutil, rsync, re

if len(sys.argv) < 2: print "Usage dsync.py [letter]"; exit()

if sys.argv[1] == "hd":
    os.system("python rsync.py c:\\Users\\burak\\Documents\\quant_at e:\\archive\\quant_at --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\Dropbox e:\\archive\\Dropbox --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\classnotes e:\\archive\\classnotes --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\kod e:\\archive\\kod --delete")

if sys.argv[1] == "hd2":
    os.system("python rsync.py c:\\Users\\burak\\Documents\\quant_at d:\\archive\\quant_at --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\Dropbox d:\\archive\\Dropbox --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\classnotes d:\\archive\\classnotes --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\kod d:\\archive\\kod --delete")

if sys.argv[1] == "flash":
    os.system("python rsync.py c:\\Users\\burak\\Documents\\Dropbox\\resmi\\newbusiness d:\\newbusiness")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\quant_at d:\\quant_at --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\classnotes d:\\classnotes --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\kod d:\\kod --delete")

if sys.argv[1] == "kitaplar":
    os.system("python rsync.py E:\\archive\\kitaplar D:\\kitaplar")
    
if sys.argv[1] == "ed":
    os.system("python rsync.py E:\\archive\\kitaplar D:\\archive\\kitaplar --delete")
    
if sys.argv[1] == "systematic":
    os.system("python rsync.py c:\\Users\\burak\\Downloads\\pysystemtrade c:\\Users\\burak\\Documents\\kod\\books\\Systematic_Trading\\pysystemtrade")
    rsync.deleteDir("c:\\Users\\burak\\Documents\\kod\\books\\Systematic_Trading\\pysystemtrade\\.git")
    rsync.purge("c:/Users/burak/Documents/kod/books/Systematic_Trading/pysystemtrade","\.pck")
    
