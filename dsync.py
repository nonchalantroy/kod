import os, sys, shutil, rsync

if len(sys.argv) < 2: print "Usage dsync.py [letter]"; exit()

if sys.argv[1] == "a":
    os.system("python rsync.py c:\\Users\\burak\\Documents\\Dropbox D:\\archive\\Dropbox --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\classnotes D:\\archive\\classnotes --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\kod D:\\archive\\kod")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\book_idx D:\\archive\\book_idx --delete")
    
if sys.argv[1] == "c":
    os.system("python rsync.py c:\\Users\\burak\\Documents\\kod E:\\kod  --delete")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\classnotes E:\\classnotes")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\book_idx E:\\book_idx")
    os.system("python rsync.py c:\\Users\\burak\\Documents\\mon E:\\mon")

if sys.argv[1] == "ca":
    os.system("python rsync.py E:\\kitaplar D:\\archive\\kitaplar")
    
if sys.argv[1] == "pst":
    os.system("python rsync.py c:\\Users\\burak\\Downloads\\pysystemtrade c:\\Users\\burak\\Documents\\kod\\books\\Systematic_Trading\\pysystemtrade")
    rsync.deleteDir("c:\\Users\\burak\\Documents\\kod\\books\\Systematic_Trading\\pysystemtrade\\.git")
