#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 19:14:50 2016
 
 USE MIT image memorability API to get memscores for each logo

@author: sheng
"""
#import json
#import simplejson as json
#from nltk.stem import WordNetLemmatizer
#import nltk
import pandas as pd
import numpy as np
import glob
import os
import io  
import urllib
import pickle
import string
import codecs
import time
import random
#from collections import defaultdict
#import matplotlib
#matplotlib.use('Agg')
#from matplotlib import pyplot as plt
#from nltk.tokenize import PunktSentenceTokenizer
#from senti_classifier import senti_classifier
#from textblob import TextBlob
from bs4 import BeautifulSoup
from urllib2 import Request, urlopen, URLError
import requests
import re
import sys
#import scipy as sp
#import scipy.stats
#from collections import Counter
#import pickle
#from compiler.ast import flatten
#from joblib import Parallel, delayed
#import multiprocessing
#num_cores = multiprocessing.cpu_count()

# take the first argument as path to data folder
# take alpha's as the second argument
abc = sys.argv[1] 
#alpha = sys.argv[2]
wora = sys.argv[2]
main = "http://www.brandsoftheworld.com/"
#letter = "logos/letter/"
#abc = [str(x) for x in range(9)]+list(string.ascii_uppercase)

#def logo_scrape(alpha):
for alpha in abc:
    os.chdir("/mnt/saswork/sh2264/vision/data/") # need to change
    #os.chdir("/Users/sheng/Dropbox/AAA/logos/data/")
    #os.chdir(sys.argv[1])
    #os.system("mkdir "+alpha)
    #if not os.path.exists(alpha):
    #    os.makedirs(alpha)

    os.chdir(alpha)
    
    #oldsoup = soup = BeautifulSoup(urllib.urlopen(main+letter+alpha).read(), "lxml")
    #titles = soup.find_all("ul", class_="titles twoCols")
    #rows = titles[0].find_all("li")#, class_="row")
    #rows_right = titles[0].find_all("li")
    #urls = temp = [ x.a['href'] for x in rows ]
    #page = 1
    #soup = BeautifulSoup(urllib.urlopen(main+letter+alpha+"?page="+str(page)).read(), "lxml")
    #print(soup!=oldsoup)
    #while soup != oldsoup:
    #    titles = soup.find_all("ul", class_="titles twoCols")
    #    rows = titles[0].find_all("li")#, class_="row")
    #    if temp == [ x.a['href'] for x in rows ]:
    #        break
    #    temp = [ x.a['href'] for x in rows ]
    #    urls += temp
    #    print(page)
    #    page += 1
    #    urlpages = "?page="+str(page)
        #oldsoup = soup
    #    soup = BeautifulSoup(urllib.urlopen(main+letter+alpha+urlpages).read(), "lxml")
        #print("hello?")
        #if soup == oldsoup:
        #    print(soup == oldsoup)
        #    break
    
    #print("all urls collected...")
    urls = []
    with codecs.open("urls.txt","r",encoding="utf-8") as urltxt:
        for url in urltxt:
            urls.append(url.strip().split("/")[-1])
    
    exist_url404 = 'urls404.txt' in os.listdir(os.getcwd())

    if exist_url404:
        urls_404 = []
        with codecs.open("urls404.txt","r",encoding="utf-8") as urls404:
            for url in urls404:
                urls_404.append(url.strip().split("/")[-1])
    # scrape each logo web page
    meta = []
    imgs = []
    with codecs.open('meta.txt', 'r', encoding='utf-8') as metatxt:
        for line in metatxt:
            meta.append(line.strip())
            imgs.append(meta[-1].split("\t")[0])
    
    imgs = set(imgs)
    
    if exist_url404:
        todo = ["/logo/"+x for x in set.difference(set.difference( set(urls), set(urls_404) ),imgs)]
    else:
        todo = ["/logo/"+x for x in set.difference(set(urls),imgs)]

    
    print(str(len(todo))+" images left to be APIed for "+alpha)
    
    if len(todo)>0:
        if not exist_url404:
            urls_404 = []
#        with codecs.open('memscore.txt', 'w', encoding='utf-8') as memscore:
        with codecs.open('memscore.txt', wora, encoding='utf-8') as memscore:
            for i,each in enumerate(reversed(todo)):
                print(i)
                count = 1
                while True and count<=5:
                    count += 1
                    try:
                        soup = BeautifulSoup(urllib.urlopen(main+each).read(), "lxml")
                        # image
                        url_logo = soup.find_all("img", class_="image")[0]['src']#.a['href']
                        break
                    except (IndexError, IOError):
                        time.sleep(0.5)
                        continue
                    else:
                        raise

                if count<=5:
                    #urls_404.append(each.split("/")[-1])
                #else:         
                    # if images not hidden, download
                    if not url_logo.endswith("hidden-logo.png"):
                        try:
                            time.sleep(random.random())
                            response = requests.get("http://memorability.csail.mit.edu/cgi-bin/image.py?url="+url_logo)
                            rating = response.json()['memscore']
                        except IOError:
                            #urls_404.append(each.split("/")[-1])
                            pass

                        memscore.write( each.split("/")[-1] + "\t" + str(rating)+ "\n" )
                
                    # meta-data -- tags
                    #terms = soup.find_all("div",class_="terms")[0].find_all("a")
                    #metatxt.write( each.split("/")[-1] + "\t" + "\t".join( [term.getText() for term in terms] ) + "\n" )
                    # meta-data -- dl
                    #dl = soup.find_all("dl")[0]#, class_="designer")
                    #metatxt.write(each.split("/")[-1] + "\t" + "\t".join([x.getText() for x in dl.find_all("dd")]) + "\n")#,class_="designer")
                    # class_="contributor"
                    # class_="format"
                    # class_="status"
                    # class_="status" Vector Quality
                    # class_="updated" 
                    #with io.open("scrapes/comments/"+str(i)+".txt", 'w', encoding="utf-8") as f:
                    #        f.write(unicode.join(u'\n',map(unicode, result)))
    print("all done!")
    with codecs.open("mscore_done.txt","w",encoding="utf-8") as txt:
        txt.write("mscore all done!")
#    if exist_url404:
#        with codecs.open("urls404.txt","a",encoding="utf-8") as txt:
#            for url in urls_404:
#                txt.write(url.strip().split("/")[-1])
#    else:
#        with codecs.open("urls404.txt","w",encoding="utf-8") as txt:
#            for url in urls_404:
#                txt.write(url.strip().split("/")[-1])
    
#if __name__ == '__main__':
#    Parallel(n_jobs = num_cores*1.0/2)(delayed(logo_scrape)(i) for i in abc)