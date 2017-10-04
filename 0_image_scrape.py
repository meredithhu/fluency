#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 19:14:50 2016

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
#from collections import defaultdict
#import matplotlib
#matplotlib.use('Agg')
#from matplotlib import pyplot as plt
#from nltk.tokenize import PunktSentenceTokenizer
#from senti_classifier import senti_classifier
#from textblob import TextBlob
from bs4 import BeautifulSoup
import re
import sys
#import scipy as sp
#import scipy.stats
#from collections import Counter
#import pickle
#from compiler.ast import flatten
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

# take the first argument as path to data folder
# take alpha's as the second argument
abc = sys.argv[2] 
#alpha = sys.argv[2]
main = "http://www.brandsoftheworld.com/"
letter = "logos/letter/"
#abc = [str(x) for x in range(9)]+list(string.ascii_uppercase)

#def logo_scrape(alpha):
for alpha in abc:
    #os.chdir("/mnt/saswork/sh2264/vision/data/") # need to change
    #os.chdir("/Users/sheng/Dropbox/AAA/logos/data/")
    os.chdir(sys.argv[1])
    #os.system("mkdir "+alpha)
    if not os.path.exists(alpha):
        os.makedirs(alpha)

    os.chdir(alpha)
    
    oldsoup = soup = BeautifulSoup(urllib.urlopen(main+letter+alpha).read(), "lxml")
    titles = soup.find_all("ul", class_="titles twoCols")
    rows = titles[0].find_all("li")#, class_="row")
    #rows_right = titles[0].find_all("li")
    urls = temp = [ x.a['href'] for x in rows ]
    page = 1
    soup = BeautifulSoup(urllib.urlopen(main+letter+alpha+"?page="+str(page)).read(), "lxml")
    #print(soup!=oldsoup)
    while soup != oldsoup:
        titles = soup.find_all("ul", class_="titles twoCols")
        rows = titles[0].find_all("li")#, class_="row")
        if temp == [ x.a['href'] for x in rows ]:
            break
        temp = [ x.a['href'] for x in rows ]
        urls += temp
        print(page)
        page += 1
        urlpages = "?page="+str(page)
        #oldsoup = soup
        soup = BeautifulSoup(urllib.urlopen(main+letter+alpha+urlpages).read(), "lxml")
        #print("hello?")
        #if soup == oldsoup:
        #    print(soup == oldsoup)
        #    break
    
    print("all urls collected...")
    with codecs.open("urls.txt","w",encoding="utf-8") as urltxt:
        for url in urls:
            urltxt.write(url+"\n")
            
    # scrape each logo web page
    with codecs.open('meta.txt', 'w', encoding='utf-8') as meta:
        
        for i,each in enumerate(urls):
            print(i)
            
            while True:
                try:
                    soup = BeautifulSoup(urllib.urlopen(main+each).read(), "lxml")
                    # image
                    url_logo = soup.find_all("img", class_="image")[0]['src']#.a['href']
                    break
                except IndexError:
                    time.sleep(1)
                    continue
                else:
                    raise
                    
            urllib.urlretrieve(url_logo, each.split("/")[-1]+".jpeg") 
            # meta-data -- tags
            terms = soup.find_all("div",class_="terms")[0].find_all("a")
            meta.write( each.split("/")[-1] + "\t" + "\t".join( [term.getText() for term in terms] ) + "\n" )
            # meta-data -- dl
            dl = soup.find_all("dl")[0]#, class_="designer")
            meta.write(each.split("/")[-1] + "\t" + "\t".join([x.getText() for x in dl.find_all("dd")]) + "\n")#,class_="designer")
            # class_="contributor"
            # class_="format"
            # class_="status"
            # class_="status" Vector Quality
            # class_="updated" 
        #with io.open("scrapes/comments/"+str(i)+".txt", 'w', encoding="utf-8") as f:
        #        f.write(unicode.join(u'\n',map(unicode, result)))

#if __name__ == '__main__':
#    Parallel(n_jobs = num_cores*1.0/2)(delayed(logo_scrape)(i) for i in abc)