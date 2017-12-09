#! usr/bin env python
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:31:50 2017
@author: shengli
sorting thru BAM dataset
"""

import os
import sys
import numpy as np
import pandas as pd
import sqlite3

os.chdir("/mnt/saswork/sh2264/bam/")


df = pd.read_sql("select * from scores",
	sqlite3.connect("./bam0.sqlite"),
	index_col="mid")
df.to_csv("bam_scores.csv")