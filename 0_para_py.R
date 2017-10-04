#!/usr/bin/env Rscript

.libPaths("/mnt/saswork/sh2264/packages")
require(dplyr)
require(data.table)
require(digest)
#require(rjson)
require(doMC)

args = commandArgs()
print("arg1: #cores\n")
print("arg2: filename\n")
print("arg3: path to vision folder\n")

registerDoMC(cores=as.integer(args[1]))
scrplist = c(as.character(0:9),LETTERS)

setwd(args[3])
foreach(alpha = scrplist)%dopar%{
      print(paste('scraping list...',alpha,Sys.time()))
      #out=path:"indiv_crd_":crd:'.pdf'
      datafolder = ifelse(substr(args[3],nchar(args[3]),nchar(args[3]))=="/", paste0(args[3],"data/"), paste0(args[3],"/data/"))   
      script='/usr/local/anaconda/bin/python %s %s %s' %>% sprintf(args[2], datafolder, alpha)
      #script='python %s %s %s' %>% sprintf(args[2], datafolder, alpha)
      
      system(script)
}
      #email(subj='indiv crd scrape thread start',body = 'crd was ':crd)
      
      #email(subj='indiv crd scrape',body = 'crd was ':crd)
      #simpleUpload(data.table(crd,date=Sys.Date() %>% as.character,time=Sys.time() %>% as.character),'system.tried_crds')