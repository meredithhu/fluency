#!/usr/bin/env Rscript

.libPaths("/mnt/saswork/sh2264/packages")
require(dplyr)
require(data.table)
require(digest)
#require(rjson)
require(doMC)

args = commandArgs()[6:length(commandArgs())]
print("arg1: #cores\n")
print("arg2: python script\n")

registerDoMC(cores=as.integer(args[1]))
#scrplist = c(as.character(0:9),LETTERS)
#done = c("Q","K","Y","X","Z","O","N","J","W","B","T","F")
#remaining = setdiff(LETTERS, c("A"))
remaining = c(as.character(0:9),LETTERS)
#remaining = c("G","I","S","D","M")
setwd("/mnt/saswork/sh2264/vision/code")
#opts = list(preschedule=FALSE)
foreach(alpha = remaining)%dopar%{
#foreach(alpha = remaining, .options.multicore=opts)%dopar%{
      print(paste('converting folder...',alpha,Sys.time()))
      #out=path:"indiv_crd_":crd:'.pdf'
      script='python %s %s' %>% sprintf(args[2], alpha)
      #script='python %s %s %s' %>% sprintf(args[2], datafolder, alpha)
      
      system(script)
}
      #email(subj='indiv crd scrape thread start',body = 'crd was ':crd)
      
      #email(subj='indiv crd scrape',body = 'crd was ':crd)
      #simpleUpload(data.table(crd,date=Sys.Date() %>% as.character,time=Sys.time() %>% as.character),'system.tried_crds')