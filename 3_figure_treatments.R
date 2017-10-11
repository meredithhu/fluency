#!/usr/bin/env Rscript

.libPaths("/mnt/saswork/sh2264/packages")
require(dplyr)
require(data.table)
require(digest)
#require(rjson)
require(doMC)
#require(RcppCNPy)
#require(rPython)
library(ggplot2)
library(xkcd)
library(stargazer)
merged_plots = fread("merged_plots.csv")
merged_plots = data.table(merged_plots)
table(merged_plots$country_y) %>% sort(dec=T) %>% names -> countryrank
country100=countryrank[1:100]
country50 = countryrank[1:50]

country_plot = merged_plots[country_y %in% country50,]
median_score = merged_plots$score %>% median
median_mean = merged_plots$mean %>% median
median_std = merged_plots$std %>% median
median_entropy = merged_plots$entropy %>% median
median_kl = merged_plots$kl %>% median
tr1 = merged_plots[std<=median_std & entropy<=median_entropy,]
tr1_1 = merged_plots[std<median_std & entropy<median_entropy,]
tr2 = merged_plots[std>=median_std & entropy<=median_entropy,]
tr2_1 = merged_plots[std>median_std & entropy<median_entropy,]
tr3 = merged_plots[std<=median_std & entropy>=median_entropy,]
tr3_1 = merged_plots[std<median_std & entropy>median_entropy,]
tr4 = merged_plots[std>=median_std & entropy>=median_entropy,]
tr4_1 = merged_plots[std>median_std & entropy>median_entropy,]

p <- ggplot(aes(score,fill=score), data=tr1)
p = p + geom_histogram(bins = 30) + xlab("Memorability Score") + ylab("Frequency")
p

variable = merged_plots$entropy
variable = merged_plots$kl
variable = merged_plots$std
p <- ggplot(aes(variable,fill=variable), data=merged_plots)
p = p + xlim(c(0,20)) + ylim(c(0,5000))
p = p + geom_histogram(bins = 50) + ylab("Frequency") + xlab("Style Complexity Score (Std of Local Shannon Entropies)")
p = p + geom_histogram(bins = 50) + ylab("Frequency") + xlab("Content Ambiguity Score (KL Divergence)")
p = p + geom_histogram(bins = 50) + ylab("Frequency") + xlab("Content Ambiguity Score (Shannon Entropy)")
p = p + ggtitle("Distribution of Kullback-Leibler Divergence as Content Ambiguity")
p = p + ggtitle("Distribution of Shannon Entropy as Content Ambiguity")
p = p + ggtitle("Distribution of Std. of Local Shannon Entropies as Style Complexity")
p


p <- ggplot(aes(variable, fill=variable), data=merged_plots)
p <- ggplot(aes(x=country_y, fill=country_y), data=country_plot)
# using geom_bar will automatically make a new "count" column
# available in an internal, transformed data frame. the help
# for geom_bar says as much

p <- p + geom_bar(width=1, colour="white")
#p = p + coord_polar("x", start=0)
# geom_text can then access this computed variable with
# ..count.. (I still thin that's horrible syntax, hadley :-)
p <- p + geom_text(aes( label=..count.., y=..count..),
                   stat="count",
                   color="white",
                   hjust=1.0, size=3)

p <- p + theme(legend.position="none")
p <- p + coord_flip()
p = p + theme_xkcd()
p = p + ggtitle("Distribution of Logo Country")+xlab("Country")+ylab("Frequency")
p

# to be more explicit to other readers of your code, you
# could also do this instead of the `geom_bar` call
p <- p + stat_count(width=1, colour="white", geom="bar")

ggplot(aes(x=category_y), data=merged_plots)+
geom_histogram()+#+theme_xkcd()





args = commandArgs()[6:length(commandArgs())]

vision = "/mnt/saswork/sh2264/vision/"
setwd(paste0(vision,"code"))

features = npyLoad("features_model0_15.npy")

setwd(paste0(vision,"data"))
folders = list.files()
entropy = list()
entropynm = list()
memscore = list()
for (i in 1:length(folders)){
	setwd(paste0(vision,"data/",folders[i]))
	entropy[[i]] = npyLoad("entropy.npy")
	entropynm[[i]] = npyLoad("entropy_name.npy")
	memscore[[i]] = fread("memscore.txt")
}