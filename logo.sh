# parallelized image scraping from brandsoftheworld.com
# save the images, meta info, urls and errors
Rscript 0_para_py.R 36 0_image_scrape_url.py /mnt/saswork/sh2264/logo/vision/data
# 36 is the number of cores run in parallel
# basically the R script just scrapes 0:9, A:Z in parallel

# now convert these images to numbers of pixels
Rscript 1_1_image_to_numpy_parallel.R 36 1_1_image_to_numpy_parallel.py
# 36 is the number of cores in parallel

# parse the meta.txt files into category and country npy arrays
python 1_1_meta_parse.py

# no freaking idea of what this is
python 1_2_meta_category.py

# planned to extract traditional image features
# but unfinished
### but hey, you dont need to!
python 1_3_image_features.py

# AlexNet
python 2_img_classify.py 15 1
# 15 epochs model1
# returns features/history/scores_model1_15_1.npy and the like to the code folder

# measure image complexity
Rscript 1_1_image_to_numpy_parallel.R 36 2_img_complex.py
# save entropy of images folder by folder

# calculate popularity scores
# first run: 
Rscript 1_parallel_pythons.R 36 2_img_popularity.py w
# later runs:
Rscript 1_parallel_pythons.R 36 2_img_popularity.py a

# calculate memorability scores
# first run: 
Rscript 1_parallel_pythons.R 36 2_img_memorability.py w
# later runs:
Rscript 1_parallel_pythons.R 36 2_img_memorability.py a

# regress scores on features
python 3_correlation.py features_model1_15 1 1
# print "system argument 1: which feature file to use, no need to include .npy"
# print "system argument 2: take log of memscore or not: 0, don't; 1, do."
# print "system argument 3: take log of merged['mean'] or not: 0, don't; 1, do"
python 3_correlation_pop.py features_model1_15 1 1 1 #Error:SVD did not converge
# print "system argument 1: which feature file to use, no need to include .npy"
# print "system argument 2: take log of memscore or not: 0, don't; 1, do."
# print "system argument 3: take log of merged['mean'] or not: 0, don't; 1, do"
# print "syste, argument 4: take log of popscore or not"

# summary plots
python -i 3_figure_treatments.py features_model1_15 # does not work?
# change directory to ../vision/results to use the merged_plots.csv file
# fire up RStudio to get the summary plots done modifying 3_figure_treatments.R!
# Rscript 3_figure_treatments.R #

# 05/13/2017
# trained ResNet on logos 
cd /mnt/saswork/sh2264/vision/code/tensorflow/models/tutorials/image/imagenet/deep-learning-models
# processing for folder A, outputs are imagenet_A.npy and imagenet_index_A.npy in folder A
python 5_resnet50_classify.py A
# in parallel
Rscript 1_parallel_pythons.R 36 5_resnet50_classify.py
# or not in parallel since our server sucks..
Rscript 1_noparallel_pythons.R 36 5_resnet50_classify.py

## simple regressions based on 5_resnet
python 6_correlation.py features_model1_15 0 0 0 0

# CNN for memorability/likability prediction (regression problem)
python 2_img_regression.py 15 1
python 2_img_regression.py 15 2
#print "system argument 1: number of training epochs"
#print "system argument 2: which y to use: 1 then mscore, 2 then pscore"


# FUN with GANs
cd /mnt/saswork/sh2264/vision/code/DCGAN-tensorflow/
python -i main.py --dataset=bookcover_small --input_fname_pattern="*.png" --input_height=200 --output_height=100 --train --crop 

python -i main.py --dataset=logos --input_fname_pattern="*.jpeg" --input_height=200 --output_height=100 --train --crop 
# ValueError: could not broadcast input array
## https://github.com/carpedm20/DCGAN-tensorflow/issues/162
cd data/mypics
# Create dir to store images that aren't the same size`
mkdir misshaped
# Identify misshaped images using Imagemagick's 'identify' tool 
and move to above dir. (Replace with your desired resolution)
identify * | grep -v "200x200" | awk '{ print $1 }' | xargs -I {} bash -c "mv {} misshaped"
# argument list too long:
identify [a-g]* | grep -v "200x200" | awk '{ print $1 }' |sed 's/\[[^]]*\]//g'| xargs -I {} bash -c "mv {} misshaped"
identify [h-n]* | grep -v "200x200" | awk '{ print $1 }' |sed 's/\[[^]]*\]//g'| xargs -I {} bash -c "mv {} misshaped"
identify [o-z]* | grep -v "200x200" | awk '{ print $1 }' |sed 's/\[[^]]*\]//g'| xargs -I {} bash -c "mv {} misshaped"

# still didnot work need to remove [*] first, tried 
# tr -d "[]" but this only removes the []
# added sed 's/\[[^]]*\]//g'

identify -format "%i %[colorspace]\n" *.jpeg | grep -v sRGB

## utility files: 
## move top ranked logo designs (memoscore/popscore) into folder:
## /mnt/saswork/sh2264/vision/graphs/samples 
00_movefile_figure1.py