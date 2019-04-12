#!/bin/bash

# create a new folder and download Omniglot dataset 
mkdir data/
pushd data/
wget "https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_background.zip"
wget "https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_evaluation.zip"

# unzip images
unzip -q '*.zip'

# move zip files to raw dir
mkdir raw/
mv *.zip raw/

# rename folders
mv images_background/ background/    # contains 30 folders
mv images_evaluation/ evaluation/    # contains 20 folders

# move 10 first evaluation subdirs to background dir
pushd evaluation/
folders=(*/)  # get all folder names
popd
for ((i=0; i<10; i++))
do
  mv "evaluation/${folders[i]}" background/
done

# move the 40 training and 10 testing dirs to processed dir
mkdir processed/
mv background processed/
mv evaluation processed/

echo "done"
