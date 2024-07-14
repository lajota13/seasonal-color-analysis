#!/bin/bash
# Simple bash script to download the LFW dataset locally.
# It accepts as input argument the destination directory where to download the dataset

dst_dir=$1
cd $dst_dir
wget http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
tar -xf lfw-deepfunneled.tgz lfw-deepfunneled
cd -