#!/bin/bash
mkdir -p ../data/ISRUC/RawData/
mkdir -p ../data/ISRUC/ExtractedChannels/
cd ../data/ISRUC/RawData/
for s in {1..10}; do
    wget http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupIII/$s.rar
    unrar x $s.rar
done
cd ../ExtractedChannels/
for s in {1..10}; do
  wget http://dataset.isr.uc.pt/ISRUC_Sleep/ExtractedChannels/subgroupIII-Extractedchannels/subject$s.mat
done