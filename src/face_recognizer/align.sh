#!/bin/bash
rm -r aligned_images
mkdir aligned_images
for N in {1..4}; do \
python align/align_dataset_mtcnn.py \
images/ \
aligned_images/ \
--image_size 160 \
--margin 32 \
--random_order \
--gpu_memory_fraction 0.5 \
& done

