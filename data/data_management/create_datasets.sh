#!/bin/bash
# ============================================================
# Recorded commands for the bag datasets. Run from data/data_management/.
# The bag files on the USB stick are the ones used in the thesis (see README, "Data").
# ============================================================

# MNIST (reproduces data/datasets/bags/mnist_bags_base.h5 bit for bit)
python mnist_bags_loader_h5.py --mode sweep      # bag-size sweep, groups bags{N}_len{L}
python mnist_bags_loader_h5.py --mode even       # high-density bags, group mnist_bags_even

# GWHD, centred patches, earlier version (groups gwhd_bags_dense / gwhd_bags_dense_color in
# ../datasets/bags/gwhd_bags.h5, 1000/100/100 bags). Recorded calls of an earlier version of
# create_gwhd_bags.py; the current script ignores --dense (see README, "Notes").
# python create_gwhd_bags.py --dataset_name gwhd_bags_dense --num_bags 1000 100 100 --patch_size 64 --overlap_threshold 1.0 --grayscale --dense
# python create_gwhd_bags.py --dataset_name gwhd_bags_dense_color --num_bags 1000 100 100 --patch_size 64 --overlap_threshold 1.0 --dense
