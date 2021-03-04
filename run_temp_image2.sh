#!/usr/bin/env bash

python train.py -ei="temp" -et "cnn_train" -ec "data~imagenet2012+model~resnet50_+optim~taso_l0.1_e100_w0.0001_m0.9_nf_a25_b0.7" -gpu 1 -x 1 -bs 256
