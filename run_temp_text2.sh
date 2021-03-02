#!/usr/bin/env bash

#python train.py -ei="temp" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~adam_l0.001_e20_w0.0001_af_bf0.9_bs0.99_l2" -gpu 0 -x 1 -bs 32
python train.py -ei="temp" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~taso_l0.05_e20_w0.0001_m0.9_nf_a25_b0.7_l2" -gpu 0 -x 1 -bs 32
#python train.py -ei="temp" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~adam_l0.001_e20_w0.0001_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 32
#python train.py -ei="temp" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~taso_l0.05_e20_w0.0001_m0.9_nf_a25_b0.7" -gpu 0 -x 1 -bs 32
#python train.py -ei="temp" -et "cnn_train" -ec "data~agnews+model~s2satt+optim~adam_l0.001_e20_w0.0001_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 32
#python train.py -ei="temp" -et "cnn_train" -ec "data~agnews+model~s2satt+optim~taso_l0.05_e20_w0.0001_m0.9_nf_a25_b0.7" -gpu 0 -x 1 -bs 32

python train.py -ei="temp" -et "cnn_train" -ec "data~yelprf+model~rcnn+optim~adam_l0.001_e20_w0.0001_af_bf0.9_bs0.99_l2" -gpu 0 -x 1 -bs 32
python train.py -ei="temp" -et "cnn_train" -ec "data~yelprf+model~rcnn+optim~taso_l0.05_e20_w0.0001_m0.9_nf_a25_b0.7_l2" -gpu 0 -x 1 -bs 32
#python train.py -ei="temp" -et "cnn_train" -ec "data~yelprf+model~textrnn+optim~adam_l0.001_e20_w0.0001_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="temp" -et "cnn_train" -ec "data~yelprf+model~textrnn+optim~taso_l0.05_e20_w0.0001_m0.9_nf_a25_b0.7" -gpu 0 -x 1 -bs 32
#python train.py -ei="temp" -et "cnn_train" -ec "data~yelprf+model~s2satt+optim~adam_l0.001_e20_w0.0001_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="temp" -et "cnn_train" -ec "data~yelprf+model~s2satt+optim~taso_l0.05_e20_w0.0001_m0.9_nf_a25_b0.7" -gpu 0 -x 1 -bs 32
