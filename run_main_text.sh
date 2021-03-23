#!/usr/bin/env bash

#python train.py -ei="main" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~adagrad_l0.5_e20_w0_d0_a0.1" -gpu 1 -x 1 -bs 32
#python train.py -ei="main" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~rmsprop_l0.001_e20_w0_m0_cf_a0.99" -gpu 1 -x 1 -bs 32
#python train.py -ei="main" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~adam_l0.0005_e20_w0_at_bf0.9_bs0.99" -gpu 1 -x 1 -bs 32
#python train.py -ei="main" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~taso_l0.05_e20_w0.0001_m0.9_nt_a25_b0.7" -gpu 1 -x 1 -bs 32

#python train.py -ei="main" -et "cnn_train" -ec "data~agnews+model~s2satt+optim~adagrad_l0.5_e20_w0_d0_a0.1" -gpu 1 -x 1 -bs 32
#python train.py -ei="main" -et "cnn_train" -ec "data~agnews+model~s2satt+optim~rmsprop_l0.001_e20_w0_m0_cf_a0.99" -gpu 1 -x 1 -bs 32
#python train.py -ei="main" -et "cnn_train" -ec "data~agnews+model~s2satt+optim~adam_l0.0005_e20_w0_at_bf0.9_bs0.99" -gpu 1 -x 1 -bs 32
#python train.py -ei="main" -et "cnn_train" -ec "data~agnews+model~s2satt+optim~taso_l0.05_e20_w0.0001_m0.9_nt_a25_b0.7" -gpu 1 -x 1 -bs 32

#python train.py -ei="main" -et "cnn_train" -ec "data~yelprf+model~rcnn+optim~adagrad_l0.5_e20_w0_d0_a0.1" -gpu 1 -x 1 -bs 32
#python train.py -ei="main" -et "cnn_train" -ec "data~yelprf+model~rcnn+optim~rmsprop_l0.001_e20_w0_m0_cf_a0.99" -gpu 1 -x 1 -bs 32
#python train.py -ei="main" -et "cnn_train" -ec "data~yelprf+model~rcnn+optim~adam_l0.0005_e20_w0_at_bf0.9_bs0.99" -gpu 1 -x 1 -bs 32
#python train.py -ei="main" -et "cnn_train" -ec "data~yelprf+model~rcnn+optim~taso_l0.05_e20_w0.0001_m0.9_nt_a25_b0.7" -gpu 1 -x 1 -bs 32

#python train.py -ei="main" -et "cnn_train" -ec "data~yelprf+model~s2satt+optim~adagrad_l0.5_e20_w0_d0_a0.1" -gpu 1 -x 1 -bs 32
#python train.py -ei="main" -et "cnn_train" -ec "data~yelprf+model~s2satt+optim~rmsprop_l0.001_e20_w0_m0_cf_a0.99" -gpu 1 -x 1 -bs 32
#python train.py -ei="main" -et "cnn_train" -ec "data~yelprf+model~s2satt+optim~adam_l0.0005_e20_w0_at_bf0.9_bs0.99" -gpu 1 -x 1 -bs 32
#python train.py -ei="main" -et "cnn_train" -ec "data~yelprf+model~s2satt+optim~taso_l0.05_e20_w0.0001_m0.9_nt_a25_b0.7" -gpu 1 -x 1 -bs 32

python train.py -ei="main" -et "cnn_train" -ec "data~yahooa+model~rcnn+optim~adagrad_l0.5_e20_w0_d0_a0.1" -gpu 1 -x 1 -bs 32
python train.py -ei="main" -et "cnn_train" -ec "data~yahooa+model~rcnn+optim~rmsprop_l0.001_e20_w0_m0_cf_a0.99" -gpu 1 -x 1 -bs 32
python train.py -ei="main" -et "cnn_train" -ec "data~yahooa+model~rcnn+optim~adam_l0.0005_e20_w0_at_bf0.9_bs0.99" -gpu 1 -x 1 -bs 32
python train.py -ei="main" -et "cnn_train" -ec "data~yahooa+model~rcnn+optim~taso_l0.05_e20_w0.0001_m0.9_nt_a25_b0.7" -gpu 1 -x 1 -bs 32

python train.py -ei="main" -et "cnn_train" -ec "data~yahooa+model~s2satt+optim~adagrad_l0.5_e20_w0_d0_a0.1" -gpu 1 -x 1 -bs 32
python train.py -ei="main" -et "cnn_train" -ec "data~yahooa+model~s2satt+optim~rmsprop_l0.001_e20_w0_m0_cf_a0.99" -gpu 1 -x 1 -bs 32
python train.py -ei="main" -et "cnn_train" -ec "data~yahooa+model~s2satt+optim~adam_l0.0005_e20_w0_at_bf0.9_bs0.99" -gpu 1 -x 1 -bs 32
python train.py -ei="main" -et "cnn_train" -ec "data~yahooa+model~s2satt+optim~taso_l0.05_e20_w0.0001_m0.9_nt_a25_b0.7" -gpu 1 -x 1 -bs 32
