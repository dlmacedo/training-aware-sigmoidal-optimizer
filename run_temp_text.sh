#!/usr/bin/env bash

#python train.py -ei="temp" -et "cnn_train" -ec "data~yelprf+model~textrnn+optim~adam_l0.001_e20_w0.0001_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 32
#python train.py -ei="temp" -et "cnn_train" -ec "data~yelprf+model~rcnn+optim~adam_l0.001_e20_w0.0001_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 32
#python train.py -ei="temp" -et "cnn_train" -ec "data~yelprf+model~s2satt+optim~adam_l0.001_e20_w0.0001_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 32

#python train.py -ei="temp" -et "cnn_train" -ec "data~yelprf+model~textrnn+optim~taso_l0.1_e20_w0.0001_m0.9_nf_a25_b0.7" -gpu 0 -x 1 -bs 32
#python train.py -ei="temp" -et "cnn_train" -ec "data~yelprf+model~rcnn+optim~taso_l0.1_e20_w0.0001_m0.9_nf_a25_b0.7" -gpu 0 -x 1 -bs 32
#python train.py -ei="temp" -et "cnn_train" -ec "data~yelprf+model~s2satt+optim~taso_l0.1_e20_w0.0001_m0.9_nf_a25_b0.7" -gpu 0 -x 1 -bs 32

#python train.py -ei="temp" -et "cnn_train" -ec "data~yelprf+model~textcnn+optim~adagrad_l0.01_e1_w0_d0_a0_nogrove" -gpu 1 -x 1 -bs 32
#python train.py -ei="temp" -et "cnn_train" -ec "data~yelprf+model~rcnn+optim~adagrad_l0.01_e1_w0_d0_a0_nogorve" -gpu 1 -x 1 -bs 32
#python train.py -ei="temp" -et "cnn_train" -ec "data~yelprf+model~s2satt+optim~adagrad_l0.01_e1_w0_d0_a0_nogrove" -gpu 1 -x 1 -bs 32

#python train.py -ei="temp" -et "cnn_train" -ec "data~yahooa+model~textcnn+optim~adagrad_l0.01_e1_w0_d0_a0_nogrove" -gpu 1 -x 1 -bs 32
#python train.py -ei="temp" -et "cnn_train" -ec "data~yahooa+model~rcnn+optim~adagrad_l0.01_e1_w0_d0_a0_nogrove" -gpu 1 -x 1 -bs 32
#python train.py -ei="temp" -et "cnn_train" -ec "data~yahooa+model~s2satt+optim~adagrad_l0.01_e1_w0_d0_a0_nogrove" -gpu 1 -x 1 -bs 32

#python train.py -ei="temp" -et "cnn_train" -ec "data~amazonrf+model~textcnn+optim~adagrad_l0.01_e1_w0_d0_a0_nogrove" -gpu 1 -x 1 -bs 32
#python train.py -ei="temp" -et "cnn_train" -ec "data~amazonrf+model~rcnn+optim~adagrad_l0.01_e1_w0_d0_a0_nogrove" -gpu 1 -x 1 -bs 32
#python train.py -ei="temp" -et "cnn_train" -ec "data~amazonrf+model~s2satt+optim~adagrad_l0.01_e1_w0_d0_a0_nogrove" -gpu 1 -x 1 -bs 32
