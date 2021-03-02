#!/usr/bin/env bash

python train.py -ei="temp" -et "cnn_train" -ec "data~yahooa+model~rcnn+optim~adam_l0.001_e1_w0.0001_af_bf0.9_bs0.99_l4" -gpu 0 -x 1 -bs 32
python train.py -ei="temp" -et "cnn_train" -ec "data~yahooa+model~textrnn+optim~adam_l0.001_e1_w0.0001_af_bf0.9_bs0.99_l4" -gpu 0 -x 1 -bs 32
python train.py -ei="temp" -et "cnn_train" -ec "data~yahooa+model~s2satt+optim~adam_l0.001_e1_w0.0001_af_bf0.9_bs0.99_l4" -gpu 0 -x 1 -bs 32

python train.py -ei="temp" -et "cnn_train" -ec "data~yelprf+model~rcnn+optim~adam_l0.001_e1_w0.0001_af_bf0.9_bs0.99_l4" -gpu 0 -x 1 -bs 32
python train.py -ei="temp" -et "cnn_train" -ec "data~yelprf+model~textrnn+optim~adam_l0.001_e1_w0.0001_af_bf0.9_bs0.99_l4" -gpu 0 -x 1 -bs 32
python train.py -ei="temp" -et "cnn_train" -ec "data~yelprf+model~s2satt+optim~adam_l0.001_e1_w0.0001_af_bf0.9_bs0.99_l4" -gpu 0 -x 1 -bs 32

python train.py -ei="temp" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~adam_l0.001_e1_w0.0001_af_bf0.9_bs0.99_l4" -gpu 0 -x 1 -bs 32
python train.py -ei="temp" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~adam_l0.001_e1_w0.0001_af_bf0.9_bs0.99_l4" -gpu 0 -x 1 -bs 32
python train.py -ei="temp" -et "cnn_train" -ec "data~agnews+model~s2satt+optim~adam_l0.001_e1_w0.0001_af_bf0.9_bs0.99_l4" -gpu 0 -x 1 -bs 32
