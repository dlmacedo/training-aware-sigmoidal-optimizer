#!/usr/bin/env bash

#python train.py -ei="pretune" -et "cnn_train" -ec "data~agnews+model~textcnn+optim~adagrad_l0.01_e2_w0_d0_a0" -gpu 0 -bs 16 -x 1
python train.py -ei="pretune" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~adagrad_l0.01_e2_w0_d0_a0" -gpu 0 -bs 32 -x 1
python train.py -ei="pretune" -et "cnn_train" -ec "data~agnews+model~s2satt+optim~adagrad_l0.01_e2_w0_d0_a0" -gpu 0 -bs 32 -x 1
