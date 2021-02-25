#!/usr/bin/env bash

python train.py -ei="pretune" -et "cnn_train" -ec "data~yelprf+model~textcnn+optim~adagrad_l0.01_e1_w0_d0_a0" -gpu 1 -bs 32 -x 1
python train.py -ei="pretune" -et "cnn_train" -ec "data~yelprf+model~rcnn+optim~adagrad_l0.01_e1_w0_d0_a0" -gpu 1 -bs 32 -x 1
python train.py -ei="pretune" -et "cnn_train" -ec "data~yelprf+model~s2satt+optim~adagrad_l0.01_e1_w0_d0_a0" -gpu 1 -bs 32 -x 1

python train.py -ei="pretune" -et "cnn_train" -ec "data~yahooa+model~textcnn+optim~adagrad_l0.01_e1_w0_d0_a0" -gpu 1 -bs 32 -x 1
python train.py -ei="pretune" -et "cnn_train" -ec "data~yahooa+model~rcnn+optim~adagrad_l0.01_e1_w0_d0_a0" -gpu 1 -bs 32 -x 1
python train.py -ei="pretune" -et "cnn_train" -ec "data~yahooa+model~s2satt+optim~adagrad_l0.01_e1_w0_d0_a0" -gpu 1 -bs 32 -x 1

python train.py -ei="pretune" -et "cnn_train" -ec "data~amazonrf+model~textcnn+optim~adagrad_l0.01_e1_w0_d0_a0" -gpu 1 -bs 32 -x 1
python train.py -ei="pretune" -et "cnn_train" -ec "data~amazonrf+model~rcnn+optim~adagrad_l0.01_e1_w0_d0_a0" -gpu 1 -bs 32 -x 1
python train.py -ei="pretune" -et "cnn_train" -ec "data~amazonrf+model~s2satt+optim~adagrad_l0.01_e1_w0_d0_a0" -gpu 1 -bs 32 -x 1
