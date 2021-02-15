#!/usr/bin/env bash

python train.py -ei="batch" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~rmsprop_l0.01_e5_w0.0001_m0.9_ct_a0.99" -gpu 0
python train.py -ei="batch" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~rmsprop_l0.01_e5_w0_m0.9_ct_a0.99" -gpu 0
python train.py -ei="batch" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~rmsprop_l0.01_e5_w0.0001_m0_ct_a0.99" -gpu 0
