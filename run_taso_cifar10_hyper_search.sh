#!/usr/bin/env bash

python train.py -ei="taso" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~taso_l0.1_e5_w0.0001_m0.9_nt_a25_b0.6" -gpu 0
python train.py -ei="taso" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~taso_l0.1_e5_w0.0001_m0.9_nt_a25_b0.7" -gpu 0
python train.py -ei="taso" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~taso_l0.1_e5_w0.0001_m0.9_nt_a25_b0.8" -gpu 0
