#!/usr/bin/env bash

python train.py -ei="batch" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~adagrad_l0.01_e5_w0_d0_a0" -gpu 1
python train.py -ei="batch" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~adagrad_l0.01_e5_w0.0001_d0_a0" -gpu 1
python train.py -ei="batch" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~adagrad_l0.001_e5_w0.0001_d0_a0" -gpu 1
