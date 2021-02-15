#!/usr/bin/env bash

#python train.py -ei="pretune" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~adagrad_l0.01_e100_w0_d0_a0" -gpu 0
#python train.py -ei="pretune" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~adagrad_l0.01_e100_w0.0001_d0_a0" -gpu 0

python train.py -ei="pretune" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~rmsprop_l0.01_e100_w0_m0_cf_a0.99" -gpu 0
python train.py -ei="pretune" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~rmsprop_l0.01_e100_w0.0001_m0_cf_a0.99" -gpu 0
#python train.py -ei="pretune" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~rmsprop_l0.01_e100_w0_m0.9_cf_a0.99" -gpu 0
#python train.py -ei="pretune" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~rmsprop_l0.01_e100_w0.0001_m0.9_cf_a0.99" -gpu 0

#python train.py -ei="pretune" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~adam_l0.001_e100_w0_af_bf0.9_bs0.99" -gpu 1
#python train.py -ei="pretune" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~adam_l0.001_e100_w0.0001_af_bf0.9_bs0.99" -gpu 1

#python train.py -ei="pretune" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~taso_l0.1_e100_w0_m0_nf_a25_b0.7" -gpu 1
#python train.py -ei="pretune" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~taso_l0.1_e100_w0.0001_m0_nf_a25_b0.7" -gpu 1
#python train.py -ei="pretune" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~taso_l0.1_e100_w0_m0.9_nf_a25_b0.7" -gpu 1
#python train.py -ei="pretune" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~taso_l0.1_e100_w0.0001_m0.9_nf_a25_b0.7" -gpu 1
