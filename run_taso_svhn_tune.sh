#!/usr/bin/env bash

python train.py -ei="batch" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~adam_l0.001_e5_w0.0001_at_bf0.9_bs0.99" -gpu 0
python train.py -ei="batch" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~adam_l0.001_e5_w0.0001_af_bf0.9_bs0.99" -gpu 0
python train.py -ei="batch" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~adam_l0.001_e5_w0_at_bf0.9_bs0.99" -gpu 0
