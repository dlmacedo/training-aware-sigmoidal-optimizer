#!/usr/bin/env bash

#python train.py -ei="temp" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.001_e100_w0.0001_af_bf0.9_bs0.99" -gpu 1 -x 1
#python train.py -ei="temp" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~adam_l0.001_e100_w0.0001_af_bf0.9_bs0.99" -gpu 1 -x 1
#python train.py -ei="temp" -et "cnn_train" -ec "data~cifar10+model~efficientnetb0+optim~adam_l0.001_e100_w0.0001_af_bf0.9_bs0.99" -gpu 1 -x 1
#python train.py -ei="temp" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l0.05_e100_w0.0001_m0.9_nf_a25_b0.7" -gpu 1 -x 1
#python train.py -ei="temp" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~taso_l0.05_e100_w0.0001_m0.9_nf_a25_b0.7" -gpu 1 -x 1
#python train.py -ei="temp" -et "cnn_train" -ec "data~cifar10+model~efficientnetb0+optim~taso_l0.1_e100_w0.0001_m0.9_nf_a25_b0.7" -gpu 1 -x 1

#python train.py -ei="temp" -et "cnn_train" -ec "data~cifar100+model~resnet50+optim~adam_l0.001_e100_w0.0001_af_bf0.9_bs0.99" -gpu 1 -x 1
#python train.py -ei="temp" -et "cnn_train" -ec "data~cifar100+model~densenetbc100+optim~adam_l0.001_e100_w0.0001_af_bf0.9_bs0.99" -gpu 1 -x 1
#python train.py -ei="temp" -et "cnn_train" -ec "data~cifar100+model~efficientnetb0+optim~adam_l0.001_e100_w0.0001_af_bf0.9_bs0.99" -gpu 1 -x 1
#python train.py -ei="temp" -et "cnn_train" -ec "data~cifar100+model~resnet50+optim~taso_l0.05_e100_w0.0001_m0.9_nf_a25_b0.7" -gpu 1 -x 1
#python train.py -ei="temp" -et "cnn_train" -ec "data~cifar100+model~densenetbc100+optim~taso_l0.05_e100_w0.0001_m0.9_nf_a25_b0.7" -gpu 1 -x 1
#python train.py -ei="temp" -et "cnn_train" -ec "data~cifar100+model~efficientnetb0+optim~taso_l0.05_e100_w0.0001_m0.9_nf_a25_b0.7" -gpu 1 -x 1

#python train.py -ei="temp" -et "cnn_train" -ec "data~tinyimagenet200+model~resnet50+optim~adam_l0.001_e100_w0.0001_af_bf0.9_bs0.99" -gpu 1 -x 1
#python train.py -ei="temp" -et "cnn_train" -ec "data~tinyimagenet200+model~densenetbc100+optim~adam_l0.001_e100_w0.0001_af_bf0.9_bs0.99" -gpu 1 -x 1
#python train.py -ei="temp" -et "cnn_train" -ec "data~tinyimagenet200+model~efficientnetb0+optim~adam_l0.001_e100_w0.0001_af_bf0.9_bs0.99" -gpu 1 -x 1
python train.py -ei="temp" -et "cnn_train" -ec "data~tinyimagenet200+model~resnet50+optim~taso_l0.05_e100_w0.0001_m0.9_nf_a25_b0.7" -gpu 1 -x 1
python train.py -ei="temp" -et "cnn_train" -ec "data~tinyimagenet200+model~densenetbc100+optim~taso_l0.05_e100_w0.0001_m0.9_nf_a25_b0.7" -gpu 1 -x 1
python train.py -ei="temp" -et "cnn_train" -ec "data~tinyimagenet200+model~efficientnetb0+optim~taso_l0.05_e100_w0.0001_m0.9_nf_a25_b0.7" -gpu 1 -x 1
