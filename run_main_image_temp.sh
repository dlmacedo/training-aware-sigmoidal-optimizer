#!/usr/bin/env bash

#python train.py -ei="main" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l0.1_e100_w0.0001_m0.9_nt_l-6_u3" -gpu 0 -x 4 -sx 2 -bs 64
#python train.py -ei="main" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.0005_e100_w0_m0_ct_a0.99" -gpu 0 -x 4 -sx 2 -bs 64
#python train.py -ei="main" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.0005_e100_w0_at_bf0.9_bs0.99" -gpu 0 -x 4 -sx 2 -bs 64
#python train.py -ei="main" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l0.5_e100_w0_d0_a0.1" -gpu 0 -x 4 -sx 2 -bs 64

python train.py -ei="main" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~htd_l0.1_e100_w0.0001_m0.9_nt_l-6_u3" -gpu 0 -x 4 -sx 2 -bs 64
python train.py -ei="main" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~rmsprop_l0.0005_e100_w0_m0_ct_a0.99" -gpu 0 -x 4 -sx 2 -bs 64
python train.py -ei="main" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~adam_l0.0005_e100_w0_at_bf0.9_bs0.99" -gpu 0 -x 4 -sx 2 -bs 64
python train.py -ei="main" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+optim~adagrad_l0.5_e100_w0_d0_a0.1" -gpu 0 -x 4 -sx 2 -bs 64

#python train.py -ei="main" -et "cnn_train" -ec "data~cifar100+model~resnet50+optim~htd_l0.1_e100_w0.0001_m0.9_nt_l-6_u3" -gpu 0 -x 1 -bs 64
#python train.py -ei="main" -et "cnn_train" -ec "data~cifar100+model~resnet50+optim~rmsprop_l0.0005_e100_w0_m0_ct_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="main" -et "cnn_train" -ec "data~cifar100+model~resnet50+optim~adam_l0.0005_e100_w0_at_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="main" -et "cnn_train" -ec "data~cifar100+model~resnet50+optim~adagrad_l0.5_e100_w0_d0_a0.1" -gpu 0 -x 1 -bs 64

#python train.py -ei="main" -et "cnn_train" -ec "data~cifar100+model~densenetbc100+optim~htd_l0.1_e100_w0.0001_m0.9_nt_l-6_u3" -gpu 0 -x 1 -bs 64
#python train.py -ei="main" -et "cnn_train" -ec "data~cifar100+model~densenetbc100+optim~rmsprop_l0.0005_e100_w0_m0_ct_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="main" -et "cnn_train" -ec "data~cifar100+model~densenetbc100+optim~adam_l0.0005_e100_w0_at_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="main" -et "cnn_train" -ec "data~cifar100+model~densenetbc100+optim~adagrad_l0.5_e100_w0_d0_a0.1" -gpu 0 -x 1 -bs 64
