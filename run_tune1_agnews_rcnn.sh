#!/usr/bin/env bash

python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~adagrad_l0.1_e20_w0.0001_d0_a0" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~adagrad_l0.05_e20_w0.0001_d0_a0" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~adagrad_l0.01_e20_w0.0001_d0_a0" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~adagrad_l0.005_e20_w0.0001_d0_a0" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~adagrad_l0.001_e20_w0.0001_d0_a0" -gpu 0 -x 1 -bs 32

python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~rmsprop_l0.1_e20_w0.0001_m0.9_cf_a0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~rmsprop_l0.05_e20_w0.0001_m0.9_cf_a0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~rmsprop_l0.01_e20_w0.0001_m0.9_cf_a0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~rmsprop_l0.005_e20_w0.0001_m0.9_cf_a0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~rmsprop_l0.001_e20_w0.0001_m0.9_cf_a0.99" -gpu 0 -x 1 -bs 32

python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~adam_l0.01_e20_w0.0001_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~adam_l0.005_e20_w0.0001_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~adam_l0.001_e20_w0.0001_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~adam_l0.0005_e20_w0.0001_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~adam_l0.0001_e20_w0.0001_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 32

python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~sgd_l1_e20_w0.0001_m0.9_nf" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~sgd_l0.5_e20_w0.0001_m0.9_nf" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~sgd_l0.1_e20_w0.0001_m0.9_nf" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~sgd_l0.05_e20_w0.0001_m0.9_nf" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~sgd_l0.01_e20_w0.0001_m0.9_nf" -gpu 0 -x 1 -bs 32

python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~taso_l1_e20_w0.0001_m0.9_nf_a25_b0.7_l4" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~taso_l0.5_e20_w0.0001_m0.9_nf_a25_b0.7_l4" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~taso_l0.1_e20_w0.0001_m0.9_nf_a25_b0.7_l4" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~taso_l0.05_e20_w0.0001_m0.9_nf_a25_b0.7_l4" -gpu 0 -x 1 -bs 32
python train.py -ei="tune1" -et "cnn_train" -ec "data~agnews+model~textrnn+optim~taso_l0.01_e20_w0.0001_m0.9_nf_a25_b0.7_l4" -gpu 0 -x 1 -bs 32
