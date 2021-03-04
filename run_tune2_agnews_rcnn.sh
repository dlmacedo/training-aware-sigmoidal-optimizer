#!/usr/bin/env bash

python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~adagrad_l0.05_e20_w0.0001_d0_a0" -gpu 0 -x 1 -bs 32
python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~adagrad_l0.05_e20_w0_d0_a0" -gpu 0 -x 1 -bs 32
python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~adagrad_l0.05_e20_w0.0001_d0_a0.1" -gpu 0 -x 1 -bs 32
python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~adagrad_l0.05_e20_w0_d0_a0.1" -gpu 0 -x 1 -bs 32

python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~rmsprop_l0.001_e20_w0.0001_m0.9_cf_a0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~rmsprop_l0.001_e20_w0_m0.9_cf_a0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~rmsprop_l0.001_e20_w0.0001_m0_cf_a0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~rmsprop_l0.001_e20_w0_m0_cf_a0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~rmsprop_l0.001_e20_w0.0001_m0.9_ct_a0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~rmsprop_l0.001_e20_w0_m0.9_ct_a0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~rmsprop_l0.001_e20_w0.0001_m0_ct_a0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~rmsprop_l0.001_e20_w0_m0_ct_a0.99" -gpu 0 -x 1 -bs 32

python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~adam_l0.001_e20_w0.0001_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~adam_l0.001_e20_w0_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~adam_l0.001_e20_w0.0001_at_bf0.9_bs0.99" -gpu 0 -x 1 -bs 32
python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~adam_l0.001_e20_w0_at_bf0.9_bs0.99" -gpu 0 -x 1 -bs 32

python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~sgd_l0.01_e20_w0.0001_m0.9_nf" -gpu 0 -x 1 -bs 32
python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~sgd_l0.01_e20_w0_m0.9_nf" -gpu 0 -x 1 -bs 32
python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~sgd_l0.01_e20_w0.0001_m0.9_nt" -gpu 0 -x 1 -bs 32
python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~sgd_l0.01_e20_w0_m0.9_nt" -gpu 0 -x 1 -bs 32

python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~taso_l0.05_e20_w0.0001_m0.9_nf_a25_b0.7_l4" -gpu 0 -x 1 -bs 32
python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~taso_l0.05_e20_w0_m0.9_nf_a25_b0.7_l4" -gpu 0 -x 1 -bs 32
python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~taso_l0.05_e20_w0.0001_m0.9_nt_a25_b0.7_l4" -gpu 0 -x 1 -bs 32
python train.py -ei="tune2" -et "cnn_train" -ec "data~agnews+model~rcnn+optim~taso_l0.05_e20_w0_m0.9_nt_a25_b0.7_l4" -gpu 0 -x 1 -bs 32
