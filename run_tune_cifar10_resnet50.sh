#!/usr/bin/env bash

#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l0.1_e100_w0.0001_d0_a0" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l0.05_e100_w0.0001_d0_a0" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l0.01_e100_w0.0001_d0_a0" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l0.5_e100_w0.0001_d0_a0" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l1_e100_w0.0001_d0_a0" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l0.1_e100_w0_d0_a0" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l0.05_e100_w0_d0_a0" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l0.01_e100_w0_d0_a0" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l0.5_e100_w0_d0_a0" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l1_e100_w0_d0_a0" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l0.1_e100_w0.0001_d0_a0.1" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l0.05_e100_w0.0001_d0_a0.1" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l0.01_e100_w0.0001_d0_a0.1" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l0.5_e100_w0.0001_d0_a0.1" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l1_e100_w0.0001_d0_a0.1" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l0.1_e100_w0_d0_a0.1" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l0.05_e100_w0_d0_a0.1" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l0.01_e100_w0_d0_a0.1" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l0.5_e100_w0_d0_a0.1" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adagrad_l1_e100_w0_d0_a0.1" -gpu 0 -x 1 -bs 64

#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.0001_e100_w0.0001_m0_cf_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.0005_e100_w0.0001_m0_cf_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.01_e100_w0.0001_m0_cf_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.005_e100_w0.0001_m0_cf_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.001_e100_w0.0001_m0_cf_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.0001_e100_w0_m0_cf_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.0005_e100_w0_m0_cf_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.01_e100_w0_m0_cf_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.005_e100_w0_m0_cf_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.001_e100_w0_m0_cf_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.0001_e100_w0.0001_m0_ct_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.0005_e100_w0.0001_m0_ct_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.01_e100_w0.0001_m0_ct_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.005_e100_w0.0001_m0_ct_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.001_e100_w0.0001_m0_ct_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.0001_e100_w0_m0_ct_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.0005_e100_w0_m0_ct_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.01_e100_w0_m0_ct_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.005_e100_w0_m0_ct_a0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~rmsprop_l0.001_e100_w0_m0_ct_a0.99" -gpu 0 -x 1 -bs 64

#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.00001_e100_w0.0001_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.00005_e100_w0.0001_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.001_e100_w0.0001_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.0005_e100_w0.0001_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.0001_e100_w0.0001_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.00001_e100_w0_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.00005_e100_w0_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.001_e100_w0_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.0005_e100_w0_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.0001_e100_w0_af_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.00001_e100_w0.0001_at_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.00005_e100_w0.0001_at_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.001_e100_w0.0001_at_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.0005_e100_w0.0001_at_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.0001_e100_w0.0001_at_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.00001_e100_w0_at_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.00005_e100_w0_at_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.001_e100_w0_at_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.0005_e100_w0_at_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~adam_l0.0001_e100_w0_at_bf0.9_bs0.99" -gpu 0 -x 1 -bs 64

#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l1_e100_w0.0001_m0.9_nf_a25_b0.7" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l0.5_e100_w0.0001_m0.9_nf_a25_b0.7" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l0.1_e100_w0.0001_m0.9_nf_a25_b0.7" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l0.05_e100_w0.0001_m0.9_nf_a25_b0.7" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l0.01_e100_w0.0001_m0.9_nf_a25_b0.7" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l1_e100_w0.0001_m0.9_nt_a25_b0.7" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l0.5_e100_w0.0001_m0.9_nt_a25_b0.7" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l0.1_e100_w0.0001_m0.9_nt_a25_b0.7" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l0.05_e100_w0.0001_m0.9_nt_a25_b0.7" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l0.01_e100_w0.0001_m0.9_nt_a25_b0.7" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l1_e100_w0_m0.9_nf_a25_b0.7" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l0.5_e100_w0_m0.9_nf_a25_b0.7" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l0.1_e100_w0_m0.9_nf_a25_b0.7" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l0.05_e100_w0_m0.9_nf_a25_b0.7" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l0.01_e100_w0_m0.9_nf_a25_b0.7" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l1_e100_w0_m0.9_nt_a25_b0.7" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l0.5_e100_w0_m0.9_nt_a25_b0.7" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l0.1_e100_w0_m0.9_nt_a25_b0.7" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l0.05_e100_w0_m0.9_nt_a25_b0.7" -gpu 0 -x 1 -bs 64
#python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~taso_l0.01_e100_w0_m0.9_nt_a25_b0.7" -gpu 0 -x 1 -bs 64

python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l1_e100_w0.0001_m0.9_nf_l-6_u3" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l0.5_e100_w0.0001_m0.9_nf_l-6_u3" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l0.1_e100_w0.0001_m0.9_nf_l-6_u3" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l0.05_e100_w0.0001_m0.9_nf_l-6_u3" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l0.01_e100_w0.0001_m0.9_nf_l-6_u3" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l1_e100_w0.0001_m0.9_nt_l-6_u3" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l0.5_e100_w0.0001_m0.9_nt_l-6_u3" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l0.1_e100_w0.0001_m0.9_nt_l-6_u3" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l0.05_e100_w0.0001_m0.9_nt_l-6_u3" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l0.01_e100_w0.0001_m0.9_nt_l-6_u3" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l1_e100_w0_m0.9_nf_l-6_u3" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l0.5_e100_w0_m0.9_nf_l-6_u3" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l0.1_e100_w0_m0.9_nf_l-6_u3" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l0.05_e100_w0_m0.9_nf_l-6_u3" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l0.01_e100_w0_m0.9_nf_l-6_u3" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l1_e100_w0_m0.9_nt_l-6_u3" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l0.5_e100_w0_m0.9_nt_l-6_u3" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l0.1_e100_w0_m0.9_nt_l-6_u3" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l0.05_e100_w0_m0.9_nt_l-6_u3" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~htd_l0.01_e100_w0_m0.9_nt_l-6_u3" -gpu 0 -x 1 -bs 64

python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l1_e100_w0.0001_m0.9_nf" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l0.5_e100_w0.0001_m0.9_nf" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l0.1_e100_w0.0001_m0.9_nf" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l0.05_e100_w0.0001_m0.9_nf" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l0.01_e100_w0.0001_m0.9_nf" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l1_e100_w0.0001_m0.9_nt" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l0.5_e100_w0.0001_m0.9_nt" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l0.1_e100_w0.0001_m0.9_nt" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l0.05_e100_w0.0001_m0.9_nt" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l0.01_e100_w0.0001_m0.9_nt" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l1_e100_w0_m0.9_nf" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l0.5_e100_w0_m0.9_nf" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l0.1_e100_w0_m0.9_nf" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l0.05_e100_w0_m0.9_nf" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l0.01_e100_w0_m0.9_nf" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l1_e100_w0_m0.9_nt" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l0.5_e100_w0_m0.9_nt" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l0.1_e100_w0_m0.9_nt" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l0.05_e100_w0_m0.9_nt" -gpu 0 -x 1 -bs 64
python train.py -ei="tune" -et "cnn_train" -ec "data~cifar10+model~resnet50+optim~cos_l0.01_e100_w0_m0.9_nt" -gpu 0 -x 1 -bs 64
