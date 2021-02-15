#!/usr/bin/env bash

#python train.py -ei="odd3" -et "cnn_train" -ec "data~cifar100+model~densenetbc100+loss~sml1_naa_id_no_no_no_no" -gpu 0
#python train.py -ei="odd3" -et "cnn_train" -ec "data~cifar100+model~densenetbc100+loss~dml1_pn2_id_no_no_no_10_no_no_0" -gpu 0
python train.py -ei="odd3" -et "cnn_train" -ec "data~cifar100+model~densenetbc100+loss~eml1_pn2_id_no_no_lz_10_sc1_no_0.1_sx1" -gpu 0
#python train.py -ei="odd3" -et "cnn_train" -ec "data~cifar100+model~resnet34+loss~sml1_naa_id_no_no_no_no" -gpu 0
#python train.py -ei="odd3" -et "cnn_train" -ec "data~cifar100+model~resnet34+loss~dml1_pn2_id_no_no_no_10_no_no_0" -gpu 0
#python train.py -ei="odd3" -et "cnn_train" -ec "data~cifar100+model~resnet34+loss~eml1_pn2_id_no_no_lz_10_sc1_no_0.1_sx1" -gpu 0

#python OOD_Baseline_and_ODIN.py --dir odd3 --dataset cifar100 --net_type densenetbc100 --loss sml1_naa_id_no_no_no_no --gpu 0
#python OOD_Baseline_and_ODIN.py --dir odd3 --dataset cifar100 --net_type densenetbc100 --loss dml1_pn2_id_no_no_no_10_no_no_0 --gpu 0
python OOD_Baseline_and_ODIN.py --dir odd3 --dataset cifar100 --net_type densenetbc100 --loss eml1_pn2_id_no_no_lz_10_sc1_no_0.1_sx1 --gpu 0
#python OOD_Baseline_and_ODIN.py --dir odd3 --dataset cifar100 --net_type resnet34 --loss sml1_naa_id_no_no_no_no --gpu 0
#python OOD_Baseline_and_ODIN.py --dir odd3 --dataset cifar100 --net_type resnet34 --loss dml1_pn2_id_no_no_no_10_no_no_0 --gpu 0
#python OOD_Baseline_and_ODIN.py --dir odd3 --dataset cifar100 --net_type resnet34 --loss eml1_pn2_id_no_no_lz_10_sc1_no_0.1_sx1 --gpu 0

#python train.py -ei="odd3" -et "cnn_odd_infer" -ec "data~cifar100+model~densenetbc100+loss~sml1_naa_id_no_no_no_no" -gpu 0
#python train.py -ei="odd3" -et "cnn_odd_infer" -ec "data~cifar100+model~densenetbc100+loss~dml1_pn2_id_no_no_no_10_no_no_0" -gpu 0
python train.py -ei="odd3" -et "cnn_odd_infer" -ec "data~cifar100+model~densenetbc100+loss~eml1_pn2_id_no_no_lz_10_sc1_no_0.1_sx1" -gpu 0
#python train.py -ei="odd3" -et "cnn_odd_infer" -ec "data~cifar100+model~resnet34+loss~sml1_naa_id_no_no_no_no" -gpu 0
#python train.py -ei="odd3" -et "cnn_odd_infer" -ec "data~cifar100+model~resnet34+loss~dml1_pn2_id_no_no_no_10_no_no_0" -gpu 0
#python train.py -ei="odd3" -et "cnn_odd_infer" -ec "data~cifar100+model~resnet34+loss~eml1_pn2_id_no_no_lz_10_sc1_no_0.1_sx1" -gpu 0

#python train.py -ei="odd3" -et "cnn_train" -ec "data~cifar10+model~densenetbc100+loss~dml1_pn2_id_no_1.0_no_no" -gpu 0
#python train.py -ei="odd3" -et "cnn_train" -ec "data~imagenet2012+model~resnet18_+loss~sml1_na_id_no_no_no_no" -gpu 0
