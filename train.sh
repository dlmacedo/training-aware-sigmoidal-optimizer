#!/usr/bin/env bash

#file="${0##*/}"
#file=${file%%.*}
#echo $file

#export ei=""
#export et="cnn.train"
#export ec="data~mnist+model~lenet5+loss~cel+fpec~999999+part~1:data~mnist+model~lenet5+loss~del+spec~999999+part~2"
#export ec="data~mnist+model~lenet5+loss~cel+fpec~999999+part~1"
#python execute.py -ei=$ei -et $et -ec $ec -x 1 -e 300
#export ei="cnn.train/data~mnist+model~lenet5+loss~cel+fpec~999999+part~1:cnn.train/data~mnist+model~lenet5+loss~del+spec~999999+part~2"
#export et="cnn.finetune"
#export ec="data~mnist+model~lenet5+loss~cel+fpec~999999+part~1"
#python execute.py -ei=$ei -et $et -ec $ec -x 1 -e 5
#export et="ml.train"
#export ec="data~mnist+model~svm+kernel~gauss+fpec~999999+part~1"
#python execute.py -ei=$ei -et $et -ec $ec -x 1 -e 5
##python analize_odd.py

export ei=""
export et="cnn.train"
#export ec="data~cifar10+model~vgg16bn_+loss~cel+fpec~999999+part~1:data~cifar10+model~vgg16bn_+loss~del+fpec~999999+part~1"
export ec="data~cifar10+model~densenet100bc_+loss~cel+fpec~999999+part~1"
python execute.py -ei=$ei -et $et -ec $ec -x 1 -e 1000 -lr 0.01 -lrdr 1 -bs 128 -mm 0 -wd 0
#export ei="cnn.train/data~cifar10+model~vgg16bn_+loss~cel+fpec~999999+part~1:cnn.train/data~cifar10+model~vgg16bn_+loss~del+fpec~999999+part~1"
#export et="cnn.finetune"
#export ec="data~cifar10+model~vgg16bn_+loss~cel+fpec~999999+part~1"
#python execute.py -ei=$ei -et $et -ec $ec -x 1 -e 5
#export et="ml.train"
#export ec="data~cifar10+model~svm+kernel~gauss+fpec~999999+part~1"
#python execute.py -ei=$ei -et $et -ec $ec -x 1 -e 5
##python analize_odd.py
