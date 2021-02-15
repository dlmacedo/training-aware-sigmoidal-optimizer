import os
import sys
import torch
import torch.nn as nn
import models
import loaders
import losses
import statistics
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchnet as tnt
import numpy as np
import time
import utils
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
import matplotlib.pyplot as plt
import data_loader
import pickle

sns.set(style="darkgrid")


class CNNAgent:

    def __init__(self, args):

        self.args = args
        self.epoch = None
        self.cluster_predictions_transformation = []

        # create dataset
        image_loaders = loaders.ImageLoader(args)
        (self.trainset_first_partition_loader_for_train,
         self.trainset_second_partition_loader_for_train,
         self.trainset_first_partition_loader_for_infer,
         self.trainset_second_partition_loader_for_infer,
         self.valset_loader) = image_loaders.get_loaders()
        if self.args.partition == "1":
            self.trainset_loader_for_train = self.trainset_first_partition_loader_for_train
        elif self.args.partition == "2":
            self.trainset_loader_for_train = self.trainset_second_partition_loader_for_train
        print("\nDATASET:", args.dataset_full)

        # create model
        torch.manual_seed(self.args.execution_seed)
        torch.cuda.manual_seed(self.args.execution_seed)
        print("=> creating model '{}'".format(self.args.model_name))
        if self.args.model_name == "densenetbc100":
            self.model = models.DenseNet3(
                100, int(self.args.number_of_model_classes), loss=self.args.loss)
        elif self.args.model_name == "resnet32":
            self.model = models.ResNet32(
                num_c=self.args.number_of_model_classes, loss=self.args.loss)
        elif self.args.model_name == "resnet34":
            self.model = models.ResNet34(
                num_c=self.args.number_of_model_classes, loss=self.args.loss)
        elif self.args.model_name == "resnet18_":
            self.model = models.resnet34_(
                num_classes=self.args.number_of_model_classes, loss=self.args.loss)
        elif self.args.model_name == "resnet34_":
            self.model = models.resnet34_(
                num_classes=self.args.number_of_model_classes, loss=self.args.loss)
        elif self.args.model_name == "resnet101_":
            self.model = models.resnet101_(
                num_classes=self.args.number_of_model_classes, loss=self.args.loss)
        elif self.args.model_name == "wideresnet3410":
            self.model = models.Wide_ResNet(
                depth=34, widen_factor=10, num_classes=self.args.number_of_model_classes, loss=self.args.loss)
        #elif self.args.model_name == "vgg":
        #    self.model = models.VGG19(num_classes=self.args.number_of_model_classes, loss=self.args.loss)
        self.model.cuda()
        torch.manual_seed(self.args.base_seed)
        torch.cuda.manual_seed(self.args.base_seed)

        # print and save model arch...
        if self.args.exp_type == "cnn_train":
            print("\nMODEL:", self.model)
            with open(os.path.join(self.args.experiment_path, 'model.arch'), 'w') as file:
                print(self.model, file=file)

        # create loss
        self.criterion = losses.GenericLossSecondPart(self.model.classifier).cuda()

        # create train
        if self.args.loss.startswith("eml"):
            parameters = self.parameters_plus(
                self.args.loss.split("_")[8], float(self.args.loss.split("_")[9]), self.args.loss.split("_")[5].startswith("f"))
            #print(parameters)
            self.optimizer = torch.optim.SGD(
                parameters,
                lr=self.args.original_learning_rate,
                momentum=self.args.momentum,
                nesterov=True,
                weight_decay=args.weight_decay
            )
        elif self.args.loss.startswith("kml"):
            parameters = self.parameters_plus(
                self.args.loss.split("_")[8], float(self.args.loss.split("_")[9]), self.args.loss.split("_")[5].startswith("f"))
            #print(parameters)
            self.optimizer = torch.optim.SGD(
                parameters,
                lr=self.args.original_learning_rate,
                momentum=self.args.momentum,
                nesterov=True,
                weight_decay=args.weight_decay
            )
        else:
            parameters = self.model.parameters()
            #print(parameters)
            self.optimizer = torch.optim.SGD(
                parameters,
                lr=self.args.original_learning_rate,
                momentum=self.args.momentum,
                nesterov=True,
                weight_decay=args.weight_decay
            )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.args.learning_rate_decay_epochs, gamma=args.learning_rate_decay_rate)
        print("\nTRAIN:", self.criterion, self.optimizer, self.scheduler)

    def parameters_weight_decay(self, special_weight_decay_list=()):
        regular_parameters, no_weight_decay_parameters = [], []
        print()
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in special_weight_decay_list:
                print("NO WEIGHT DECAY:", name)
                no_weight_decay_parameters.append(param)
                #regular_parameters.append(param)
            else:
                regular_parameters.append(param)
        print()
        return [
            {'params': regular_parameters},
            {'params': no_weight_decay_parameters, 'weight_decay': 0.}]

    def parameters_plus(self, weight_decay, learn_rate, frozen_prototypes):
        regular_parameters, classifier_prototypes_parameters, classifier_scales_tilts = [], [], []
        print()
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if name == "classifier.weights":
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                print("SPECIAL OPTIMIZATION PROTOTYPES:", name)
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                classifier_prototypes_parameters.append(param)
            elif name == "classifier.scales":
                print("##########################################################")
                print("SPECIAL OPTIMIZATION SCALES AND TILTS:", name)
                print("##########################################################")
                classifier_scales_tilts.append(param)
            elif name == "classifier.tilts":
                print("##########################################################")
                print("SPECIAL OPTIMIZATION SCALES AND TILTS:", name)
                print("##########################################################")
                classifier_scales_tilts.append(param)
            else:
                regular_parameters.append(param)
        print()
        if (weight_decay == "WD") and (not frozen_prototypes):
            print("\nLEARNABLE PROTOTYPES")
            print("WEIGHT DECAY SCALES AND TILTS\n")
            dict_to_return = [
            {'params': regular_parameters},
            {'params': classifier_prototypes_parameters},
            {'params': classifier_scales_tilts, 'lr': learn_rate}]
        elif (weight_decay == "NO") and (not frozen_prototypes):
            print("\nLEARNABLE PROTOTYPES")
            print("NO WEIGHT DECAY SCALES AND TILTS\n")
            dict_to_return = [
            {'params': regular_parameters},
            {'params': classifier_prototypes_parameters},
            {'params': classifier_scales_tilts, 'lr': learn_rate, 'weight_decay': 0.}]
        if (weight_decay == "WD") and (frozen_prototypes):
            print("\nFROZEN PROTOTYPES")
            print("WEIGHT DECAY SCALES AND TILTS\n")
            dict_to_return = [
            {'params': regular_parameters},
            {'params': classifier_prototypes_parameters, 'lr': 0., 'weight_decay': 0.},
            {'params': classifier_scales_tilts, 'lr': learn_rate}]
        elif (weight_decay == "NO") and (frozen_prototypes):
            print("\nFROZEN PROTOTYPES")
            print("NO WEIGHT DECAY SCALES AND TILTS\n")
            dict_to_return = [
            {'params': regular_parameters},
            {'params': classifier_prototypes_parameters, 'lr': 0., 'weight_decay': 0.},
            {'params': classifier_scales_tilts, 'lr': learn_rate, 'weight_decay': 0.}]
        return dict_to_return

    """
    def parameters_xml_plus(self, weight_decay, learn_rate):
        regular_parameters, classifier_scales_tilts_bias = [], []
        print()
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue 
            if name == "classifier.scales":
                print("##########################################################")
                print("SPECIAL OPTIMIZATION SCALES, TILTS, AND BIAS:", name)
                print("##########################################################")
                classifier_scales_tilts_bias.append(param)
            elif name == "classifier.tilts":
                print("##########################################################")
                print("SPECIAL OPTIMIZATION SCALES, TILTS, AND BIAS:", name)
                print("##########################################################")
                classifier_scales_tilts_bias.append(param)
            elif name == "classifier.bias":
                print("##########################################################")
                print("SPECIAL OPTIMIZATION SCALES, TILTS, AND BIAS:", name)
                print("##########################################################")
                classifier_scales_tilts_bias.append(param)
            else:
                regular_parameters.append(param)
        print()
        if (weight_decay == "WD"):
            print("WEIGHT DECAY SCALES, TILTS, AND BIAS\n")
            dict_to_return = [
            {'params': regular_parameters},
            {'params': classifier_scales_tilts_bias, 'lr': learn_rate}]
        elif (weight_decay == "NO"):
            print("NO WEIGHT DECAY SCALES, TILTS, AND BIAS\n")
            dict_to_return = [
            {'params': regular_parameters},
            {'params': classifier_scales_tilts_bias, 'lr': learn_rate, 'weight_decay': 0.}]
        #print(dict_to_return)
        return dict_to_return
    """

    """
    def parameters_zml_plus(self, weight_decay, learn_rate):
        regular_parameters, classifier_scales_tilts = [], []
        print()
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue 
            if name == "classifier.scales":
                print("##########################################################")
                print("SPECIAL OPTIMIZATION SCALES, TILTS AND NORMALIZATION:", name)
                print("##########################################################")
                classifier_scales_tilts.append(param)
            elif name == "classifier.tilts":
                print("##########################################################")
                print("SPECIAL OPTIMIZATION SCALES, TILTS AND NORMALIZATION:", name)
                print("##########################################################")
                classifier_scales_tilts.append(param)
            else:
                regular_parameters.append(param)
        print()
        if (weight_decay == "WD"):
            print("WEIGHT DECAY SCALES AND TILTS\n")
            dict_to_return = [
            {'params': regular_parameters},
            {'params': classifier_scales_tilts, 'lr': learn_rate}]
        elif (weight_decay == "NO"):
            print("NO WEIGHT DECAY SCALES AND TILTS\n")
            dict_to_return = [
            {'params': regular_parameters},
            {'params': classifier_scales_tilts, 'lr': learn_rate, 'weight_decay': 0.}]
        #print(dict_to_return)
        return dict_to_return
    """

    """
    def process_weight_decay_original(self, weight_decay, skip_list=()):
        decay, no_decay = [], []
        print()
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            #if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            if name in skip_list:
                print("NO WEIGHT DECAY:", name)
                no_decay.append(param)
            else:
                decay.append(param)
        print()
        return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]
    """
    
    def train_validate(self):
        # template for others procedures of this class...
        # building results and raw results files...
        if self.args.execution == 1:
            with open(self.args.executions_best_results_file_path, "w") as best_results:
                best_results.write(
                    "DATA,MODEL,LOSS,EXECUTION,EPOCH,"
                    "TRAIN LOSS,TRAIN ACC1,TRAIN ODD_ACC,"
                    #"TRAIN LOSS,TRAIN ODD_LOSS,TRAIN ACC1,TRAIN ODD_ACC,"
                    "TRAIN INTRA_LOGITS MEAN,TRAIN INTRA_LOGITS STD,TRAIN INTER_LOGITS MEAN,TRAIN INTER_LOGITS STD,"
                    "TRAIN MAX_PROBS MEAN,TRAIN MAX_PROBS STD,"
                    "TRAIN ENTROPIES MEAN,TRAIN ENTROPIES STD,"
                    "VALID LOSS,VALID ACC1,VALID ODD_ACC,"
                    #"VALID LOSS,VALID ODD_LOSS,VALID ACC1,VALID ODD_ACC,"
                    "VALID INTRA_LOGITS MEAN,VALID INTRA_LOGITS STD,VALID INTER_LOGITS MEAN,VALID INTER_LOGITS STD,"
                    "VALID MAX_PROBS MEAN,VALID MAX_PROBS STD,"
                    "VALID ENTROPIES MEAN,VALID ENTROPIES STD\n"
                )
            with open(self.args.executions_raw_results_file_path, "w") as raw_results:
                #raw_results.write("EXECUTION,EPOCH,SET,TYPE,VALUE\n")
                raw_results.write("DATA,MODEL,LOSS,EXECUTION,EPOCH,SET,METRIC,VALUE\n")

        print("\n################ TRAINING ################")

        #best_model_results = {"TRAIN LOSS": float("inf")}
        #best_model_results = {"TRAIN ODD_LOSS": float("inf")}
        best_model_results = {"VALID ACC1": 0}
        #best_model_results = {"VALID ODD_ACC": 0}

        for self.epoch in range(1, self.args.epochs + 1):
            print("\n######## EPOCH:", self.epoch, "OF", self.args.epochs, "########")

            # Adjusting learning rate (if not using reduce on plateau)...
            # self.scheduler.step()

            # Print current learning rate...
            for param_group in self.optimizer.param_groups:
                print("\nLEARNING RATE:\t\t", param_group["lr"])

            train_loss, train_acc1, train_odd_acc, train_epoch_logits, train_epoch_metrics = self.train_epoch()
            #train_loss, train_odd_loss, train_acc1, train_odd_acc, train_epoch_logits, train_epoch_metrics = self.train_epoch()

            # Adjusting learning rate (if not using reduce on plateau)...
            self.scheduler.step()
            valid_loss, valid_acc1, valid_odd_acc, valid_epoch_logits, valid_epoch_metrics = self.validate_epoch()
            #valid_loss, valid_odd_loss, valid_acc1, valid_odd_acc, valid_epoch_logits, valid_epoch_metrics = self.validate_epoch()

            train_intra_logits_mean = statistics.mean(train_epoch_logits["intra"])
            train_intra_logits_std = statistics.pstdev(train_epoch_logits["intra"])
            train_inter_logits_mean = statistics.mean(train_epoch_logits["inter"])
            train_inter_logits_std = statistics.pstdev(train_epoch_logits["inter"])
            #######################################################################
            train_max_probs_mean = statistics.mean(train_epoch_metrics["max_probs"])
            train_max_probs_std = statistics.pstdev(train_epoch_metrics["max_probs"])
            train_entropies_mean = statistics.mean(train_epoch_metrics["entropies"])
            train_entropies_std = statistics.pstdev(train_epoch_metrics["entropies"])
            #######################################################################
            valid_intra_logits_mean = statistics.mean(valid_epoch_logits["intra"])
            valid_intra_logits_std = statistics.pstdev(valid_epoch_logits["intra"])
            valid_inter_logits_mean = statistics.mean(valid_epoch_logits["inter"])
            valid_inter_logits_std = statistics.pstdev(valid_epoch_logits["inter"])
            #######################################################################
            valid_max_probs_mean = statistics.mean(valid_epoch_metrics["max_probs"])
            valid_max_probs_std = statistics.pstdev(valid_epoch_metrics["max_probs"])
            valid_entropies_mean = statistics.mean(valid_epoch_metrics["entropies"])
            valid_entropies_std = statistics.pstdev(valid_epoch_metrics["entropies"])
            #######################################################################

            print("\n####################################################")
            print("TRAIN MAX PROB MEAN:\t", train_max_probs_mean)
            print("TRAIN MAX PROB STD:\t", train_max_probs_std)
            print("VALID MAX PROB MEAN:\t", valid_max_probs_mean)
            print("VALID MAX PROB STD:\t", valid_max_probs_std)
            print("####################################################\n")

            print("\n####################################################")
            print("TRAIN ENTROPY MEAN:\t", train_entropies_mean)
            print("TRAIN ENTROPY STD:\t", train_entropies_std)
            print("VALID ENTROPY MEAN:\t", valid_entropies_mean)
            print("VALID ENTROPY STD:\t", valid_entropies_std)
            print("####################################################\n")

            with open(self.args.executions_raw_results_file_path, "a") as raw_results:
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "TRAIN", "LOSS", train_loss))
                #raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                #    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                #    "TRAIN", "ODD_LOSS", train_odd_loss))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "TRAIN", "ACC1", train_acc1))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "TRAIN", "ODD_ACC", train_odd_acc))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "TRAIN", "INTRA_LOGITS MEAN", train_intra_logits_mean))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "TRAIN", "INTRA_LOGITS STD", train_intra_logits_std))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "TRAIN", "INTER_LOGITS MEAN", train_inter_logits_mean))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "TRAIN", "INTER_LOGITS STD", train_inter_logits_std))
                #########################################################
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "TRAIN", "MAX_PROBS MEAN", train_max_probs_mean))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "TRAIN", "MAX_PROBS STD", train_max_probs_std))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "TRAIN", "ENTROPIES MEAN", train_entropies_mean))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "TRAIN", "ENTROPIES STD", train_entropies_std))
                #########################################################               
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "VALID", "LOSS", valid_loss))
                #raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                #    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                #    "VALID", "ODD_LOSS", valid_odd_loss))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "VALID", "ACC1", valid_acc1))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "VALID", "ODD_ACC", valid_odd_acc))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "VALID", "INTRA_LOGITS MEAN", valid_intra_logits_mean))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "VALID", "INTRA_LOGITS STD", valid_intra_logits_std))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "VALID", "INTER_LOGITS MEAN", valid_inter_logits_mean))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "VALID", "INTER_LOGITS STD", valid_inter_logits_std))
                #########################################################
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "VALID", "MAX_PROBS MEAN", valid_max_probs_mean))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "VALID", "MAX_PROBS STD", valid_max_probs_std))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "VALID", "ENTROPIES MEAN", valid_entropies_mean))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.loss, self.args.execution, self.epoch,
                    "VALID", "ENTROPIES STD", valid_entropies_std))
                #########################################################               

            print()
            print("TRAIN ==>>\tIADM: {0:.8f}\tIADS: {1:.8f}\tIEDM: {2:.8f}\tIEDS: {3:.8f}".format(
                train_intra_logits_mean, train_intra_logits_std, train_inter_logits_mean, train_inter_logits_std))
            print("VALID ==>>\tIADM: {0:.8f}\tIADS: {1:.8f}\tIEDM: {2:.8f}\tIEDS: {3:.8f}".format(
                valid_intra_logits_mean, valid_intra_logits_std, valid_inter_logits_mean, valid_inter_logits_std))
            print()

            #############################################
            print("\nDATA:", self.args.dataset_full)
            print("MODEL:", self.args.model_name)
            print("LOSS:", self.args.loss, "\n")
            #############################################

            # if is best...
            #if train_loss < best_model_results["TRAIN LOSS"]:
            #if train_odd_loss < best_model_results["TRAIN ODD_LOSS"]:
            if valid_acc1 > best_model_results["VALID ACC1"]:
            #if valid_odd_acc > best_model_results["VALID ODD_ACC"]:
                print("!+NEW BEST MODEL VALID ACC1!")
                best_model_results = {
                    "DATA": self.args.dataset_full,
                    "MODEL": self.args.model_name,
                    "LOSS": self.args.loss,
                    "EXECUTION": self.args.execution,
                    "EPOCH": self.epoch,
                    ###########################################################################
                    "TRAIN LOSS": train_loss,
                    #"TRAIN ODD_LOSS": train_odd_loss,
                    "TRAIN ACC1": train_acc1,
                    "TRAIN ODD_ACC": train_odd_acc,
                    "TRAIN INTRA_LOGITS MEAN": train_intra_logits_mean,
                    "TRAIN INTRA_LOGITS STD": train_intra_logits_std,
                    "TRAIN INTER_LOGITS MEAN": train_inter_logits_mean,
                    "TRAIN INTER_LOGITS STD": train_inter_logits_std,
                    ###########################################################################
                    "TRAIN MAX_PROBS MEAN": train_max_probs_mean,
                    "TRAIN MAX_PROBS STD": train_max_probs_std,
                    "TRAIN ENTROPIES MEAN": train_entropies_mean,
                    "TRAIN ENTROPIES STD": train_entropies_std,
                    ###########################################################################
                    "VALID LOSS": valid_loss,
                    #"VALID ODD_LOSS": valid_odd_loss,
                    "VALID ACC1": valid_acc1,
                    "VALID ODD_ACC": valid_odd_acc,
                    "VALID INTRA_LOGITS MEAN": valid_intra_logits_mean,
                    "VALID INTRA_LOGITS STD": valid_intra_logits_std,
                    "VALID INTER_LOGITS MEAN": valid_inter_logits_mean,
                    "VALID INTER_LOGITS STD": valid_inter_logits_std,
                    ###########################################################################
                    "VALID MAX_PROBS MEAN": valid_max_probs_mean,
                    "VALID MAX_PROBS STD": valid_max_probs_std,
                    "VALID ENTROPIES MEAN": valid_entropies_mean,
                    "VALID ENTROPIES STD": valid_entropies_std,
                    ###########################################################################
                }

                #print("!+NEW BEST MODEL TRAIN LOSS:\t\t{0:.4f} IN EPOCH {1}! SAVING {2}".format(
                #    train_loss, self.epoch, self.args.best_model_file_path))
                #print("!+NEW BEST MODEL TRAIN ODD LOSS:\t\t{0:.4f} IN EPOCH {1}! SAVING {2}".format(
                #    train_odd_loss, self.epoch, self.args.best_model_file_path))
                print("!+NEW BEST MODEL VALID ACC1:\t\t{0:.4f} IN EPOCH {1}! SAVING {2}\n".format(
                    valid_acc1, self.epoch, self.args.best_model_file_path))
                #print("!+NEW BEST MODEL VALID ODD ACC:\t\t{0:.4f} IN EPOCH {1}! SAVING {2}\n".format(
                #    valid_odd_acc, self.epoch, self.args.best_model_file_path))

                torch.save(self.model.state_dict(), self.args.best_model_file_path)
                #torch.save(self.model.state_dict(), self.args.best_model_file_alternative_path)

                np.save(os.path.join(
                    self.args.experiment_path, "best_model"+str(self.args.execution)+"_train_epoch_logits.npy"), train_epoch_logits)
                np.save(os.path.join(
                    self.args.experiment_path, "best_model"+str(self.args.execution)+"_train_epoch_metrics.npy"), train_epoch_metrics)
                np.save(os.path.join(
                    self.args.experiment_path, "best_model"+str(self.args.execution)+"_valid_epoch_logits.npy"), valid_epoch_logits)
                np.save(os.path.join(
                    self.args.experiment_path, "best_model"+str(self.args.execution)+"_valid_epoch_metrics.npy"), valid_epoch_metrics)
                #filename = os.path.join(
                #    self.args.experiment_path, "best_model"+str(self.args.execution)+"_train_epoch_entropies_per_classes.pkl")
                #with open(filename, 'wb') as file:  # Overwrites any existing file.
                #    pickle.dump(train_epoch_entropies_per_classes, file, pickle.HIGHEST_PROTOCOL)
                #with open(filename, "rb") as file:
                #    testando = pickle.load(file)
                #    print(testando)
                #filename = os.path.join(
                #    self.args.experiment_path, "best_model"+str(self.args.execution)+"_valid_epoch_entropies_per_classes.pkl")
                #with open(filename, 'wb') as file:  # Overwrites any existing file.
                #    pickle.dump(valid_epoch_entropies_per_classes, file, pickle.HIGHEST_PROTOCOL)
                #with open(filename, "rb") as file:
                #    testando = pickle.load(file)
                #    print(testando)

            print('!$$$$ BEST MODEL TRAIN ACC1:\t\t{0:.4f}'.format(best_model_results["TRAIN ACC1"]))
            print('!$$$$ BEST MODEL VALID ACC1:\t\t{0:.4f}'.format(best_model_results["VALID ACC1"]))
            ########################################################################################################
            ########################################################################################################
            #print('!$$$$ BEST MODEL TRAIN ODD ACC:\t\t{0:.4f}'.format(best_model_results["TRAIN ODD_ACC"]))
            #print('!$$$$ BEST MODEL VALID ODD ACC:\t\t{0:.4f}'.format(best_model_results["VALID ODD_ACC"]))
            ########################################################################################################
            ########################################################################################################

            # Adjusting learning rate (if using reduce on plateau)...
            # scheduler.step(valid_acc1)

        with open(self.args.executions_best_results_file_path, "a") as best_results:
            #best_results.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            best_results.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                best_model_results["DATA"],
                best_model_results["MODEL"],
                best_model_results["LOSS"],
                best_model_results["EXECUTION"],
                best_model_results["EPOCH"],
                #############################################
                best_model_results["TRAIN LOSS"],
                #best_model_results["TRAIN ODD_LOSS"],
                best_model_results["TRAIN ACC1"],
                best_model_results["TRAIN ODD_ACC"],
                best_model_results["TRAIN INTRA_LOGITS MEAN"],
                best_model_results["TRAIN INTRA_LOGITS STD"],
                best_model_results["TRAIN INTER_LOGITS MEAN"],
                best_model_results["TRAIN INTER_LOGITS STD"],
                #############################################
                best_model_results["TRAIN MAX_PROBS MEAN"],
                best_model_results["TRAIN MAX_PROBS STD"],
                best_model_results["TRAIN ENTROPIES MEAN"],
                best_model_results["TRAIN ENTROPIES STD"],
                #############################################
                best_model_results["VALID LOSS"],
                #best_model_results["VALID ODD_LOSS"],
                best_model_results["VALID ACC1"],
                best_model_results["VALID ODD_ACC"],
                best_model_results["VALID INTRA_LOGITS MEAN"],
                best_model_results["VALID INTRA_LOGITS STD"],
                best_model_results["VALID INTER_LOGITS MEAN"],
                best_model_results["VALID INTER_LOGITS STD"],
                #############################################
                best_model_results["VALID MAX_PROBS MEAN"],
                best_model_results["VALID MAX_PROBS STD"],
                best_model_results["VALID ENTROPIES MEAN"],
                best_model_results["VALID ENTROPIES STD"],
                #############################################
                )
            )

        # extracting features from best model...
        self.extract_features_for_all_sets(self.args.best_model_file_path)
        print()

    def train_epoch(self):
        print()
        # switch to train mode
        self.model.train()
        self.criterion.train()

        # Meters...
        loss_meter = utils.MeanMeter()
        #odd_loss_meter = utils.MeanMeter()
        accuracy_meter = tnt.meter.ClassErrorMeter(topk=[1], accuracy=True)
        odd_accuracy_meter = tnt.meter.ClassErrorMeter(topk=[1], accuracy=True)
        epoch_logits = {"intra": [], "inter": []}
        epoch_metrics = {"max_probs": [], "entropies": []}
        #epoch_entropies_per_classes =  [[] for i in range(self.model.classifier.weights.size(0))]

        for batch_index, (inputs, targets) in enumerate(self.trainset_loader_for_train):
            batch_index += 1

            # moving to GPU...
            inputs = inputs.cuda()
            targets = targets.cuda(non_blocking=True)

            # adding noise...
            #noisy_data = torch.clamp(noisy_data, min_pixel, max_pixel)

            #"""
            # preparing structural augmentation...
            if self.args.loss.split("_")[4] != "no":
                #print("entering 1!!!")
                #lam = np.random.beta(args.beta, args.beta)
                #lam = 0.5
                #batch_slice = 1/self.args.loss.split("_")[4][0]
                #slice_size = inputs.size()[0]//int(self.args.loss.split("_")[4][0])
                #slice_size = 2**int(self.args.loss.split("_")[4][0])
                slice_size = inputs.size(0)//(2**int(self.args.loss.split("_")[4][0]))
                #print(slice_size)
                W = inputs.size(2)
                H = inputs.size(3)
                #rand_index = torch.randperm(input.size()[0]).cuda()
                rand_index = torch.randperm(slice_size).cuda()
                #print(rand_index)
                #target_a = target
                #target_b = target[rand_index]
                #bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                #bbx1, bby1, bbx2, bby2 = utils.rand_bbox(inputs.size(), lam)
                #inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]  
                ####################################################################################          
                ####################################################################################          
                """
                plt.imshow(inputs[0].cpu().permute(1, 2, 0))  
                plt.show()      
                plt.imshow(inputs[rand_index[0]].cpu().permute(1, 2, 0))  
                plt.show()      
                """
                ####################################################################################          
                ####################################################################################          
                r = np.random.rand(1)
                if r < 0.5:
                    inputs[:slice_size, :, int(W/2):, :] = inputs[rand_index, :, int(W/2):, :]
                else:
                    inputs[:slice_size, :, :, int(H/2):] = inputs[rand_index, :, :, int(H/2):]    
                ####################################################################################          
                ####################################################################################          
                """
                plt.imshow(inputs[0].cpu().permute(1, 2, 0))  
                plt.show()      
                plt.imshow(inputs[slice_size].cpu().permute(1, 2, 0))  
                plt.show()      
                """
                ##########################################################################          
                ##########################################################################          
                # adjust lambda to exactly match pixel ratio
                #lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                #print(lam)
            #"""

            # compute output
            features = self.model(inputs)

            # compute loss
            # loss, intra_logits, and inter_logits are already allways using the correct batch size in the bellow line of code...
            loss, outputs, odd_outputs, intra_logits, inter_logits = self.criterion(features, targets)
          
            # executing structural augmentation...
            if self.args.loss.split("_")[4] != "no":
                #print("entering 3!!!")
                #print(slice_size)
                augmentation_multiplier = float(self.args.loss.split("_")[4][1:])                   
                #print(augmentation_multiplier)
                ############################################################
                uniform_dist = torch.Tensor(slice_size, self.args.number_of_model_classes).fill_((1./self.args.number_of_model_classes)).cuda()
                kl_divergence = F.kl_div(F.log_softmax(odd_outputs[:slice_size], dim=1), uniform_dist, reduction='batchmean')
                augmentation = augmentation_multiplier * kl_divergence
                ############################################################
                ####uniform_dist = torch.Tensor(slice_size, self.args.number_of_model_classes).fill_((1./self.args.number_of_model_classes)).cuda()
                ####kl_divergence = F.kl_div(uniform_dist, F.log_softmax(odd_outputs[:slice_size], dim=1), reduction='batchmean')
                ####augmentation = augmentation_multiplier * kl_divergence
                ############################################################
                # cross-entropy from softmax distribution to uniform distribution
                #loss += 0.5 * -(x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()
                #cross_entropy = -(odd_outputs[:slice_size].mean(1) - torch.logsumexp(odd_outputs[:slice_size], dim=1)).mean()
                #augmentation = augmentation_multiplier * cross_entropy
                ############################################################
                #cross_entropy = -(1./self.args.number_of_model_classes * F.log_softmax(odd_outputs[:slice_size], dim=1)).sum(dim=1).mean()
                #augmentation = augmentation_multiplier * cross_entropy
                ############################################################
                ####cross_entropy = -(F.softmax(odd_outputs[:slice_size], dim=1) * math.log(1./self.args.number_of_model_classes)).sum(dim=1).mean()
                ####augmentation = augmentation_multiplier * cross_entropy
                ############################################################
                #entropy = utils.entropies_from_logits(odd_outputs[:slice_size]).mean()
                #augmentation = - augmentation_multiplier * entropy
                ############################################################
                #entropy = utils.entropies_from_logits(odd_outputs[:slice_size]).mean()
                #augmentation = augmentation_multiplier * entropy
                ############################################################
                inputs = inputs[slice_size:]
                targets = targets[slice_size:]
                outputs = outputs[slice_size:]
                odd_outputs = odd_outputs[slice_size:]
                loss += augmentation
            else:
                augmentation = torch.tensor(0)

            ##########################################################################
            ##########################################################################
            max_probs = nn.Softmax(dim=1)(odd_outputs).max(dim=1)[0]
            entropies = utils.entropies_from_logits(odd_outputs)
            #odd_loss = self.loss(odd_outputs, targets)
            ##########################################################################
            ##########################################################################

            ##########################################################################
            ##########################################################################
            # executing regularization...
            if self.args.loss.split("_")[3] != "no":
                regularization_multiplier = float(self.args.loss.split("_")[3][1:])
                if self.args.loss.split("_")[3][0] == "P":
                    regularization = regularization_multiplier * max_probs.std()
                elif self.args.loss.split("_")[3][0] == "E":
                    regularization = regularization_multiplier * entropies.std()
                loss += regularization
            else:
                regularization = torch.tensor(0)
            #print(max_probs.std().item())
            #print(entropies.std().item())
            ##########################################################################
            ##########################################################################

            # accumulate metrics over batches...
            loss_meter.add(loss.item()-augmentation.item()-regularization.item(), inputs.size(0))
            #odd_loss_meter.add(odd_loss.item(), inputs.size(0))
            accuracy_meter.add(outputs.detach(), targets.detach())
            odd_accuracy_meter.add(odd_outputs.detach(), targets.detach())

            intra_logits = intra_logits.tolist()
            inter_logits = inter_logits.tolist()
            if self.args.number_of_model_classes > 100:
                print("WARMING!!! DO NOT BLINDLY TRUST EPOCH LOGITS STATISTICS!!!")
                epoch_logits["intra"] = intra_logits
                epoch_logits["inter"] = inter_logits
            else:
                epoch_logits["intra"] += intra_logits
                epoch_logits["inter"] += inter_logits
            epoch_metrics["max_probs"] += max_probs.tolist()
            epoch_metrics["entropies"] += entropies.tolist()

            # zero grads, compute gradients and do optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_index % self.args.print_freq == 0:
                print('Train Epoch: [{0}][{1:3}/{2}]\t'
                      'Loss {loss:.8f}\t\t'
                      'Acc1 {acc1_meter:.2f}\t'
                      'IALM {intra_logits_mean:.4f}\t'
                      'IALS {intra_logits_std:.8f}\t\t'
                      'IELM {inter_logits_mean:.4f}\t'
                      'IELS {inter_logits_std:.8f}'
                      .format(self.epoch, batch_index, len(self.trainset_loader_for_train),
                              loss=loss_meter.avg,
                              acc1_meter=accuracy_meter.value()[0],
                              intra_logits_mean=statistics.mean(intra_logits),
                              intra_logits_std=statistics.stdev(intra_logits),
                              inter_logits_mean=statistics.mean(inter_logits),
                              inter_logits_std=statistics.stdev(inter_logits),
                              )
                      )

        print('\n#### TRAIN ACC1:\t{0:.4f}\n\n'.format(accuracy_meter.value()[0]))

        ###########################################
        #torch.set_printoptions(profile="full")
        print()
        print(self.model.classifier.weights[:5])
        print()
        print(self.model.classifier.weights[0].mean())
        print(self.model.classifier.weights[0].std())
        print(self.model.classifier.weights[1].mean())
        print(self.model.classifier.weights[1].std())
        print(self.model.classifier.weights[2].mean())
        print(self.model.classifier.weights[2].std())
        print(self.model.classifier.weights[3].mean())
        print(self.model.classifier.weights[3].std())
        print(self.model.classifier.weights[4].mean())
        print(self.model.classifier.weights[4].std())
        print()
        print("================================================================")
        print("SIZE: [DIM=0]\n", self.model.classifier.weights.mean(dim=0).size())
        print("SIZE: [DIM=0]\n", self.model.classifier.weights.std(dim=0).size())
        print("WEIGHTS MEAN [DIM=0]:\n", self.model.classifier.weights.mean(dim=0))
        print("WEIGHTS  STD [DIM=0]:\n", self.model.classifier.weights.std(dim=0))
        print("WEIGHTS MEAN [DIM=0] (mean):\n", self.model.classifier.weights.mean(dim=0).mean())
        print("WEIGHTS  STD [DIM=0] (mean):\n", self.model.classifier.weights.std(dim=0).mean())
        print("WEIGHTS MEAN [DIM=0] (std):\n", self.model.classifier.weights.mean(dim=0).std())
        print("WEIGHTS  STD [DIM=0] (std):\n", self.model.classifier.weights.std(dim=0).std())
        print("SIZE: [DIM=1]\n", self.model.classifier.weights.mean(dim=1).size())
        print("SIZE: [DIM=1]\n", self.model.classifier.weights.std(dim=1).size())
        print("WEIGHTS MEAN [DIM=1]:\n", self.model.classifier.weights.mean(dim=1))
        print("WEIGHTS  STD [DIM=1]:\n", self.model.classifier.weights.std(dim=1))
        print("WEIGHTS MEAN [DIM=1] (mean):\n", self.model.classifier.weights.mean(dim=1).mean())
        print("WEIGHTS  STD [DIM=1] (mean):\n", self.model.classifier.weights.std(dim=1).mean())
        print("WEIGHTS MEAN [DIM=1] (std):\n", self.model.classifier.weights.mean(dim=1).std())
        print("WEIGHTS  STD [DIM=1] (std):\n", self.model.classifier.weights.std(dim=1).std())
        print("================================================================")
        print()
        #torch.set_printoptions(profile="default")
        ###########################################

        return loss_meter.avg, accuracy_meter.value()[0], odd_accuracy_meter.value()[0], epoch_logits, epoch_metrics
        #return loss_meter.avg, odd_loss_meter.avg, accuracy_meter.value()[0], odd_accuracy_meter.value()[0], epoch_logits, epoch_metrics

    def validate_epoch(self):
        print()
        # switch to evaluate mode
        self.model.eval()
        self.criterion.eval()

        # Meters...
        loss_meter = utils.MeanMeter()
        #odd_loss_meter = utils.MeanMeter()
        accuracy_meter = tnt.meter.ClassErrorMeter(topk=[1], accuracy=True)
        odd_accuracy_meter = tnt.meter.ClassErrorMeter(topk=[1], accuracy=True)
        epoch_logits = {"intra": [], "inter": []}
        epoch_metrics = {"max_probs": [], "entropies": []}
        #epoch_entropies_per_classes =  [[] for i in range(self.model.classifier.weights.size(0))]

        with torch.no_grad():

            for batch_index, (inputs, targets) in enumerate(self.valset_loader):
                batch_index += 1

                # moving to GPU...
                inputs = inputs.cuda()
                targets = targets.cuda(non_blocking=True)

                # compute output
                self.model.classifier.metrics_evaluation_mode = True
                features = self.model(inputs)

                # compute loss
                # loss, intra_logits, and inter_logits are already allways using the correct batch size in the bellow line of code...
                loss, outputs, odd_outputs, intra_logits, inter_logits = self.criterion(features, targets)
                self.model.classifier.metrics_evaluation_mode = False

                ##########################################################################
                ##########################################################################
                max_probs = nn.Softmax(dim=1)(odd_outputs).max(dim=1)[0]
                entropies = utils.entropies_from_logits(odd_outputs)
                #odd_loss = self.loss(odd_outputs, targets)
                ##########################################################################
                ##########################################################################

                # accumulate metrics over batches...
                loss_meter.add(loss.item(), inputs.size(0))
                #odd_loss_meter.add(odd_loss.item(), inputs.size(0))
                accuracy_meter.add(outputs.detach(), targets.detach())
                odd_accuracy_meter.add(odd_outputs.detach(), targets.detach())

                intra_logits = intra_logits.tolist()
                inter_logits = inter_logits.tolist()
                if self.args.number_of_model_classes > 100:
                    print("WARMING!!! DO NOT BLINDLY TRUST EPOCH LOGITS STATISTICS!!!")
                    epoch_logits["intra"] = intra_logits
                    epoch_logits["inter"] = inter_logits
                else:
                    epoch_logits["intra"] += intra_logits
                    epoch_logits["inter"] += inter_logits
                epoch_metrics["max_probs"] += max_probs.tolist()
                epoch_metrics["entropies"] += entropies.tolist()

                if batch_index % self.args.print_freq == 0:
                    print('Valid Epoch: [{0}][{1:3}/{2}]\t'
                          'Loss {loss:.8f}\t\t'
                          'Acc1 {acc1_meter:.2f}\t'
                          'IALM {intra_logits_mean:.4f}\t'
                          'IALS {intra_logits_std:.8f}\t\t'
                          'IELM {inter_logits_mean:.4f}\t'
                          'IELS {inter_logits_std:.8f}'
                          .format(self.epoch, batch_index, len(self.valset_loader),
                                  loss=loss_meter.avg,
                                  acc1_meter=accuracy_meter.value()[0],
                                  intra_logits_mean=statistics.mean(intra_logits),
                                  intra_logits_std=statistics.stdev(intra_logits),
                                  inter_logits_mean=statistics.mean(inter_logits),
                                  inter_logits_std=statistics.stdev(inter_logits),
                                  )
                          )

        print('\n#### VALID ACC1:\t{0:.4f}\n\n'.format(accuracy_meter.value()[0]))

        return loss_meter.avg, accuracy_meter.value()[0], odd_accuracy_meter.value()[0], epoch_logits, epoch_metrics
        #return loss_meter.avg, odd_loss_meter.avg, accuracy_meter.value()[0], odd_accuracy_meter.value()[0], epoch_logits, epoch_metrics

    def extract_features_for_all_sets(self, model_file_path):
        print("\n################ EXTRACTING FEATURES ################")

        # Loading best model...
        if os.path.isfile(model_file_path):
            print("\n=> loading checkpoint '{}'".format(model_file_path))
            #checkpoint = torch.load(model_file_path)
            #self.model.load_state_dict(checkpoint['best_model_state_dict'])
            #print("=> loaded checkpoint '{}' (epoch {})".format(model_file_path, checkpoint['best_model_epoch']))
            self.model.load_state_dict(torch.load(model_file_path, map_location="cuda:" + str(self.args.gpu_id)))
            print("=> loaded checkpoint '{}'".format(model_file_path))
        else:
            print("=> no checkpoint found at '{}'".format(model_file_path))
            return

        features_trainset_first_partition_file_path = '{}.pth'.format(os.path.splitext(model_file_path)[0]+'_trainset_first_partition')
        features_trainset_second_partition_file_path = '{}.pth'.format(os.path.splitext(model_file_path)[0]+'_trainset_second_partition')
        features_valset_file_path = '{}.pth'.format(os.path.splitext(model_file_path)[0]+'_valset')

        if len(self.trainset_first_partition_loader_for_infer) != 0:
            self.extract_features_from_loader(
                self.trainset_first_partition_loader_for_infer, features_trainset_first_partition_file_path)
        if len(self.trainset_second_partition_loader_for_infer) != 0:
            self.extract_features_from_loader(
                self.trainset_second_partition_loader_for_infer, features_trainset_second_partition_file_path)
        self.extract_features_from_loader(self.valset_loader, features_valset_file_path)

    def extract_features_from_loader(self, loader, file_path):
        # switch to evaluate mode
        self.model.eval()
        # print('\nExtract features on {}set'.format(loader.dataset.set))
        print('Extract features on {}'.format(loader.dataset))

        with torch.no_grad():
            for batch_id, (input_tensor, target_tensor) in enumerate(tqdm(loader)):
                # moving to GPU...
                input_tensor = input_tensor.cuda()
                # target_tensor = target_tensor.cuda(non_blocking=True)
                # compute batch logits and features...
                batch_logits, batch_features = self.model.logits_features(input_tensor)
                if batch_id == 0:
                    logits = torch.Tensor(len(loader.sampler), self.args.number_of_model_classes)
                    features = torch.Tensor(len(loader.sampler), batch_features.size()[1])
                    targets = torch.Tensor(len(loader.sampler))
                    print("LOGITS:", logits.size())
                    print("FEATURES:", features.size())
                    print("TARGETS:", targets.size())
                current_bsize = input_tensor.size(0)
                from_ = int(batch_id * loader.batch_size)
                to_ = int(from_ + current_bsize)
                logits[from_:to_] = batch_logits.cpu()
                features[from_:to_] = batch_features.cpu()
                targets[from_:to_] = target_tensor

        os.system('mkdir -p {}'.format(os.path.dirname(file_path)))
        print('save ' + file_path)
        torch.save((logits, features, targets), file_path)
        return logits, features, targets

    def odd_infer(self):
        print("\n################ INFERING ################")

        # Loading best model...
        if os.path.isfile(self.args.best_model_file_path):
            print("\n=> loading checkpoint '{}'".format(self.args.best_model_file_path))
            self.model.load_state_dict(torch.load(self.args.best_model_file_path, map_location="cuda:" + str(self.args.gpu_id)))
            print("=> loaded checkpoint '{}'".format(self.args.best_model_file_path))
        else:
            print("=> no checkpoint found at '{}'".format(self.args.best_model_file_path))
            return

        # preparing and normalizing data
        if self.args.dataset == 'cifar10':
            in_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))])
        elif self.args.dataset == 'cifar100':
            in_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.507, 0.486, 0.440), (0.267, 0.256, 0.276))])
        elif self.args.dataset == 'svhn':
            in_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.437, 0.443, 0.472), (0.198, 0.201, 0.197))])

        """
        # defining out-distribution...
        if self.args.dataset == 'svhn':
            out_dist_list = ['cifar10', 'imagenet_resize', 'lsun_resize']
        else:
            out_dist_list = ['svhn', 'imagenet_resize', 'lsun_resize']
        """
        if self.args.dataset == 'cifar10':
            #out_dist_list = ['svhn', 'imagenet_resize', 'lsun_resize']
            out_dist_list = ['svhn', 'imagenet_resize', 'lsun_resize', 'fooling_images', 'gaussian_noise','uniform_noise']
        elif self.args.dataset == 'cifar100':
            #out_dist_list = ['svhn', 'imagenet_resize', 'lsun_resize']
            out_dist_list = ['svhn', 'imagenet_resize', 'lsun_resize', 'fooling_images', 'gaussian_noise','uniform_noise']
        elif self.args.dataset == 'svhn':
            #out_dist_list = ['cifar100', 'cifar10', 'imagenet_resize', 'lsun_resize']
            out_dist_list = ['cifar100', 'cifar10', 'imagenet_resize', 'lsun_resize', 'fooling_images', 'gaussian_noise','uniform_noise']
        #elif args.dataset == 'imagenet32':
            ##out_dist_list = ['svhn', 'cifar10', 'lsun_resize']
            #out_dist_list = ['svhn', 'cifar10', 'lsun_resize', 'fooling_images', 'gaussian_noise','uniform_noise']

        # Storing logits and metrics in out-distribution...
        for out_dist in out_dist_list:
            print('Out-distribution: ' + out_dist)
            self.valset_loader = data_loader.getNonTargetDataSet(out_dist, self.args.batch_size, in_transform, "data")
            #_, _, valid_epoch_logits, valid_epoch_metrics, valid_epoch_entropies_per_classes = self.validate_epoch()
            _, _, _, valid_epoch_logits, valid_epoch_metrics = self.validate_epoch()
            np.save(os.path.join(
                self.args.experiment_path,
                "best_model"+str(self.args.execution)+"_valid_epoch_logits_"+out_dist+".npy"),
                valid_epoch_logits)
            np.save(os.path.join(
                self.args.experiment_path,
                "best_model"+str(self.args.execution)+"_valid_epoch_metrics_"+out_dist+".npy"),
                valid_epoch_metrics)
            #################################################
            #filename = os.path.join(
            #    self.args.experiment_path,
            #    "best_model"+str(self.args.execution)+"_valid_epoch_entropies_per_classes_"+out_dist+".pkl")
            #with open(filename, 'wb') as file:  # Overwrites any existing file.
            #    pickle.dump(valid_epoch_entropies_per_classes, file, pickle.HIGHEST_PROTOCOL)
            #with open(filename, "rb") as file:
            #    testando = pickle.load(file)
            #    print(testando)
            #################################################

    def adv_infer(self):
        print("\n################ INFERING ################")

        args_outf = os.path.join('./output/adv/', self.args.loss, self.args.model_name + '_' + self.args.dataset_full + '/')  # + '/'

        # Loading best model...
        if os.path.isfile(self.args.best_model_file_path):
            print("\n=> loading checkpoint '{}'".format(self.args.best_model_file_path))
            self.model.load_state_dict(torch.load(self.args.best_model_file_path, map_location="cuda:" + str(self.args.gpu_id)))
            print("=> loaded checkpoint '{}'".format(self.args.best_model_file_path))
        else:
            print("=> no checkpoint found at '{}'".format(self.args.best_model_file_path))
            return

        #attacks = ['FGSM', 'BIM', 'DeepFool', 'CWL2']
        attacks = ['FGSM', 'BIM', 'CWL2']
        #attacks = ['FGSM', 'BIM']

        for attack in attacks:
            print('Attack: ' + attack)

            test_clean_data = torch.load(args_outf + 'clean_data_%s_%s_%s_100.pth' % (self.args.model_name, self.args.dataset_full, attack))
            test_adv_data = torch.load(args_outf + 'adv_data_%s_%s_%s_100.pth' % (self.args.model_name, self.args.dataset_full, attack))
            test_noisy_data = torch.load(args_outf + 'noisy_data_%s_%s_%s_100.pth' % (self.args.model_name, self.args.dataset_full, attack))
            test_label = torch.load(args_outf + 'label_%s_%s_%s_100.pth' % (self.args.model_name, self.args.dataset_full, attack))

            dataset = torch.utils.data.TensorDataset(test_clean_data, test_label)
            self.valset_loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size)
            #_, _, valid_epoch_logits, valid_epoch_metrics, _ = self.validate_epoch()
            _, _, _, valid_epoch_logits, valid_epoch_metrics = self.validate_epoch()
            np.save(os.path.join(
                self.args.experiment_path, "best_model"+str(self.args.execution)+"_valid_epoch_logits_"+attack+"_clean_100.npy"), valid_epoch_logits)
            np.save(os.path.join(
                self.args.experiment_path, "best_model"+str(self.args.execution)+"_valid_epoch_metrics_"+attack+"_clean_100.npy"), valid_epoch_metrics)

            dataset = torch.utils.data.TensorDataset(test_noisy_data, test_label)
            self.valset_loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size)
            #_, _, valid_epoch_logits, valid_epoch_metrics, _ = self.validate_epoch()
            _, _, _, valid_epoch_logits, valid_epoch_metrics = self.validate_epoch()
            np.save(os.path.join(
                self.args.experiment_path, "best_model"+str(self.args.execution)+"_valid_epoch_logits_"+attack+"_noise_100.npy"), valid_epoch_logits)
            np.save(os.path.join(
                self.args.experiment_path, "best_model"+str(self.args.execution)+"_valid_epoch_metrics_"+attack+"_noise_100.npy"), valid_epoch_metrics)

            dataset = torch.utils.data.TensorDataset(test_adv_data, test_label)
            self.valset_loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size)
            #_, _, valid_epoch_logits, valid_epoch_metrics, _ = self.validate_epoch()
            _, _, _, valid_epoch_logits, valid_epoch_metrics = self.validate_epoch()
            np.save(os.path.join(
                self.args.experiment_path, "best_model"+str(self.args.execution)+"_valid_epoch_logits_"+attack+"_adv_100.npy"), valid_epoch_logits)
            np.save(os.path.join(
                self.args.experiment_path, "best_model"+str(self.args.execution)+"_valid_epoch_metrics_"+attack+"_adv_100.npy"), valid_epoch_metrics)


"""
def update_misc_stat(misc_stat, batch_distances, num_classes, total_targets_per_class):
    for i in range(num_classes):
        if total_targets_per_class[i].item() != 0:
            if i not in misc_stat["intraclass"]:
                misc_stat["intraclass"][i] = batch_distances["intraclass"][i]
            else:
                misc_stat["intraclass"][i] += batch_distances["intraclass"][i]
    if "logits" not in misc_stat["interclass"]:
        misc_stat["interclass"]["distances"] = batch_distances["interclass"]
    else:
        misc_stat["interclass"]["distances"] += batch_distances["interclass"]
    return misc_stat


def calculate_misc_stat(misc_stat, num_classes):
    misc_stat["intraclass"]["mean"] = {}
    misc_stat["intraclass"]["std"] = {}
    for i in range(num_classes):
        if i not in misc_stat["intraclass"]:
            misc_stat["intraclass"]["mean"][i] = math.nan
            misc_stat["intraclass"]["std"][i] = math.nan
        else:
            misc_stat["intraclass"]["mean"][i] = statistics.mean(misc_stat["intraclass"][i])
            misc_stat["intraclass"]["std"][i] = statistics.pstdev(misc_stat["intraclass"][i])
    #print("\nINTRACLASS MEAN:", misc_stat["intraclass"]["mean"])
    #print("INTRACLASS STD:\n", misc_stat["intraclass"]["std"])
    misc_stat["intraclass"]["mean"]["mean"] = statistics.mean(list(misc_stat["intraclass"]["mean"].values()))
    misc_stat["intraclass"]["mean"]["std"] = statistics.pstdev(list(misc_stat["intraclass"]["mean"].values()))
    misc_stat["intraclass"]["std"]["mean"] = statistics.mean(list(misc_stat["intraclass"]["std"].values()))
    misc_stat["interclass"]["mean"] = statistics.mean(misc_stat["interclass"]["distances"])
    misc_stat["interclass"]["std"] = statistics.pstdev(misc_stat["interclass"]["distances"])
    return misc_stat
"""
