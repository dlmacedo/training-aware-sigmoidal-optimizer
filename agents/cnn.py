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
         self.valset_loader, self.normalize) = image_loaders.get_loaders()
        self.batch_normalize = loaders.BatchNormalize(self.normalize.mean, self.normalize.std, inplace=True, device=torch.cuda.current_device())
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
                100, int(self.args.number_of_model_classes))
        elif self.args.model_name == "resnet32":
            self.model = models.ResNet32(
                num_c=self.args.number_of_model_classes)
        elif self.args.model_name == "resnet34":
            self.model = models.ResNet34(
                num_c=self.args.number_of_model_classes)
        elif self.args.model_name == "resnet110":
            self.model = models.ResNet110(
                num_c=self.args.number_of_model_classes)
        elif self.args.model_name == "resnet56":
            self.model = models.ResNet56(
                num_c=self.args.number_of_model_classes)
        elif self.args.model_name == "resnet18_":
            self.model = models.resnet18_(
                num_classes=self.args.number_of_model_classes)
        elif self.args.model_name == "resnet34_":
            self.model = models.resnet34_(
                num_classes=self.args.number_of_model_classes)
        elif self.args.model_name == "resnet101_":
            self.model = models.resnet101_(
                num_classes=self.args.number_of_model_classes)
        elif self.args.model_name == "wideresnet3410":
            self.model = models.Wide_ResNet(
                depth=34, widen_factor=10, num_classes=self.args.number_of_model_classes)
        #elif self.args.model_name == "vgg":
        #    self.model = models.VGG19(num_classes=self.args.number_of_model_classes)
        elif self.args.model_name == "efficientnetb0":
            self.model = models.EfficientNet(
                num_classes=self.args.number_of_model_classes, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2)
        self.model.cuda()
        torch.manual_seed(self.args.base_seed)
        torch.cuda.manual_seed(self.args.base_seed)

        # print and save model arch...
        if self.args.exp_type == "cnn_train":
            print("\nMODEL:", self.model)
            with open(os.path.join(self.args.experiment_path, 'model.arch'), 'w') as file:
                print(self.model, file=file)

        print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        utils.print_num_params(self.model)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        # create loss
        #self.criterion = losses.GenericLossSecondPart(self.model.classifier).cuda()
        self.criterion = nn.CrossEntropyLoss().cuda()

        # create train
        if self.args.optim.startswith("taso"):
            print("\n$$$$$$$$$$$$$$$")
            print("OPTIMIZER: TASO")
            print("$$$$$$$$$$$$$$$\n")
            initial_learning_rate = float(self.args.optim.split("_")[1][2:])
            #final_learning_rate = initial_learning_rate / 1000
            self.args.epochs = int(self.args.optim.split("_")[2][1:])
            weight_decay = float(self.args.optim.split("_")[3][1:])
            momentum = float(self.args.optim.split("_")[4][1:])
            nesterov = True if (self.args.optim.split("_")[5][1:] == "t") else False
            alpha = float(self.args.optim.split("_")[6][1:])
            beta = float(self.args.optim.split("_")[7][1:])
            parameters = self.model.parameters()
            self.optimizer = torch.optim.SGD(
                parameters, lr=initial_learning_rate, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
            #our_lambda = lambda epoch: (1 - final_learning_rate)/(1 + math.exp(alpha*(epoch/parameters['epochs']-beta))) + final_learning_rate
            #self.our_lambda = lambda epoch: 1/(1 + math.exp(alpha*((epoch+1)/self.args.epochs-beta))) + final_learning_rate
            #self.our_lambda = lambda epoch: 1/(1 + alpha*(epoch/self.args.epochs))
            taso_function = lambda epoch: 1/(1 + math.exp(alpha*(((epoch+1)/self.args.epochs)-beta))) + 0.001
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=taso_function, verbose=True)
            print("INITIAL LEARNING RATE: ", initial_learning_rate)
            #print("FINAL LEARNING RATE: ", final_learning_rate)
            print("TOTAL EPOCHS: ", self.args.epochs)
            print("WEIGHT DECAY: ", weight_decay)
            print("MOMENTUM: ", momentum)
            print("NESTEROV: ", nesterov)
            print("ALPHA: ", alpha)
            print("BETA: ", beta)
        elif self.args.optim.startswith("adam"):
            print("\n$$$$$$$$$$$$$$$")
            print("OPTIMIZER: ADAM")
            print("$$$$$$$$$$$$$$$\n")
            parameters = self.model.parameters()
            self.optimizer = torch.optim.SGD(
                parameters,
                lr=self.args.original_learning_rate,
                momentum=self.args.momentum,
                nesterov=True,
                weight_decay=args.weight_decay)
        elif self.args.optim.startswith("rmsprop"):
            print("\n$$$$$$$$$$$$$$$$$$")
            print("OPTIMIZER: RMSPROP")
            print("$$$$$$$$$$$$$$$$$$\n")
            parameters = self.model.parameters()
            self.optimizer = torch.optim.SGD(
                parameters,
                lr=self.args.original_learning_rate,
                momentum=self.args.momentum,
                nesterov=True,
                weight_decay=args.weight_decay)
        elif self.args.optim.startswith("adagrad"):
            print("\n$$$$$$$$$$$$$$$$$$")
            print("OPTIMIZER: ADAGRAD")
            print("$$$$$$$$$$$$$$$$$$\n")
            parameters = self.model.parameters()
            self.optimizer = torch.optim.SGD(
                parameters,
                lr=self.args.original_learning_rate,
                momentum=self.args.momentum,
                nesterov=True,
                weight_decay=args.weight_decay)

        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.learning_rate_decay_epochs, gamma=args.learning_rate_decay_rate)

        print("\nTRAIN:")
        print(self.criterion)
        print(self.optimizer)
        print(self.scheduler)

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

    
    def train_validate(self):
        # template for others procedures of this class...
        # building results and raw results files...
        if self.args.execution == 1:
            with open(self.args.executions_best_results_file_path, "w") as best_results:
                best_results.write("DATA,MODEL,OPTIM,EXECUTION,EPOCH,TRAIN LOSS,TRAIN ACC1,VALID LOSS,VALID ACC1\n")
            with open(self.args.executions_raw_results_file_path, "w") as raw_results:
                raw_results.write("DATA,MODEL,OPTIM,EXECUTION,EPOCH,SET,METRIC,VALUE\n")

        print("\n################ TRAINING ################")

        #best_model_results = {"TRAIN LOSS": float("inf")}
        #best_model_results = {"TRAIN ODD_LOSS": float("inf")}
        best_model_results = {"VALID ACC1": 0}
        #best_model_results = {"VALID ODD_ACC": 0}

        for self.epoch in range(1, self.args.epochs + 1):
            #self.epoch = epoch
            print("\n######## EPOCH:", self.epoch, "OF", self.args.epochs, "########")

            """
            ##########################################################################################
            ##########################################################################################
            if self.epoch == 1:
                if self.args.model_name in ['resnet110','resnet1202'] and self.args.dataset_full.startswith("cifar"):
                    print("Starting warm up training!!!\n" * 10)
                    # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
                    # then switch back. In this setup it will correspond for first epoch.
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.args.original_learning_rate*0.1
            ##########################################################################################
            ##########################################################################################
            """

            # Print current learning rate...
            for param_group in self.optimizer.param_groups:
                print("\nLEARNING RATE:\t\t", param_group["lr"])

            #train_loss, train_acc1, train_odd_acc, train_epoch_logits, train_epoch_metrics = self.train_epoch()
            train_loss, train_acc1 = self.train_epoch()
            #train_loss, train_odd_loss, train_acc1, train_odd_acc, train_epoch_logits, train_epoch_metrics = self.train_epoch()

            """
            ##########################################################################################
            ##########################################################################################
            if self.epoch == 1:
                if self.args.model_name in ['resnet110','resnet1202'] and self.args.dataset_full.startswith("cifar"):
                    print("Finishing warm up training!!!\n" * 10)
                    # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
                    # then switch back. In this setup it will correspond for first epoch.
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.args.original_learning_rate
            ##########################################################################################
            ##########################################################################################
            """

            
            #valid_loss, valid_acc1, valid_odd_acc, valid_epoch_logits, valid_epoch_metrics = self.validate_epoch()
            valid_loss, valid_acc1 = self.validate_epoch()
            #valid_loss, valid_odd_loss, valid_acc1, valid_odd_acc, valid_epoch_logits, valid_epoch_metrics = self.validate_epoch()


            # Adjusting learning rate (if not using reduce on plateau)...
            if self.args.optim.startswith("taso"):
                #print(epoch)
                #print(self.our_lambda(epoch))
                self.scheduler.step()
                #print(self.criterion)
                #print(self.optimizer)
                #print(self.scheduler)


            """
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
            """

            with open(self.args.executions_raw_results_file_path, "a") as raw_results:
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.optim, self.args.execution, self.epoch,
                    "TRAIN", "LOSS", train_loss))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.optim, self.args.execution, self.epoch,
                    "TRAIN", "ACC1", train_acc1))
                #########################################################               
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.optim, self.args.execution, self.epoch,
                    "VALID", "LOSS", valid_loss))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.optim, self.args.execution, self.epoch,
                    "VALID", "ACC1", valid_acc1))

            #print()
            #print("TRAIN ==>>\tIADM: {0:.8f}\tIADS: {1:.8f}\tIEDM: {2:.8f}\tIEDS: {3:.8f}".format(
            #    train_intra_logits_mean, train_intra_logits_std, train_inter_logits_mean, train_inter_logits_std))
            #print("VALID ==>>\tIADM: {0:.8f}\tIADS: {1:.8f}\tIEDM: {2:.8f}\tIEDS: {3:.8f}".format(
            #    valid_intra_logits_mean, valid_intra_logits_std, valid_inter_logits_mean, valid_inter_logits_std))
            #print()

            #############################################
            print("\nDATA:", self.args.dataset_full)
            print("MODEL:", self.args.model_name)
            print("OPTIM:", self.args.optim, "\n")
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
                    "OPTIM": self.args.optim,
                    "EXECUTION": self.args.execution,
                    "EPOCH": self.epoch,
                    "TRAIN LOSS": train_loss,
                    "TRAIN ACC1": train_acc1,
                    "VALID LOSS": valid_loss,
                    "VALID ACC1": valid_acc1,
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

                """
                np.save(os.path.join(
                    self.args.experiment_path, "best_model"+str(self.args.execution)+"_train_epoch_logits.npy"), train_epoch_logits)
                np.save(os.path.join(
                    self.args.experiment_path, "best_model"+str(self.args.execution)+"_train_epoch_metrics.npy"), train_epoch_metrics)
                np.save(os.path.join(
                    self.args.experiment_path, "best_model"+str(self.args.execution)+"_valid_epoch_logits.npy"), valid_epoch_logits)
                np.save(os.path.join(
                    self.args.experiment_path, "best_model"+str(self.args.execution)+"_valid_epoch_metrics.npy"), valid_epoch_metrics)
                """
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


        with open(self.args.executions_best_results_file_path, "a") as best_results:
            #best_results.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            best_results.write("{},{},{},{},{},{},{},{},{}\n".format(
                best_model_results["DATA"],
                best_model_results["MODEL"],
                best_model_results["OPTIM"],
                best_model_results["EXECUTION"],
                best_model_results["EPOCH"],
                best_model_results["TRAIN LOSS"],
                best_model_results["TRAIN ACC1"],
                best_model_results["VALID LOSS"],
                best_model_results["VALID ACC1"],
                )
            )

        # extracting features from best model...
        ####self.extract_features_for_all_sets(self.args.best_model_file_path)
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

            # compute output
            outputs = self.model(inputs)

            # compute loss
            loss = self.criterion(outputs, targets)

            #max_probs = nn.Softmax(dim=1)(odd_outputs).max(dim=1)[0]
            #entropies = utils.entropies_from_logits(odd_outputs)
            #max_probs = odd_probabilities.max(dim=1)[0]
            #entropies = utils.entropies_from_probabilities(odd_probabilities)


            # accumulate metrics over batches...
            loss_meter.add(loss.item(), inputs.size(0))
            accuracy_meter.add(outputs.detach(), targets.detach())
            #odd_accuracy_meter.add(odd_outputs.detach(), targets.detach())
            #accuracy_meter.add(cls_probabilities.detach(), targets.detach())
            #odd_accuracy_meter.add(odd_probabilities.detach(), targets.detach())

            """
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
            """

            # zero grads, compute gradients and do optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_index % self.args.print_freq == 0:
                print('Train Epoch: [{0}][{1:3}/{2}]\t'
                      'Loss {loss:.8f}\t\t'
                      'Acc1 {acc1_meter:.2f}\t'
                      #'IADM {intra_logits_mean:.4f}\t'
                      #'IADS {intra_logits_std:.8f}\t\t'
                      #'IEDM {inter_logits_mean:.4f}\t'
                      #'IEDS {inter_logits_std:.8f}'
                      .format(self.epoch, batch_index, len(self.trainset_loader_for_train),
                              loss=loss_meter.avg,
                              acc1_meter=accuracy_meter.value()[0],
                              #intra_logits_mean=statistics.mean(intra_logits),
                              #intra_logits_std=statistics.stdev(intra_logits),
                              #inter_logits_mean=statistics.mean(inter_logits),
                              #inter_logits_std=statistics.stdev(inter_logits),
                              )
                      )

        print('\n#### TRAIN ACC1:\t{0:.4f}\n\n'.format(accuracy_meter.value()[0]))

        """
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
        """

        return loss_meter.avg, accuracy_meter.value()[0]#, odd_accuracy_meter.value()[0], epoch_logits, epoch_metrics
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
                #self.model.classifier.metrics_evaluation_mode = True
                outputs = self.model(inputs)

                # compute loss
                # loss, intra_logits, and inter_logits are already allways using the correct batch size in the bellow line of code...
                #loss, outputs, odd_outputs, intra_logits, inter_logits = self.criterion(features, targets)
                loss = self.criterion(outputs, targets)
                #self.model.classifier.metrics_evaluation_mode = False

                #max_probs = nn.Softmax(dim=1)(odd_outputs).max(dim=1)[0]
                #entropies = utils.entropies_from_logits(odd_outputs)
                #max_probs = odd_probabilities.max(dim=1)[0]
                #entropies = utils.entropies_from_probabilities(odd_probabilities)

                # accumulate metrics over batches...
                loss_meter.add(loss.item(), inputs.size(0))
                accuracy_meter.add(outputs.detach(), targets.detach())
                #odd_accuracy_meter.add(odd_outputs.detach(), targets.detach())
                #accuracy_meter.add(cls_probabilities.detach(), targets.detach())
                #odd_accuracy_meter.add(odd_probabilities.detach(), targets.detach())

                """
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
                """

                if batch_index % self.args.print_freq == 0:
                    print('Valid Epoch: [{0}][{1:3}/{2}]\t'
                          'Loss {loss:.8f}\t\t'
                          'Acc1 {acc1_meter:.2f}\t'
                          #'IADM {intra_logits_mean:.4f}\t'
                          #'IADS {intra_logits_std:.8f}\t\t'
                          #'IEDM {inter_logits_mean:.4f}\t'
                          #'IEDS {inter_logits_std:.8f}'
                          .format(self.epoch, batch_index, len(self.valset_loader),
                                  loss=loss_meter.avg,
                                  acc1_meter=accuracy_meter.value()[0],
                                  #intra_logits_mean=statistics.mean(intra_logits),
                                  #intra_logits_std=statistics.stdev(intra_logits),
                                  #inter_logits_mean=statistics.mean(inter_logits),
                                  #inter_logits_std=statistics.stdev(inter_logits),
                                  )
                          )

        print('\n#### VALID ACC1:\t{0:.4f}\n\n'.format(accuracy_meter.value()[0]))

        return loss_meter.avg, accuracy_meter.value()[0]#, odd_accuracy_meter.value()[0], epoch_logits, epoch_metrics
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

            ########################################################################################################
            ########################################################################################################
            """
            from collections import OrderedDict
            model_state = torch.load("/tmp/efficientnet_weights/efficientnet-b0-08094119.pth")
            # A basic remapping is required
            mapping = {k: v for k, v in zip(model_state.keys(), model.state_dict().keys())}
            mapped_model_state = OrderedDict([(mapping[k], v) for k, v in model_state.items()])
            model.load_state_dict(mapped_model_state, strict=False)
            """
            ########################################################################################################
            ########################################################################################################

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
