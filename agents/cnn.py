import os
import sys
import torch
import torch.nn as nn
import models
import loaders
#import losses
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
#import data_loader
import pickle
from torchinfo import summary


sns.set(style="darkgrid")


class CNNAgent:

    def __init__(self, args):

        self.args = args
        self.epoch = None
        self.cluster_predictions_transformation = []

        # create dataset
        if self.args.data_type == 'image':
            image_loaders = loaders.ImageLoader(args)
            (self.trainset_first_partition_loader_for_train,
            self.trainset_second_partition_loader_for_train,
            self.trainset_first_partition_loader_for_infer,
            self.trainset_second_partition_loader_for_infer,
            self.valset_loader, self.normalize) = image_loaders.get_loaders()
            self.batch_normalize = loaders.BatchNormalize(
                self.normalize.mean, self.normalize.std, inplace=True, device=torch.cuda.current_device())
        elif self.args.data_type == 'text':
            text_loaders = loaders.TextLoader(args)
            (self.trainset_first_partition_loader_for_train,
            self.trainset_second_partition_loader_for_train,
            self.trainset_first_partition_loader_for_infer,
            self.trainset_second_partition_loader_for_infer,
            self.valset_loader, self.normalize) = text_loaders.get_loaders()

            
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
        elif self.args.model_name == "resnet50":
            self.model = models.ResNet50(
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
        #############################################################################################################################
        #############################################################################################################################
        #elif self.args.model_name == "textrnn":
        #    self.model = models.TextRNN(
        #        #self.args.text_config,len(self.args.text_dataset.vocab), self.args.text_dataset.word_embeddings)
        #        len(self.args.text_dataset.vocab), self.args.text_dataset.word_embeddings, self.args.number_of_model_classes)
        ##############################################################
        elif self.args.model_name == "textcnn":
            self.model = models.TextCNN(
                len(self.args.text_dataset.vocab), self.args.text_dataset.word_embeddings, self.args.number_of_model_classes)
        elif self.args.model_name == "rcnn":
            self.model = models.RCNN(
                #self.args.text_config,len(self.args.text_dataset.vocab), self.args.text_dataset.word_embeddings)
                len(self.args.text_dataset.vocab), self.args.text_dataset.word_embeddings, self.args.number_of_model_classes)
        elif self.args.model_name == "s2satt":
            self.model = models.Seq2SeqAttention(
                #self.args.text_config,len(self.args.text_dataset.vocab), self.args.text_dataset.word_embeddings)
                len(self.args.text_dataset.vocab), self.args.text_dataset.word_embeddings, self.args.number_of_model_classes)
        #elif self.args.model_name == "fasttext":
        #    embed_dim = 64
        #    self.model = models.FastText(
        #        #self.args.text_config,len(self.args.text_dataset.vocab), self.args.text_dataset.word_embeddings)
        #        #len(train_dataset.get_vocab()), embed_dim, len(train_dataset.get_labels()))
        #        len(text_loaders.train_set.get_vocab()), embed_dim, len(text_loaders.train_set.get_labels()))
        self.model.cuda()
        torch.manual_seed(self.args.base_seed)
        torch.cuda.manual_seed(self.args.base_seed)

        # print and save model arch...
        if self.args.exp_type == "cnn_train":
            print("\nMODEL:", self.model)
            with open(os.path.join(self.args.experiment_path, 'model.arch'), 'w') as file:
                print(self.model, file=file)
        

        print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        #collect_batch = next(iter(self.trainset_loader_for_train))
        #if self.args.data_type == "image":
        #    input_data = collect_batch[0][0]#.cuda()
        #elif self.args.data_type == "text":
        #    input_data = collect_batch[0].text#.cuda()
        #summary(self.model, input_data=input_data[0])
        utils.print_num_params(self.model)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        # create loss
        #self.criterion = losses.GenericLossSecondPart(self.model.classifier).cuda()
        self.criterion = nn.CrossEntropyLoss().cuda()

        # create train
        if self.args.optim.startswith("sgd"):
            print("\n$$$$$$$$$$$$$$$")
            print("OPTIMIZER: SGD")
            print("$$$$$$$$$$$$$$$\n")
            initial_learning_rate = float(self.args.optim.split("_")[1][2:])
            self.args.epochs = int(self.args.optim.split("_")[2][1:])
            weight_decay = float(self.args.optim.split("_")[3][1:])
            momentum = float(self.args.optim.split("_")[4][1:])
            nesterov = True if (self.args.optim.split("_")[5][1:] == "t") else False
            #alpha = float(self.args.optim.split("_")[6][1:])
            #beta = float(self.args.optim.split("_")[7][1:])
            parameters = self.model.parameters()
            self.optimizer = torch.optim.SGD(
                parameters,
                lr=initial_learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov)
            #taso_function = lambda epoch: 1/(1 + math.exp(alpha*(((epoch+1)/self.args.epochs)-beta))) + 0.001
            #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=taso_function, verbose=True)
            print("INITIAL LEARNING RATE: ", initial_learning_rate)
            print("TOTAL EPOCHS: ", self.args.epochs)
            print("WEIGHT DECAY: ", weight_decay)
            print("MOMENTUM: ", momentum)
            print("NESTEROV: ", nesterov)
            #print("ALPHA: ", alpha)
            #print("BETA: ", beta)
        elif self.args.optim.startswith("taso"):
            print("\n$$$$$$$$$$$$$$$")
            print("OPTIMIZER: TASO")
            print("$$$$$$$$$$$$$$$\n")
            initial_learning_rate = float(self.args.optim.split("_")[1][2:])
            self.args.epochs = int(self.args.optim.split("_")[2][1:])
            weight_decay = float(self.args.optim.split("_")[3][1:])
            momentum = float(self.args.optim.split("_")[4][1:])
            nesterov = True if (self.args.optim.split("_")[5][1:] == "t") else False
            alpha = float(self.args.optim.split("_")[6][1:])
            beta = float(self.args.optim.split("_")[7][1:])
            parameters = self.model.parameters()
            self.optimizer = torch.optim.SGD(
                parameters,
                lr=initial_learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov)
            taso_function = lambda epoch: 1/(1 + math.exp(alpha*(((epoch+1)/self.args.epochs)-beta))) + 0.001
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=taso_function, verbose=True)
            print("INITIAL LEARNING RATE: ", initial_learning_rate)
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
            initial_learning_rate = float(self.args.optim.split("_")[1][2:])
            self.args.epochs = int(self.args.optim.split("_")[2][1:])
            weight_decay = float(self.args.optim.split("_")[3][1:])
            amsgrad = True if (self.args.optim.split("_")[4][1:] == "t") else False
            beta_first = float(self.args.optim.split("_")[5][2:])
            beta_second = float(self.args.optim.split("_")[6][2:])
            parameters = self.model.parameters()
            self.optimizer = torch.optim.Adam(
                parameters,
                lr=initial_learning_rate,
                amsgrad=amsgrad,
                weight_decay=weight_decay,
                betas=(beta_first, beta_second))
            print("INITIAL LEARNING RATE: ", initial_learning_rate)
            print("TOTAL EPOCHS: ", self.args.epochs)
            print("WEIGHT DECAY: ", weight_decay)
            print("AMSGRAD: ", amsgrad)
            print("BETA_FIRST: ", beta_first)
            print("BEST_SECOND: ", beta_second)
        elif self.args.optim.startswith("rmsprop"):
            print("\n$$$$$$$$$$$$$$$$$$")
            print("OPTIMIZER: RMSPROP")
            print("$$$$$$$$$$$$$$$$$$\n")
            initial_learning_rate = float(self.args.optim.split("_")[1][2:])
            self.args.epochs = int(self.args.optim.split("_")[2][1:])
            weight_decay = float(self.args.optim.split("_")[3][1:])
            momentum = float(self.args.optim.split("_")[4][1:])
            centered = True if (self.args.optim.split("_")[5][1:] == "t") else False
            alpha = float(self.args.optim.split("_")[6][1:])
            parameters = self.model.parameters()
            self.optimizer = torch.optim.RMSprop(
                parameters,
                lr=initial_learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
                centered=centered,
                alpha=alpha)
            print("INITIAL LEARNING RATE: ", initial_learning_rate)
            print("TOTAL EPOCHS: ", self.args.epochs)
            print("WEIGHT DECAY: ", weight_decay)
            print("MOMENTUM: ", momentum)
            print("ALPHA: ", alpha)
        elif self.args.optim.startswith("adagrad"):
            print("\n$$$$$$$$$$$$$$$$$$")
            print("OPTIMIZER: ADAGRAD")
            print("$$$$$$$$$$$$$$$$$$\n")
            initial_learning_rate = float(self.args.optim.split("_")[1][2:])
            self.args.epochs = int(self.args.optim.split("_")[2][1:])
            weight_decay = float(self.args.optim.split("_")[3][1:])
            lr_decay = float(self.args.optim.split("_")[4][1:])
            initial_accumulator_value = float(self.args.optim.split("_")[5][1:])
            parameters = self.model.parameters()
            self.optimizer = torch.optim.Adagrad(
                parameters,
                lr=initial_learning_rate,
                weight_decay=weight_decay,
                lr_decay=lr_decay,
                initial_accumulator_value=initial_accumulator_value)
            print("INITIAL LEARNING RATE: ", initial_learning_rate)
            print("TOTAL EPOCHS: ", self.args.epochs)
            print("WEIGHT DECAY: ", weight_decay)
            print("LR DECAY: ", lr_decay)
            print("INITIAL ACCUMULATOR VALUE: ", initial_accumulator_value)

        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        # self.optimizer, milestones=self.args.learning_rate_decay_epochs, gamma=args.learning_rate_decay_rate)

        print("\nTRAIN:")
        #print(self.criterion)
        print(self.optimizer)
        #print(self.scheduler)

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
        best_model_results = {"VALID ACC1": 0}

        for self.epoch in range(1, self.args.epochs + 1):
            #self.epoch = epoch
            print("\n######## EPOCH:", self.epoch, "OF", self.args.epochs, "########")

            # Print current learning rate...
            for param_group in self.optimizer.param_groups:
                print("\nLEARNING RATE:\t\t", param_group["lr"])
                temp_learning_rate = param_group["lr"]

            train_loss, train_acc1 = self.train_epoch()
            valid_loss, valid_acc1 = self.validate_epoch()

            # Adjusting learning rate (if not using reduce on plateau)...
            if self.args.optim.startswith("taso"):
                self.scheduler.step()


            with open(self.args.executions_raw_results_file_path, "a") as raw_results:
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.optim, self.args.execution, self.epoch,
                    "TRAIN", "LOSS", train_loss))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.optim, self.args.execution, self.epoch,
                    "TRAIN", "ACC1", train_acc1))
                ###################################################################################################
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.optim, self.args.execution, self.epoch,
                    "HYPER", "LR", temp_learning_rate))
                ###################################################################################################
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.optim, self.args.execution, self.epoch,
                    "VALID", "LOSS", valid_loss))
                raw_results.write("{},{},{},{},{},{},{},{}\n".format(
                    self.args.dataset_full, self.args.model_name, self.args.optim, self.args.execution, self.epoch,
                    "VALID", "ACC1", valid_acc1))

            #############################################
            print("\nDATA:", self.args.dataset_full)
            print("MODEL:", self.args.model_name)
            print("OPTIM:", self.args.optim, "\n")
            #############################################

            # if is best...
            #if train_loss < best_model_results["TRAIN LOSS"]:
            if valid_acc1 > best_model_results["VALID ACC1"]:
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
                print("!+NEW BEST MODEL VALID ACC1:\t\t{0:.4f} IN EPOCH {1}! SAVING {2}\n".format(
                    valid_acc1, self.epoch, self.args.best_model_file_path))

                torch.save(self.model.state_dict(), self.args.best_model_file_path)

            print('!$$$$ BEST MODEL TRAIN ACC1:\t\t{0:.4f}'.format(best_model_results["TRAIN ACC1"]))
            print('!$$$$ BEST MODEL VALID ACC1:\t\t{0:.4f}'.format(best_model_results["VALID ACC1"]))


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


    def train_epoch(self):
        print()
        # switch to train mode
        self.model.train()
        self.criterion.train()

        # Meters...
        loss_meter = utils.MeanMeter()
        #odd_loss_meter = utils.MeanMeter()
        accuracy_meter = tnt.meter.ClassErrorMeter(topk=[1], accuracy=True)
        #odd_accuracy_meter = tnt.meter.ClassErrorMeter(topk=[1], accuracy=True)
        #epoch_logits = {"intra": [], "inter": []}
        #epoch_metrics = {"max_probs": [], "entropies": []}
        ##epoch_entropies_per_classes =  [[] for i in range(self.model.classifier.weights.size(0))]

        #for batch_index, (inputs, targets) in enumerate(self.trainset_loader_for_train):
        for batch_index, batch_data in enumerate(self.trainset_loader_for_train):
            batch_index += 1

            # compatibilize...
            if self.args.data_type == "image":
                inputs = batch_data[0]
                targets = batch_data[1]
            elif self.args.data_type == "text":
                """
                inputs = batch_data[0]
                #print(inputs.size())
                #print(inputs)
                targets = (batch_data[1] - 1).type(torch.LongTensor)
                #print(targets.size())
                #print(targets)
                """
                """
                if self.args.dataset_full in ['yelprf']:
                    inputs = batch_data[0]
                    #print(inputs.size())
                    #print(inputs)
                    offsets = batch_data[1].cuda()
                    #print(offsets.size())
                    #print(offsets)
                    targets = batch_data[2]
                    #print(targets.size())
                    #print(targets)
                else:
                    inputs = batch_data.text
                    targets = (batch_data.label - 1).type(torch.LongTensor)
                """
                inputs = batch_data.text
                targets = (batch_data.label - 1).type(torch.LongTensor)


            # moving to GPU...
            inputs = inputs.cuda()
            targets = targets.cuda(non_blocking=True)
            
            #print("\n############")
            #print("INPUTS SIZE:")
            #print(inputs.size())
            #print("############\n")

            # compute output
            """
            if self.args.model_name == 'fasttext':
                outputs = self.model(inputs, offsets)
            else:
                outputs = self.model(inputs)
            """
            outputs = self.model(inputs)
            #print("$$$$$$$$$$$$$")
            #print(outputs.size())
            #print("$$$$$$$$$$$$$")

            # compute loss
            loss = self.criterion(outputs, targets)

            # accumulate metrics over batches...
            loss_meter.add(loss.item(), inputs.size(0))
            accuracy_meter.add(outputs.detach(), targets.detach())

            # zero grads, compute gradients and do optimizer step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_index % self.args.print_freq == 0:
                print('Train Epoch: [{0}][{1:3}/{2}]\t'
                      'Loss {loss:.8f}\t\t'
                      'Acc1 {acc1_meter:.2f}\t'
                      .format(self.epoch, batch_index, len(self.trainset_loader_for_train),
                              loss=loss_meter.avg,
                              acc1_meter=accuracy_meter.value()[0],
                              )
                      )

        print('\n#### TRAIN ACC1:\t{0:.4f}\n\n'.format(accuracy_meter.value()[0]))


        return loss_meter.avg, accuracy_meter.value()[0]#, odd_accuracy_meter.value()[0], epoch_logits, epoch_metrics

    def validate_epoch(self):
        print()
        # switch to evaluate mode
        self.model.eval()
        self.criterion.eval()

        # Meters...
        loss_meter = utils.MeanMeter()
        #odd_loss_meter = utils.MeanMeter()
        accuracy_meter = tnt.meter.ClassErrorMeter(topk=[1], accuracy=True)
        #odd_accuracy_meter = tnt.meter.ClassErrorMeter(topk=[1], accuracy=True)
        #epoch_logits = {"intra": [], "inter": []}
        #epoch_metrics = {"max_probs": [], "entropies": []}
        #epoch_entropies_per_classes =  [[] for i in range(self.model.classifier.weights.size(0))]

        with torch.no_grad():

            #for batch_index, (inputs, targets) in enumerate(self.valset_loader):
            for batch_index, batch_data in enumerate(self.valset_loader):
                batch_index += 1

                # compatibilize...
                if self.args.data_type == "image":
                    inputs = batch_data[0]
                    targets = batch_data[1]
                elif self.args.data_type == "text":
                    """
                    if self.args.dataset_full in ['yelprf']:
                        inputs = batch_data[0]
                        #print(inputs.size())
                        #print(inputs)
                        offsets = batch_data[1].cuda()
                        #print(offsets.size())
                        #print(offsets)
                        targets = batch_data[2]
                        #print(targets.size())
                        #print(targets)
                    else:
                        inputs = batch_data.text
                        targets = (batch_data.label - 1).type(torch.LongTensor)
                    """
                    inputs = batch_data.text
                    targets = (batch_data.label - 1).type(torch.LongTensor)

                # moving to GPU...
                inputs = inputs.cuda()
                targets = targets.cuda(non_blocking=True)

                # compute output
                """
                if self.args.model_name == 'fasttext':
                    outputs = self.model(inputs, offsets)
                else:
                    outputs = self.model(inputs)
                """
                outputs = self.model(inputs)

                # compute loss
                loss = self.criterion(outputs, targets)
                #self.model.classifier.metrics_evaluation_mode = False

                # accumulate metrics over batches...
                loss_meter.add(loss.item(), inputs.size(0))
                accuracy_meter.add(outputs.detach(), targets.detach())

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

