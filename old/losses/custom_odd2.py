import torch.nn as nn
import numpy as np
import os
import sklearn
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import pdist, cdist, squareform
import torch.nn.functional as F
#import utils
import torch
import math
import sys

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

__all__ = ['GenericLossFirstPart', 'GenericLossSecondPart']


class GenericLossFirstPart(nn.Module):
    """Replaces classifier layer"""
    def __init__(self, in_features, out_features, type):
        super(GenericLossFirstPart, self).__init__()
        self.type = type
        #################################
        self.inference_transform = False # Apply or not distance transformation during inference...
        self.inference_learn = "NO" # Define which scaling will be used during inference...
        #################################
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.metrics_evaluation_mode = False

        if self.type.startswith("sml"):
            ########################################################
            self.alpha = float(self.type.split("_")[0].strip("sml"))
            ########################################################
            self.bias = nn.Parameter(torch.Tensor(out_features))
            #self.bias.data.zero_()
            #nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
            nn.init.uniform_(self.bias, a=-math.sqrt(3/self.in_features), b=math.sqrt(3/self.in_features))
            nn.init.uniform_(self.weights, a=-math.sqrt(3/self.in_features), b=math.sqrt(3/self.in_features))
            print("init kaiming")

        elif self.type.startswith("dml"):
            #self.register_parameter('bias', None) # Fix this!!!
            ########################################################
            self.alpha = float(self.type.split("_")[0].strip("dml")) ####### NEW!!!!
            ########################################################
            if self.type.split("_")[1].startswith("pn"):
                self.pnorm = float(self.type.split("_")[1].strip("pn"))
                nn.init.constant_(self.weights, 0.0)
                print("init zero for prototypes!!!")
            if self.type.split("_")[1].startswith("ad"):
                nn.init.normal_(self.weights, mean=0.0, std=1.0)
                print("init normal for prototypes!!!")

        elif self.type.startswith("eml"):
            #self.register_parameter('bias', None) # Fix this!!!
            ########################################################
            #self.alpha = float(self.type.split("_")[0].strip("eml")) ####### NEW!!!!
            #self.scales = nn.Parameter(torch.Tensor(1, out_features)) # torch.matmul() tested with BATCH NORMALIZATION!!!
            #self.scales = nn.Parameter(torch.Tensor(1)) # NEVER TESTED!!!
            if self.type.split("_")[7] == "ON":
                self.scales = nn.Parameter(torch.Tensor(1))
                self.tilts = None
                #self.normalization = None
                self.learn = lambda x: self.scales * x
                #################################################
                self.init_scales = float(self.type.split("_")[6])
                nn.init.constant_(self.scales, self.init_scales)
                print("init constant for scales:", self.init_scales)
                print("initialized scales:", self.scales)
            elif self.type.split("_")[7] == "SC":
                self.scales = nn.Parameter(torch.Tensor(out_features))
                self.tilts = None
                #self.normalization = None
                self.learn = lambda x: self.scales * x
                #################################################
                self.init_scales = float(self.type.split("_")[6])
                nn.init.constant_(self.scales, self.init_scales)
                print("init constant for scales:", self.init_scales)
                print("initialized scales:", self.scales)
            elif self.type.split("_")[7] == "ST":
                self.scales = nn.Parameter(torch.Tensor(out_features))
                self.tilts = nn.Parameter(torch.Tensor(out_features))
                #self.normalization = None
                self.learn = lambda x: (self.scales * x) + self.tilts
                #################################################
                self.init_scales = float(self.type.split("_")[6])
                nn.init.constant_(self.scales, self.init_scales)
                print("init constant for scales:", self.init_scales)
                print("initialized scales:", self.scales)
                nn.init.constant_(self.tilts, 0)
                print("init constant for tilts:", 0)
                print("initialized tilts:", self.tilts)
            elif self.type.split("_")[7] == "XC":
                self.scales = nn.Parameter(torch.Tensor(out_features, out_features)) # torch.matmul()
                self.tilts = None
                #self.normalization = None
                self.learn = lambda x: self.scales.matmul(x.t()).t()
                #self.normalization = nn.BatchNorm1d(self.out_features)
                #self.learn = lambda x: self.scales.matmul(self.normalization(x).t()).t()
                ##################################################
                ##################################################
                nn.init.constant_(self.scales, 0.0)
                self.init_scales = float(self.type.split("_")[6])
                self.scales.data.add_(self.init_scales * torch.eye(out_features))
                print("init constant for scales:", self.init_scales)
                print("initialized scales:", self.scales)
            elif self.type.split("_")[7] == "XT":
                self.scales = nn.Parameter(torch.Tensor(out_features, out_features)) # torch.matmul()
                self.tilts = nn.Parameter(torch.Tensor(out_features))
                #self.normalization = None
                self.learn = lambda x: self.scales.matmul(x.t()).t() + self.tilts
                #self.normalization = nn.BatchNorm1d(self.out_features)
                #self.learn = lambda x: self.scales.matmul(self.normalization(x).t()).t() + self.tilts
                ##################################################
                ##################################################
                nn.init.constant_(self.scales, 0.0)
                self.init_scales = float(self.type.split("_")[6])
                self.scales.data.add_(self.init_scales * torch.eye(out_features))
                print("init constant for scales:", self.init_scales)
                print("initialized scales:", self.scales)
                nn.init.constant_(self.tilts, 0)
                print("init constant for tilts:", 0)
                print("initialized tilts:", self.tilts)
            elif self.type.split("_")[7] == "BN":
                self.scales = None
                self.tilts = None
                self.normalization = nn.BatchNorm1d(self.out_features)
                #self.normalization = nn.InstanceNorm1d(self.out_features) #### "IN"
                #self.normalization = nn.LayerNorm(self.out_features) #### "LN"
                self.normalization.weight.data.fill_(float(self.type.split("_")[6]))
                print("\nNORMALIZATION WEIGHTS SIZE:\n", self.normalization.weight.size())
                print("NORMALIZATION WEIGHTS:\n", self.normalization.weight)
                self.normalization.bias.data.fill_(100) #### THE BIAS MAY BE IMPORTANT!!! TRY WITH BIAS=0???
                print("\nNORMALIZATION BIASES SIZE:\n", self.normalization.bias.size())
                print("NORMALIZATION BIASES:\n", self.normalization.bias)
                self.learn = lambda x: self.normalization(x)
            #######################################################

            if self.type.split("_")[1].startswith("pn"):
                self.pnorm = float(self.type.split("_")[1].strip("pn"))
            if self.type.split("_")[2].startswith("id"):
                self.transform = lambda x: x
            elif self.type.split("_")[2].startswith("log"):
                self.transform = lambda x: torch.log(x)

            if self.type.split("_")[5][1] == "z":
                nn.init.constant_(self.weights, 0.0)
                print("\ninit zero for prototypes!!!")
            elif self.type.split("_")[5][1] == "n":
                self.init_prototypes_std = float(self.type.split("_")[5][2])
                nn.init.normal_(self.weights, mean=0.0, std=self.init_prototypes_std)
                print("\ninit normal for prototypes:", self.init_prototypes_std)

            """
            self.init_scales = float(self.type.split("_")[6])
            print("init constant for scales:", self.init_scales)
            nn.init.constant_(self.scales, self.init_scales)
            print("initialized scales:", self.scales)
            """    

        elif self.type.startswith("xml"):
            ########################################################
            #self.alpha = float(self.type.split("_")[0].strip("sml"))
            ########################################################
            self.bias = nn.Parameter(torch.Tensor(out_features))
            #self.bias.data.zero_()
            #nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
            nn.init.uniform_(self.bias, a=-math.sqrt(3/self.in_features), b=math.sqrt(3/self.in_features))
            nn.init.uniform_(self.weights, a=-math.sqrt(3/self.in_features), b=math.sqrt(3/self.in_features))
            print("init kaiming")

            #self.register_parameter('bias', None) # Fix this!!!
            ########################################################
            #self.alpha = float(self.type.split("_")[0].strip("eml"))
            #self.scales = nn.Parameter(torch.Tensor(1, out_features)) # torch.matmul() tested with BATCH NORMALIZATION!!!
            #self.scales = nn.Parameter(torch.Tensor(1)) # NEVER TESTED!!!
            if self.type.split("_")[7] == "ON":
                self.scales = nn.Parameter(torch.Tensor(1))
                self.tilts = None
                #self.normalization = None
                self.learn = lambda x: self.scales * x
                #################################################
                self.init_scales = float(self.type.split("_")[6])
                nn.init.constant_(self.scales, self.init_scales)
                print("init constant for scales:", self.init_scales)
                print("initialized scales:", self.scales)
            elif self.type.split("_")[7] == "SC":
                self.scales = nn.Parameter(torch.Tensor(out_features))
                self.tilts = None
                #self.normalization = None
                self.learn = lambda x: self.scales * x
                #################################################
                self.init_scales = float(self.type.split("_")[6])
                nn.init.constant_(self.scales, self.init_scales)
                print("init constant for scales:", self.init_scales)
                print("initialized scales:", self.scales)
            elif self.type.split("_")[7] == "ST":
                self.scales = nn.Parameter(torch.Tensor(out_features))
                self.tilts = nn.Parameter(torch.Tensor(out_features))
                #self.normalization = None
                self.learn = lambda x: (self.scales * x) + self.tilts
                #################################################
                self.init_scales = float(self.type.split("_")[6])
                nn.init.constant_(self.scales, self.init_scales)
                print("init constant for scales:", self.init_scales)
                print("initialized scales:", self.scales)
                nn.init.constant_(self.tilts, 0)
                print("init constant for tilts:", 0)
                print("initialized tilts:", self.tilts)

            #if self.type.split("_")[1].startswith("pn"):
            #    self.pnorm = float(self.type.split("_")[1].strip("pn"))
            if self.type.split("_")[2].startswith("id"):
                self.transform = lambda x: x
            elif self.type.split("_")[2].startswith("log"):
                self.transform = lambda x: torch.log(x)

            #if self.type.split("_")[5][1] == "z":
            #    nn.init.constant_(self.weights, 0.0)
            #    print("\ninit zero for prototypes!!!")
            #elif self.type.split("_")[5][1] == "n":
            #    self.init_prototypes_std = float(self.type.split("_")[5][2])
            #    nn.init.normal_(self.weights, mean=0.0, std=self.init_prototypes_std)
            #    print("\ninit normal for prototypes:", self.init_prototypes_std)

        print("WEIGHTS/PROTOTYPES SIZE:\n", self.weights.size())
        print("WEIGHTS/PROTOTYPES INITIALIZED:\n", self.weights)
        print("WEIGHTS/PROTOTYPES INITIALIZED [MEAN]:\n", self.weights.mean(dim=0).mean())
        print("WEIGHTS/PROTOTYPES INITIALIZED [STD]:\n", self.weights.std(dim=0).mean())

    def forward(self, features):

        if self.type.startswith("sml"):
            if self.training or self.metrics_evaluation_mode:
                #print("training or in metrics evaluation mode!!!")
                return features
            else:
                #print("pure inferecing!!!")
                affines = features.matmul(self.weights.t()) + self.bias
                logits = affines

                if self.inference_learn == 'NO':
                    inference_logits = logits
                elif self.inference_learn == 'LR':
                    inference_logits = self.alpha*logits

                return inference_logits

        elif self.type.startswith("dml"):
            if self.training or self.metrics_evaluation_mode:
                #print("training or in metrics evaluation mode!!!")
                return features
            else:
                #print("pure inferecing!!!")
                if self.type.split("_")[1].startswith("pn"):
                    #print("euclidean")
                    #pnorm = float(self.type.split("_")[1].strip("pn"))
                    #distances = utils.euclidean_distances(features, self.weights, pnorm)
                    distances = utils.euclidean_distances(features, self.weights, self.pnorm)
                    logits = distances
                elif self.type.split("_")[1].startswith("ad"):
                    print("angular")
                    similarities = utils.cosine_similarity(features, self.weights).clamp_(-0.999999, 0.999999)
                    distances = (torch.acos(similarities) / math.pi).clamp_(0.000001, 0.999999)
                    logits = distances

                if self.inference_learn == 'NO':
                    inference_logits = logits
                elif self.inference_learn == 'LR':
                    inference_logits = self.alpha*logits

                return -inference_logits

        elif self.type.startswith("eml"):
            if self.training or self.metrics_evaluation_mode:
                #print("training or in metrics evaluation mode!!!")
                return features
            else:
                #print("pure inferecing!!!")
                if self.type.split("_")[1].startswith("pn"):
                    #print("euclidean")
                    #pnorm = float(self.type.split("_")[1].strip("pn"))
                    #distances = utils.euclidean_distances(features, self.weights, pnorm)
                    distances = utils.euclidean_distances(features, self.weights, self.pnorm)
                    #logits = distances
                elif self.type.split("_")[1].startswith("ad"):
                    print("angular")
                    similarities = utils.cosine_similarity(features, self.weights).clamp_(-0.999999, 0.999999)
                    distances = (torch.acos(similarities) / math.pi).clamp_(0.000001, 0.999999)
                    #logits = distances

                if self.inference_transform:
                    logits = self.transform(distances)
                else:
                    logits = distances

                if self.inference_learn == 'NO':
                    inference_logits = logits
                elif self.inference_learn == 'LR':
                    inference_logits = self.learn(logits)

                return -inference_logits

        elif self.type.startswith("xml"):
            if self.training or self.metrics_evaluation_mode:
                #print("training or in metrics evaluation mode!!!")
                return features
            else:
                #print("pure inferecing!!!")
                affines = features.matmul(self.weights.t()) + self.bias

                if self.inference_transform:
                    logits = self.transform(affines)
                else:
                    logits = affines

                if self.inference_learn == 'NO':
                    inference_logits = logits
                elif self.inference_learn == 'LR':
                    inference_logits = self.learn(logits)

                return inference_logits

    def extra_repr(self):
        #return 'in_features={}, out_features={}, type={}, bias={}'.format(
        #    self.in_features, self.out_features, self.type, self.bias is not None)
        #return 'in_features={}, out_features={}, type={}'.format(
        #    self.in_features, self.out_features, self.type)
        if self.type.startswith("sml"):
            return 'in_features={}, out_features={}, type={}, bias={}'.format(
                self.in_features, self.out_features, self.type, self.bias is not None)

        elif self.type.startswith("dml"):
            return 'in_features={}, out_features={}, type={}'.format(
                self.in_features, self.out_features, self.type)

        elif self.type.startswith("eml"):
            return 'in_features={}, out_features={}, type={}, scales={}, tilts={}'.format(
                self.in_features, self.out_features, self.type, self.scales is not None, self.tilts is not None)

        elif self.type.startswith("xml"):
            return 'in_features={}, out_features={}, type={}, scales={}, tilts={}'.format(
                self.in_features, self.out_features, self.type, self.scales is not None, self.tilts is not None)


class GenericLossSecondPart(nn.Module):
    def __init__(self, loss_first_part):
        super().__init__()
        self.weights = loss_first_part.weights
        self.type = loss_first_part.type
        self.loss = nn.CrossEntropyLoss()
        #self.alpha = loss_first_part.alpha ####### NEW!!!!
        #self.PLEASE = loss_first_part.training
        if self.type.startswith("sml"):
            self.bias = loss_first_part.bias
            self.alpha = loss_first_part.alpha ####### NEW!!!!
        ########################################
        elif self.type.startswith("dml"): ####### NEW!!!!
            self.alpha = loss_first_part.alpha ####### NEW!!!!
        ########################################
        elif self.type.startswith("eml"):
            self.scales = loss_first_part.scales
            self.tilts = loss_first_part.tilts
            #self.normalization = loss_first_part.normalization
            self.transform = loss_first_part.transform
            self.learn = loss_first_part.learn
        ########################################
        elif self.type.startswith("xml"):
            self.bias = loss_first_part.bias
            self.scales = loss_first_part.scales
            self.tilts = loss_first_part.tilts
            self.transform = loss_first_part.transform
            self.learn = loss_first_part.learn
        ########################################
        if self.type.split("_")[1].startswith("pn"):
            self.pnorm = loss_first_part.pnorm

    def forward(self, features, targets):

        if self.type.startswith("sml"):
            #print("ALPHA:", self.alpha)
            # targets_one_hot = F.one_hot(targets).long().cuda()
            targets_one_hot = torch.eye(self.weights.size(0))[targets].long().cuda()

            if self.type.split("_")[1].startswith("na"):
                #print("affines")
                affines = features.matmul(self.weights.t()) + self.bias
                logits = affines

            intra_inter_affine = torch.where(targets_one_hot != 0, affines, torch.Tensor([float('Inf')]).cuda())
            intra_affines = intra_inter_affine[intra_inter_affine != float('Inf')]
            intra_inter_affine = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), affines)
            inter_affines = intra_inter_affine[intra_inter_affine != float('Inf')]
            #affines = features.matmul(self.weights.t()) + self.bias
            if self.training and self.type.split("_")[3] != "no":
                if self.type.split("_")[3].startswith("rega"):
                    print("rega")
                    gamma = float(self.type.split("_")[3].strip("rega"))
                    print(gamma)
                    first_part = intra_affines.std()
                    second_part = inter_affines.std()
                elif self.type.split("_")[3].startswith("regb"):
                    print("regb")
                    gamma = float(self.type.split("_")[3].strip("regb"))
                    print(gamma)
                    first_part = intra_affines.std() / intra_affines.mean().detach().item()
                    second_part = inter_affines.std() / inter_affines.mean().detach().item()
                # print("FIRST PART:", first_part.item())
                # print("SECOND PART:", second_part.item())
                regularization = gamma * first_part + gamma * second_part
                print("regularization")
            else:
                #print("no_reg")
                regularization = 0

            ####################################################################################
            #probabilities_for_training = nn.Softmax(dim=1)(self.alpha*logits)
            #probabilities_at_targets = probabilities_for_training[range(features.size(0)), targets]
            #loss = -torch.log(probabilities_at_targets).mean() + regularization
            loss = self.loss(self.alpha*logits, targets) + regularization
            ####################################################################################
            """
            probabilities = nn.Softmax(dim=1)(logits)
            entropies = utils.entropies_from_logits(logits)
            entropies_per_classes =  [[] for i in range(self.weights.size(0))]
            for index in range(entropies.size(0)):
                entropies_per_classes[targets[index].item()].append(entropies[index].item())
            """
            ####################################################################################
            #return (loss, logits, intra_affines.tolist(), inter_affines.tolist(),
            return (loss, self.alpha*logits, logits, intra_affines.tolist(), inter_affines.tolist(),
                   #probabilities.max(dim=1)[0].tolist(), entropies.tolist()#, entropies_per_classes
                   )

        elif self.type.startswith("dml"):
            #print("ALPHA:", self.alpha)
            # targets_one_hot = F.one_hot(targets).long().cuda()
            targets_one_hot = torch.eye(self.weights.size(0))[targets].long().cuda()

            if self.type.split("_")[1].startswith("pn"):
                #print("euclidean")
                #pnorm = float(self.type.split("_")[1].strip("pn"))
                #distances = utils.euclidean_distances(features, self.weights, pnorm)
                distances = utils.euclidean_distances(features, self.weights, self.pnorm)
                logits = distances
                ####################################################################
            elif self.type.split("_")[1].startswith("ad"):
                """
                weights_norms = self.weights.norm(p=2, dim=1, keepdim=True)
                normalized_weights = self.weights / (weights_norms+0.001)  # this sum may change final results...
                feature_norms = features.norm(p=2, dim=1, keepdim=True)
                normalized_features = features / (feature_norms+0.001)  # this sum may change final results...
                """
                print("angular")
                similarities = utils.cosine_similarity(features, self.weights).clamp_(-0.999999, 0.999999)
                distances = (torch.acos(similarities) / math.pi).clamp_(0.000001, 0.999999)
                logits = distances

            ####################################################################################################
            ####################################################################################################
            #guaged_distances = distances - distances.mean(dim=1).unsqueeze(1).detach() + 10
            #print(distances)
            #guaged_distances = distances #- distances.mean(dim=1).unsqueeze(1).detach() + 10
            #print(guaged_distances)
            intra_inter_distances = torch.where(targets_one_hot != 0, -distances, distances)
            intra_distances = -intra_inter_distances[intra_inter_distances <= 0]
            inter_distances = intra_inter_distances[intra_inter_distances > 0]
            ####################################################################################################
            #if self.type.split("_")[3] != "no":
            if self.training and self.type.split("_")[3] != "no":
                if self.type.split("_")[3].startswith("rega"):
                    print("rega")
                    gamma = float(self.type.split("_")[3].strip("rega"))
                    print(gamma)
                    first_part = intra_distances.std()
                    second_part = inter_distances.std()
                elif self.type.split("_")[3].startswith("regb"):
                    print("regb")
                    gamma = float(self.type.split("_")[3].strip("regb"))
                    print(gamma)
                    first_part = intra_distances.std() / intra_distances.mean().detach().item()
                    #first_part = 0
                    second_part = inter_distances.std() / inter_distances.mean().detach().item()
                    #second_part = 0
                #print("FIRST PART:", first_part.item())
                #print("SECOND PART:", second_part.item())
                regularization = gamma*first_part + gamma*second_part
                print("regularization!!!")
            else:
                #print("no_reg")
                regularization = 0
            ####################################################################################################
            ####################################################################################################

            ####################################################################################
            probabilities_for_training = nn.Softmax(dim=1)(-self.alpha*logits)
            probabilities_at_targets = probabilities_for_training[range(features.size(0)), targets]
            loss = -torch.log(probabilities_at_targets).mean() + regularization
            #loss = self.loss(-self.alpha*logits, targets) + regularization
            ####################################################################################
            """
            probabilities = nn.Softmax(dim=1)(-logits)
            entropies = utils.entropies_from_logits(-logits)
            entropies_per_classes =  [[] for i in range(self.weights.size(0))]
            for index in range(entropies.size(0)):
                entropies_per_classes[targets[index].item()].append(entropies[index].item())
            """
            ####################################################################################
            #return (loss, -logits, intra_distances.tolist(), inter_distances.tolist(),
            return (loss, -self.alpha*logits, -logits, intra_distances.tolist(), inter_distances.tolist(),
                   #probabilities.max(dim=1)[0].tolist(), entropies.tolist()#, entropies_per_classes
                   )

############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
        elif self.type.startswith("eml"):
            #print("ALPHA:", self.alpha)
            # targets_one_hot = F.one_hot(targets).long().cuda()
            targets_one_hot = torch.eye(self.weights.size(0))[targets].long().cuda()
            #################################################################
            """
            if self.training and (not self.type.split("_")[4].startswith("no")):
                noise = float(self.type.split("_")[4])
                #features = torch.add(features, noise, torch.randn(features.size()).cuda())
                features = torch.add(features, noise, features * torch.randn(features.size()).cuda())
                #features = torch.add(features, noise, torch.norm(features, p=2, dim=1, keepdim=True) * torch.randn(features.size()).cuda())
                print("Noise!!!")
            """
            #################################################################

            if self.type.split("_")[1].startswith("pn"):
                #print("euclidean")
                #pnorm = float(self.type.split("_")[1].strip("pn"))
                #distances = utils.euclidean_distances(features, self.weights, pnorm)
                distances = utils.euclidean_distances(features, self.weights, self.pnorm)
                #logits = distances
            elif self.type.split("_")[1].startswith("ad"):
                """
                weights_norms = self.weights.norm(p=2, dim=1, keepdim=True)
                normalized_weights = self.weights / (weights_norms+0.001)  # this sum may change final results...
                feature_norms = features.norm(p=2, dim=1, keepdim=True)
                normalized_features = features / (feature_norms+0.001)  # this sum may change final results...
                """
                print("angular")
                similarities = utils.cosine_similarity(features, self.weights).clamp_(-0.999999, 0.999999)
                distances = (torch.acos(similarities) / math.pi).clamp_(0.000001, 0.999999)
                #logits = distances

            logits = self.transform(distances)
            learned_logits = self.learn(logits)
            #print("\nLEARNED LOGITS:\n", learned_logits, "\n")

            ####################################################################################################
            intra_inter_distances = torch.where(targets_one_hot != 0, -distances, distances)
            intra_distances = -intra_inter_distances[intra_inter_distances <= 0].view(distances.size(0),-1)
            inter_distances = intra_inter_distances[intra_inter_distances > 0].view(distances.size(0),-1)
            ####################################################################################################
            #if self.type.split("_")[3] != "no":
            if self.training and self.type.split("_")[3] != "no":
                if self.type.split("_")[3].startswith("rega"):
                    #print("rega")
                    gamma = float(self.type.split("_")[3].strip("rega"))
                    #print(gamma)
                    #first_part = intra_distances.std()
                    #################################################
                    first_part = (intra_distances.flatten() - inter_distances.mean(dim=1)).var()
                    print((intra_distances.flatten() - inter_distances.mean(dim=1)).var().item())
                    #################################################
                    #second_part = inter_distances.std()
                    #################################################
                    second_part = inter_distances.var(dim=1).mean()
                    print(inter_distances.var(dim=1).mean().item())
                    #################################################
                elif self.type.split("_")[3].startswith("regb"):
                    print("regb")
                    gamma = float(self.type.split("_")[3].strip("regb"))
                    print(gamma)
                    first_part = intra_distances.std() / intra_distances.mean().detach().item()
                    #first_part = 0
                    second_part = inter_distances.std() / inter_distances.mean().detach().item()
                    #second_part = 0
                #print("FIRST PART:", first_part.item())
                #print("SECOND PART:", second_part.item())
                #regularization = gamma*first_part + gamma*second_part
                regularization = gamma*first_part
                print("regularization!!!")
            else:
                #print("no_reg")
                regularization = 0
            #### WE CAN ALSO USE ENTROPY FOR REGULARIZATION!!!
            #regularization = utils.entropies_from_probabilities(nn.Softmax(dim=1)(-alpha*logits)).mean()

            ################################################################################################################
            probabilities_for_training = nn.Softmax(dim=1)(-learned_logits)
            probabilities_at_targets = probabilities_for_training[range(features.size(0)), targets]
            loss = -torch.log(probabilities_at_targets).mean() + regularization
            #loss = -((1-probabilities_at_targets)*torch.log(probabilities_at_targets)).mean() + regularization
            #loss = -(torch.exp(probabilities_at_targets)*torch.log(probabilities_at_targets)).mean() + regularization
            #loss = self.loss(-learned_logits, targets) + regularization
            ################################################################################################################
            """
            probabilities = nn.Softmax(dim=1)(-logits) #### <<== LEARNED LOGITS??? MAY HAVE SCALE AND 10X FACTOR!!!
            entropies = utils.entropies_from_logits(-logits) #### <<== LEARNED LOGITS??? MAY HAVE SCALE AND 10X FACTOR!!!
            entropies_per_classes =  [[] for i in range(self.weights.size(0))]
            for index in range(entropies.size(0)):
                entropies_per_classes[targets[index].item()].append(entropies[index].item())
            """
            ################################################################################################################
            #return (loss, -learned_logits, intra_distances.flatten().tolist(), inter_distances.flatten().tolist(), #### <<== LOGITS???
            return (loss, -learned_logits, -logits, intra_distances.flatten().tolist(), inter_distances.flatten().tolist(), #### <<== LOGITS???
                   #probabilities.max(dim=1)[0].tolist(), entropies.tolist()#, entropies_per_classes
                   )
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################

        if self.type.startswith("xml"):
            #print("ALPHA:", self.alpha)
            # targets_one_hot = F.one_hot(targets).long().cuda()
            targets_one_hot = torch.eye(self.weights.size(0))[targets].long().cuda()

            if self.type.split("_")[1].startswith("na"):
                #print("affines")
                affines = features.matmul(self.weights.t()) + self.bias
                #logits = affines

            #print(self.scales)
            #print(self.tilts)
            logits = self.transform(affines)
            learned_logits = self.learn(logits)
            #print("\nLEARNED LOGITS:\n", learned_logits, "\n")

            intra_inter_affine = torch.where(targets_one_hot != 0, affines, torch.Tensor([float('Inf')]).cuda())
            intra_affines = intra_inter_affine[intra_inter_affine != float('Inf')]
            intra_inter_affine = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), affines)
            inter_affines = intra_inter_affine[intra_inter_affine != float('Inf')]
            #affines = features.matmul(self.weights.t()) + self.bias
            if self.training and self.type.split("_")[3] != "no":
                if self.type.split("_")[3].startswith("rega"):
                    print("rega")
                    gamma = float(self.type.split("_")[3].strip("rega"))
                    print(gamma)
                    first_part = intra_affines.std()
                    second_part = inter_affines.std()
                elif self.type.split("_")[3].startswith("regb"):
                    print("regb")
                    gamma = float(self.type.split("_")[3].strip("regb"))
                    print(gamma)
                    first_part = intra_affines.std() / intra_affines.mean().detach().item()
                    second_part = inter_affines.std() / inter_affines.mean().detach().item()
                # print("FIRST PART:", first_part.item())
                # print("SECOND PART:", second_part.item())
                regularization = gamma * first_part + gamma * second_part
                print("regularization")
            else:
                #print("no_reg")
                regularization = 0

            ####################################################################################
            #probabilities_for_training = nn.Softmax(dim=1)(learned_logits*logits)
            #probabilities_at_targets = probabilities_for_training[range(features.size(0)), targets]
            #loss = -torch.log(probabilities_at_targets).mean() + regularization
            loss = self.loss(learned_logits, targets) + regularization
            ####################################################################################
            """
            probabilities = nn.Softmax(dim=1)(logits)
            entropies = utils.entropies_from_logits(logits)
            entropies_per_classes =  [[] for i in range(self.weights.size(0))]
            for index in range(entropies.size(0)):
                entropies_per_classes[targets[index].item()].append(entropies[index].item())
            """
            ####################################################################################
            #return (loss, learned_logits, intra_affines.tolist(), inter_affines.tolist(),
            return (loss, learned_logits, logits, intra_affines.tolist(), inter_affines.tolist(),
                   #probabilities.max(dim=1)[0].tolist(), entropies.tolist()#, entropies_per_classes
                   )



###########################################################################
#transformed_distances = torch.log(angular_distances)
#transformed_distances = torch.log(angular_distances/(1-angular_distances))
#transformed_distances = -torch.log(1-angular_distances) # tested and did not work... Maybe using F.cosine_similarity...
#transformed_distances = -(1-angular_distances)
#transformed_distances = angular_distances/(1-angular_distances)
###########################################################################

############################################################################################
#RUN WITH ALPHA = 10, SCALE INIT in [10, 100, 1000, 10000] (similar to small learning rate, but different!!!)
#Do I really need to train this option??? This rally appears to be the second option below!!!
#scaled_distances = (self.scales/self.scales.mean().detach().item()) * distances
#print("EFFECTIVE SCALES:\n", self.scales/self.scales.mean().detach().item())
#print("DISTANCES:\n", distances)
#print("SCALED DISTANCES:\n", scaled_distances)
