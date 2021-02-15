import torch.nn as nn
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
        self.inference_transform = False # Apply or not distance transformation during inference...
        self.inference_learn = "NO" # Define which scaling will be used during inference...
        self.in_features = in_features
        self.out_features = out_features
        #self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.metrics_evaluation_mode = False
        #self.batches_processed = 0 ########

        if self.type.startswith("sml"):
            self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias = nn.Parameter(torch.Tensor(out_features))
            self.alpha = float(self.type.split("_")[0].strip("sml"))
            self.split = self.type.split("_")[4]

            #nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
            #nn.init.uniform_(self.bias, -math.sqrt(1/self.in_features), math.sqrt(1/self.in_features))
            #### IF WE CHANGE THE ORDER OF INITIALIZATION, ALPHA=10 FOR SOFTMAX DOES WORK!!! ####
            #### IF WE CHANGE THE ORDER OF INITIALIZATION, ALPHA=10 FOR SOFTMAX DOES WORK!!! ####
            nn.init.uniform_(self.weights, a=-math.sqrt(3/self.in_features), b=math.sqrt(3/self.in_features)) # _bs64 DEFINITIVE!!!
            nn.init.uniform_(self.bias, a=-math.sqrt(3/self.in_features), b=math.sqrt(3/self.in_features)) # _bs64 DEFINITIVE!!!
            #### IF WE CHANGE THE ORDER OF INITIALIZATION, ALPHA=10 FOR SOFTMAX DOES WORK!!! ####
            #### IF WE CHANGE THE ORDER OF INITIALIZATION, ALPHA=10 FOR SOFTMAX DOES WORK!!! ####
            ##nn.init.uniform_(self.weights, -math.sqrt(1/self.in_features), math.sqrt(1/self.in_features)) #<<<<====
            ##nn.init.uniform_(self.bias, -math.sqrt(1/self.in_features), math.sqrt(1/self.in_features)) #<<<<====
            print("init softmax!!!")

        elif self.type.startswith("dml"):
            #self.register_parameter('bias', None) # Fix this!!!
            #############################################################################################
            self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
            #############################################################################################
            #self.weights = nn.Parameter(torch.Tensor(5, out_features, in_features))
            self.alpha = float(self.type.split("_")[0].strip("dml")) ####### NEW!!!!
            self.split = self.type.split("_")[4]
            ##########################################
            self.init_scales = float(self.type.split("_")[6])
            self.scales = nn.Parameter(torch.Tensor(1))            
            nn.init.constant_(self.scales, self.init_scales)
            print("init constant for scales:", self.init_scales)
            print("initialized scales:", self.scales)
            self.learn = lambda x: self.scales * x
            ##########################################

            self.distance = self.type.split("_")[1]
            #"""
            if self.distance.startswith("pn"):
                nn.init.constant_(self.weights, 0)
                print("\ninit ==>> ZERO <<== for prototypes!!!")
                #nn.init.normal_(self.weights, mean=0.0, std=0.1)
                #print("init normal for prototypes!!!")
            elif self.distance.startswith("ad"):
                nn.init.normal_(self.weights, mean=0.0, std=1.0)
                print("init normal for prototypes!!!")
            #"""
            
            #self.proc_prototypes = self.type.split("_")[5]
            """
            if self.proc_prototypes[1] == "z":
                nn.init.constant_(self.weights, 0)
                print("\ninit ==>> ZERO <<== for prototypes!!!")
            elif self.proc_prototypes[1] == "o":
                nn.init.constant_(self.weights, 0)
                print("\ninit ==>> ZERO <<== for prototypes!!!")
            elif self.proc_prototypes[1] == "b":
                nn.init.constant_(self.weights, 0.3)
                print("\ninit ==>> 0.3 <<== for prototypes!!!")
            else:
                nn.init.constant_(self.weights, 0)
                print("\ninit ==>> ZERO <<== for prototypes!!!")
            """
            #nn.init.constant_(self.weights, 0)
            #print("\ninit ==>> ZERO <<== for prototypes!!!")


            """
            ################################################
            self.initprot = self.type.split("_")[6]
            self.vectors_sum = torch.zeros(out_features, in_features).cuda()
            self.vectors_total = torch.ones(out_features).cuda()#[0] * out_features
            print("\nVECTORS SUM:\n", self.vectors_sum)
            print("VECTORS TOTAL:\n", self.vectors_total, "\n")
            ################################################
            """
            #self.plus = False

        elif self.type.startswith("eml"):
            #self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
            self.alpha = float(self.type.split("_")[0].strip("eml")) ####### NEW!!!!
            self.prototypes_per_class = int(self.type.split("_")[10][2:]) # working fine...
            print("Number of prototypes:", self.prototypes_per_class)
            self.weights = nn.Parameter(torch.Tensor(self.prototypes_per_class, out_features, in_features))
            self.init_scales = float(self.type.split("_")[6])
            self.split = self.type.split("_")[4]

            if self.type.split("_")[7] == "ON":
                self.scales = nn.Parameter(torch.Tensor(1))
                self.tilts = None
                self.learn = lambda x: self.scales * x
                nn.init.constant_(self.scales, self.init_scales)
                print("init constant for scales:", self.init_scales)
                print("initialized scales:", self.scales)
            elif self.type.split("_")[7].startswith("sc"):
                ##########################################################################################
                ##########################################################################################
                if self.type.split("_")[7].strip("sc") == "1":
                    self.scales = nn.Parameter(torch.Tensor(1)) # 1
                    nn.init.constant_(self.scales, self.init_scales)
                    print("init constant for scales:", self.init_scales)
                elif self.type.split("_")[7].strip("sc") == "2":
                    self.scales = nn.Parameter(torch.Tensor(out_features)) # 2
                    nn.init.constant_(self.scales, self.init_scales)
                    print("init constant for scales:", self.init_scales)
                elif self.type.split("_")[7].strip("sc") == "3":
                    self.scales = nn.Parameter(torch.Tensor(self.prototypes_per_class, 1)) # 3
                    nn.init.constant_(self.scales, self.init_scales)
                    print("init constant for scales:", self.init_scales)
                elif self.type.split("_")[7].strip("sc") == "4":
                    self.scales = nn.Parameter(torch.Tensor(self.prototypes_per_class, out_features)) # 4
                    nn.init.constant_(self.scales, self.init_scales)
                    print("init constant for scales:", self.init_scales)
                elif self.type.split("_")[7].strip("sc") == "x":
                    self.scales = nn.Parameter(torch.Tensor(self.prototypes_per_class, 1)) # 3
                    if self.prototypes_per_class == 1:
                        self.scales.data = torch.Tensor([10.0]).unsqueeze(1)
                    elif self.prototypes_per_class == 2:
                        self.scales.data = torch.Tensor([10.0,10.5]).unsqueeze(1)
                    elif self.prototypes_per_class == 4:
                        self.scales.data = torch.Tensor([10.0,10.5,11.0,11.5]).unsqueeze(1)
                    elif self.prototypes_per_class == 8:
                        #self.scales.data = torch.Tensor([10,10,10,10,10,10,10,10]).unsqueeze(1)
                        self.scales.data = torch.Tensor([10.0,10.5,11.0,11.5,12.0,12.5,13.0,13.5]).unsqueeze(1)
                    elif self.prototypes_per_class == 16:
                        #self.scales.data = torch.Tensor([10,10,10,10,10,10,10,10]).unsqueeze(1)
                        self.scales.data = torch.Tensor([10.0,10.5,11.0,11.5,12.0,12.5,13.0,13.5,
                                                         14.0,14.5,15.0,15.5,16.0,16.5,17.0,17.5]).unsqueeze(1)
                print("########################################")
                print("SCALES DIMENSIONS:", self.scales.size())
                print("INITIALIZED SCALES:", self.scales)
                print("########################################")
                ##########################################################################################
                ##########################################################################################
                self.tilts = None
                self.learn = lambda x: self.scales * x
                #nn.init.constant_(self.tilts, 0)
                #print("init constant for tilts:", 0)
                #print("initialized tilts:", self.tilts)
            elif self.type.split("_")[7] == "st":
                self.scales = nn.Parameter(torch.Tensor(out_features))
                self.tilts = nn.Parameter(torch.Tensor(out_features))
                self.learn = lambda x: (self.scales * x) + self.tilts
                nn.init.constant_(self.scales, self.init_scales)
                print("init constant for scales:", self.init_scales)
                print("initialized scales:", self.scales)
                nn.init.constant_(self.tilts, 0)
                print("init constant for tilts:", 0)
                print("initialized tilts:", self.tilts)
            elif self.type.split("_")[7] == "tt":
                self.scales = None
                self.tilts = nn.Parameter(torch.Tensor(out_features))
                self.learn = lambda x: (self.init_scales * x) + self.tilts
                #nn.init.constant_(self.scales, self.init_scales)
                #print("init constant for scales:", self.init_scales)
                #print("initialized scales:", self.scales)
                nn.init.constant_(self.tilts, 0)
                print("init constant for tilts:", 0)
                print("initialized tilts:", self.tilts)

            if self.type.split("_")[2].startswith("id"):
                self.transform = lambda x: x
            elif self.type.split("_")[2].startswith("log"):
                self.transform = lambda x: torch.log(x)

            self.distance = self.type.split("_")[1]
            self.model_data_not_initialized = True
            self.proc_prototypes = self.type.split("_")[5]

            #"""
            if self.distance.startswith("pn"):
                if self.proc_prototypes[1] == "z":
                    nn.init.constant_(self.weights, 0)
                    print("\ninit ==>> ZERO <<== for prototypes!!!")
                ###############################################################
                ###############################################################
                elif self.proc_prototypes[1] == "n":
                    nn.init.normal_(self.weights, mean=0.0, std=0.1)
                    print("\ninit ==>> NORMAL STD=0.1 <<== for prototypes!!!")
                elif self.proc_prototypes[1:] == "n1":
                    nn.init.normal_(self.weights, mean=0.0, std=0.1)
                    print("\ninit ==>> NORMAL STD=0.1 <<== for prototypes!!!")
                ###############################################################
                ###############################################################
                elif self.proc_prototypes[1:] == "n2":
                    nn.init.normal_(self.weights, mean=0.0, std=0.01)
                    print("\ninit ==>> NORMAL STD=0.01 <<== for prototypes!!!")
                elif self.proc_prototypes[1:] == "n3":
                    nn.init.normal_(self.weights, mean=0.0, std=0.001)
                    print("\ninit ==>> NORMAL STD=0.001 <<== for prototypes!!!")
                elif self.proc_prototypes[1] == "d":
                    if self.prototypes_per_class == 1:
                        #temp_tensor = torch.Tensor([0.0])
                        temp_tensor = torch.Tensor([0])
                        self.weights.data = temp_tensor.unsqueeze(1).unsqueeze(2).repeat(1,self.weights.size(1),self.weights.size(2))                   
                    if self.prototypes_per_class == 2:
                        #temp_tensor = torch.Tensor([0.5,-0.5])
                        temp_tensor = torch.Tensor([0.1,-0.1])
                        self.weights.data = temp_tensor.unsqueeze(1).unsqueeze(2).repeat(1,self.weights.size(1),self.weights.size(2))                   
                    if self.prototypes_per_class == 4:
                        #temp_tensor = torch.Tensor([0.5,-0.5,1.0,-1.0])
                        temp_tensor = torch.Tensor([0.1,-0.1,0.2,-0.2])
                        self.weights.data = temp_tensor.unsqueeze(1).unsqueeze(2).repeat(1,self.weights.size(1),self.weights.size(2))                   
                    if self.prototypes_per_class == 8:
                        #temp_tensor = torch.Tensor([0.5,-0.5,1.0,-1.0,1.5,-1.5,2.0,-2.0])
                        temp_tensor = torch.Tensor([0.1,-0.1,0.2,-0.2,0.3,-0.3,0.4,-0.4])
                        self.weights.data = temp_tensor.unsqueeze(1).unsqueeze(2).repeat(1,self.weights.size(1),self.weights.size(2))                   
                    print("\ninit ==>> CONSTANT!!! <<== for prototypes!!!")
                #nn.init.constant_(self.weights, 0)
                #print("\ninit ==>> ZERO <<== for prototypes!!!")
                #################################################
                #################################################
                #nn.init.normal_(self.weights, mean=0.0, std=0.01)
                #nn.init.normal_(self.weights, mean=0.0, std=0.1)
                #nn.init.normal_(self.weights, mean=0.0, std=1.0)
                #print("init normal for prototypes!!!")
                #################################################
                #################################################
            elif self.distance.startswith("ad"):
                nn.init.normal_(self.weights, mean=0.0, std=1.0)
                print("\ninit ==>> NORMAL STD=1.0 <<== for prototypes!!!")
            """
            elif self.distance.startswith("mm"):
                nn.init.normal_(self.weights, mean=0.0, std=0.01)
                print("init normal std=0.01 for prototypes!!!")
            """
            #"""

            """
            if self.proc_prototypes[1] == "z":
                nn.init.constant_(self.weights, 0)
                print("\ninit ==>> ZERO <<== for prototypes!!!")
            elif self.proc_prototypes[1:] == "n0.1":
                nn.init.normal_(self.weights, mean=0.0, std=0.1)
                print("\ninit ==>> NORMAL STD=0.1 <<== for prototypes!!!")
            """
            #nn.init.constant_(self.weights, 0)
            #print("\ninit ==>> ZERO <<== for prototypes!!!")
            

        elif self.type.startswith("cml"):
            self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
            #self.prototypes = torch.Tensor(out_features, in_features)
            #self.prototypes = nn.Parameter(torch.Tensor(out_features, in_features))
            self.init_scales = float(self.type.split("_")[6])
            self.split = self.type.split("_")[4]
            #self.plus = True

            if self.type.split("_")[7] == "ON":
                self.scales = nn.Parameter(torch.Tensor(1))
                self.tilts = None
                self.learn = lambda x: self.scales * x
                #self.init_scales = float(self.type.split("_")[6])
                nn.init.constant_(self.scales, self.init_scales)
                print("init constant for scales:", self.init_scales)
                print("initialized scales:", self.scales)
            elif self.type.split("_")[7] == "SC":
                self.scales = nn.Parameter(torch.Tensor(out_features))
                self.tilts = None
                self.learn = lambda x: self.scales * x
                #self.init_scales = float(self.type.split("_")[6])
                nn.init.constant_(self.scales, self.init_scales)
                print("init constant for scales:", self.init_scales)
                print("initialized scales:", self.scales)
            elif self.type.split("_")[7] == "ST":
                self.scales = nn.Parameter(torch.Tensor(out_features))
                self.tilts = nn.Parameter(torch.Tensor(out_features))
                self.learn = lambda x: (self.scales * x) + self.tilts
                #self.init_scales = float(self.type.split("_")[6])
                nn.init.constant_(self.scales, self.init_scales)
                print("init constant for scales:", self.init_scales)
                print("initialized scales:", self.scales)
                nn.init.constant_(self.tilts, 0)
                print("init constant for tilts:", 0)
                print("initialized tilts:", self.tilts)
            elif self.type.split("_")[7] == "AA":
                self.scales = nn.Parameter(torch.Tensor(out_features))
                self.tilts = nn.Parameter(torch.Tensor(out_features))
                self.learn = lambda x: (self.scales * x) + self.tilts
                #self.init_scales = float(self.type.split("_")[6])
                nn.init.constant_(self.scales, self.init_scales)
                print("init constant for scales:", self.init_scales)
                print("initialized scales:", self.scales)
                nn.init.constant_(self.tilts, 0)
                print("init constant for tilts:", 0)
                print("initialized tilts:", self.tilts)

            if self.type.split("_")[2].startswith("id"):
                self.transform = lambda x: x
            elif self.type.split("_")[2].startswith("log"):
                print("log!!!")
                self.transform = lambda x: torch.log(x)

            self.distance = self.type.split("_")[1]
            #"""
            if self.distance.startswith("pn"):
                nn.init.constant_(self.weights, 0)
                print("\ninit ==>> ZERO <<== for prototypes!!!")
            elif self.distance.startswith("ad"):
                nn.init.normal_(self.weights, mean=0.0, std=1.0)
                print("init normal for prototypes!!!")
            #"""

            #self.proc_prototypes = self.type.split("_")[5]
            """
            if self.proc_prototypes[1] == "z":
                nn.init.constant_(self.weights, 0)
                print("\ninit ==>> ZERO <<== for prototypes!!!")
            elif self.proc_prototypes[1] == "o":
                nn.init.constant_(self.weights, 0)
                print("\ninit ==>> ZERO <<== for prototypes!!!")
            elif self.proc_prototypes[1] == "b":
                nn.init.constant_(self.weights, 0.3)
                print("\ninit ==>> 0.3 <<== for prototypes!!!")
            else:
                nn.init.constant_(self.weights, 0)
                print("\ninit ==>> ZERO <<== for prototypes!!!")
            """
            #nn.init.constant_(self.weights, 0)
            #print("\ninit ==>> ZERO <<== for prototypes!!!")


        print("\nWEIGHTS/PROTOTYPES SIZE:\n", self.weights.size())
        print("WEIGHTS/PROTOTYPES INITIALIZED [MEAN]:\n", self.weights.mean(dim=0).mean())
        print("WEIGHTS/PROTOTYPES INITIALIZED [STD]:\n", self.weights.std(dim=0).mean())
        print("WEIGHTS/PROTOTYPES INITIALIZED:\n", self.weights)


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
                    ood_logits = logits
                elif self.inference_learn == 'LR':
                    ood_logits = self.alpha*logits

                #return ood_logits
                #print(nn.Softmax(dim=1)(ood_logits).size())
                return nn.Softmax(dim=1)(ood_logits)

        elif self.type.startswith("dml"):
            if self.training or self.metrics_evaluation_mode:
                #print("training or in metrics evaluation mode!!!")
                return features
            else:
                #print("pure inferecing!!!")
                if self.distance.startswith("pn"):
                    #print("euclidean")
                    distances = utils.euclidean_distances(features, self.weights, 2)
                elif self.distance.startswith("ad"):
                    print("angular")
                    #similarities = utils.cosine_similarity(features, self.weights).clamp_(-0.999999, 0.999999)
                    similarities = torch.mm(F.normalize(features), torch.transpose(F.normalize(self.weights), 0, 1))
                    distances = (torch.acos(similarities) / math.pi)#.clamp_(0.000001, 0.999999)
                    #distances = -similarities

                logits = distances

                if self.inference_learn == 'NO':
                    ood_logits = logits
                #elif self.inference_learn == 'LR':
                #    ood_logits = self.alpha*logits
                elif self.inference_learn == 'LR':
                    ood_logits = self.learn(logits)

                #return -ood_logits
                #print(nn.Softmax(dim=1)(-ood_logits).size())
                return nn.Softmax(dim=1)(-ood_logits)

        elif self.type.startswith("eml"):
            if self.training or self.metrics_evaluation_mode:
                #print("training or in metrics evaluation mode!!!")
                return features
            else:
                #print("pure inferecing!!!")
                if self.distance.startswith("pn"):
                    #print("euclidean")
                    distances = utils.euclidean_distances(features, self.weights, 2)
                elif self.distance.startswith("ad"):
                    print("angular")
                    #similarities = utils.cosine_similarity(features, self.weights).clamp_(-0.999999, 0.999999)
                    similarities = torch.mm(F.normalize(features), torch.transpose(F.normalize(self.weights), 0, 1))
                    distances = (torch.acos(similarities) / math.pi)#.clamp_(0.000001, 0.999999)
                    #distances = -similarities

                #"""
                if self.inference_transform:
                    logits = self.transform(distances)
                else:
                    logits = distances
                #"""

                if self.inference_learn == 'NO':
                    ood_logits = logits
                #elif self.inference_learn == 'ML':
                #    ood_logits = self.prototypes_per_class * logits
                #elif self.inference_learn == 'DV':
                #    ood_logits = logits / self.prototypes_per_class
                #elif self.inference_learn == 'XR':
                #    ood_logits = self.learn(logits)
                elif self.inference_learn == 'LR':
                    ood_logits = self.learn(logits)
                #############################################################
                #############################################################

                #return -ood_logits
                return nn.Softmax(dim=2)(-ood_logits).mean(dim=1)
                #return nn.Softmax(dim=2)(-ood_logits)

        elif self.type.startswith("cml"):
            if self.training or self.metrics_evaluation_mode:
                #print("training or in metrics evaluation mode!!!")
                return features
            else:
                #print("pure inferecing!!!")
                if self.distance.startswith("pn"):
                    #print("euclidean")
                    distances = utils.euclidean_distances(features, self.weights, 2)
                elif self.distance.startswith("ad"):
                    print("angular")
                    #similarities = utils.cosine_similarity(features, self.weights).clamp_(-0.999999, 0.999999)
                    similarities = torch.mm(F.normalize(features), torch.transpose(F.normalize(self.weights), 0, 1))
                    distances = (torch.acos(similarities) / math.pi)#.clamp_(0.000001, 0.999999)
                    #distances = -similarities

                #"""
                if self.inference_transform:
                    logits = self.transform(distances)
                else:
                    logits = distances
                #"""

                if self.inference_learn == 'NO':
                    ood_logits = logits
                elif self.inference_learn == 'LR':
                    ood_logits = self.learn(logits)
                #############################################################
                #############################################################

                return -ood_logits


    def extra_repr(self):
        if self.type.startswith("sml"):
            return 'in_features={}, out_features={}, type={}, bias={}'.format(
                self.in_features, self.out_features, self.type, self.bias is not None)

        elif self.type.startswith("dml"):
            return 'in_features={}, out_features={}, type={}, scales={}'.format(
                self.in_features, self.out_features, self.type, self.scales is not None)

        elif self.type.startswith("eml"):
            return 'in_features={}, out_features={}, type={}, scales={}, tilts={}'.format(
                self.in_features, self.out_features, self.type, self.scales is not None, self.tilts is not None)

        elif self.type.startswith("cml"):
            return 'in_features={}, out_features={}, type={}, scales={}, tilts={}'.format(
                self.in_features, self.out_features, self.type, self.scales is not None, self.tilts is not None)


class GenericLossSecondPart(nn.Module):
    def __init__(self, loss_first_part):
        super().__init__()
        self.weights = loss_first_part.weights
        self.type = loss_first_part.type
        self.loss = nn.CrossEntropyLoss()
        self.split = loss_first_part.split
        if self.type.startswith("sml"):
            self.bias = loss_first_part.bias
            self.alpha = loss_first_part.alpha
        elif self.type.startswith("dml"): 
            self.distance = loss_first_part.distance
            self.alpha = loss_first_part.alpha
            self.scales = loss_first_part.scales
            self.learn = loss_first_part.learn
            self.init_scales = loss_first_part.init_scales
            #self.plus = loss_first_part.plus
            ################################################
            ################################################
            #self.vectors_sum = loss_first_part.vectors_sum
            #self.vectors_total = loss_first_part.vectors_total
            """
            self.initprot = self.type.split("_")[6]
            self.vectors_sum = torch.zeros(loss_first_part.out_features, loss_first_part.in_features).cuda()
            self.vectors_total = torch.ones(loss_first_part.out_features).cuda()#[0] * out_features
            """
            #print("\nVECTORS SUM:\n", self.vectors_sum)
            #print("VECTORS TOTAL:\n", self.vectors_total, "\n")
            ################################################
            ################################################
        elif self.type.startswith("eml"):
            self.alpha = loss_first_part.alpha
            self.distance = loss_first_part.distance
            self.scales = loss_first_part.scales
            self.tilts = loss_first_part.tilts
            self.transform = loss_first_part.transform
            self.learn = loss_first_part.learn
            self.proc_prototypes = loss_first_part.proc_prototypes
            self.model_data_not_initialized = loss_first_part.model_data_not_initialized
            self.prototypes_per_class = loss_first_part.prototypes_per_class
            #self.batches_processed = loss_first_part.batches_processed
        elif self.type.startswith("cml"):
            self.distance = loss_first_part.distance
            self.scales = loss_first_part.scales
            self.tilts = loss_first_part.tilts
            self.transform = loss_first_part.transform
            self.learn = loss_first_part.learn
            #self.plus = loss_first_part.plus


    def forward(self, features, targets, batch_index=None):
    #def forward(self, features, targets, last_batch=False):

        if self.type.startswith("sml"):
            #print("ALPHA:", self.alpha)
            # targets_one_hot = F.one_hot(targets).long().cuda()
            targets_one_hot = torch.eye(self.weights.size(0))[targets].long().cuda()

            affines = features.matmul(self.weights.t()) + self.bias

            ################################################################
            distances = utils.euclidean_distances(features, self.weights, 2)
            ################################################################

            logits = affines
            logits_to_training = self.alpha*logits
            logits_to_inference = self.alpha*logits

            ################################################################################################################
            ################################################################################################################
            # executing structural augmentation...
            if self.training and self.split != "no":
                print("augmentation!!!")
                slice_size = features.size(0)//2
                #probabilities_for_training = nn.Softmax(dim=1)(logits_to_training[slice_size:])
                #probabilities_at_targets = probabilities_for_training[range(features.size(0)-slice_size), targets[slice_size:]]
                #loss = -torch.log(probabilities_at_targets).mean()
                loss = self.loss(logits_to_training[slice_size:], targets[slice_size:])
                #################################################################################################################
                #intra_inter_logits = torch.where(targets_one_hot != 0, logits, torch.Tensor([float('Inf')]).cuda())[slice_size:]
                #inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), logits)[slice_size:]
                intra_inter_logits = torch.where(targets_one_hot != 0, distances, torch.Tensor([float('Inf')]).cuda())[slice_size:]
                inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), distances)[slice_size:]
                ###################################################################################################################
            else:
                #probabilities_for_training = nn.Softmax(dim=1)(logits_to_training)
                #probabilities_at_targets = probabilities_for_training[range(features.size(0)), targets]
                #loss = -torch.log(probabilities_at_targets).mean()
                loss = self.loss(logits_to_training, targets)
                ####################################################################################################
                #intra_inter_logits = torch.where(targets_one_hot != 0, logits, torch.Tensor([float('Inf')]).cuda())
                #inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), logits)
                intra_inter_logits = torch.where(targets_one_hot != 0, distances, torch.Tensor([float('Inf')]).cuda())
                inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), distances)
                ######################################################################################################
            ################################################################################################################
            ################################################################################################################

            intra_logits = intra_inter_logits[intra_inter_logits != float('Inf')]
            inter_logits = inter_intra_logits[inter_intra_logits != float('Inf')]
            #affines = features.matmul(self.weights.t()) + self.bias

            #return loss, logits_to_inference, logits, intra_logits, inter_logits
            cls_probabilities = nn.Softmax(dim=1)(logits_to_inference)
            #print(cls_probabilities)
            ood_probabilities = nn.Softmax(dim=1)(logits)
            #print(ood_probabilities)
            return loss, cls_probabilities, ood_probabilities, intra_logits, inter_logits

        elif self.type.startswith("dml"):
            #print("ALPHA:", self.alpha)
            # targets_one_hot = F.one_hot(targets).long().cuda()
            #############################################################################################
            targets_one_hot = torch.eye(self.weights.size(0))[targets].long().cuda()
            #targets_one_hot = torch.eye(self.weights.size(1))[targets].long().cuda()
            #############################################################################################
            #print(targets_one_hot.size())
            #print(targets)
            #print(batch_index)

            ############################
            ############################
            """
            if self.training and (self.initprot == "initprot"):
                if batch_index==1:
                    print("############################################")
                    print("############################################")
                    print("First batch!!! First batch!!! First batch!!!")
                    #print(self.weights)
                    #print(self.vectors_sum)
                    #print(self.vectors_total.unsqueeze(1))
                    self.weights.data = self.vectors_sum/self.vectors_total.unsqueeze(1)
                    del self.vectors_sum
                    del self.vectors_total
                    self.vectors_sum = torch.zeros(self.weights.size(0), self.weights.size(1)).cuda()
                    self.vectors_total = torch.zeros(self.weights.size(0)).cuda()#[0] * out_features
                    #self.vectors_sum.fill_(0)
                    #self.vectors_total.fill_(0)
                    #print(self.weights)
                    print("############################################")
                    print("############################################")

                #for item in range(len(features)):
                #    #self.vectors_sum.index_add_(0, torch.LongTensor([targets[item]]).cuda(), features[item].unsqueeze(0))
                #    self.vectors_total[targets[item]] += 1
                #print(features)
                self.vectors_sum.index_add_(0, targets, features)
                #print(self.vectors_sum.size())
                #torch_ones = torch.ones(features.size(0)).cuda()
                self.vectors_total.index_add_(0, targets, torch.ones(features.size(0)).cuda())
                #print(self.vectors_total.size())
                #del torch_ones
                #print(self.vectors_sum)
                #print(self.vectors_total)
            """
            ############################
            ############################

            if self.distance.startswith("pn"):
                #print("euclidean")
                distances = utils.euclidean_distances(features, self.weights, 2)
            elif self.distance.startswith("ad"):
                """
                weights_norms = self.weights.norm(p=2, dim=1, keepdim=True)
                normalized_weights = self.weights / (weights_norms+0.001)  # this sum may change final results...
                feature_norms = features.norm(p=2, dim=1, keepdim=True)
                normalized_features = features / (feature_norms+0.001)  # this sum may change final results...
                """
                #print("angular")
                #similarities = utils.cosine_similarity(features, self.weights).clamp_(-0.999999, 0.999999)
                similarities = torch.mm(F.normalize(features), torch.transpose(F.normalize(self.weights), 0, 1))
                distances = (torch.acos(similarities) / math.pi)#.clamp_(0.000001, 0.999999)
                #distances = -similarities
            
            #print(self.weights.size())
            #print(self.weights)

            logits = distances
            #if self.initprot == "initprot":
            #    print(torch.isnan(features).any().item())
            #    print(torch.isnan(self.weights).any().item())
            #    print(torch.isnan(distances).any().item())

            """
            if self.training and self.plus:
                print("PLUS!!!")
                target_logits_mean = ((targets_one_hot*logits).sum()/targets_one_hot.size(0)).item()
                #print(target_logits_mean)
                logits_to_training = torch.where(targets_one_hot > 0.5, (self.alpha*logits)+target_logits_mean, self.alpha*logits)
                logits_to_inference = torch.where(targets_one_hot > 0.5, (self.alpha*logits)+target_logits_mean, self.alpha*logits)
            else:
                #print("NO PLUS!!!")
                logits_to_training = self.alpha*logits
                logits_to_inference = self.alpha*logits
            """
            #logits_to_training = self.alpha*logits
            #logits_to_inference = self.alpha*logits
            logits_to_training = self.alpha * self.learn(logits)
            logits_to_inference = self.learn(logits)
            #print(self.scales.item())

            ################################################################################################################
            ################################################################################################################
            # executing structural augmentation...
            if self.training and self.split != "no":
                print("augmentation!!!")
                slice_size = features.size(0)//2
                probabilities_for_training = nn.Softmax(dim=1)(-logits_to_training[slice_size:])
                probabilities_at_targets = probabilities_for_training[range(features.size(0)-slice_size), targets[slice_size:]]
                loss = -torch.log(probabilities_at_targets).mean()
                #loss = self.loss(-logits_to_training[slice_size:], targets[slice_size:])
                """
                if self.initprot != "initprot":
                    probabilities_for_training = nn.Softmax(dim=1)(-logits_to_training[slice_size:])
                    probabilities_at_targets = probabilities_for_training[range(features.size(0)-slice_size), targets[slice_size:]]
                    loss = -torch.log(probabilities_at_targets).mean()
                else:
                    print("initprot")
                    loss = self.loss(-logits_to_training[slice_size:], targets[slice_size:])
                    #print(torch.isnan(loss).any().item())
                """
                #################################################################################################################
                #intra_inter_logits = torch.where(targets_one_hot != 0, logits, torch.Tensor([float('Inf')]).cuda())[slice_size:]
                #inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), logits)[slice_size:]
                intra_inter_logits = torch.where(targets_one_hot != 0, distances, torch.Tensor([float('Inf')]).cuda())[slice_size:]
                inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), distances)[slice_size:]
                ###################################################################################################################
            else:
                probabilities_for_training = nn.Softmax(dim=1)(-logits_to_training)
                probabilities_at_targets = probabilities_for_training[range(features.size(0)), targets]
                loss = -torch.log(probabilities_at_targets).mean()#/1.2
                #print("right and corrected is dividing by the ratio to entropic score equals to ten!!!")
                #loss = self.loss(-logits_to_training, targets)
                """
                if self.initprot != "initprot":
                    probabilities_for_training = nn.Softmax(dim=1)(-logits_to_training)
                    probabilities_at_targets = probabilities_for_training[range(features.size(0)), targets]
                    loss = -torch.log(probabilities_at_targets).mean()
                else:
                    print("initprot")
                    loss = self.loss(-logits_to_training, targets)
                    #print(torch.isnan(loss).any().item())
                """
                ####################################################################################################
                #intra_inter_logits = torch.where(targets_one_hot != 0, logits, torch.Tensor([float('Inf')]).cuda())
                #inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), logits)
                intra_inter_logits = torch.where(targets_one_hot != 0, distances, torch.Tensor([float('Inf')]).cuda())
                inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), distances)
                ######################################################################################################
            ################################################################################################################
            ################################################################################################################

            intra_logits = intra_inter_logits[intra_inter_logits != float('Inf')]
            inter_logits = inter_intra_logits[inter_intra_logits != float('Inf')]

            #return loss, -logits_to_inference, -logits, intra_logits, inter_logits
            cls_probabilities = nn.Softmax(dim=1)(-logits_to_inference)
            #print(cls_probabilities)
            ood_probabilities = nn.Softmax(dim=1)(-logits)
            #print(ood_probabilities)
            return loss, cls_probabilities, ood_probabilities, intra_logits, inter_logits

        elif self.type.startswith("eml"):
            #print("ALPHA:", self.alpha)
            # targets_one_hot = F.one_hot(targets).long().cuda()

            #"""
            if self.training and self.proc_prototypes[1]=="x" and self.model_data_not_initialized and self.distance.startswith("pn"):
                print("model_data_not_initialized!!!") 

                #if self.split != "no":
                #    slice_size = features.size(0)//2
                #    #print("FEATURES:", features[slice_size:])
                #    print("FEATURES MEAN:", features[slice_size:].mean().item())
                #    nn.init.constant_(self.weights, features[slice_size:].mean().item())
                #else:
                #    #print("FEATURES:", features)
                #    print("FEATURES MEAN:", features.mean().item())
                #    nn.init.constant_(self.weights, features.mean().item())

                #print("FEATURES:", features)
                print()
                print("FEATURES MEAN:", features.mean().item())
                print("FEATURES STD:", features.std().item())
                #nn.init.normal_(self.weights, mean=0, std=features.std().item())
                nn.init.normal_(self.weights, mean=features.mean().item(), std=0.1)
                #nn.init.constant_(self.weights, features.mean().item())
                #print("WEIGHTS:", self.weights)                   
                print("WEIGHTS MEAN:", self.weights.mean().item())
                print("WEIGHTS STD:", self.weights.std().item())
                print()
                self.model_data_not_initialized = False
            #"""

            """
            if self.training and (self.batches_processed < 100) and self.distance.startswith("mm"):
                self.learn = lambda x: x
                self.batches_processed += 1
            else:
                self.learn = lambda x: self.scales * x
            """

            #########################################################################
            #########################################################################
            #targets_one_hot = torch.eye(self.weights.size(0))[targets].long().cuda()
            targets_one_hot = torch.eye(self.weights.size(1))[targets].long().cuda()
            #########################################################################
            #########################################################################

            if self.distance.startswith("pn"):
                #print("euclidean") 
                distances = utils.euclidean_distances(features, self.weights, 2)#.permute(0, 2, 1)
            elif self.distance.startswith("ad"):
                """
                weights_norms = self.weights.norm(p=2, dim=1, keepdim=True)
                normalized_weights = self.weights / (weights_norms+0.001)  # this sum may change final results...
                feature_norms = features.norm(p=2, dim=1, keepdim=True)
                normalized_features = features / (feature_norms+0.001)  # this sum may change final results...
                """
                print("angular")
                #similarities = utils.cosine_similarity(features, self.weights).clamp_(-0.999999, 0.999999)
                similarities = torch.mm(F.normalize(features), torch.transpose(F.normalize(self.weights), 0, 1))
                distances = (torch.acos(similarities) / math.pi)#.clamp_(0.000001, 0.999999)
                #distances = -similarities
            """
            elif self.distance.startswith("mm"):
                print("matrix multiplication")
                print(features.transpose(1, 0).size())
                print(self.weights.size())
                similarities = torch.matmul(self.weights, features.transpose(1, 0)).permute(2, 0, 1)
                print(similarities.size())
                distances = -similarities
            """

            logits = self.transform(distances)
            #logits_to_training = self.learn(logits)
            #logits_to_inference = self.learn(logits)
            logits_to_training = self.alpha * self.learn(logits)
            logits_to_inference = self.learn(logits)
            """
            print("#############################################")
            print("#############################################")
            print(self.weights.size())
            print(self.weights)
            #print(distances.size())
            #print(self.alpha)
            #print(logits.size())
            print(self.scales.size())
            print(self.scales)
            #print(logits_to_training.size())
            #print(logits_to_inference.size())
            print("#############################################")
            print("#############################################")
            """

            ################################################################################################################
            ################################################################################################################
            # executing structural augmentation...
            if self.training and self.split != "no":
                print("augmentation!!!")
                slice_size = features.size(0)//2
                probabilities_for_training = nn.Softmax(dim=1)(-logits_to_training[slice_size:])
                probabilities_at_targets = probabilities_for_training[range(features.size(0)-slice_size), targets[slice_size:]]
                loss = -torch.log(probabilities_at_targets).mean()
                #loss = self.loss(-logits_to_training[slice_size:], targets[slice_size:])
                #################################################################################################################
                #intra_inter_logits = torch.where(targets_one_hot != 0, logits, torch.Tensor([float('Inf')]).cuda())[slice_size:]
                #inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), logits)[slice_size:]
                intra_inter_logits = torch.where(targets_one_hot != 0, distances, torch.Tensor([float('Inf')]).cuda())[slice_size:]
                inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), distances)[slice_size:]
                ###################################################################################################################
            else:
                #print("NOT growding!!!")
                #probabilities_for_training = nn.Softmax(dim=1)(-logits_to_training)
                probabilities_for_training = nn.Softmax(dim=2)(-logits_to_training)
                #print("probabilities_for_training size:", probabilities_for_training.size())
                #print("probabilities_for_training:\n", probabilities_for_training)
                #probabilities_at_targets = probabilities_for_training[range(features.size(0)), targets]
                probabilities_at_targets = probabilities_for_training.permute(0, 2, 1)[range(features.size(0)), targets]
                #print("probabilities_at_targets size:", probabilities_at_targets.size())
                #print("probabilities_at_targets:\n", probabilities_at_targets)
                ############################################$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                ############################################$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                #loss_by_example = -torch.log(probabilities_at_targets.mean(dim=1)) # x
                #loss_by_example = -torch.log(probabilities_at_targets).mean(dim=1) # xx
                ########################################################################################
                loss_by_example = -torch.log(probabilities_at_targets).sum(dim=1) # sx
                ########################################################################################
                #loss_by_example = -torch.log(probabilities_at_targets.sum(dim=1)) # sxx
                ############################################$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                ############################################$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                #print("loss_by_example size:", loss_by_example.size())
                #print("loss_by_example:\n", loss_by_example)
                loss = loss_by_example.mean()
                #loss = self.prototypes_per_class * loss_by_example.mean()
                #loss = self.loss(-logits_to_training, targets)
                ####################################################################################################
                #intra_inter_logits = torch.where(targets_one_hot != 0, distances, torch.Tensor([float('Inf')]).cuda())
                #inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), distances)
                intra_inter_logits = torch.where(targets_one_hot != 0, distances.mean(dim=1), torch.Tensor([float('Inf')]).cuda())
                inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), distances.mean(dim=1))
                ######################################################################################################
            ################################################################################################################
            ################################################################################################################

            intra_logits = intra_inter_logits[intra_inter_logits != float('Inf')]
            inter_logits = inter_intra_logits[inter_intra_logits != float('Inf')]

            #return loss, -logits_to_inference, -logits, intra_logits, inter_logits
            cls_probabilities = nn.Softmax(dim=2)(-logits_to_inference).mean(dim=1)
            #print("cls_probabilities size:", cls_probabilities.size())
            #print("cls_probabilities:\n", cls_probabilities)
            ood_probabilities = nn.Softmax(dim=2)(-logits).mean(dim=1)
            #print("ood_probabilities size:", ood_probabilities.size())
            #print("ood_probabilities:\n", ood_probabilities)
            return loss, cls_probabilities, ood_probabilities, intra_logits, inter_logits

        elif self.type.startswith("cml"):
            # CONTRASTIVE LEARNING INSPIRED ISOMAX!!! COMBINES ISOMAX PROTOTYPES AND CONTRASTIVE LEARNING!!! MAY BE USED TO UNSPUER/SELF/SEMI!!!
            # CONTRASTIVE LEARNING INSPIRED ISOMAX!!! COMBINES ISOMAX PROTOTYPES AND CONTRASTIVE LEARNING!!! MAY BE USED TO UNSPUER/SELF/SEMI!!!
            # CONTRASTIVE LEARNING INSPIRED ISOMAX!!! COMBINES ISOMAX PROTOTYPES AND CONTRASTIVE LEARNING!!! MAY BE USED TO UNSPUER/SELF/SEMI!!!
            # CONTRASTIVE LEARNING INSPIRED ISOMAX!!! COMBINES ISOMAX PROTOTYPES AND CONTRASTIVE LEARNING!!! MAY BE USED TO UNSPUER/SELF/SEMI!!!
            #print("ALPHA:", self.alpha)
            # targets_one_hot = F.one_hot(targets).long().cuda()
            targets_one_hot = torch.eye(self.weights.size(0))[targets].long().cuda()

            if self.distance.startswith("pn"):
                #print("euclidean") 
                distances = utils.euclidean_distances(features, self.weights, 2)
            elif self.distance.startswith("ad"):
                """
                weights_norms = self.weights.norm(p=2, dim=1, keepdim=True)
                normalized_weights = self.weights / (weights_norms+0.001)  # this sum may change final results...
                feature_norms = features.norm(p=2, dim=1, keepdim=True)
                normalized_features = features / (feature_norms+0.001)  # this sum may change final results...
                """
                print("angular")
                #similarities = utils.cosine_similarity(features, self.weights).clamp_(-0.999999, 0.999999)
                similarities = torch.mm(F.normalize(features), torch.transpose(F.normalize(self.weights), 0, 1))
                distances = (torch.acos(similarities) / math.pi)#.clamp_(0.000001, 0.999999)
                #distances = -similarities

            logits = distances
            if self.training:
                pairwise_distaces = utils.euclidean_distances(features, features, 2)
                pairwise_distaces = torch.where(torch.eye(pairwise_distaces.size(0)).bool().cuda(), torch.Tensor([0]).cuda(), pairwise_distaces)
                logits_contrastive_term = torch.Tensor(pairwise_distaces.size(0), self.weights.size(0)).cuda()
                for i in range(self.weights.size(0)):
                    logits_contrastive_term[:, i] = torch.sum(pairwise_distaces[:, targets==i], dim=1)/pairwise_distaces.size(0)
                    #logits_contrastive_term[:, i] = torch.mean(pairwise_distaces[:, targets==i], dim=1) # this also work in log type 2!!!
                logits_to_training = self.learn(self.transform(logits + logits_contrastive_term))
                #### the logits_contrastive_term deveria estar fora da SoftMax Funxtion???
                #### the logits_contrastive_term deveria estar fora da SoftMax Funxtion???
                #### the logits_contrastive_term deveria estar fora da SoftMax Funxtion???
                #### the logits_contrastive_term deveria estar fora da SoftMax Funxtion???
                logits_to_inference = self.learn(self.transform(logits))
                """
                if self.plus:
                    #print("PLUS+CML!!!")
                    target_logits_mean = ((targets_one_hot*logits).sum()/targets_one_hot.size(0)).item()
                    #print(target_logits_mean)
                    logits_to_training = torch.where(targets_one_hot > 0.5, logits_to_training+target_logits_mean, logits_to_training)
                    logits_to_inference = torch.where(targets_one_hot > 0.5, logits_to_inference+target_logits_mean, logits_to_inference)
                """

                """
                equal_targets = torch.eq(targets.expand(
                    [features.size(0),features.size(0)]), targets.expand([features.size(0),features.size(0)]).t()).fill_diagonal_(False)
                #equal_targets_full = torch.eq(targets.expand(
                #    [features.size(0),features.size(0)]), targets.expand([features.size(0),features.size(0)]).t())
                #print(equal_targets)

                equal_paiwise_distances = torch.where(equal_targets, pairwise_distaces, torch.Tensor([0]).cuda())
                #not_equal_paiwise_distances = torch.where(equal_targets_full, torch.Tensor([0]).cuda(), pairwise_distaces)
                #print(equal_paiwise_distances.size(0))

                equal_paiwise_distances_sum_normalized = (equal_paiwise_distances.sum(dim=1)/(torch.sum(equal_targets==True, dim=1)+1)).unsqueeze(1)
                #equal_paiwise_distances_sum_normalized = (equal_paiwise_distances.sum(dim=1)/equal_paiwise_distances.size(0)).unsqueeze(1)
                #not_equal_paiwise_distances_sum_normalized = (not_equal_paiwise_distances.sum(dim=1)/(torch.sum(equal_targets_full==False, dim=1))).unsqueeze(1)

                logits_contrastive_term = targets_one_hot * equal_paiwise_distances_sum_normalized
                #contrastive_enhanced_logits = logits + targets_one_hot * equal_paiwise_distances_sum_normalized
                #"""
            else:
                logits_to_training = self.learn(self.transform(logits))
                logits_to_inference = self.learn(self.transform(logits))

            ################################################################################################################
            ################################################################################################################
            # executing structural augmentation...
            if self.training and self.split != "no":
                print("augmentation!!!")
                slice_size = features.size(0)//2
                probabilities_for_training = nn.Softmax(dim=1)(-logits_to_training[slice_size:])
                probabilities_at_targets = probabilities_for_training[range(features.size(0)-slice_size), targets[slice_size:]]
                loss = -torch.log(probabilities_at_targets).mean()
                #loss = self.loss(-logits_to_training[slice_size:], targets[slice_size:])
                #################################################################################################################
                #intra_inter_logits = torch.where(targets_one_hot != 0, logits, torch.Tensor([float('Inf')]).cuda())[slice_size:]
                #inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), logits)[slice_size:]
                intra_inter_logits = torch.where(targets_one_hot != 0, distances, torch.Tensor([float('Inf')]).cuda())[slice_size:]
                inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), distances)[slice_size:]
                ###################################################################################################################
            else:
                probabilities_for_training = nn.Softmax(dim=1)(-logits_to_training)
                probabilities_at_targets = probabilities_for_training[range(features.size(0)), targets]
                loss = -torch.log(probabilities_at_targets).mean()
                #loss = self.loss(-logits_to_training, targets)
                ####################################################################################################
                #intra_inter_logits = torch.where(targets_one_hot != 0, logits, torch.Tensor([float('Inf')]).cuda())
                #inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), logits)
                intra_inter_logits = torch.where(targets_one_hot != 0, distances, torch.Tensor([float('Inf')]).cuda())
                inter_intra_logits = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), distances)
                ######################################################################################################
            ################################################################################################################
            ################################################################################################################

            intra_logits = intra_inter_logits[intra_inter_logits != float('Inf')]
            inter_logits = inter_intra_logits[inter_intra_logits != float('Inf')]

            #return loss, -logits_to_inference, -logits, intra_logits, inter_logits
            return loss, -logits_to_inference, -logits, intra_logits, inter_logits


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

#################################################################
#if self.training and (not self.type.split("_")[4].startswith("no")):
#    noise = float(self.type.split("_")[4])
#    #features = torch.add(features, noise, torch.randn(features.size()).cuda())
#    features = torch.add(features, noise, features * torch.randn(features.size()).cuda())
#    #features = torch.add(features, noise, torch.norm(features, p=2, dim=1, keepdim=True) * torch.randn(features.size()).cuda())
#    print("Noise!!!")
#################################################################

"""
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
    geometric_regularization = gamma * first_part + gamma * second_part
    print("geometric_regularization")
else:
    #print("no geometric_regularization")
    geometric_regularization = 0
"""

"""
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
    geometric_regularization = gamma*first_part + gamma*second_part
    print("geometric_regularization!!!")
else:
    #print("no geometric_regularization")
    geometric_regularization = 0
"""

"""
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
    geometric_regularization = gamma*first_part + gamma*second_part
    print("geometric_regularization!!!")
else:
    #print("no geometric_regularization")
    geometric_regularization = 0
#### WE CAN ALSO USE ENTROPY FOR GEOMETRIC regularization!!!
#geometric_regularization = utils.entropies_from_probabilities(nn.Softmax(dim=1)(-alpha*logits)).mean()            
"""

"""
elif self.type.startswith("kml"):
    #print("ALPHA:", self.alpha)
    # targets_one_hot = F.one_hot(targets).long().cuda()
    targets_one_hot = torch.eye(self.weights.size(0))[targets].long().cuda()

    if self.distance.startswith("pn"):
        #print("euclidean")
        distances = utils.euclidean_distances(features, self.weights, 2)
    elif self.distance.startswith("ad"):
        #weights_norms = self.weights.norm(p=2, dim=1, keepdim=True)
        #normalized_weights = self.weights / (weights_norms+0.001)  # this sum may change final results...
        #feature_norms = features.norm(p=2, dim=1, keepdim=True)
        #normalized_features = features / (feature_norms+0.001)  # this sum may change final results...
        print("angular")
        similarities = utils.cosine_similarity(features, self.weights).clamp_(-0.999999, 0.999999)
        distances = (torch.acos(similarities) / math.pi).clamp_(0.000001, 0.999999)
        #distances = (torch.acos(similarities) / math.pi).clamp_(0.000001, 0.999999) + self.bias

    logits = self.transform(distances)
    learned_logits = self.learn(logits)
    #print("\nLEARNED LOGITS:\n", learned_logits, "\n")

    ################################################################################################################
    ################################################################################################################

    #print(targets_one_hot)
    p = self.probability
    #print(p)
    target_distrbutions = torch.where(targets_one_hot != 0, torch.tensor(p).cuda(), torch.tensor((1-p)/(targets_one_hot.size(1)-1)).cuda())
    #print(target_distrbutions)

    # executing structural augmentation...
    if self.training and self.type.split("_")[4] != "no":
        #slice_size = features.size(0)//(2**int(self.type.split("_")[4][0]))
        slice_size = features.size(0)/@ln

        #uniform_dist = torch.Tensor(slice_size, self.args.number_of_model_classes).fill_((1./self.args.number_of_model_classes)).cuda()
        #kl_divergence = F.kl_div(F.log_softmax(odd_outputs[:slice_size], dim=1), uniform_dist, reduction='batchmean')
        #augmentation = augmentation_multiplier * kl_divergence

        #probabilities_for_training = nn.Softmax(dim=1)(-learned_logits[slice_size:])
        #probabilities_at_targets = probabilities_for_training[range(features.size(0)-slice_size), targets[slice_size:]]
        #loss = -torch.log(probabilities_at_targets).mean()
        #loss = self.loss(-learned_logits[slice_size:], targets[slice_size:])
        loss = F.kl_div(F.log_softmax(-learned_logits[slice_size:], dim=1), target_distrbutions[slice_size:], reduction='batchmean')
        intra_inter_distances = torch.where(targets_one_hot != 0, -distances, distances)[slice_size:]
    else:

        #uniform_dist = torch.Tensor(slice_size, self.args.number_of_model_classes).fill_((1./self.args.number_of_model_classes)).cuda()
        #kl_divergence = F.kl_div(F.log_softmax(odd_outputs[:slice_size], dim=1), uniform_dist, reduction='batchmean')
        #augmentation = augmentation_multiplier * kl_divergence

        #probabilities_for_training = nn.Softmax(dim=1)(-learned_logits)
        #probabilities_at_targets = probabilities_for_training[range(features.size(0)), targets]
        #loss = -torch.log(probabilities_at_targets).mean()
        #loss = self.loss(-learned_logits, targets)
        loss = F.kl_div(F.log_softmax(-learned_logits, dim=1), target_distrbutions, reduction='batchmean')
        intra_inter_distances = torch.where(targets_one_hot != 0, -distances, distances)
    ################################################################################################################
    ################################################################################################################

    intra_distances = -intra_inter_distances[intra_inter_distances <= 0]
    inter_distances = intra_inter_distances[intra_inter_distances > 0]

    #return loss, -learned_logits, -logits, intra_distances.flatten().tolist(), inter_distances.flatten().tolist()
    #return loss, -learned_logits, -logits, intra_distances.tolist(), inter_distances.tolist()
    return loss, -learned_logits, -logits, intra_distances, inter_distances
"""

"""
else:
    mean_temperature = self.scales.mean().item()
    mean_logits = logits.mean(dim=1).unsqueeze(1)
    #mean_temperature_logits = self.learn(logits)/mean_temperature # should remove tilts before normalization???
    mean_temperature_logits = (self.learn(logits)-self.tilts)/mean_temperature # now with remove tilts!!!
    normalized_logits = self.learn(logits)/self.learn(logits).mean(dim=1).unsqueeze(1)
    normalized_learn_logits = self.learn(logits)/mean_logits
    ###############################################################
    ###############################################################
    #print("#####################################################")
    #print("MEAN TEMPERATURE:", mean_temperature)
    ##print(mean_temperature_logits)
    #print("MEAN TEMPERATURE LOGITS [MEAN]:", mean_temperature_logits.mean().item())
    ##print(normalized_logits)
    #print("NORMALIZED LOGITS [MEAN]:", normalized_logits.mean().item())
    if self.inference_learn == 'NORMLR':
        inference_logits = normalized_logits
    elif self.inference_learn == 'NORMLR1X':
        inference_logits = mean_temperature*normalized_logits
    elif self.inference_learn == 'NORMLR2X':
        inference_logits = mean_logits*normalized_logits
    elif self.inference_learn == 'NORMLR3X':
        inference_logits = mean_logits*mean_temperature*normalized_logits
    elif self.inference_learn == 'NORMLR4X':
        inference_logits = normalized_learn_logits
    elif self.inference_learn == 'MEANTEMPLR':
        inference_logits = mean_temperature_logits
    elif self.inference_learn == 'SQRTLR':
        inference_logits = math.sqrt(mean_temperature)*mean_temperature_logits
    elif self.inference_learn == 'LOGLR':
        inference_logits = math.log1p(mean_temperature)*mean_temperature_logits
    elif self.inference_learn == 'LOGMEANLR':
        inference_logits = ((mean_temperature-1)/math.log(mean_temperature))*mean_temperature_logits
    elif self.inference_learn == 'HARMMEANLR':
        inference_logits = (2/(1+(1/mean_temperature)))*mean_temperature_logits
    elif self.inference_learn == 'ARITHMEANLR':
        inference_logits = ((mean_temperature+1)/2)*mean_temperature_logits
    elif self.inference_learn == '1/SQRTLR':
        inference_logits = mean_temperature_logits/math.sqrt(mean_temperature)
    elif self.inference_learn == '1/LOGLR':
        inference_logits = mean_temperature_logits/math.log1p(mean_temperature)
    elif self.inference_learn == '1/LOGMEANLR':
        inference_logits = mean_temperature_logits/((mean_temperature-1)/math.log(mean_temperature))
    elif self.inference_learn == '1/HARMMEANLR':
        inference_logits = mean_temperature_logits/(2/(1+(1/mean_temperature)))
    elif self.inference_learn == '1/ARITHMEANLR':
        inference_logits = mean_temperature_logits/((mean_temperature+1)/2)
#print("INFERENCE LOGITS [MEAN]:", inference_logits.mean().item())
"""

"""
elif self.proc_prototypes[1] == "n":
    ########self.proc_prototypes_std = float(self.proc_prototypes[2])
    nn.init.normal_(self.weights, mean=0.0, std=float(self.proc_prototypes[2]))
    print("\ninit normal for prototypes:", float(self.proc_prototypes[2]))
elif self.proc_prototypes[1] == "x":
    ########self.proc_prototypes_option = self.proc_prototypes[2]
    print("\nadvanced init for prototypes [option]:", self.proc_prototypes[2])
    nn.init.normal_(self.weights, mean=0.0, std=1.0)
    ########self.weights = torch.nn.Parameter((self.weights-self.weights.mean(dim=0).unsqueeze(0))/self.weights.std(dim=0).unsqueeze(0))
    print("verify pre-normalization:", torch.norm(self.weights, dim=1))
    #print(utils.euclidean_distances(self.weights, self.weights, 2))
    if self.proc_prototypes[2] == "1":
        self.weights.data = self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
    elif self.proc_prototypes[2] == "2":
        self.weights.data = torch.norm(self.weights, dim=1).mean().item()*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
    if self.proc_prototypes[2] == "3":
        self.weights.data = 0.1*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
    elif self.proc_prototypes[2] == "4":
        self.weights.data = 2*torch.norm(self.weights, dim=1).mean().item()*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
    print("verify post-normalization:", torch.norm(self.weights, dim=1))
    #print(utils.euclidean_distances(self.weights, self.weights, 2))
"""

"""
if self.proc_prototypes[1] == "x":
    if self.proc_prototypes[2] == "a":
        #print("a")
        norm_weights = self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "b":
        #print("b")
        norm_weights = torch.norm(self.weights, dim=1).mean().item()*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, norm_weights, 2)
    if self.proc_prototypes[2] == "c":
        #print("c")
        norm_weights = 0.1*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "d":
        #print("d")
        #norm_weights = self.weights/torch.norm(self.weights, dim=1).unsqueeze(1).detach()
        norm_weights = self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "e":
        #print("e")
        #norm_weights = torch.norm(self.weights, dim=1).mean().item()*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1).detach()
        norm_weights = torch.norm(self.weights, dim=1).mean().item()*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "f":
        #print("f")
        #norm_weights = self.weights/torch.norm(self.weights, dim=1).unsqueeze(1).detach()
        norm_weights = 0.1*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, norm_weights, 2)
    #elif self.proc_prototypes[2] == "g":
    #    #print("g")
    #    self.weights.data = self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
    #    distances = utils.euclidean_distances(features, self.weights, 2)
    #elif self.proc_prototypes[2] == "h":
    #    #print("h")
    #    self.weights.data = torch.norm(self.weights, dim=1).mean().item()*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
    #    distances = utils.euclidean_distances(features, self.weights, 2)
    #elif self.proc_prototypes[2] == "i":
    #    #print("i")
    #    self.weights.data = 0.1*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
    #    distances = utils.euclidean_distances(features, self.weights, 2)
    else:
        #print("not special x!!!")
        distances = utils.euclidean_distances(features, self.weights, 2)
else:
    #print("not even x!!!")
    distances = utils.euclidean_distances(features, self.weights, 2)
"""

"""
if self.proc_prototypes[1] == "x":
    if self.proc_prototypes[2] == "a":
        #print("a")
        norm_weights = self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "b":
        #print("b")
        norm_weights = torch.norm(self.weights, dim=1).mean().item()*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, norm_weights, 2)
    if self.proc_prototypes[2] == "c":
        #print("c")
        norm_weights = 0.1*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "d":
        #print("d")
        #norm_weights = self.weights/torch.norm(self.weights, dim=1).unsqueeze(1).detach()
        norm_weights = self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "e":
        #print("e")
        #norm_weights = torch.norm(self.weights, dim=1).mean().item()*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1).detach()
        norm_weights = torch.norm(self.weights, dim=1).mean().item()*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "f":
        #print("f")
        #norm_weights = self.weights/torch.norm(self.weights, dim=1).unsqueeze(1).detach()
        norm_weights = 0.1*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, norm_weights, 2)
    #elif self.proc_prototypes[2] == "g":
    #    #print("g")
    #    self.weights.data = self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
    #    distances = utils.euclidean_distances(features, self.weights, 2)
    #elif self.proc_prototypes[2] == "h":
    #    #print("h")
    #    self.weights.data = torch.norm(self.weights, dim=1).mean().item()*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
    #    distances = utils.euclidean_distances(features, self.weights, 2)
    #elif self.proc_prototypes[2] == "i":
    #    #print("i")
    #    self.weights.data = 0.1*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
    #    distances = utils.euclidean_distances(features, self.weights, 2)
    else:
        #print("not special x!!!")
        distances = utils.euclidean_distances(features, self.weights, 2)
else:
    #print("not even x!!!")
    distances = utils.euclidean_distances(features, self.weights, 2)
"""

"""
else:
    mean_temperature = self.scales.mean().item()
    mean_logits = logits.mean(dim=1).unsqueeze(1)
    #mean_temperature_logits = self.learn(logits)/mean_temperature # should remove tilts before normalization???
    mean_temperature_logits = (self.learn(logits)-self.tilts)/mean_temperature # now with remove tilts!!!
    normalized_logits = self.learn(logits)/self.learn(logits).mean(dim=1).unsqueeze(1)
    normalized_learn_logits = self.learn(logits)/mean_logits
    ###############################################################
    ###############################################################
    #print("#####################################################")
    #print("MEAN TEMPERATURE:", mean_temperature)
    ##print(mean_temperature_logits)
    #print("MEAN TEMPERATURE LOGITS [MEAN]:", mean_temperature_logits.mean().item())
    ##print(normalized_logits)
    #print("NORMALIZED LOGITS [MEAN]:", normalized_logits.mean().item())
    if self.inference_learn == 'NORMLR':
        inference_logits = normalized_logits
    elif self.inference_learn == 'NORMLR1X':
        inference_logits = mean_temperature*normalized_logits
    elif self.inference_learn == 'NORMLR2X':
        inference_logits = mean_logits*normalized_logits
    elif self.inference_learn == 'NORMLR3X':
        inference_logits = mean_logits*mean_temperature*normalized_logits
    elif self.inference_learn == 'NORMLR4X':
        inference_logits = normalized_learn_logits
    elif self.inference_learn == 'MEANTEMPLR':
        inference_logits = mean_temperature_logits
    elif self.inference_learn == 'SQRTLR':
        inference_logits = math.sqrt(mean_temperature)*mean_temperature_logits
    elif self.inference_learn == 'LOGLR':
        inference_logits = math.log1p(mean_temperature)*mean_temperature_logits
    elif self.inference_learn == 'LOGMEANLR':
        inference_logits = ((mean_temperature-1)/math.log(mean_temperature))*mean_temperature_logits
    elif self.inference_learn == 'HARMMEANLR':
        inference_logits = (2/(1+(1/mean_temperature)))*mean_temperature_logits
    elif self.inference_learn == 'ARITHMEANLR':
        inference_logits = ((mean_temperature+1)/2)*mean_temperature_logits
    elif self.inference_learn == '1/SQRTLR':
        inference_logits = mean_temperature_logits/math.sqrt(mean_temperature)
    elif self.inference_learn == '1/LOGLR':
        inference_logits = mean_temperature_logits/math.log1p(mean_temperature)
    elif self.inference_learn == '1/LOGMEANLR':
        inference_logits = mean_temperature_logits/((mean_temperature-1)/math.log(mean_temperature))
    elif self.inference_learn == '1/HARMMEANLR':
        inference_logits = mean_temperature_logits/(2/(1+(1/mean_temperature)))
    elif self.inference_learn == '1/ARITHMEANLR':
        inference_logits = mean_temperature_logits/((mean_temperature+1)/2)
#print("INFERENCE LOGITS [MEAN]:", inference_logits.mean().item())
"""

"""               
if self.proc_prototypes[1] == "x":
    if self.proc_prototypes[2] == "a":
        #print("a")
        norm_weights = self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "b":
        #print("b")
        norm_weights = torch.norm(self.weights, dim=1).mean().item()*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "c":
        #print("c")
        norm_weights = 0.1*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "d":
        #print("d")
        norm_weights = self.weights/torch.norm(self.weights, dim=1).unsqueeze(1).detach()
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "e":
        #print("e")
        norm_weights = torch.norm(self.weights, dim=1).mean().item()*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1).detach()
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "f":
        #print("f")
        norm_weights = 0.1*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1).detach()
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "g":
        #print("g")
        self.weights.data = self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, self.weights, 2)
    elif self.proc_prototypes[2] == "h":
        #print("h")
        self.weights.data = torch.norm(self.weights, dim=1).mean().item()*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, self.weights, 2)
    elif self.proc_prototypes[2] == "i":
        #print("i")
        self.weights.data = 0.1*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, self.weights, 2)
    else:
        #print("not special x!!!")
        distances = utils.euclidean_distances(features, self.weights, 2)
else:
    #print("not even x!!!")
    distances = utils.euclidean_distances(features, self.weights, 2)
"""

"""                
if self.proc_prototypes[1] == "x":
    if self.proc_prototypes[2] == "a":
        #print("a")
        norm_weights = self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "b":
        #print("b")
        norm_weights = torch.norm(self.weights, dim=1).mean().item()*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "c":
        #print("c")
        norm_weights = 0.1*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "d":
        #print("d")
        norm_weights = self.weights/torch.norm(self.weights, dim=1).unsqueeze(1).detach()
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "e":
        #print("e")
        norm_weights = torch.norm(self.weights, dim=1).mean().item()*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1).detach()
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "f":
        #print("f")
        norm_weights = 0.1*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1).detach()
        distances = utils.euclidean_distances(features, norm_weights, 2)
    elif self.proc_prototypes[2] == "g":
        #print("g")
        self.weights.data = self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, self.weights, 2)
    elif self.proc_prototypes[2] == "h":
        #print("h")
        self.weights.data = torch.norm(self.weights, dim=1).mean().item()*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, self.weights, 2)
    elif self.proc_prototypes[2] == "i":
        #print("i")
        self.weights.data = 0.1*self.weights/torch.norm(self.weights, dim=1).unsqueeze(1)
        distances = utils.euclidean_distances(features, self.weights, 2)
    else:
        #print("not special x!!!")
        distances = utils.euclidean_distances(features, self.weights, 2)
else:
    #print("not even x!!!")
    distances = utils.euclidean_distances(features, self.weights, 2)
"""
