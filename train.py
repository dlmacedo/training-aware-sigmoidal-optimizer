import sys
import argparse
import os
import random
import numpy
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import agents
import utils


#torch.set_default_tensor_type(torch.DoubleTensor)
#torch.set_default_dtype(torch.float64)

cudnn.benchmark = False
cudnn.deterministic = True

numpy.set_printoptions(edgeitems=5, linewidth=160, formatter={'float': '{:0.6f}'.format})
torch.set_printoptions(edgeitems=5, precision=6, linewidth=160)
pd.options.display.float_format = '{:,.6f}'.format
pd.set_option('display.width', 160)

parser = argparse.ArgumentParser(description='Train')

parser.add_argument('-x', '--executions', default=1, type=int, metavar='N', help='Number of executions')
parser.add_argument('-sx', '--start-executions', default=1, type=int, metavar='N', help='Number of the start execution')
parser.add_argument('-w', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('-bs', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size')
#parser.add_argument('-e', '--epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
#parser.add_argument('-lr', '--original-learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
##parser.add_argument('-lrdr', '--learning-rate-decay-rate', default=0.1, type=float, metavar='LRDR', help='learning rate decay rate')
##parser.add_argument('-lrde', '--learning-rate-decay-epochs', default="150 200 250", metavar='LRDE', help='learning rate decay epochs')
##parser.add_argument('-lrdp', '--learning-rate-decay-period', default=500, type=int, metavar='LRDP', help='learning rate decay period')
#parser.add_argument('-mm', '--momentum', default=0.9, type=float, metavar='M', help='momentum')
#parser.add_argument('-wd', '--weight-decay', default=1*1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('-pf', '--print-freq', default=1, type=int, metavar='N', help='print frequency')
parser.add_argument('-gpu', '--gpu-id', default='0', type=int, help='id for CUDA_VISIBLE_DEVICES')
parser.add_argument('-ei', '--exps-inputs', default="", type=str, metavar='PATHS', help='Inputs paths for the experiments')
parser.add_argument('-et', '--exps-types', default="", type=str, metavar='EXPERIMENTS', help='Experiments types to be performed')
parser.add_argument('-ec', '--exps-configs', default="", type=str, metavar='CONFIGS', help='Experiments configs to be used')
##parser.add_argument('-base', '--base-seed', default="10000", type=int, metavar='CONFIGS', help='Base seed to be used')


args = parser.parse_args()
args.exps_inputs = args.exps_inputs.split(":")
args.exps_types = args.exps_types.split(":")
args.exps_configs = args.exps_configs.split(":")
#args.learning_rate_decay_epochs = sorted([int(item) for item in args.learning_rate_decay_epochs.split()])
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
#torch.cuda.device(args.gpu_id)
torch.cuda.set_device(args.gpu_id)

#######################################################
print('\n__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION')
#from subprocess import call
# call(["nvcc", "--version"]) does not work
#! nvcc --version
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
#print('__Devices')
#call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print('Active CUDA Device: GPU', torch.cuda.current_device())
#print('Available devices ', torch.cuda.device_count())
#print('Current cuda device ', torch.cuda.current_device())
#use_cuda = torch.cuda.is_available()
#FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
#LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
#Tensor = FloatTensor
#######################################################


def main():

    print("\n\n\n\n\n\n")
    print("***************************************************************")
    print("***************************************************************")
    print("***************************************************************")
    print("***************************************************************")

    all_experiment_results = {}

    for args.exp_input in args.exps_inputs:
        for args.exp_type in args.exps_types:
            for args.exp_config in args.exps_configs:

                print("\n\n\n\n")
                print("***************************************************************")
                args.base_seed = 1000000 ###### <<<<<===== x1 = 100001
                args.number_of_first_partition_examples_per_class = 1000000000
                args.number_of_second_partition_examples_per_class = 0
                args.number_of_model_classes = None
                args.partition = "1"

                print("EXPERIMENT INPUT:", args.exp_input.upper())
                print("EXPERIMENT TYPE:", args.exp_type.upper())
                print("EXPERIMENT CONFIG:", args.exp_config.upper())

                args.experiment_path = os.path.join("expers", args.exp_input, args.exp_type, args.exp_config)

                if not os.path.exists(args.experiment_path):
                    os.makedirs(args.experiment_path)
                print("EXPERIMENT PATH:", args.experiment_path.upper())

                args.executions_best_results_file_path = os.path.join(args.experiment_path, "results_best.csv")
                args.executions_raw_results_file_path = os.path.join(args.experiment_path, "results_raw.csv")

                for config in args.exp_config.split("+"):
                    config = config.split("~")
                    if config[0] == "data":
                        args.dataset_full = str(config[1])
                        print("DATASET FULL:", args.dataset_full.upper())
                        if "_" in str(config[1]):
                            args.number_of_first_partition_examples_per_class = int(str(config[1]).split("_")[1])
                            print("NUMBER OF FIRST PARTITION EXAMPLES PER CLASS:", args.number_of_first_partition_examples_per_class)
                            args.dataset = str(config[1]).split("_")[0]
                            print("DATASET:", args.dataset.upper())
                        else:
                            args.dataset = str(config[1])
                            print("DATASET:", args.dataset.upper())
                    elif config[0] == "fpec":
                        args.number_of_first_partition_examples_per_class = int(config[1])
                        print("NUMBER OF FIRST PARTITION EXAMPLES PER CLASS:", args.number_of_first_partition_examples_per_class)
                    elif config[0] == "spec":
                        args.number_of_second_partition_examples_per_class = int(config[1])
                        print("NUMBER OF SECOND PARTITION EXAMPLES PER CLASS:", args.number_of_second_partition_examples_per_class)
                    elif config[0] == "part":
                        args.partition = str(config[1])
                        print("PARTITION TO BE USED ['1' OR '2']:", args.partition.upper())
                    elif config[0] == "model":
                        args.model_name = str(config[1])
                        print("MODEL:", args.model_name.upper())
                        """
                        if args.model_name == "textcnn":
                            args.max_sen_len = 30
                            #args.max_sen_len = None
                        elif args.model_name == "textcnn1d":
                            #args.max_sen_len = 30
                            args.max_sen_len = None
                        elif args.model_name == "attbilstm":
                            #args.max_sen_len = 30
                            args.max_sen_len = None
                        elif args.model_name == "textrnn":
                            args.max_sen_len = None
                        elif args.model_name == "rcnn":
                            args.max_sen_len = None
                        elif args.model_name == "s2satt":
                            args.max_sen_len = None
                        """
                    elif config[0] == "optim":
                        args.optim = str(config[1])
                        print("OPTIM:", args.optim.upper())

                if args.dataset == "mnist":
                    args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 10
                    args.data_type = "image"
                elif args.dataset == "cifar10":
                    args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 10
                    args.data_type = "image"
                elif args.dataset == "cifar100":
                    args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 100
                    args.data_type = "image"
                elif args.dataset == "svhn":
                    args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 10
                    args.data_type = "image"
                elif args.dataset == "stl10":
                    args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 10
                    args.data_type = "image"
                elif args.dataset == "imagenet32":
                    args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 1000
                    args.data_type = "image"
                elif args.dataset == "tinyimagenet200":
                    args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 200
                    args.data_type = "image"
                elif args.dataset == "imagenet2012":
                    args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 1000
                    args.data_type = "image"
                elif args.dataset == "agnews":
                    args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 4
                    args.data_type = "text"
                elif args.dataset == "yelprf":
                    args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 5
                    args.data_type = "text"
                elif args.dataset == "yahooa":
                    args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 10
                    args.data_type = "text"
                elif args.dataset == "amazonrf":
                    args.number_of_model_classes = args.number_of_model_classes if args.number_of_model_classes else 5
                    args.data_type = "text"

                #args.experiment_alternative_path = os.path.join("pretrained", args.exp_input, args.loss)
                #if not os.path.exists(args.experiment_alternative_path):
                #    os.makedirs(args.experiment_alternative_path)

                print("***************************************************************")

                for args.execution in range(args.start_executions, args.start_executions + args.executions):
                #for args.execution in range(1, args.executions + 1):

                    print("\n\n################ EXECUTION:", args.execution, "OF", args.executions, "################")

                    args.best_model_file_path = os.path.join(args.experiment_path, "model" + str(args.execution) + ".pth")

                    random.seed(args.base_seed)
                    numpy.random.seed(args.base_seed)
                    torch.manual_seed(args.base_seed)
                    torch.cuda.manual_seed(args.base_seed)
                    args.execution_seed = args.base_seed + args.execution
                    print("EXECUTION SEED:", args.execution_seed)

                    # For all experiment types, only the last args is kept for all executions...
                    utils.save_dict_list_to_csv([vars(args)], args.experiment_path, args.exp_type+"_args")
                    print("\nARGUMENTS:", dict(utils.load_dict_list_from_csv(args.experiment_path, args.exp_type+"_args")[0]))

                    #if args.exp_type == "cnn_train":
                    cnn_agent = agents.ClassificationAgent(args)
                    cnn_agent.train_validate()
                    #elif args.exp_type == "cnn_odd_infer":
                    #    cnn_agent = agents.CNNAgent(args)
                    #    cnn_agent.odd_infer()
                    #elif args.exp_type == "cnn_adv_infer":
                    #    cnn_agent = agents.CNNAgent(args)
                    #    cnn_agent.adv_infer()

                experiment_results = pd.read_csv(os.path.join(os.path.join(args.experiment_path, "results_best.csv")))
                print("\n################################\n", "EXPERIMENT RESULTS", "\n################################")
                print("\n", args.experiment_path.upper())
                #print("\n", experiment_results.transpose())
                print("\n", experiment_results)
                print("\n", experiment_results.describe())

                all_experiment_results[args.experiment_path] = experiment_results

    #print("\n\n\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n", "ALL EXPERIMENT RESULTS", "\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    #for key in all_experiment_results:
    #    print("\n", key.upper())
    #    print("\n", all_experiment_results[key].transpose())
    #    print("\n", all_experiment_results[key].describe().reindex(['count', 'avg', 'std']))


if __name__ == '__main__':
    main()
