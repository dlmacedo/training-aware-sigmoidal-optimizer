import os
import torch
import pandas as pd
import seaborn as sns
import numpy
from sklearn.metrics import accuracy_score
import utils

sns.set(style="darkgrid")


class MLAgent:

    def __init__(self, args):

        self.args = args

        # basic load block to get logits and targets tensors for the  trainset...
        trainset_first_partition_features_file = '{}/{}/{}.pth'.format(
            'experiments', args.exp_input, 'best_model_' + str(args.execution) + '_trainset_first_partition')
        trainset_first_partition_features = torch.load(trainset_first_partition_features_file)
        trainset_first_partition_features_logits = trainset_first_partition_features[0]
        trainset_first_partition_features_targets = trainset_first_partition_features[2].long()
        trainset_first_partition_features_targets_numpy = numpy.asarray(trainset_first_partition_features_targets)

        trainset_second_partition_features_file = '{}/{}/{}.pth'.format(
            'experiments', args.exp_input, 'best_model_' + str(args.execution) + '_trainset_second_partition')
        trainset_second_partition_features = torch.load(trainset_second_partition_features_file)
        trainset_second_partition_features_logits = trainset_second_partition_features[0]
        trainset_second_partition_features_targets = trainset_second_partition_features[2].long()
        trainset_second_partition_features_targets_numpy = numpy.asarray(trainset_second_partition_features_targets)

        # basic load block to get logits and targets tensors for the valset...
        valset_features_file = '{}/{}/{}.pth'.format(
            'experiments', args.exp_input, 'best_model_' + str(args.execution) + '_valset')
        valset_features = torch.load(valset_features_file)
        self.valset_features_logits = valset_features[0]
        self.valset_features_targets = valset_features[2].long()
        self.valset_features_targets_numpy = numpy.asarray(self.valset_features_targets)

        print("\nDATASET:", args.dataset)

        # create model
        torch.manual_seed(self.args.execution_seed)
        torch.cuda.manual_seed(self.args.execution_seed)
        ####################
        ### Code to create ml model (ex: SVM)
        ### Code to create ml model (ex: SVM)
        ### Code to create ml model (ex: SVM)
        ### Code to create ml model (ex: SVM)
        ### Code to create ml model (ex: SVM)
        ### Code to create ml model (ex: SVM)
        #print("=> creating model '{}'".format(self.model_name))
        #self.model = models.__dict__[self.model_name](num_classes=self.number_of_model_classes)
        #self.model.cuda()
        #print("\nMODEL:", self.model)
        ####################
        torch.manual_seed(self.args.base_seed)
        torch.cuda.manual_seed(self.args.base_seed)

        #self.executions_results_file_path = os.path.join(self.args.experiment_path, "results.csv")
        #self.executions_raw_results_file_path = os.path.join(self.args.experiment_path, "raw_results.csv")

        """
        self.original_learning_rate = args.original_learning_rate
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.learning_rate_decay_epochs = args.learning_rate_decay_epochs
        self.learning_rate_decay_rate = args.learning_rate_decay_rate

        # create training
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.original_learning_rate, momentum=self.momentum, weight_decay=args.weight_decay, nesterov=False)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.learning_rate_decay_epochs, gamma=args.learning_rate_decay_rate)
        """

    def train_validate(self):

        # building results and raw results files...
        if self.args.execution == 1:
            with open(self.args.executions_results_file_path, "w") as results:
                results.write("VAL ACC1\n")
            #with open(self.args.executions_raw_results_file_path, "w") as raw_results:
            #    raw_results.write("Execution,Epoch,Set,Type,Value\n")

        print("\n################ TRAINING ################")

        best_results = {}
        # Calculating map val acc1...
        map_val_predictions = utils.probabilities(self.valset_features_logits).max(1)[1]  # get the index of the max probability
        map_val_predictions_numpy = map_val_predictions.numpy()
        best_results["VAL ACC1"] = accuracy_score(self.valset_features_targets_numpy, map_val_predictions_numpy)
        print("VAL ACC1:\t", best_results["VAL ACC1"], "\n\n")

        """
        for self.epoch in range(1, self.epochs + 1):
            print("\n######## EPOCH:", self.epoch, "OF", self.epochs, "########")

            # Adjusting learning rate (if not using reduce on plateau)...
            self.scheduler.step()

            # Print current learning rate...
            for param_group in self.optimizer.param_groups:
                print("\nLEARNING RATE:\t\t", param_group["lr"])

            train_acc1, train_purity, train_loss, train_mean_entropy, train_entropy_mean = self.train()
            val_acc1, val_purity = self.validate()

            # Saving raw results...
            criterion_raw_dict["Execution"].append(self.execution)
            criterion_raw_dict["Epoch"].append(self.epoch)
            criterion_raw_dict["Set"].append("Train")
            criterion_raw_dict["Type"].append("Loss")
            criterion_raw_dict["Value"].append(train_loss)

            criterion_raw_dict["Execution"].append(self.execution)
            criterion_raw_dict["Epoch"].append(self.epoch)
            criterion_raw_dict["Set"].append("Train")
            criterion_raw_dict["Type"].append("Mean Entropy")
            criterion_raw_dict["Value"].append(train_mean_entropy)

            criterion_raw_dict["Execution"].append(self.execution)
            criterion_raw_dict["Epoch"].append(self.epoch)
            criterion_raw_dict["Set"].append("Train")
            criterion_raw_dict["Type"].append("Entropy Mean")
            criterion_raw_dict["Value"].append(train_entropy_mean)

            # if is best...
            if train_loss < best_model_results["TRAIN LOSS"]:
                best_model_results = {"TRAIN ACC1": train_acc1, "TRAIN PURITY": train_purity,
                                      "VALID ACC1": val_acc1, "VALID PURITY": val_purity,
                                      "TRAIN LOSS": train_loss,
                                      "TRAIN MEAN ENTROPY": train_mean_entropy,
                                      "TRAIN ENTROPIES MEAN": train_entropy_mean}

                best_model_file_path = os.path.join(self.experiment_path, 'best_model_' + str(self.execution) + '.pth.tar')

                print("!+NEW BEST MODEL TRAIN LOSS:\t\t{0:.4f} IN EPOCH {1}! SAVING {2}".format(
                    train_loss, self.epoch, best_model_file_path))
                print("!+NEW BEST MODEL TRAIN PURITY:\t\t{0:.4f} IN EPOCH {1}! SAVING {2}".format(
                    train_purity, self.epoch, best_model_file_path))
                print("!+NEW BEST MODEL VALID ACCURACY:\t{0:.4f} IN EPOCH {1}! SAVING {2}\n".format(
                    val_acc1, self.epoch, best_model_file_path))

                full_state = {'epoch': self.epoch, 'model': self.model_name, 'model_state_dict': self.model.state_dict(),
                              'best_train_purity': best_model_results["TRAIN PURITY"], 'best_val_acc1': best_model_results["VALID ACC1"]}

                torch.save(full_state, best_model_file_path)

            print('!$$$$ BEST MODEL TRAIN PURITY:\t\t{0:.3f}'.format(best_model_results["TRAIN PURITY"]))
            print('!$$$$ BEST MODEL VALID PURITY:\t\t{0:.3f}'.format(best_model_results["VALID PURITY"]))
            print('!$$$$ BEST MODEL TRAIN ACCURACY:\t{0:.3f}'.format(best_model_results["TRAIN ACC1"]))
            print('!$$$$ BEST MODEL VALID ACCURACY:\t{0:.3f}\n'.format(best_model_results["VALID ACC1"]))

            # Adjusting learning rate (if using reduce on plateau)...
            # scheduler.step(val_acc1)
        """

        #criterion_raw_dataframe = pd.DataFrame(data=criterion_raw_dict)

        """
        # plt.figure()
        ax = sns.lineplot(x="Epoch", y="Value", hue="Type", data=criterion_raw_dataframe)
        ax.set_title("Criteria")
        # plt.show()
        plt.savefig(os.path.join(self.experiment_path, 'criterion_raw'))
        # tikz_save(os.path.join(path, 'criterion_raw.tex'), figurewidth='\\0.5textwidth')
        criterion_raw_dataframe.to_csv(os.path.join(self.experiment_path, 'criterion_raw.csv'), encoding='utf-8', index=False)
        plt.close()
        """

        #utils.save_object(best_results, self.args.experiment_path, "execution_results")
        with open(self.args.executions_results_file_path, "a") as results:
            results.write("{}\n".format(
                best_results["VAL ACC1"],
                )
            )


#def compute_probabilities(logits, dim=1):
#    return nn.Softmax(dim=dim)(logits)

