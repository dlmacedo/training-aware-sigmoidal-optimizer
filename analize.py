
import argparse
import os
import torch
import math
import sys
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import statistics
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import time
import matplotlib.patheffects as PathEffects
import pickle
import statistics
from scipy.interpolate import interp1d 

#import numpy as np
from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

"""
#######################
#matplotlib.rcParams.update({'font.size': 24})
#sns.set(font_scale=3)
##plt.rc('font', weight="medium")
##plt.rc('text', usetex=True)
sns.set_style('darkgrid')
sns.set_palette('muted')
#sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set_context("paper")
#sns.set_context("paper", font_scale=1.5)
#######################
"""

pd.options.display.float_format = '{:,.4f}'.format
pd.set_option('display.width', 160)

parser = argparse.ArgumentParser(description='Analize results in csv files')

parser.add_argument('-p', '--path', default="", type=str, help='Path for the experiments to be analized')
parser.set_defaults(argument=True)

args = parser.parse_args()
path = os.path.join("expers", args.path)
if not os.path.exists(path):
    sys.exit('You should pass a valid path to analyze!!!')

random.seed(1000000)
np.random.seed(1000000)
torch.manual_seed(1000000)
torch.cuda.manual_seed(1000000)


def main():

    #############################################################################
    #############################################################################
    DATASETS = ['cifar10', 'cifar100', 'svhn', 'stl10']
    #############################################################################
    #############################################################################

    EXTRA_DATASETS = [
        'svhn_250', 'svhn_500', 'svhn_1000', 'svhn_2000', 'svhn_3000', 'svhn_4000', 'svhn_5000',
        'cifar10_250', 'cifar10_500', 'cifar10_1000', 'cifar10_2000', 'cifar10_3000', 'cifar10_4000', 'cifar10_5000', 
        'cifar100_25', 'cifar100_50', 'cifar100_100', 'cifar100_200', 'cifar100_300', 'cifar100_400', 'cifar100_500',
        'stl10_25', 'stl10_50', 'stl10_100', 'stl10_200', 'stl10_300', 'stl10_400', 'stl10_500',
        ]

    #############################################################################
    #############################################################################
    #MODELS = ['vgg19', 'resnet34', 'resnet50', 'wideresnet3410', 'densenetbc100'] 
    MODELS = ['vgg19', 'resnet34', 'resnet50', 'densenetbc100'] 
    #############################################################################
    #############################################################################

    """
    #LOSSES = ['sml1_na_id_no_no_no_no', 'dml10_pn2_id_no_no_no_no', 'eml1_pn2_id_no_no_lz0_10_ST_NO_0.01']
    LOSSES = ['sml1_na_id_no_no_no_no', 'dml10_pn2_id_no_no_no_no']
    #LOSSES_TEXTS = [r"SoftMax", r"IsoMax", r"IsoMax$_2$"]
    #LOSSES_TEXTS = [r"SoftMax", r"IsoMax"]
    PRINT_LOSS = {
        'sml1_na_id_no_no_no_no': r"SoftMax",
        'dml10_pn2_id_no_no_no_no': r"IsoMax",
        #'eml1_pn2_id_no_no_lz0_10_ST_NO_0.01': r"IsoMax$_2$",
        #'eml1_pn2_id_no_no_lz0_10_SC_NO_0.01': r"IsoMax$_2$"
        }
    """

    #OPTIM = ['taso_l0.1_e5_w0.0001_m0.9_nt_a25_b0.6','taso_l0.1_e5_w0.0001_m0.9_nt_a25_b0.7','taso_l0.1_e5_w0.0001_m0.9_nt_a25_b0.8']
    #PRINT_OPTIM = ['TASO1','TASO2','TASO3']

    PRINT_MODEL = {'densenetbc100': r'DenseNetBC100', 'resnet50': r'ResNet50', 'vgg19': r'VGG19'}
    PRINT_DATA = {'svhn': r'SVHN', 'cifar10': r'CIFAR10', 'cifar100': r'CIFAR100'}
    #####################################################################################################################
    #SVHN_LOSSES = ['sml1_na_id_no_no_no_no', 'dml1_pn2_id_no_no_no', 'dml3_pn2_id_no_no_no', 'dml10_pn2_id_no_no_no_no']
    #SVHN_LOSSES_TEXTS = [r"SoftMax", r"IsoMax $(E_s\!=\!1)$", r"IsoMax $(E_s\!=\!3)$", r"IsoMax $(E_s\!=\!10)$"]
    #####################################################################################################################

    print(DATASETS)
    print(MODELS)
    #print(LOSSES)
    ##print(LOSSES_TEXTS)
    ##print(SVHN_LOSSES)
    ##print(SVHN_LOSSES_TEXTS)
    #print(OPTIM)

    ############################################################################
    ############################################################################
    #sns.set_context("paper", font_scale=1.6)
    print("\n###########################")
    print("####### PLOT TASO #########")
    print("###########################")
    
    x = np.linspace(0,100,100)
    y = 1/(1 + np.exp(40*((x/100)-0.6))) + 0.001

    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)
    fig, ax = plt.subplots()

    # We need to draw the canvas, otherwise the labels won't be positioned and won't have values yet.
    fig.canvas.draw()
    ax.set_xticks([0,25,50,75,100])
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = '0'
    labels[1] = r'$\mathrm{E_F}/4$'
    labels[2] = r'$\mathrm{E_F}/2$'
    labels[3] = r'$\mathrm{3E_F}/4$'
    labels[4] = r'$\mathrm{E_F}$'
    ax.set_xticklabels(labels)
    #ax.set_yticks([1])
    #labels = [item.get_text() for item in ax.get_yticklabels()]
    #labels[0] = r'$\mathrm{LR_I}$'
    ax.set_yticklabels([])

    ax.axvline(55, color='red', linestyle='dashed')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Learning Rate')
    #ax.set_title('Training Aware Sigmoidal Optimization')

    #label = r'$\mathrm{LR}=\frac{\mathrm{LR_I}}{1+\exp\left(\alpha \left(\frac{\mathrm{E}}{\mathrm{E_F}}-\beta\right)\right)}+\mathrm{LR_F}$'
    label = r'$\mathrm{LR}=\frac{\mathrm{LR_I}}{1+\exp\left(\alpha \left(\mathrm{E}/\mathrm{E_F}-\beta\right)\right)}+\mathrm{LR_F}$'
    #label = r'$LR=\frac{LR_I}{1+\exp\left(\alpha \left(\frac{E}{E_F}-\beta\right)\right)}+LR_F$'
    plt.plot(x, y, 'y', label=label, color='blue')  
    plt.legend(loc='lower left', fontsize=11)
    #with sns.axes_style("white"):
    #    plt.legend(loc='lower right', bbox_to_anchor=(1, -0.25), ncol=6, fontsize=13)

    #plt.show()
    plt.savefig(os.path.join(path, 'plot_taso_curve.png'), bbox_inches='tight', dpi=300)
    plt.close()
    ############################################################################
    ############################################################################


    ############################################################################
    ############################################################################
    #sns.set_context("paper", font_scale=1.6)
    print("\n#########################################")
    print("####### PLOT SADDLE AND MINIMUM #########")
    print("#########################################")

    #fig = plt.figure(figsize=(3, 6))
    fig = plt.figure()   
    ax = fig.add_subplot(111, projection='3d')
    #ax = plt.subplot(2, 1, 1, projection='3d')
    plot_args = {'rstride': 1, 'cstride': 1, 'cmap':"Blues_r", 'linewidth': 0.4, 'antialiased': True, 'vmin': -1, 'vmax': 1}
    x, y = np.mgrid[-1:1:31j, -1:1:31j]
    z = 0.5*x**2 - 0.5*y**2
    #z = 0.5*x**2 + 0.5*y**2 - 1
    ax.plot_surface(x, y, z, **plot_args)
    #ax.plot([0], [0], [0], 'ro')
    ax.view_init(azim=-60, elev=30)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.xticks([-1, -0.5, 0, 0.5, 1], [r"$-1$", r"$-1/2$", r"$0$", r"$1/2$", r"$1$"])
    plt.yticks([-1, -0.5, 0, 0.5, 1], [r"$-1$", r"$-1/2$", r"$0$", r"$1/2$", r"$1$"])
    ax.set_zticks([-1, -0.5, 0, 0.5, 1])
    ax.set_zticklabels([r"$-1$", r"$-1/2$", r"$0$", r"$1/2$", r"$1$"])
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    #plt.savefig("Saddle_point.svg", bbox_inches="tight", transparent=True)
    plt.savefig(os.path.join(path, 'plot_saddle_point.png'), bbox_inches='tight', dpi=300)
    #plt.show()
    plt.close()

    #fig = plt.figure(figsize=(3, 6))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax = plt.subplot(2, 1, 2, projection='3d')
    plot_args = {'rstride': 1, 'cstride': 1, 'cmap':"Blues_r", 'linewidth': 0.4, 'antialiased': True, 'vmin': -1, 'vmax': 1}
    x, y = np.mgrid[-1:1:31j, -1:1:31j]
    #z = 0.5*x**2 - 0.5*y**2
    z = 0.5*x**2 + 0.5*y**2 - 1
    ax.plot_surface(x, y, z, **plot_args)
    #ax.plot([0], [0], [0], 'ro')
    ax.view_init(azim=-60, elev=30)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.xticks([-1, -0.5, 0, 0.5, 1], [r"$-1$", r"$-1/2$", r"$0$", r"$1/2$", r"$1$"])
    plt.yticks([-1, -0.5, 0, 0.5, 1], [r"$-1$", r"$-1/2$", r"$0$", r"$1/2$", r"$1$"])
    ax.set_zticks([-1, -0.5, 0, 0.5, 1])
    ax.set_zticklabels([r"$-1$", r"$-1/2$", r"$0$", r"$1/2$", r"$1$"])
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    #plt.savefig("Saddle_point.svg", bbox_inches="tight", transparent=True)
    plt.savefig(os.path.join(path, 'plot_local_minimum.png'), bbox_inches='tight', dpi=300)
    #plt.show()
    plt.close()
    ############################################################################
    ############################################################################



    #######################
    #matplotlib.rcParams.update({'font.size': 24})
    #sns.set(font_scale=3)
    ##plt.rc('font', weight="medium")
    ##plt.rc('text', usetex=True)
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    #sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    sns.set_context("paper")
    #sns.set_context("paper", font_scale=1.5)
    #######################




    print("\n#####################################")
    print("########## FINDING FILES ############")
    print("#####################################")
    list_of_files = []
    file_names_dict_of_lists = {}
    for (dir_path, dir_names, file_names) in os.walk(path):
        for filename in file_names:
            if filename.endswith('.csv') or filename.endswith('.npy') or filename.endswith('.pth'):
                if filename not in file_names_dict_of_lists:
                    file_names_dict_of_lists[filename] = [os.path.join(dir_path, filename)]
                else:
                    file_names_dict_of_lists[filename] += [os.path.join(dir_path, filename)]
                list_of_files += [os.path.join(dir_path, filename)]
    print()
    for key in file_names_dict_of_lists:
        print(key)
        #print(file_names_dict_of_lists[key])


    print("\n#####################################")
    print("######## TABLE: RAW RESULTS #########")
    print("#####################################")
    data_frame_list = []
    for file in file_names_dict_of_lists['results_raw.csv']:
        data_frame_list.append(pd.read_csv(file))
    raw_results_data_frame = pd.concat(data_frame_list)
    #raw_results_data_frame.to_csv(os.path.join(path, 'all_results_raw.csv'), index=False)
    print(raw_results_data_frame)


    print("\n#####################################")
    print("###### TABLE: BEST ACCURACIES #######")
    print("#####################################")
    data_frame_list = []
    for file in file_names_dict_of_lists['results_best.csv']:
        #df = pd.read_csv(file)
        #print(df)
        data_frame_list.append(pd.read_csv(file))
    best_results_data_frame = pd.concat(data_frame_list)
    """
    print("#########################")
    print("#########################")
    print(best_results_data_frame)
    print("#########################")
    print("#########################")
    """
    best_results_data_frame.to_csv(os.path.join(path, 'all_results_best.csv'), index=False)
    #######################################################################################
    dfx = best_results_data_frame.loc[best_results_data_frame['DATA'].isin(EXTRA_DATASETS)]
    dfx.to_csv(os.path.join(path, 'extra_results_best.csv'), index=False)
    #######################################################################################
    #print(best_results_data_frame, "\n")
    for data in DATASETS:
    #for data in (DATASETS + EXTRA_DATASETS):
        for model in MODELS:
            print("\n########")
            print(data)
            print(model)
            ####best_results_data_frame['VALID DIFF'] = best_results_data_frame['VALID INTER_LOGITS MEAN'] - best_results_data_frame['VALID INTRA_LOGITS MEAN']
            ####best_results_data_frame['VALID DIV'] = best_results_data_frame['VALID INTER_LOGITS MEAN'] / best_results_data_frame['VALID INTRA_LOGITS MEAN']
            df = best_results_data_frame.loc[
                best_results_data_frame['DATA'].isin([data]) &
                best_results_data_frame['MODEL'].isin([model])
            ]
            #df = df[['OPTIM','TRAIN LOSS', 'TRAIN ACC1','VALID LOSS', 'VALID ACC1',]]
            print(df)
            print()
            dfx = df.groupby('OPTIM', as_index=False)[['TRAIN LOSS', 'TRAIN ACC1','VALID LOSS', 'VALID ACC1']].mean()
            dfx = dfx.rename(columns={'TRAIN LOSS': 'TRAIN LOSS MEAN', 'TRAIN ACC1': 'TRAIN ACC1 MEAN',
                'VALID LOSS': 'VALID LOSS MEAN', 'VALID ACC1': 'VALID ACC1 MEAN'})
            dfx = dfx.sort_values('VALID ACC1 MEAN', ascending=False)#.drop_duplicates(["LOSS"])
            print(dfx)
            print()
            #dfx = df.groupby('OPTIM', as_index=False)[['TRAIN LOSS', 'TRAIN ACC1','VALID LOSS', 'VALID ACC1']].agg([np.mean, np.std, np.count])
            dfx = df.groupby('OPTIM', as_index=False)[['TRAIN LOSS', 'TRAIN ACC1','VALID LOSS', 'VALID ACC1']].agg(['mean', 'std', 'count'])
            print(dfx)
            ####df.to_csv(os.path.join(path, data+'+'+model+'+results_best.csv'), index=False)
            ####df = df.rename(columns={'VALID INTRA_LOGITS MEAN': 'VIALM', 'VALID INTER_LOGITS MEAN': 'VIELM'})
            print("########\n")

    sys.exit()










    print("\n########################################")
    print("######## TABLE: INFERENCE DEALYS #######")
    print("########################################")
    data_frame_list = []
    for file in file_names_dict_of_lists['results_odd.csv']:
        data_frame_list.append(pd.read_csv(file))
    best_results_data_frame = pd.concat(data_frame_list)
    for model in MODELS:
        print("\n########")
        print(model)
        #for data in DATASETS:
        #    print(data)
        df = best_results_data_frame.loc[
            #best_results_data_frame['IN-DATA'].isin([data]) &
            best_results_data_frame['MODEL'].isin([model]) &
            #best_results_data_frame['LOSS'].isin(['eml_pn2_id_no_no_lz_10_ST_NO_0']) &
            #best_results_data_frame['LOSS'].isin(['eml_pn2_id_no_no_lz_10_ST_NO_0.1']) &
            #best_results_data_frame['LOSS'].isin(['eml_pn2_id_no_no_lz_10_ST_NO_0.01']) &
            best_results_data_frame['INFER-LEARN'].isin(['NO']) &
            #best_results_data_frame['INFER-LEARN'].isin(['NO','LR']) &
            best_results_data_frame['INFER-TRANS'].isin([False]) &
            #best_results_data_frame['SCORE'].isin(["NE"]) &
            #best_results_data_frame['SCORE'].isin(["NE","MP"]) &
            # THE MEAN OF THE METRICS SHOULD EXCLUDE NOISE OUT DATA???
            best_results_data_frame['OUT-DATA'].isin(['svhn','cifar10'])
            #best_results_data_frame['OUT-DATA'].isin(['gaussian_noise','uniform_noise'])
            #best_results_data_frame['OUT-DATA'].isin(['svhn','lsun_resize','imagenet_resize','cifar10','gaussian_noise','uniform_noise'])
            ####best_results_data_frame['OUT-DATA'].isin(['svhn','lsun_resize','imagenet_resize','cifar10','gaussian_noise','uniform_noise','fooling_images'])
            #best_results_data_frame['OUT-DATA'].isin(['fooling_images'])
            # DO NOT USE FOOLING IMAGES ANYMORE SINCE NOBODY USES AND IT IS UNLIKE IN REALWORLD!!!
            ]
        df = df[['MODEL','IN-DATA','LOSS','OUT-DATA','SCORE','CPU_FALSE','CPU_TRUE','GPU_FALSE','GPU_TRUE']]
        ######################################################################################
        #####df = df.sort_values(['LOSS','OUT-DATA','AUROC'], ascending=False)
        #df = df.sort_values(['OUT-DATA','AUROC'], ascending=False)
        ##df = df.sort_values(['OUT-DATA','DTACC'], ascending=False)
        #####df.to_csv(os.path.join(path, data+'_'+model+'_results_best.csv'), index=False)
        #####df = df.groupby(['LOSS','AD-HOC','SCORE','INFER-LEARN','INFER-TRANS'], as_index=False)['AUROC'].mean()
        ###df = df.groupby(['LOSS','INFER-LEARN','SCORE'], as_index=False)['TNR'].mean()
        ###df = df.sort_values(['TNR'], ascending=False)
        #df = df.groupby(['LOSS','INFER-LEARN','INFER-TRANS','SCORE'], as_index=False)['AUROC'].mean()
        ##df = df.groupby(['LOSS','INFER-LEARN','INFER-TRANS','SCORE'], as_index=False)['DTACC'].mean()
        #df = df.sort_values(['AUROC'], ascending=False)
        ##df = df.sort_values(['DTACC'], ascending=False)
        ######################################################################################
        df = df.groupby(['MODEL','IN-DATA','LOSS','SCORE'], as_index=False)['CPU_FALSE','CPU_TRUE','GPU_FALSE','GPU_TRUE'].mean()
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
    


    sns.set_context("paper", font_scale=1.6)
    #MODELS = ['densenetbc100','resnet34']
    #METRICS = ['VALID ACC1','MEAN AUROC','MEAN TNR']
    #METRICS = ['MEAN AUROC']
    #PAPERS = ['odd2']
    #PAPERS = ['odd1','odd2']
    #DATA = ['svhn', 'cifar10','cifar100']        
    print("\n###############################################")
    print("####### FIGURES: ENTROPIC SCORE STUDY #########")
    print("###############################################")
    fig = plt.figure(figsize=(18, 3))
    ax = plt.subplot(1, 2, 1)   
    ax.plot([1,4,7,10,13,16], [95.1,95.3,95.0,95.2,95.0,95.0], label="DenseNet|CIFAR10")
    ax.plot([1,4,7,10,13,16], [95.4,95.6,95.4,95.6,95.6,95.2], label="ResNet|CIFAR10")
    ax.plot([1,4,7,10,13,16], [77.0,77.9,77.7,77.5,77.3,76.9], label="DenseNet|CIFAR100")
    ax.plot([1,4,7,10,13,16], [77.3,78.1,77.6,77.4,77.0,77.2], label="ResNet|CIFAR100")
    ax.plot([1,4,7,10,13,16], [96.8,96.9,96.6,96.6,96.7,96.7], label="DenseNet|SVHN")
    ax.plot([1,4,7,10,13,16], [97.0,96.8,96.8,96.8,96.6,96.7], label="ResNet|SVHN")
    ax.axvline(10, color='black', linestyle='dashed')
    ax.set_xticks([1,4,7,10,13,16])
    ax.set_xlabel('Entropic Score')
    ax.set_ylim(70, 100)
    ax.set_title('Classification [Accuracy] (%)', loc='center')
    ax = plt.subplot(1, 2, 2)   
    ax.plot([1,4,7,10,13,16], [92.4,95.1,97.3,97.3,98.0,98.0], label="DenseNet|CIFAR10")
    ax.plot([1,4,7,10,13,16], [91.8,94.3,95.7,96.2,96.1,96.1], label="ResNet|CIFAR10")
    ax.plot([1,4,7,10,13,16], [75.5,82.2,91.0,91.9,91.9,91.5], label="DenseNet|CIFAR100")
    ax.plot([1,4,7,10,13,16], [78.8,83.7,86.7,91.2,92.3,91.8], label="ResNet|CIFAR100")
    ax.plot([1,4,7,10,13,16], [97.6,98.9,99.1,98.9,99.0,98.8], label="DenseNet|SVHN")
    ax.plot([1,4,7,10,13,16], [93.3,95.5,98.3,98.1,98.3,98.1], label="ResNet|SVHN")
    ax.axvline(10, color='black', linestyle='dashed')
    ax.set_xticks([1,4,7,10,13,16])
    ax.set_xlabel('Entropic Score')
    ax.set_ylim(70, 100)
    ax.set_title(r'OOD Detection [$\overline{\mathrm{AUROC}}$] (%)', loc='center')
    with sns.axes_style("white"):
        plt.legend(loc='upper right', bbox_to_anchor=(1, -0.25), ncol=6, fontsize=13)
    #fig.subplots_adjust(top=0.83)
    #fig.suptitle("ENTROPIC SCORE STUDY", fontsize=13)
    plt.savefig(os.path.join(path, 'plot_odd1_entropic_score.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("############################## ACABOU!!! ######################")  



    sns.set_context("paper", font_scale=1.4)
    print("\n#####################################")
    print("###### entropic scale parametrization ########")
    print("#####################################")
    for data in ["svhn"]:
        for model in ["densenetbc100"]:
            print("\n########")
            #df = pd.read_csv(os.path.join("expers", "cnn_train", "svhn_tnr.csv"))
            df = pd.read_csv("svhn_tnr.csv")
            print(df.columns)
            print(df)
            plt.figure(figsize=(15, 4))
            plt.title(#"IN-DIST=" + PRINT_DATA[data] +
            "MODEL=" + PRINT_MODEL[model] + 
            " | CLASSIFICATION METRIC=Accuracy" +
            " | SCORE=Entropic Score" +
            " | OOD DETECTION METRIC=TNR@TPR95")
            #ax = sns.barplot(x="metric", y="value", hue="loss", palette="YlGnBu", data=df)
            ax = sns.barplot(x="metric", y="value", hue="loss", data=df)
            ax.set(ylabel=r"Value (%)")
            ax.set_ylim(75, 100)
            #ax.set(xlabel=r"Loss")
            ax.set(xlabel="")
            string = ''
            for i, item in enumerate(df.iterrows()):
                print("#######")
                print(item[0])
                print(item[1])
                #print(i % 5)
                print(i % 4)
                print("#######")
                string += '{:4.2f}'.format(round(item[1].value, 2)) + '    '
                ##if (i % 5) == 4:
                if (i % 4) == 3:
                    ##ax.text(i//5, 2, string.rstrip(), color='black', fontsize=8, ha="center")
                    #ax.text(i//4, 75.5, string.rstrip(), color='black', fontsize=8, ha="center")
                    string = ''
            plt.vlines(x=0.5, ymin=75, ymax=100, color='red', linestyles='dashed',)
            labels = [item.get_text() for item in ax.get_xticklabels()]
            labels[0] = 'SVHN [Classification]'
            labels[1] = 'OOD: LSUN [Detection]'
            labels[2] = 'OOD: TinyImageNet'
            labels[3] = 'OOD: CIFAR10'
            labels[4] = 'OOD: CIFAR100'
            ax.set_xticklabels(labels)
            #ax.legend(title="ggggggg")
            with sns.axes_style("white"):
                #legend = plt.legend(loc='upper right', bbox_to_anchor=(0.8, -0.175), ncol=5)
                legend = plt.legend(loc='upper right', bbox_to_anchor=(0.8, -0.125), ncol=5)
                #legend.set_title(' ')
                #for t, l in zip(plt.legend(loc='upper right', bbox_to_anchor=(0.8, -0.175), ncol=5, title=" ").get_texts(), SVHN_LOSSES_TEXTS):
                for t, l in zip(legend.get_texts(), SVHN_LOSSES_TEXTS):
                    t.set_text(l)
            plt.savefig(os.path.join(path, 'plot_odd1_entropic_scale_parametrization_'+model+'_'+data), bbox_inches='tight', dpi=150)
            plt.close()
            print("########\n")



    sns.set_context("paper", font_scale=1.5)
    print("\n############################################################################")
    print("####### FIGURES: NEW NEW NEW!!! TRAINING LOSS AND VALID ACCURACIES #########")
    print("############################################################################")
    print()
    #print("PAPER:", paper)
    df = raw_results_data_frame.loc[
        raw_results_data_frame['LOSS'].isin(LOSSES) &
        raw_results_data_frame['SET'].isin(['VALID']) &
        raw_results_data_frame['METRIC'].isin(['ACC1'])# &
        #raw_results_data_frame['EXECUTION'].isin(['1'])
        ]
    #print(df)
    df = df.replace({'cifar10': 'CIFAR10', 'cifar100': 'CIFAR100', 'svhn': 'SVHN'})
    df = df.replace({'densenetbc100': 'DenseNet', 'resnet34': 'ResNet'})
    #print(df)
    #shift = -0.3
    for metric in ['TRAINING LOSS','VALID ACCURACY']:
        print("METRIC:", metric)
        fig = plt.figure(figsize=(21, 3))
        if metric == 'TRAINING LOSS':
            #print("PAPER:", paper)
            df = raw_results_data_frame.loc[
                raw_results_data_frame['LOSS'].isin(LOSSES) &
                raw_results_data_frame['SET'].isin(['TRAIN']) &
                raw_results_data_frame['METRIC'].isin(['LOSS'])# &
                #raw_results_data_frame['EXECUTION'].isin(['1'])
                ]
        elif metric == 'VALID ACCURACY':
            #print("PAPER:", paper)
            df = raw_results_data_frame.loc[
                raw_results_data_frame['LOSS'].isin(LOSSES) &
                raw_results_data_frame['SET'].isin(['VALID']) &
                raw_results_data_frame['METRIC'].isin(['ACC1'])# &
                #raw_results_data_frame['EXECUTION'].isin(['1'])
                ]
        #print(df)
        df = df.replace({'cifar10': 'CIFAR10', 'cifar100': 'CIFAR100', 'svhn': 'SVHN'})
        df = df.replace({'densenetbc100': 'DenseNet', 'resnet34': 'ResNet'})
        #print(df)
        for model in ['DenseNet','ResNet']:
            print("MODEL:", model)
            for i, in_data in enumerate(['SVHN','CIFAR10','CIFAR100']):
                print('DATA:', in_data)                
                if model == 'DenseNet':
                    ax = plt.subplot(1, 6, i + 1)
                elif model == 'ResNet':
                    ax = plt.subplot(1, 6, i + 4)
                dfx = df.loc[df['MODEL'].isin([model]) & df['DATA'].isin([in_data])]
                #dfx = dfx.sort_values(['MODEL','DATA','EXAMPLES_PER_CLASS','VALID ACC1'], ascending=True)
                print(dfx)
                dftemp = dfx.loc[dfx['LOSS'].isin(['sml1_na_id_no_no_no_no'])] 
                ax.plot(dftemp["EPOCH"], dftemp["VALUE"], label="SoftMax")
                ##ax.plot([0,1,2,3,4], dftemp['VALID ACC1'], label="SoftMax")
                #if paper == 'odd1':
                dftemp = dfx.loc[dfx['LOSS'].isin(['dml10_pn2_id_no_no_no_no'])]
                ax.plot(dftemp["EPOCH"], dftemp["VALUE"], label="IsoMax")
                ##ax.plot([0,1,2,3,4], dftemp['VALID ACC1'], label="IsoMax")      
                ax.set_xlabel('Epoch')
                ax.set_title(model+" | "+in_data, loc='center')
                if metric == 'TRAINING LOSS':
                    ax.set_ylim(0, 1)
                    if i == 0 and model == 'DenseNet':
                        ax.set_ylabel('Loss (Train)')
                elif metric == 'VALID ACCURACY':
                    ax.set_ylim(0, 100)
                    if i == 0 and model == 'DenseNet':
                        ax.set_ylabel(r'Accuracy (Test) [%]')
        with sns.axes_style("white"):
            #plt.legend(loc='upper right', bbox_to_anchor=(0.3, -0.25), ncol=4, fontsize=13)
            #plt.legend(loc='upper right', bbox_to_anchor=(-0.3, -0.25), ncol=4, fontsize=13)
            #plt.legend(loc='upper right', bbox_to_anchor=(shift, -0.5), ncol=4, fontsize=12)
            plt.legend(loc='upper right', bbox_to_anchor=(-1.75, -0.25), ncol=4)
        #fig.subplots_adjust(top=0.9)
        #fig.suptitle(metric, fontsize=18)
        
        if metric == 'TRAINING LOSS':
            plt.savefig(os.path.join(path, 'plot_odd1_train_losses.png'), bbox_inches='tight', dpi=150)
        elif metric == 'VALID ACCURACY':
            plt.savefig(os.path.join(path, 'plot_odd1_test_accuracies.png'), bbox_inches='tight', dpi=150)
        plt.close()



    sns.set_context("paper", font_scale=1.2)
    print("\n##########################################################")
    print("######## FIGURES: TRAINING LOSSES AND ENTROPIES ##########")
    print("##########################################################")
    print()
    #print(raw_results_data_frame)
    #for data in DATASETS:
    for data in ['cifar10']:
        #for model in MODELS:
        for model in ['densenetbc100']:
            #for loss in LOSSES:
            #for loss in ['sml1_na_id_no_no_no_no', 'dml1_pn2_id_no_no', 'dml3_pn2_id_no_no', 'dml10_pn2_id_no_no_no_no']:
            for loss in ['sml1_na_id_no_no_no_no', 'dml10_pn2_id_no_no_no_no', 'sml1_na_id_no_no_no_no', 'dml10_pn2_id_no_no_no_no']:
                    print(data,model,loss)
                    df = raw_results_data_frame.loc[
                        raw_results_data_frame['DATA'].isin([data]) &
                        raw_results_data_frame['MODEL'].isin([model]) &
                        raw_results_data_frame['LOSS'].isin([loss]) &
                        #raw_results_data_frame['SET'].isin(['VALID']) &
                        raw_results_data_frame['METRIC'].isin(['LOSS','ENTROPIES MEAN'])# &
                        #raw_results_data_frame['EXECUTION'].isin(['1'])
                    ]
                    print(df)
                    #plt.figure(figsize=(6,3))
                    plt.figure(figsize=(3,3))
                    ax = sns.lineplot(x="EPOCH", y="VALUE",
                    hue="METRIC",
                    style="SET",
                    data=df)
                    #ax.set(ylabel=r"Value")
                    ax.set(ylabel=r"")
                    ax.set(xlabel=r"Epoch")
                    #ax.set_title('DATA='+PRINT_DATA[data]+' | MODEL='+PRINT_MODEL[model]+' | LOSS='+PRINT_LOSS[loss], loc='center')
                    #ax.set_title('LOSS='+PRINT_LOSS_ESPECIAL[loss], loc='center')
                    ax.set_title('LOSS='+PRINT_LOSS[loss], loc='center')
                    #ax.set_ylim(0, 3)
                    with sns.axes_style():
                        #for t, l in zip(plt.legend(loc='upper right', bbox_to_anchor=(-0.35, -0.2), ncol=5, title="Loss").get_texts(), LOSSES_TEXTS):
                        #for t, l in zip(plt.legend(loc='upper right', bbox_to_anchor=(-0.3, -0.2), ncol=5).get_texts(), LOSSES_TEXTS):
                        #for t, l in zip(plt.legend().get_texts(), LOSSES_TEXTS):
                        #for t in plt.legend().get_texts():
                        #    print(t)
                        #    #t.set_text(l)
                        #legends = plt.legend().get_texts()
                        #legends = plt.legend(fontsize=8).get_texts()
                        legend = plt.legend()
                        legends = legend.get_texts()
                        legends[0].set_text(r"METRIC")
                        legends[1].set_text(r"Loss")
                        legends[2].set_text(r"Entropy")
                        legends[3].set_text(r"SET")
                        legends[4].set_text(r"Train")
                        legends[5].set_text(r"Test")
                    if loss != 'sml1_na_id_no_no_no_no':
                        legend.remove()
                    plt.savefig(os.path.join(path, 'plot_train_losses_entropies+'+model+'+'+data+'+'+loss+'.png'), bbox_inches='tight', dpi=150)
                    plt.close()



    sns.set_context("paper", font_scale=1.3)
    print("\n#####################################")
    print("####### FIGURES: HISTOGRAMS #########")
    print("#####################################")
    print()
    bins = 30

    for data in DATASETS:
    #for data in ['cifar10']:
        if data == 'cifar10':
            out_data_list = ['svhn', 'imagenet_resize', 'lsun_resize']
            #out_data_list = ['svhn', 'imagenet_resize', 'lsun_resize', 'gaussian_noise']
            #out_data_list = ['svhn', 'imagenet_resize', 'lsun_resize', 'gaussian_noise','uniform_noise']
        elif data == 'cifar100':
            out_data_list = ['svhn', 'imagenet_resize', 'lsun_resize']
            #out_data_list = ['svhn', 'imagenet_resize', 'lsun_resize', 'gaussian_noise']
            #out_data_list = ['svhn', 'imagenet_resize', 'lsun_resize', 'gaussian_noise','uniform_noise']
        elif data == 'svhn':
            out_data_list = ['cifar10', 'imagenet_resize', 'lsun_resize']
            #out_data_list = ['cifar10', 'imagenet_resize', 'lsun_resize', 'gaussian_noise']
            #out_data_list = ['cifar10', 'imagenet_resize', 'lsun_resize', 'gaussian_noise','uniform_noise']
        #elif data == 'imagenet32':
            ##out_data_list = ['svhn', 'cifar10', 'lsun_resize']
            #out_data_list = ['svhn', 'cifar10', 'lsun_resize', 'gaussian_noise','uniform_noise']
        for model in MODELS:
        #for model in ['densenetbc100']:
            for loss in LOSSES:
                base_path = 'data~'+data+'+model~'+model+'+loss~'+loss
                fig = plt.figure(figsize=(15, 3))
                #########################
                #ax = plt.subplot(1, 4, 1)
                #########################
                in_data_file = os.path.join(path, base_path, 'best_model1_valid_epoch_logits.npy')
                in_logits = np.load(in_data_file, allow_pickle=True).item()
                in_logits_intra = random.sample(in_logits['intra'], 10000)
                print(len(in_logits_intra))
                ############################
                #ax.hist(in_logits_intra, bins=bins, alpha=1.0, label='in-distribution intraclass', density=True)
                ############################
                #ax.axvline(statistics.mean(in_logits_intra), color='k', linestyle='dotted')
                in_logits_inter = random.sample(in_logits['inter'], 10000)
                print(len(in_logits_inter))
                ############################
                #ax.hist(in_logits_inter, bins=bins, alpha=1.0, label='in-distribution interclass', density=True)
                ############################
                #ax.axvline(statistics.mean(in_logits_inter), color='k', linestyle='dotted')
                #ax.legend(loc='upper right', prop={'size': 8})
                ###########################
                """
                if loss.startswith("sml"):
                    ax.set_xlabel('Logit (Affine Transformations)')
                else:
                    ax.set_xlabel('Logit (Distancies)')
                ax.set_ylabel('Density')
                ax.set_title('in-distribution: '+PRINT_DATA[data], loc='center')
                """
                ##############################
                for i, out_data in enumerate(out_data_list):
                    #############################
                    #ax = plt.subplot(1, 4, i + 2)
                    ax = plt.subplot(1, 3, i + 1)
                    #ax = plt.subplot(1, 4, i + 1)
                    if i == 0:
                        ax.set_ylabel('Density')
                    #############################
                    ax.hist(in_logits_intra, bins=bins, alpha=1.0, label='in-distribution intraclass', density=True)
                    #ax.axvline(statistics.mean(in_logits_intra), color='k', linestyle='dotted')
                    ax.hist(in_logits_inter, bins=bins, alpha=1.0, label='in-distribution interclass', density=True)
                    #ax.axvline(statistics.mean(in_logits_inter), color='k', linestyle='dotted')
                    out_data_file = os.path.join(path, base_path, 'best_model1_valid_epoch_logits_'+out_data +'.npy')
                    out_logits = np.load(out_data_file, allow_pickle=True).item()
                    out_logits = random.sample(out_logits['inter'], 10000)
                    print(len(out_logits))
                    ax.hist(out_logits, bins=bins, alpha=0.5, label='out-of-distribution', density=True, linewidth=0)
                    #ax.axvline(statistics.mean(out_logits), color='k', linestyle='dotted')
                    #ax.legend(loc='upper right', prop={'size': 8})
                    if loss.startswith("sml"):
                        ax.set_xlabel('Logit (Affine Transformation)')
                    else:
                        ax.set_xlabel('Logit (Distance)')
                    #if i == 0:
                    #    ax.set_ylabel('Quantity')
                    ax.set_title('out-of-distribution: '+PRINT_DATA[out_data], loc='center')
                #with sns.set_style('white'):
                #with sns.set(style="whitegrid"):
                with sns.axes_style("white"):
                    ####plt.legend(loc='lower center', bbox_to_anchor=(-1.35, -0.35), ncol=4)
                    #plt.legend(loc='upper right', bbox_to_anchor=(-0.2, -0.2), ncol=4)
                    ##################################
                    #plt.legend(loc='upper right', bbox_to_anchor=(0.3, -0.25), ncol=4, fontsize=13)
                    ##################################
                    plt.legend(loc='upper right', bbox_to_anchor=(0.35, -0.25), ncol=4, fontsize=13)
                    #plt.legend(loc='upper right', bbox_to_anchor=(-1.3, -0.2), ncol=4)
                #plt.suptitle("MODEL=" + PRINT_MODEL[model] + " | LOSS=" + PRINT_LOSS[loss])
                #g.fig.subplots_adjust(top=0.9)
                #g.fig.suptitle('Title')
                fig.subplots_adjust(top=0.83)
                #################################
                #fig.suptitle("MODEL=" + PRINT_MODEL[model] + " | LOSS=" + PRINT_LOSS[loss], fontsize=13)
                fig.suptitle("IN-DISTRIBUTION=" + PRINT_DATA[data] + " | MODEL=" + PRINT_MODEL[model] + " | LOSS=" + PRINT_LOSS[loss], fontsize=13)
                #################################
                plt.savefig(os.path.join(path, 'plot_histogram_logits+'+model+'+'+data+'+'+loss+'.png'), bbox_inches='tight', dpi=150)
                plt.close()

    for data in DATASETS:
    #for data in ['cifar10']:
        if data == 'cifar10':
            out_data_list = ['svhn', 'imagenet_resize', 'lsun_resize']
            #out_data_list = ['svhn', 'imagenet_resize', 'lsun_resize', 'gaussian_noise','uniform_noise']
        elif data == 'cifar100':
            out_data_list = ['svhn', 'imagenet_resize', 'lsun_resize']
            #out_data_list = ['svhn', 'imagenet_resize', 'lsun_resize', 'gaussian_noise','uniform_noise']
        elif data == 'svhn':
            out_data_list = ['cifar10', 'imagenet_resize', 'lsun_resize']
            #out_data_list = ['cifar10', 'imagenet_resize', 'lsun_resize', 'gaussian_noise','uniform_noise']
        #elif data == 'imagenet32':
            ##out_data_list = ['svhn', 'cifar10', 'lsun_resize']
            #out_data_list = ['svhn', 'cifar10', 'lsun_resize', 'gaussian_noise','uniform_noise']
        for model in MODELS:
            for loss in LOSSES:
                base_path = 'data~'+data+'+model~'+model+'+loss~'+loss
                fig = plt.figure(figsize=(15, 3))
                ###########################
                #ax = plt.subplot(1, 4, 1)
                ###########################
                in_data_file = os.path.join(path, base_path, 'best_model1_valid_epoch_metrics.npy')
                in_metrics = np.load(in_data_file, allow_pickle=True).item()
                in_max_probs = random.sample(in_metrics['max_probs'], 10000)
                print(len(in_max_probs))
                ###########################
                #ax.hist(in_max_probs, bins=bins, alpha=1.0, label='in-distribution', density=True)
                ###########################
                ###########################
                """
                ax.set_xlabel('Maximum Probability')
                ax.set_ylabel('Density')
                ax.set_title('in-distribution: '+PRINT_DATA[data], loc='center')
                """
                ############################
                for i, out_data in enumerate(out_data_list):
                    ###############################
                    #ax = plt.subplot(1, 4, i + 2)
                    ax = plt.subplot(1, 3, i + 1)
                    if i == 0:
                        ax.set_ylabel('Density')
                    ##############################
                    ax.hist(in_max_probs, bins=bins, alpha=1.0, label='in-distribution', density=True)
                    #if loss.startswith('dml'):
                    #    plt.axvline(0.2, color='r', linestyle='dashed')
                    out_data_file = os.path.join(path, base_path, 'best_model1_valid_epoch_metrics_'+out_data +'.npy')
                    out_metrics = np.load(out_data_file, allow_pickle=True).item()
                    out_max_probs = random.sample(out_metrics['max_probs'], 10000)
                    print(len(out_max_probs))
                    ax.hist(out_max_probs, bins=bins, alpha=0.5, label='out-of-distribution', density=True, linewidth=0)
                    #ax.legend(loc='upper right', prop={'size': 8})
                    ax.set_xlabel('Maximum Probability')
                    #if i == 0:
                    #    ax.set_ylabel('Quantity')
                    ax.set_title('out-of-distribution: '+PRINT_DATA[out_data], loc='center')
                #with sns.set_style('white'):
                #with sns.set(style="whitegrid"):
                with sns.axes_style("white"):
                    ####plt.legend(loc='lower center', bbox_to_anchor=(-1.35, -0.35), ncol=4)
                    #plt.legend(loc='upper right', bbox_to_anchor=(-0.5, -0.25), ncol=4, fontsize=13)
                    plt.legend(loc='upper right', bbox_to_anchor=(-0.1, -0.25), ncol=4, fontsize=13)
                    #plt.legend(loc='upper right', bbox_to_anchor=(-1.8, -0.2), ncol=4)
                #plt.suptitle(loss)
                fig.subplots_adjust(top=0.83)
                ################################
                #fig.suptitle("MODEL=" + PRINT_MODEL[model] + " | LOSS=" + PRINT_LOSS[loss], fontsize=13)
                fig.suptitle("IN-DISTRIBUTION=" + PRINT_DATA[data] + " | MODEL=" + PRINT_MODEL[model] + " | LOSS=" + PRINT_LOSS[loss], fontsize=13)
                ################################
                plt.savefig(os.path.join(path, 'plot_histogram_maxprobs+'+model+'+'+data+'+'+loss+'.png'), bbox_inches='tight', dpi=150)
                plt.close()

    for data in DATASETS:
    #for data in ['cifar10']:
        if data == 'cifar10':
            out_data_list = ['svhn', 'imagenet_resize', 'lsun_resize']
            #out_data_list = ['svhn', 'imagenet_resize', 'lsun_resize', 'gaussian_noise','uniform_noise']
            norm_factor = math.log(10)
        elif data == 'cifar100':
            out_data_list = ['svhn', 'imagenet_resize', 'lsun_resize']
            #out_data_list = ['svhn', 'imagenet_resize', 'lsun_resize', 'gaussian_noise','uniform_noise']
            norm_factor = math.log(100)
        elif data == 'svhn':
            out_data_list = ['cifar10', 'imagenet_resize', 'lsun_resize']
            #out_data_list = ['cifar10', 'imagenet_resize', 'lsun_resize', 'gaussian_noise','uniform_noise']
            norm_factor = math.log(10)
        #elif data == 'imagenet32':
            ##out_data_list = ['svhn', 'cifar10', 'lsun_resize']
            #out_data_list = ['svhn', 'cifar10', 'lsun_resize', 'gaussian_noise','uniform_noise']
        for model in MODELS:
            for loss in LOSSES:
                base_path = 'data~'+data+'+model~'+model+'+loss~'+loss
                fig = plt.figure(figsize=(15, 3))
                ################################
                #ax = plt.subplot(1, 4, 1)
                ################################
                in_data_file = os.path.join(path, base_path, 'best_model1_valid_epoch_metrics.npy')
                in_metrics = np.load(in_data_file, allow_pickle=True).item()
                #in_entropies = in_metrics['entropies']
                in_entropies = random.sample(in_metrics['entropies'], 10000)
                print(len(in_entropies))
                ################################
                #ax.hist([x/norm_factor for x in in_entropies], bins=bins, alpha=1.0, label='in-distribution', density=True)
                ################################
                #####################################
                """
                ax.set_xlabel('Normalized Entropy')
                ax.set_ylabel('Density')
                ax.set_title('in-distribution: '+PRINT_DATA[data], loc='center')
                """
                #####################################
                for i, out_data in enumerate(out_data_list):
                    #################################
                    #ax = plt.subplot(1, 4, i + 2)
                    ax = plt.subplot(1, 3, i + 1)
                    if i == 0:
                        ax.set_ylabel('Density')
                    ################################
                    ax.hist([x/norm_factor for x in in_entropies], bins=bins, alpha=1.0, label='in-distribution', density=True)
                    #if loss.startswith('dml'):
                    #    plt.axvline(0.95, color='r', linestyle='dashed')
                    out_data_file = os.path.join(path, base_path, 'best_model1_valid_epoch_metrics_'+out_data +'.npy')
                    out_metrics = np.load(out_data_file, allow_pickle=True).item()
                    out_entropies = random.sample(out_metrics['entropies'], 10000)
                    print(len(out_entropies))
                    ax.hist([x/norm_factor for x in out_entropies], bins=bins, alpha=0.5, label='out-of-distribution', density=True, linewidth=0)
                    #ax.legend(loc='upper right', prop={'size': 8})
                    ax.set_xlabel('Normalized Entropy')
                    #if i == 0:
                    #    ax.set_ylabel('Quantity')
                    ax.set_title('out-of-distribution: '+PRINT_DATA[out_data], loc='center')
                #with sns.set_style('white'):
                #with sns.set(style="whitegrid"):
                with sns.axes_style("white"):
                    ####plt.legend(loc='lower center', bbox_to_anchor=(-1.35, -0.35), ncol=4)
                    #plt.legend(loc='upper right', bbox_to_anchor=(-0.5, -0.25), ncol=4, fontsize=13)
                    plt.legend(loc='upper right', bbox_to_anchor=(-0.1, -0.25), ncol=4, fontsize=13)
                    #plt.legend(loc='upper right', bbox_to_anchor=(-2.0, -0.2), ncol=4)
                fig.subplots_adjust(top=0.83)
                #######################################
                #fig.suptitle("MODEL=" + PRINT_MODEL[model] + " | LOSS=" + PRINT_LOSS[loss], fontsize=13)
                fig.suptitle("IN-DISTRIBUTION=" + PRINT_DATA[data] + " | MODEL=" + PRINT_MODEL[model] + " | LOSS=" + PRINT_LOSS[loss], fontsize=13)
                #######################################
                plt.savefig(os.path.join(path, 'plot_histogram_entropies+'+model+'+'+data+'+'+loss+'.png'), bbox_inches='tight', dpi=150)
                plt.close()



    sns.set_context("paper", font_scale=1.6)
    MODELS = ['densenetbc100','resnet34']
    METRICS = ['VALID ACC1','MEAN AUROC','MEAN TNR']
    #METRICS = ['MEAN AUROC']
    #PAPERS = ['odd2']
    #PAPERS = ['odd1','odd2']
    DATA = ['svhn', 'cifar10','cifar100']        
    print("\n#####################################")
    print("####### FIGURES: ROBUSTNESS #########")
    print("#####################################")
    #print("PAPER:", paper)
    shift = -0.3
    for model in MODELS:
        print("MODEL:", model)
        for metric in METRICS:
            print("METRIC:", metric)
            if metric == 'VALID ACC1':
                df = pd.read_csv(os.path.join(path, 'extra_results_best.csv'))
                splited = df['DATA'].str.split("_", expand=True)
                df['DATA'] = splited[0]
                df['EXAMPLES_PER_CLASS'] = splited[1] 
                df['EXAMPLES_PER_CLASS'] = pd.to_numeric(df['EXAMPLES_PER_CLASS'])
                df = df[['MODEL','DATA','EXAMPLES_PER_CLASS','LOSS','VALID ACC1']]
                #print(df)
                fig = plt.figure(figsize=(15, 3))
                for i, in_data in enumerate(DATA):
                    print('IN-DATA:', in_data)
                    ax = plt.subplot(1, 3, i + 1)                   
                    dfx = df.loc[df['MODEL'].isin([model]) & df['DATA'].isin([in_data])]
                    dfx = dfx.sort_values(['MODEL','DATA','EXAMPLES_PER_CLASS','VALID ACC1'], ascending=True)
                    print(dfx)
                    dftemp = dfx.loc[dfx['LOSS'].isin(['sml1_na_id_no_no_no_no'])] 
                    ax.plot(dftemp['EXAMPLES_PER_CLASS'], dftemp['VALID ACC1'], label="SoftMax")
                    ##ax.plot([0,1,2,3,4], dftemp['VALID ACC1'], label="SoftMax")
                    #if paper == 'odd1':
                    dftemp = dfx.loc[dfx['LOSS'].isin(['dml10_pn2_id_no_no_no_no'])]
                    ax.plot(dftemp['EXAMPLES_PER_CLASS'], dftemp['VALID ACC1'], label="IsoMax")
                    ##ax.plot([0,1,2,3,4], dftemp['VALID ACC1'], label="IsoMax")
                    #"""
                    if in_data == 'cifar10':
                        #ax.set_ylim(70, 100)
                        ax.set_ylim(80, 100)
                    elif in_data == 'cifar100':
                        ax.set_ylim(50, 80)
                        #ax.set_ylim(50, 100)
                    elif in_data == 'svhn':
                        #ax.set_ylim(70, 100)
                        ax.set_ylim(90, 100)
                    #"""
                    ax.set_xlabel('Training Examples per Class')
                    if i == 0:
                        ax.set_ylabel('Classification [Accuracy] (%)')
                    ax.set_title('in-distribution: '+PRINT_DATA[in_data], loc='center')
                    #if in_data == 'cifar100':
                    #    ax.set_xticks([0,1,2,3,4],[25,50,100,200,400])
                    #else:
                    #    ax.set_xticks([0,1,2,3,4],[250,500,1000,2000,4000])
                    #ax.set_xscale("log")
                    #ax.grid(True)
                with sns.axes_style("white"):
                    #plt.legend(loc='upper right', bbox_to_anchor=(0.3, -0.25), ncol=4, fontsize=13)
                    #plt.legend(loc='upper right', bbox_to_anchor=(-0.3, -0.25), ncol=4, fontsize=13)
                    plt.legend(loc='upper right', bbox_to_anchor=(shift, -0.25), ncol=4, fontsize=13)
                fig.subplots_adjust(top=0.83)
                fig.suptitle("MODEL=" + PRINT_MODEL[model], fontsize=13)
                plt.savefig(os.path.join(path, 'plot_odd1_robustness_acc1_'+model+'.png'), bbox_inches='tight', dpi=150)
                plt.close()
            elif metric == 'MEAN AUROC':
                df = pd.read_csv(os.path.join(path, 'extra_results_odd.csv'))
                splited = df['IN-DATA'].str.split("_", expand=True)
                df['IN-DATA'] = splited[0]
                df['EXAMPLES_PER_CLASS'] = splited[1] 
                df['EXAMPLES_PER_CLASS'] = pd.to_numeric(df['EXAMPLES_PER_CLASS'])
                df = df[['MODEL','IN-DATA','SCORE','EXAMPLES_PER_CLASS','LOSS','OUT-DATA','AUROC']]
                #print(df)
                fig = plt.figure(figsize=(15, 3))
                for i, in_data in enumerate(DATA):
                    #if in_data == 'cifar100':
                    #    in_data = 'cifar10'
                    print('IN-DATA:', in_data)
                    ax = plt.subplot(1, 3, i + 1)                   
                    dfx = df.loc[
                        df['MODEL'].isin([model]) &
                        df['IN-DATA'].isin([in_data]) &
                        df['SCORE'].isin(['NE']) &
                        df['OUT-DATA'].isin(['imagenet_resize','lsun_resize','cifar10','svhn'])
                        ]                        
                    dfx = dfx.groupby(['SCORE','EXAMPLES_PER_CLASS','LOSS'], as_index=False)['AUROC'].mean()
                    dfx = dfx.sort_values(['SCORE','EXAMPLES_PER_CLASS','LOSS'], ascending=True)
                    print(dfx)
                    dftemp = dfx.loc[dfx['LOSS'].isin(['sml1_na_id_no_no_no_no'])] 
                    ax.plot(dftemp['EXAMPLES_PER_CLASS'], dftemp['AUROC'], label="SoftMax")
                    #ax.plot([0,1,2,3], dftemp['AUROC'], label="SoftMax")
                    #if paper == 'odd1':
                    dftemp = dfx.loc[dfx['LOSS'].isin(['dml10_pn2_id_no_no_no_no'])]
                    ax.plot(dftemp['EXAMPLES_PER_CLASS'], dftemp['AUROC'], label="IsoMax")
                    #ax.plot([0,1,2,3], dftemp['AUROC'], label="IsoMax")
                    """
                    if in_data == 'cifar10':
                        if model == 'densenetbc100':
                            ax.set_ylim(80, 100)
                        elif model == 'resnet34':
                            ax.set_ylim(75, 100)
                    elif in_data == 'cifar100':
                        ax.set_ylim(60, 100)
                        #if model == 'densenetbc100':
                        #    ax.set_ylim(60, 100)
                        #elif model == 'resnet34':
                        #    ax.set_ylim(60, 100)
                    elif in_data == 'svhn':
                        ax.set_ylim(90, 100)
                        #if model == 'densenetbc100':
                        #    ax.set_ylim(90, 100)
                        #elif model == 'resnet34':
                        #    ax.set_ylim(92, 98)
                    """
                    ax.set_xlabel('Training Examples per Class')
                    if i == 0:
                        #ax.set_ylabel(r'OOD Detection [$\overline{\mathrm{AUROC}}$] (%)')
                        ax.set_ylabel(r'Detection [$\overline{\mathrm{AUROC}}$] (%)')
                    ax.set_title('in-distribution: '+PRINT_DATA[in_data], loc='center')
                    #if in_data == 'cifar100':
                    #    ax.set_xticks([0,1,2,3],[50,100,250,500])
                    #else:
                    #    ax.set_xticks([0,1,2,3],[500,1000,2500,5000])
                    #ax.set_xscale("log")
                    #ax.grid(True)
                with sns.axes_style("white"):
                    #plt.legend(loc='upper right', bbox_to_anchor=(0.3, -0.25), ncol=4, fontsize=13)
                    #plt.legend(loc='upper right', bbox_to_anchor=(-0.3, -0.25), ncol=4, fontsize=13)
                    plt.legend(loc='upper right', bbox_to_anchor=(shift, -0.25), ncol=4, fontsize=13)
                fig.subplots_adjust(top=0.83)
                fig.suptitle("MODEL=" + PRINT_MODEL[model], fontsize=13)
                plt.savefig(os.path.join(path, 'plot_odd1_robustness_auroc_'+model+'.png'), bbox_inches='tight', dpi=150)
                plt.close()
            elif metric == 'MEAN TNR':
                df = pd.read_csv(os.path.join(path, 'extra_results_odd.csv'))
                splited = df['IN-DATA'].str.split("_", expand=True)
                df['IN-DATA'] = splited[0]
                df['EXAMPLES_PER_CLASS'] = splited[1] 
                df['EXAMPLES_PER_CLASS'] = pd.to_numeric(df['EXAMPLES_PER_CLASS'])
                df = df[['MODEL','IN-DATA','SCORE','EXAMPLES_PER_CLASS','LOSS','OUT-DATA','TNR']]
                #print(df)
                fig = plt.figure(figsize=(15, 3))
                for i, in_data in enumerate(DATA):
                    #if in_data == 'cifar100':
                    #    in_data = 'cifar10'
                    print('IN-DATA:', in_data)
                    ax = plt.subplot(1, 3, i + 1)                   
                    dfx = df.loc[
                        df['MODEL'].isin([model]) &
                        df['IN-DATA'].isin([in_data]) &
                        df['SCORE'].isin(['NE']) &
                        df['OUT-DATA'].isin(['imagenet_resize','lsun_resize','cifar10','svhn'])
                        ]                        
                    dfx = dfx.groupby(['SCORE','EXAMPLES_PER_CLASS','LOSS'], as_index=False)['TNR'].mean()
                    dfx = dfx.sort_values(['SCORE','EXAMPLES_PER_CLASS','LOSS'], ascending=True)
                    print(dfx)
                    dftemp = dfx.loc[dfx['LOSS'].isin(['sml1_na_id_no_no_no_no'])] 
                    ax.plot(dftemp['EXAMPLES_PER_CLASS'], dftemp['TNR'], label="SoftMax")
                    #ax.plot([0,1,2,3], dftemp['TNR'], label="SoftMax")
                    #if paper == 'odd1':
                    dftemp = dfx.loc[dfx['LOSS'].isin(['dml10_pn2_id_no_no_no_no'])]
                    ax.plot(dftemp['EXAMPLES_PER_CLASS'], dftemp['TNR'], label="IsoMax")
                    #ax.plot([0,1,2,3], dftemp['TNR'], label="IsoMax")
                    """
                    if in_data == 'cifar10':
                        if model == 'densenetbc100':
                            ax.set_ylim(20, 90)
                        elif model == 'resnet34':
                            ax.set_ylim(20, 80)
                    elif in_data == 'cifar100':
                        ax.set_ylim(0, 50)
                        #if model == 'densenetbc100':
                        #    ax.set_ylim(0, 100)
                        #elif model == 'resnet34':
                        #    ax.set_ylim(0, 100)
                    elif in_data == 'svhn':
                        ax.set_ylim(50, 100)
                        #if model == 'densenetbc100':
                        #    ax.set_ylim(0, 100)
                        #elif model == 'resnet34':
                        #    ax.set_ylim(0, 100)
                    """
                    ax.set_xlabel('Training Examples per Class')
                    if i == 0:
                        #ax.set_ylabel(r'OOD Detection [$\overline{\mathrm{TNR@TPR95}}$] (%)')
                        ax.set_ylabel(r'Detection [$\overline{\mathrm{TNR@TPR95}}$] (%)')
                    ax.set_title('in-distribution: '+PRINT_DATA[in_data], loc='center')
                    #if in_data == 'cifar100':
                    #    ax.set_xticks([0,1,2,3],[50,100,250,500])
                    #else:
                    #    ax.set_xticks([0,1,2,3],[500,1000,2500,5000])
                    #ax.set_xscale("log")
                    #ax.grid(True)
                with sns.axes_style("white"):
                    #plt.legend(loc='upper right', bbox_to_anchor=(0.3, -0.25), ncol=4, fontsize=13)
                    #plt.legend(loc='upper right', bbox_to_anchor=(-0.3, -0.25), ncol=4, fontsize=13)
                    plt.legend(loc='upper right', bbox_to_anchor=(shift, -0.25), ncol=4, fontsize=13)
                fig.subplots_adjust(top=0.83)
                fig.suptitle("MODEL=" + PRINT_MODEL[model], fontsize=13)
                plt.savefig(os.path.join(path, 'plot_odd1_robustness_tnr_'+model+'.png'), bbox_inches='tight', dpi=150)
                plt.close()



if __name__ == '__main__':
    main()

    """
    print("\n#####################################")
    print("####### FIGURE: T-SNE PLOTS #########")
    print("#####################################")
    # basic load block to get logits and targets tensors for the  trainset...
    perplexities = [1000]
    #perplexities = [2, 5, 30, 50, 100, 500, 2000, 5000]
    for data in DATASETS:
        for model in MODELS:
            for loss in LOSSES:
                for perplexity in perplexities:
                    print("\n########")
                    file = os.path.join(path, 'data~'+data+'+model~'+model+'+loss~'+loss, 'model1_valset.pth')

                    trainset_first_partition_features = torch.load(file)
                    trainset_first_partition_features_logits = trainset_first_partition_features[1]
                    trainset_first_partition_features_logits_numpy = np.asarray(trainset_first_partition_features_logits)
                    print(trainset_first_partition_features_logits_numpy.shape)
                    trainset_first_partition_features_targets = trainset_first_partition_features[2].long()
                    trainset_first_partition_features_targets_numpy = np.asarray(trainset_first_partition_features_targets)
                    print(trainset_first_partition_features_targets_numpy.shape)

                    time_start = time.time()
                    tsne = TSNE(
                        n_components=2, perplexity=perplexity, random_state=1234, verbose=10, n_iter=5000, learning_rate=200,
                    ).fit_transform(trainset_first_partition_features_logits_numpy)
                    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

                    scatter(tsne, trainset_first_partition_features_targets_numpy)
                    plt.savefig(os.path.join(
                        path, data+'_'+model+'_'+loss.replace('.', ',')+'_TSNE_'+str(perplexity)), bbox_inches='tight', dpi=150)
                    plt.close()
                    print("\n########")
    """

    """
    #for paper in ['odd1','odd2']:
    #if paper == 'odd1':
    #    LOSSES = ['sml1_na_id_no_no_no_no', 'dml10_pn2_id_no_no_no_no']
    #    LOSSES_TEXTS = [r"SoftMax", r"IsoMax"]
    #elif paper == 'odd2':
    #    LOSSES = ['sml1_na_id_no_no_no_no', 'dml10_pn2_id_no_no_no_no', 'eml1_pn2_id_no_no_lz0_10_ST_NO_0.01']
    #    LOSSES_TEXTS = [r"SoftMax", r"IsoMax", r"IsoMax$_2$"]
    #LOSSES = ['sml1_na_id_no_no_no_no', 'dml10_pn2_id_no_no_no_no']
    #LOSSES_TEXTS = [r"SoftMax", r"IsoMax"]
    #####################################################
    #####################################################
    #####################################################
    sns.set_context("paper", font_scale=1.8)
    print("\n#####################################")
    print("### FIGURE: ALL VALID ACCURACIES ####")
    print("#####################################")
    df = raw_results_data_frame.loc[
        raw_results_data_frame['LOSS'].isin(LOSSES) &
        raw_results_data_frame['SET'].isin(['VALID']) &
        raw_results_data_frame['METRIC'].isin(['ACC1'])# &
        #raw_results_data_frame['EXECUTION'].isin(['1'])
        ]
    #print(df)
    df = df.replace({'cifar10': 'CIFAR10', 'cifar100': 'CIFAR100', 'svhn': 'SVHN'})
    df = df.replace({'densenetbc100': 'DenseNet', 'resnet34': 'ResNet'})
    #print(df)
    #plt.figure(figsize=(15, 1))
    #sns.catplot("EPOCH", "VALUE", "LOSS", df, kind="point", size=1, aspect=5)
    grid = sns.FacetGrid(
        df,
        #palette=sns.color_palette("BrBG", 2),
        #col="DATA", col_order=['cifar10', 'cifar100', 'svhn'],
        col="DATA", col_order=['SVHN', 'CIFAR10', 'CIFAR100'],
        #row="MODEL", row_order=MODELS,
        row="MODEL", row_order=['DenseNet', 'ResNet'],
        hue="LOSS", hue_order=LOSSES,
        #margin_titles=True,
        aspect=2#1.5
    )#.set_titles('{col_name}')
    grid.map(plt.plot, "EPOCH", "VALUE", marker="")
    print(LOSSES)
    print(LOSSES_TEXTS)
    #grid.set(xlabel="xxxxx")
    #grid.set(ylabel="yyyyy")
    grid.set(ylim=(50, 100))
    grid.axes[0, 0].set_ylabel('Accuracy (Test)')
    grid.axes[1, 0].set_ylabel('Accuracy (Test)')
    grid.axes[1, 0].set_xlabel('Epoch')
    grid.axes[1, 1].set_xlabel('Epoch')
    grid.axes[1, 2].set_xlabel('Epoch')
    #grid.add_legend()
    #grid._legend.set_title("Loss")
    #for t, l in zip(grid._legend.texts, LOSSES_TEXTS):
    #    t.set_text(l)
    shift = -0.25
    with sns.axes_style("white"):
        #for t, l in zip(plt.legend(loc='upper right', bbox_to_anchor=(-0.35, -0.2), ncol=5, title="Loss").get_texts(), LOSSES_TEXTS):
        #for t, l in zip(plt.legend(loc='upper right', bbox_to_anchor=(-0.3, -0.2), ncol=5).get_texts(), LOSSES_TEXTS):
        #for t, l in zip(plt.legend(loc='upper right', bbox_to_anchor=(0.0, -0.2), ncol=5).get_texts(), LOSSES_TEXTS):
        #for t, l in zip(plt.legend(loc='upper right', bbox_to_anchor=(shift, -0.2), ncol=5).get_texts(), LOSSES_TEXTS):
        for t, l in zip(plt.legend(loc='upper right', bbox_to_anchor=(shift, -0.325), ncol=5).get_texts(), LOSSES_TEXTS):
            print(t,l)
            t.set_text(l)
    plt.savefig(os.path.join(path, 'plot_odd1_all_test_accuracies'), bbox_inches='tight', dpi=150)
    #######################
    plt.close() #### NEW!!!
    #######################


    sns.set_context("paper", font_scale=1.8)
    print("\n#####################################")
    print("##### FIGURE: ALL TRAIN LOSSES ######")
    print("#####################################")
    df = raw_results_data_frame.loc[
        raw_results_data_frame['LOSS'].isin(LOSSES) &
        raw_results_data_frame['SET'].isin(['TRAIN']) &
        raw_results_data_frame['METRIC'].isin(['LOSS'])# &
        #raw_results_data_frame['EXECUTION'].isin(['1'])
        ]
    #print(df)
    df = df.replace({'cifar10': 'CIFAR10', 'cifar100': 'CIFAR100', 'svhn': 'SVHN'})
    df = df.replace({'densenetbc100': 'DenseNet', 'resnet34': 'ResNet'})
    #print(df)
    #plt.figure(figsize=(15, 5))
    grid = sns.FacetGrid(
        df,
        #palette=sns.color_palette("BrBG", 2),
        #col="DATA", col_order=['cifar10', 'cifar100', 'svhn'],
        col="DATA", col_order=['SVHN', 'CIFAR10', 'CIFAR100'],
        #row="MODEL", row_order=MODELS,
        row="MODEL", row_order=['DenseNet', 'ResNet'],
        hue="LOSS", hue_order=LOSSES,
        #margin_titles=True,
        #aspect=2#1.5
    )#.set_titles('{col_name}')
    grid.map(plt.plot, "EPOCH", "VALUE", marker="")
    print(LOSSES)
    print(LOSSES_TEXTS)
    #grid.set(xlabel="xxxxx")
    #grid.set(ylabel="yyyyy")
    grid.set(ylim=(0, 1))
    grid.axes[0, 0].set_ylabel('Loss (Train)')
    grid.axes[1, 0].set_ylabel('Loss (Train)')
    grid.axes[1, 0].set_xlabel('Epoch')
    grid.axes[1, 1].set_xlabel('Epoch')
    grid.axes[1, 2].set_xlabel('Epoch')
    #grid.add_legend()
    #grid._legend.set_title("Loss")
    #for t, l in zip(grid._legend.texts, LOSSES_TEXTS):
    #    t.set_text(l)
    shift = -0.25
    with sns.axes_style("white"):
        #for t, l in zip(plt.legend(loc='upper right', bbox_to_anchor=(-0.35, -0.2), ncol=5, title="Loss").get_texts(), LOSSES_TEXTS):
        #for t, l in zip(plt.legend(loc='upper right', bbox_to_anchor=(-0.3, -0.2), ncol=5).get_texts(), LOSSES_TEXTS):
        #for t, l in zip(plt.legend(loc='upper right', bbox_to_anchor=(0.0, -0.2), ncol=5).get_texts(), LOSSES_TEXTS):
        #for t, l in zip(plt.legend(loc='upper right', bbox_to_anchor=(shift, -0.2), ncol=5).get_texts(), LOSSES_TEXTS):
        for t, l in zip(plt.legend(loc='upper right', bbox_to_anchor=(shift, -0.325), ncol=5).get_texts(), LOSSES_TEXTS):
            print(t,l)
            t.set_text(l)
    plt.savefig(os.path.join(path, 'plot_odd1_all_train_losses'), bbox_inches='tight', dpi=150)
    #######################
    plt.close() #### NEW!!!
    #######################
    #####################################################
    #####################################################
    #####################################################
    #LOSSES = ['sml1_na_id_no_no_no_no', 'dml10_pn2_id_no_no_no_no', 'eml1_pn2_id_no_no_lz0_10_ST_NO_0.01']
    #LOSSES_TEXTS = [r"SoftMax", r"IsoMax", r"IsoMax$_2$"]
    """

    """
    print("\n#####################################")
    print("###### ENTROPIES PER CLASS ##########")
    print("#####################################")
    for data in DATASETS:
    #for data in ['cifar100']:
        for model in MODELS:
        #for model in ['densenetbc100']:
            for loss in LOSSES:
            #for loss in ['eml1_pn2_id_no_no_lz0_10_ST_NO_0.1']:
                print("DATASET:", data)
                print("MODEL:", model)
                print("LOSS:", loss)
                #base_path = 'data~cifar10+model~densenetbc100+loss~eml1_pn2_id_no_no_lz0_10_ST_NO_0.1'
                base_path = 'data~'+data+'+model~'+model+'+loss~'+loss
                in_data_file = os.path.join(path, base_path, 'best_model1_valid_epoch_entropies_per_classes.pkl')
                with open(in_data_file, "rb") as file:
                    testando = pickle.load(file)
                    #print(type(testando))
                    #print(len(testando))
                    #print(type(testando[0]))
                    #print(len(testando[0]))
                    means = []
                    pstds = []
                    for i in range(len(testando)):
                        #print(statistics.mean(testando[i]))
                        #print(statistics.pstdev(testando[i]))
                        means.append(statistics.mean(testando[i]))
                        pstds.append(statistics.pstdev(testando[i]))
                    print("Mean of Means",statistics.mean(means))
                    print("PStd of Means",statistics.pstdev(means))
                    print("Mean of PStds",statistics.mean(pstds))
                    print("PStd of PStds",statistics.pstdev(pstds),"\n")

    print("\n#####################################")
    print("### FIGURES: ENTROPIES BOXPLOTS #####")
    print("#####################################")
    #print(raw_results_data_frame)
    for data in DATASETS:
    #for data in ['cifar10']:
        for model in MODELS:
        #for model in ['densenetbc100']:
            for loss in LOSSES:
            #for loss in ['eml1_pn2_id_no_no_lz0_10_ST_NO_0.1']:
                base_path = 'data~'+data+'+model~'+model+'+loss~'+loss
                in_data_file = os.path.join(path, base_path, 'best_model1_valid_epoch_metrics.npy')
                in_metrics = np.load(in_data_file, allow_pickle=True).item()
                in_entropies = in_metrics['entropies']
                df1 = pd.DataFrame(data=in_entropies, columns=["VALUE"])
                df1['DATA'] = data
                df1['LOSS'] = r'IsoMax'
                df2 = pd.DataFrame(data=in_entropies, columns=["VALUE"])
                df2['DATA'] = data
                df2['LOSS'] = r'IsoMax2'
                df3 = pd.DataFrame(data=in_entropies, columns=["VALUE"])
                df3['DATA'] = 'cifar100'
                df3['LOSS'] = r'IsoMax'
                df4 = pd.DataFrame(data=in_entropies, columns=["VALUE"])
                df4['DATA'] = 'cifar100'
                df4['LOSS'] = r'IsoMax2'
                df5 = pd.DataFrame(data=in_entropies, columns=["VALUE"])
                df5['DATA'] = 'svhn'
                df5['LOSS'] = r'IsoMax'
                df6 = pd.DataFrame(data=in_entropies, columns=["VALUE"])
                df6['DATA'] = 'svhn'
                df6['LOSS'] = r'IsoMax2'
                df = pd.concat([df1,df2,df3,df4,df5,df6])
                print(df)
                #plt.figure(figsize=(3,3))
                #ax = sns.boxplot(data=[in_entropies,in_entropies])
                ax = sns.boxplot(data=df, y='VALUE', x='DATA', hue='LOSS')
                #ax.set_ylim(top=3)
                #labels = [item.get_text() for item in ax.get_xticklabels()]
                #labels[0] = r'IsoMax'
                #labels[1] = r'IsoMax2'
                #ax.set_xticklabels(labels)
                plt.legend(loc='upper right')
                plt.savefig(os.path.join(path, 'entropy_boxplots_'+data+'_'+model+'.png'), bbox_inches='tight', dpi=150)
                plt.close()
    """
