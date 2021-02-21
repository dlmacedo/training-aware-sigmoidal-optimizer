import os
import random
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
#import torch.utils.data as data
import torchtext.data as data
import torchtext.datasets as datasets
from torchtext.vocab import GloVe
from torchtext.vocab import Vectors
import numpy as np
import torch
from PIL import Image
from PIL.Image import BICUBIC
import pandas as pd
import spacy

class Dataset(object):
    def __init__(self, args):
        self.args = args
        self.config = self.args.text_config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}
    
    def parse_label(self, label):
        '''
        Get the actual labels from label string
        Input:
            label (string) : labels of the form '__label__2'
        Returns:
            label (int) : integer value corresponding to label string
        '''
        return int(label.strip()[-1])

    def get_pandas_df(self, filename):
        '''
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
        '''
        with open(filename, 'r') as datafile:     
            data = [line.strip().split(',', maxsplit=1) for line in datafile]
            data_text = list(map(lambda x: x[1], data))
            data_label = list(map(lambda x: self.parse_label(x[0]), data))

        full_df = pd.DataFrame({"text":data_text, "label":data_label})
        return full_df
    
    def load_data(self, w2v_file, train_file, test_file, val_file=None):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data
        
        Inputs:
            w2v_file (String): absolute path to file containing word embeddings (GloVe/Word2Vec)
            train_file (String): absolute path to training file
            test_file (String): absolute path to test file
            val_file (String): absolute path to validation file
        '''

        ############################
        #NLP = spacy.load('en')
        #NLP = spacy.load('en_core_web_sm')
        NLP = spacy.load('en_core_web_trf')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
        ############################

        # Creating Field for data
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        #TEXT = data.Field(sequential=True, tokenize='spacy', lower=True, fix_length=self.config.max_sen_len)
        LABEL = data.Field(sequential=False, use_vocab=False)
        
        ##################################################################################################
        datafields = [("text",TEXT),("label",LABEL)]
        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)
        
        test_df = self.get_pandas_df(test_file)
        test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, datafields)
        ##################################################################################################
        
        #train_data, test_data = datasets.AG_NEWS.splits(TEXT, LABEL)
        ####train_data, test_data = datasets.YelpReviewFull.iters(batch_size=4)
        #train_data, test_data = datasets.YelpReviewFull.splits(TEXT, LABEL)
        
        ##################################################################################################
        # If validation file exists, load it. Otherwise get validation data from training data
        if val_file:
            val_df = self.get_pandas_df(val_file)
            val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8)
        ##################################################################################################
        
        TEXT.build_vocab(train_data, vectors=Vectors(w2v_file))
        #TEXT.build_vocab(train_data, vectors=GloVe(name='840B', dim=300))
        #LABEL.build_vocab(train_data)
        self.word_embeddings = TEXT.vocab.vectors
        self.vocab = TEXT.vocab

        ##################################################################################################
        self.train_iterator = data.BucketIterator(
            (train_data),
            #batch_size=self.config.batch_size,
            batch_size=self.args.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)
        
        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            #batch_size=self.config.batch_size,
            batch_size=self.args.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)
        ##################################################################################################

        # make iterator for splits
        #self.train_iterator, self.test_iterator = data.BucketIterator.splits(
        #    (train_data, test_data),
        #    batch_size=self.args.batch_size,
        #    sort_key=lambda x: len(x.text),
        #    repeat=False,
        #    shuffle=False)
        #    #device=0)

        print ("Loaded {} training examples".format(len(train_data)))
        print ("Loaded {} test examples".format(len(test_data)))
        print ("Loaded {} validation examples".format(len(val_data)))


"""
def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for idx,batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
        else:
            x = batch.text
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score
"""

class TextLoader:
    def __init__(self, args):

        self.args = args

        if args.dataset == "agnews":
            train_file = 'data/agnews/ag_news.train'
            test_file = 'data/agnews/ag_news.test'  
            w2v_file = 'data/agnews/glove.840B.300d.txt'
            self.args.text_dataset = Dataset(self.args)
            self.args.text_dataset.load_data(w2v_file, train_file, test_file)
            #text_dataset precisa estar no nivel args???
            #text_dataset precisa estar no nivel args???

            """
            self.normalize = transforms.Normalize((0.491, 0.482, 0.446), (0.247, 0.243, 0.261))
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
            self.inference_transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])
            self.dataset_path = "data/cifar10"
            self.trainset_for_train = torchvision.datasets.CIFAR10(
                root=self.dataset_path, train=True, download=True, transform=self.train_transform)
            self.trainset_for_infer = torchvision.datasets.CIFAR10(
                root=self.dataset_path, train=True, download=True, transform=self.inference_transform)
            self.val_set = torchvision.datasets.CIFAR10(
                root=self.dataset_path, train=False, download=True, transform=self.inference_transform)
            """



    def get_loaders(self):



        return (self.args.text_dataset.train_iterator, None,
                self.args.text_dataset.train_iterator, None,
                self.args.text_dataset.test_iterator, None)


    def _worker_init(self, worker_id):
        random.seed(self.args.base_seed)
