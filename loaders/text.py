import os
import random
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
#import torch.utils.data as data
from torch.utils.data.dataset import random_split
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
        #self.config = self.args.text_config
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

    def get_pandas_df(self, filename, original):
        '''
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
        '''
        with open(filename, 'r') as datafile:     
            data = [line.strip().split(',', maxsplit=1) for line in datafile]

            if original:
                data_text = list(map(lambda x: x[1], data))
                data_label = list(map(lambda x: self.parse_label(x[0]), data))
            else:
                data_text = list(map(lambda x: x[1][1:-1], data))
                data_label = list(map(lambda x: int(x[0][1:-1]), data))

        full_df = pd.DataFrame({"text":data_text, "label":data_label})
        return full_df
    
    #def load_data(self, w2v_file, train_file, test_file, val_file=None):
    def load_data(self, train_file, test_file):
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
        #TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len, batch_first=False)
        #LABEL = data.Field(sequential=False, use_vocab=False)
        ##TEXT = data.Field(sequential=True, tokenize='spacy', lower=True, fix_length=self.config.max_sen_len)
        TEXT = data.Field(
            #sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len,
            #sequential=True, tokenize=tokenizer, lower=True, fix_length=self.args.max_sen_len, include_lengths=False, batch_first=False)
            sequential=True, tokenize=tokenizer, lower=True, fix_length=None, include_lengths=False, batch_first=False)
        LABEL = data.Field(sequential=False)

        ##################################################################################################
        #"""
        datafields = [("text",TEXT),("label",LABEL)]
        # Load data from pd.DataFrame into torchtext.data.Dataset
        #train_df = self.get_pandas_df(train_file, True)
        train_df = self.get_pandas_df(train_file, False)
        print(train_df)
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)
        ####train_data, _ = random_split(train_data, [1000, len(train_data) - 1000])
        
        #test_df = self.get_pandas_df(test_file, True)
        test_df = self.get_pandas_df(test_file, False)
        print(test_df)
        test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, datafields)
        #"""
        ##################################################################################################
        
        #DATASET = datasets.AG_NEWS()
        #train_data, test_data = datasets.AG_NEWS.splits(TEXT, LABEL)
        #train_data, test_data = datasets.AG_NEWS()
        ###############################################train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
        ####train_data, test_data = datasets.YelpReviewFull.iters(batch_size=4)
        #train_data, test_data = datasets.YelpReviewFull.splits(TEXT, LABEL)
        
        ##################################################################################################
        # If validation file exists, load it. Otherwise get validation data from training data
        """
        if val_file:
            val_df = self.get_pandas_df(val_file)
            val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8)
        """
        ##################################################################################################
        
        #TEXT.build_vocab(train_data, vectors=Vectors(w2v_file))
        TEXT.build_vocab(train_data, vectors=GloVe(name='840B', dim=300))
        LABEL.build_vocab(train_data)
        self.word_embeddings = TEXT.vocab.vectors
        self.vocab = TEXT.vocab

        ##################################################################################################
        self.train_iterator = data.BucketIterator(
            (train_data), batch_size=self.args.batch_size, sort_key=lambda x: len(x.text), train=True, repeat=False, sort=True, shuffle=False)
        
        self.test_iterator = data.BucketIterator(
            (test_data), batch_size=self.args.batch_size, sort_key=lambda x: len(x.text), train=False, repeat=False, sort=True, shuffle=False)
        ##################################################################################################

        print ("Loaded {} training examples".format(len(train_data)))
        print ("Loaded {} test examples".format(len(test_data)))
        #print ("Loaded {} validation examples".format(len(val_data)))


class TextLoader:
    def __init__(self, args):

        self.args = args

        if self.args.dataset == "agnews":
            self.args.iteractions_per_epoch = 2000
            #train_file = 'data/agnews/ag_news.train'
            #test_file = 'data/agnews/ag_news.test'  
            train_file = 'data/ag_news_csv/train.csv'
            test_file = 'data/ag_news_csv/test.csv'  

        if self.args.dataset == "yelprf":
            self.args.iteractions_per_epoch = 10000
            #train_file = 'data/yelp_review_full_csv/train.csv'
            #test_file = 'data/yelp_review_full_csv/test.csv'  
            train_file = '/mnt/ssd/yelp_review_full_csv/train.csv'
            test_file = '/mnt/ssd/yelp_review_full_csv/test.csv'  

        elif self.args.dataset == "yahooa":
            #train_file = 'data/yahoo_answers_csv/train.csv'
            #test_file = 'data/yahoo_answers_csv/test.csv'
            train_file = '/mnt/ssd/yahoo_answers_csv/train.csv'
            test_file = '/mnt/ssd/yahoo_answers_csv/test.csv'

        #elif self.args.dataset == "amazonrf":
        #    #self.train_set, self.test_set = datasets.YelpReviewFull()
        #    train_file = 'data/amazon_review_full_csv/train.csv'
        #    test_file = 'data/amazon_review_full_csv/test.csv'  

        self.args.text_dataset = Dataset(self.args)
        #self.args.text_dataset.load_data(w2v_file, train_file, test_file)
        self.args.text_dataset.load_data(train_file, test_file)
    


    def get_loaders(self):

        return self.args.text_dataset.train_iterator, None, None, None, self.args.text_dataset.test_iterator, None


    def _worker_init(self, worker_id):
        random.seed(self.args.base_seed)


    def generate_batch(self, batch):
        r"""
        Since the text entries have different lengths, a custom function
        generate_batch() is used to generate data batches and offsets,
        which are compatible with EmbeddingBag. The function is passed
        to 'collate_fn' in torch.utils.data.DataLoader. The input to
        'collate_fn' is a list of tensors with the size of batch_size,
        and the 'collate_fn' function packs them into a mini-batch.
        Pay attention here and make sure that 'collate_fn' is declared
        as a top level def. This ensures that the function is available
        in each worker.

        Output:
            text: the text entries in the data_batch are packed into a list and
                concatenated as a single tensor for the input of nn.EmbeddingBag.
            offsets: the offsets is a tensor of delimiters to represent the beginning
                index of the individual sequence in the text tensor.
            cls: a tensor saving the labels of individual text entries.
        """
        """
        label = torch.tensor([entry[0] for entry in batch])
        text = [entry[1] for entry in batch]
        offsets = [0] + [len(entry) for entry in text]
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text = torch.cat(text)
        return text, offsets, label
        """
        label = torch.tensor([entry[0] for entry in batch])
        #text = [entry[1] for entry in batch]
        max_len = max([len(entry[1]) for entry in batch])
        text = [torch.tensor(entry[1].tolist() + [0] * (max_len - len(entry[1]))) for entry in batch]
        text = torch.stack(text, 0).permute(1,0)
        return text, label


"""
from torchtext.experimental.datasets import AG_NEWS
train, test = AG_NEWS(ngrams=3)
"""

"""
from torch.utils.data import DataLoader
def collate_fn(batch):
    texts, labels = [], []
    for label, txt in batch:
        texts.append(txt)
        labels.append(label)
    return texts, labels
dataloader = DataLoader(train, batch_size=8, collate_fn=collate_fn)
for idx, (texts, labels) in enumerate(dataloader):
    print(idx, texts, labels)
"""

"""
self.train_set, self.test_set = datasets.YelpReviewFull()
# split train_dataset into train and valid
#train_len = int(len(self.train_set) * 0.95)
#self.train_set_less_valid, self.valid_set = random_split(self.train_set, [train_len, len(self.train_set) - train_len])
"""

"""
if self.args.dataset in ["old"]:
    self.train_loader = DataLoader(
        self.train_set, batch_size=self.args.batch_size, shuffle=True,
        collate_fn=self.generate_batch, num_workers=self.args.workers, worker_init_fn=self._worker_init)
    self.test_loader = DataLoader(
        self.test_set, batch_size=self.args.batch_size,
        collate_fn=self.generate_batch, worker_init_fn=self._worker_init)

    self.train_loader = data.BucketIterator(
        (self.train_set), batch_size=self.args.batch_size, sort_key=lambda x: len(x.text), repeat=False, shuffle=True,)
        #sort=False, sort_within_batch=True) # repeat???
    
    self.test_loader = data.BucketIterator(
        (self.test_set), batch_size=self.args.batch_size, sort_key=lambda x: len(x.text), repeat=False, shuffle=False,)
        #sort=False, sort_within_batch=True) # repeat???

    return self.train_loader, None, None, None, self.test_loader, None
"""
