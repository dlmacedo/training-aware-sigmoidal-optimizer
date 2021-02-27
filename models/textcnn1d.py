# https://www.aclweb.org/anthology/D14-1181.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
TextCNN1D
attributes:
    n_classes: number of classes
    vocab_size: number of words in the vocabulary of the model
    embeddings: word embedding weights
    emb_size: size of word embeddings
    fine_tune: allow fine-tuning of embedding layer?
               (only makes sense when using pre-trained embeddings)
    n_kernels: number of kernels
    kernel_sizes (list): size of each kernel
    dropout: dropout
    n_channels: number of channels (1 / 2)
'''

class TextCNN1D(nn.Module):
    #def __init__(self, n_classes, vocab_size, embeddings, emb_size, fine_tune=False, n_kernels=100, kernel_sizes=[3, 4, 5], dropout=0.3, n_channels=1):
    def __init__(
        self, vocab_size, embeddings, n_classes, embed_size=300, fine_tune=False, n_kernels=100, kernel_sizes=[3, 4, 5], dropout=0.3, n_channels=1):
    
        super(TextCNN1D, self).__init__()

        # embedding layer
        self.embedding1 = nn.Embedding(vocab_size, embed_size)
        self.set_embeddings(embeddings, 1, fine_tune)

        if n_channels == 2:
            # multichannel: a static channel and a non-static channel
            # which means embedding2 is frozen
            self.embedding2 = nn.Embedding(vocab_size, embed_size)
            self.set_embeddings(embeddings, 1, False)
        else:
            self.embedding2 = None

        # 1d conv layer
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels = n_channels, 
                out_channels = n_kernels, 
                #kernel_size = size * embed_size,
                #stride = embed_size
                kernel_size = size,
                #stride = 1
            ) 
            for size in kernel_sizes
        ])

        self.fc = nn.Linear(len(kernel_sizes) * n_kernels, n_classes) 
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()


    '''
    set weights of embedding layer
    input param:
        embeddings: word embeddings
        layer_id: embedding layer 1 or 2 (when adopting multichannel architecture)
        fine_tune: allow fine-tuning of embedding layer? 
                   (only makes sense when using pre-trained embeddings)
    '''
    def set_embeddings(self, embeddings, layer_id=1, fine_tune=False):
        if embeddings is None:
            # initialize embedding layer with the uniform distribution
            if layer_id == 1:
                self.embedding1.weight.data.uniform_(-0.1, 0.1)
            else:
                self.embedding2.weight.data.uniform_(-0.1, 0.1)
        else:
            # initialize embedding layer with pre-trained embeddings
            if layer_id == 1:
                self.embedding1.weight = nn.Parameter(embeddings, requires_grad = fine_tune)
            else:
                self.embedding2.weight = nn.Parameter(embeddings, requires_grad = fine_tune)


    '''
    input param:
        text: input data (batch_size, word_pad_len)
        words_per_sentence: sentence lengths (batch_size)
    return: 
        scores: class scores (batch_size, n_classes)
    '''
    #def forward(self, text, words_per_sentence):
    #def forward(self, text, words_per_sentence):
    def forward(self, text):

        text = text.permute(1,0)
    
        batch_size = text.size(0)

        # word embedding
        embeddings = self.embedding1(text).view(batch_size, 1, -1)  # (batch_size, 1, word_pad_len * emb_size)
        # multichannel
        if self.embedding2:
            embeddings2 = self.embedding2(text).view(batch_size, 1, -1)  # (batch_size, 1, word_pad_len * emb_size)
            embeddings = torch.cat((embeddings, embeddings2), dim = 1) # (batch_size, 2, word_pad_len * emb_size)

        # conv
        conved = [self.relu(conv(embeddings)) for conv in self.convs]  # [(batch size, n_kernels, word_pad_len - kernel_sizes[n] + 1)]

        # pooling
        pooled = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in conved]  # [(batch size, n_kernels)]
        
        # flatten
        flattened = self.dropout(torch.cat(pooled, dim = 1))  # (batch size, n_kernels * len(kernel_sizes))
        scores = self.fc(flattened)  # (batch size, n_classes)
        
        return scores

"""
# global parameters
model_name: textcnn  # 'han', 'fasttext', 'attbilstm', 'textcnn', 'transformer'
                     # refer to README.md for more info about each model

# dataset parameters
dataset: ag_news  # 'ag_news', 'dbpedia', 'yelp_review_polarity', 'yelp_review_full', 'yahoo_answers', 'amazon_review_polarity', 'amazon_review_full'
                  # refer to README.md for more info about each dataset
dataset_path: /Users/zou/Desktop/Text-Classification/data/ag_news  # folder with dataset
output_path: /Users/zou/Desktop/Text-Classification/data/outputs/ag_news/sents  # folder with data files saved by preprocess.py

# preprocess parameters
word_limit: 200
min_word_count: 5

# word embeddings parameters
emb_pretrain: True  # false: initialize embedding weights randomly
                    # true: load pre-trained word embeddings
emb_folder: /Users/zou/Desktop/Text-Classification/data/glove  # only makes sense when `emb_pretrain: True`
emb_filename: glove.6B.300d.txt  # only makes sense when `emb_pretrain: True`
emb_size: 256  # word embedding size
               # only makes sense when `emb_pretrain: False`
fine_tune_word_embeddings: True  # fine-tune word embeddings?

# model parameters
conv_layer: '1D' # '1D', '2D', use 1D or 2D convolution layer
n_kernels: 100
kernel_sizes: [3, 4, 5]
n_channels: 2
dropout: 0.3  # dropout

# checkpoint saving parameters
checkpoint_path: /Users/zou/Desktop/Text-Classification/checkpoints  # path to save checkpoints, null if never save checkpoints
checkpoint_basename: checkpoint_textcnn_agnews  # basename of the checkpoint

# training parameters
start_epoch: 0  # start at this epoch
batch_size: 64  # batch size
lr: 0.001  # learning rate
lr_decay: 0.3  # a factor to multiply learning rate with (0, 1)
workers: 4  # number of workers for loading data in the DataLoader
num_epochs: 5  # number of epochs to run
grad_clip: null  # clip gradients at this value, null if never clip gradients
print_freq: 2000  # print training status every __ batches
checkpoint: null  # path to model checkpoint, null if none
# tensorboard
tensorboard: True  # enable tensorboard or not?
log_dir: /Users/zou/Desktop/Text-Classification/logs/textcnn  # folder for saving logs for tensorboard
                                                              # only makes sense when `tensorboard: True`
"""