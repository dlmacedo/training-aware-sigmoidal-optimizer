# https://www.aclweb.org/anthology/P16-2034.pdf

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
#from .attention import *

'''
attention-based bidirectional LSTM
attributes:
    n_classes: number of classes
    vocab_size: number of words in the vocabulary of the model
    embeddings: word embedding weights
    emb_size: size of word embeddings
    fine_tune: allow fine-tuning of embedding layer?
               (only makes sense when using pre-trained embeddings)
    rnn_size: size of bi-LSTM
    rnn_layers: number of layers in bi-LSTM
    dropout: dropout
'''
class AttBiLSTM(nn.Module):

    def __init__(self, vocab_size, embeddings, n_classes, emb_size=300, fine_tune=False, rnn_size=10, rnn_layers=1, dropout=0.3):

        super(AttBiLSTM, self).__init__()

        self.rnn_size = rnn_size
        
        # embedding layer
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.set_embeddings(embeddings, fine_tune)

        # bidirectional LSTM
        self.BiLSTM = nn.LSTM(
            emb_size, rnn_size, 
            num_layers = rnn_layers, 
            bidirectional = True,
            dropout = (0 if rnn_layers == 1 else dropout), 
            batch_first = True
        )

        self.attention = Attention(rnn_size)
        self.fc = nn.Linear(rnn_size, n_classes)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = 1)


    '''
    set weights of embedding layer
    input param:
        embeddings: word embeddings
        fine_tune: allow fine-tuning of embedding layer? 
                   (only makes sense when using pre-trained embeddings)
    '''
    def set_embeddings(self, embeddings, fine_tune = True):
        if embeddings is None:
            # initialize embedding layer with the uniform distribution
            self.embeddings.weight.data.uniform_(-0.1, 0.1)
        else:
            # initialize embedding layer with pre-trained embeddings
            self.embeddings.weight = nn.Parameter(embeddings, requires_grad = fine_tune)


    '''
    input param:
        text: input data (batch_size, word_pad_len)
        words_per_sentence: sentence lengths (batch_size)
    return: 
        scores: class scores (batch_size, n_classes)
    '''
    #def forward(self, text, words_per_sentence):
    def forward(self, text):

        text = text.permute(1, 0)
        
        # word embedding, apply dropout
        embeddings = self.dropout(self.embeddings(text)) # (batch_size, word_pad_len, emb_size)
        
        # pack sequences (remove word-pads, SENTENCES -> WORDS)
        #packed_words = pack_padded_sequence(
        #    embeddings,
        #    lengths = words_per_sentence.tolist(),
        #    batch_first = True,
        #    enforce_sorted = False
        #)  # a PackedSequence object, where 'data' is the flattened words (n_words, emb_size)
        
        # run through bidirectional LSTM (PyTorch automatically applies it on the PackedSequence)
        #rnn_out, _ = self.BiLSTM(packed_words)  # a PackedSequence object, where 'data' is the output of the LSTM (n_words, 2 * rnn_size)
        rnn_out, _ = self.BiLSTM(embeddings)  # a PackedSequence object, where 'data' is the output of the LSTM (n_words, 2 * rnn_size)
        
        # unpack sequences (re-pad with 0s, WORDS -> SENTENCES)
        #rnn_out, _ = pad_packed_sequence(rnn_out, batch_first = True)  # (batch_size, word_pad_len, 2 * word_rnn_size)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first = False)  # (batch_size, word_pad_len, 2 * word_rnn_size)

        # eq.8: h_i = [\overrightarrow{h}_i ⨁ \overleftarrow{h}_i ]
        # H = {h_1, h_2, ..., h_T}
        H = rnn_out[ :, :, : self.rnn_size] + rnn_out[ :, :, self.rnn_size : ] # (batch_size, word_pad_len, rnn_size)

        # attention module
        r, alphas = self.attention(H)  # (batch_size, rnn_size), (batch_size, word_pad_len)

        # eq.12: h* = tanh(r)
        h = self.tanh(r)  # (batch_size, rnn_size)

        scores = self.fc(self.dropout(h))  # (batch_size, n_classes)

        return scores #, alphas


'''
attention network
attributes:
    rnn_size: size of bi-LSTM
'''
class Attention(nn.Module):
    def __init__(self, rnn_size):
        super(Attention, self).__init__()
        self.w = nn.Linear(rnn_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = 1)
    
    '''
    input param: 
        H: output of bi-LSTM (batch_size, word_pad_len, hidden_size)
    
    return:
        r: sentence representation r (batch_size, rnn_size)
        alpha: attention weights (batch_size, word_pad_len)
    '''
    def forward(self, H):

        # eq.9: M = tanh(H)
        M = self.tanh(H)  # (batch_size, word_pad_len, rnn_size)
        
        # eq.10: α = softmax(w^T M)
        alpha = self.w(M).squeeze(2)  # (batch_size, word_pad_len)
        alpha = self.softmax(alpha)  # (batch_size, word_pad_len)
        
        # eq.11: r = H 
        r = H * alpha.unsqueeze(2)  # (batch_size, word_pad_len, rnn_size)
        r = r.sum(dim = 1)  # (batch_size, rnn_size)
        
        return r, alpha


"""
# global parameters
model_name: attbilstm  # 'han', 'fasttext', 'attbilstm', 'textcnn', 'transformer'
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
rnn_size: 10  # bi-RNN size
rnn_layers: 1  # number of layers in bi-RNN
dropout: 0.3  # dropout

# checkpoint saving parameters
checkpoint_path: /Users/zou/Desktop/Text-Classification/checkpoints  # path to save checkpoints, null if never save checkpoints
checkpoint_basename: checkpoint_attbilstm_agnews  # basename of the checkpoint

# training parameters
start_epoch: 0  # start at this epoch
batch_size: 64  # batch size
lr: 0.001  # learning rate
lr_decay: 0.9  # a factor to multiply learning rate with (0, 1)
workers: 4  # number of workers for loading data in the DataLoader
num_epochs: 5  # number of epochs to run
grad_clip: null  # clip gradients at this value, null if never clip gradients
print_freq: 2000  # print training status every __ batches
checkpoint: null  # path to model checkpoint, null if none
# tensorboard
tensorboard: True  # enable tensorboard or not?
log_dir: /Users/zou/Desktop/Text-Classification/logs/attbilstm  # folder for saving logs for tensorboard
                                                                # only makes sense when `tensorboard: True`
"""