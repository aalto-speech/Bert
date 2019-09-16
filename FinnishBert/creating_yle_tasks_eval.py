from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import collections
import os
import json
import model_1
#import tokenization
import numpy as np
import tensorflow as tf

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab
vocab= load_vocab('yle_dev_clean')
vocab_words=load_vocab('vocab_with_aff')
def convert_tokens_to_ids_UNK(tokens,vocab_words):
    ids = []
    for token in tokens:
        ids.append(vocab_words[token])
    return ids

UNK_ID=convert_tokens_to_ids_UNK(['[UNK]'],vocab_words)[0]

def convert_tokens_to_ids(tokens,vocab_words):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            #print(token)
            if token not in vocab_words:
                ids.append(UNK_ID)
            else:
                ids.append(vocab_words[token])
            
        return ids
def main():

    tokk=[]
    #print(vocab)
    for tok, ids in vocab.items():
        tok=tok.split()
        tokk.append(tok)
    #print(tokk)
    gg=[]
    for tokens in tokk:
        gg.append(convert_tokens_to_ids(tokens,vocab_words))
    print(gg)
    counter=0
    print(np.array(gg).shape)
    for ids in gg:
        for id in ids:
            if id ==UNK_ID:
                counter+=1
    print(counter)


if __name__=="__main__":
   main()