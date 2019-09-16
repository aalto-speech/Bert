
import logging
import sys
import os
import lzma
import random

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


def change_tokens(text_file):
    for line in open(, encoding='utf-8'):
        parts =line.strip().split()
        
        for i,part in enumerate(parts):
            if part == '<s>':
                parts[i]='[CLS]'
            if part == '</s>':
                parts[i]='[SEP]'
    for 
if __name__ == "__main__":
    change_tokens('data/kielipankki.dev')
    