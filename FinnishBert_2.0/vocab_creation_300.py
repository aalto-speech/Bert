import logging
import sys
import os
import lzma
import random
import numpy as np
import six

vocab=[]
with open('data/kielipankki.train', "r", encoding="utf-8") as reader:
    while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            tokens=token.split()
            if len(tokens) < 300:
                for tok in tokens:
                    if tok not in vocab:
                        vocab.append(tok)
with tf.gfile.GFile('data/vocab_pre_300', "w") as writer:
    for item in vocab:
        writer.write(str(item)+"\n")
# token="hello there hello there i rock"
# tokens=token.split()
# if len(tokens) < 2:
#     for tok in tokens:
#         if tok not in vocab:
#             vocab.append(tok)
# print(vocab)