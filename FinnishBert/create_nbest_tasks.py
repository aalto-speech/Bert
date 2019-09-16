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

def parse_result(result, all_tokens,all_ids, output_file=None):
  with tf.gfile.GFile(output_file, "w") as writer:
    tf.logging.info("***** Predict results *****")
    i = 0
    #sentences ={}
    result_str=""
    ppl=[]
    #sentence_loss=0.0
    #word_count_per_sent=0
    sentence_counter=0
    for word_loss in result:
      #print(word_loss)
      # start of a sentence
      if all_tokens[i] == "[CLS]":
        #sentence = {}
        #tokens = []
        sentence_loss = 0.0
        #word_count_per_sent = 0
        i += 1

      # add token
      #tokens.append({"token": printable_text(all_tokens[i]),
     #                "prob": float(np.exp(-word_loss[0])) })
      sentence_loss += word_loss[0]
      #word_count_per_sent += 1
      i += 1

      #token_count_per_word = 0
      #while is_subtoken(all_tokens[i]):
        #token_count_per_word += 1
        #tokens.append({"token": printable_text(all_tokens[i]),
         #              "prob": float(np.exp(-word_loss[token_count_per_word]))})
        #sentence_loss += word_loss[token_count_per_word]
        #i += 1

      # end of a sentence
      if all_tokens[i] == "[SEP]":
        #sentence["tokens"] = tokens
        ppl.append(sentence_loss)
        result_str+=all_ids[sentence_counter]+" "+sentence_loss+"/n"
        #sentences.append(sentence)
        sentence_counter+=1
        i += 1
    #sentences["ppl"]=float(2**(sentence_loss/word_count_per_sent))
    if output_file is not None:
      tf.logging.info("Saving results to %s" % output_file)
      writer.write(result_str)


age=23
gender="male"


st=""
st+=str(age)+" "+gender+"\n"
print(st)


dev_set=collections.OrderedDict()
index = 0
all_ids=[]
with open('yle_testing_stuff', "r", encoding="utf-8") as reader:
    while True:
        line = reader.readline()
        if not line:
            break
        line = line.strip()
        Splitted=line.split(" ", 1)
        if len(Splitted) == 1:
            Splitted.append('[SEP]')
        all_ids.append(Splitted[0])
        token=Splitted[1]
        dev_set[token] = index
        index += 1
print(dev_set)
