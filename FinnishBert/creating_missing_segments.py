
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
dev_set=collections.OrderedDict()
all_ids=[]
alt_ids=[]
with open('yle_nbest_1000', "r", encoding="utf-8") as reader:
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
        dev_set[Splitted[0]] = token


with open('outputckpoints_1506/lm_cost', "r", encoding="utf-8") as reader:
    while True:
        line = reader.readline()
        if not line:
            break
        line = line.strip()
        Splitted=line.split(" ", 1)
        if len(Splitted) == 1:
            Splitted.append('[SEP]')
        alt_ids.append(Splitted[0])

result_ids=np.setdiff1d(all_ids,alt_ids)
resulted_segment=""
for id_n in result_ids:
    resulted_segment+=id_n+" "+dev_set[id_n]+"\n"
with tf.gfile.GFile('last_segment', "w") as writer:
    writer.write(resulted_segment)