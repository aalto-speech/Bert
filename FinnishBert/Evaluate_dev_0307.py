
"""BERT language model predict."""

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
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("max_predictions_per_seq", 40,
"In this task, it also refers to maximum number of masked tokens per word.")

flags.DEFINE_string(
    "bert_config_file", 'bert_config_1506.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "output_dir", 'outputckpoints_1506',
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 300,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer("predict_batch_size", 32, "Total batch size for predict.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")



#%%


class InputExample(object):
  def __init__(self, unique_id, text):
    self.unique_id = unique_id
    self.text = text





def model_fn_builder(bert_config, init_checkpoint, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    #segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]


    model = model_1.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        #token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
    compute_type=tf.float16)

    masked_lm_example_loss = get_masked_lm_output(
        bert_config, model.get_sequence_output(), model.get_embedding_table(),
        masked_lm_positions, masked_lm_ids)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = model_1.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.PREDICT:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=masked_lm_example_loss, scaffold_fn=scaffold_fn)  # 输出mask_word的score
    return output_spec

  return model_fn




def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=model_1.get_activation(bert_config.hidden_act),
          kernel_initializer=model_1.create_initializer(
              bert_config.initializer_range))
      input_tensor = model_1.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    loss = tf.reshape(per_example_loss, [-1, tf.shape(positions)[1]])
    # TODO: dynamic gather from per_example_loss
  return loss



def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = model_1.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def input_fn_builder(features, seq_length, max_predictions_per_seq):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  #all_segment_ids = []
  all_masked_lm_positions = []
  all_masked_lm_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    #all_segment_ids.append(feature.segment_ids)
    all_masked_lm_positions.append(feature.masked_lm_positions)
    all_masked_lm_ids.append(feature.masked_lm_ids)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]
    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "masked_lm_positions":
            tf.constant(
                all_masked_lm_positions,
                shape=[num_examples, max_predictions_per_seq],
                dtype=tf.int32),
        "masked_lm_ids":
            tf.constant(
                all_masked_lm_ids,
                shape=[num_examples, max_predictions_per_seq],
                dtype=tf.int32)
    })

    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d

  return input_fn


def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
      raise ValueError("Not running on Python2 or Python 3?")
# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, max_seq_length):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  all_features = []
  all_tokens = []

  for (ex_index, example) in enumerate(examples):
    #print(example)
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    features, tokens = convert_single_example(ex_index, example,
                                     max_seq_length)
    all_features.extend(features)
    all_tokens.extend(tokens)

  return all_features, all_tokens

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

def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    #print('amazing',item)  
    output.append(vocab[item])
  return output

def convert_ids_to_tokens(vocab, ids):
  ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in vocab.items()])
  tokens = []
  for i in ids:
      tokens.append(ids_to_tokens[i])
  return tokens

MASKED_TOKEN = "[MASK]"
vocab_words=load_vocab('vocab_with_aff')

def convert_tokens_to_ids(tokens,vocab_words):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            #print(token)
            ids.append(vocab_words[token])
            
        return ids
MASKED_ID = convert_tokens_to_ids([MASKED_TOKEN],vocab_words)[0]


def create_masked_lm_prediction(input_ids, mask_position, mask_count=1):
  new_input_ids = list(input_ids)
  masked_lm_labels = []
  masked_lm_positions = list(range(mask_position, mask_position + mask_count))
  for i in masked_lm_positions:
    new_input_ids[i] = MASKED_ID
    masked_lm_labels.append(input_ids[i])
  return new_input_ids, masked_lm_positions, masked_lm_labels


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, masked_lm_positions,
               masked_lm_ids):
    self.input_ids = input_ids,
    #self.segment_ids = segment_ids,
    self.input_mask = input_mask,
    self.masked_lm_positions = masked_lm_positions,
    self.masked_lm_ids = masked_lm_ids


def convert_single_example(ex_index, example, max_seq_length):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  #print(example.text)
  tokens = example.text.split()

  # Account for [CLS] and [SEP] with "- 2"
  if len(tokens) > max_seq_length - 2:
    tokens = tokens[0:(max_seq_length - 2)]

  input_tokens = []
  #segment_ids = []
  #input_tokens.append("[CLS]")
  #segment_ids.append(0)
  for token in tokens:
    input_tokens.append(token)
    #segment_ids.append(0)
  #input_tokens.append("[SEP]")
  #segment_ids.append(0)
  #Tprint(input_tokens)

  input_ids = convert_tokens_to_ids(input_tokens,vocab_words)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    #segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  #assert len(segment_ids) == max_seq_length

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("id: %s" % (example.unique_id))
    tf.logging.info("tokens: %s" % " ".join(
        [printable_text(x) for x in input_tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    #tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

  features = create_sequential_mask(input_tokens, input_ids, input_mask,
                                    FLAGS.max_predictions_per_seq)
  return features, input_tokens


def is_subtoken(x):
  return '+' in x

def create_sequential_mask(input_tokens, input_ids, input_mask,
                           max_predictions_per_seq):
  """Mask each token/word sequentially"""
  features = []
  i = 1
  while i < len(input_tokens) - 1:
    mask_count = 1
    #while is_subtoken(input_tokens[i+mask_count]):
      #mask_count += 1

    input_ids_new, masked_lm_positions, masked_lm_labels = create_masked_lm_prediction(input_ids, i, mask_count)
    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_labels.append(0)

    feature = InputFeatures(
      input_ids=input_ids_new,
      input_mask=input_mask,
      #segment_ids=segment_ids,
      masked_lm_positions=masked_lm_positions,
      masked_lm_ids=masked_lm_labels)
    features.append(feature)
    i += mask_count
  return features

def create_sequential_mask2(input_tokens, input_ids, input_mask,
                           max_predictions_per_seq):
  cand_indices=[]
  for (i, token) in enumerate(input_tokens):
      if token == "[CLS]" or token == "[SEP]":
              continue
      cand_indices.append(i)
  covered_indices=[]
  features=[]
  output_tokens=[]
  #masked_lm_positions = []
  for i in cand_indices:
      if i in covered_indices:
          continue
      covered_indices.append(i)    
      temp_sent=input_tokens[:]
      #print(temp_sent)
      temp_sent[i]='[MASK]'
      input_id=input_ids[:]
      input_id[i]=MASKED_ID
      #print(temp_sent)
      masked_lm_positions=[i]
      masked_lm_labels=[input_tokens[i]]
      while len(masked_lm_positions) < max_predictions_per_seq:
        masked_lm_positions.append(0)
        masked_lm_labels.append(0)
      feature = InputFeatures(
      input_ids=input_id,
      input_mask=input_mask,
      #segment_ids=segment_ids,
      masked_lm_positions=masked_lm_positions,
      masked_lm_ids=masked_lm_labels)
      features.append(feature)
      continue
      #op=TrainingInstance(temp_sent,masked_lm_positions,masked_lm_labels)
      #output_tokens.append(op)
  return features

def parse_result(result, all_tokens, output_file=None):
  with tf.gfile.GFile(output_file, "w") as writer:
    tf.logging.info("***** Predict results *****")
    i = 0
    sentences = []
    for word_loss in result:
      #print(word_loss)
      # start of a sentence
      if all_tokens[i] == "[CLS]":
        sentence = {}
        tokens = []
        sentence_loss = 0.0
        word_count_per_sent = 0
        i += 1

      # add token
      tokens.append({"token": printable_text(all_tokens[i]),
                     "prob": float(np.exp(-word_loss[0])) })
      sentence_loss += word_loss[0]
      word_count_per_sent += 1
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
        sentence["tokens"] = tokens
        sentence["ppl"] = float(np.exp(sentence_loss / word_count_per_sent))
        sentences.append(sentence)
        i += 1

    if output_file is not None:
      tf.logging.info("Saving results to %s" % output_file)
      writer.write(json.dumps(sentences, indent=2, ensure_ascii=False))

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = model_1.BertConfig.from_json_file(FLAGS.bert_config_file)
  config = tf.ConfigProto()
  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=None,
      model_dir=FLAGS.output_dir,
      session_config=config,
      save_checkpoints_steps=5000,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))


  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=FLAGS.predict_batch_size)


  #predict_examples =[InputExample(1,'yhdysvaltain sen+ +aatti on hyväksy+ +nyt pitkän kiist+ +e +ly+ +n velka+ +kat+ +on nosta+ +en ääni+ +n seitsemänkymmentä+ +neljä kaksikymmentä kuusi')]
  #predict_examples =[InputExample(1,'auto+ +ni on punainen')]
  #predict_examples =[InputExample(1,'hän esittää itsenäisyys+ +päivän puheen pikku myy+ +n ään+ +ellä ja pyytää linnan juhli+ +lle ka+ +val+ +je+ +erikseen eli linnan isän+ +nä+ +ksi edes+ +menne+ +en matti pello+ +n+ +pää+ +n jota tässä elämässä tuu+ +raa a+ +ki kau+ +ris+ +mäki')]  
  predict_examples=[]
  dev_set=load_vocab('data/kielipankki_clean.dev')
  for tok, ids in dev_set.items():
      temp_example=InputExample(ids,tok)
      predict_examples.append(temp_example)


  features, all_tokens = convert_examples_to_features(predict_examples,FLAGS.max_seq_length)
  #print(np.array(features))
  # print('vocabbb',len(vocab_words))
  # for feat in features:
  #     print('main function input_ids',feat.input_ids[0])
  #     print('main function input_mask',feat.input_mask[0])
  #     print('main function masked_lm_ids',feat.masked_lm_ids[0])
  #     print('main function masked_lm_positions ',feat.masked_lm_positions[0])
  #     temp= convert_ids_to_tokens(vocab_words,feat.input_ids[0])
  #     print(temp)
  tf.logging.info("***** Running prediction*****")
  tf.logging.info("  Num examples = %d", len(predict_examples))
  tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

  if FLAGS.use_tpu:
    # Warning: According to tpu_estimator.py Prediction on TPU is an
    # experimental feature and hence not supported here
    raise ValueError("Prediction in TPU not supported")

  predict_input_fn = input_fn_builder(
      features=features,
      seq_length=FLAGS.max_seq_length,
      max_predictions_per_seq=FLAGS.max_predictions_per_seq)
  #print(all_tokens)
  result = estimator.predict(input_fn=predict_input_fn)
  output_predict_file = os.path.join(FLAGS.output_dir, "test_results_dev_kiel.json")
   #print(list(result))
  parse_result(result, all_tokens, output_predict_file)
  # if output_predict_file is not None:
  #     tf.logging.info("Saving results to %s" % output_predict_file)
  #     writer.write(json.dumps(sentences, indent=2, ensure_ascii=False))

if __name__ == "__main__":
  tf.app.run()




