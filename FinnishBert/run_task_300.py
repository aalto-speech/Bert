import os
import time
import model_1
import optimization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS
flags.DEFINE_bool("use_fp16",False, "Whether to use fp32 or fp16 arithmetic on GPU.")
flags.DEFINE_bool("do_train", True, "Whether to run training.")
flags.DEFINE_bool("amp", False, "Whether to enable AMP ops.")

class _LogSessionRunHook(tf.train.SessionRunHook):
  def __init__(self, global_batch_size, display_every=1000, hvd_rank=-1):
    self.global_batch_size = global_batch_size
    self.display_every = display_every
    self.hvd_rank = hvd_rank
  def after_create_session(self, session, coord):
    if FLAGS.use_fp16 or FLAGS.amp:
      print('  Step samples/sec   MLM Loss  NSP Loss  Loss  Learning-rate  Loss-scaler')
    else:
      print('  Step samples/sec   MLM Loss  NSP Loss  Loss  Learning-rate')
    self.elapsed_secs = 0.
    self.count = 0
  def before_run(self, run_context):
    self.t0 = time.time()
    if FLAGS.use_fp16 or FLAGS.amp:
      return tf.train.SessionRunArgs(
          fetches=['step_update:0', 'total_loss:0',
                   'learning_rate:0','mlm_loss:0', 'loss_scale:0'])
    else:
      return tf.train.SessionRunArgs(
          fetches=['step_update:0', 'total_loss:0',
                   'learning_rate:0','mlm_loss:0'])
  def after_run(self, run_context, run_values):
    self.elapsed_secs += time.time() - self.t0
    self.count += 1
    if FLAGS.use_fp16 or FLAGS.amp:
      global_step, total_loss, lr, mlm_loss, loss_scaler = run_values.results
    else:
      global_step, total_loss, lr, mlm_loss = run_values.results
    print_step = global_step + 1 # One-based index for printing.
    if print_step == 1 or print_step % self.display_every == 0:
        dt = self.elapsed_secs / self.count
        img_per_sec = self.global_batch_size / dt
        if self.hvd_rank >= 0:
          if FLAGS.use_fp16 or FLAGS.amp:
            print('%2d :: %6i %11.1f %10.4e %6.3f     %6.4e  %6.4e' %
                  (self.hvd_rank, print_step, img_per_sec, mlm_loss, total_loss, lr, loss_scaler))
          else:
            print('%2d :: %6i %11.1f %10.4e %6.3f     %6.4e' %
                  (self.hvd_rank, print_step, img_per_sec, mlm_loss,total_loss, lr))
        else:
          if FLAGS.use_fp16 or FLAGS.amp:
            print('%6i %11.1f %10.4e %6.3f     %6.4e  %6.4e' %
                  (print_step, img_per_sec, mlm_loss, total_loss, lr, loss_scaler))
          else:
            print('%6i %11.1f %10.4e  %6.3f     %6.4e' %
                  (print_step, img_per_sec, mlm_loss, total_loss, lr))
        self.elapsed_secs = 0.
        self.count = 0


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, hvd=None):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    #segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = model_1.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        #token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        compute_type=tf.float16)

    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)


    masked_lm_loss = tf.identity(masked_lm_loss, name="mlm_loss")
    total_loss = masked_lm_loss
    total_loss = tf.identity(total_loss, name='total_loss')

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint and (hvd is None or hvd.rank() == 0):
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
      tf.logging.info("  %d :: name = %s, shape = %s%s", 0 if hvd is None else hvd.rank(), var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu,
          hvd, FLAGS.use_fp16, FLAGS.amp)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights):
        """Computes the loss and accuracy of the model."""
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

      

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss
            
        }

      eval_metrics = (metric_fn, [
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
          masked_lm_weights
      ])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn

def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
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
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)

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

def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4,
                     hvd=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        #"segment_ids":
         #   tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      if hvd is not None: d = d.shard(hvd.size(), hvd.rank())
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))#Default was True
    return d

  return input_fn

def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

   #tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example



def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    bert_config = model_1.BertConfig.from_json_file('bert_config.json')
    input_files = []
    input_files.extend(tf.gfile.Glob('kielipankki_clean_train_300_final.tfrecord'))
    tf.logging.info("*** Input Files ***")
    for input_file in input_files:
        tf.logging.info("  %s" % input_file)
    
    
    config = tf.ConfigProto()
    '''
    The TensorFlow/XLA JIT compiler compiles and runs parts of TensorFlow graphs via XLA. The benefit of this over the standard TensorFlow implementation is that XLA can fuse multiple operators (kernel fusion) into a small number of compiled kernels. Fusing operators can reduce memory bandwidth requirements and improve performance compared to executing operators one-at-a-time, as the TensorFlow executor does.'''
    tpu_cluster_resolver = None
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=None,
        model_dir='outputckpoints',
        session_config=config,
        save_checkpoints_steps=1000,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=1000,
            num_shards=8,
            per_host_input_for_training=is_per_host),
        # This variable controls how often estimator reports examples/sec.
        # Default value is every 100 steps.
        # When --report_loss is True, we set to very large value to prevent
        # default info reporting from estimator.
        # Ideally we should set it to None, but that does not work.
        log_step_count_steps=10000)
    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=None,
        learning_rate=5e-5,
        num_train_steps=100000,
        num_warmup_steps=10000,
        use_tpu=False,
        use_one_hot_embeddings=False,
        hvd=None)
    training_hooks = []
    training_hooks.append(_LogSessionRunHook(32,1000,-1))
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=False,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=12,
        eval_batch_size=8)
    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d",12)
        train_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=300,
            max_predictions_per_seq=40,
            is_training=True,
            hvd=None )
        estimator.train(input_fn=train_input_fn, hooks=training_hooks, max_steps=100000)
    
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", 8)

    eval_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=300,
        max_predictions_per_seq=40,
        is_training=False)

    result = estimator.evaluate(
        input_fn=eval_input_fn, steps=100)

    output_eval_file = os.path.join('outputckpoints', "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    tf.app.run()
