import os
import csv
import six

import numpy as np
import time
import json
import warnings
import io

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import keras
import keras.backend as K
from keras.utils.generic_utils import Progbar
from keras.engine.training_utils import standardize_input_data

from collections import deque
from collections import OrderedDict
from collections import Iterable
from collections import defaultdict

try:
  import requests
except ImportError:
  requests = None

class TensorBoard_TimestepEmbeddings(keras.callbacks.TensorBoard):
  """TensorBoard basic visualizations.
  [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard)
  is a visualization tool provided with TensorFlow.
  This callback writes a log for TensorBoard, which allows
  you to visualize dynamic graphs of your training and test
  metrics, as well as activation histograms for the different
  layers in your model.
  If you have installed TensorFlow with pip, you should be able
  to launch TensorBoard from the command line:
  ```sh
  tensorboard --logdir=/full_path_to_your_logs
  ```
  When using a backend other than TensorFlow, TensorBoard will still work
  (if you have TensorFlow installed), but the only feature available will
  be the display of the losses and metrics plots.
  # Arguments
    log_dir: the path of the directory where to save the log
      files to be parsed by TensorBoard.
    histogram_freq: frequency (in epochs) at which to compute activation
      and weight histograms for the layers of the model. If set to 0,
      histograms won't be computed. Validation data (or split) must be
      specified for histogram visualizations.
    batch_size: size of batch of inputs to feed to the network
      for histograms computation.
    write_graph: whether to visualize the graph in TensorBoard.
      The log file can become quite large when
      write_graph is set to True.
    write_grads: whether to visualize gradient histograms in TensorBoard.
      `histogram_freq` must be greater than 0.
    write_images: whether to write model weights to visualize as
      image in TensorBoard.
    embeddings_freq: frequency (in epochs) at which selected embedding
      layers will be saved. If set to 0, embeddings won't be computed.
      Data to be visualized in TensorBoard's Embedding tab must be passed
      as `embeddings_data`.
    embeddings_layer_names: a list of names of layers to keep eye on. If
      None or empty list all the embedding layer will be watched.
    embeddings_metadata: a dictionary which maps layer name to a file name
      in which metadata for this embedding layer is saved. See the
      [details](https://www.tensorflow.org/guide/embedding#metadata)
      about metadata files format. In case if the same metadata file is
      used for all embedding layers, string can be passed.
    embeddings_data: data to be embedded at layers specified in
      `embeddings_layer_names`. Numpy array (if the model has a single
      input) or list of Numpy arrays (if the model has multiple inputs).
      Learn [more about embeddings](
      https://www.tensorflow.org/guide/embedding).
    update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`, writes
      the losses and metrics to TensorBoard after each batch. The same
      applies for `'epoch'`. If using an integer, let's say `10000`,
      the callback will write the metrics and losses to TensorBoard every
      10000 samples. Note that writing too frequently to TensorBoard
      can slow down your training.
  """

  def __init__(self, *args, **kwargs):
    super(TensorBoard_TimestepEmbeddings, self).__init__(*args, **kwargs)

  def set_model(self, model):
    self.model = model
    if K.backend() == 'tensorflow':
      self.sess = K.get_session()
    if self.histogram_freq and self.merged is None:
      for layer in self.model.layers:
        for weight in layer.weights:
          mapped_weight_name = weight.name.replace(':', '_')
          tf.summary.histogram(mapped_weight_name, weight)
          if self.write_grads and weight in layer.trainable_weights:
            grads = model.optimizer.get_gradients(model.total_loss,
                                weight)

            def is_indexed_slices(grad):
              return type(grad).__name__ == 'IndexedSlices'
            grads = [
              grad.values if is_indexed_slices(grad) else grad
              for grad in grads]
            tf.summary.histogram('{}_grad'.format(mapped_weight_name),
                       grads)
          if self.write_images:
            w_img = tf.squeeze(weight)
            shape = K.int_shape(w_img)
            if len(shape) == 2:  # dense layer kernel case
              if shape[0] > shape[1]:
                w_img = tf.transpose(w_img)
                shape = K.int_shape(w_img)
              w_img = tf.reshape(w_img, [1,
                             shape[0],
                             shape[1],
                             1])
            elif len(shape) == 3:  # convnet case
              if K.image_data_format() == 'channels_last':
                # switch to channels_first to display
                # every kernel as a separate image
                w_img = tf.transpose(w_img, perm=[2, 0, 1])
                shape = K.int_shape(w_img)
              w_img = tf.reshape(w_img, [shape[0],
                             shape[1],
                             shape[2],
                             1])
            elif len(shape) == 1:  # bias case
              w_img = tf.reshape(w_img, [1,
                             shape[0],
                             1,
                             1])
            else:
              # not possible to handle 3D convnets etc.
              continue

            shape = K.int_shape(w_img)
            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
            tf.summary.image(mapped_weight_name, w_img)

        if hasattr(layer, 'output'):
          if isinstance(layer.output, list):
            for i, output in enumerate(layer.output):
              tf.summary.histogram('{}_out_{}'.format(layer.name, i),
                         output)
          else:
            tf.summary.histogram('{}_out'.format(layer.name),
                       layer.output)
    self.merged = tf.summary.merge_all()

    if self.write_graph:
      self.writer = tf.summary.FileWriter(self.log_dir,
                        self.sess.graph)
    else:
      self.writer = tf.summary.FileWriter(self.log_dir)

    if self.embeddings_freq and self.embeddings_data is not None:
      self.embeddings_data = standardize_input_data(self.embeddings_data,
                              model.input_names)

      embeddings_layer_names = self.embeddings_layer_names

      if not embeddings_layer_names:
        embeddings_layer_names = [layer.name for layer in self.model.layers if type(layer).__name__ == 'Embedding']
      self.assign_embeddings = []
      embeddings_vars = {}

      self.batch_id = batch_id = tf.placeholder(tf.int32)
      self.step = step = tf.placeholder(tf.int32)

      for layer in self.model.layers:
        if layer.name in embeddings_layer_names:
          if 'ReusableEmbed' in type(layer).__name__:
            embedding_input = self.model.get_layer(layer.name).output[0]
          else:
            embedding_input = self.model.get_layer(layer.name).output
          embedding_size = np.prod(embedding_input.shape[1:])
          embedding_input = tf.reshape(embedding_input,
                         (step, int(embedding_size)))
          shape = (self.embeddings_data[0].shape[0], int(embedding_size))
          embedding = tf.Variable(tf.zeros(shape),
                      name=layer.name + '_embedding')
          embeddings_vars[layer.name] = embedding
          batch = tf.assign(embedding[batch_id:batch_id + step],
                    embedding_input)
          self.assign_embeddings.append(batch)

      self.saver = tf.train.Saver(list(embeddings_vars.values()))

      if not isinstance(self.embeddings_metadata, str):
        embeddings_metadata = self.embeddings_metadata
      else:
        embeddings_metadata = {layer_name: self.embeddings_metadata
                     for layer_name in embeddings_vars.keys()}

      config = projector.ProjectorConfig()

      for layer_name, tensor in embeddings_vars.items():
        embedding = config.embeddings.add()
        embedding.tensor_name = tensor.name

        if layer_name in embeddings_metadata:
          embedding.metadata_path = embeddings_metadata[layer_name]

      projector.visualize_embeddings(self.writer, config)
