import datetime
import json
import numpy as np
import keras
import os

from keras.utils import np_utils

from keras.models import Model

import keras.callbacks as callbacks

from transformer_utils import load_optimizer_weights, CosineLRSchedule
from multihot_utils import recall_at_10

class MOOC_Model(object):
    """
    Base Class for Model
    """
    def __init__(self, embedding_vocab_size, multihot_input=False):
        """
        Initialize miscellaneous model attributes
        """
        self.model = None
        self.model_params = None
        self.model_history = None
        self.best_accuracy = None

        self.multihot_input = multihot_input
        if multihot_input:
            self.embedding_vocab_size = int(embedding_vocab_size)
        else:
            self.embedding_vocab_size = int(embedding_vocab_size) +1

    def set_model_name(self, name):
        if not self.model_params:
            print("WARNING: Create model before setting model name.")
            return -1
        self.model_name = name + self.model_params_to_string

    @property
    def model_params_to_string(self):
        return '_{layers!s}_{seq_len!s}_{embed_dim!s}_{e_vocab_size!s}_{lrate!s}'.format(**self.model_params)

    def compile_and_load(self, optimizer, model_load_path=None):
        '''
        Compile model with optimizer and load weights saved from model_load_path
        '''
        assert self.model_params is not None, 'Please create model before training'
        print('Compiling model with params: {}'.format(self.model_params))

        if self.multihot_input:
            self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[recall_at_10])
        else:
            self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        # load model weights if specified
        if model_load_path is not None and os.path.exists(model_load_path):
            self.model.load_weights(model_load_path, skip_mismatch=True, by_name=True)
            load_optimizer_weights(self.model, model_load_path)
            print('Old model from {} successfully loaded.\n'.format(model_load_path))
            self.model.summary()
            return

        print('Model compiled successfully; Model summary below:\n')
        print('-'*80)
        self.model.summary()

    def early_stopping_fit(self, train_x, train_y, val_x, val_y, epoch_limit=200, loss_nonimprove_limit=3, batch_size=64, use_cosine_lr=False, model_save_path=None, tensorboard_log_path=None):
        '''
        Train model with early stopping condition
        '''
        assert self.model_params is not None, 'Please create model before training'
        print('Training model with params: {}'.format(self.model_params))

        val_metric = 'val_recall_at_10' if self.multihot_input else 'val_acc'
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=loss_nonimprove_limit, verbose=1)
        model_callbacks = [callbacks.TerminateOnNaN(), early_stopping]

        if use_cosine_lr:
            model_callbacks.append(callbacks.LearningRateScheduler(
            CosineLRSchedule(lr_high=self.model_params['lrate'], lr_low=self.model_params['lrate'] / 32, initial_period=10), verbose=1))
        if model_save_path is not None:
            model_callbacks.append(callbacks.ModelCheckpoint(model_save_path, monitor=val_metric, mode='max', save_best_only=True, verbose=True))
        if tensorboard_log_path is not None:
            model_callbacks.append(callbacks.TensorBoard(tensorboard_log_path, \
                histogram_freq=1, batch_size=batch_size, write_grads=True, write_images=True))

        self.model_history = self.model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=batch_size, epochs=epoch_limit, callbacks=model_callbacks)

        try:
            self.best_accuracy = max(self.model_history.history[val_metric])
            print("Best accuracy:", self.best_accuracy)
        except KeyError:
            print("NaN encountered in training at epoch {}, terminating training.".format(i))
        finally:
            return

    def test_set_eval(self, test_x, test_y, batch_size=64, tensorboard_log_path=None):
        '''
        Evaluate model using test set
        '''
        assert self.model_params is not None, 'Please create model before testing'
        print('Testing model with params: {}'.format(self.model_params))

        test_metrics = self.model.evaluate(test_x, test_y, batch_size=batch_size)
        
        for metric_name, metric_value in zip(self.model.metrics_names, test_metrics):
            print('Test {}: {:.8f}'.format(metric_name, metric_value))
