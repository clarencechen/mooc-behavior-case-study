import datetime
import json
import numpy as np
import keras
import os

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad, Adam
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed
import keras.callbacks as callbacks

from transformer_utils import load_optimizer_weights, CosineLRSchedule
from transformer_models import vanilla_transformer_gpt_model

class MOOC_Keras_Model(object):
    """
    
    """
    def __init__(self, embedding_vocab_size):
        """
        """
        self.X = None
        self.y = None
        self.padded_y_windows = None
        self.model = None
        self.model_params = None
        self.model_histories = []
        self.embedding_vocab_size = int(embedding_vocab_size) +1
        self.best_epoch = None
        self.previous_val_loss = []

    def import_data(self, X, y, additional_params = []):
        """
        """
    
    def set_model_name(self, name):
        if not self.model_params:
            print("WARNING: Create LSTM model before setting model name.")
            return -1
        self.model_name = name + self.model_params_to_string

    @property
    def model_params_to_string(self):
        return '_{layers!s}_{seq_len!s}_{embed_dim!s}_{e_vocab_size!s}_{lrate!s}'.format(**self.model_params)

    def create_basic_transformer_model(self, lrate=2e-3, layers=4, embed_dim=128, seq_len=256, model_load_path=None):
        """
        Returns a Vanilla Transformer model
        """
        self.model_params = {'layers': layers,
            'embed_dim': embed_dim,
            'e_vocab_size': self.embedding_vocab_size,
            'seq_len': seq_len,
            'lrate': lrate}

        optimizer = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, clipvalue=5.0)
        model = vanilla_transformer_gpt_model(
                max_seq_length=seq_len,
                vocabulary_size=self.embedding_vocab_size,
                word_embedding_size=embed_dim,
                transformer_depth=layers,
                num_heads=8)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        if model_load_path is not None and os.path.exists(model_load_path):
            model.load_weights(model_load_path, skip_mismatch=True, by_name=True)
            load_optimizer_weights(model, model_load_path)
            print('Old model from {} successfully loaded.\n'.format(model_load_path))
            model.summary()
            self.model = model
            return
        
        print('Compiled new model.\n')
        print('-'*80)
        model.summary()
        self.model = model

    def create_basic_lstm_model(self, lrate=0.01, opt=Adagrad, layers=2, hidden_size=128, embed_dim=128, seq_len=256, model_load_path=None):
        """
        Returns a LSTM model
        """
        self.model_params = {'layers': layers,
            'hidden_size': hidden_size,
            'embed_dim': embed_dim,
            'e_vocab_size': self.embedding_vocab_size,
            'seq_len': seq_len,
            'lrate': lrate,
            'opt': opt}

        main_input = Input(shape=(seq_len,), name='main_input', dtype='int32')
        x = Embedding(input_dim=self.embedding_vocab_size, output_dim=embed_dim, input_length=seq_len, mask_zero=True)(main_input)
        for i in range(layers):
            print("Adding layer: " + str(i))
            x = LSTM(hidden_size, dropout=0.2, return_sequences=True)(x)
        main_loss = TimeDistributed(Dense(self.embedding_vocab_size, activation='softmax'))(x)
        
        model = Model(inputs=[main_input], outputs=[main_loss])
        model.compile(optimizer=opt(lr=lrate), loss='categorical_crossentropy', metrics=['accuracy'])

        if model_load_path is not None and os.path.exists(model_load_path):
            model.load_weights(model_load_path, skip_mismatch=True, by_name=True)
            load_optimizer_weights(model, model_load_path)
            print('Old model from {} successfully loaded.\n'.format(model_load_path))
            model.summary()
            self.model = model
            return

        print('Compiled new model.\n')
        print('-'*80)
        model.summary()
        self.model = model

    def transformer_model_fit(self, train_x, train_y, val_x, val_y, epoch_limit=5, batch_size=64, model_save_path=None, tensorboard_log_path=None):

        assert self.model_params is not None, 'Please create model before training'
        print('Training model with params: {}'.format(self.model_params))

        lr_scheduler = callbacks.LearningRateScheduler(
            CosineLRSchedule(lr_high=self.model_params['lrate'], lr_low=self.model_params['lrate'] / 32, initial_period=3),
            verbose=1)
        model_callbacks = [lr_scheduler]
        
        if model_save_path is not None:
            model_callbacks.append(callbacks.ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1))
        if tensorboard_log_path is not None:
            model_callbacks.append(callbacks.TensorBoard(tensorboard_log_path, \
                histogram_freq=1, batch_size=batch_size, write_grads=True, write_images=True, \
                embeddings_freq=epoch_limit, embeddings_layer_names='bpe_embeddings', embeddings_data=np.arange(self.embedding_vocab_size)))

        self.model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=batch_size, epochs=epoch_limit, callbacks=model_callbacks)

    def transformer_model_eval(self, test_x, test_y, batch_size=64, tensorboard_log_path=None):
        '''
        Evaluate model using test set
        '''
        assert self.model_params is not None, 'Please create model before testing'
        print('Testing model with params: {}'.format(self.model_params))

        model_callbacks = []

        if tensorboard_log_path is not None:
            model_callbacks.append(callbacks.TensorBoard(tensorboard_log_path))

        test_metrics = self.model.evaluate(test_x, test_y, batch_size=batch_size, callbacks=model_callbacks)
        for metric_name, metric_value in zip(self.model.metrics_names, test_metrics):
            print('Test {}: {:.8f}'.format(metric_name, metric_value))

    def early_stopping_model_fit(self, train_x, train_y, val_x, val_y, epoch_limit=200, loss_nonimprove_limit=3, batch_size=64, model_save_path=None, tensorboard_log_path=None):
        
        assert self.model_params is not None

        model_callbacks = []
        if model_save_path is not None:
            model_callbacks.append(callbacks.ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=True))
        if tensorboard_log_path is not None:
            model_callbacks.append(callbacks.TensorBoard(tensorboard_log_path))

        early_stopping_met = False
        for i in range(epoch_limit):
            print("Epoch: ", i)
            current_history = self.model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=batch_size, epochs=1)
            current_history = current_history.history
            validation_loss, validation_accuracy = current_history['val_loss'][0], current_history['val_acc'][0]
            #average_of_average_accuracy = np.mean(validation_accuracy_dictionary['averages'])
            self.previous_val_loss.append(validation_loss)
            if len(self.previous_val_loss) > loss_nonimprove_limit:
                min_val_loss = min(self.previous_val_loss)
                recent_losses = self.previous_val_loss[-loss_nonimprove_limit-1:]
                print(recent_losses)
                if min(recent_losses) > min_val_loss:
                    early_stopping_met = True
                if validation_loss == min_val_loss:
                    self.best_epoch = i
                    #self.best_average_of_average_accuracy = average_of_average_accuracy
                    self.best_accuracy = validation_accuracy
            if early_stopping_met:
                print("Early stopping reached.")
                print("Best epoch according to validation loss:", self.best_epoch)
                print("Best epoch's accuracy:", self.best_accuracy)
                #print("Best epoch's average accuracy:", self.best_average_of_average_accuracy)
                return

"""
    def simple_model_fit(self, epochs = 10, batch_size = 64, validation_proportion = 0.1):
        for i in range(epochs):
            print("epoch:", i)
            validation_i = int(len(self.X) * .9)
            hist = self.model.fit(self.X[:validation_i], self.y[:validation_i], batch_size = batch_size, nb_epoch = 1, validation_data = (self.X[validation_i:], self.y[validation_i:]))
            self.model_histories.append(hist)
"""
