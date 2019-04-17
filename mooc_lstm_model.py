import datetime
import json
import numpy as np
import keras
import os

from keras.optimizers import RMSprop
from keras.utils import np_utils

from keras.models import Model
from keras.layers import Input, Dense

from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed

import keras.callbacks as callbacks

from mooc_enhanced_lstm import enhanced_lstm_model

class MOOC_LSTM_Model(object):
    """
    
    """
    def __init__(self, embedding_vocab_size):
        """
        """
        self.model = None
        self.model_params = None
        self.model_histories = []
        self.embedding_vocab_size = int(embedding_vocab_size) +1
        self.best_epoch = None
        self.previous_val_loss = []

    def set_model_name(self, name):
        if not self.model_params:
            print("WARNING: Create LSTM model before setting model name.")
            return -1
        self.model_name = name + self.model_params_to_string

    @property
    def model_params_to_string(self):
        return '_{layers!s}_{seq_len!s}_{embed_dim!s}_{e_vocab_size!s}_{lrate!s}'.format(**self.model_params)

    def create_basic_lstm_model(self, use_enhancements=True, lrate=0.01, layers=2, hidden_size=128, embed_dim=128, seq_len=256, model_load_path=None):
        """
        Returns a LSTM model
        """
        self.model_params = {'layers': layers,
            'hidden_size': hidden_size,
            'embed_dim': embed_dim,
            'e_vocab_size': self.embedding_vocab_size,
            'seq_len': seq_len,
            'lrate': lrate}

        if(use_enhancements):
            model = enhanced_lstm_model(self.model_params)
            model.compile(optimizer=RMSprop(lr=lrate), loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            main_input = Input(shape=(seq_len,), name='main_input', dtype='int32')
            x = Embedding(input_dim=self.embedding_vocab_size, output_dim=embed_dim, input_length=seq_len, mask_zero=True)(main_input)
            for i in range(layers):
                print("Adding layer: " + str(i))
                x = LSTM(hidden_size, dropout=0.2, return_sequences=True)(x)
            main_loss = TimeDistributed(Dense(self.embedding_vocab_size, activation='softmax'))(x)

            model = Model(inputs=[main_input], outputs=[main_loss])
            model.compile(optimizer=RMSprop(lr=lrate), loss='categorical_crossentropy', metrics=['accuracy'])

        # load model weights if specified
        if model_load_path is not None and os.path.exists(model_load_path):
            model.load_weights(model_load_path, skip_mismatch=True, by_name=True)
            load_optimizer_weights(model, model_load_path)
            print('Old model from {} successfully loaded.\n'.format(model_load_path))
            model.summary()
            self.model = model
            return

        print('Model compiled successfully; will train from new initialization.\n')
        print('-'*80)
        model.summary()
        self.model = model

    def early_stopping_model_fit(self, train_x, train_y, val_x, val_y, epoch_limit=200, loss_nonimprove_limit=3, batch_size=64, model_save_path=None, tensorboard_log_path=None):
        
        assert self.model_params is not None

        model_callbacks = []
        if model_save_path is not None:
            model_callbacks.append(callbacks.ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=True))
        if tensorboard_log_path is not None:
            model_callbacks.append(callbacks.TensorBoard(tensorboard_log_path, \
                histogram_freq=1, batch_size=batch_size, write_grads=True, write_images=True))

        early_stopping_met = False
        for i in range(epoch_limit):
            print("Epoch: ", i)
            current_history = self.model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=batch_size, epochs=1, callbacks=model_callbacks)
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