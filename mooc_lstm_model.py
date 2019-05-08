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

from transformer_utils import load_optimizer_weights
from mooc_enhanced_lstm import enhanced_lstm_model

import keras.callbacks as callbacks

from mooc_model import *

class MOOC_LSTM_Model(MOOC_Model):
    """
    
    """
    def __init__(self, embedding_vocab_size):
        super().__init__(embedding_vocab_size)


    def create_lstm_model(self, use_enhancements=True, lrate=0.01, layers=2, embed_dim=128, seq_len=256, model_load_path=None):
        """
        Returns a LSTM model
        """
        self.model_params = {'layers': layers,
            'embed_dim': embed_dim,
            'e_vocab_size': self.embedding_vocab_size,
            'seq_len': seq_len,
            'lrate': lrate}

        if(use_enhancements):
            model = enhanced_lstm_model(self.model_params)
        else:
            main_input = Input(shape=(seq_len,), name='main_input', dtype='int32')
            x = Embedding(input_dim=self.embedding_vocab_size, output_dim=embed_dim, input_length=seq_len, mask_zero=True)(main_input)
            for i in range(layers):
                print("Adding layer: " + str(i))
                x = LSTM(embed_dim, dropout=0.2, return_sequences=True)(x)
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

        print('Model compiled successfully; Model summary below:\n')
        print('-'*80)
        model.summary()
        self.model = model
