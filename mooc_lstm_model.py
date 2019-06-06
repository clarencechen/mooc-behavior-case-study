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

    def create_lstm_model(self, lrate=0.01, layers=2, embed_dim=128, seq_len=256, model_load_path=None, confidence_penalty_weight=0, use_tied_embedding=False, lstm_dropout=0.2):
        """
        Returns a LSTM model
        """
        self.model_params = {'layers': layers,
            'embed_dim': embed_dim,
            'e_vocab_size': self.embedding_vocab_size,
            'seq_len': seq_len,
            'lrate': lrate}

        main_input = Input(shape=(model_params['seq_len'],), dtype='int32', name='node_ids')

        # Tied Embedding Layer
        if use_tied_embedding:
            l2_regularizer = regularizers.l2(1e-6)
            embedding_layer = ReusableEmbedding(model_params['e_vocab_size'],\
                model_params['embed_dim'],\
                input_length=model_params['seq_len'],\
                name='embedding_layer',\
                # Regularization is based on paper "A Comparative Study on
                # Regularization Strategies for Embedding-based Neural Networks"
                # https://arxiv.org/pdf/1508.03721.pdf
                embeddings_regularizer=l2_regularizer)
            output_layer = TiedOutputEmbedding(projection_regularizer=l2_regularizer,\
                projection_dropout=0.6,\
                name='word_prediction_logits')
            output_softmax_layer = Softmax(name='word_predictions')
            next_step_input, embedding_matrix = embedding_layer(main_input)
        else:
            embedding_layer = Embedding(model_params['e_vocab_size'],\
                    model_params['embed_dim'],\
                    input_length=model_params['seq_len'],\
                    mask_zero=True, name='embedding_layer')
            output_layer = TimeDistributed(Dense(model_params['e_vocab_size'], activation='softmax', name='word_predictions'))
            next_step_input = embedding_layer(main_input)

        for i in range(model_params['layers']):
            next_step_input = LSTM(model_params['embed_dim'], dropout=0.2, return_sequences=True, name='LSTM_layer_{}'.format(i))(next_step_input)

        # Tied Embedding Layer
        if use_tied_embedding:
            main_loss = output_softmax_layer(output_layer([next_step_input, embedding_matrix]))
        else:
            main_loss = output_layer(next_step_input)

        model = Model(inputs=[main_input], outputs=[main_loss])
        
        # Penalty for confidence of the output distribution, as described in
        # "Regularizing Neural Networks by Penalizing Confident
        # Output Distributions" (https://arxiv.org/abs/1701.06548)
        if confidence_penalty_weight > 0:
            confidence_penalty = K.mean(
                confidence_penalty_weight *
                K.sum(word_predictions * K.log(word_predictions +K.epsilon()), axis=-1))
            model.add_loss(confidence_penalty)

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
