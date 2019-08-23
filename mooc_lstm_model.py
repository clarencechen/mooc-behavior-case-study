import datetime
import json
import numpy as np
import keras
import os

from keras.optimizers import Adam
from keras.utils import np_utils

from keras.models import Model
from keras.layers import Input, Dense, Softmax
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed

from keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding
from multihot_utils import ReusableEmbed_Multihot

import keras.callbacks as callbacks
import keras.backend as K

from mooc_model import *

class MOOC_LSTM_Model(MOOC_Model):
    """
    Represents an LSTM Model and accompanying metadata
    """
    def __init__(self, embedding_vocab_size, embed_dim=128, seq_len=256, layers=2, lrate=0.01, multihot_input=False, confidence_penalty_weight=0, use_tied_embedding=False, lstm_dropout=0.2, model_load_path=None, **kwargs):
        """
        Builds and Compiles an LSTM model
        """
        super().__init__(embedding_vocab_size, embed_dim, seq_len, layers, lrate, multihot_input, **kwargs)

        # Input Layer
        if self.multihot_input:
            main_input = Input(shape=(self.model_params['seq_len'], self.model_params['vocab_size']), dtype='float', name='onehot_ids')
        else:
            main_input = Input(shape=(self.model_params['seq_len'],), dtype='int32', name='node_ids')

        # Tied Embedding Layer
        if use_tied_embedding:
            l2_regularizer = keras.regularizers.l2(1e-6)
            if self.multihot_input:
                embedding_layer = ReusableEmbed_Multihot(
                    input_dim=self.model_params['vocab_size'],
                    output_dim=self.model_params['embed_dim'],
                    input_length=self.model_params['seq_len'],
                    name='multihot_embeddings',
                    # Regularization is based on paper "A Comparative Study on
                    # Regularization Strategies for Embedding-based Neural Networks"
                    embeddings_regularizer=l2_regularizer)
            else:
                embedding_layer = ReusableEmbedding(
                    input_dim=self.model_params['vocab_size'],
                    output_dim=self.model_params['embed_dim'],
                    input_length=self.model_params['seq_len'],
                    name='token_embeddings',
                    # Regularization is based on paper "A Comparative Study on
                    # Regularization Strategies for Embedding-based Neural Networks"
                    # https://arxiv.org/pdf/1508.03721.pdf
                    embeddings_regularizer=l2_regularizer)

            output_layer = TiedOutputEmbedding(
                projection_regularizer=l2_regularizer,
                projection_dropout=0.6,
                name='next_step_logits')
            output_softmax_layer = Softmax(name='next_step_predictions')
            next_step_input, embedding_matrix = embedding_layer(main_input)
        # Regular Embedding Layer
        else:
            if self.multihot_input:
                embedding_layer = TimeDistributed(Dense(
                        units=self.model_params['embed_dim'],
                        activation=None, use_bias=False,
                        name='multihot_embeddings'))
            else:
                embedding_layer = Embedding(
                        input_dim=self.model_params['vocab_size'],
                        output_dim=self.model_params['embed_dim'],
                        input_length=self.model_params['seq_len'],
                        name='token_embeddings')
            output_layer = TimeDistributed(Dense(self.model_params['vocab_size'], 
                activation='softmax', 
                name='next_step_predictions'))
            next_step_input = embedding_layer(main_input)

        for i in range(self.model_params['layers']):
            next_step_input = LSTM(self.model_params['embed_dim'], dropout=0.2, return_sequences=True, name='LSTM_layer_{}'.format(i))(next_step_input)

        # Tied Embedding Layer
        if use_tied_embedding:
            word_predictions = output_softmax_layer(output_layer([next_step_input, embedding_matrix]))
        else:
            word_predictions = output_layer(next_step_input)

        self.model = Model(inputs=[main_input], outputs=[word_predictions])
        
        # Penalty for confidence of the output distribution, as described in
        # "Regularizing Neural Networks by Penalizing Confident
        # Output Distributions" (https://arxiv.org/abs/1701.06548)
        if confidence_penalty_weight > 0:
            confidence_penalty = K.mean(confidence_penalty_weight *
                K.sum(word_predictions * K.log(word_predictions +K.epsilon()), axis=-1))
            self.model.add_loss(confidence_penalty)

        optimizer = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, clipvalue=5.0)
        self.compile_and_load(optimizer, model_load_path)
