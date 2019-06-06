import datetime
import json
import numpy as np
import keras
import os

from keras.optimizers import Adam
from keras.utils import np_utils

from keras.models import Model

from transformer_models import vanilla_transformer_gpt_model

import keras.callbacks as callbacks

from mooc_model import *

class MOOC_Transformer_Model(MOOC_Model):
    """
    
    """
    def __init__(self, embedding_vocab_size, **kwargs):
        super().__init__(embedding_vocab_size, **kwargs)

    def create_basic_transformer_model(self, lrate=2e-3, layers=4, embed_dim=128, seq_len=256, model_load_path=None):
        """
        Returns a Vanilla Transformer model
        """
        self.model_params = {'layers': layers,
            'embed_dim': embed_dim,
            'e_vocab_size': self.embedding_vocab_size,
            'seq_len': seq_len,
            'lrate': lrate}

        model = vanilla_transformer_gpt_model(
                max_seq_length=seq_len,
                vocabulary_size=self.embedding_vocab_size,
                word_embedding_size=embed_dim,
                transformer_depth=layers,
                num_heads=8,
                multihot_input=self.multihot_input)

        optimizer = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, clipvalue=5.0)
        self.compile_and_load(optimizer, model_load_path)