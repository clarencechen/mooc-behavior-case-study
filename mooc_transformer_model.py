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
    Represents a Transformer Model and accompanying metadata
    """
    def __init__(self, embedding_vocab_size: int, embed_dim: int=128, seq_len: int=256, layers: int=4, lrate: float=2e-3, multihot_input=False, model_load_path=None, **kwargs):
        """
        Builds and Compiles a Vanilla Transformer model
        """
        super().__init__(embedding_vocab_size, embed_dim, seq_len, layers, lrate, multihot_input, **kwargs)

        self.model = vanilla_transformer_gpt_model(
                max_seq_length=self.model_params['seq_len'],
                vocabulary_size=self.model_params['vocab_size'],
                word_embedding_size=self.model_params['embed_dim'],
                transformer_depth=self.model_params['layers'],
                num_heads=8,
                multihot_input=self.multihot_input)

        optimizer = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, clipvalue=5.0)
        self.compile_and_load(optimizer, model_load_path)