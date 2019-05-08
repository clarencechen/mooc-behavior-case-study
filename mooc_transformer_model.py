import datetime
import json
import numpy as np
import keras
import os

from keras.optimizers import Adam
from keras.utils import np_utils

from keras.models import Model

from transformer_utils import load_optimizer_weights
from transformer_models import vanilla_transformer_gpt_model
from transformer_mooc_finetune import build_finetune_model

import keras.callbacks as callbacks

from mooc_model import *

class MOOC_Transformer_Model(MOOC_Model):
    """
    
    """
    def __init__(self, embedding_vocab_size):
        super().__init__(embedding_vocab_size)

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
                num_heads=8)

        optimizer = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, clipvalue=5.0)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
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

    def create_finetune_transformer_model(self, model_load_path, old_embedding_size, lrate=2e-3, layers=4, embed_dim=128, seq_len=256, freeze_layers=False):
        """
        Loads Vanilla Transformer model for finetuning on a new dataset
        """
        self.model_params = {'layers': layers,
            'embed_dim': embed_dim,
            'e_vocab_size': self.embedding_vocab_size,
            'seq_len': seq_len,
            'lrate': lrate}

        old_model = vanilla_transformer_gpt_model(
                max_seq_length=seq_len,
                vocabulary_size=int(old_embedding_size +1),
                word_embedding_size=embed_dim,
                transformer_depth=layers,
                num_heads=8)

        optimizer = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, clipvalue=5.0)
        old_model.compile(optimizer=optimizer, metrics=['accuracy'])
        
        # load model weights if specified
        if os.path.exists(model_load_path):
            old_model.load_weights(model_load_path, skip_mismatch=True, by_name=True)
            load_optimizer_weights(old_model, model_load_path)
            print('Old model weights from {} successfully loaded.\n'.format(model_load_path))
        else:
            raise FileNotFoundError('Please specify a valid model_load_path to load pretrained weights.')

        self.model = build_finetune_model(old_model, self.model_params, freeze_layers)
