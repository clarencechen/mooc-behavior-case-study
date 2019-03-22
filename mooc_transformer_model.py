import datetime
import json
import numpy as np
import keras
import os

from keras.regularizers import l2 as l2_reg
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Input, Softmax
from keras.models import Model
from keras.layers.embeddings import Embedding
import keras.callbacks as callbacks

from keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding
from transformer_utils import load_optimizer_weights, CosineLRSchedule
from transformer_models import vanilla_transformer_gpt_model

def build_finetune_model(old_model, model_params, freeze_layers=False):
    
    # freeze all relevant layers of old model
    if freeze_layers:
        for layer in old_model.layers[2:-2]:
            layer.trainable = False

    # build new finetuning layers
    reg = l2_reg(1e-4)
    new_course_ids = Input(shape=(int(model_params['seq_len']),), dtype='int32', name='new_ids')
    new_embedding = ReusableEmbedding(int(model_params['e_vocab_size']), int(model_params['embed_dim']), input_length=int(model_params['seq_len']), \
        name='new_embeddings', embeddings_regularizer=reg)
    
    new_tied_disembedding = TiedOutputEmbedding(projection_regularizer=reg, projection_dropout=0.6, name='new_embedding_logits')
    new_softmax = Softmax(name='new_predictions')

    # connect new finetuning layers
    fed_input, new_embedding_matrix = new_embedding(word_ids)
    fed_output = old_model.layers[2:-2](fed_input)
    new_predictions = new_softmax(new_tied_disembedding([fed_output, new_embedding_matrix]))

    # build and compule new model
    finetune_model = Model(inputs=[new_course_ids], outputs=[new_predictions])
    optimizer = Adam(lr=model_params['lrate'], beta_1=0.9, beta_2=0.999, clipvalue=5.0)
    finetune_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print('Finetuned model compiled successfully.\n')
    finetune_model.summary()
    return finetune_model

class MOOC_Transformer_Model(object):
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
            print("WARNING: Create Transformer model before setting model name.")
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

        print('Model compiled successfully; Model summary below:')
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
        old_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        # load model weights if specified
        if os.path.exists(model_load_path):
            old_model.load_weights(model_load_path, skip_mismatch=True, by_name=True)
            load_optimizer_weights(old_model, model_load_path)
            print('Old model weights from {} successfully loaded.\n'.format(model_load_path))
        else:
            raise FileNotFoundError('Please specify a valid model_load_path to load pretrained weights.')

        self.model = build_finetune_model(old_model, self.model_params, freeze_layers)

    def transformer_model_fit(self, train_x, train_y, val_x, val_y, epoch_limit=5, batch_size=64, model_save_path=None, tensorboard_log_path=None):

        assert self.model_params is not None, 'Please create model before training'
        print('Training model with params: {}'.format(self.model_params))

        lr_scheduler = callbacks.LearningRateScheduler(
            CosineLRSchedule(lr_high=self.model_params['lrate'], lr_low=self.model_params['lrate'] / 32, initial_period=10),
            verbose=1)
        model_callbacks = [lr_scheduler]
        
        if model_save_path is not None:
            model_callbacks.append(callbacks.ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1))
        if tensorboard_log_path is not None:
            model_callbacks.append(callbacks.TensorBoard(tensorboard_log_path, \
                histogram_freq=1, batch_size=batch_size, write_grads=True, write_images=True))

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

        test_metrics = self.model.evaluate(test_x, test_y, batch_size=batch_size)
        for metric_name, metric_value in zip(self.model.metrics_names, test_metrics):
            print('Test {}: {:.8f}'.format(metric_name, metric_value))

"""
    def simple_model_fit(self, epochs = 10, batch_size = 64, validation_proportion = 0.1):
        for i in range(epochs):
            print("epoch:", i)
            validation_i = int(len(self.X) * .9)
            hist = self.model.fit(self.X[:validation_i], self.y[:validation_i], batch_size = batch_size, nb_epoch = 1, validation_data = (self.X[validation_i:], self.y[validation_i:]))
            self.model_histories.append(hist)
"""
