import datetime
import json
import numpy as np
import keras
import os

from keras.optimizers import Adam
from keras.regularizers import l2 as l2_reg
from keras.layers import Input, Softmax

import keras.callbacks as callbacks
from keras.models import Model

from keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding

def build_finetune_model(old_model, model_params, freeze_layers=False):
    
    # freeze all relevant layers of old model
    trained_core = Model(inputs=[old_model.layers[2].input], outputs=[old_model.layers[-3].output])
    if freeze_layers:
        for layer in trained_core:
            layer.trainable = False
    trained_core.summary()

    # build new finetuning layers
    reg = l2_reg(1e-4)
    new_course_ids = Input(shape=(int(model_params['seq_len']),), dtype='int32', name='new_ids')
    new_embedding = ReusableEmbedding(int(model_params['e_vocab_size']), int(model_params['embed_dim']), input_length=int(model_params['seq_len']), \
        name='new_embeddings', embeddings_regularizer=reg)
    new_tied_disembedding = TiedOutputEmbedding(projection_regularizer=reg, projection_dropout=0.6, name='new_embedding_logits')
    new_softmax = Softmax(name='new_predictions')

    # connect new finetuning layers
    fed_input, new_embedding_matrix = new_embedding(new_course_ids)
    fed_output = trained_core(fed_input)
    new_predictions = new_softmax(new_tied_disembedding([fed_output, new_embedding_matrix]))

    # build and compule new model
    finetune_model = Model(inputs=[new_course_ids], outputs=[new_predictions])
    optimizer = Adam(lr=model_params['lrate'], beta_1=0.9, beta_2=0.999, clipvalue=5.0)
    finetune_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print('Finetuned model compiled successfully.\n')
    finetune_model.summary()
    return finetune_model
