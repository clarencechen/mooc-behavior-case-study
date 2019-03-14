import datetime
import json
import numpy as np
import keras
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

    def create_basic_transformer_model(self, lrate, layers=4, embed_dim=128, seq_len=256, model_load_path=None):
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

    def create_basic_lstm_model(self, layers, lrate, hidden_size, opt, embed_dim, seq_len, model_save_path=None):
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
        x = Embedding(input_dim=self.embedding_vocab_size+1, output_dim=embed_dim, input_length=seq_len, mask_zero=True)(main_input)
        for i in range(layers):
            print("adding layer " + str(i))
            x = LSTM(hidden_size, dropout_W = 0.2, return_sequences = True)(x)
        main_loss = TimeDistributed(Dense(embedding_vocab_size, activation='softmax'))(x)
        model = Model(input=[main_input], output = [main_loss])
        model.compile(loss='categorical_crossentropy', optimizer=opt(lr=lrate), metrics=['accuracy'])

        if model_save_path is not None and os.path.exists(model_save_path):
            model.load_weights(model_save_path, skip_mismatch=True, by_name=True)
            load_optimizer_weights(model, model_save_path)
            print('Old model from {} successgully loaded.\n'.format(model_save_path))
            model.summary()
            self.model = model
            return

        print('Compiled new model.\n')
        print('-'*80)
        model.summary()
        self.model = model

    def transformer_model_fit(self, train_x, train_y, val_x, val_y, test_x, test_y, epoch_limit=100, batch_size=64, model_save_path=None, tensorboard_log_path=None):

        assert self.model_params is not None

        lr_scheduler = callbacks.LearningRateScheduler(
            CosineLRSchedule(lr_high=self.model_params['lrate'], lr_low=self.model_params['lrate'] / 32, initial_period=epoch_limit),
            verbose=1)
        model_callbacks = [lr_scheduler]
        
        if model_save_path is not None:
            model_callbacks.append(callbacks.ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=True))
        if tensorboard_log_path is not None:
            model_callbacks.append(callbacks.TensorBoard(tensorboard_log_path))
        
        self.model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=batch_size, epochs=epoch_limit, callbacks=model_callbacks)

        # Evaluation using test set
        print('-' * 80)
        test_metrics = self.model.evaluate(test_x, test_y, batch_size=batch_size)
        for metric_name, metric_value in zip(model.metrics_names, test_metrics):
            print('Test {}: {.6f}'.format(metric_name, metric_value))

    def early_stopping_model_fit(self, train_x, train_y, validation_data, epoch_limit = 200, loss_nonimprove_limit = 3, batch_size=64, save_models_to_folder = None):
        """
        """
        early_stopping_met = False
        for i in range(epoch_limit):
            print("epoch:", i)
            current_history = self.model.fit(train_x, train_y, batch_size = batch_size, nb_epoch = 1, validation_data = validation_data)
            current_history = current_history.history
            validation_loss = current_history['val_loss'][0]
            validation_accuracy_dictionary = self.compute_validation_accuracy(validation_data, b_s = batch_size)
            average_of_average_accuracy = np.mean(validation_accuracy_dictionary['averages'])
            accuracy = validation_accuracy_dictionary['accuracy']
            self.previous_val_loss.append(validation_loss)
            if len(self.previous_val_loss) > loss_nonimprove_limit:
                min_val_loss = min(self.previous_val_loss)
                recent_losses = self.previous_val_loss[-loss_nonimprove_limit-1:]
                print(recent_losses)
                if min(recent_losses) > min_val_loss:
                    early_stopping_met = True
                if validation_loss == min_val_loss:
                    self.best_epoch = i
                    self.best_average_of_average_accuracy = average_of_average_accuracy
                    self.best_accuracy = accuracy
            if early_stopping_met:
                print("Early stopping reached.")
                print("Best epoch according to validation loss:", self.best_epoch)
                print("Best epoch's accuracy:", self.best_accuracy)
                print("Best epoch's average accuracy:", self.best_average_of_average_accuracy)
                return

    def compute_validation_accuracy(self, validation_data, b_s = 64):
        """
        """
        validation_x, validation_y = validation_data
        just_x_indices = validation_x
        if isinstance(validation_x, list):
            just_x_indices = validation_x[0]
        
        predictions = self.model.predict(validation_x, batch_size = b_s)

        per_student_accuracies = np.zeros(len(just_x_indices))
        total_correct_predictions = 0
        
        for student_sequence_index, current_x in enumerate(just_x_indices):
            current_studen_preds = np.max(predictions[student_sequence_index], axis=-1)
            current_student_correct = np.count_nonzero(np.where(current_studen_preds == validation_y[student_sequence_index]))
            total_predictions += current_student_correct
            total_correct_predictions += len(validation_y[student_sequence_index])
            acc = float(current_student_correct) / len(validation_y[student_sequence_index])
            per_student_accuracies[current_x] = acc

        total_val_acc = float(total_correct_predictions) / total_predictions
        print("Total validation accuracy:", total_val_acc)
        print("Average accuracy:", np.mean(per_student_accuracies))
        return_dict = {}
        return_dict['accuracy'] = total_val_acc
        return_dict['averages'] = per_student_accuracies
        return return_dict

"""
    def simple_model_fit(self, epochs = 10, batch_size = 64, validation_proportion = 0.1):
        for i in range(epochs):
            print("epoch:", i)
            validation_i = int(len(self.X) * .9)
            hist = self.model.fit(self.X[:validation_i], self.y[:validation_i], batch_size = batch_size, nb_epoch = 1, validation_data = (self.X[validation_i:], self.y[validation_i:]))
            self.model_histories.append(hist)
"""
