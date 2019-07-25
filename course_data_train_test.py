from course_data_processing import *
from mooc_transformer_model import *
from mooc_lstm_model import *
from mooc_loader import *

dataset_name = 'stu_sem_grade_condense.pkl'
train_proportion, val_proportion = 0.81, 0.09

my_verticals = UCB_Course_Data_Vertical('../../{}'.format(dataset_name))
sequence_len = my_verticals.num_semesters -1
vocab_size = my_verticals.num_courses
train_x, train_y, val_x, val_y, test_x, test_y = my_verticals.expose_course_x_y(train_proportion, val_proportion)

#Step 2: Build a Keras LSTM Model and train on data from the Step 2 Bridge.
'''
print("Now training baseline LSTM Model for {}:".format(dataset_name))
lstm_model = MOOC_LSTM_Model(vocab_size, multihot_input=True)
lstm_model.create_lstm_model(lrate=0.001, layers=1, embed_dim=256, seq_len=sequence_len, \
    lstm_dropout=0, confidence_penalty_weight=0, use_tied_embedding=False, model_load_path=None)
lstm_model.early_stopping_fit(train_x, train_y, val_x, val_y, batch_size=128,\
    model_save_path='lstm_weights_baseline_{}'.format(dataset_name))
lstm_model.test_set_eval(test_x, test_y, batch_size=1024)
'''
print("Now training LSTM Model with confidence penalty for {}:".format(dataset_name))
lstm_confidence_penalty = MOOC_LSTM_Model(vocab_size, multihot_input=True)
lstm_confidence_penalty.create_lstm_model(lrate=0.001, layers=1, embed_dim=256, seq_len=sequence_len, \
    confidence_penalty_weight=0.1, use_tied_embedding=True, model_load_path=None)
lstm_confidence_penalty.early_stopping_fit(train_x, train_y, val_x, val_y, batch_size=128,\
    model_save_path='lstm_weights_penalty_{}'.format(dataset_name))
lstm_confidence_penalty.test_set_eval(test_x, test_y, batch_size=1024)
'''
print("Now training LSTM Model with tied embeddings for {}:".format(dataset_name))
lstm_tied_embeddings = MOOC_LSTM_Model(vocab_size, multihot_input=True)
lstm_tied_embeddings.create_lstm_model(lrate=0.001, layers=1, embed_dim=256, seq_len=sequence_len, \
    confidence_penalty_weight=0.1, use_tied_embedding=False, model_load_path=None)
lstm_tied_embeddings.early_stopping_fit(train_x, train_y, val_x, val_y, batch_size=128,\
    model_save_path='lstm_weights_tied_embedding_{}'.format(dataset_name))
lstm_tied_embeddings.test_set_eval(test_x, test_y, batch_size=1024)
'''
print("Now training enhanced LSTM Model for {}:".format(dataset_name))
lstm_enhanced = MOOC_LSTM_Model(vocab_size, multihot_input=True)
lstm_enhanced.create_lstm_model(lrate=0.001, layers=1, embed_dim=256, seq_len=sequence_len, \
    confidence_penalty_weight=0.1, use_tied_embedding=True, model_load_path=None)
lstm_enhanced.early_stopping_fit(train_x, train_y, val_x, val_y, batch_size=128,\
    model_save_path='lstm_weights_enhanced_{}'.format(dataset_name))
lstm_enhanced.test_set_eval(test_x, test_y, batch_size=1024)

#Step 3: Build a Keras Transformer Model and train on same data as the LSTM from Step 2.

print("Now training Transformer Model for {}:".format(dataset_name))
transformer_model = MOOC_Transformer_Model(vocab_size, multihot_input=True)
transformer_model.create_basic_transformer_model(lrate=0.001, layers=4, embed_dim=256, seq_len=sequence_len, \
    model_load_path=None)
transformer_model.early_stopping_fit(train_x, train_y, val_x, val_y, batch_size=128, \
    use_cosine_lr=True, model_save_path='transformer_weights_{}'.format(dataset_name), tensorboard_log_path='./tensorboard/')
transformer_model.test_set_eval(test_x, test_y, batch_size=1024)
