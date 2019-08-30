import keras

from mooc_data_processing import *
from mooc_transformer_model import *
from mooc_lstm_model import *
from mooc_loader import *

COURSE_NAMES = ['DelftX_AE1110x_2T2015', 'BerkeleyX_EE40LX_2T2015', 'BerkeleyX_Fin101x_1T2016', \
    'BerkeleyX_ColWri2.2x_1T2016', 'BerkeleyX_CS169.2x_1T2016', 'BerkeleyX_Policy01x_1T2016']

HAS_HEADER = [True, False, True, False, True, True]
sequence_max_len = 256
train_proportion, val_proportion = 0.63, 0.07

for i in range(6):

    my_verticals = Vertical_Output(COURSE_NAMES[i], HAS_HEADER[i])

    old_embedding_size = int(my_verticals.pre_index_data.vertical_index.max())

    my_verticals.create_full_indices_based_on_pre_index_data_ignoring_time_spent()
    my_verticals.prepend_1_to_current_full_indices()

    print("Number of users registered for coursename {}: {}".format(COURSE_NAMES[i], len(my_verticals.current_full_indices)))
    print("Number of vertical page ids for coursename {}: {}".format(COURSE_NAMES[i], old_embedding_size))

    print("Example index sequence for student 5: ", my_verticals.current_full_indices[5])
    print("Example time_spent sequence for student 5: ", my_verticals.current_full_time_spent[5])

    train_x, train_y, val_x, val_y, test_x, test_y = my_verticals.expose_x_y(seq_len=sequence_max_len, train_proportion=train_proportion, val_proportion=val_proportion)

    #Step 2: Build a Keras LSTM Model and train on data from the Step 2 Bridge.
    '''
    print("Now training a LSTM Model with confidence penalty for {}:".format(COURSE_NAMES[i]))
    lstm_model = MOOC_LSTM_Model(old_embedding_size, embed_dim=128, seq_len=sequence_max_len, layers=2, \
        lrate=0.01, model_load_path=None, confidence_penalty_weight=0, use_tied_embedding=False)
    lstm_model.early_stopping_fit(train_x, train_y, val_x, val_y, \
        model_save_path='./weights/lstm_weights_conf_penalty_{}'.format(COURSE_NAMES[i]))
    lstm_model.test_set_eval(test_x, test_y, batch_size=64)
    lstm_model.extract_embedding_weights('./embeddings/lstm_weights_conf_penalty_{}'.format(COURSE_NAMES[i]))
    keras.backend.clear_session()

    print("Now training an LSTM Model with both enhancements for {}:".format(COURSE_NAMES[i]))
    lstm_tied_embeddings = MOOC_LSTM_Model(old_embedding_size, embed_dim=128, seq_len=sequence_max_len, layers=2, \
        lrate=0.01, model_load_path=None, confidence_penalty_weight=0, use_tied_embedding=True)
    lstm_tied_embeddings.early_stopping_fit(train_x, train_y, val_x, val_y, \
        model_save_path='./weights/lstm_weights_enhanced_{}'.format(COURSE_NAMES[i]))
    lstm_tied_embeddings.test_set_eval(test_x, test_y, batch_size=64)
    lstm_tied_embeddings.extract_embedding_weights('./embeddings/lstm_weights_enhanced_{}'.format(COURSE_NAMES[i]))
    keras.backend.clear_session()
    '''
    #Step 3: Build a Keras Transformer Model and train on same data as the LSTM from Step 2.
    print("Now training first Transformer Model for {}:".format(COURSE_NAMES[i]))
    transformer_model = MOOC_Transformer_Model(old_embedding_size, embed_dim=128, seq_len=sequence_max_len, layers=4, \
        lrate=1e-3, model_load_path=None)
    transformer_model.early_stopping_fit(train_x, train_y, val_x, val_y, batch_size=128, \
        use_cosine_lr=False, model_save_path='./weights/transformer_weights_{}'.format(COURSE_NAMES[i]))
    transformer_model.test_set_eval(test_x, test_y, batch_size=128)
    transformer_model.extract_embedding_weights('./embeddings/transformer_weights_{}'.format(COURSE_NAMES[i]))
    keras.backend.clear_session()
