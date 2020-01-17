import argparse
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


parser = argparse.ArgumentParser(description='Launch Course Trajectory Modeling Script.')
parser.add_argument('--data_dir', type=str, default='mooc_data')
parser.add_argument('--weights_save_dir', type=str, default='weights')
parser.add_argument('--weights_load_dir', type=str, required=False)
parser.add_argument('--embeddings_dir', type=str, default='embeddings')
parser.add_argument('-e', '--lstm_tied_embeds', type=str, action='store_true')
parser.add_argument('-d', '--embed_dim', type=int, default=128)
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-l', '--layers', type=int, default=2)
args = parser.parse_args()

for i in range(6):

    my_verticals = Vertical_Output(args.data_dir, COURSE_NAMES[i], HAS_HEADER[i])

    old_embedding_size = int(my_verticals.pre_index_data.vertical_index.max())

    my_verticals.create_full_indices_based_on_pre_index_data_ignoring_time_spent()
    my_verticals.prepend_1_to_current_full_indices()

    print("Number of users registered for coursename {}: {}".format(COURSE_NAMES[i], len(my_verticals.current_full_indices)))
    print("Number of vertical page ids for coursename {}: {}".format(COURSE_NAMES[i], old_embedding_size))

    print("Example index sequence for student 5: ", my_verticals.current_full_indices[5])
    print("Example time_spent sequence for student 5: ", my_verticals.current_full_time_spent[5])

    train_x, train_y, val_x, val_y, test_x, test_y = my_verticals.expose_x_y(seq_len=sequence_max_len, train_proportion=train_proportion, val_proportion=val_proportion)

    if args.weights_load_dir is not None:
        lstm_load_path, transformer_load_path = './{}/lstm_baseline/'.format(args.weights_load_dir), './{}/transformer/'.format(args.weights_load_dir)
    else:
        lstm_load_path, transformer_load_path = None, None
    #Step 2: Build a Keras LSTM Model and train on data from the Step 2 Bridge.

    print("Now training baseline a LSTM Model for {}:".format(COURSE_NAMES[i]))
    lstm_model = MOOC_LSTM_Model(old_embedding_size, embed_dim=args.embed_dim, seq_len=sequence_max_len, layers=args.layers, \
        lrate=0.01, model_load_path=lstm_load_path, use_tied_embedding=args.lstm_tied_embeds)
    lstm_model.early_stopping_fit(train_x, train_y, val_x, val_y, \
        model_save_path='./{}/lstm_baseline/{}.h5'.format(args.weights_save_dir, COURSE_NAMES[i]))
    lstm_model.test_set_eval(test_x, test_y, batch_size=args.batch_size)
    lstm_model.extract_embedding_weights('./{}/lstm_baseline/{}'.format(args.embeddings_dir, COURSE_NAMES[i]))
    keras.backend.clear_session()
    
    #Step 3: Build a Keras Transformer Model and train on same data as the LSTM from Step 2.
    print("Now training first Transformer Model for {}:".format(COURSE_NAMES[i]))
    transformer_model = MOOC_Transformer_Model(old_embedding_size, embed_dim=args.embed_dim, seq_len=sequence_max_len, layers=args.layers, \
        lrate=0.0005, model_load_path=transformer_load_path)
    transformer_model.early_stopping_fit(train_x, train_y, val_x, val_y, batch_size=2*args.batch_size, \
        use_cosine_lr=False, model_save_path='./{}/transformer/{}.h5'.format(args.weights_save_dir, COURSE_NAMES[i]))
    transformer_model.test_set_eval(test_x, test_y, batch_size=2*args.batch_size)
    transformer_model.extract_embedding_weights('./{}/transformer/{}'.format(args.embeddings_dir, COURSE_NAMES[i]))
    keras.backend.clear_session()
