from data_processing import *
from mooc_transformer_model import *
from mooc_lstm_model import *
from mooc_loader import *

COURSE_NAMES = ['DelftX_AE1110x_2T2015', 'BerkeleyX_EE40LX_2T2015', 'BerkeleyX_Fin101x_1T2016', \
    'BerkeleyX_ColWri2.2x_1T2016', 'BerkeleyX_CS169.2x_1T2016', 'BerkeleyX_Policy01x_1T2016']

HAS_HEADER = [True, False, True, False, True, True]
DATA_DIR = '../../mooc-data/'
dataset_names = ['{}{}_parsed_v2.tsv'.format(DATA_DIR, course) for course in COURSE_NAMES]
sequence_max_len = 256
train_proportion, val_proportion = 0.63, 0.07

for i in range(6):

    my_verticals = Vertical_Output(dataset_names[i], HAS_HEADER[i])

    old_embedding_size = my_verticals.pre_index_data.vertical_index.max()

    my_verticals.create_full_indices_based_on_pre_index_data_ignoring_time_spent()
    my_verticals.prepend_1_to_current_full_indices()

    print("Number of users registered for coursename {}: {}".format(COURSE_NAMES[i], len(my_verticals.current_full_indices)))
    print("Number of vertical page ids for coursename {}: {}".format(COURSE_NAMES[i], old_embedding_size))

    print("Example index sequence for student 5: ", my_verticals.current_full_indices[5])
    print("Example time_spent sequence for student 5: ", my_verticals.current_full_time_spent[5])

    old_X, old_y = my_verticals.expose_x_y(max_len=sequence_max_len)

    train_index, val_index = int(len(old_X)*train_proportion), int(len(old_X)*(train_proportion +val_proportion))
    train_x, val_x, test_x = old_X[:train_index], old_X[train_index:val_index], old_X[val_index:] 
    train_y, val_y, test_y = old_y[:train_index], old_y[train_index:val_index], old_y[val_index:]


    #Step 2: Build a Keras LSTM Model and train on data from the Step 2 Bridge.
    print("Now training first enhanced LSTM Model for {}:".format(COURSE_NAMES[i]))
    lstm_model = MOOC_LSTM_Model(old_embedding_size)
    lstm_model.create_lstm_model(lrate=0.01, layers=2, embed_dim=128, seq_len=sequence_max_len, \
        model_load_path=None, confidence_penalty_weight=0.1, use_tied_embedding=False)
    lstm_model.early_stopping_fit(train_x, train_y, val_x, val_y, \
        model_save_path='lstm_weights_penalty_only_{}'.format(COURSE_NAMES[i]))
    lstm_model.test_set_eval(test_x, test_y, batch_size=64)

    lstm_model_2 = MOOC_LSTM_Model(old_embedding_size)
    print("Now training second enhanced LSTM Model for {}:".format(COURSE_NAMES[i]))
    lstm_model_2.create_lstm_model(lrate=0.01, layers=2, embed_dim=128, seq_len=sequence_max_len, \
        model_load_path=None, use_tied_embedding=True)
    lstm_model_2.early_stopping_fit(train_x, train_y, val_x, val_y, \
        model_save_path='lstm_weights_tied_embedding_only_{}'.format(COURSE_NAMES[i]))
    lstm_model_2.test_set_eval(test_x, test_y, batch_size=64)

'''
    #Step 3: Build a Keras Transformer Model and train on same data as the LSTM from Step 2.
    print("Now training first Transformer Model for {}:".format(COURSE_NAMES[i]))
    transformer_model = MOOC_Transformer_Model(old_embedding_size)
    transformer_model.create_basic_transformer_model(lrate=1e-3, layers=4, embed_dim=128, seq_len=sequence_max_len, \
        model_load_path=None)
    transformer_model.early_stopping_fit(train_x, train_y, val_x, val_y, batch_size=128, \
        use_cosine_lr=True, model_save_path='transformer_weights_{}'.format(COURSE_NAMES[i]))

    transformer_model.test_set_eval(test_x, test_y, batch_size=128)

new_verticals = Vertical_Output(dataset_names[1])
new_verticals.current_full_indices = new_verticals.create_full_indices_based_on_pre_index_data_ignoring_time_spent()
new_verticals.prepend_1_to_current_full_indices()

new_X, new_y = new_verticals.expose_x_y(max_len=sequence_max_len)
print("Length of exposed X for dataset with name {}: {}".format(dataset_names[1], len(new_X)))
print("Length of a sample sequence:", len(new_X[20]))

train_proportion = 0.9
train_index = int(len(new_X)*train_proportion)
train_x, val_x = new_X[:train_index], new_X[train_index:]
train_y, val_y = new_y[:train_index], new_y[train_index:]

#Step 4: Load an old Keras Transformer Model and finetune it on new dataset with new embedding layers
print("Now fine-tuning Transformer Model on dataset named {} ...".format(COURSE_NAMES[1]))
finetune_model = MOOC_Transformer_Model(new_verticals.pre_index_data.vertical_index.max())
finetune_model.create_finetune_transformer_model('transformer_weights_{}'.format(COURSE_NAMES[0]), old_embedding_size, \
    lrate=1e-4, layers=4, embed_dim=128, seq_len=sequence_max_len, freeze_layers=True)
transformer_model.transformer_model_fit(train_x, train_y, val_x, val_y, epoch_limit=20, batch_size=32, \
    model_save_path='transformer_weights_{}'.format(COURSE_NAMES[1]), tensorboard_log_path='finetune_logs')
'''

#Step 5: Build a recommendation oracle or other downstream task that utilizes the keras model.
