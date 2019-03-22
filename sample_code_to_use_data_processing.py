from data_processing import *
from mooc_transformer_model import *
from mooc_lstm_model import *
from mooc_loader import *

COURSE_NAMES = ['DelftX_AE1110x_2T2015', \
'BerkeleyX_BJC.3x_1T2016', 'BerkeleyX_ColWri2.3x_1T2016', 'BerkeleyX_ColWri.3.12x_1T2016', 'BerkeleyX_ColWri3.3x_2T2016', 'BerkeleyX_EE40LX_2T2015', \
'BerkeleyX_BJC.4x_1T2016', 'BerkeleyX_ColWri.3.10_1T2016', 'BerkeleyX_ColWri3.1x_2_3T2015', 'BerkeleyX_ColWri3.4x_1T2015', 'BerkeleyX_Fin101x_1T2016', \
'BerkeleyX_ColWri2.2x_1T2016', 'BerkeleyX_ColWri.3.11_1T2016', 'BerkeleyX_ColWri3.2x_2T2016', 'BerkeleyX_CS169.2x_1T2016', 'BerkeleyX_Policy01x_1T2016']
DATA_DIR = '../../mooc-data/'
dataset_names = ['{}{}_parsed_v2.tsv'.format(DATA_DIR, course) for course in COURSE_NAMES]
sequence_max_len = 256

my_verticals = Vertical_Output(dataset_names[0])

old_embedding_size = my_verticals.pre_index_data.vertical_index.max()

'''
my_verticals.current_full_indices, my_verticals.current_full_indices_userids = my_verticals.create_full_indices_based_on_pre_index_data_ignoring_time_spent()
my_verticals.prepend_1_to_current_full_indices()

print("Number of users registered: ", len(my_verticals.current_full_indices))
print("Example sequence for student 5: ", my_verticals.current_full_indices[5])

old_X, old_y = my_verticals.expose_x_y(max_len=sequence_max_len)

train_proportion = 0.9
train_index = int(len(old_X)*train_proportion)
train_x, val_x = old_X[:train_index], old_X[train_index:]
train_y, val_y = old_y[:train_index], old_y[train_index:]

#Step 2: Build a Keras LSTM Model and train on data from the Step 2 Bridge.
print("Now training LSTM Model...")
lstm_model = MOOC_LSTM_Model(old_embedding_size)
lstm_model.create_basic_lstm_model(lrate=0.01, layers=2, hidden_size=128, embed_dim=128, seq_len=sequence_max_len)
lstm_model.early_stopping_model_fit(train_x, train_y, val_x, val_y, \
    model_save_path='lstm_weights_{}'.format(COURSE_NAMES[0]), tensorboard_log_path='lstm_logs', loss_nonimprove_limit=3)

#Step 3: Build a Keras Transformer Model and train on same data as the LSTM from Step 2.
print("Now training first Transformer Model...")
transformer_model = MOOC_Transformer_Model(old_embedding_size)
transformer_model.create_basic_transformer_model(lrate=1e-3, layers=4, embed_dim=128, seq_len=sequence_max_len, \
    model_load_path='transformer_weights_{}'.format(COURSE_NAMES[0]))

transformer_model.transformer_model_fit(train_x, train_y, val_x, val_y, epoch_limit=20, batch_size=128, \
    model_save_path='transformer_weights_{}'.format(COURSE_NAMES[0]), tensorboard_log_path='tensorboard_logs')
'''

new_verticals = Vertical_Output(dataset_names[1])
new_verticals.current_full_indices, new_verticals.current_full_indices_userids = new_verticals.create_full_indices_based_on_pre_index_data_ignoring_time_spent()
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
finetune_model.create_finetune_transformer_model('transformer_weights_{}'.format(COURSE_NAMES[1]), old_embedding_size, \
    lrate=1e-4, layers=4, embed_dim=128, seq_len=sequence_max_len, freeze_layers=True)
transformer_model.transformer_model_fit(train_x, train_y, val_x, val_y, epoch_limit=20, batch_size=32, \
    model_save_path='transformer_weights_{}'.format(COURSE_NAMES[1]), tensorboard_log_path='finetune_logs')

#Step 5: Build a recommendation oracle or other downstream task that utilizes the keras model.
