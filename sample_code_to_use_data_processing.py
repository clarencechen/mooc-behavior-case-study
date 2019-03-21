from data_processing import *
from mooc_model import *
from mooc_loader import *

COURSE_NAMES = ['DelftX_AE1110x_2T2015', \
'BerkeleyX_BJC.3x_1T2016', 'BerkeleyX_ColWri2.3x_1T2016', 'BerkeleyX_ColWri.3.12x_1T2016', 'BerkeleyX_ColWri3.3x_2T2016', 'BerkeleyX_EE40LX_2T2015', \
'BerkeleyX_BJC.4x_1T2016', 'BerkeleyX_ColWri.3.10_1T2016', 'BerkeleyX_ColWri3.1x_2_3T2015', 'BerkeleyX_ColWri3.4x_1T2015', 'BerkeleyX_Fin101x_1T2016', \
'BerkeleyX_ColWri2.2x_1T2016', 'BerkeleyX_ColWri.3.11_1T2016', 'BerkeleyX_ColWri3.2x_2T2016', 'BerkeleyX_CS169.2x_1T2016', 'BerkeleyX_Policy01x_1T2016']
DATA_DIR = '../../mooc-data/'
dataset_names = ['{}{}_parsed_v2.tsv'.format(DATA_DIR, course) for course in COURSE_NAMES]
sequence_max_len = 256

my_verticals = Vertical_Output(dataset_names[0])
my_verticals.current_full_indices, my_verticals.current_full_indices_userids = my_verticals.create_full_indices_based_on_pre_index_data_ignoring_time_spent()
my_verticals.prepend_1_to_current_full_indices()

print("Number of users registered: ", len(my_verticals.current_full_indices))
print("Example sequence for student 5: ", my_verticals.current_full_indices[5])

X, y = my_verticals.expose_x_y(max_len=sequence_max_len)
print("Length of exposed X:", len(X))
print("Length of a sample sequence:", len(X[20]))

#Step 3: Build a Keras LSTM Model and train on data from the Step 2 Bridge.

print("Building keras model and attempting to train...")
my_keras_model = MOOC_Keras_Model(my_verticals.pre_index_data.vertical_index.max())

my_keras_model.create_basic_transformer_model(lrate=2e-3, layers=4, embed_dim=128, seq_len=256, model_load_path='transformer_weights')
#my_keras_model.create_basic_lstm_model(lrate=2e-3, layers=2, hidden_size=128, embed_dim=128, seq_len=256)

my_keras_model.set_model_name('Baseline_Input_Output')

train_proportion = 0.9
train_index = int(len(X)*train_proportion)
train_x, val_x = X[:train_index], X[train_index:]
train_y, val_y = y[:train_index], y[train_index:]

my_keras_model.transformer_model_fit(train_x, train_y, val_x, val_y, epoch_limit=20, batch_size=128, model_save_path='transformer_weights', tensorboard_log_path='tensorboard_logs')

'''
for i in range(1, len(COURSE_NAMES)):
	test_verticals = Vertical_Output(dataset_names[1])
	test_verticals.current_full_indices, test_verticals.current_full_indices_userids = test_verticals.create_full_indices_based_on_pre_index_data_ignoring_time_spent()
	test_verticals.prepend_1_to_current_full_indices()
	print("Number of users registered: ", len(my_verticals.current_full_indices))
	print("Example sequence for student 5: ", my_verticals.current_full_indices[5])
	test_x, test_y = my_verticals.expose_x_y(max_len=sequence_max_len)
	print("Testing model on the following dataset: {}".format(COURSE_NAMES[i]))
	my_keras_model.transformer_model_fit(test_x, test_y, batch_size=128)
'''

#my_keras_model.early_stopping_model_fit(train_x, train_y, val_x, val_y, loss_nonimprove_limit = 3)

#Step 4: Build a recommendation oracle or other downstream task that utilizes the keras model.
