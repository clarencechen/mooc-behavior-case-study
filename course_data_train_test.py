from data_processing_course import *
from mooc_transformer_model import *
from mooc_lstm_model import *
from mooc_loader import *

dataset_name = '../../stu_sem_grade_condense.pkl'
train_proportion, val_proportion = 0.63, 0.07
sequence_max_len = 32

my_verticals = UCB_Course_Data_Vertical(dataset_name)
train_data, val_data, test_data = my_verticals.expose_course_x_y(train_proportion, val_proportion)

