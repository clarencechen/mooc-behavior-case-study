from keras.preprocessing import sequence as sequence
import datetime
import pandas as pd
import json
import numpy as np
import csv
import abc
import io

def process_time(time):
    return datetime.datetime.strptime(time[:-6], '%Y-%m-%dT%H:%M:%S.%f' if '.' in time[:-6] else '%Y-%m-%dT%H:%M:%S')

def construct_vertical(sequential, chapter, vertical=-1):
    return '/' + chapter + '/' + sequential + '/' + str(vertical)

def write_tb_metadata(course_name, raw_data):
    metadata_list = raw_data[['vertical_index', 'chapter_title', 'sequential_title', 'vertical_title']].sort_values('vertical_index').drop_duplicates()
    header = ['Vertical Index', 'Chapter Title', 'Section Title', 'Vertical Title']
    zero_token = ['0', 'N/A', 'N/A', 'Not In Course']

    file_obj = io.open('./embeddings/metadata/{}_tb_metadata.tsv'.format(course_name), 'w', encoding='utf-8')
    file_obj.write('\t'.join(header) + '\n')
    file_obj.write('\t'.join(zero_token) + '\n')
    metadata_list.to_csv(file_obj, sep='\t', columns=['vertical_index', 'chapter_title', 'sequential_title', 'vertical_title'], header=False, index=False)
    file_obj.close()

class Vertical_Output():
    """
    Accepts MOOC_Data object, which contains a sorted_data attribute as well as a course axis
    Creates a new internal DataFrame named pre_index_data that contains columns for userid, timestamp, and unique representation of action. Can contain additional columns.
    The unique representation of action is what should be converted to indices (for output into embedding layer).
    Outputs X y used to train model, along with vocab_size for keras model.
    """
    def __init__(self, course_name, has_header, write_metadata=False):

        input_file = '{}{}_parsed_v2.tsv'.format('../../mooc-data/', course_name)
        if has_header:
            raw_data = pd.read_csv(input_file, delimiter='\t', \
                usecols=['username', 'vertical_index', 'basic_type', 'time', 'chapter_title', 'sequential_title', 'vertical_title',])
        else:
            raw_data = pd.read_csv(input_file, delimiter='\t', \
                usecols=[0, 4, 5, 7, 8, 9, 10], names=['username', 'vertical_index', 'basic_type', 'time', 'chapter_title', 'sequential_title', 'vertical_title'], header=None)

        if write_metadata:
            write_tb_metadata(course_name, raw_data)
        
        self.pre_index_data = raw_data[raw_data['basic_type'].isin(['seq_goto', 'seq_next', 'seq_prev'])].sort_values('time')
        self.current_full_indices = []
        self.current_full_time_spent = []
        print('Done reading pre_index_data.')

    def create_full_indices_based_on_pre_index_data_ignoring_time_spent(self):
        """
        Fills pre_index_data mapped to indices (list of lists), as well as corresponding list of user ids
        """
        list_of_indices, list_of_time_spent = [], []
        grouped_by_user = self.pre_index_data.groupby('username')
        for user_id, data in grouped_by_user:
            # calculate time elapsed from last index for this user
            time_stamps = pd.to_datetime(data['time'], infer_datetime_format=True)
            time_diffs = time_stamps.diff()[1:].map(lambda x : x.total_seconds())
            bucketed_time_spent = list(pd.cut(time_diffs, bins=[0, 10, 60, 1800, np.inf], labels=False, right=True))

            list_of_indices.append(list(data['vertical_index']))
            list_of_time_spent.append([0.] + bucketed_time_spent)
        
        self.current_full_indices = list_of_indices
        self.current_full_time_spent = list_of_time_spent
        
    def prepend_1_to_current_full_indices(self):
        """
        MUTATES self.current_full_indices such that a 1 is prepended to all lists
        Will not prepend a 1 if there is already a 1 at the start
        """
        for index_seq, time_spent in zip(self.current_full_indices, self.current_full_time_spent):
            if index_seq[0] == 1:
                continue
            else:
                index_seq.insert(0, 1)
                time_spent.insert(0, 0)
                continue

    def remove_contiguous_repeats_from_pre_index_data(self, keep_highest_time_spent = True):
        """
        Returns a copy of pre_index_data where contiguous repeats are removed
        """
        grouped_by_user = self.pre_index_data.groupby('username')
        data_to_dataframe = [] #will be 4 columns, matching pre_index_data
        for user_id, data in grouped_by_user:
            previous_element = None
            for row in data.iterrows():
                index = row[0]
                values = row[1]
                u, t, vert_index = values.username, values.time, values.vertical_index
                t_spent = values.time_spent
                if not previous_element:
                    previous_element = url
                    data_to_dataframe.append([u, t, url, t_spent])
                    continue
                else:
                    if url == previous_element:
                        if keep_highest_time_spent:
                            currently_recorded_time_spent = data_to_dataframe[-1][-1]
                            if isinstance(t_spent, str):
                                #t_spent is therefore 'endtime'
                                data_to_dataframe[-1][-1] = t_spent
                            elif t_spent > currently_recorded_time_spent:
                                data_to_dataframe[-1][-1] = t_spent
                            else:
                                continue
                        continue
                    else:
                        previous_element = url
                        data_to_dataframe.append([u, t, url, t_spent])
        temp_df = pd.DataFrame(data_to_dataframe, columns = self.pre_index_data.columns)
        return temp_df

    def expose_x_y(self, seq_len=256, min_len=3, train_proportion=0.63, val_proportion=0.07):
        """
        Returns train, val, and test numpy arrays based on current_full_indices
        """
        x_seq = [seq[:-1] for seq in self.current_full_indices if len(seq) >= min_len]
        padded_x_seq = sequence.pad_sequences(x_seq, maxlen=seq_len, padding='post', truncating='post')
        
        #time_seq = [seq[:-1] for seq in self.current_full_time_spent if len(seq) >= min_len]
        #padded_time_spent_seq = sequence.pad_sequences(time_seq, maxlen=seq_len, padding='post', truncating='post')
        
        y_seq = [seq[1:] for seq in self.current_full_indices if len(seq) >= min_len]
        padded_y_seq = sequence.pad_sequences(y_seq, maxlen=seq_len, padding='post', truncating='post')
        
        y_one_hot = np.zeros((len(padded_y_seq), seq_len, int(self.pre_index_data.vertical_index.max() +1)), dtype=np.bool)
        for i, output in enumerate(padded_y_seq):
            for t, vert_index in enumerate(output):
                y_one_hot[int(i), int(t), int(vert_index)] = 1
        
        train_index, val_index = int(len(padded_x_seq)*train_proportion), int(len(y_one_hot)*(train_proportion +val_proportion))
        train_x, val_x, test_x = padded_x_seq[:train_index], padded_x_seq[train_index:val_index], padded_x_seq[val_index:] 
        train_y, val_y, test_y = y_one_hot[:train_index], y_one_hot[train_index:val_index], y_one_hot[val_index:]
        return train_x, train_y, val_x, val_y, test_x, test_y
