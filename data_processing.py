from keras.preprocessing import sequence as sequence
import datetime
import pandas as pd
import json
import numpy as np
import csv
import abc

def process_time(time):
    return datetime.datetime.strptime(time[:-6], '%Y-%m-%dT%H:%M:%S.%f' if '.' in time[:-6] else '%Y-%m-%dT%H:%M:%S')

class Abstract_Bridge_Between_MOOC_Data_and_Embedding_Indices(object):
    
    @abc.abstractmethod
    def __init__(self, input_file):
        pass

    @abc.abstractmethod
    def expose_x_y(self):
        pass

def construct_vertical(sequential, chapter, vertical=-1):
    return '/' + chapter + '/' + sequential + '/' + str(vertical)

class Vertical_Output(Abstract_Bridge_Between_MOOC_Data_and_Embedding_Indices):
    """
    Accepts MOOC_Data object, which contains a sorted_data attribute as well as a course axis
    Creates a new internal DataFrame named pre_index_data that contains columns for userid, timestamp, and unique representation of action. Can contain additional columns.
    The unique representation of action is what should be converted to indices (for output into embedding layer).
    Outputs X y used to train model, along with vocab_size for keras model.
    """
    def __init__(self, input_file, has_header):
        if has_header:
            raw_data = pd.read_csv(input_file, delimiter='\t', \
                usecols=['username', 'vertical_index', 'basic_type', 'time', 'vertical_title',])
        else:
            raw_data = pd.read_csv(input_file, delimiter='\t', \
                usecols=[0, 4, 5, 7, 10], names=['username', 'vertical_index', 'basic_type', 'time', 'vertical_title'], header=None)

        seq_rows = raw_data[raw_data['basic_type'].isin(['seq_goto', 'seq_next', 'seq_prev'])]

        self.pre_index_data = seq_rows.sort_values('time')
        self.current_full_indices = []
        self.current_full_time_spent = []
        print('Done reading pre_index_data.')

    def populate_time_spent_in_pre_index_data(self):
        return NotImplementedError
    '''
    def create_vertical_index_to_title_mapping(self):
        vertical_index_to_title = {}
        grouped_by_title = self.pre_index_data.groupby('vertical_title')
        for title, data in grouped_by_title:
            if title not in vertical_index_to_title:
                vertical_index_to_title[title] = str(data.vertical_index.mode())
        self.vertical_index_to_title = vertical_index_to_title
    '''
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

    def populate_time_spent(self):
        """
        Returns a copy of self.pre_index_data with the time_spent column populated with integer between 0 and 3
        """
        pre_index_data = self.pre_index_data
        grouped_by_user = pre_index_data.groupby('username')
        data_to_append = []
        for user_id, data in grouped_by_user:
            new_time_sequence = []
            timestamps = [process_time(elem) for elem in list(data.time)]
            for i in range(0, len(timestamps) - 1):
                current_time = timestamps[i]
                next_time = timestamps[i+1]
                second_difference = (next_time - current_time).total_seconds()
                new_time_sequence.append(second_difference)
            new_time_sequence.append('endtime')

            i = 0
            for row in data.iterrows():
                index = row[0]
                values = row[1]
                user = values.user
                timestamp = timestamps[i]
                rep = values.unique_representation_of_event
                time_spent = new_time_sequence[i]
                data_to_append.append([user, timestamp, rep, time_spent])
                i+=1

        temp_df = pd.DataFrame(data_to_append, columns = ['user', 'time', 'vertical_index', 'time_spent'])
        return temp_df

    def expose_x_y(self, max_len = 256, min_len = 3):
        """
        Returns X, y numpy arrays based on current_full_indices
        """
        idx_windows = [seq[:-1] for seq in self.current_full_indices if len(seq) >= min_len]
        time_windows = [seq[:-1] for seq in self.current_full_time_spent if len(seq) >= min_len]
        
        y_windows = [seq[1:] for seq in self.current_full_indices if len(seq) >= min_len]
        X = sequence.pad_sequences(idx_windows, maxlen=max_len, padding='post', truncating='post')
        padded_y_windows = sequence.pad_sequences(y_windows, maxlen=max_len, padding='post', truncating='post')
        self.padded_y_windows = padded_y_windows
        y = np.zeros((len(padded_y_windows), max_len, int(self.pre_index_data.vertical_index.max() +1)), dtype=np.bool)
        for i, output in enumerate(padded_y_windows):
            for t, vert_index in enumerate(output):
                y[int(i), int(t), int(vert_index)] = 1
        return X, y
