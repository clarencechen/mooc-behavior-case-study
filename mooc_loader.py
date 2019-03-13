import datetime
import pandas as pd
import json
import numpy as np
import csv
import abc

def process_time(time):
    return datetime.datetime.strptime(time[:-6], '%Y-%m-%dT%H:%M:%S.%f' if '.' in time[:-6] else '%Y-%m-%dT%H:%M:%S')

class MOOC_Data(object):
    """
    Accepts raw edx log of json objects
    Converts to pandas dataframe
    Has several filtering methods
    Ultimate output is sorted_data to be fed to an Abstract_Mapped_To_Indices_Data 
    """
    @abc.abstractmethod
    def __init__(self):
        pass

    def output_to_disk(self, output_name):
        print("Outputting sorted dataframe to disk.")
        self.sorted_data.to_csv(output_name)

    def filter_data_problem_check(self, minimum_problem_checks = 1):
        """
        Returns a COPY of sorted_data that is filtered to only include users with a problem check.
        If you want to overwrite .sorted_data, then reassign it to the returned dataframe of this method
        """
        data_df = self.sorted_data
        print("Number of rows before filtering for problem check:", len(data_df))
        only_problem_check = data_df[data_df['event_type'] == 'problem_check']
        users_with_problem_check = set(only_problem_check.username)
        df_only_users_with_problem_check = data_df[data_df['username'].isin(users_with_problem_check)].sort_values('time')
        print("Number of rows from users with a problem check:", len(df_only_users_with_problem_check))
        return df_only_users_with_problem_check

    def filter_data_navigation_only(self):
        """
        Returns a COPY of sorted_data that is filtered to only include rows from navigation events.
        """
        data_df = self.sorted_data
        print("Number of rows before filtering by only navigation:", len(data_df))
        seq_rows = data_df[data_df['event_type'].isin(['seq_goto', 'seq_next', 'seq_prev'])]
        print('seq rows', len(seq_rows))
        slash_rows = data_df[data_df['event_type'].str.startswith('/')]
        slash_rows = slash_rows[slash_rows['event_type'].str.contains('courseware')]
        print('slash rows with courseware', len(slash_rows))
        navigation_rows = pd.concat([seq_rows, slash_rows]).sort_values('time')
        print("Length of navigation rows:", len(navigation_rows))
        return navigation_rows
    
    def filter_data_by_time(self, earliest_time = datetime.datetime(1900, 1, 1, 0, 0, 0, 000000), latest_time = datetime.datetime(2100, 12, 31, 23, 59, 59, 999999)):
        """
        Returns a COPY of sorted_data that is filtered to only include rows that are between earliest_time and latest_time
        """
        data_df = self.sorted_data
        print("Length of data before filtering by time:", len(data_df))
        if 'datetime_time' not in data_df.columns:
            data_df['datetime_time'] = data_df['time'].apply(process_time)
        data_df = data_df[data_df['datetime_time'] <= latest_time]
        data_df = data_df[data_df['datetime_time'] >= earliest_time]
        print("Length of data after filtering by time:", len(data_df))
        return data_df

class MOOC_Data_From_Log(MOOC_Data):
    def __init__(self, log_file, course_axis_file):
        """
        Most important attribute is sorted_data
        sorted_data is a pandas dataframe with columns from the original 
        Attributes:
        sorted_data: DataFrame of all rows of data sorted by time.
                     Columns are: 
        course_axis: DataFrame of the course axis
        problem_check_data: DataFrame of all rows of data that are done by users with a problem check within time constraints        
        Reads in json event log and converts to pandas DataFrame
        """
        working_data = []
        fail_count = 0
        with open(log_file) as data_file:
            for line in data_file.readlines():
                try:
                    data = json.loads(line)
                    working_data.append(line)
                except:
                    fail_count+=1
                    print("Failed to read line:")
                    print(line)
                    continue
        print("Number of failed lines: ", fail_count)
        print("Length of successfuly read data: ", len(working_data))
        print("Converting to pandas dataframe now...")
        data_json_str = "[" + ','.join(working_data) + "]" #Converts json to one large string, since that is what pandas needs
        data_df = pd.read_json(data_json_str)
#        self.raw_data = data_df
        self.sorted_data = data_df.sort_values('time')
        self.course_axis = pd.read_csv(course_axis_file)
        print("Completed loading json file and converted to Pandas DataFrame processing")

class MOOC_Data_From_Disk(MOOC_Data):
    def __init__(self, dataframe_file_name, course_axis_file):
        """
        Accepts dataframe csv and converts it into pandas dataframe
        """
        self.sorted_data = pd.read_csv(dataframe_file_name)
        self.course_axis = pd.read_csv(course_axis_file)
        print("Successfully read Dataframe from disk.", dataframe_file_name)