import pickle
import numpy as np
import keras

from keras.preprocessing import sequence as sequence

class UCB_Course_Data_Vertical(object):
    def __init__(self, input_file):
            with open(input_file, 'rb') as file:
                student_data = pickle.load(file)
            self.course_grade_data = student_data['stu_sem_grade_condense']
            
            print('Done reading course_grade_data.')

            self.num_students = len(self.course_grade_data)
            self.num_semesters = np.max([len(student) for student in self.course_grade_data])
            self.num_courses = np.max(np.max(self.course_grade_data))
            self.num_grades = 7

            print('Number of students: {0} \t Total semesters per student: {1} \t Number of courses: {2}'.format(self.num_students, self.num_semesters, self.num_courses))

            index_matrix = np.array(self.course_grade_data)

            self.course_data = np.zeros((self.num_students, self.num_semesters, self.num_courses))
            self.grade_data = np.zeros((self.num_students, self.num_semesters, self.num_grades))


            it_1 = np.nditer(index_matrix, ['multi_index', 'refs_ok'], ['readonly'])
            with it_1:
                one_hot_mold = np.eye(self.num_courses)
                while not it_1.finished:
                    elem = it_1[0].item()
                    # check for empty semesters
                    if len(elem) > 0:
                        temp_arr = np.array(elem, dtype='int32')
                        courses_one_hot = np.sum(one_hot_mold[temp_arr[:,0] -1], axis=0)
                        self.course_data[it_1.multi_index[0], it_1.multi_index[1], :] = courses_one_hot
                    it_1.iternext()

            print('Done parsing course_data.')
            print('Example course sequence for student 5: ', self.course_data[5])

            '''
            it_2 = np.nditer(index_matrix, ['multi_index', 'refs_ok'], ['readonly'])
            with it_2:
                one_hot_mold = np.eye(self.num_grades)
                while not it_2.finished:
                    elem = it_2[0].item()
                    # check for empty semesters
                    if len(elem) > 0:
                        temp_arr = np.array(elem, dtype='int32')
                        grades_one_hot = np.sum(one_hot_mold[temp_arr[:,1] -1], axis=0)
                        self.grade_data[it_2.multi_index[0], it_2.multi_index[1], :] = grades_one_hot
                    it_2.iternext()

            print('Done parsing grade_data.')
            print('Example grade sequence for student 5: ', self.grade_data[5])
            '''

    def expose_course_x_y(self, train_proportion=0.63, val_proportion=0.07):
        """
        Returns train, val, and test numpy arrays based on course_data array
        """
        '''
        train_x, train_y = self.course_data[:,:-3,:], self.course_data[:,1:-2,:]
        val_x, val_y = self.course_data[:,1:-2,:], self.course_data[:,2:-1,:]
        test_x, test_y = self.course_data[:,2:-1,:], self.course_data[:,3:,:]
        '''

        train_index, val_index = int(self.num_students*train_proportion), int(self.num_students*(train_proportion +val_proportion))
        train_x, val_x, test_x = self.course_data[:train_index,:-1,:], self.course_data[train_index:val_index,:-1,:], self.course_data[val_index:,:-1,:] 
        train_y, val_y, test_y = self.course_data[:train_index,1:,:], self.course_data[train_index:val_index,1:,:], self.course_data[val_index:,1:,:]

        return train_x, train_y, val_x, val_y, test_x, test_y