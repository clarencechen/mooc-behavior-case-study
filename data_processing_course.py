import pickle
import numpy as np

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
            self.course_data.dump('../../berkeley_course_data.pkl')

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
            self.grade_data.dump('../../berkeley_grade_data.pkl')


            '''
            it = np.nditer([index_matrix, course_data, grade_data], flags=['external_loop', 'refs_ok'], op_flags=[['readonly'], ['writeonly'], ['writeonly']], op_axes = [[0, 1], [0, 1], [0, 1]])
            with it:
                #flattened arrays by student and semester:
                for (semesters, courses, grades) in it:
                    print(semesters.shape)
                    for i in range(len(semesters)):
                        print(courses[i].shape)
                        if(len(semesters[i])) > 0:
                            temp = np.array(semesters[i], dtype='int32')
                            # 0 values do not exist in courses or grades
                            courses[i] = np.sum(np.eye(self.num_courses)[temp[:,0] -1], axis=0)
                            grades[i] = np.sum(np.eye(self.num_grades)[temp[:,1] -1], axis=0)
                        else:
                            courses[i] = np.zeros(self.num_courses)
                            grades[i] = np.zeros(self.num_grades)
            for i in range(self.num_students):
                for t in range(self.num_semesters):
                    if len(self.course_grade_data[i][t]) > 0:
                        # semester is a rectangular np array
                        semester = np.array(self.course_grade_data[i][t], dtype='int32')
                        # 0 values do not exist in courses
                        course_data[i, t, :] = np.sum(np.eye(self.num_courses)[semester[:,0] -1], axis=0)
                        # 0 values do not exist in grades
                        grade_data[i, t, :] = np.sum(np.eye(self.num_grades)[semester[:,1] -1], axis=0)
                    else:
                        course_data[i, t, :] = np.zeros((self.num_courses))
                        grade_data[i, t, :] = np.zeros((self.num_grades))
            '''

            self.course_x, self.course_y = course_data[:,:-1,:], course_data[:,1:,:]

    def expose_course_x_y(self, train_proportion = 0.63, val_proportion = 0.07):
        train_index, val_index = int(len(self.course_x)*train_proportion), int(len(self.course_x)*(train_proportion +val_proportion))
        train_x, val_x, test_x = self.course_x[:train_index], self.course_x[train_index:val_index], self.course_x[val_index:] 
        train_y, val_y, test_y = self.course_y[:train_index], self.course_y[train_index:val_index], self.course_y[val_index:]
        return (train_x, train_y), (val_x, val_y), (test_x, test_y)