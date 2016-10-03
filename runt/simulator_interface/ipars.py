import inspect
import numpy as np
import math
import os
import time
from copy import deepcopy

# import SimulatorClass
from .simulator_class import SimulatorClass

class IPARS_interface(SimulatorClass):
    def __init__(self):
        self.name = 'IPARS_interface'

    def write_input_file(self, param, t, path):
        raise NotImplementedError(inspect.stack()[0][3])

    def run(self, path):
        os.system('\\rm -f ' + path + '/ipars_done')
        os.system('\\rm -f ' + path + '/ipars_error')
        os.system('cd ' + path + '; ipars > /dev/null && (echo "done">ipars_done) || (echo "error">ipars_error) &')

    def check_finish(self, path, start_time = 0.0):
        return os.path.isfile(path + '/ipars_done')

    def check_break(self, path, start_time):
        file = os.path.isfile(path + '/ipars_error')
        t = (time.time()-start_time) > 300.
        return (file or t)

    def write_input_ens(self, path, ens_status, ParamArray, TimeArray):
        raise NotImplementedError(inspect.stack()[0][3])

    def read_output_ens(self, path, ens_status, well_type_time_dic, ReadTimeArray):
        EnSize = len(ens_status)

        # create path list
        path_list = []
        for i in range(EnSize):
            p = path + '/ens' + str(i+1).zfill(3)
            path_list.append(p)

        if not type(ReadTimeArray) == list and not type(ReadTimeArray) == np.ndarray:
            ReadTimeArray = [ReadTimeArray]

        dic = deepcopy(well_type_time_dic)
        for k in dic.keys():
            dic[k] = [ReadTimeArray, ]

        ObserSize = len(dic.keys())
        StepSize = len(ReadTimeArray)


        result = np.zeros((StepSize, ObserSize, EnSize))
        for i in range(EnSize):
            if ens_status[i]:
                filepath = path_list[i] + '/spe5.out'
                try:
                    result[:, :, i] = \
                            self.read_wellout_extract(filepath, dic, tolerance = 1.0)

                except:
                    ens_status[i] = False

        result = result.reshape((StepSize * ObserSize, EnSize))

        #final output result is a 2D matrix

        # write output data
        dic_path = path + '/output_data_type'
        #forecast
        if os.path.isfile(dic_path):
            dic_path = path + '/fc_output_data_type'
        f = open(dic_path, 'w')
        for k in sorted(dic.keys()):
            print>>f, k
            print>>f, dic[k]
        f.close()

        data_path = path + '/data'
        # forecast
        if os.path.isfile(data_path):
            data_path = path + '/fc_data'

        np.savetxt(data_path, result)

        return [result, ens_status]
        

    def read_wellout_raw(self, filename):
        # Index Assigned to Data Types                      
        # 1      ==> WATER INJECTION RATE
        # 2      ==> OIL PRODUCTION RATE                                               
        # 3      ==> WATER PRODUCTION RATE
        # 4      ==> GAS PRODUCTION RATE
        # 5      ==> WATER/OIL RATIO
        # 6      ==> GAS/OIL RATIO
        # 7      ==> BOTTOM-HOLE PRESSURE
        # 8      ==> GAS INJECTION RATE
        # 9      ==> OIL INJECTION RATE
        
        filelist = open(filename).readlines()

        well_data_dic = {}
        
        while True:
            try:
                textline = filelist.pop(0)
                indexline = map(int, filelist.pop(0).split())
        
                key = (indexline[0], indexline[1])
                
                try:
                    initial_len = len(well_data_dic[key][0])
                except KeyError:
                    well_data_dic[key] = [[],[]]
                    initial_len = 0
        
                for i in [0,1]:
                    while len(well_data_dic[key][i]) - initial_len < indexline[2]:
                        well_data_dic[key][i].extend(map(float, filelist.pop(0).split()))
        
            # end of file
            except IndexError:
                break

        return well_data_dic

    def read_wellout_extract(self, filename, well_type_time_dic, tolerance = 0.01):
        # return a np array, size: StepSize * ObserSize

        # well_type_time_dic:
        # key: a tuple (well number, data type)
        # value: a list with a list as only member [[TimeArray], ]
        # example: {(1,3): [[10.0, 20.0, 30.0], ]}

        # read raw data from file
        well_data_raw = self.read_wellout_raw(filename)
       
        # well_data: dictionary
        well_data = deepcopy(well_type_time_dic)


        # loop keys
        for k in well_type_time_dic.keys():
            # target_time_array is a list
            target_time_array = well_type_time_dic[k][0]



            stepsize = len(target_time_array)
            # raw_time_array is a list, usually a fine scale target_time_array
            raw_time_array = well_data_raw[k][0]
            # result_data_array is created to save target_data according to target_time_array
            result_data_array = []


            # loop target time array
            t_index = 0
            for i in range(len(raw_time_array)):

                if abs(raw_time_array[i] - target_time_array[t_index]) < tolerance:

                    result_data_array.append(well_data_raw[k][1][i])
                    t_index += 1


                    if t_index == len(target_time_array):
                        break

            try:
                assert len(target_time_array) == len(result_data_array)
                well_data[k].append(result_data_array)

            except:
                raise IndexError('Target Time Array Error')



        data_matrix = np.zeros((stepsize, len(well_data.keys())))
        i = 0
        for k in sorted(well_data.keys()):
            data_matrix[:,i] = well_data[k][1]
            i += 1

        # data_matrix size: StepSize * ObserSize

        return data_matrix

