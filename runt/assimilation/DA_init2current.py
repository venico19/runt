import inspect
import time
from subprocess import call
import numpy as np

#import DataAssimilationClass
from .data_assimilation_class import DataAssimilationClass

class DA_init2current(DataAssimilationClass):

    def __init__(self, 
                 EnSize, 
                 ParamArray,
                 DATimeArray, 
                 FCTimeArray, 
                 EnsembleClass, 
                 AnalysisClass, 
                 SimulatorClass):

        # Ensize: number of ensembles
        self.EnSize = EnSize

        # load ParamArray
        self.ParamArray = ParamArray

        # load TimeArray
        self.DATimeArray = DATimeArray
        self.FCTimeArray = FCTimeArray
        self.TimeStepSize = DATimeArray.shape[0]

        # 
        self._ensemble = EnsembleClass
        self._analysis = AnalysisClass
        self._simulator = SimulatorClass

        # set obser noise
        self.set_obser_noise()

        # set parallel number
        self.set_parallel_num()

        # initialize ens status list
        self.ens_status = [True] * self.EnSize

    def param_initialization(self):
        # initialize param array
        pass

    def parameterization(self):
        # transform parameters
        pass

    def Obser_initialization(self):
        # initialize observation
        # read from file or run simulation
        raise NotImplementedError(inspect.stack()[0][3])

    def Dynamic2Data(self, DynamicArray):
        # define operator H
        # return DataArray
        return DynamicArray

    def param_write(self, step):
        # write analysis result

        # write parameters
        path = './output/param' + str(step).zfill(2) +'.dat'
        np.savetxt(path, self.ParamArray)

        # write ens_status
        ens_filename = './output/ens_status' + str(step).zfill(2) + '.dat'
        ens = np.array(self.ens_status).astype(int)
        np.savetxt(ens_filename, np.append(ens, sum(ens)), delimiter='\n', fmt = '%d')

    def data_write(self, step):
        # write simulation result
        path = './output/data' + str(step).zfill(2) + '.dat'
        np.savetxt(path, self.DataArray)

    def set_obser_noise(self, noise = 0.01):
        # default noise = 1%
        self.obser_noise = noise

    def set_parallel_num(self, parallel=1):
        self.parallel_num = parallel

    def DArun(self):
        # Data Assimilation Process:

        # 0) clear old output folder, mkdir new one
        path = "output" + time.strftime("%y%m%d%H%M")
        call(["mv", "output", path])
        call(["mkdir", "output"])

        # 1) intialization
        # 1.1 initialize param array
        self.param_initialization()

        # 1.2 parameterization (optional)
        self.parameterization()

        # 1.3 initialize Obser
        self.Obser_initialization()

        # 1.4 write initial param
        self.param_write(0)

        for i in range(self.TimeStepSize):

            print "updating step: ", str(i+1), "/", str(self.TimeStepSize)

            # 2 ensemble generation
            # 2.1 define SimulationTime
            SimulationTime = self.DATimeArray[i]
            
            # 2.2 define path
            path = './output/step' + str(i+1).zfill(2)
            call(["mkdir", path])

            # 2.3 write input files

            self._simulator.write_input_ens(param = self.ParamArray,
                                            t = SimulationTime,
                                            path = path,
                                            ens_status = self.ens_status)
                                            

            # 2.4 run simulation
            self.ens_status = \
                        self._ensemble.fwd_propagate(Simulator = self._simulator, 
                                                     path = path, 
                                                     parallel_num = self.parallel_num,
                                                     ens_status = self.ens_status)

            # 2.5 read output files

            [self.DynamicArray, self.ens_status] = \
                    self._simulator.read_output_ens(path = path, 
                                                    ens_status = self.ens_status,
                                                    well_type_time_dic = self.well_type_time_dic,
                                                    ReadTimeArray = SimulationTime)

            self.DataArray = self.Dynamic2Data(self.DynamicArray)
            self.data_write(i+1)

            # 2 analysis
            [AnalysisParam, AnalysisDynamic] = \
                    self._analysis.create_analysis(self.extract_ens(self.ParamArray), 
                                                   self.extract_ens(self.DynamicArray),
                                                   self.extract_ens(self.DataArray), 
                                                   self.extract_ens(self.ObserArray[i,:,:]), 
                                                   self.ObserCov[i,:,:])

            # 3 write result and assign self.Param
            self.ParamArray = self.extract_ens_inv(AnalysisParam)
            self.param_write(i+1)



    def forecast(self, FCTime, StepList = None):
        # FCTime is a list of output time list
        # FCTime[-1] is the time of forecast time ending

        if StepList == None:
            StepList = [1, self.TimeStepSize]

        #  SimulationTime
        SimulationTime = FCTime[-1]

        #  path
        for step in StepList:
            print "forecating step: ", str(step)
            # path
            path = './output/step' + str(step).zfill(2)

            # 2.3 write input files

            self._simulator.write_input_ens_fc(param = self.ParamArray,
                                            t = SimulationTime,
                                            path = path,
                                            ens_status = self.ens_status)

            # 2.4 run simulation
             
            self.ens_status = \
                        self._ensemble.fwd_propagate(Simulator = self._simulator, 
                                                     path = path, 
                                                     parallel_num = self.parallel_num,
                                                     ens_status = self.ens_status)

            # 2.5 read output files

            [self.DynamicArray, self.ens_status] = \
                    self._simulator.read_output_ens(path = path, 
                                                    ens_status = self.ens_status,
                                                    well_type_time_dic = self.well_type_time_dic,
                                                    ReadTimeArray = FCTime)

            self.DataArray = self.Dynamic2Data(self.DynamicArray)
            self.data_write(step)


    def extract_ens(self, Array):
        # Array is a numpy array
        # Array size: xxx * EnSize
        # extract ens columns according self.ens_status
        # self.ens_status is a list, length = self.EnSize

        index_list = []
        for i in range(self.EnSize):
            if self.ens_status[i]:
                index_list.append(i)

        return Array[:, index_list]

    def extract_ens_inv(self, Array):
        # Array is a numpy array
        # Array size: xxx * True_EnSize
        NewArray = np.zeros((Array.shape[0], self.EnSize))
        index = 0
        for i in range(self.EnSize):
            if self.ens_status[i]:
                NewArray[:, i] = Array[:, index]
                index += 1

        return NewArray



