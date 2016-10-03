import inspect
import time
from subprocess import call
import numpy as np

class DataAssimilationClass(object):

    def __init__(self, EnSize, DATimeArray, FCTimeArray, EnsembleClass, AnalysisClass, SimulatorClass):
        self.EnSize = EnSize
        self.TimeArray = TimeArray
        self.FCTimeArray = FCTimeArray
        self.TimeStepSize = TimeArray.shape[0]
        self._ensemble = EnsembleClass
        self._analysis = AnalysisClass
        self._simulator = SimulatorClass

    def param_initialization(self):
        # initialize param array
        raise NotImplementedError(inspect.stack()[0][3])

    def parameterization(self):
        # transform parameters 
        raise NotImplementedError(inspect.stack()[0][3])

    def Obser_initialization(self):
        # initialize observation
        # read from file or run simulation
        raise NotImplementedError(inspect.stack()[0][3])

    def ObserCov_initialization(self):
        #define observation error coviance 
        raise NotImplementedError(inspect.stack()[0][3])

    def Dynamic2Data(self, DynamicArray):
        # define operator H
        # return DataArray
        raise NotImplementedError(inspect.stack()[0][3])

    def param_write(self):
        # write analysis result
        raise NotImplementedError(inspect.stack()[0][3])

    def dynamic_write(self):
        # write simulation result
        raise NotImplementedError(inspect.stack()[0][3])

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

            print "updating step: ", str(i+1), "/", str(self.TimeStepSize+1)

            # 2 ensemble generation
            # 2.1 define SimulationTime
            SimulationTime = np.array([self.TimeStepSize[i], self.TimeStepSize[i+1]])
            
            # 1.2 define path
            # PathList is a list of simulation path
            # PathList = []

            # 1.3 run simulation
            DynamicArray = self._ensemble.fwd_propagate(ParamArray = self.ParamArray, 
                                                     TimeArray = SimulationTime, 
                                                     Simulator = self._simulator)
            DataArray = self.Dynamic2Data(DynamicArray)

            # 2 analysis
            [AnalysisParam, AnalysisDynamic] = self._analysis.create_analysis(self.ParamArray, 
                                                                              DynamicArray,
                                                                              DataArray, 
                                                                              self.ObserArray, 
                                                                              self.ObserCov)

            # 3 write result and assign self.Param
            self.ParamArray = AnalysisParam
            self.param_write()

        # 4 forecast
        ForecastArray = self._ensemble.fwd_propagate(ParamArray = self.ParamArray, 
                                                     TimeArray = self.FCTimeArray,
                                                     Simulator = self.Simulator)
        self.dynamic_write()

        raise NotImplementedError(inspect.stack()[0][3])
