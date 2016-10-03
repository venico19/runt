import inspect
from subprocess import call
import time

import numpy as np
from scipy.stats import norm

#import DataAssimilationClass
from .data_assimilation_class import DataAssimilationClass

class DA_levelset_fracture(DataAssimilationClass):
    def __init__(self, 
                 EnSize,
                 NodeArray, 
                 ParamArray, 
                 DATimeArray,
                 FCTimeArray,
                 EnsembleClass,
                 AnalysisClass,
                 SimulatorClass):

        # EnSize: number of ensembles
        self.EnSize = EnSize

        # load node array
        # size: NodeSize * EnSize
        self.NodeArray = NodeArray
        self.NodeSize = self.NodeArray.shape[0]

        # load param array includes
        #      levelset 
        #      rho
        #      theta
        # size: (3 * NodeSize) * EnSize
        self.ParamArray = ParamArray

        # load time array
        self.DATimeArray = DATimeArray
        self.FCTimeArray = FCTimeArray
        self.TimeStepSize = len(DATimeArray)

        self._ensemble = EnsembleClass
        self._analysis = AnalysisClass
        self._simulator = SimulatorClass


        # set obser noise
        self.set_obser_noise()

        # set length_threshold
        self.set_length_threshold()

        # intialize ens_status list
        self.ens_status = [True] * self.EnSize


    def Obser_initialization(self):
        # initialize observation
        # read from file or run simulation
        raise NotImplementedError(inspect.stack()[0][3])

    def ObserCov_initialization(self):
        #define observation error coviance 
        raise NotImplementedError(inspect.stack()[0][3])

    def ParamTransform(self):
        # transform normal distributed parameters to other distributions
        # for example, transform theta from normal to uniform 
        self.angle = norm.cdf(self.theta) * np.pi * 2.0
    
    def set_length_threshold(self, threshold = 0.0):
        # default threshold 0.0
        self.length_threshold = threshold

    def set_obser_noise(self, noise = 0.01):
        # default noise is 1%
        self.obser_noise = noise

    def Param2Frac(self):
        # transform parameters for constucting fractures    
        self.levelset = self.ParamArray[0:self.NodeSize, :]
        self.rho = self.ParamArray[self.NodeSize:2*self.NodeSize, :]
        self.theta = self.ParamArray[2*self.NodeSize:3*self.NodeSize, :]

        self.ParamTransform()

        # list for all ensembles, size = EnSize, each member is a list
        FracLocationAll = []

        for i in range(self.EnSize):
            # list for each ensemble
            # each member is a tuple (x0, y0, x1, y1)
            # this list could be empty if these is no fracture
            FracLocation = []
            
            for j in range(self.NodeSize):
                # center location
                [x0, y0] = self.NodeArray[j, :]

                # phi
                if self.levelset[j][i] > 0:
                    # length and angle
                    length = self.rho[j][i]
                    angle = self.angle[j][i]

                    if length > self.length_threshold:
                        x1 = x0 + length * np.cos(angle)
                        y1 = y0 + length * np.sin(angle)

                        # trim if necessery
                        # e.g. trim at 4 boundaries x = 0, x = 1, y = 0, y = 1
                        [x1, y1] = self.trim_fracture(x0, y0, x1, y1, angle)

                        # calculate trimed length
                        length = np.sqrt((x1-x0)**2 + (y1-y0)**2)

                        if length > self.length_threshold:     
                            # append to FracLocation list
                            FracLocation.append((x0, y0, x1, y1))

            FracLocationAll.append(FracLocation)

        return FracLocationAll

    def trim_fracture(self, x0, y0, x1, y1, angle):

        # trim at 4 boundaries x = 0, x = 1, y = 0, y = 1
        if abs(x0-x1) < 0.0001:
            xedge_small = x0
            xedge_large = x0
            yedge_small = -1.0
            yedge_large = 2.0
        elif abs(y0-y1) < 0.0001:
            xedge_small = -1.0
            xedge_large = 2.0
            yedge_small = y0
            yedge_large = y0
        else:
            xedge_large = x0 + (1.0-y0)/np.tan(angle)
            yedge_large = y0 + (1.0-x0)*np.tan(angle)
            xedge_small = x0 - (y0-0.0)/np.tan(angle)
            yedge_small = y0 - (x0-0.0)*np.tan(angle)

        # generator an array of 6 points coordinate, then sort them
        dtype = [('name', 'S10'), ('x_co', float), ('y_co', float)]
        value = [('x0y0', x0, y0), ('x1y1', x1, y1), 
                 ('x0yS', 0.0, yedge_small), ('xSy0', xedge_small, 0.0), 
                 ('x1yL', 1.0, yedge_large), ('xLy1', xedge_large, 1.0)]
        points = np.array(value, dtype = dtype)
        points = np.sort(points, order = ['x_co', 'y_co'])

        # search x0 y0 and x1 y1
        for i in range(points.shape[0]):
            if points[i][0]=='x0y0':
                x0y0_index = i
            elif points[i][0]=='x1y1':
                x1y1_index = i
        
        if x0y0_index < x1y1_index:
            # search from right to left
            increase = -1
        else:
            # search from left to right
            increase = 1

        #search starts here
        s = x1y1_index

        while (points[s][1]<0.0 or points[s][1]>1.0 or points[s][2]<0.0 or points[s][2]>1.0):
            s += increase

        return [points[s][1], points[s][2]]

    def param_write(self, step):
        # write parameters
        filename = ['levelset', 'rho', 'theta']
        for i in range(3):
            path = './output/' + filename[i] + str(step).zfill(2) + '.dat'
            np.savetxt(path, self.ParamArray[i*self.NodeSize : (i+1)*self.NodeSize, :], fmt = '%9.5f')
        
        # write ens_status
        ens_filename = './output/ens_status' + str(step).zfill(2) + '.dat'
        ens = np.array(self.ens_status).astype(int)
        np.savetxt(ens_filename, np.append(ens, sum(ens)), delimiter='\n', fmt = '%d')

        # write FracNumber
        filename = './output/FracNumber' + str(step+1).zfill(2) + '.dat'
        f = open(filename, 'w')
        for item in self.Param2Frac():
            print>>f, len(item)
        

    def set_parallel_num(self, parallel=1):
        self.parallel_num = parallel

    def DArun(self):
        # Data Assimilation Process:

        # 0) clear old output folder, mkdir new one
        path = "output" + time.strftime("%y%m%d%H%M")
        call(["mv", "output", path])
        call(["mkdir", "output"])

        # 1) initialization
        # 1.1) initialize param
        # done in __init__

        # 1.2) initialize Obser and ObserCov
        self.Obser_initialization()

        # 1.3) write initial param
        self.param_write(0)

        for i in range(self.TimeStepSize):
            print "updating step: ", str(i+1)
            # 2) ensemble generation

            # 2.2) define SimulationTime
            SimulationTime = self.DATimeArray[i]
            
            # 2.3) define path
            path = './output/step' + str(i+1).zfill(2)
            call(["mkdir", path])

            # 1.3 run simulation
            # self.Param2Frac() returns FracLocationAll 
            [DynamicArray, self.ens_status] = \
                self._ensemble.fwd_propagate(ParamArray = self.Param2Frac(), 
                                             DynamicArray = [self.pressure[i],self.WellType[i, :]],
                                             TimeArray = SimulationTime, 
                                             Simulator = self._simulator, 
                                             path = path, 
                                             parallel_num = self.parallel_num,
                                             ens_status = self.ens_status)

            
            DataArray = DynamicArray


            # 2 analysis

            [AnalysisParam, AnalysisDynamic] = \
                self._analysis.create_analysis(self.extract_ens(self.ParamArray), 
                                               self.extract_ens(DynamicArray),
                                               self.extract_well(self.extract_ens(DataArray), i, 1), 
                                               self.extract_well(self.extract_ens(self.ObserArray[i,:,:]), i, 1), 
                                               self.extract_well(self.ObserCov[i,:,:],i, 2))

            # 3 write result and assign self.Param
            self.ParamArray = self.extract_ens_inv(AnalysisParam)
            self.param_write(i+1)

        # 4 forecast
        # ForecastArray = self._ensemble.fwd_propagate(ParamArray = self.ParamArray, 
        #                                             TimeArray = self.FCTimeArray,
        #                                             Simulator = self.Simulator)

    def extract_well(self, Array, i, tag):
        # reduce non-production wells 
        # if tag = 1, only apply on first dimension
        #             e.g. DataArray, ObserArray
        # if tag = 2, apply on two dimensions
        #             e.g. ObserCov
        if tag==1:
            return Array[0:self.ProdSize[i], :]
        elif tag==2:
            return Array[0:self.ProdSize[i], 0:self.ProdSize[i]]

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



