import inspect
import numpy as np

#import AnalysisClass
from .enkf import EnKF

class EnKF_levelset(EnKF):
    # input: {numpy arrays}
    #   ParamArray      = (ParamSize * EnSize)
    #   DynamicArray    = (DynamicSize * EnSize)
    #   DataArray       = (ObserSize * EnSize)
    #                     usually DataArray is part of DynamicArray
    #                     a function Dynamic2Data is defined in DA method
    #                     acctually Dynamic2Data is operator H
    #   ObserArray      = (ObserSize * EnSize) 
    #                     already disturbed by noises
    #   ObserCov        = (ObserSize * ObserSize)
    #
    # output:
    #   AnalysisDynamic = (DynamicSize * EnSize)
    #   AnalysisParam   = (ParamSize * EnSize)
    def __init__(self):
        self.Name = 'Ensemble Kalman Filter modified for levelset parameterization'

    def create_analysis(self, 
                        ParamArray, 
                        DynamicArray, 
                        DataArray, 
                        ObserArray, 
                        ObserCov,
                        levelset_num = 3):

        # collect data size 
        EnSize = ParamArray.shape[1]
        ParamSize = ParamArray.shape[0]
        DynamicSize = DynamicArray.shape[0]
        ObserSize = ObserArray.shape[0]

        levelset_num = levelset_num
        NodeSize = ParamSize / levelset_num

        #check data size
        assert EnSize == DynamicArray.shape[1]
        assert EnSize == DataArray.shape[1]
        assert EnSize == ObserArray.shape[1]
        assert ObserSize == ObserCov.shape[0]
        assert ObserSize == ObserCov.shape[1]

        ##################
        # start analysis #
        ##################

        # constuct state vector (y_f)
        # A = [parameter dynamic]^T
        # size: (ParamSize + DynamicSize) * EnSize
        A = np.vstack([ParamArray, DynamicArray])

        # modify A: if levelset < 0, set rho and theta as 0
        A_original = np.copy(A)



        # if level_num = 3:
        # levelset: A[0:NodeSize, :]
        # rho: A[NodeSize:2*NodeSize, ;]
        # theta: A[2*NodeSize:3*NodeSize, :]

        level_status = np.array(A[0:NodeSize, :]>0)
        level_pos_num = level_status.sum(1)
        assert level_pos_num.shape[0] == NodeSize

        A[NodeSize:2*NodeSize, :] = A[NodeSize:2*NodeSize, :] * level_status
        A[2*NodeSize:3*NodeSize, :] = A[2*NodeSize:3*NodeSize, :] * level_status

        # calculate mean and perturbation from mean
        # size: (ParamSize + DynamicSize) * EnSize
        Amean = (1./float(EnSize)) * np.tile(A.sum(1), (EnSize, 1)).transpose()

        # modify Amean
        for i in range(NodeSize):
            if level_pos_num[i] > 0:
                m = float(EnSize) / float(level_pos_num[i])
            elif level_pos_num[i] == 0:
                m = 0.0
            Amean[NodeSize+i, :] = Amean[NodeSize+i, :] * m
            Amean[2*NodeSize+i, :] = Amean[2*NodeSize+i, :] * m

        dA = A - Amean

        # observation perturbation
        # size: ObserSize * EnSize
        dD = ObserArray - DataArray

        # data perturbation from mean
        # size: ObserSize * EnSize
        DataMean = (1./float(EnSize)) * np.tile(DataArray.sum(1), (EnSize, 1)).transpose()
        S = DataArray - DataMean

        # calculate data coviance + observation coviance
        # size: ObserSize * ObserSize
        COVD = (1./float(EnSize-1)) * np.dot(S, S.transpose()) + ObserCov

        # compute COV^(-1) * dD
        # size: ObserSize * EnSize
        B = np.linalg.solve(COVD, dD)

        # calculate coviance matrix between A and Data
        # size: (ParamSize + DynamicSize) * ObserSize
        COV = (1./float(EnSize-1)) * np.dot(dA, S.transpose())

        #modify ACOV
#        for i in range(NodeSize):
#            if level_pos_num[i] > 0:
#                m = float(EnSize) / float(level_pos_num[i])
#            elif level_pos_num[i] == 0:
#                m = 0.0
#            COV[NodeSize+i, :] = COV[NodeSize+i, :] * m
#            COV[2*NodeSize+i, :] = COV[2*NodeSize+i, :] * m

        # compute analysis
        # size: (ParamSize + DynamicSize) * EnSize
        AnalysisArray = A_original + np.dot(COV, B)

        # seperate parameter
        # size: ParamSize * EnSize
        #       DynamicSize * EnSize
        AnalysisParam = AnalysisArray[0:ParamSize, :]
        AnalysisDynamic = AnalysisArray[ParamSize, :]

        return [AnalysisParam, AnalysisDynamic]
        
