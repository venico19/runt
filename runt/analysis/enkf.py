import inspect
import numpy as np

#import AnalysisClass
from .analysis_class import AnalysisClass

class EnKF(AnalysisClass):
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
        self.Name = 'Ensemble Kalman Filter'

    def create_analysis(self, 
                        ParamArray, DynamicArray, DataArray, 
                        ObserArray, ObserCov):
        # collect data size 
        EnSize = ParamArray.shape[1]
        ParamSize = ParamArray.shape[0]
        DynamicSize = DynamicArray.shape[0]
        ObserSize = ObserArray.shape[0]

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

        # calculate mean and perturbation from mean
        # size: (ParamSize + DynamicSize) * EnSize
        Amean = (1./float(EnSize)) * np.tile(A.sum(1), (EnSize, 1)).transpose()
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

        # compute analysis
        # size: (ParamSize + DynamicSize) * EnSize
        AnalysisArray = A + np.dot(COV, B)

        # seperate parameter
        # size: ParamSize * EnSize
        #       DynamicSize * EnSize
        AnalysisParam = AnalysisArray[0:ParamSize, :]
        AnalysisDynamic = AnalysisArray[ParamSize, :]

        return [AnalysisParam, AnalysisDynamic]
        
