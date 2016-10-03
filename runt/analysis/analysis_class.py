import inspect
import numpy as np

class AnalysisClass(object):
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
    #   AnalysisArray   = (ParamSize+DynamicSize) * EnSize
    #   AnalysisParam   = (ParamSize * EnSize)

    def __init__(self):
        pass

    def create_analysis(self, 
                        ParamArray, DynamicArray, DataArray, 
                        ObserArray, ObserCov):
        
        raise NotImplementedError(inspect.stack()[0][3])

