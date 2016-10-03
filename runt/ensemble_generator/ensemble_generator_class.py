import inspect
import numpy as np


class EnsembleGeneratorClass(object):
    '''
    1,  initialize ensemble:
        create state vector ensembles, 
        including static parameters, dynamic data, observation

    2,  call simulator:
        input: static parameters, dynamic data, 
               time array, simulator_class, etc... 
        output: dynamic data
    '''

    def __init__(self):
        pass

    def fwd_propagate(self, 
                      Param = None, Dynamic = None, TimeArray = None, 
                      Simulator = None, path = None):

        #call simulator
        #
        # input:
        #   Param = (Param size * EnSize) numpy array
        #   Dynamic = (Dynamic size * EnSize) numpy array
        #   TimeArray = (Time step size * Ensize) numpy array
        #   Simulator = object of SimulatorInterface(object)
        #   path = path of simulation
        #
        # output:
        #   DataArray

        self.Param = Param
        self.Dynamic = Dynamic
        self.TimeArray = TimeArray
        self.EnSize = Param.shape[1]
        self._simulator = Simulator
        self.path = path

        #write input file, run simulator, read output file
        Simulator.write_input_file(self.Param, self.Dynamic, self.TimeArray, self.path)
        Simulator.run_simulator(self.path)
        DataArray = Simulator.read_output_file(self.path)

        raise NotImplementedError(inspect.stack()[0][3])



