import inspect

class SimulatorClass(object):
    def __init__(self):
        pass

    def write_input_file(self, param, dynamic, time, path):
        raise NotImplementedError(inspect.stack()[0][3]) 

    def run_simulator(self, path):
        raise NotImplementedError(inspect.stack()[0][3]) 

    def check_finish(self, path):
        raise NotImplementedError(inspect.stack()[0][3]) 

    def read_output_file(self, path):
        raise NotImplementedError(inspect.stack()[0][3]) 

