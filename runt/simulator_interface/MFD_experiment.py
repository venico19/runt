import inspect
from subprocess import call
import os
import time
import numpy as np

#import SimulatorClass
from .simulator_class import SimulatorClass

class MFD_experiment(SimulatorClass):
    def __init__(self):
        self.x_scale = 0.32385
        self.y_scale = 0.4826

    def write_input_file(self, param, dynamic, t, path):
        self.write_mesher_file(param, path)
        self.write_mfd_file(dynamic, t, path)

    def write_mesher_file(self, param, path):
        # 1) write frac location
        fraclocfile = path + '/add_frac_cmd'
        f = open(fraclocfile, 'w')
        for item in param:
            print>>f, 'add_vertex', item[0]*self.x_scale, item[1]*self.y_scale
            print>>f, 'add_vertex', item[2]*self.x_scale, item[3]*self.y_scale
            print>>f, 'add_edge -1 -2'
            print>>f, 'set_fracture -1'
        print>>f, 'intersect_fractures'
        print>>f, 'remove_leaves'
        print>>f, 'remove_fractures_from_holes'
        print>>f, 'merge_fracture_vertices  9.99999975E-05'
        print>>f, 'set_fracture_width .0001'
        f.close()

        # 2) copy mesher file
        call(['cp', './input/mesher_cmd', path])

    def write_mfd_file(self, dynamic, t, path):
        # 3) write input_example.py
        inputfile = path + '/input_example.py'
        f = open(inputfile, 'w')
        print>>f, "output_file = open('pressure', 'w')"
        print>>f, 'K = 4.e-12'
        print>>f, 'K_f = 1.e-6'
        print>>f, 'mu = 8.9e-4'
        print>>f, '#well_type: 0: close, 1: inj, 2:prod'
        print>>f, 'well_type = ', list(dynamic[1])
        print>>f, 'pressures = [', str(dynamic[0]), ']'
        print>>f, 'total_time =', t
        f.close()

        # 4) copy run_example.py
        call(['cp', './input/run_example.py', path])

    def run_mesher(self, path):
        os.system('\ln -s /h1/jing/repos/mesher/build/lib/mesher/mesher.py ' + path)
        os.system('\\rm -f ' + path + '/done')
        os.system('\\rm -f ' + path + '/error')
        os.system('cd ' + path + '; python mesher.py mesher_cmd > /dev/null && (echo "done">mesher_done) || (echo "error">mesher_error) &') 

    def run_mfd(self, path):
        os.system('\\rm -f ' + path + '/flag')
        # run mfd
        os.system('cd ' + path + '; python run_example.py > /dev/null && (echo "done">mfd_done) || (echo "error">mfd_error) &') 

    def check_finish_mfd(self, path):
        return os.path.isfile(path + '/flag')

    def check_finish_mesher(self, path):
        return os.path.isfile(path + '/mesh')

    def check_break_mfd(self, path, start_time):
        file = os.path.isfile(path + '/mfd_error')
        t = (time.time() - start_time) > 300.
        return (file or t)

    def check_break_mesher(self, path, start_time):
        file = os.path.isfile(path + '/mesher_error')
        t = (time.time() - start_time) > 600.
        return (file or t)

    def read_output_file(self, path, ens_status, well_type):
        # path is a list for all ensembles 
        # len(path) = EnSize
        # len(ens_status) = EnSize
        # well_type is a np array, size: WellSize * 1

        EnSize = len(path)

        WellSize = 9
        SimulationArray = np.zeros((WellSize, EnSize))
        for i in range(EnSize):
            if ens_status[i]:
                try:
                    index = 0
                    f = open(path[i] + '/pressure', 'r')
                    f.readline()
                    for w in range(WellSize):
                        temp = f.readline().split(' ')
                        if well_type[w] == 1:
                            SimulationArray[index, i] = float(temp[-1])
                            index += 1
                    f.close()
                except:
                    ens_status[i]=False

        return [SimulationArray, ens_status]

