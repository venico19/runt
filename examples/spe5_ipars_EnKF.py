import numpy as np
import math
from scipy.stats import norm
from subprocess import call
import os.path

if __name__=='__main__':
    call(["ln", "-f", "-s", "/h1/jing/repos/runt/runt"])

# import ensemble class
from runt.ensemble_generator.ensemble_generator_parallel import EnsembleGeneratorParallel

# import analysis class
from runt.analysis.enkf import EnKF

# import simulator class
from runt.simulator_interface.ipars import IPARS_interface

# import assimilation class
from runt.assimilation.DA_init2current import DA_init2current

class IPARS_spe5_const_interface(IPARS_interface):
    # rewrite class function if necessary
    def write_input_param(self, param, path):
        #  write perm
        permfile = path + '/perm.dat'
        f = open(permfile, 'w')
        print>>f, 'XPERM1(1,,) = 50.' 
        print>>f, 'XPERM1(2,,) = 50.'
        print>>f, 'XPERM1(3,,) = 25.'
        print>>f, 'YPERM1(1,,) = ', math.exp(param[0])
        print>>f, 'YPERM1(2,,) = ', math.exp(param[1])
        print>>f, 'YPERM1(3,,) = ', math.exp(param[2])
        print>>f, 'ZPERM1(1,,) = ', math.exp(param[0])
        print>>f, 'ZPERM1(2,,) = ', math.exp(param[1])
        print>>f, 'ZPERM1(3,,) = ', math.exp(param[2])
        f.close()

        #  copy data file
        call(['cp', './spe5/spe5.dat', path])
        call(['cp', './spe5/IPARS.IN', path])
        call(['ln', '-f', '-s', '/org/centers/csm/jping/IPARSv3.1/workc/ipars', path])

    def write_input_time(self, t, path):
        #  write simulation time
        timefile = path + '/timeend.dat'
        f = open(timefile, 'w')
        print>>f, 'TIMEEND = ', t
        f.close()

    def write_input_ens(self, param, t, path, ens_status):
        EnSize = len(ens_status)
        path_list = []
        for i in range(EnSize):
            p = path + '/ens' + str(i+1).zfill(3)
            call (["mkdir", p])
            path_list.append(p)

            if ens_status[i]:
                self.write_input_param(param[:,i], path_list[i])
                self.write_input_time(t, path_list[i])

    def write_input_ens_fc(self, param, t, path, ens_status):
        EnSize = len(ens_status)
        path_list = []
        for i in range(EnSize):
            p = path + '/ens' + str(i+1).zfill(3)
            path_list.append(p)

            if ens_status[i]:
                self.write_input_time(t, path_list[i])



class spe5_IPARS_DA(DA_init2current):
    # rewrite class function if necessary
    def Obser_initialization(self):

        self.ObserSize = 3

        # create well_type_time_dic
        self.well_type_time_dic = {}
        self.well_type_time_dic[(2, 2)] = [self.FCTimeArray]
        self.well_type_time_dic[(2, 3)] = [self.FCTimeArray]
        self.well_type_time_dic[(2, 4)] = [self.FCTimeArray]
    
        # self.Obser size: StepSize * Obsersize
        self.Obser = self._simulator.read_wellout_extract('./spe5/spe5.out', self.well_type_time_dic, tolerance = 1.0)

        np.savetxt('./output/observation.dat', self.Obser)

        # self.ObserCov size: StepSize * Obsersize * Obsersize
        self.ObserCov = np.zeros((self.TimeStepSize, self.ObserSize, self.ObserSize))
        for s in range(self.TimeStepSize):
            for i in range(self.ObserSize):
                self.ObserCov[s][i][i] = (self.Obser[s][i] * self.obser_noise)**2

        # self.ObserArray size: Stepsize * ObserSize * EnSize
        np.random.seed(0)
        
        self.ObserArray = np.repeat(self.Obser[:, :, np.newaxis], self.EnSize, axis = 2) + \
                np.repeat(self.Obser[:, :, np.newaxis], self.EnSize, axis = 2) * \
                np.random.randn(self.TimeStepSize, self.ObserSize, self.EnSize) * self.obser_noise



if __name__=='__main__':

    # 1) specify ensemble and analysis methods
    ensemble_method = EnsembleGeneratorParallel()
    analysis_method = EnKF()
    simulator = IPARS_spe5_const_interface()

    # 2) specify parameters
    EnSize = 40
    # if not RandMode, set random.seed as 0 
    RandMode = False
    ParamSize = 3
    Param_mean = [math.log(500), math.log(50), math.log(200)]
    Param_std = [1.5,1.,1.5]
    DATimeArray = np.array([365.0, 821.25, 1095.0, 1460.0, 1825.0, 2190.0, 2555.0, 2920.0, 3285.0, 3650.0])
    FCTimeArray = np.array([365.0, 821.25, 1095.0, 1460.0, 1825.0, 2190.0, 2555.0, 2920.0, 3285.0, 3650.0])

    # 3) generate (or load) initial parameters

    # 3.1) Parameters
    if not RandMode:
        np.random.seed(0)
    
    ParamArray = np.empty([ParamSize, EnSize])
    for i in range(ParamSize):
        ParamArray[i,:] = np.random.randn(EnSize) * Param_std[i] + Param_mean[i]

    # 4) generate DA method object
    DA_method = spe5_IPARS_DA(EnSize = EnSize, 
                              ParamArray = ParamArray, 
                              DATimeArray = DATimeArray,
                              FCTimeArray = FCTimeArray,
                              EnsembleClass = ensemble_method,
                              AnalysisClass = analysis_method,
                              SimulatorClass = simulator)

    DA_method.set_obser_noise(0.05)

    DA_method.set_parallel_num(5)

    DA_method.DArun()

    # initialize obser if run forecast only
    #DA_method.Obser_initialization()

    DA_method.forecast(FCTime = FCTimeArray)


