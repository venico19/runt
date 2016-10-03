import numpy as np
from scipy.stats import norm
from subprocess import call

if __name__=='__main__':
    call(["ln", "-f", "-s", "/h1/jing/repos/runt/runt"])

# import ensemble class
from runt.ensemble_generator.ens_gen_fracture_parallel import Ens_Gen_Fracture_Parallel

# import analysis class
from runt.analysis.enkf_levelset import EnKF_levelset

# import assimilation class
from runt.assimilation.DA_levelset_fracture import DA_levelset_fracture

# import simulator class
from runt.simulator_interface.MFD_experiment import MFD_experiment


class levelset_MFD_DA(DA_levelset_fracture):
    # rewrite class function if necessary
    def Obser_initialization(self):
        # load observation
        self.WellSize = 9

        step = self.TimeStepSize
        self.StepSize = step
        self.pressure = np.zeros(step)
        # size: StepSize * WellSize
        self.WellType = np.zeros((step, self.WellSize), dtype = np.int)
        self.ProdSize = np.zeros(step, dtype = np.int)
        # size: StepSize * ObserSize
        self.Obser = np.zeros((step, self.WellSize))

        f = open('./input/obser.dat', 'r')
        for i in range(step):
            self.pressure[i] = f.readline()
            self.WellType[i,:] = map(int, f.readline().split(' '))
            for item in self.WellType[i, :]:
                if item == 1:
                    self.ProdSize[i] += 1
            for k in range(self.ProdSize[i]):
                temp = f.readline().split(' ')
                self.Obser[i,k] = float(temp[-1])

        # build self.ObserCov 
        # size: StepSize * ObserSize * ObserSize
        self.ObserCov = np.zeros((step, self.WellSize, self.WellSize))
        for s in range(step):
            for i in range(self.WellSize):
                self.ObserCov[s][i][i] = (self.Obser[s][i] * self.obser_noise)**2

        # build self.ObserArray
        # size: StepSize * ObserSize * EnSize
        np.random.seed(0)
        self.ObserArray = np.repeat(self.Obser[:, :, np.newaxis], self.EnSize, axis = 2) + np.random.randn(step, self.WellSize, self.EnSize)*self.obser_noise*np.mean(self.Obser)


    ### for reload mode only
    def DArun_reload(self):
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
            
            # 1.3 run simulation
            # self.Param2Frac() returns FracLocationAll 
#            [DynamicArray, self.ens_status] = \
#                self._ensemble.fwd_propagate(ParamArray = self.Param2Frac(), 
#                                             DynamicArray = [self.pressure[i],self.WellType[i, :]],
#                                             TimeArray = SimulationTime, 
#                                             Simulator = self._simulator, 
#                                             path = path, 
#                                             parallel_num = self.parallel_num,
#                                             ens_status = self.ens_status)
            # read output only without running simulator

            [DynamicArray, self.ens_status] = \
                self._ensemble.read_output_only(ParamArray = self.Param2Frac(), 
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


if __name__=='__main__':

    # 1) specify ensemble and analysis methods
    ensemble_method = Ens_Gen_Fracture_Parallel()
    analysis_method = EnKF_levelset()
    simulator = MFD_experiment()


    # 2) specify parameters
    NodeFileName = './input/node.dat'
    EnSize = 100
    RandMode = False
    [levelset_mean, levelset_std] = [-1.0, 0.5]
    [rho_mean, rho_std] = [0.2, 0.5]
    DATimeArray = np.array([300., ] *15)
    FCTimeArray = np.array([300.])

    # 3) generate (or load) initial parameters
    # 3.1) Node
    NodeArray = np.loadtxt(NodeFileName)
    NodeSize = NodeArray.shape[0]

    # 3.2) Parameters
    if not RandMode:
        np.random.seed(0)
#    levelset = np.random.randn(NodeSize, EnSize) * levelset_std + levelset_mean
#    rho = np.random.randn(NodeSize, EnSize) * rho_std + rho_mean
#    theta = np.random.randn(NodeSize, EnSize) 
    levelset = np.loadtxt('./input/levelset.dat')
    levelset = levelset[0:NodeSize*EnSize].reshape((EnSize, NodeSize)).transpose()
    levelset = levelset * levelset_std + levelset_mean

    rho = np.loadtxt('./input/rho_norm.dat')
    rho = rho[0:NodeSize*EnSize].reshape((EnSize, NodeSize)).transpose()
    rho = rho * rho_std + rho_mean

    theta = np.loadtxt('./input/theta_norm.dat')
    theta = theta[0:NodeSize*EnSize].reshape((EnSize, NodeSize)).transpose()


    ParamArray = np.vstack((levelset, rho, theta))
    

    # 4) generate DA method object
    DA_method = levelset_MFD_DA(EnSize = EnSize, 
                                NodeArray = NodeArray, 
                                ParamArray = ParamArray, 
                                DATimeArray = DATimeArray,
                                FCTimeArray = FCTimeArray,
                                EnsembleClass = ensemble_method,
                                AnalysisClass = analysis_method,
                                SimulatorClass = simulator)

    DA_method.set_length_threshold(0.2)
    DA_method.set_obser_noise(0.05)

    DA_method.set_parallel_num(10)

    DA_method.DArun()


