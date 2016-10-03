import inspect
from subprocess import call
import numpy as np
import time

#import the EnsembleGeneratorClass
from .ensemble_generator_class import EnsembleGeneratorClass

class EnsembleGeneratorParallel(EnsembleGeneratorClass):
    '''
    1,  initialize ensemble:
        create state vector ensembles, 
        including static parameters, dynamic data, observation

    2,  call simulator in parallel:
        input: static parameters, dynamic data, 
               time array, simulator_class, etc... 
        output: dynamic data
    '''

    def __init__(self):
        pass

    def fwd_propagate(self, 
                      ParamArray = None, 
                      DynamicArray = None, 
                      TimeArray = None, 
                      Simulator = None, 
                      path = None,
                      ens_status = None,
                      parallel_num = 1):

        #call simulator
        #
        # input:
        #   Param        = (Param size * Ensize) numpy array
        #   Dynamic      = (Dynamic size * Ensize) numpy array
        #   TimeArray    = (Time step size * 1) numpy array
        #   Simulator    = object of SimulatorInterface(object)
        #   path         = a list of simulation path for each realization
        #               length(path) = self.EnSize = Param.shape[1]
        #   parallel_num = number of parallel run, default 1
        #
        # output:
        #   DataArray    = (Dynamic size * Ensize) numpy array

        self.Param = ParamArray
        self.EnSize = len(ens_status)
        self.Dynamic = DynamicArray
        self.TimeArray = TimeArray
        self._simulator = Simulator
        self.parallel_num = parallel_num
        self.ens_status = ens_status

        # create paths for each realization
        # self.path is a list of pathes for all ensembles, length: EnSize
        self.path = []
        for i in range(self.EnSize):
            p = path + '/ens' + str(i+1).zfill(3)
            self.path.append(p)
            
        # set parallel num
        if self.parallel_num > self.EnSize:
            print "parallel_num", self.parallel_num, "larger than EnSize, set parallel_num as EnSize", self.Ensize
            self.parallel_num = self.EnSize
        if self.parallel_num < 1:
            self.parallel_num = 1

        #############################
        # run simulator in parallel #
        #############################

        # if finished, marked finish_index as True
        finish_index = [False] * self.EnSize

        # run first bunch of simulation
        # para: number of running simulations
        # now_index: number of processed ens
        # now_index could be larger than para, as ens_status could be False

        now_index = 0
        para = 0

        start_time = [1.e11] * self.EnSize

        while para < self.parallel_num:
            if self.ens_status[now_index]:
                print "forward propagate, realization: ", \
                      now_index + 1, "/", self.EnSize
                print self.path[now_index]

                # run
                Simulator.run(self.path[now_index])
                start_time[now_index] = time.time()

                para += 1

            now_index += 1

        # check if anyone finish
        active_ens_number = sum(self.ens_status)
        while sum(finish_index) < active_ens_number:
            # loop in all realizations check if anyone finish
            for i in range(now_index):
                # if (not marked finish) and (this ensemble status is active)
                if (not finish_index[i]) and self.ens_status[i]:
                    # check if finish or break
                    if Simulator.check_finish(self.path[i]) or Simulator.check_break(self.path[i], start_time[i]):
                        # break :(
                        if Simulator.check_break(self.path[i], start_time[i]):
                            self.ens_status[i] = False
                            with open(path+'/broken', 'a') as file:
                                file.write('simulation breaks, ens: ' + str(i+1) + '\n')
                        # finish! or break
                        finish_index[i] = True
                        
                        if now_index<self.EnSize:
                            #run a new one
                            while not self.ens_status[now_index] and now_index<self.EnSize-1:
                                now_index += 1

                            if self.ens_status[now_index]:
                                print "forward propagate realization:", now_index + 1, "/", self.EnSize
                                print self.path[now_index]
                                
                                # run 
                                Simulator.run(self.path[now_index])
                                start_time[now_index] = time.time()
                                
                                now_index += 1

            time.sleep(1)

        return self.ens_status
