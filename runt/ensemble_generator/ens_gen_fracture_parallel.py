import inspect
from subprocess import call
import time
import numpy as np

#import the EnsembleGeneratorClass
from .ensemble_generator_parallel import EnsembleGeneratorParallel

class Ens_Gen_Fracture_Parallel(EnsembleGeneratorParallel):
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
                      ParamArray, 
                      DynamicArray, 
                      TimeArray, 
                      Simulator, 
                      path,
                      ens_status,
                      parallel_num = 1):

        #call simulator
        #
        # input:
        #   Param        = a list of fracture location, length = EnSize
        #   Dynamic      = (Dynamic size) numpy array
        #   TimeArray    = (Time step size * 1) numpy array
        #   Simulator    = object of SimulatorInterface(object)
        #   path         = a list of simulation path for each realization
        #               length(path) = self.EnSize = Param.shape[1]
        #   parallel_num = number of parallel run, default 1
        #
        # output:
        #   DataArray    = (Dynamic size * Ensize) numpy array

        self.Param = ParamArray    
        self.EnSize = len(ParamArray)
        self.Dynamic = DynamicArray
        self.TimeArray = TimeArray
        self._simulator = Simulator
        self.parallel_num = parallel_num
        self.ens_status = ens_status


        # create paths for each realization
        # path list, length: EnSize
        self.path = []
        for i in range(self.EnSize):
            p = path + '/ens' + str(i+1).zfill(3)
            call (["mkdir", p])
            self.path.append(p)

        # set parallel num
        if self.parallel_num > self.EnSize:
            print "parallel_num", self.parallel_num, "larger than EnSize, set parallel_num as EnSize", self.Ensize
            self.parallel_num = self.EnSize

        ##########################
        # run mesher in parallel #
        ##########################

        # start_time for each realization
        start_time = [1.e11] * self.EnSize
        
        now_index = 0
        finish_index = [False] * self.EnSize

        # write input file, run simulator, read output file in parallel
        # run first bunch of simulation
        # para: number of running simulations
        # now_index: number of processed ens

        # run first bunch of simulation
        para = 0
        while para < self.parallel_num:
            if self.ens_status[now_index]:
                print "mesher forward propagate realization:", now_index + 1, "/", self.EnSize
                print self.path[now_index]

                # write input file
                Simulator.write_mesher_file(self.Param[now_index], 
                                            self.path[now_index])
                # run mesher
                Simulator.run_mesher(self.path[now_index])
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
                    if Simulator.check_finish_mesher(self.path[i]) or Simulator.check_break_mesher(self.path[i], start_time[i]):
                        # break :(
                        if Simulator.check_break_mesher(self.path[i], start_time[i]):
                            self.ens_status[i] = False
                            with open(path+'/broken', 'a') as file:
                                file.write('mesher breaks, ens: ' + str(i+1) + '\n')
                        # finish! or break
                        finish_index[i] = True
                        
                        if now_index<self.EnSize:
                            #run a new one
                            while not self.ens_status[now_index] and now_index<self.EnSize-1:
                                now_index += 1

                            if self.ens_status[now_index]:
                                print "mesher forward propagate realization:", now_index + 1, "/", self.EnSize
                                print self.path[now_index]
                                # write input file
                                Simulator.write_mesher_file(self.Param[now_index], 
                                                           self.path[now_index])
                                # run mesher
                                Simulator.run_mesher(self.path[now_index])
                                start_time[now_index] = time.time()
                                
                                now_index += 1

            time.sleep(1)

        ####################################



        ##########################
        #  run MFD in parallel   #
        ##########################

        # start_time for each realization
        start_time = [1.e11] * self.EnSize
        
        now_index = 0
        finish_index = [False] * self.EnSize

        # write input file, run simulator, read output file in parallel
        # run first bunch of simulation
        # para: number of running simulations
        # now_index: number of processed ens

        # run first bunch of simulation
        para = 0
        while para < self.parallel_num:
            if self.ens_status[now_index]:
                print "mfd forward propagate realization:", now_index + 1, "/", self.EnSize
                print self.path[now_index]
                # write input file
                Simulator.write_mfd_file(self.Dynamic, 
                                         self.TimeArray,
                                         self.path[now_index])
                # run mfd
                Simulator.run_mfd(self.path[now_index])
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
                    if Simulator.check_finish_mfd(self.path[i]) or Simulator.check_break_mfd(self.path[i], start_time[i]):
                        # break :(
                        if Simulator.check_break_mfd(self.path[i], start_time[i]):
                            self.ens_status[i] = False
                            with open(path+'/broken', 'a') as file:
                                file.write('mfd breaks, ens: ' + str(i+1) + '\n')
                        # finish! or break
                        finish_index[i] = True
                        
                        if now_index<self.EnSize:
                            #run a new one
                            while not self.ens_status[now_index] and now_index<self.EnSize-1:
                                now_index += 1

                            if self.ens_status[now_index]:
                                print "mfd forward propagate realization:", now_index + 1, "/", self.EnSize
                                print self.path[now_index]
                                # write input file
                                Simulator.write_mfd_file(self.Dynamic, 
                                                         self.TimeArray,
                                                         self.path[now_index])
                                # run mesher
                                Simulator.run_mfd(self.path[now_index])
                                start_time[now_index] = time.time()
                                
                                now_index += 1

            time.sleep(1)

        ####################################

        # read output

        [SimulationArray, self.ens_status] = Simulator.read_output_file(self.path, self.ens_status, self.Dynamic[1])

        return [SimulationArray, self.ens_status]



    def read_output_only(self, 
                         ParamArray, 
                         DynamicArray, 
                         TimeArray, 
                         Simulator, 
                         path,
                         ens_status,
                         parallel_num = 1):

        #call simulator
        #
        # input:
        #   Param        = a list of fracture location, length = EnSize
        #   Dynamic      = (Dynamic size) numpy array
        #   TimeArray    = (Time step size * 1) numpy array
        #   Simulator    = object of SimulatorInterface(object)
        #   path         = a list of simulation path for each realization
        #               length(path) = self.EnSize = Param.shape[1]
        #   parallel_num = number of parallel run, default 1
        #
        # output:
        #   DataArray    = (Dynamic size * Ensize) numpy array

        self.Param = ParamArray    
        self.EnSize = len(ParamArray)
        self.Dynamic = DynamicArray
        self.TimeArray = TimeArray
        self._simulator = Simulator
        self.parallel_num = parallel_num
        self.ens_status = ens_status


        # create paths for each realization
        # path list, length: EnSize
        self.path = []
        for i in range(self.EnSize):
            p = path + '/ens' + str(i+1).zfill(3)
            call (["mkdir", p])
            self.path.append(p)

        # read output

        [SimulationArray, self.ens_status] = Simulator.read_output_file(self.path, self.ens_status, self.Dynamic[1])

        return [SimulationArray, self.ens_status]

