import datetime
import time
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plot
import math as m
from scipy.stats import poisson
from sklearn.preprocessing import normalize
from get_data import Data
from Trial import Trial


class Neuron:

    def __init__(self):
        T1O8 = Trial("T1O8", Data.T1O8())
        T2O1 = Trial("T2O1", Data.T2O1())
        T3O6 = Trial("T3O6", Data.T3O6())
        T4O3 = Trial("T4O3", Data.T4O3())
        T5O5 = Trial("T5O5", Data.T5O5())
        T6O2 = Trial("T6O2", Data.T6O2())
        T7O4 = Trial("T7O4", Data.T7O4())
        T8O9 = Trial("T8O9", Data.T8O9())
        T9O7 = Trial("T9O7", Data.T9O7())
        T10O10 = Trial("T10O10", Data.T10O10())

        self._data = [T1O8.data(), T2O1.data(), T3O6.data(), T4O3.data(), T5O5.data(), T6O2.data(), T7O4.data(), T8O9.data(), T9O7.data(), T10O10.data()]
        self._trial_list = [T1O8, T2O1, T3O6, T4O3, T5O5, T6O2, T7O4, T8O9, T9O7, T10O10]

        self._nmb_odors = 10
        self._nmb_data_per_trial = 1000
        self._nmb_trials = 10

    def get_trial_list(self):
        return self._trial_list

    # def get_data(self):
    #     return self._data

    # def trial_matrix(self):
    #     array = [[] for i in range(self._nmb_trials)]
    #     c = 0
    #     for trial in self._trial_list:
    #         for i in range(self._nmb_trials):
    #             if trial.odor() == i+1:
    #                 array[c].append(1)
    #             else:
    #                 array[c].append(0)
    #         c += 1
    #     return array

    # def concatenate_horizontal(self, m1, m2):
    #     return np.concatenate((m1, m2), axis=1)


    def temp_kernel_matrix(self):
        array = []
        to_append = 0
        for trial in self._trial_list:
            temp = []
            if trial.get_activation() == True:
                to_append = 1
            if trial.get_activation() == False:
                to_append = 0
            for i in range(self._nmb_data_per_trial):
                temp.append(to_append)
            array.append(temp)
        return array


    # def bias_matrix(self):
    #     array = [0]*self._nmb_odors
    #     for trial in self._trial_list:
    #         array[trial.odor()-1] = trial.sorption()
    #     return array


    # def coeff_matrix(self, bias_matrix, temp_kernel_matrix):
    #     array = []
    #     for coeff in bias_matrix:
    #         array.append(coeff)
    #     for coeff in temp_kernel_matrix:
    #         array.append(coeff)  
    #     return array


    def L_stage(self, to_lf, coeff_matrix):
        # print(to_lf[0])
        # print(coeff_matrix)
        return np.matmul(to_lf,coeff_matrix)


    def norm_L_stage(self, lf_matrix):
        norm = []
        for nmb in lf_matrix:
            # print(nmb)
            norm.append(nmb/1000000)
        # print(norm)
        return norm



    def N_stage(self, norm_L_stage):
        lam = []
        for nmb in norm_L_stage:
            out = 0
            if nmb != 0:
                out = float(1/(1+m.exp((-nmb))))
            # out = float(np.log(1+m.exp(nmb)))
            # out = float(nmb**2)
            lam.append(out)
        return lam


    def gen_spikes(self, lam):
        spike = []
        for l in lam:
            spike.append(np.random.poisson(l, size=len(self._data[0])))
        return spike


    def calculate_distance(self, spike_array):
        out = []
        for train in spike_array:
            spike_time = []
            timestamp = float(0.0)
            for i in range(len(train)):
                if train[i] > 0:
                    for i in range(1, train[i]+1):
                        spike_time.append(timestamp)
                timestamp += float(0.001)
            out.append(spike_time)
        return out


    def calc_nmb_spikes(self, spike_array):
        # out = []
        c = 0
        for train in spike_array:
            sum = 0
            for nmb in train:
                if nmb > 0:
                    sum += 1
            self._trial_list[c].set_total_spikes(sum)
            c += 1
            # out.append(sum)
        # return out


    def to_print(self):
        for trial in self._trial_list:
            print("trial: " + str(trial.get_name()) + "; odor: " + str(trial.odor()) + "; sorption: " + trial.sorption_level() + "; spikes: " + str(trial.get_total_spikes()))

    def mult_matrices(self, m1, m2):
        arr = []
        for i in range(len(m1)):
            c = 0
            sum = 0
            while c < len(m1[i]):
                val = float(m1[i][c] * m2[i][c])
                sum += val
                c += 1
            arr.append(val)
        return arr

    def run_trials(self):
        # trial_matrix = self.trial_matrix()
        # flow_matrix = self._data
        # to_lf = self.concatenate_horizontal(trial_matrix, flow_matrix)
        temp_kernel_matrix = self.temp_kernel_matrix()
        to_lf = self._data
        # bias_matrix = self.bias_matrix()
        # coeff_matrix = self.coeff_matrix(bias_matrix, temp_kernel_matrix)
        # kernel = np.array(temp_kernel_matrix)
        # lf_matrix = self.L_stage(to_lf, temp_kernel_matrix)
        lf_matrix = self.mult_matrices(to_lf, temp_kernel_matrix)
        # print(lf_matrix)
        norm_L_stage = self.norm_L_stage(lf_matrix)
        lam = self.N_stage(norm_L_stage)
        spikes = self.gen_spikes(lam)
        dist = self.calculate_distance(spikes)
        self.calc_nmb_spikes(spikes)
        # to_print = to_print(trial_list)
        array = []
        for trial in self._trial_list:
            array.append(trial.get_total_spikes())
        return array


    # print(to_print)


    # tp1 = dist[0][100:200]
    # tp2 = dist[1][100:200]
    # tp3 = dist[2][100:200]
    # tp = [tp1,tp2,tp3]
    # plot.eventplot(dist)  
    # plot.show()