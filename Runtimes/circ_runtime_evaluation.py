# -*- coding: utf-8 -*-
"""
@author: QUTAC Quantum

SPDX-FileCopyrightText: 2023 QUTAC

SPDX-License-Identifier: Apache-2.0

"""

import numpy as np
import collections

import warnings

from qiskit import Aer, transpile
from qiskit.utils import QuantumInstance
from qiskit_optimization import QuadraticProgram
from qiskit.algorithms.optimizers import SLSQP
from qiskit.algorithms import QAOA

warnings.filterwarnings("ignore", category=DeprecationWarning) 

from QAOA.sources.KnapsackQAOA_Qiskit import QAOASolver
from VQE.VQE import *


class circ_runtime_evaluation:

    """
    A class for deriving the mean circuit depth and circuit runtimes for QAOA and VQE on a given backend
    ...
    
    Methods
    -------

    get_gate_times_from_backend():
        return gate execution times for the given IBM-Q backend averaged over all qubits of the backend

    get_transpiled_circs_and_mean_depths():
        transpile the qaoa or vqe circuit on the backend and measure its depth for each scenario
    
    get_circ_runtimes():
        calculate the mean circuit execution time for each scenario
    
    """

    def __init__(self,dataset, scenarios, num_layers, objective_weight, formulation, comp_averages, opt_level, backend, algorithm):
        """
        Constructs all the necessary attributes.

        Parameters
        ----------
            dataset : dict
                contains information about all scenarios (problem instances)
            scenarios : list
                numbers (starting from 1) of the scenarios which should be considered in the evaluation
            n_layers : int
                number of layers for QAOA or VQE circuit
            objective_weight : int
                weight for the objective term in the QUBO
            formulation : str
                type of formulation for slack bits 'Linear' or 'Binary'
            comp_averages : int
                nmumber of times to repeat the transpilation
            opt_level : int
                optimization level for transpilation
            backend : Qiskit backend object
                the backend to be used for transpilation (defines the native gate set and gate execution times)
            algorithm: str
                whether QAOA or VQE should be considered. Values are: 'qaoa', 'vqe'

        Returns
        ----------
            None
        """
        
        self.dataset = dataset
        self.scenarios = scenarios
        self.num_layers = num_layers
        self.objective_weight = objective_weight
        self.formulation = formulation
        self.comp_averages = comp_averages
        self.opt_level = opt_level
        self.backend = backend
        self.algorithm = algorithm

        self.gate_times = self.get_gate_times_from_backend()
        (self.depths_mean, self.circs) = self.get_transpiled_circs_and_mean_depths()
        self.runtimes_mean = self.get_circ_runtimes()


    ##########################################################################################
    # Calculate mean gate execution times for the given backend
    ##########################################################################################

    def get_gate_times_from_backend(self):
        
        """
        Method to calculate the execution times (given in ns) for each gate, averaged over all qubits of the backend

        Returns
        ----------
            gate_times: dictionary of the form {gate name, mean gate time, std of mean gate time}
        """

        config = self.backend.configuration()
        self.gate_set=config.basis_gates
        self.gate_set.append('save_expval')
        self.cmap=config.coupling_map
        self.backend_qubits = len(set(sum(self.cmap, [])))
        
        prop_dict = self.backend.properties().to_dict()
        gate_times={}

        # Gate times in ns 
        for gate in self.gate_set:
            if gate == 'save_expval':
                gate_times[gate] = (0,0)
            else:
                gate_times_tmp=[]
                for j in range(len(prop_dict['gates'])):

                    if prop_dict['gates'][j]['gate'] == gate and prop_dict['gates'][j]['gate'] != 'reset':
                        gate_times_tmp.append(prop_dict['gates'][j]['parameters'][1]['value'])

                    if prop_dict['gates'][j]['gate']==gate and prop_dict['gates'][j]['gate'] == 'reset':
                        gate_times_tmp.append(prop_dict['gates'][j]['parameters'][0]['value'])
                gate_times[gate] = round(np.mean(gate_times_tmp),3), round(np.std(gate_times_tmp),3)

        return gate_times
    

    ##########################################################################################
    # Create and transpile circuits for given algorithm on the backend
    ##########################################################################################
    def get_transpiled_circs_and_mean_depths(self):
        
        """
        Method to transpile the qaoa or vqe circuit on the backend and measure its depth

        Returns
        ----------
            depths_mean: np.array with shape (scenarios, 2)
                mean and std depth of the transpiled circuit, averaged over several transpilation runs for each scenario
            circs: dict
                conatins the number of qubits, number of transpilation run and the transpiled circuit for all scenarios.
        """

        depths_mean=np.zeros((len(self.scenarios), 2))
        circs = []

        for num in range(len(self.scenarios)):

            scenario = f'scenario_{num+1}'
            print(scenario)
            article_reward = np.array(self.dataset[scenario]['article_reward'])
            article_weight = np.array(self.dataset[scenario]['article_weight'])
            knapsack_capacity = np.array(self.dataset[scenario]['knapsack_capacity'])
            (number_of_knapsacks, number_of_articles) = article_reward.shape
            num_qubits = number_of_knapsacks*number_of_articles
            for i in range(number_of_knapsacks):
                num_qubits += len(bin(knapsack_capacity[i])) - 2

            single_penalty = int(2*np.max(article_reward))
            capacity_penalty = int(2*np.max(article_reward))

            if self.algorithm == 'qaoa':

                qaoa = QAOASolver(article_reward, article_weight, knapsack_capacity, self.formulation,
                            single_penalty, capacity_penalty, self.objective_weight, self.num_layers)

                qp = QuadraticProgram()
                for i in range(len(qaoa.H)):
                    qp.binary_var('x{0}'.format(i))
                qp.minimize(constant=qaoa.offset, linear = np.diag(qaoa.H), quadratic = np.triu(qaoa.H, 1))

                # dummy initial params for qaoa: just placeholders
                init_params=[0.1, 0.1] 
                params=np.repeat(init_params, self.num_layers)

                qins = QuantumInstance(backend=Aer.get_backend("qasm_simulator"))
                optimizer=SLSQP()
                qaoa_mes = QAOA(optimizer=optimizer, 
                                reps=self.num_layers, 
                                include_custom=True, 
                                quantum_instance=qins, 
                                initial_point=params)
                

                qc_ising=qp.to_ising()[0]
                qc=qaoa_mes.construct_circuit(params, qc_ising)[0]

            if self.algorithm == 'vqe':
                qc = VQE_circuit(num_qubits, self.num_layers) 

            depths_raw=[]
            for av in range(self.comp_averages):
                transpiled_circ=transpile(qc, basis_gates=self.gate_set, coupling_map=self.cmap, optimization_level=self.opt_level)
                depths_raw.append(transpiled_circ.depth())
                circs.append([num_qubits, av, transpiled_circ])


            depths_mean[num,0]=np.mean(depths_raw)
            depths_mean[num,1]=np.std(depths_raw)
        
        return depths_mean, circs

    ##########################################################################################
    # Derive circuit runtime for each transpiled circ and average over transpilation runs
    ##########################################################################################
    def get_circ_runtimes(self):
                
        """
        Method to calculate the circuit execution time from the transpiled circuit and the gate execution times.
        The transpiled circuits are converted into a schedule, taking into account parallel execution of gates.
        From this, the circuit runtime is derived as the maximum duration over all gates.
        The runtimes for the several transpilation runs are averaged for each scenario.

        Returns
        ----------
            runtimes_mean: np.array with shape (scenarios, 2)
                mean and std depth of the circuit runtime, averaged over several transpilation runs for each scenario
        """

        runtimes=[]
        qubits_used=[] # for testing puposes
        circ_schedules=[] # for testing purposes

        for cc in range(len(self.circs)):
            transpiled_circ = self.circs[cc][2]
            num_qubits=self.circs[cc][0]

            circ_data = list(transpiled_circ.data)

            qubits_used=[]
            for n in range(self.backend_qubits): #go through all qubits of the backend
                for j in range(len(list(transpiled_circ.data))):
                    if list(transpiled_circ.data)[j][1][0].index == n and list(transpiled_circ.data)[j][0].name != 'barrier' and list(transpiled_circ.data)[j][0].name != 'measure' and circ_data[j][0].name != 'save_expval':
                        qubits_used.append(n)



            qubits_used_nums=list(collections.Counter(qubits_used).keys())

            circ_dict={}
            step=dict((el,0) for el in qubits_used_nums)
            time_step=dict((el,0) for el in qubits_used_nums)
            for j in range(len(circ_data)):
                if circ_data[j][0].name != 'measure' and circ_data[j][0].name != 'barrier' and circ_data[j][0].name != 'save_expval':
                    if j == 0:
                        step[circ_data[j][1][0].index]+=1
                        time_step[circ_data[j][1][0].index]+=self.gate_times[circ_data[j][0].name][0]
                        circ_dict[circ_data[j][1][0].index, step[circ_data[j][1][0].index]] = [time_step[circ_data[j][1][0].index], circ_data[j][0].name, circ_data[j][1][0].index]

                    if j > 0:
                        if circ_data[j][0].num_qubits == 1: 
                            step[circ_data[j][1][0].index]+=1
                            time_step[circ_data[j][1][0].index]+=self.gate_times[circ_data[j][0].name][0]
                            circ_dict[circ_data[j][1][0].index, step[circ_data[j][1][0].index]] = [time_step[circ_data[j][1][0].index], circ_data[j][0].name, circ_data[j][1][0].index]

                        if circ_data[j][0].num_qubits == 2: 
                            id1=circ_data[j][1][0].index
                            id2=circ_data[j][1][1].index
                            max_step = max([step[id1], step[id2]])
                            max_time = max([time_step[id1], time_step[id2]])
                            step[id1] = max_step + 1
                            step[id2] = max_step + 1
                            time_step[id1] = max_time + self.gate_times[circ_data[j][0].name][0]
                            time_step[id2] = max_time + self.gate_times[circ_data[j][0].name][0]

                            circ_dict[id1, step[id1]] = [time_step[id1], circ_data[j][0].name, id2]



            od = collections.OrderedDict(sorted(circ_dict.items()))
            qubit_times=np.array(list(od.values()))[:,0].astype(float)
            max_qubit_time = max(qubit_times)

            circ_schedules.append([num_qubits, od])
            runtimes.append([num_qubits, max_qubit_time]) #in ns
            qubits_used.append([num_qubits, qubits_used_nums])


        runtimes_final=np.array(runtimes)[:,1].reshape((len(self.scenarios), self.comp_averages))
        runtimes_mean=np.zeros((len(self.scenarios), 2))
        for j in range(len(self.scenarios)):
            runtimes_mean[j,0]=np.mean(runtimes_final[j])
            runtimes_mean[j,1]=np.std(runtimes_final[j])
        
        return runtimes_mean
