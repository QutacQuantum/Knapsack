# -*- coding: utf-8 -*-
"""
@author: QUTAC Quantum

SPDX-FileCopyrightText: 2023 QUTAC

SPDX-License-Identifier: Apache-2.0

"""
from pyqubo import Binary
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit.algorithms import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.utils import QuantumInstance
from qiskit import Aer
from qiskit.algorithms.optimizers import SLSQP

class QAOASolver:
    """
    A class for implementing standad QAOA as per https://arxiv.org/abs/1411.4028 
    
    """

    def __init__(self, article_value, article_weight, knapsack_capacity, formulation,
                 single_penalty, capacity_penalty, objective_weight, n_layers):
        """
        Constructs all the necessary attributes for the QAOA object.

        Parameters
        ----------
            article_value : numpy array
                contains the value (reward) for filling each item in a knapsack
            article_weight : numpy array
                contains the weights for all items
            knapsack_capacity : numpy array
                contains the capacity of each knapsack
            formulation : str
                type of formulation for slack bits 'Linear' or 'Binary'
            single_penalty : int
                penalty coefficient to ensure each item is palced in maximum one knapsack
            capacity_penalty : int
                penalty coefficient to ensure capacity constraint of each knapsack
            objective_weight : int
                weight for the objective term in the QUBO
            n_layers : int
                number of QAOA layers
        Returns
        ----------
            None

        """
        self.article_value = article_value
        self.article_weight = article_weight
        self.knapsack_capacity = knapsack_capacity
        self.single_penalty = single_penalty
        self.capacity_penalty = capacity_penalty
        self.objective_weight = objective_weight
        self.formulation = formulation
        self.n_layers = n_layers
        self.H_capacity = None
        self.H_single = None
        self.H_obj = None
        self.number_of_knapsacks = len(self.knapsack_capacity)
        self.number_of_articles = len(self.article_weight)

        (self.H,
         self.offset) = self._create_ising_model()
        


    ##########################################################################################
    # create the ising model for knapsack problem
    ##########################################################################################
    def _create_ising_model(self):
        """
        Internal method to create the ising model for multi-knapsack problem

        Parameters
        ----------
            None
        Returns
        ----------
             H: numpy array
                created QUBO from `pyqubo`
             offset: float
                offset term in the qubo
            
        """
        x = dict()
        for i in range(self.number_of_knapsacks):
            for j in range(self.number_of_articles):
                x[(i, j)] = Binary(f'x_{i}_{j}')

        y = dict()
        for i in range(self.number_of_knapsacks):
            if self.formulation == 'Binary':
                num_slack_bits = len(bin(self.knapsack_capacity[i])) - 2
            elif self.formulation == 'Linear':
                num_slack_bits = self.knapsack_capacity[i]
            for b in range(num_slack_bits):
                y[i, b] = Binary(f'y_{i}_{b}')

        self.H_single = 0
        for j in range(self.number_of_articles):
            temp = sum([x[i, j] for i in range(self.number_of_knapsacks)])
            self.H_single += temp * (temp - 1)
        self.H_single *= self.single_penalty # 20 A

        self.H_capacity = 0
        for i in range(self.number_of_knapsacks):
            temp = sum([self.article_weight[j] * x[i, j] for j in range(self.number_of_articles)])
            if self.formulation == 'Binary':
                num_slack_bits = len(bin(self.knapsack_capacity[i])) - 2
                temp += sum([2 ** b * y[i, b] for b in range(num_slack_bits)])
                
            elif self.formulation == 'Linear':
                num_slack_bits = self.knapsack_capacity[i]
                temp += sum([y[i, b] for b in range(num_slack_bits)])
            temp -= self.knapsack_capacity[i]
            self.H_capacity += temp ** 2
        self.H_capacity *= self.capacity_penalty # 5=B

        self.H_obj = sum([-self.article_value[i, j] * x[i, j] for i in range(self.number_of_knapsacks) for j in
                          range(self.number_of_articles)])
        self.H_obj *= self.objective_weight
        H = self.H_single + self.H_capacity + self.H_obj
        model = H.compile()
        self.variables = model.variables
        self.num_qubits = len(self.variables)
        
        
        quadratic, offset = model.to_qubo()

        var_to_index = dict([(n, i) for i, n in enumerate(self.variables)])
        
        H=np.zeros((self.num_qubits, self.num_qubits))
        for (var1, var2) in quadratic:
            (a,b)=(var_to_index[var1], var_to_index[var2])
            H[a,b]=quadratic[(var1, var2)]
            
        for i in range(self.num_qubits):
            H[i,i]/=2
        H=(H+H.T)
                

        return H, offset

    
    
    ##########################################################################################
    # Non-Gradient descent method on QAOA circuit to optimize the unitary rotation gate angles
    ##########################################################################################

    def solve_qaoa_knapsack(self, initial_params=None, total_iterations=None, num_layers=None, nshots=None):
        """
        Method to perform the classical optimization of QAOA parameters

        Parameters
        ----------
            initial_params : numpy array
                initial parameters for QAOA circuit for all the layers
            total_iterations : int
                total number of classical iterations to optimize the rotation angles
            num_layers : int
                number of QAOA layers
            nshots : int
                number of shots for the quantum circuit to compute the expectation value
            
        Returns
        ----------
            all_solutions: list
                List containing the qubo values and the probability for each sampled solution
            qaoa_result: MinimumEigenOptimizer.solve object
                result object obtained from QisKit QAOA solver 
            total_runtime: float
                total runtime required for classical optimization
            classical_iterations: int
                total number of classical iterations

        """
        
        #seed = np.random.seed(123)
        # Define quadratic program for QAOA solver
        qp = QuadraticProgram()
        for i in range(len(self.H)):
            qp.binary_var('x{0}'.format(i))
        qp.minimize(constant=self.offset, linear = np.diag(self.H), quadratic = np.triu(self.H, 1))

         # for layers > 1: use same initial value for all gammas and all betas. parameters are given as: [beta1, beta2, ..., gamma1, gamma2, ...]
        params=np.repeat(initial_params, num_layers)

        counts = []
        values = []
        def store_intermediate_result(eval_count, parameters, mean, std):
            counts.append(eval_count)
            values.append(mean)
        
        qins = QuantumInstance(backend=Aer.get_backend("qasm_simulator"), shots=nshots)#, seed_simulator=seed, seed_transpiler=seed)
        optimizer=SLSQP(maxiter=total_iterations)
        qaoa_mes = QAOA(optimizer=optimizer, 
                        reps=num_layers, 
                        include_custom=True, 
                        quantum_instance=qins, 
                        initial_point=params, 
                        callback=store_intermediate_result)

        qaoa = MinimumEigenOptimizer(qaoa_mes) 
        qaoa_result = qaoa.solve(qp)


        all_solutions = []
        for i in range(len(qaoa_result.samples)):
            all_solutions.append((qaoa_result.samples[i].fval, qaoa_result.samples[i].probability))
        
        converge_counts = counts
        converge_vals = values
        classical_iterations = len(converge_counts)        
        total_runtime = qaoa_result.min_eigen_solver_result.optimizer_time
        
        print("result QAOA:\n", qaoa_result)
        print("\ntime:", total_runtime, "\n\n")
        print("\nOpt. iterations:", classical_iterations, "\n\n")
        

        return all_solutions, qaoa_result, total_runtime, classical_iterations
    
