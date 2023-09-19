# -*- coding: utf-8 -*-
"""
@author: QUTAC Quantum

SPDX-FileCopyrightText: 2023 QUTAC

SPDX-License-Identifier: Apache-2.0

"""

from pyqubo import Binary
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from qiskit_optimization import QuadraticProgram
from qiskit.algorithms import QAOA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.utils import QuantumInstance
from qiskit import Aer
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import SLSQP



class WSQAOASolver:
    """
    A class for implementing warm-start QAOA based on https://arxiv.org/pdf/2009.10095.pdf
    """
    
    def __init__(self, article_value, article_weight, knapsack_capacity, formulation,
                 single_penalty, capacity_penalty, objective_weight, n_layers,
                 epsilon):
        """
        Constructs all the necessary attributes for the WSQAOASolver object.

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
            epsilon : float
                egularization parameter for creating mixer Hamiltonian
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
        self.epsilon = epsilon
        self.H_capacity = None
        self.H_single = None
        self.H_obj = None
        self.number_of_knapsacks = len(self.knapsack_capacity)
        self.number_of_articles = len(self.article_weight)

        self.continuous_solution =  self._relaxed_qubo_solution()

        (self.H,
         self.variables,
         self.quadratic,
         self.var_to_index,
         self.index_to_var,
         self.model,
         self.offset) = self._create_ising_model()

        self.c_stars = [self.continuous_solution[key] for key in self.var_to_index]

        self.thetas = []
        for c_star in self.c_stars:
            if self.epsilon <= c_star <= 1-self.epsilon:
                self.thetas.append(2 * np.arcsin(np.sqrt(c_star)))
            elif c_star < self.epsilon:
                self.thetas.append(2 * np.arcsin(np.sqrt(self.epsilon)))
            elif c_star > 1-self.epsilon:
                self.thetas.append(2 * np.arcsin(np.sqrt(1-self.epsilon)))
        
    ##########################################################################################
    # compute solution to relaxed QUBO
    ##########################################################################################
    def _relaxed_qubo_solution(self):
        """
        Internal method to compute the solution to the realxed QUBO for warm-starting QAOA

        Parameters
        ----------
            None
        Returns
        ----------
            continuous_solution: dict
                dictionary containing the continuous solution value for all the problem variables

        """
        m = gp.Model("qp")
        x = dict()
        for i in range(self.number_of_knapsacks):
            for j in range(self.number_of_articles):
                x[(i,j)] = m.addVar(lb=0.0, ub=1.0, name=f'x_{i}_{j}')

        y = dict()
        for i in range(self.number_of_knapsacks):
            if self.formulation == 'Binary':
                num_slack_bits = len(bin(self.knapsack_capacity[i])) - 2
            elif self.formulation == 'Linear':
                num_slack_bits = self.knapsack_capacity[i]
            for b in range(num_slack_bits):
                y[i, b] = m.addVar(lb=0.0, ub=1.0, name=f'y_{i}_{b}')

        H_single = 0
        for j in range(self.number_of_articles):
            temp = sum([x[i, j] for i in range(self.number_of_knapsacks)])
            H_single += temp *(temp-1)
        H_single *= self.single_penalty

        H_capacity = 0
        for i in range(self.number_of_knapsacks):
            temp = sum([self.article_weight[j]*x[i,j] for j in range(self.number_of_articles)])
            if self.formulation == 'Binary':
                num_slack_bits = len(bin(self.knapsack_capacity[i])) - 2
                temp += sum([2**b * y[i, b] for b in range(num_slack_bits)])
            elif self.formulation == 'Linear':
                num_slack_bits = self.knapsack_capacity[i]
                temp += sum([y[i, b] for b in range(num_slack_bits)])
            temp -= self.knapsack_capacity[i]
            H_capacity += temp**2
        H_capacity *= self.capacity_penalty

        H_obj = sum([-self.article_value[i,j]*x[i, j] for i in range(self.number_of_knapsacks) for j in range(self.number_of_articles)])
        H_obj *= self.objective_weight
        H = H_single + H_capacity + H_obj

        m.setObjective(H,GRB.MINIMIZE)
        m.params.NonConvex=2
        m.optimize()
        continuous_solution=dict()
        for v in m.getVars():
            continuous_solution[v.VarName]= v.X
        return continuous_solution

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
            variables : list
                list of qubo variables
            quadratic : dict
                a dictionary containing all the interactions in the QUBO model, including
            var_to_index : dict
                a dictionary containing the mapping from QUBO variables to seqence index of the variables
            index_to_var : dict
                a dictionary containing the mapping from the variable index to the variable name in QUBO
            model : `pyqubo`
                pyqubo model object
            offset : float
                the constant offset term in the QUBO
            
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
        self.H_single *= self.single_penalty

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
        self.H_capacity *= self.capacity_penalty

        self.H_obj = sum([-self.article_value[i, j] * x[i, j] for i in range(self.number_of_knapsacks) for j in
                          range(self.number_of_articles)])
        self.H_obj *= self.objective_weight
        H = self.H_single + self.H_capacity + self.H_obj
        model = H.compile()
        self.variables = model.variables
        self.num_qubits = len(self.variables)
        
        
        self.quadratic, self.offset = model.to_qubo()

        self.var_to_index = dict([(n, i) for i, n in enumerate(self.variables)])
        self.index_to_var = dict([(i, n) for i, n in enumerate(self.variables)])

        
        H=np.zeros((self.num_qubits, self.num_qubits))
        for (var1, var2) in self.quadratic:
            (a,b)=(self.var_to_index[var1], self.var_to_index[var2])
            H[a,b]=self.quadratic[(var1, var2)]
        for i in range(self.num_qubits):
            H[i,i]/=2
        H=(H+H.T)
                
        
        return H, self.variables, self.quadratic, self.var_to_index, self.index_to_var, model, self.offset

      
    def solve_qaoa_knapsack(self, mixer_hamiltonian='Pauli-X', initial_params=None, total_iterations=None, num_layers=None, nshots=None):
        """
        Method to perform the classical optimization of QAOA parameters

        Parameters
        ----------
            mixer_hamiltonian : string
                mixer hamiltonian for warm-start QAOA, options= 'WS-QAOA', 'Pauli-X'
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
            ws_qaoa_result: MinimumEigenOptimizer.solve object
                result object obtained from QisKit warm-start QAOA solver 
            total_runtime: float
                total runtime required for classical optimization
            classical_iterations: int
                total number of classical iterations

        """
        # Define quadratic program for QAOA solver
        qp = QuadraticProgram()
        for i in range(len(self.H)):
            qp.binary_var('x{0}'.format(i))
        qp.minimize(constant=self.offset, linear = np.diag(self.H), quadratic = np.triu(self.H, 1))

         # for layers > 1: use same initial value for all gammas and all betas. parameters are given as: [beta1, beta2, ..., gamma1, gamma2, ...]
        params=np.repeat(initial_params, num_layers)
        
        #####################
        init_qc = QuantumCircuit(len(self.H))
        for idx, theta in enumerate(self.thetas):
            init_qc.ry(theta, idx)
        
        beta = Parameter("β")
        if mixer_hamiltonian=='Pauli-X':
            ws_mixer = QuantumCircuit(len(self.H))
            for idx in range(len(self.H)):
                ws_mixer.rx(-2 * beta, idx)
        elif mixer_hamiltonian=='WS-Mixer':
            ws_mixer = QuantumCircuit(len(self.H))
            for idx, theta in enumerate(self.thetas):
                ws_mixer.ry(-theta, idx)
                ws_mixer.rz(-2 * beta, idx)
                ws_mixer.ry(theta, idx)
        else:
            print("Error Message: Mixer Hamiltonian not generated. Supported mixers are 'Pauli-X' and 'WS-Mixer'")
        ###################################
        
        counts = []
        values = []
        def store_intermediate_result(eval_count, parameters, mean, std):
            counts.append(eval_count)
            values.append(mean)

        quantum_instance = QuantumInstance(backend=Aer.get_backend("qasm_simulator"), 
                                           shots=nshots)
        optimizer=SLSQP(maxiter=total_iterations)
        ws_qaoa_mes = QAOA( optimizer=optimizer, 
                            quantum_instance=quantum_instance, 
                            initial_state=init_qc,
                            mixer=ws_mixer,
                            reps=num_layers, 
                            include_custom=True,                             
                            initial_point=params, 
                            callback=store_intermediate_result)

        ws_qaoa = MinimumEigenOptimizer(ws_qaoa_mes) 
        self.ws_qaoa_result = ws_qaoa.solve(qp)

        all_solutions = []
        for i in range(len(self.ws_qaoa_result.samples)):
            all_solutions.append((self.ws_qaoa_result.samples[i].fval, 
                                  self.ws_qaoa_result.samples[i].probability))
        

        converge_counts = counts
        converge_vals = values
        classical_iterations = len(converge_counts)
        total_runtime = self.ws_qaoa_result.min_eigen_solver_result.optimizer_time


        print("result QAOA:\n", self.ws_qaoa_result)
        print("\ntime:", total_runtime, "\n\n")
        print("\nOpt. iterations:", classical_iterations, "\n\n")

        return all_solutions, self.ws_qaoa_result, total_runtime, classical_iterations
    
    