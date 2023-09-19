# -*- coding: utf-8 -*-
"""
@author: QUTAC Quantum

SPDX-FileCopyrightText: 2023 QUTAC

SPDX-License-Identifier: Apache-2.0

"""

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter

from scipy.optimize import minimize

import numpy as np
from numpy.random import uniform




class VQE:
   
    def __init__(self, n):
        """
        Constructs an n-qubit VQE instance with a (so far) empty quantum circuit

        Parameters
        ----------
            n : int
                number of qubits
        Returns
        ----------
            None

        """
        self.qc = QuantumCircuit(n, n)
        self.parameters = []
        self.n = n


    def initial_layer(self):
        """
        Adds an initial layer of parameterized rotations to the quantum circuit

        Parameters
        ----------
            None
        Returns
        ----------
            None

        """
        for i in range(self.n):
            self.parameters.append(Parameter("θ"+"_"+str(len(self.parameters))))
            self.qc.ry(self.parameters[i], i)
        self.qc.barrier()


    def add_layer(self):
        """
        Adds a layer of parameterized rotations and CNOTs to the quantum circuit 
        following the circuit architecture shown in https://arxiv.org/pdf/2102.05566.pdf

        Parameters
        ----------
            None
        Returns
        ----------
            None

        """        
        for i in range(self.n - 1):
            self.qc.cx(i,i+1)
            
            self.parameters.append(Parameter("θ"+"_"+str(len(self.parameters))))
            self.qc.ry(self.parameters[-1], i)
            self.parameters.append(Parameter("θ"+"_"+str(len(self.parameters))))
            self.qc.ry(self.parameters[-1], i+1)
            self.qc.cx(i,i+1)
            
            self.parameters.append(Parameter("θ"+"_"+str(len(self.parameters))))
            self.qc.ry(self.parameters[-1], i)
            self.parameters.append(Parameter("θ"+"_"+str(len(self.parameters))))
            self.qc.ry(self.parameters[-1], i+1)
        self.qc.barrier()



    def compute_energy(self, params, backend, Q, offset):
        '''
        Assign concrete values to the circuit parameters evaluate the bitstrings sampled on backend 
        w.r.t the QUBO function f(x) = x^T Q x + offset

        Parameters
        ---------- 
            params: numpy array
                values for the circuit parameters 
            backend: qiskit backend
                backend for circuit sampling
            Q: numpy matrix
                QUBO matrix
            offset: float
                QUBO offset
            num_iterations: int
                maximum number of optimizer iterations

        Returns
        ----------    
            mean_energy: float
                average energy obtained from sampling the quantum circuit
        '''                
        assert len(Q) == self.qc.num_qubits, "Circuit must have same number of qubits as there are variables in the QUBO!"
        # assert that number of parameters be equal to number of params in the circuit
        # bind the parameters to the parameters
        values = dict(zip(self.parameters, params))
        circ = self.qc.bind_parameters(values)
        for i in range(circ.num_qubits):
            circ.measure(i,i)

        #transpile circuit
        qc = transpile(circ, backend=backend)
        # run the circuit with the parameters passed in params
        job = backend.run(qc, shots=10000)   
        counts = job.result().get_counts()

        #calculate mean energy w.r.t the Hamiltonian resulting from the qubo
        mean_energy = 0
        for k in counts:
            bitstring = np.array([int(x) for x in k])
            # evaluate bitstring k for the QUBO matrix Q with numpy matrix multiplication
            mean_energy += (bitstring @ Q @ bitstring + offset)*counts[k]
        mean_energy /= sum([counts[k] for k in counts])

        return mean_energy        
    


    def optimize_layer(self, backend, Q, offset, num_iterations, initial_params):
        '''
        Train the parameterized circuit self.qc wrt the loss function induced by the QUBO matrix Q

        Parameters
        ---------- 
            backend: qiskit backend
                backend for circuit sampling
            Q: numpy matrix
                QUBO matrix
            offset: float
                QUBO offset
            num_iterations: int
                maximum number of optimizer iterations
            initial_params: numpy array
                initial guess for the circuit parameters

        Returns
        ----------    
            theta: numpy array
                optimized circuit parameters
        '''        
        func = lambda x: self.compute_energy(x, backend, Q, offset)

        
        theta = minimize(func, 
        x0=initial_params, 
        method='COBYLA',
        options={"maxiter": num_iterations})

        return theta



def solve_VQE(backend, Q, offset, num_layers):
    '''
    Solve a QUBO using VQE

    Input: 
    backend: IBMQ backend to run circuits on
    Q: QUBO matrix
    offset: offset in QUBO problem f(x) = x^T Q x + offset
    num_layers: number of layers in VQE ansatz

    Output: results dictionary containing
    results["solution"]: best bitstring
    results["fval"]: function value of best bitstring
    results["mean_energy"]: average function value among all bitstrings sampled
    results["counts"]: dictionary with bitstrings as keys and how often they were sampled from the final circuit as values
    results["params"]: list of optimal circuit parameters
    results["optimizer_loops"]: number of iterations until convergence
    
    '''

    n = len(Q)
    
    results = dict()
    vqe = VQE(n)
    vqe.initial_layer()
    for _ in range(num_layers):
        vqe.add_layer()
    
    num_params = len(vqe.parameters)
    initial = uniform(0, 2*np.pi, num_params)

    res=vqe.optimize_layer(backend, Q, offset,
        num_iterations=10000, 
        initial_params=initial)

    opt_params = res.x

    # find out best bitstring for circuit with optimal parameters
    values = dict(zip(vqe.parameters, opt_params))
    circ = vqe.qc.bind_parameters(values)
    for i in range(circ.num_qubits):
        circ.measure(i,i)
    

    #transpile circuit
    qc = transpile(circ, backend=backend)
    # run the circuit with the parameters passed in params
    job = backend.run(qc, shots=10000)   
    counts = job.result().get_counts()

    best = min(counts, key=lambda x: cost_function(x, Q))
 
    results["solution"] = best
    results["fval"] = cost_function(best, Q) + offset
    results["mean_energy"]=res.fun
    results["counts"]=counts
    results["params"]=opt_params.tolist()
    results["optimizer_loops"]=res.nfev
    
    print("VQE best solution: ", results["solution"])
    print("VQE best energy: ", results["fval"])
    print("Opt. iterations:", results["optimizer_loops"], "\n\n")


    return results

# evaluate QUBO function f(x) = x^T Q x
def cost_function(x, Q):
    '''
    Evaluate QUBO function f(x) = x^T Q x

    Parameters
    ---------- 
        x: numpy array
            bitstring
        Q: numpy matrix
            QUBO matrix

    Returns
    ----------    
        parameterized qiskit Circuit
    '''
    x_vector = [int(b) for b in x]
    assert Q.shape == (len(x),len(x)), "Dimensions of QUBO matrix and input vector don't match up!"
    return x_vector @ Q @ x_vector

def VQE_circuit(num_qubits, num_layers):
    '''
    Return parameterized VQE circuit following the circuit architecture described in https://arxiv.org/pdf/2102.05566.pdf

    Parameters
    ---------- 
        num_qubits: int
            number of qubits
        num_layers: int
            number of layers in VQE ansatz

    Returns
    ----------    
        parameterized qiskit Circuit
    '''
    vqe = VQE(num_qubits)
    vqe.initial_layer()
    for _ in range(num_layers):
        vqe.add_layer()
    return vqe.qc
