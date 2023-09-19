#  Copyright 2021 The QUARK Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import itertools
import logging
from typing import TypedDict, Union
import numpy as np
from dimod import *

from applications.Mapping import *
from solvers.Annealer_without_DWave import Annealer
from solvers.IterativeHeuristic_without_DWave import IterativeHeuristic


class MultiKSQUBO(Mapping):
    """
    QUBO formulation for the Knapsack

    """

    def __init__(self):
        super().__init__()
        self.solver_options = ["Annealer", "IterativeHeuristic"]
        self.problem = None
        self.n_aux = []

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this mapping

        :return:
                 .. code-block:: python

                     return {
                                        }
    }

        """
        return {
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

        """

    def map(self, problem: dict, config: Config) -> (dict, float):
        """
        Maps the networkx graph to a QUBO formulation.

        :param problem: dictionary with items and capacities
        :type problem: dict
        :param config: config with the parameters specified in Config class
        :type config: Config
        :return: dict with the QUBO, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = time() * 1000

        self.problem = problem

        weights = np.array(problem['weights'], dtype=int)
        values = np.array(problem['values'], dtype=int)
        capacities = np.array(problem['bin_capacities'], dtype=int)
        n_items = len(weights)
        n_knapsacks = len(capacities)

        b = 2 * np.max(np.concatenate(values).ravel())  # coefficient for value function
        a = 2 * np.max(np.concatenate(values).ravel())  # coefficients for knapsack capacity penalties
        gamma = 1  # coefficient for multiple knapsack penalties

        logging.info(f"Alpha penalty value: {a}")
        logging.info(f"Beta penalty value: {b}")

        generated_matrix, offset = self.multi_ks_qubo(n_items, capacities, weights, values, a, b, gamma)

        bqm = BinaryQuadraticModel(generated_matrix, 'BINARY', offset=offset)
        logging.info("Created Qubo")

        return {"Q": bqm.to_qubo()[0], "raw_matrix": generated_matrix}, round(time() * 1000 - start, 3)

    def multi_ks_qubo(self, num_items, C, weights, vals, alpha, beta, gamma):
        """
        QUBO for multi knapsack problem using binary slack bits

        Input:
        num_items: number of items to be distributed n
        C: capacity of the knapsacks
        weights: np-array with weights for each item.
        vals: np-arrays with values of each item within each knapsack
        alpha: penalty for putting one item into several knapsacks
        beta: penalty for overstepping capacities
        gamma: prefactor for objective function that maximizes the total value

        encoding of the bitstrings:
        binary variables for items: x_{i, j} = 1 if item j is in knapsack i
        binary variables for fillings of knapsack i (slack bits): y_{i, b}. The filling of knapsack i is encoded using binary digits by C_i - \sum_b^M_i (2^b * y_b),
        where M_i=floor(log2(C_i)). The binary digits encode the numbers {0, ..., C_i},
        corresponding to capacities {C_i, ..., 0} --> note the reversed order. In this way, no larger capacities than C_i can be encoded.

        bitstring: (x_{0, 0}, x_{0, 1}, ..., x_{0, n-1}, y_{0, 0}, ..., y_{0, M_0}, x_{1, 0}, ..., y_{m-1, M_m})
        where n denotes the number of items and m the number of knapsacks.

        Output: H, offset
        H: triagonal Hamiltonian matrix, without offset
        offset: constant offset
        """

        # number of binary slack bits for each knapsack = M_i+1. With M_i = floor(log2(C_i)). Thus, the number of slack bits is different for each knapsack.
        num_knapsacks = len(C)

        # array with number of slack bits for each knapsack
        num_slack_bits = (np.floor(np.log2(C)) + 1).astype(int)
        self.n_aux = num_slack_bits
        num_qubits = (num_items * num_knapsacks + num_slack_bits.sum()).astype(int)

        H = np.zeros((num_qubits, num_qubits))

        # diagonal terms: capacity penalty and objective
        for i in range(num_knapsacks):

            # diagonal terms in x_ij
            for j in range(num_items):
                ind = i * num_items + num_slack_bits[:i].sum() + j
                H[ind, ind] = beta * weights[j] ** 2 - 2 * beta * C[i] * weights[j] - gamma * vals[i][j]
            # diagonal terms in y_ib
            for b in range(num_slack_bits[i]):
                ind = (i + 1) * num_items + num_slack_bits[:i].sum() + b
                H[ind, ind] = beta * 2 ** (2 * b) - 2 * beta * C[i] * 2 ** b

        # off-diagonal terms: create list of pairs of unequal indices in i, j, b. Use only (i,k) and not (k,i) since this is already accounted for by a factor 2 in the x-x and y-y mixed terms.
        # The number of slack bits can be different
        knapsack_ind_pairs = list(itertools.combinations(range(num_knapsacks), 2))
        item_ind_pairs = list(itertools.combinations(range(num_items), 2))

        slack_ind_pairs = []
        for i in range(num_knapsacks):
            slack_ind_pairs.append(list(itertools.combinations(range(num_slack_bits[i]), 2)))

        # off-diagonal x-x-terms
        # penalty for one item being in two knapsacks
        for j in range(num_items):
            for pair in knapsack_ind_pairs:
                ind1 = pair[0] * num_items + num_slack_bits[:pair[0]].sum() + j
                ind2 = pair[1] * num_items + num_slack_bits[:pair[1]].sum() + j
                # print(ind1, ind2)
                H[ind1, ind2] = 2 * alpha
        # capacity penalty terms
        for i in range(num_knapsacks):
            for pair in item_ind_pairs:
                ind1 = i * num_items + num_slack_bits[:i].sum() + pair[0]
                ind2 = i * num_items + num_slack_bits[:i].sum() + pair[1]
                H[ind1, ind2] = 2 * beta * weights[pair[0]] * weights[pair[1]]

        # off-diagonal y-y terms
        # capacity penalty terms
        for i in range(num_knapsacks):
            for pair in slack_ind_pairs[i]:
                ind1 = (i + 1) * num_items + num_slack_bits[:i].sum() + pair[0]
                ind2 = (i + 1) * num_items + num_slack_bits[:i].sum() + pair[1]
                H[ind1, ind2] = 2 * beta * 2 ** (pair[0] + pair[1])

        # off-diagonal x-y terms
        # capacity penalty terms
        for i in range(num_knapsacks):
            for j in range(num_items):
                for b in range(num_slack_bits[i]):
                    ind1 = i * num_items + num_slack_bits[:i].sum() + j
                    ind2 = (i + 1) * num_items + num_slack_bits[:i].sum() + b
                    H[ind1, ind2] = 2 * beta * weights[j] * 2 ** b

        offset = beta * (C ** 2).sum()

        return H, offset

    def reverse_map(self, solution: dict) -> (list, float):
        """
        Maps the solution back to the representation needed by the Knapsack class for validation/evaluation.

        :param solution: dictionary containing the solution
        :type solution: dict
        :return: solution mapped accordingly, time it took to map it
        :rtype: tuple(dict, float)
        """
        start = time() * 1000
        mapped_solution = {}

        raw_bitstring = [None] * len(solution.items())
        for k, v in solution.items():
            raw_bitstring[k] = v
        # Filter out values that are 0
        solution = {k: v for k, v in solution.items() if v}

        slacks_per_bin = {}

        for item in solution:
            for x in self.problem['all_bins']:
                if x not in mapped_solution:
                    mapped_solution[x] = []
                # Item x can be in all of them y bins, therefore we have to find out which item is in which bin
                # + sum(self.n_aux[:x]) is there to skip the n_aux variables
                range_lower = x * self.problem['num_items'] + sum(self.n_aux[:x])
                range_upper = self.problem['num_items'] * (x + 1) + sum(self.n_aux[:x])
                # Check if item is in the range of this bin
                if item in range(range_lower, range_upper):
                    mapped_solution[x].append(item - x * self.problem['num_items'] - sum(self.n_aux[:x]))
                if x not in slacks_per_bin:
                    slacks_per_bin[x] = raw_bitstring[range_upper: range_upper + self.n_aux[x]]

        for k, v in mapped_solution.items():
            weights = sum([self.problem['weights'][x] for x in v]) if len(v) > 0 else 0
            if k in slacks_per_bin:
                slacks_for_specific_bin = slacks_per_bin[k]
                temp = sum([2 ** b * slacks_for_specific_bin[b] for b in range(len(slacks_for_specific_bin))])
            else:
                temp = 0
            if weights + temp != self.problem['bin_capacities'][k]:
                logging.info(f"Violated slack bit in bin {k}:  {weights + temp} (weights: {weights}, temp: {temp}) vs {self.problem['bin_capacities'][k]}")
                mapped_solution = False

        return mapped_solution, round(time() * 1000 - start, 3)

    def get_solver(self, solver_option: str) -> Union[Annealer, IterativeHeuristic]:

        if solver_option == "Annealer":
            return Annealer()
        elif solver_option == "IterativeHeuristic":
            return IterativeHeuristic()
        else:
            raise NotImplementedError(f"Solver Option {solver_option} not implemented")
