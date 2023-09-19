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

import random
from typing import TypedDict, Union
import dimod
import numpy as np

from devices.SimulatedAnnealingSampler import SimulatedAnnealingSampler
from solvers.Solver import *


class IterativeHeuristic(Solver):
    """
    Class for Iterative Heuristic Solver originally from Ruben Pfeiffer.
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()
        self.device_options = ["SimulatedAnnealer"]

    def get_device(self, device_option: str) -> SimulatedAnnealingSampler:
        if device_option == "SimulatedAnnealer":
            return SimulatedAnnealingSampler()
        else:
            raise NotImplementedError(f"Device Option {device_option}  not implemented")

    def get_parameter_options(self) -> dict:
        """
        Returns the configurable settings for this solver

        :return:
                 .. code-block:: python

                      return {
                                "number_of_reads": {
                                    "values": [100, 250, 500, 750, 1000],
                                    "description": "How many reads do you need?"
                                },
                                "iterations": {
                                    "values": [100, 250, 500, 750, 1000],
                                    "description": "How many iterations do you need?"
                                }
                            }

        """
        return {
            "number_of_reads": {
                "values": [100, 250, 500, 750, 1000],
                "description": "How many reads do you need?"
            },
            "iterations": {
                "values": [50, 100, 250, 500, 750, 1000],
                "description": "How many iterations do you need?"
            },
            "numOptVars": {
                "values": [12],
                "description": "How many numOptVars do you need?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config.

        .. code-block:: python

            number_of_reads: int
            iterations: int
            numOptVars: int

        """
        number_of_reads: int
        iterations: int
        numOptVars: int

    def run(self, mapped_problem: dict, device_wrapper: any, config: Config, **kwargs: dict) -> (dict, float):
        """
        Iterative Heuristic Solver.

        :param mapped_problem: dictionary with the key 'raw_matrix' where its value should be the QUBO matrix i
        :type mapped_problem: dict
        :param device_wrapper: Annealing device
        :type device_wrapper: any
        :param config: Annealing settings
        :type config: Config
        :param kwargs:
        :type kwargs: any
        :return: Solution and the time it took for the annealer to compute it
        :rtype: tuple(dict, float)
        """

        Q = mapped_problem['raw_matrix']

        device = device_wrapper.get_device()
        start = time() * 1000

        # Generating random initial solution
        arr = np.random.binomial(1, 0.5, size=Q.shape[0])
        numOptVars = config['numOptVars']
        val = np.dot(arr, np.dot(Q, arr))
        oldVal = val
        oldArr = arr
        numIters = 0
        solution = None

        if numOptVars > arr.size:
            numOptVars = arr.size
            logging.info("numOptVars to big, falling back to maximum")

        # Main loop
        while (numIters < config['iterations']):
            # Random choice of variables to optimize
            optVars = np.asarray(random.sample(range(arr.size), numOptVars))

            # Generating reduced QUBO
            Q_red = np.zeros((numOptVars, numOptVars))
            for i in range(arr.size):
                if i in optVars:
                    ind = np.where(optVars == i)[0][0]
                    Q_red[(ind, ind)] += Q[(i, i)]
                    for j in range(i + 1, arr.size):
                        if j in optVars:
                            Q_red[(ind, np.where(optVars == j)[0][0])] += Q[(i, j)]
                        else:
                            Q_red[(ind, ind)] += Q[(i, j)] * arr[j]
                else:
                    for j in range(i + 1, arr.size):
                        if j in optVars:
                            ind = np.where(optVars == j)[0][0]
                            Q_red[(ind, ind)] += Q[(i, j)] * arr[i]
            bqm = dimod.BinaryQuadraticModel.from_qubo(Q_red)

            # Solving reduced QUBO
            set = device.sample(bqm, num_reads=config['number_of_reads'])# Checking for improvement
            tempArr = arr
            for i in range(optVars.size):
                arr[optVars[i]] = set.first[0][i]
            newVal = np.dot(arr, np.dot(Q, arr))
            if newVal > val:
                arr = tempArr
                numIters += 1
            elif newVal == val:
                solution = set.first
                numIters += 1
            else:
                solution = set.first
                val = newVal
                numIters = 0

        # Map back array to the dict
        response = dict(zip(range(arr.size), arr.T))
        time_to_solve = round(time() * 1000 - start, 3)

        logging.info(f'Annealing finished in {time_to_solve} ms.')

        return response, time_to_solve, {}
