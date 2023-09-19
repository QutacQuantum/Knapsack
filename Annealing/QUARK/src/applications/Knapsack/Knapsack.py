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

import logging
import os
from time import time
from typing import TypedDict
import json
import numpy as np

from applications.Application import *
from applications.Knapsack.mappings.MultiKSQUBO import MultiKSQUBO


class Knapsack(Application):
    """
    Multiple Knapsacks Problem
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__("Knapsack")
        self.mapping_options = [ "MultiKSQUBO"]

    def get_solution_quality_unit(self) -> str:
        return "Total Value"

    def get_mapping(self, mapping_option):

        if mapping_option == "MultiKSQUBO":
            return MultiKSQUBO()
        else:
            raise NotImplementedError(f"Mapping Option {mapping_option} not implemented")

    def get_parameter_options(self):
        """
        Returns the configurable settings for this application

        :return:
                 .. code-block:: python

                      return {
                                ""problem_type": {
                                                    "values": ['Scenario_1', 'Scenario_2', 'Scenario_3', 'Scenario_4', 'Scenario_5', 'Scenario_6',
                           'Scenario_7', 'Scenario_8', 'Scenario_9', 'Scenario_10'],
                                                    "description": "Which problem type do you want?"
                                                }
                            }

        """
        return {
            "problem_type": {
                "values": ['Scenario_1', 'Scenario_2', 'Scenario_3', 'Scenario_4', 'Scenario_5', 'Scenario_6',
                           'Scenario_7', 'Scenario_8', 'Scenario_9', 'Scenario_10'],
                "description": "Which problem type do you want?"
            }
        }

    class Config(TypedDict):
        """
        Attributes of a valid config

        .. code-block:: python

            problem_type: str

        """
        problem_type: str

    def generate_problem(self, config: Config) -> dict:
        """
        Creates a multiple Knapsack problem

        :param config: Config specifying the number of knapsacks
        :type config: Config
        :return: dictionary with items and capacities
        :rtype: dict
        """

        # structure from https://developers.google.com/optimization/bin/multiple_knapsack
        size = config['problem_type']  # ['Scenario_1', 'Scenario_2', 'Scenario_3', 'Scenario_4', 'Scenario_5']:
        file = os.path.join(os.path.dirname(__file__), "reference_data.json")
        with open(file, 'r') as f:
            dataset = json.load(f)

        self.application = {}
        self.application['values'] = dataset[size]['article_reward']
        self.application['weights'] = dataset[size]['article_weight']
        self.application['num_items'] = len(self.application['weights'])
        self.application['all_items'] = range(self.application['num_items'])
        self.application['bin_capacities'] = dataset[size]['knapsack_capacity']
        self.application['num_bins'] = len(self.application['bin_capacities'])
        self.application['all_bins'] = range(self.application['num_bins'])
        optimal = np.array(dataset[size]['optimal_solution'])

        return self.application

    # def process_solution(self, solution):
    #     start_time = time() * 1000
    #     return  solution, round(time() * 1000 - start_time, 3)

    def validate(self, solution: list) -> (bool, float):
        """
        Checks solutions produces overfull and items are packed only once

        :param solution: dict containing the picked items
        :type solution: dict
        :return: Boolean whether the solution is valid, time it took to validate
        :rtype: tuple(bool, float)
        """
        start_time = time() * 1000
        validation = True
        packed_items = []

        # Fixme this is very hacky
        # Purpose of this is if a slack va is MultiKSQUBO was violated solution is set to False and is not a dict with the packed weight as usual
        if type(solution) == bool and solution is False:
            logging.info(f"Violation of slack bits!")
            return solution, round(time() * 1000 - start_time, 3)

        # Check if any bins are overfull
        for x in self.application['all_bins']:
            weight_packed_in_bin = 0
            packed_items += solution[x]
            for i in solution[x]:
                weight_packed_in_bin += self.application['weights'][i]
            if weight_packed_in_bin > self.application['bin_capacities'][x]:
                validation = False
                logging.info(f"Violation of bin capacity in bin {x}: {weight_packed_in_bin} vs {self.application['bin_capacities'][x]} ❌")
        if len(packed_items) != len(list(set(packed_items))):
            logging.info("Packed at least one item more than once! ❌")
            validation = False

        if validation:
            logging.info("No capacities go violated ✅ !")
        else:
            logging.info(f"No valid solution: {solution}")
        return validation, round(time() * 1000 - start_time, 3)

    def evaluate(self, solution: dict) -> (int, float):
        """
        Calculates the values of the picked items

        :param solution: dict containing the picked items
        :type solution: dict
        :return: Packed value
        :rtype: tuple(int, float)
        """
        start = time() * 1000
        total_value = 0
        total_weight = 0
        # Check the value of the solution
        for b in solution:
            for i in solution[b]:
                total_value += self.application['values'][b][i]
                total_weight += self.application['weights'][i]

        logging.info(f"Total packed weight: {total_weight} with a value of {total_value}")
        return total_value, round(time() * 1000 - start, 3)

    def _serialize_range(obj):
        if isinstance(obj, range):
            return list(obj)
        return obj

    class RangeEncoder(json.JSONEncoder):
        """ Helper to save the problem"""

        def default(self, obj):
            if isinstance(obj, range):
                return list(obj)
            return json.JSONEncoder.default(self, obj)

    def save(self, path):
        with open(f"{path}/problem.json", 'w') as fp:
            json.dump(self.application, fp, cls=self.RangeEncoder)




