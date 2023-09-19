# -*- coding: utf-8 -*-
"""
@author: QUTAC Quantum

SPDX-FileCopyrightText: 2023 QUTAC

SPDX-License-Identifier: Apache-2.0

"""
import pulp



class IPSolver:
    """
    A class for implementing the integer program formulaton of the knapsack problem
    
    """
    def __init__(self, article_value, article_weight, knapsack_capacity, formulation):
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
        Returns
        ----------
            None

        """
        self.article_value = article_value
        self.article_weight = article_weight
        self.knapsack_capacity = knapsack_capacity
        self.number_of_knapsacks = len(self.knapsack_capacity)
        self.number_of_articles = len(self.article_weight)
        self.formulation = formulation
        self.model = None
        self.x = None


    def create_model(self):
        """
        Method to create the integer program formulation for multi-knapsack problem

        Parameters
        ----------
            None
        Returns
        ----------
             None : store the model, variables as class attributes
            
        """
        x = dict()
        for i in range(self.number_of_knapsacks):
            for j in range(self.number_of_articles):
                x[(i, j)] = pulp.LpVariable(f'x_{i}_{j}',
                                           lowBound=0,
                                           upBound=1,
                                           cat='Binary') # Continuous Binary

        y = dict()
        for i in range(self.number_of_knapsacks):
            if self.formulation == 'Binary':
                num_slack_bits = len(bin(self.knapsack_capacity[i])) - 2
            elif self.formulation == 'Linear':
                num_slack_bits = self.knapsack_capacity[i]
            for b in range(num_slack_bits):
                y[i, b] = pulp.LpVariable(f'y_{i}_{b}',
                                         lowBound=0,
                                         upBound=1,
                                         cat='Binary') # Continuous Binary

        model = pulp.LpProblem("Knapsack", pulp.LpMaximize)

        print('Variables Created')

        ##########################################################
        # Any item $i$ can be assigned to a maximum one knapsack.
        # It is possible that an item is not assigned to any knapsack
        ##########################################################
        for j in range(self.number_of_articles):
            model += pulp.lpSum([x[i, j] for i in range(self.number_of_knapsacks)]) <= 1
        print('Single Knapsack done')

        ##########################################################
        # Ensure that the capacity of any knapsack is not exceeded.
        # This is achieved by incorporated by introducing slack bits
        ##########################################################
        for i in range(self.number_of_knapsacks):
            first = pulp.lpSum([self.article_weight[j]*x[i, j] for j in range(self.number_of_articles)])
            if self.formulation == 'Binary':
                ###################################################
                num_slack_bits = len(bin(self.knapsack_capacity[i])) - 2
                second = pulp.lpSum([2 ** b * y[i, b] for b in range(num_slack_bits)])
            elif self.formulation == 'Linear':
                num_slack_bits = self.knapsack_capacity[i]
                second = pulp.lpSum([ y[i, b] for b in range(num_slack_bits)])

            model += first + second == self.knapsack_capacity[i]
        print('Constraint with Slack Done')

        ##########################################################
        # Maximize the total value of assigned items in knapsacks
        ##########################################################
        model  += pulp.lpSum([self.article_value[i,j]*x[i, j] for i in range(self.number_of_knapsacks) for j in range(self.number_of_articles)])

        print('Objective Done')
        self.model = model

        self.x = x
        self.y= y
        self.num_slack_bits=num_slack_bits


    ##########################################################################################
    # Solver method for the IP model
    ##########################################################################################
    def solve(self, solver):
        """
        Method to perform the classical optimization of QAOA parameters

        Parameters
        ----------
            solver : string
                the solver used to optimize the IP model                
            
        Returns
        ----------
            status: string
                model status after optimization, example: optimal, infeasible, etc.
            solution: dict
                dictionary containing the variable values for each variable in the integer programming model
            sol_y: dict
                dictionary containing the solutoin values for slack variables, only
            consumed: dict
                dictionary containing the consumption value of each knapsack
            objective_value: float
                the objective value of the optimized solution

        """
        if solver == 'GUROBI':
            self.model.solve(pulp.GUROBI())
        else:
            self.model.solve()
        status = pulp.LpStatus[self.model.status]

        solution = dict()
        for i in range(self.number_of_knapsacks):
            for j in range(self.number_of_articles):
                if 1e-10 < self.x[i,j].varValue:
                    solution[(i,j)]=self.x[i,j].varValue

        #############################################
        solution = dict()
        for i in range(self.number_of_knapsacks):
            for j in range(self.number_of_articles):
                solution[f'x_{i}_{j}'] = self.x[i, j].varValue

        for i in range(self.number_of_knapsacks):
            if self.formulation == 'Binary':
                num_slack_bits = len(bin(self.knapsack_capacity[i])) - 2
            elif self.formulation == 'Linear':
                num_slack_bits = self.knapsack_capacity[i]
            for b in range(num_slack_bits):
                solution[f'y_{i}_{b}'] = self.y[i, b].varValue
        ######################################################
        objective_value = self.model.objective.value()

        sol_y = dict()
        consumed=dict()
        for i in range(self.number_of_knapsacks):
            consumed[i]=0
            for j in range(self.number_of_articles):
                consumed[i] += self.article_weight[j]*self.x[i,j].varValue


        slacks=dict()
        for i in range(self.number_of_knapsacks):
            slacks[i]=0
            if self.formulation == 'Binary':
                num_slack_bits = len(bin(self.knapsack_capacity[i])) - 2
                for b in range(num_slack_bits):
                    if self.y[i, b].varValue > 1e-10:
                        sol_y[i, b] = (self.y[i, b].varValue, 2 ** b)
                        slacks[i] += self.y[i, b].varValue * 2 ** b
            elif self.formulation == 'Linear':
                num_slack_bits = self.knapsack_capacity[i]
                for b in range(num_slack_bits):
                    if self.y[i, b].varValue > 1e-10:
                        sol_y[i, b] = (self.y[i, b].varValue, 1)
                        slacks[i] += self.y[i, b].varValue


        total=dict()
        for i in range(self.number_of_knapsacks):
            total[i] = consumed[i] + slacks[i]

        return status, solution, sol_y, consumed, objective_value

#################################################

