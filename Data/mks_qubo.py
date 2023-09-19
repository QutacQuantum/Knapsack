# -*- coding: utf-8 -*-
"""
@author: QUTAC Quantum

SPDX-FileCopyrightText: 2023 QUTAC

SPDX-License-Identifier: Apache-2.0

"""
import numpy as np
import itertools



def multi_ks_qubo(num_items, C, weights, vals, alpha, beta, gamma):

    '''
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
    '''

    #number of binary slack bits for each knapsack = M_i+1. With M_i = floor(log2(C_i)). Thus, the number of slack bits is different for each knapsack.
    num_knapsacks = len(C)
        
    # array with number of slack bits for each knapsack
    num_slack_bits = (np.floor(np.log2(C)) + 1).astype(int)

    num_qubits=(num_items*num_knapsacks + num_slack_bits.sum()).astype(int)

    H=np.zeros((num_qubits, num_qubits))

    #diagonal terms: capacity penalty and objective
    for i in range(num_knapsacks):
        # diagonal terms in x_ij
        for j in range(num_items):
            ind = i*num_items + num_slack_bits[:i].sum() + j
            H[ind, ind] = beta*weights[j]**2 - 2*beta*C[i]*weights[j] - gamma*vals[i][j]
        #diagonal terms in y_ib
        for b in range(num_slack_bits[i]):
            ind = (i+1)*num_items + num_slack_bits[:i].sum() + b
            H[ind, ind] = beta*2**(2*b) - 2*beta*C[i]*2**b


    #off-diagonal terms: create list of pairs of unequal indices in i, j, b. Use only (i,k) and not (k,i) since this is already accounted for by a factor 2 in the x-x and y-y mixed terms.
    # The number of slack bits can be different 
    knapsack_ind_pairs=list(itertools.combinations(range(num_knapsacks), 2))
    item_ind_pairs=list(itertools.combinations(range(num_items), 2))

    slack_ind_pairs=[]
    for i in range(num_knapsacks):
        slack_ind_pairs.append(list(itertools.combinations(range(num_slack_bits[i]), 2)))

    #off-diagonal x-x-terms
    # penalty for one item being in two knapsacks
    for j in range(num_items):
        for pair in knapsack_ind_pairs:
            ind1 = pair[0]*num_items + num_slack_bits[:pair[0]].sum() + j
            ind2 = pair[1]*num_items + num_slack_bits[:pair[1]].sum() + j
            #print(ind1, ind2)
            H[ind1, ind2] = 2*alpha
    #capacity penalty terms
    for i in range(num_knapsacks):
        for pair in item_ind_pairs:
            ind1 = i*num_items + num_slack_bits[:i].sum() + pair[0]
            ind2 = i*num_items + num_slack_bits[:i].sum() + pair[1]
            H[ind1, ind2] = 2*beta*weights[pair[0]]*weights[pair[1]]

    #off-diagonal y-y terms
    #capacity penalty terms
    for i in range(num_knapsacks):
        for pair in slack_ind_pairs[i]:
            ind1 = (i+1)*num_items + num_slack_bits[:i].sum() + pair[0]   
            ind2 = (i+1)*num_items + num_slack_bits[:i].sum() + pair[1] 
            H[ind1, ind2] = 2*beta*2**(pair[0]+pair[1])

    #off-diagonal x-y terms
    #capacity penalty terms
    for i in range(num_knapsacks):
        for j in range(num_items):
            for b in range(num_slack_bits[i]):
                ind1 = i*num_items + num_slack_bits[:i].sum() + j
                ind2 = (i+1)*num_items + num_slack_bits[:i].sum() + b
                H[ind1, ind2] = 2*beta*weights[j]*2**b


    offset = beta*(C**2).sum()

    return H, offset

def multi_ks_qubo_reduced_count(num_items, C, weights, vals, alpha, beta, gamma):

    '''
    QUBO for multi knapsack problem using binary slack bits and reduced count

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
    binary variables for fillings of knapsack i (slack bits): y_{i, b}. The filling of knapsack i is encoded using binary digits by C_i - \sum_b^L (2^b * y_b), where L=floor(log2(C_i - C_i^min)).
    C_i^min: The minimal useful capacity to start counting with (to reduce the number of needed slack bits even further). The binary digits encode the numbers {0,..., C_i - C_i^min}, 
    corresponding to capacities {C_i, ..., C_i^min} --> note the reversed order. In this way, no larger capacities than C_i can be encoded.

    bitstring: (x_{0, 0}, x_{0, 1}, ..., x_{0, n-1}, y_{0, 0}, ..., y_{0, L}, x_{1, 0}, ..., y_{m-1, L})
    where n denotes the number of items and m the number of knapsacks.

    Output: H, offset
    H: triagonal Hamiltonian matrix, without offset
    offset: constant offset
    '''

    #number of binary slack bits for each knapsack = L+1. With L = floor(log2(C - c_min)) = floor(log2(max(weights)-1)), since c_min = C - max(weights) + 1. Thus, the number of slack bits is the same for all knapsacks.
    num_knapsacks = len(C)
    
    num_slack_bits = np.floor(np.log2(int(np.max(weights))-1)).astype(int) + 1

    num_qubits=(num_items + num_slack_bits)*num_knapsacks

    H=np.zeros((num_qubits, num_qubits))

    #diagonal terms: capacity penalty and objective
    for i in range(num_knapsacks):
        # diagonal terms in x_ij
        for j in range(num_items):
            ind = i*(num_items+num_slack_bits) + j
            H[ind, ind] = beta*weights[j]**2 - 2*beta*C[i]*weights[j] - gamma*vals[i][j]
        #diagonal terms in y_ib
        for b in range(num_slack_bits):
            ind = i*(num_items+num_slack_bits) + num_items + b
            H[ind, ind] = beta*2**(2*b) - 2*beta*C[i]*2**b


    #off-diagonal terms: create list of pairs of unequal indices in i, j, b. Use only (i,k) and not (k,i) since this is already accounted for by a factor 2 in the x-x and y-y mixed terms.
    # The number of slack bits can be different 
    knapsack_ind_pairs=list(itertools.combinations(range(num_knapsacks), 2))
    item_ind_pairs=list(itertools.combinations(range(num_items), 2))
    slack_ind_pairs=list(itertools.combinations(range(num_slack_bits), 2))

    #off-diagonal x-x-terms
    # penalty for one item being in two knapsacks
    for j in range(num_items):
        for pair in knapsack_ind_pairs:
            ind1 = pair[0]*(num_items+num_slack_bits) + j
            ind2 = pair[1]*(num_items+num_slack_bits) + j
            #print(ind1, ind2)
            H[ind1, ind2] = 2*alpha
    #capacity penalty terms
    for i in range(num_knapsacks):
        for pair in item_ind_pairs:
            ind1 = i*(num_items+num_slack_bits) + pair[0]
            ind2 = i*(num_items+num_slack_bits) + pair[1]
            H[ind1, ind2] = 2*beta*weights[pair[0]]*weights[pair[1]]

    #off-diagonal y-y terms
    #capacity penalty terms
    for i in range(num_knapsacks):
        for pair in slack_ind_pairs:   
            ind1 = i*(num_items+num_slack_bits) + num_items + pair[0]
            ind2 = i*(num_items+num_slack_bits) + num_items + pair[1]
            H[ind1, ind2] = 2*beta*2**(pair[0]+pair[1])

    #off-diagonal x-y terms
    #capacity penalty terms
    for i in range(num_knapsacks):
        for j in range(num_items):
            for b in range(num_slack_bits):
                ind1 = i*(num_items+num_slack_bits) + j
                ind2 = i*(num_items+num_slack_bits) + num_items + b
                H[ind1, ind2] = 2*beta*weights[j]*2**b


    offset = beta*(C**2).sum()

    return H, offset



def single_ks_qubo_linear(n, c_max, w, v, A, B, c_min=None):
    
    '''
    QUBO for single knapsack problem using linear slack bits

    Input: n, c_max, w, v, A, B, c_min
    n: number of items to be distributed
    C_max: capacity of the knapsack
    w: np-array with weights for each item. dim=n
    v: np-arrays with values of each item, dim=n
    A: penalty for overstepping capacity 
    B: penalty for not maximizing the total value. A > max(values)*B > 0 
    c_min: Optional mininmal capacity to start counting for auxiliary variables, if not specified, c_max - max(w) + 1 is being used.

    
    Output: q
    q: triagonal Hamiltonian matrix, without offset
    offset: constant offset
    '''

    # compute single qubo size

    if c_min is None:
        c_min = np.asarray(c_max)-int(np.max(w))+1
    
    n_aux = (c_max - c_min + 1) if c_max > c_min else 0
    n_qubo = n + n_aux
   
    q = np.zeros((n_qubo, n_qubo))
    for i in range(n):
        # add cost elements (x_i, x_i)
        q[i, i] += - B * v[i]
        # add capacity penalty elements (x_i, x_i)
        q[i, i] += A * w[i]**2
        if n_aux == 0:
            q[i, i] += - 2 * A * c_max * w[i] # factor of 2 due to mixed quadratic term

    # add capacity penalty elements (x_i, x_ip)
    for i in range(n):
        for ip in range(i+1, n):
            q[i, ip] += 2 * A * w[i] * w[ip] # factor of 2 due to summation over (i, ip): i < ip

    # add capacity penalty elements (y_k, y_k) -> q_i_i
    # transformation: i = n + k - c_min
    if n_aux > 0:
        for k in range(c_min, c_max + 1):
            i = n + k - c_min
            q[i, i] += A * (k**2 - 1)

    # add capacity penalty elements (y_k, y_kp) -> q_i_ip
    # transformation: i = n + k - c_min
    #                 ip = n + kp - c_min
    if n_aux > 0:
        for k in range(c_min, c_max + 1):
            i = n + k - c_min
            for kp in range(k + 1, c_max + 1):
                ip = n + kp - c_min
                q[i, ip] += 2 * A * (k * kp + 1) # factor of 2 due to summation over (k, kp): k < kp
    
    # add capacity penalty elements (x_i, y_k) -> q_i_ip
    # transformation: ip = k + n - c_min
    if n_aux > 0:
        for i in range(n):
            for k in range(c_min, c_max + 1):
                ip = n + k - c_min
                q[i, ip] += - 2 * A * k * w[i] # factor of 2 due to mixed quadratic term
 

    if n_aux == 0:
        offset = np.dot(A,c_max**2)
    if n_aux > 0:
        offset = np.sum(A)
    
    return q, offset


def multi_ks_qubo_linear(m, n, c_max, w, v, A, B, P, c_min=None):

    '''
    QUBO for multiple knapsack problem using linear slack bits

    Input: m, n, c_max, w, v, A, B, P, c_min
    m: number of knapsacks
    n: number of items to be distributed
    c_max: array with capacities for each knapsack. dim=m
    w: np-array with weights for each item. dim=n
    v: np-arrays with values of each item for each knapsack, dim=(n, m)
    A: array with penalties for overstepping capacity for each knapsack. dim=m
    B: array with penalties for not maximizing the total value. A > max(values)*B > 0 for each knapsack. dim=m
    P: penalty for having one item in several knapsacks. Assumed to be the same for each item
    c_min: Optional mininmal capacity to start counting for auxiliary variables, if not specified, c_max - max(w) +1 is being used.
    
    Output: q, offset
    q: triagonal Hamiltonian matrix without constant offset
    offset: constant offset
    '''


    # number auxiliary variables y_ij per knapsack j
    if c_min is None:
        c_min = np.asarray(c_max)-int(np.max(w))+1

    n_aux = [c1 - c2 + 1 if c1 > c2 else 0 for (c1, c2) in zip(c_max, c_min)]

    # total size of QUBO
    n_qubo = m * n + sum(n_aux)

    # initialize QUBO with 0
    q = np.zeros((n_qubo, n_qubo))

    # sizes of the single QUBOs
    q_size = [n + s for s in n_aux]

    o = 0
    # add single QUBOs for knapsacks j=0,m-1
    for j in range(m):
        qs = single_ks_qubo_linear(n, c_max[j], w, v[j], A[j], B[j], c_min[j])[0]
        s = len(qs)
        q[o:o+s, o:o+s] = qs
        o += s

    
    # add cross-QUBO penalty elements (x_i_j, x_i_jp) -> q_l_lp
    # transformation: l = sum(q_size[0:j]) + i
    # transformation: lp = sum(q_size[0:jp]) + i
    for i in range(n):
        for j in range(m):
            l = sum(q_size[0:j]) + i
            for jp in range(j+1, m):
                lp = sum(q_size[0:jp]) + i
                q[l, lp] = P
    
   
    #calculate constant offset
    if n_aux[0] == 0:
        offset = np.dot(A,c_max**2)
    else:
        offset = np.sum(A)
    
    return q, offset


if __name__ == '__main__':
    '''example problem with 2 knapsacks and 6 items'''
    
    
    m_knapsacks = 2 # len(capacities), len(values)
    n_items = 4 # len(weights), len(values[0])
    max_capacities = [ 5, 5 ]
    #min_capacities = [ 3, 3]
    weights = [3, 3, 2, 2 ]
    values = [ 
        [3, 3, 2, 2 ],
        [ 3, 3, 2, 2 ]
    ]

    # coefficients for the Hamiltonian
    B = [5] * m_knapsacks                     # coefficient for value function
    A = [ 20 ]  * m_knapsacks # coefficients for knapsack capacity penalties
    P = 50   # coefficient for multiple knapsack penalties


    qubo, offset = multi_ks_qubo_linear(m_knapsacks, n_items, max_capacities, weights, values, P, A, B)
    

    np.set_printoptions(linewidth=240)
    print(qubo)
