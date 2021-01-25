import numpy as np
import networkx as nx
from config import K, dtype


def maxcut(nqubits, norm=40, random_graph=True):
    """Builds maxcut hamiltonian"""
    if random_graph:
        aa = np.random.randint(1, nqubits*(nqubits-1)/2+1)
        graph = nx.random_graphs.dense_gnm_random_graph(nqubits, aa)
        V = nx.adjacency_matrix(graph).toarray()

    ham = K.zeros(shape=(2**nqubits,2**nqubits), dtype=dtype)
    Z = K.array([[1,0],[0,-1]], dtype=dtype)
    I = K.array([[1,0],[0,1]], dtype=dtype)
    for i in range(nqubits):
        for j in range(nqubits):
            h = K.eye(1, dtype=int)
            for k in range(nqubits):
                if (k == i) ^ (k == j):
                    h = K.kron(h, Z)
                else:
                    h = K.kron(h, I)
            M = K.eye(2**nqubits, dtype=int) - h
            if random_graph:
                ham += V[i,j] * M
            else:
                ham += M
    return - 1/norm * ham


def weighted_maxcut(nqubits, norm=40, random_graph=True):
    """Builds maxcut hamiltonian
    weights = K.array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318, 0.4236548 ],
                       [0.64589411, 0.43758721, 0.891773  , 0.96366276, 0.38344152],
                       [0.79172504, 0.52889492, 0.56804456, 0.92559664, 0.07103606],
                       [0.0871293 , 0.0202184 , 0.83261985, 0.77815675, 0.87001215],
                       [0.97861834, 0.79915856, 0.46147936, 0.78052918, 0.11827443]],
                       dtype=dtype)
    """
    if random_graph:
        aa = np.random.randint(1, nqubits*(nqubits-1)/2+1)
        graph = nx.random_graphs.dense_gnm_random_graph(nqubits, aa)
        V = nx.adjacency_matrix(graph).toarray()

    ham = K.zeros(shape=(2**nqubits,2**nqubits), dtype=dtype)
    Z = K.array([[1,0],[0,-1]], dtype=dtype)
    I = K.array([[1,0],[0,1]], dtype=dtype)
    for i in range(nqubits):
        for j in range(nqubits):
            h = K.eye(1, dtype=int)
            for k in range(nqubits):
                if (k == i) ^ (k == j):
                    h = K.kron(h, Z)
                else:
                    h = K.kron(h, I)
            w = dtype(np.random.uniform(0, 1))
            M = w * (K.eye(2**nqubits, dtype=int) - h)
            if random_graph:
                ham += V[i,j] * M
            else:
                ham += M
    return - 1/norm * ham


def rbm(nqubits, jmax=0.1):
    """Builds RBM hamiltonian."""
    graph = nx.generators.classic.turan_graph(nqubits, 2)
    A = nx.adjacency_matrix(graph, weight=None).toarray()
    B = dtype(np.random.uniform(0, jmax, nqubits))
    c = dtype(np.random.uniform(0, jmax, nqubits))
    W = dtype(np.random.uniform(0, jmax/2, (nqubits, nqubits)))
    J = A * W
    ham = K.zeros(shape=(2**nqubits,2**nqubits), dtype=dtype)
    Z = K.array([[1,0],[0,-1]], dtype=dtype)
    X = K.array([[0,1],[1,0]], dtype=dtype)
    I = K.array([[1,0],[0,1]], dtype=dtype)
    for i in range(nqubits):
        for j in range(nqubits):
            h = K.eye(1, dtype=int)
            for k in range(nqubits):
                if (k == i) ^ (k == j):
                    h = K.kron(h, Z)
                else:
                    h = K.kron(h, I)
            ham += J[i,j] * h

        h = K.eye(1, dtype=int)
        for k in range(nqubits):
            if k == i:
                h = K.kron(h, Z)
            else:
                h = K.kron(h, I)
        ham += B[i] * h
        h = K.eye(1, dtype=int)
        for k in range(nqubits):
            if k == i:
                h = K.kron(h, X)
            else:
                h = K.kron(h, I)
        ham -= c[i] * h
    return ham
