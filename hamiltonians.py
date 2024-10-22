import numpy as np
import networkx as nx
from config import K, dtype


def maxcut(nqubits, norm=40, random_graph=True):
    """Builds maxcut hamiltonian"""
    if random_graph:
        aa = np.random.randint(1, nqubits*(nqubits-1)/2+1)
        graph = nx.random_graphs.dense_gnm_random_graph(nqubits, aa)
        V = K.array(nx.adjacency_matrix(graph).toarray(), dtype=dtype)

    ham = K.zeros(shape=(2**nqubits,2**nqubits), dtype=dtype)
    Z = K.array([[1,0],[0,-1]], dtype=dtype)
    I = K.array([[1,0],[0,1]], dtype=dtype)
    for i in range(nqubits):
        for j in range(nqubits):
            h = K.eye(1, dtype=dtype)
            for k in range(nqubits):
                if (k == i) ^ (k == j):
                    h = K.kron(h, Z)
                else:
                    h = K.kron(h, I)
            M = K.eye(2**nqubits, dtype=dtype) - h
            if random_graph:
                ham += V[i,j] * M
            else:
                ham += M
    return - 1/norm * ham


def weighted_maxcut(nqubits, norm=40, random_graph=True):
    """Builds maxcut hamiltonian"""
    if random_graph:
        aa = np.random.randint(1, nqubits*(nqubits-1)/2+1)
        graph = nx.random_graphs.dense_gnm_random_graph(nqubits, aa)
        V = K.array(nx.adjacency_matrix(graph).toarray(), dtype=dtype)

    ham = K.zeros(shape=(2**nqubits,2**nqubits), dtype=dtype)
    Z = K.array([[1,0],[0,-1]], dtype=dtype)
    I = K.array([[1,0],[0,1]], dtype=dtype)
    for i in range(nqubits):
        for j in range(nqubits):
            h = K.eye(1, dtype=dtype)
            for k in range(nqubits):
                if (k == i) ^ (k == j):
                    h = K.kron(h, Z)
                else:
                    h = K.kron(h, I)
            w = dtype(np.random.uniform(0, 1))
            M = w * (K.eye(2**nqubits, dtype=dtype) - h)
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
            h = K.eye(1, dtype=dtype)
            for k in range(nqubits):
                if (k == i) ^ (k == j):
                    h = K.kron(h, Z)
                else:
                    h = K.kron(h, I)
            ham += J[i,j] * h

        h = K.eye(1, dtype=dtype)
        for k in range(nqubits):
            if k == i:
                h = K.kron(h, Z)
            else:
                h = K.kron(h, I)
        ham += B[i] * h
        h = K.eye(1, dtype=dtype)
        for k in range(nqubits):
            if k == i:
                h = K.kron(h, X)
            else:
                h = K.kron(h, I)
        ham -= c[i] * h
    return ham


def heisenberg(nqubits, norm=40):
    """Builds heisenberg hamiltonian"""
    ham = K.zeros(shape=(2**nqubits,2**nqubits), dtype=dtype)
    X = K.array([[0,1],[1,0]], dtype=dtype)
    Z = K.array([[1,0],[0,-1]], dtype=dtype)
    I = K.array([[1,0],[0,1]], dtype=dtype)
    for i in range(nqubits):
        hx = K.eye(1, dtype=dtype)
        for j in range(nqubits):
            if i in {j % nqubits, (j+1) % nqubits}:
                hx = K.kron(hx, X)
            else:
                hx = K.kron(hx, I)
        hz = K.eye(1, dtype=dtype)
        for j in range(nqubits):
            if i in {j % nqubits, (j+1) % nqubits}:
                hz = K.kron(hz, Z)
            else:
                hz = K.kron(hz, I)
        w = dtype(np.random.uniform(-1, 1))
        M = w * (hx  + 0.5 * hz)
        ham += M
    return - 1/norm * ham


def heisenberg_fully_connected(nqubits, norm=40, random_graph=True):
    """Builds maxcut hamiltonian"""
    ham = K.zeros(shape=(2**nqubits,2**nqubits), dtype=dtype)
    X = K.array([[0,1],[1,0]], dtype=dtype)
    Z = K.array([[1,0],[0,-1]], dtype=dtype)
    I = K.array([[1,0],[0,1]], dtype=dtype)
    for i in range(nqubits):
        for j in range(nqubits):
            hx = K.eye(1, dtype=dtype)
            for k in range(nqubits):
                if (k == i) ^ (k == j):
                    hx = K.kron(hx, X)
                else:
                    hx = K.kron(hx, I)
            hz = K.eye(1, dtype=dtype)
            for k in range(nqubits):
                if (k == i) ^ (k == j):
                    hz = K.kron(hz, Z)
                else:
                    hz = K.kron(hz, I)
            w = dtype(np.random.uniform(-1, 1))
            M = w * (hx + 0.5 * hz)
            if i != j:
                ham += M
    return - 1/norm * ham