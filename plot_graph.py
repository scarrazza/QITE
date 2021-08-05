import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# weighted maxcut
nqubits = 12
np.random.seed(0)
aa = np.random.randint(1, nqubits*(nqubits-1)/2+1)
#graph = nx.random_graphs.dense_gnm_random_graph(nqubits, aa)
graph = nx.generators.classic.turan_graph(nqubits, 2)

plt.figure()
nx.draw(graph, node_color='k')
plt.savefig('draw.png', bbox_inches='tight')

plt.figure()
nx.draw_circular(graph, node_color='k')
plt.savefig('draw_circular.png', bbox_inches='tight')

plt.figure()
nx.draw_kamada_kawai(graph, node_color='k')
plt.savefig('kamada_kawai.png', bbox_inches='tight')

plt.figure()
nx.draw_spectral(graph, node_color='k')
plt.savefig('spectral.png', bbox_inches='tight')

plt.figure()
nx.draw_spring(graph, node_color='k')
plt.savefig('spring.png', bbox_inches='tight')

plt.figure()
nx.draw_shell(graph, node_color='k')
plt.savefig('shell.png', bbox_inches='tight')
