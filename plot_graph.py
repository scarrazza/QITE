import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# weighted maxcut
nqubits = 12
np.random.seed(0)
aa = np.random.randint(1, nqubits*(nqubits-1)/2+1)
graph = nx.random_graphs.dense_gnm_random_graph(nqubits, aa)

plt.figure()
nx.draw(graph, node_color='k')
plt.savefig('weighted_maxcut_draw.png', bbox_inches='tight')

plt.figure()
nx.draw_circular(graph, node_color='k')
plt.savefig('weighted_maxcut_draw_circular.png', bbox_inches='tight')

plt.figure()
nx.draw_kamada_kawai(graph, node_color='k')
plt.savefig('weighted_maxcut_kamada_kawai.png', bbox_inches='tight')

plt.figure()
nx.draw_spectral(graph, node_color='k')
plt.savefig('weighted_maxcut_spectral.png', bbox_inches='tight')

plt.figure()
nx.draw_spring(graph, node_color='k')
plt.savefig('weighted_maxcut_spring.png', bbox_inches='tight')

plt.figure()
nx.draw_shell(graph, node_color='k')
plt.savefig('weighted_maxcut_shell.png', bbox_inches='tight')
