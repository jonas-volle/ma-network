import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
# import community as louvain
from itertools import cycle

# Definitions

def color_cycle():
    """
        This is a cycle of colors.
    """
    colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6']
    return cycle(colors)
colors = color_cycle()

def to_community_list(node_community_dict):
    """
        This transforms {0: 0, 1: 1:, 2: 0, 3: 1} into [[0, 2], [1, 3]].
    """
    partition_dict = {}
    for v, k in zip(node_community_dict.keys(), node_community_dict.values()):
        if k not in partition_dict:
            partition_dict[k] = []
        partition_dict[k].append(v)
    return list(partition_dict.values())

def lcc(G):
    return max(nx.connected_component_subgraphs(G, copy=True), key=len)

# Import Data

authorships = pd.read_csv("data/scp_network_data.csv")

authorships['publication_id'] = authorships['publication_id'].astype('str') # werden in strings verwandelt, damit id 0 und id= 0 nicht der gleiche Punkt ist
authorships.head()


A = nx.Graph(name='Authorships')

A.add_edges_from(authorships.values)

C = bipartite.weighted_projected_graph(A, set(authorships['author_id'])) # Gewichtung für alle Author ids

C_lcc = C.subgraph(max(nx.connected_components(C), key=len))

pos_C_lcc = nx.spring_layout(C_lcc, k=0.5, seed=12345, iterations= 10000)

plt.figure(figsize=(6,6))
nx.draw(C_lcc, pos= pos_C_lcc)
plt.show()

nx.number_of_nodes(C_lcc), nx.number_of_edges(C_lcc)

# Größe der Punkte nach der degree-centrality
C_lcc_degree_centrality = list(nx.degree_centrality(C_lcc).values())
C_lcc_degree_centrality = [2000/max(C_lcc_degree_centrality)*x for x in C_lcc_degree_centrality]
# A_lcc_degree_centrality[:5]

plt.figure(figsize=(12, 12))
nx.draw(C_lcc, pos=pos_C_lcc, width=0.1, alpha=0.3, node_size=C_lcc_degree_centrality)
plt.show()

nodes_info = pd.read_csv("data/author_label.csv")

author = nodes_info[nodes_info['author_id'].isin(set(C_lcc.nodes))][['author_id', 'author']]
author.head()

author_dict = author.set_index('author_id')['author'].to_dict()

plt.figure(figsize=(12, 12))
nx.draw(C_lcc, pos=pos_C_lcc, width=0.5, alpha=0.3,
        node_size=C_lcc_degree_centrality, with_labels = True,
        labels = author_dict, font_size=6)
plt.savefig("plots/scp_authorships.pdf")
plt.show()