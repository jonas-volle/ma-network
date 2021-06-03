import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as louvain

from itertools import cycle
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

# Import Dataset

df = pd.read_csv("data/co-autorship-AS-DGS.csv")
df['pdf_id'] = df['pdf_id'].astype('str')

df_n = df[["pdf_id", "id"]]
df_n['pdf_id'] = df['pdf_id'].astype('str')

A = nx.Graph(name='Authorships')
A.add_edges_from(df_n.values)


from networkx.algorithms import bipartite
C = bipartite.weighted_projected_graph(A, set(df_n['id']))
C_lcc = lcc(C)

pos = nx.spring_layout(C_lcc, seed=10, iterations=50000)
pos_kk = nx.kamada_kawai_layout(C_lcc)

communities_louvain = louvain.best_partition(C_lcc, resolution=100)
communities_louvain = to_community_list(communities_louvain)

C_lcc_degree_centrality = list(nx.degree_centrality(C_lcc).values())
C_lcc_degree_centrality = [100/max(C_lcc_degree_centrality)*x for x in C_lcc_degree_centrality]

plt.figure(figsize=(12, 12))
for nodes in communities_louvain:
    nx.draw_networkx_nodes(C_lcc, nodelist=nodes, pos=pos, node_size=C_lcc_degree_centrality,
                           node_color=next(colors))
nx.draw_networkx_edges(C_lcc, pos=pos, width=0.1, alpha=0.2)

csw_names = pd.read_csv("data/author_label.csv")

author = nodes_info[nodes_info['author_id'].isin(set(A_lcc_sup_lcc.nodes))][['author_id', 'author']]
author.head()

author_dict = author.set_index('author_id')['author'].to_dict()

nx.draw(A_lcc_sup_lcc, pos=pos_A_lcc_sup_lcc, width=0.5, alpha=0.3, node_color = AS_DGS_color,
        node_size=A_lcc_sup_lcc_degree_centrality, with_labels= True, labels = author_dict, font_size=7)


plt.savefig("graph.pdf")