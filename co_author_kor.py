import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
import community as louvain
from itertools import cycle
import community as louvain
from community import modularity

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

df = pd.read_csv("data/co_author_edges_26_09_19.csv")


# Graph
A = nx.Graph(name='Authorships')
A.add_edges_from(df[["author_1_id", "author_2_id"]].values)


pos_authorship = nx.spring_layout(A, k=0.5)

plt.figure(figsize=(6, 6))
nx.draw(A, pos=pos_authorship)
plt.show()

# Size of the Graph
nx.number_of_nodes(A), nx.number_of_edges(A)

# Is the Graph connected?
nx.is_connected(A)

# How many connected components?
nx.number_connected_components(A)

# average degree
(2*nx.number_of_edges(A))/nx.number_of_nodes(A)

# density
nx.number_of_edges(A)/(nx.number_of_nodes(A)*(nx.number_of_nodes(A)-1)/2)

# cluster coefficent
nx.average_clustering(A)

# the largest connected component
A_lcc = lcc(A)
nx.is_connected(A_lcc)
nx.number_of_nodes(A_lcc), nx.number_of_edges(A_lcc)

# average degree
(2*nx.number_of_edges(A_lcc))/nx.number_of_nodes(A_lcc)

# density
nx.number_of_edges(A_lcc)/(nx.number_of_nodes(A_lcc)*(nx.number_of_nodes(A_lcc)-1)/2)

# cluster coefficent
nx.average_clustering(A_lcc)

# Degree distribution
A_lcc_degrees = [len(list(A_lcc.neighbors(n))) for n in A_lcc.nodes()]

plt.figure()
plt.hist(A_lcc_degrees, bins= 100, color="black")
plt.show()

A_lcc_degrees.plot()
plt.show()

# degree centrality distribution
# Compute the degree centrality of the Twitter network: deg_cent
A_lcc_deg_cent = nx.degree_centrality(A_lcc)

# Plot a histogram of the degree centrality distribution of the graph.
plt.figure()
plt.hist(list(A_lcc_deg_cent.values()), bins = 20)
plt.show()



pos_A_lcc = nx.spring_layout(A_lcc, k=0.5, seed=12345, iterations= 2000)

plt.figure(figsize=(6,6))
nx.draw(A_lcc, pos= pos_A_lcc)
plt.show()

nx.number_of_nodes(A_lcc), nx.number_of_edges(A_lcc)

# Größe der Punkte nach der degree-centrality
A_lcc_degree_centrality = list(nx.degree_centrality(A_lcc).values())
A_lcc_degree_centrality = [2000/max(A_lcc_degree_centrality)*x for x in A_lcc_degree_centrality]
# A_lcc_degree_centrality[:5]

plt.figure(figsize=(12, 12))
nx.draw(A_lcc, pos=pos_A_lcc, width=0.1, alpha=0.3, node_size=A_lcc_degree_centrality)
plt.show()

# Farbe für AS und DGS

nodes_info = pd.read_csv("data/co_author_nodes_26_09_19.csv")

AS_DGS = nodes_info[nodes_info['author_id'].isin(set(A_lcc.nodes))][['author_id', 'as.dgs']]
AS_DGS.head()

AS_DGS_dict = AS_DGS.set_index('author_id')['as.dgs'].to_dict()

for key, value in AS_DGS_dict.items():
    if value == 'AS':
        A_lcc.nodes[key]['color'] = 'blue'
    if value == 'DGS':
        A_lcc.nodes[key]['color'] = 'orange'
    if value == 'keine_Mitgliedschaft':
        A_lcc.nodes[key]['color'] = 'grey'

AS_DGS_color = [k['color'] for (u, k) in A_lcc.nodes(data=True)]

plt.figure(figsize=(12, 12))
nx.draw(A_lcc, pos=pos_A_lcc, width=0.1, alpha=0.3, node_color = AS_DGS_color,
        node_size=A_lcc_degree_centrality)
plt.savefig("co-authorship_as_dgs.pdf")
plt.show()

nx.attribute_assortativity_coefficient(A_lcc, attribute='color')

# Names
author_names = nodes_info[nodes_info['id'].isin(set(A_lcc.nodes))][['id', 'author']]
author_names_dict = author_names.set_index('id')['author'].to_dict()


# Centrale Author*innen

# Degree centrality
deg_cen = nx.degree_centrality(A_lcc)
sorted_deg_cen = sorted(deg_cen.items(), key=lambda kv: kv[1])
sorted_deg_cen.reverse()
print(sorted_deg_cen)

plt.figure()
plt.hist(list(deg_cen.values()))
plt.show()


# Betweenness centrality
bet_cen = nx.betweenness_centrality(A_lcc)
sorted_bet_cen = sorted(bet_cen.items(), key=lambda kv: kv[1])
sorted_bet_cen.reverse()
print(sorted_bet_cen)


# Closeness centrality
clo_cen = nx.closeness_centrality(A_lcc)


# Eigenvector centrality
eig_cen = nx.eigenvector_centrality(A_lcc)


sorted_bet_cen = sorted(bet_cen.items(), key=lambda kv: kv[1])
sorted_bet_cen.reverse()
print(sorted_bet_cen)

# Community detection: Louvain method

A_lcc_communities_louvain = louvain.best_partition(A_lcc)
A_lcc_communities_louvain = to_community_list(A_lcc_communities_louvain)

plt.figure(figsize=(12, 12))
for nodes in A_lcc_communities_louvain:
    nx.draw_networkx_nodes(A_lcc, nodelist=nodes, pos=pos_A_lcc,
                           node_size=A_lcc_degree_centrality, node_color=next(colors), alpha = 0.5)
nx.draw_networkx_edges(A_lcc, pos_A_lcc, width=0.7, alpha=0.2)
nx.draw_networkx_labels(A_lcc_sup_lcc, pos_A_lcc_sup_lcc, labels=author_dict, font_size=5, alpha=0.5)
plt.axis('off')
plt.savefig("co-authorship_only_as_dgs_community.pdf")
plt.show()

nodes_info = nodes_info.set_index(nodes_info.author_id, drop = True)
nodes_info.head()

for i in range(len(A_lcc_communities_louvain)):
    nodes_info.reindex(A_lcc_communities_louvain[i])

nodes_info.reindex(A_lcc_communities_louvain[1]).groupby('as.dgs').count()


# ---------------------- Subgraph nur AS und DGS --------------------------------------

nodes = [n for n, d in A_lcc.nodes(data=True) if d['color'] != 'grey']

# Create the set of nodes: nodeset
nodeset = set(nodes)

A_lcc_sup = A_lcc.subgraph(nodeset)

# Size of the Graph
nx.number_of_nodes(A_lcc_sup), nx.number_of_edges(A_lcc_sup)

# Is the Graph connected?
nx.is_connected(A_lcc_sup)

# How many connected components?
nx.number_connected_components(A_lcc_sup)

# the largest connected component
A_lcc_sup_lcc = lcc(A_lcc_sup)
nx.is_connected(A_lcc_sup_lcc)
nx.number_of_nodes(A_lcc_sup_lcc), nx.number_of_edges(A_lcc_sup_lcc)

# average degree
(2*nx.number_of_edges(A_lcc_sup_lcc))/nx.number_of_nodes(A_lcc_sup_lcc)

# density
nx.number_of_edges(A_lcc_sup_lcc)/(nx.number_of_nodes(A_lcc_sup_lcc)*(nx.number_of_nodes(A_lcc_sup_lcc)-1)/2)

# cluster coefficent
nx.average_clustering(A_lcc_sup_lcc)

# Degree distribution
A_lcc_degrees = [len(list(A_lcc.neighbors(n))) for n in A_lcc.nodes()]

plt.figure()
plt.hist(A_lcc_degrees, bins= 100, color="black")
plt.show()


# Centrale Author*innen

# Degree centrality
deg_cen = nx.degree_centrality(A_lcc_sup_lcc)
sorted_deg_cen = sorted(deg_cen.items(), key=lambda kv: kv[1])
sorted_deg_cen.reverse()
print(sorted_deg_cen)

plt.figure()
plt.hist(list(deg_cen.values()))
plt.show()


# Betweenness centrality
bet_cen = nx.betweenness_centrality(A_lcc_sup_lcc)
sorted_bet_cen = sorted(bet_cen.items(), key=lambda kv: kv[1])
sorted_bet_cen.reverse()
print(sorted_bet_cen)

# Edge Betweenness centrality
edge_bet_cen = nx.edge_betweenness_centrality(A_lcc_sup_lcc)
sorted_edge_bet_cen = sorted(edge_bet_cen.items(), key=lambda kv: kv[1])
sorted_edge_bet_cen.reverse()
print(sorted_edge_bet_cen)


# Closeness centrality
clo_cen = nx.closeness_centrality(A_lcc_sup_lcc)
sorted_clo_cen = sorted(clo_cen.items(), key=lambda kv: kv[1])
sorted_clo_cen.reverse()
print(sorted_clo_cen)


# Eigenvector centrality
eig_cen = nx.eigenvector_centrality(A_lcc_sup_lcc, max_iter=500)
sorted_eig_cen = sorted(eig_cen.items(), key=lambda kv: kv[1])
sorted_eig_cen.reverse()
print(sorted_eig_cen)




pos_A_lcc_sup_lcc = nx.spring_layout(A_lcc_sup_lcc, k=0.5, seed=12345, iterations= 2000)

# Größe der Punkte nach der degree-centrality
A_lcc_sup_lcc_degree_centrality = list(nx.degree_centrality(A_lcc_sup_lcc).values())
A_lcc_sup_lcc_degree_centrality = [2000/max(A_lcc_sup_lcc_degree_centrality)*x for x in A_lcc_sup_lcc_degree_centrality]


AS_DGS = nodes_info[nodes_info['author_id'].isin(set(A_lcc_sup_lcc.nodes))][['author_id', 'as.dgs']]
AS_DGS.head()

AS_DGS_dict = AS_DGS.set_index('author_id')['as.dgs'].to_dict()


for key, value in AS_DGS_dict.items():
    if value == 'AS':
        A_lcc_sup_lcc.nodes[key]['color'] = 'blue'
    if value == 'DGS':
        A_lcc_sup_lcc.nodes[key]['color'] = 'orange'
    if value == 'keine_Mitgliedschaft':
        A_lcc_sup_lcc.nodes[key]['color'] = 'grey'

AS_DGS_color = [k['color'] for (u, k) in A_lcc_sup_lcc.nodes(data=True)]

author = nodes_info[nodes_info['author_id'].isin(set(A_lcc_sup_lcc.nodes))][['author_id', 'author']]
author.head()

author_dict = author.set_index('author_id')['author'].to_dict()

plt.figure(figsize=(12, 12))
nx.draw(A_lcc_sup_lcc, pos=pos_A_lcc_sup_lcc, width=0.5, alpha=0.3, node_color = AS_DGS_color,
        node_size=A_lcc_sup_lcc_degree_centrality, with_labels= True, labels = author_dict, font_size=7)
plt.savefig("co-authorship_only_as_dgs.pdf")
plt.show()

# Modularity AS and DGS Communities

nodes_as_dgs = [[n for n, d in A_lcc_sup_lcc.nodes(data=True) if d['color'] == 'blue'],[n for n, d in A_lcc_sup_lcc.nodes(data=True) if d['color'] == 'orange']]

nx.algorithms.community.quality.modularity(A_lcc_sup_lcc, nodes_as_dgs)

# Community detection: Louvain method

communities_louvain_sup = louvain.best_partition(A_lcc_sup_lcc, random_state = 1234)
communities_louvain_sup = to_community_list(communities_louvain_sup)

plt.figure(figsize=(12, 12))
for nodes in communities_louvain_sup:
    nx.draw_networkx_nodes(A_lcc_sup_lcc, nodelist=nodes, pos=pos_A_lcc_sup_lcc,
                           node_size=300, node_color=next(colors), alpha = 0.5)
nx.draw_networkx_edges(A_lcc_sup_lcc, pos_A_lcc_sup_lcc, width=0.7, alpha=0.2)
nx.draw_networkx_labels(A_lcc_sup_lcc, pos_A_lcc_sup_lcc, labels=author_dict, font_size=5, alpha=0.5)
plt.axis('off')
plt.savefig("co-authorship_only_as_dgs_community.pdf")
plt.show()

nodes_info = nodes_info.set_index(nodes_info.author_id, drop = True)
nodes_info.head()

for i in range(len(communities_louvain_sup)):
    nodes_info.reindex(communities_louvain_sup[i])

nodes_info.reindex(communities_louvain_sup[8]).groupby('as.dgs').count()

df_com = pd.DataFrame(communities_louvain_sup).transpose()

df_com.to_csv("df_com.csv")

nx.algorithms.community.quality.modularity(A_lcc_sup_lcc, communities_louvain_sup)

len(communities_louvain_sup)

# Communities
communities_louvain_sup[0]

# density
nx.number_of_edges(A_lcc_sup_lcc.subgraph(communities_louvain_sup[0]))/\
(nx.number_of_nodes(A_lcc_sup_lcc.subgraph(communities_louvain_sup[0]))*
 (nx.number_of_nodes(A_lcc_sup_lcc.subgraph(communities_louvain_sup[0]))-1)/2)

# degree centrality
# Degree centrality
sorted(nx.degree_centrality(A_lcc_sup_lcc.subgraph(communities_louvain_sup[8])).items(),
             key=lambda kv: kv[1])

# average shortest path length
nx.average_shortest_path_length(A_lcc_sup_lcc.subgraph(communities_louvain_sup[8]))

#neighbors
[n for n in A_lcc_sup_lcc.neighbors(1012)]



# Netzwerk nur innerhalb der AS

nodes_AS = [n for n, d in A_lcc.nodes(data=True) if d['color'] == 'blue']


# Create the set of nodes: nodeset
nodeset_AS = set(nodes_AS)

AS_sup = A_lcc.subgraph(nodeset_AS)

# Size of the Graph
nx.number_of_nodes(AS_sup), nx.number_of_edges(AS_sup)

# density
nx.number_of_edges(AS_sup)/(nx.number_of_nodes(AS_sup)*(nx.number_of_nodes(AS_sup)-1)/2)


# Is the Graph connected?
nx.is_connected(AS_sup)

# How many connected components?
nx.number_connected_components(AS_sup)

# the largest connected component
AS_sup_lcc = lcc(AS_sup)
nx.is_connected(AS_sup_lcc)
nx.number_of_nodes(AS_sup_lcc), nx.number_of_edges(AS_sup_lcc)

# average degree
(2*nx.number_of_edges(AS_sup_lcc))/nx.number_of_nodes(AS_sup_lcc)

# density
nx.number_of_edges(AS_sup_lcc)/(nx.number_of_nodes(AS_sup_lcc)*(nx.number_of_nodes(AS_sup_lcc)-1)/2)

# clustering coefficent
nx.average_clustering(AS_sup_lcc)

pos_AS_sup_lcc = nx.spring_layout(AS_sup_lcc, k=0.5, seed=12345, iterations= 2000)

# Größe der Punkte nach der degree-centrality
AS_sup_lcc_degree_centrality = list(nx.degree_centrality(AS_sup_lcc).values())
AS_sup_lcc_degree_centrality = [2000/max(AS_sup_lcc_degree_centrality)*x for x in AS_sup_lcc_degree_centrality]
# A_lcc_degree_centrality[:5]

AS_DGS = nodes_info[nodes_info['author_id'].isin(set(AS_sup_lcc.nodes))][['author_id', 'as.dgs']]
AS_DGS.head()

AS_DGS_dict = AS_DGS.set_index('author_id')['as.dgs'].to_dict()


for key, value in AS_DGS_dict.items():
    if value == 'AS':
        AS_sup_lcc.nodes[key]['color'] = 'blue'
    if value == 'DGS':
        AS_sup_lcc.nodes[key]['color'] = 'orange'
    if value == 'keine_Mitgliedschaft':
        AS_sup_lcc.nodes[key]['color'] = 'grey'

AS_DGS_color = [k['color'] for (u, k) in AS_sup_lcc.nodes(data=True)]

author_AS = nodes_info[nodes_info['author_id'].isin(set(AS_sup_lcc.nodes))][['author_id', 'author']]
author.head()

author_AS_dict = author_AS.set_index('author_id')['author'].to_dict()

plt.figure(figsize=(12, 12))
nx.draw(AS_sup_lcc, pos=pos_AS_sup_lcc, width=0.5, alpha=0.3, node_color = AS_DGS_color,
        node_size=AS_sup_lcc_degree_centrality, with_labels= True, labels = author_AS_dict, font_size=7)
plt.savefig("co-authorship_only_as.pdf")
plt.show()


# Netzwerk nur innerhalb der DGS

nodes_DGS = [n for n, d in A_lcc.nodes(data=True) if d['color'] == 'orange']

# Create the set of nodes: nodeset
nodeset_DGS = set(nodes_DGS)

DGS_sup = A_lcc.subgraph(nodeset_DGS)

# Size of the Graph
nx.number_of_nodes(DGS_sup), nx.number_of_edges(DGS_sup)

# density
nx.number_of_edges(DGS_sup)/(nx.number_of_nodes(DGS_sup)*(nx.number_of_nodes(DGS_sup)-1)/2)


# Is the Graph connected?
nx.is_connected(DGS_sup)

# How many connected components?
nx.number_connected_components(DGS_sup)

# the largest connected component
DGS_sup_lcc = lcc(DGS_sup)
nx.is_connected(DGS_sup_lcc)
nx.number_of_nodes(DGS_sup_lcc), nx.number_of_edges(DGS_sup_lcc)

# average degree
(2*nx.number_of_edges(DGS_sup_lcc))/nx.number_of_nodes(DGS_sup_lcc)

# density
nx.number_of_edges(DGS_sup_lcc)/(nx.number_of_nodes(DGS_sup_lcc)*(nx.number_of_nodes(DGS_sup_lcc)-1)/2)

# clustering coefficent
nx.average_clustering(DGS_sup_lcc)

pos_DGS_sup_lcc = nx.spring_layout(DGS_sup_lcc, k=0.5, seed=12345, iterations= 2000)

# Größe der Punkte nach der degree-centrality
DGS_sup_lcc_degree_centrality = list(nx.degree_centrality(DGS_sup_lcc).values())
DGS_sup_lcc_degree_centrality = [2000/max(DGS_sup_lcc_degree_centrality)*x for x in DGS_sup_lcc_degree_centrality]
# A_lcc_degree_centrality[:5]

AS_DGS = nodes_info[nodes_info['author_id'].isin(set(DGS_sup_lcc.nodes))][['author_id', 'as.dgs']]
AS_DGS.head()

AS_DGS_dict = AS_DGS.set_index('author_id')['as.dgs'].to_dict()


for key, value in AS_DGS_dict.items():
    if value == 'AS':
        DGS_sup_lcc.nodes[key]['color'] = 'blue'
    if value == 'DGS':
        DGS_sup_lcc.nodes[key]['color'] = 'orange'
    if value == 'keine_Mitgliedschaft':
        DGS_sup_lcc.nodes[key]['color'] = 'grey'

AS_DGS_color = [k['color'] for (u, k) in DGS_sup_lcc.nodes(data=True)]

author_DGS = nodes_info[nodes_info['author_id'].isin(set(DGS_sup_lcc.nodes))][['author_id', 'author']]
author.head()

author_DGS_dict = author_DGS.set_index('author_id')['author'].to_dict()

plt.figure(figsize=(12, 12))
nx.draw(DGS_sup_lcc, pos=pos_DGS_sup_lcc, width=0.5, alpha=0.3, node_color = AS_DGS_color,
        node_size=DGS_sup_lcc_degree_centrality, with_labels= True, labels = author_DGS_dict, font_size=7)
plt.savefig("co-authorship_only_dgs.pdf")
plt.show()
