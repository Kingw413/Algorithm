from ACupdate import ACO
from BCupdate import BC
from Djupdate import Dijk
import networkx as nx
import numpy as np

delays = np.random.uniform(0.1, 0.5, 9)
edges = [('A','B'),('A','C'),('B','D'),('B','E'),('C','E'),('C','F'),('D','G'),('E','G'),('F','G')]
def initial(edges):
    G = nx.Graph()
    # 生成网络图
    G.add_edges_from(edges)
    for node in G.nodes:
        G.nodes[node]['CS'] = []
        G.nodes[node]['PIT'] = {}
        G.nodes[node]['FIB'] = {}
        G.nodes[node]['Packet'] = []
    # 随机生成各节点间链路的时延、丢失率
    i = 0
    for edge in G.edges:
        G.edges[edge]['load'] = 0
        G.edges[edge]['delay'] = delays[i]
        G.edges[edge]['loss'] = round(np.random.uniform(0.1, 0.5),3)
        i += 1
    return G
G = initial(edges)
G1= initial(edges)
G2 = initial(edges)

delay_BC = BC(G, 'B','F')
delay_Dj = Dijk(G1, 'B','F')
delay_AC = ACO(G2, 'B', 'F')



# x = [x for x in range(20)]
#load_Dj, delay_Dj, overhead_Dj = Dijk(G2, 'B','F')
# with open('load.csv','w') as file:
#     for i in range(len(load_BC)):
#         file.write(str(load_BC[i],))
#         file.write('\t')
#         file.write(str(load_Dj[i]))
#         file.write('\t')
#         file.write(str(load_AC[i]))
#         file.write('\n')
#
with open('delay.csv','w',encoding='utf-8-sig') as file:
    file.write('时间\t多播\t最短路径优先\t基于蚁群的自适应\n')
    for i in range(len(delay_AC)):
        file.write(str(i/10))
        file.write('\t')
        file.write(str(delay_BC[i]))
        file.write('\t')
        file.write(str(delay_Dj[i]))
        file.write('\t')
        file.write(str(delay_AC[i]))
        file.write('\n')

#
# with open('overhead.csv','w') as file:
#     for i in range(len(overhead_AC)):
#         file.write(str(overhead_BC[i]))
#         file.write('\t')
#         file.write(str(overhead_Dj[i]))
#         file.write('\t')
#         file.write(str(overhead_AC[i]))
#         file.write('\n')