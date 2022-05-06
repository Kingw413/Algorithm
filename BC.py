import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

G = nx.Graph()
edges = [('A','B',{'tau':1}),('A','C',{'tau':1}),('B','D',{'tau':1}),('B','E',{'tau':1}),('C','E',{'tau':1}),
         ('C','F',{'tau':1}),('D','G',{'tau':1}),('E','G',{'tau':1}),('F','G',{'tau':1})]
G.add_edges_from(edges)

for node in G.nodes:
    G.nodes[node]['CS'] = []
    G.nodes[node]['PIT'] = {}
    G.nodes[node]['FIB'] = {}

for path in nx.all_simple_paths(G,'B','F'):
    #print(path)
    for k in range(len(path)-1):
        node = path[k]
        if path[k+1] not in G.nodes[node]['FIB']:
            G.nodes[node]['FIB'][path[k+1]]=1

TTL = 100
overhead_max = 200
path_tab = []
def BC(N_packet,G):
    for i in range(N_packet):
        time = i
        node = 'B'
        interest = 'HIT/' + str(np.random.randint(1000))
        for path in nx.all_simple_paths(G, 'B', 'F'):
            for k in range(1,len(path)-1):
                next_node = path[k]

                # 检查并删除超时的PIT请求
                timeout_interest = []
                for key in G.nodes[next_node]['PIT'].keys():
                    if G.nodes[next_node]['PIT'][key] < time+0.5*k:
                        G.nodes[next_node]['PIT'][key] = time+0.5*k
                    if G.nodes[next_node]['PIT'][key] + TTL < i:
                        timeout_interest.append(key)
                for out_interest in timeout_interest:
                    G.nodes[next_node]['PIT'].pop(out_interest)

                # 检查是否达到PIT最大负载，如果是则删除最旧的请求
                if len(G.nodes[next_node]['PIT']) >= overhead_max:
                    oldest_interest = min(G.nodes[next_node]['PIT'], key=lambda k: G.nodes[next_node]['PIT'][k])
                    G.nodes[next_node]['PIT'].pop(oldest_interest)

                # 添加新的请求
                G.nodes[next_node]['PIT'][interest] = time + 0.5*k

    overhead = []
    for node in G.nodes:
        if len(G.nodes[node]['PIT']):
            overhead.append(len(G.nodes[node]['PIT']))
    return sum(overhead)/(overhead_max*len(overhead))

overhead_ratio_BC = []

for i in range(30):
    overhead_ratio_BC.append(BC(30+5*i,G))
print(overhead_ratio_BC)

plt.plot([30+5*i for i in range(30)],overhead_ratio_BC)
plt.show()

# nx.draw(G, pos=nx.kamada_kawai_layout(G), with_labels=True)
# plt.show()
