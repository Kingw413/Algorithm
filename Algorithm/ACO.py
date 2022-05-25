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
def ACO(N_packet):
    for i in range(N_packet):
        time = i
        node = 'B'
        path = ['B']
        interest = 'HIT/'+str(np.random.randint(1000))
        G.nodes[node]['PIT'][interest] = time
        while node != 'F':
            node_lists = G.nodes[node]['FIB']
            passed_nodes = [x for x in node_lists.keys() if x in path]
            [node_lists.pop(x) for x in passed_nodes]
            if 'F' in node_lists:
                next_node = 'F'
            else:
                next_node = max(node_lists, key=lambda k: node_lists[k])
            path.append(next_node)

            # 检查并删除超时的PIT请求
            timeout_interest = []
            for key in G.nodes[next_node]['PIT'].keys():
                if G.nodes[next_node]['PIT'][key]+TTL<i:
                    timeout_interest.append(key)
            for out_interest in timeout_interest:
                G.nodes[next_node]['PIT'].pop(out_interest)

            # 检查是否达到PIT最大负载，如果是则删除最旧的请求
            if next_node!='F' and len(G.nodes[next_node]['PIT'])>=overhead_max:
                oldest_interest = min(G.nodes[next_node]['PIT'], key=lambda k: G.nodes[next_node]['PIT'][k])
                G.nodes[next_node]['PIT'].pop(oldest_interest)

            # 添加新的请求
            G.nodes[next_node]['PIT'][interest] = G.nodes[node]['PIT'][interest] + 0.5

            G.nodes[node]['FIB'][next_node] = 1-len(G.nodes[next_node]['PIT'])/overhead_max
            node = next_node
        path_tab.append(path)
        #print(path)
    overhead = []
    for node in G.nodes:
        if (node!='B' and node!='F') and len(G.nodes[node]['PIT']):
            overhead.append(len(G.nodes[node]['PIT']))
    return sum(overhead)/(overhead_max*len(overhead))

overhead_ratio=[]
overhead_ratio_BC = []

for i in range(15):
    overhead_ratio.append(ACO(10+10*i))
    #overhead_ratio_BC.append(BC(30+5*i,G1))
print(overhead_ratio)
#print(overhead_ratio_BC)

plt.plot([10+10*i for i in range(15)],overhead_ratio)
#plt.plot([30+5*i for i in range(30)],overhead_ratio_BC)
plt.show()
