import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def ACO(edges, Consumer, Producer, iter_times):
    # 全局化生存时间、最大负载、consumer、producer、拓扑图参数
    global TTL, overhead_max, consumer, producer, G
    TTL = 100
    overhead_max = 200
    consumer = Consumer
    producer = Producer
    # 初始化拓扑图中各个节点的表项
    G = initial(edges)
    # 计算全局路由,从而为每个节点生成FIB表
    caculateRoute(G, consumer, producer)
    overhead_ratios = []
    N_packet_lists = [10+10*k for k in range(iter_times)] # 发送的包的总数
    for N_packet in N_packet_lists:
        for i in range(N_packet):
            time = i            # 当前时间
            node = consumer     # 当前节点
            path = [consumer]   # 已经走过的节点路径
            interest = 'HIT/'+str(np.random.randint(1000))  # 随机生成兴趣包
            while node != producer:
                # 贪心地选出下一跳
                next_node, path = choose_next(G, node, path)
                # 删除超时的PIT表项
                del_expiredPIT(G, node, time, TTL)
                # 判断PIT是否超载，如果是做出相应操作
                isoverload(G, node, overhead_max)
                # 添加新的PIT表项
                insertPIT(G, interest, time, next_node)
                # 更新节点的FIB表项
                updateFIB(G, node, next_node)
                node = next_node
                time = G.nodes[next_node]['PIT'][interest]
        # 计算平均负载
        overhead_ratio = caculate_overhead(G, consumer, producer, overhead_max)
        overhead_ratios.append(overhead_ratio)
    # 作图
    plot_result(N_packet_lists, overhead_ratios, 'AC')
    return overhead_ratios


# 函数：初始化各个节点的CS、PIT、FIB表
def initial(edges):
    G = nx.Graph()
    # 生成网络图
    G.add_edges_from(edges)
    for node in G.nodes:
        G.nodes[node]['CS'] = []
        G.nodes[node]['PIT'] = {}
        G.nodes[node]['FIB'] = {}
    return G


# 函数：找到所有从consumer(B)到producer(F)的简单路径，从而构建各个节点的初始FIB表
def caculateRoute(G, consumer, producer):
    for path in nx.all_simple_paths(G, consumer, producer):
        for k in range(len(path)-1):
            node = path[k] # 当前节点
            # 如果此路径的下一个节点不在当前节点的FIB表中
            if path[k+1] not in G.nodes[node]['FIB']:
                # 将其添加到当前节点FIB表中，并初始化此链路信息素为1
                G.nodes[node]['FIB'][path[k+1]] = 1


# 函数：选择下一个转发节点
def choose_next(G, node, path):
    node_lists = G.nodes[node]['FIB']
    # 找到该请求已经转发过的上游节点，避免环路
    passed_nodes = [x for x in node_lists.keys() if x in path]
    [node_lists.pop(x) for x in passed_nodes]
    if 'F' in node_lists:
        next_node = 'F'
    else:
        next_node = max(node_lists, key=lambda k: node_lists[k])
    path.append(next_node)
    return next_node, path

# 函数：检查并删除超时的PIT请求
def del_expiredPIT(G, node, time, TTL):
    timeout_interest = []
    for key in G.nodes[node]['PIT'].keys():
        if G.nodes[node]['PIT'][key] + TTL < time:
            timeout_interest.append(key)
    for out_interest in timeout_interest:
        G.nodes[node]['PIT'].pop(out_interest)


# 函数：检查是否达到PIT最大负载，如果是则删除最旧的请求
def isoverload(G, node, overhead_max):
    if node != 'F' and len(G.nodes[node]['PIT']) >= overhead_max:
        # 找到PIT表中存在时间最旧的请求，将其删除
        oldest_interest = min(G.nodes[node]['PIT'], key=lambda k: G.nodes[node]['PIT'][k])
        G.nodes[node]['PIT'].pop(oldest_interest)


# 函数：更新PIT表项
def insertPIT(G, interest, time, node):
    # 首先查询是否已经有相同请求
    if interest in G.nodes[node]['PIT']:
        # 判断相同请求的时间先后，更新请求的最新时间
        if G.nodes[node]['PIT'][interest] < time+0.5:
            G.nodes[node]['PIT'][interest] = time + 0.5
    # 如果没有相同请求，则添加新的PIT表项
    else:
        G.nodes[node]['PIT'][interest] = time + 0.5


# 函数：更新FIB表项
def updateFIB(G, node, next_node):
    G.nodes[node]['FIB'][next_node] = 1 - len(G.nodes[next_node]['PIT']) / overhead_max


# 函数：计算平均负载
def caculate_overhead(G, consumer, producer, overhead_max):
    overhead = []
    for node in G.nodes:
        # 排除掉consumer与producer，只考虑有负载的节点
        if (node!=consumer and node!=producer) and len(G.nodes[node]['PIT']):
            overhead.append(len(G.nodes[node]['PIT']))
    overhead_ratio = sum(overhead)/(overhead_max*len(overhead))
    return overhead_ratio


# 函数：作图
def plot_result(N_packet_lists, overhead_ratios, filename):
    plt.plot(N_packet_lists, overhead_ratios, 'bo-')
    plt.xlabel('Number of Interests')
    plt.ylabel('overhead_ratio')
    plt.title(filename)
    plt.show()
