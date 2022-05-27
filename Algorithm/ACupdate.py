import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random


def ACO(graph, Consumer, Producer):
    # 全局化生存时间、最大负载、consumer、producer、拓扑图参数
    global TTL, load_max, consumer, producer, alpha, beta
    TTL = 200
    load_max = 200
    consumer = Consumer
    producer = Producer
    alpha = 0.5
    beta = 0.5
    gamma = 0
    G = graph
    # 初始化拓扑图中各个节点的表项
    G = initial(edges)
    # 计算全局路由,从而为每个节点生成FIB表
    caculateRoute(G, consumer, producer)
    load_ratios = []
    aver_delays=[]
    overheads = []
    N_packet = 500
    times = [x for x in range(int(N_packet/100))]
    all_paths = set()
    for path in nx.all_simple_paths(G,'B','F'):
        all_paths.add(tuple(path))
    had_paths = set()
    for i in range(1, N_packet):
        time = i / 100  # 当前时间
        now_node = consumer  # 当前节点
        path = [consumer]  # 已经走过的节点路径
        interest = 'HIT/' + str(i)  # 随机生成兴趣包
        G.nodes[consumer]['PIT'][interest] = {}

        while now_node != producer:
            # 检查每个节点的PIT表，删除已经接收到返回Data的请求
            for node in G.nodes:
                # 去除掉已经收到返回的Data的请求
                receiveData(G, node, time, consumer)
                # 删除超时的请求
                if node != consumer and node != producer:
                    del_expiredPIT(G, node, time, TTL)
                # 判断PIT是否超载，如果是作出相应操作
                    isoverload(G, node, load_max)

            # 贪心地选出下一跳
            next_node, path = choose_next(G, now_node, path)
            insertPIT(G, interest, time, now_node, next_node)
            G.nodes[now_node]['Packet'].append((interest, time))
            now_node = next_node
            time = G.nodes[next_node]['PIT'][interest]['forward_time']
        sendData(G, interest, time, path)
        time = i/100
        had_paths.add(tuple(path))
        # over_head = caculate_overhead(G, consumer, producer, overhead, i/100)

        if had_paths == all_paths:
            hello_update(G, time)
        # if i> 0 and i%10 == 0:
        #     hello_update(G, time)

        if (time /0.1)%1 == 0:
            # 计算平均负载
            load_ratio = caculate_load(G, consumer, producer, load_max)
            load_ratios.append(load_ratio)
            aver_delay = caculate_delay(G, consumer)
            aver_delays.append(aver_delay)
            overhead = 0
            for node in G.nodes:
                for packet in G.nodes[node]['Packet']:
                    if packet[1] <= time:
                        overhead += 1
            aver_overhead = round(overhead*1024*8/time/1000,3)
            overheads.append(aver_overhead)
            #over_head = caculate_overhead(G,i/10)
            #overheads.append(over_head/(i/10))

    # 作图
    # plot_result(times, load_ratios, aver_delays, 'AC')
    print('AC: ', overheads)
    return load_ratios, aver_delays, overheads


# 函数：初始化各个节点的CS、PIT、FIB表
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
    for edge in G.edges:
        G.edges[edge]['load'] = 0
        G.edges[edge]['delay'] = round(np.random.uniform(0.1, 0.5),3)
        G.edges[edge]['loss'] = round(np.random.uniform(0.1, 0.5),3)
    return G


# 函数：找到所有从consumer(B)到producer(F)的简单路径，从而构建各个节点的初始FIB表
def caculateRoute(G, consumer, producer):
    for path in nx.all_simple_paths(G, consumer, producer):
        for k in range(len(path)-1):
            node = path[k] # 当前节点
            # 如果此路径的下一个节点不在当前节点的FIB表中
            if path[k+1] not in G.nodes[node]['FIB']:
                # 将其添加到当前节点FIB表中，并初始化此链路信息素为1
                G.nodes[node]['FIB'][path[k+1]]= {}
                G.nodes[node]['FIB'][path[k+1]]['load'] = 0
                G.nodes[node]['FIB'][path[k+1]]['tau'] = 1


# 函数：选择下一个转发节点
def choose_next(G, node, path):
    node_lists = G.nodes[node]['FIB']
    # 找到该请求已经转发过的上游节点，避免环路
    passed_nodes = [x for x in node_lists.keys() if x in path]
    [node_lists.pop(x) for x in passed_nodes]
    # 找到信息素值最大的下一跳接口
    choice = []
    for nodes in node_lists:
        choice.append(node_lists[nodes]['tau'])
    max_tau = max(choice)
    max_nodes = []
    for x in node_lists:
        if node_lists[x]['tau'] == max_tau:
            max_nodes.append(x)
    next_node = random.choice(max_nodes)
    path.append(next_node)
    return next_node, path


# 函数：检查并删除超时的PIT请求
def del_expiredPIT(G, node, time, TTL):
    timeout_interest = []
    for interest in G.nodes[node]['PIT'].keys():
        if 'forward_time' in G.nodes[node]['PIT']:
            if G.nodes[node]['PIT'][interest]['forward_time'] + TTL < time:
                timeout_interest.append(interest)
    for out_interest in timeout_interest:
        G.nodes[node]['PIT'].pop(out_interest)


# 函数：检查是否达到PIT最大负载，如果是则删除最旧的请求
def isoverload(G, node, load_max):
    if len(G.nodes[node]['PIT']) >= load_max:
        # 找到PIT表中存在时间最旧的请求，将其删除
        oldest_interest = min(G.nodes[node]['PIT'], key=lambda interest: G.nodes[node]['PIT'][interest]['forward_time'])
        G.nodes[node]['PIT'].pop(oldest_interest)


# 函数：删除收到返回Data的相应PIT请求
def receiveData(G, node, time, consumer, load_max=200):
    received = []
    for interest, face_time in G.nodes[node]['PIT'].items():
        if 'receive_time' in face_time:
                if time >= face_time['receive_time']:
                    received.append(interest)
    for received_interest in received:
        if 'outFace' in G.nodes[node]['PIT'][received_interest]:
            next_node = G.nodes[node]['PIT'][received_interest]['outFace']
            if node != consumer:
                G.nodes[node]['PIT'].pop(received_interest)
            updateFIB(G, node, next_node, load_max)


# 函数：更新PIT表项
def insertPIT(G, interest, time, now_node, next_node):
    # 首先查询是否已经有相同请求
    delay = G.edges[now_node,next_node]['delay']
    if interest in G.nodes[next_node]['PIT'].keys():
        # 判断相同请求的时间先后，更新请求的最新时间
        if G.nodes[next_node]['PIT'][interest]['forward_time'] > time + delay:
            G.nodes[next_node]['PIT'][interest]['forward_time'] = time + delay
    # 如果没有相同请求，则添加新的PIT表项
    else:
        G.nodes[now_node]['PIT'][interest]['outFace'] = next_node
        G.nodes[now_node]['PIT'][interest]['forward_time'] = time

        G.nodes[next_node]['PIT'][interest] = {}
        G.nodes[next_node]['PIT'][interest]['inFace'] = now_node
        G.nodes[next_node]['PIT'][interest]['forward_time'] = time + delay



# 函数：更新FIB表项
def updateFIB(G, node, next_node, load_max):
    total_load = 0.00001
    total_delay = 0
    total_loss = 0
    Len = len(G.nodes[node]['FIB'])

    for all_node in G.nodes[node]['FIB'].keys():
        G.nodes[node]['FIB'][all_node]['load'] = len(G.nodes[node]['PIT'])/load_max
        total_load += G.nodes[node]['FIB'][all_node]['load']
        # total_delay += G.edges[node,all_node]['delay']
        # total_loss += G.edges[node,all_node]['loss']

    delta_tau = 1-G.nodes[node]['FIB'][next_node]['load']/total_load
    G.nodes[node]['FIB'][next_node]['tau'] = (1-0.5)*G.nodes[node]['FIB'][next_node]['tau'] + 0.5*delta_tau

    for other_node in G.nodes[node]['FIB'].keys():
        if other_node != next_node:
            G.nodes[node]['FIB'][other_node]['tau'] = (1-0.5)*G.nodes[node]['FIB'][other_node]['tau']-0.5*delta_tau/(Len-1)


def hello_update(G, time):
    for node in G.nodes:
        total_load = 0.000001
        total_delay = 0.000001
        for next_node in G.nodes[node]['FIB']:
            total_load += G.nodes[node]['FIB'][next_node]['load']
            total_delay += caculate_delay(G, next_node)
        for next_node in G.nodes[node]['FIB']:
            load= G.nodes[node]['FIB'][next_node]['load']
            delay = caculate_delay(G, next_node)
            tau_new = alpha*(1-load/total_load) + beta*(1-delay/total_delay)
            if time<1:
                G.nodes[node]['FIB'][next_node]['tau'] = tau_new
            else:
                G.nodes[node]['FIB'][next_node]['tau'] = (1-0.5)*G.nodes[node]['FIB'][next_node]['tau'] + 0.5*tau_new


# 函数：producer沿着原路返回Data
def sendData(G, interest, time, path):
    path1 = path.copy()        # 复制路径，避免后续操作更改原path
    # path1.remove(consumer)     # 去掉consumer
    for node in path1:
        k = path1.index(node)  # 节点的索引值
        # 节点接收到返回Data的时间
        receive_time = time
        while k != len(path1)-1:
            receive_time += G.edges[path1[k],path1[k+1]]['delay']
            k += 1
        # receive_time = time + 0.5*(len(path1)-1-k)
        if 'receive_time' in G.nodes[node]['PIT'][interest]:
            if G.nodes[node]['PIT'][interest]['receive_time'] > receive_time:
                G.nodes[node]['PIT'][interest]['receive_time'] = receive_time
        else:
            G.nodes[node]['PIT'][interest]['receive_time'] = receive_time


# 函数：计算平均负载
def caculate_load(G, consumer, producer, load_max):
    load = []
    for node in G.nodes:
        # 排除掉consumer与producer，只考虑有负载的节点
        if (node!=consumer and node!=producer) and len(G.nodes[node]['PIT']):
            load.append(len(G.nodes[node]['PIT']))
    load_ratio = sum(load)/(load_max*len(load))
    return load_ratio


# 函数：计算平均时延
def caculate_delay(G, node):
    round_time = []
    for interest in G.nodes[node]['PIT'].keys():
        trip_time = G.nodes[node]['PIT'][interest]['receive_time'] - G.nodes[node]['PIT'][interest]['forward_time']
        round_time.append(trip_time)
    aver_delay = sum(round_time) / len(round_time)
    return aver_delay

# 函数：计算开销
def caculate_overhead(G,time):
    overhead = 0
    for node in G.nodes:
        for packet in  G.nodes[node]['Packet']:
            if packet[1]<=time:
                overhead += 1
    return overhead


# 函数：作图
def plot_result(times, overhead_ratios, aver_delays, filename):
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    plt.plot(times, overhead_ratios, 'bo-')
    plt.xlabel('time/(s)')
    plt.ylabel('overhead_ratio')
    plt.title(filename)

    plt.subplot(2,1,2)
    plt.plot(times, aver_delays, 'rs-')
    plt.xlabel('time/(s)')
    plt.ylabel('aver_delays')
    plt.title(filename)
    plt.show()


edges = [('A','B'),('A','C'),('B','D'),('B','E'),('C','E'),('C','F'),('D','G'),('E','G'),('F','G')]
overhead_AC = ACO(edges, 'B', 'F')