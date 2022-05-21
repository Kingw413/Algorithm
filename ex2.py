import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def ACO(graph, Consumer, Producer):
    # 全局化生存时间、最大负载、consumer、producer、拓扑图参数
    global TTL, overhead_max, consumer, producer, G, alpha, beta, gamma
    TTL = 20
    overhead_max = 200
    consumer = Consumer
    producer = Producer
    alpha = 0.7
    beta = 0.2
    gamma = 0.1
    G = graph
    # 初始化拓扑图中各个节点的表项
    # G = initial(edges,consumer)
    # 计算全局路由,从而为每个节点生成FIB表
    caculateRoute(G, consumer, producer)
    overhead_ratios = []
    aver_delays=[]
    #N_packet_lists=[4]
    #N_packet_lists = [10+10*k for k in range(iter_times)] # 发送的包的总数
    #for N_packet in N_packet_lists:
    N_packet = 2000
    times = [x for x in range(int(N_packet/100))]
    for i in range(N_packet):
        time = i / 100  # 当前时间
        now_node = consumer  # 当前节点
        path = [consumer]  # 已经走过的节点路径
        interest = 'HIT/' + str(i)  # 随机生成兴趣包
        G.nodes[consumer]['Interest'][interest] = [time]
        while now_node != producer:
            # 检查每个节点的PIT表，删除已经接收到返回Data的请求
            for node in G.nodes:
                # 去除掉已经收到返回的Data的请求
                receiveData(G, node, time)
                # 删除超时的请求
                del_expiredPIT(G, node, time, TTL)
                # 判断PIT是否超载，如果是作出相应操作
                isoverload(G, node, overhead_max)
                # 对当前节点的FIB表进行更新
                for adj_node in G.nodes[now_node]['FIB']:
                    updateFIB(G, now_node, adj_node)

            # 贪心地选出下一跳
            next_node, path = choose_next(G, now_node, path)
            # 删除超时的PIT表项
            #del_expiredPIT(G, next_node, time, TTL)
            # 判断PIT是否超载，如果是做出相应操作
            #isoverload(G, next_node, overhead_max)
            # 添加新的PIT表项
            insertPIT(G, interest, time, now_node, next_node)
            # 更新节点的FIB表项
            # updateFIB(G, now_node, next_node)
            now_node = next_node
            time = G.nodes[next_node]['PIT'][interest][0]
        sendData(G, interest, time, path, consumer)
        if i % 100 == 0:
            # 计算平均负载
            overhead_ratio = caculate_overhead(G, consumer, producer, overhead_max)
            overhead_ratios.append(overhead_ratio)
            aver_delay = caculate_delay(G, consumer)
            aver_delays.append(aver_delay)


    # 作图
    plot_result(times, overhead_ratios, aver_delays, 'AC')
    return overhead_ratios,aver_delays


# 函数：初始化各个节点的CS、PIT、FIB表
def initial(edges,consumer):
    G = nx.Graph()
    # 生成网络图
    G.add_edges_from(edges)
    for node in G.nodes:
        if node == consumer:
            G.nodes[node]['PIT']={}
            G.nodes[node]['FIB']={}
            G.nodes[node]['Interest']={}
        else:
            G.nodes[node]['CS'] = []
            G.nodes[node]['PIT'] = {}
            G.nodes[node]['FIB'] = {}
    # 随机生成各节点间链路的时延、丢失率
    for edge in G.edges:
        G.edges[edge]['load'] = 0
        G.edges[edge]['delay'] = 0.01
        # G.edges[edge]['delay'] = np.random.uniform(0.001, 0.01)
        G.edges[edge]['loss'] = np.random.uniform(0.1, 0.5)
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
        if G.nodes[node]['PIT'][key][0] + TTL < time:
            timeout_interest.append(key)
    for out_interest in timeout_interest:
        G.nodes[node]['PIT'].pop(out_interest)


# 函数：检查是否达到PIT最大负载，如果是则删除最旧的请求
def isoverload(G, node, overhead_max):
    if node != 'F' and len(G.nodes[node]['PIT']) >= overhead_max:
        # 找到PIT表中存在时间最旧的请求，将其删除
        oldest_interest = min(G.nodes[node]['PIT'], key=lambda k: G.nodes[node]['PIT'][k][0])
        G.nodes[node]['PIT'].pop(oldest_interest)


# 函数：删除收到返回Data的相应PIT请求
def receiveData(G, node, time):
    received = []
    for interest, times in G.nodes[node]['PIT'].items():
        if len(times)==2:
                if time >= times[1]:
                    received.append(interest)
    for received_interest in received:
        G.nodes[node]['PIT'].pop(received_interest)


# 函数：更新PIT表项
def insertPIT(G, interest, time, now_node, next_node):
    # 首先查询是否已经有相同请求
    delay = G.edges[now_node,next_node]['delay']
    if interest in G.nodes[next_node]['PIT']:
        # 判断相同请求的时间先后，更新请求的最新时间
        if G.nodes[next_node]['PIT'][interest][0] > time + delay:
            G.nodes[next_node]['PIT'][interest][0] = time + delay
    # 如果没有相同请求，则添加新的PIT表项
    else:
        G.nodes[next_node]['PIT'][interest] = [time + delay]


# 函数：更新FIB表项
def updateFIB(G, node, next_node):
    total_load = 0.001
    total_delay = 0
    total_loss = 0
    Len = len(G.nodes[node]['FIB'])
    for all_node in G.nodes[node]['FIB'].keys():
        G.edges[node,all_node]['load'] = len(G.nodes[all_node]['PIT'])/overhead_max
        total_load += G.edges[node,all_node]['load']
        total_delay += G.edges[node,all_node]['delay']
        total_loss += G.edges[node,all_node]['loss']

    edge = G.edges[node,next_node]
    G.nodes[node]['FIB'][next_node] = (alpha*(1-edge['load']/total_load) + beta*(1-edge['delay']/total_delay)
                                      + gamma*(1-edge['loss']/total_loss))/Len
    # G.nodes[node]['FIB'][next_node] = 1 - len(G.nodes[next_node]['PIT']) / overhead_max


# 函数：producer沿着原路返回Data
def sendData(G, interest, time, path, consumer):
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
        if node == consumer:
            if len(G.nodes[node]['Interest'][interest]) == 2:
                if G.nodes[node]['Interest'][interest][1] > receive_time:
                    G.nodes[node]['Interest'][interest][1] = receive_time
            else:
                G.nodes[node]['Interest'][interest].append(receive_time)
        else:
            if interest in G.nodes[node]['PIT'].keys():
                # 如果节点的PIT表中已经记录了先前相同Data的返回时间，则比较取最小时间
                if len(G.nodes[node]['PIT'][interest]) == 2:
                    if G.nodes[node]['PIT'][interest][1] > receive_time:
                        G.nodes[node]['PIT'][interest][1] = receive_time
                # 如果之前没有返回Data,则记录返回时间
                else:
                    G.nodes[node]['PIT'][interest].append(receive_time)


# 函数：计算平均负载
def caculate_overhead(G, consumer, producer, overhead_max):
    overhead = []
    for node in G.nodes:
        # 排除掉consumer与producer，只考虑有负载的节点
        if (node!=consumer and node!=producer) and len(G.nodes[node]['PIT']):
            overhead.append(len(G.nodes[node]['PIT']))
    overhead_ratio = sum(overhead)/(overhead_max*len(overhead))
    return overhead_ratio


# 函数：计算平均时延
def caculate_delay(G, consumer):
    round_time = []
    for name in G.nodes[consumer]['Interest'].keys():
        trip_time = G.nodes[consumer]['Interest'][name][1] - G.nodes[consumer]['Interest'][name][0]
        round_time.append(trip_time)
    aver_delay = sum(round_time) / len(round_time)
    return aver_delay

# 函数：作图
def plot_result(times, overhead_ratios, aver_delays, filename):
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    plt.plot(times, overhead_ratios, 'bo-')
    # xticks = np.arange(0,20,1)
    # plt.xticks(xticks)
    plt.xlabel('time/(s)')
    plt.ylabel('overhead_ratio')
    plt.title(filename)

    plt.subplot(2,1,2)
    plt.plot(times, aver_delays, 'rs-')
    plt.xlabel('time/(s)')
    plt.ylabel('aver_delays')
    plt.title(filename)
    plt.show()


def BC(graph, Consumer, Producer):
    # 全局化生存时间、最大负载、consumer、producer、拓扑图参数
    global TTL, overhead_max, consumer, producer, G
    TTL = 20
    overhead_max = 200
    consumer = Consumer
    producer = Producer
    G = graph
    # 初始化拓扑图中各个节点的表项
    # G = initial(edges, consumer)
    # 计算全局路由,从而为每个节点生成FIB表
    caculateRoute(G, consumer, producer)
    overhead_ratios = []
    aver_delays = []
    # N_packet_lists = [10+10*k for k in range(iter_times)]
    N_packet = 2000
    times = [x for x in range(int(N_packet/100))]
    #for N_packet in N_packet_lists:
    for i in range(N_packet):
        time = i/100  # 当前时间
        interest = 'HIT/' + str(i)  # 随机生成兴趣包
        G.nodes[consumer]['Interest'][interest] = [time]
        for node in G.nodes:
            receiveData(G, node, time)
        # 找出所有的简单路径，再分别操作
        for path in nx.all_simple_paths(G, consumer, producer):
            now_time = time
            # 对路径上的每个节点进行更新
            now_node = consumer
            for k in range(1, len(path)):
                next_node = path[k]
                for node in G.nodes:
                    receiveData(G, node, now_time)
                    # 删除超时的PIT表项
                    del_expiredPIT(G, node, now_time, TTL)
                    # 判断PIT是否超载，如果是做出相应操作
                    isoverload(G, node, overhead_max)
                # 添加新的PIT表项
                insertPIT(G, interest, now_time, now_node, next_node)
                now_node = next_node
                now_time = G.nodes[next_node]['PIT'][interest][0]
            sendData(G, interest, now_time, path, consumer)
        if i % 100 == 0:
            # 计算平均负载
            overhead_ratio = caculate_overhead(G, consumer, producer, overhead_max)
            overhead_ratios.append(overhead_ratio)
            aver_delay = caculate_delay(G, consumer)
            aver_delays.append(aver_delay)

    # 作图
    plot_result(times, overhead_ratios, aver_delays, 'BC')
    return overhead_ratios,aver_delays


edges = [('A','B'),('A','C'),('B','D'),('B','E'),('C','E'),('C','F'),('D','G'),('E','G'),('F','G')]
G = initial(edges, 'B')
G1 = G.copy()
overhead_AC, delay_AC= ACO(G, 'B', 'F')
#print(overhead_AC)
overhead_BC, delay_BC = BC(G1, 'B','F')
x = [x for x in range(20)]
# x = [10+10*k for k in range(15)]
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(x,overhead_AC,'ro-')
plt.plot(x,overhead_BC,'gs-')
plt.xlabel('time(s)')
plt.ylabel('overhead_ratio')
plt.title('AC VS BC')
plt.legend(['AC','BC'])

plt.subplot(2,1,2)
plt.plot(x,delay_AC,'ro-')
plt.plot(x,delay_BC,'gs-')
plt.xlabel('time(s)')
plt.ylabel('aver_delay')
plt.title('AC VS BC')
plt.legend(['AC','BC'])
plt.show()