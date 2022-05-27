import numpy as np
import networkx as nx
from ACupdate import initial
from ACupdate import caculateRoute
from ACupdate import del_expiredPIT
from ACupdate import isoverload
from ACupdate import insertPIT
from ACupdate import sendData
from ACupdate import receiveData
from ACupdate import caculate_overhead
from ACupdate import caculate_delay
from ACupdate import caculate_load
from ACupdate import plot_result

def Dijk(graph, Consumer, Producer):
    # 全局化生存时间、最大负载、consumer、producer、拓扑图参数
    global TTL, load_max, consumer, producer
    TTL = 200
    load_max = 200
    consumer = Consumer
    producer = Producer
    G = graph
    caculateRoute(G, consumer, producer)
    load_ratios = []
    aver_delays = []
    overheads =[]
    # N_packet_lists = [10+10*k for k in range(iter_times)]
    N_packet = 500
    path = nx.shortest_path(G, consumer, producer)
    for i in range(1, N_packet):
        time = i/100  # 当前时间
        interest = 'HIT/' + str(i)  # 随机生成兴趣包
        G.nodes[consumer]['PIT'][interest] = {}
        now_node = consumer
        for k in range(len(path)-1):
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
            next_node = path [k+1]
            insertPIT(G, interest, time, now_node, next_node)
            G.nodes[now_node]['Packet'].append((interest, time))
            now_node = next_node
            time = G.nodes[next_node]['PIT'][interest]['forward_time']
        sendData(G, interest, time, path)
        time = i/100
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

    # 作图
    #plot_result(times, load_ratios, aver_delays, 'BC')
    print('Dj:', overheads)
    return load_ratios,aver_delays,overheads


edges = [('A','B'),('A','C'),('B','D'),('B','E'),('C','E'),('C','F'),('D','G'),('E','G'),('F','G')]
G = initial(edges)
Dijk(G, 'B','F')