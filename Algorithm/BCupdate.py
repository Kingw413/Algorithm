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


def BC(graph, Consumer, Producer):
    # 全局化生存时间、最大负载、consumer、producer、拓扑图参数
    global TTL, load_max, consumer, producer
    TTL = 200
    load_max = 200
    consumer = Consumer
    producer = Producer
    G = graph
    # 初始化拓扑图中各个节点的表项
    G = initial(edges)
    # 计算全局路由,从而为每个节点生成FIB表
    caculateRoute(G, consumer, producer)
    load_ratios = []
    aver_delays = []
    overheads = []
    # N_packet_lists = [10+10*k for k in range(iter_times)]
    N_packet = 500
    times = [x for x in range(int(N_packet/100))]
    #for N_packet in N_packet_lists:
    for i in range(1, N_packet):
        time = i/100  # 当前时间
        interest = 'HIT/' + str(i)  # 随机生成兴趣包
        G.nodes[consumer]['PIT'][interest] = {}
        for node in G.nodes:
            receiveData(G, node, time, consumer)
        # 找出所有的简单路径，再分别操作
        for path in nx.all_simple_paths(G, consumer, producer):
            now_time = time
            # 对路径上的每个节点进行更新
            now_node = consumer
            for k in range(1, len(path)):
                next_node = path[k]
                for node in G.nodes:
                    receiveData(G, node, now_time, consumer)
                    # 删除超时的PIT表项
                    if node != consumer:
                        del_expiredPIT(G, node, now_time, TTL)
                        # 判断PIT是否超载，如果是做出相应操作
                        isoverload(G, node, load_max)
                # 添加新的PIT表项
                insertPIT(G, interest, now_time, now_node, next_node)
                G.nodes[now_node]['Packet'].append((interest, now_time))
                now_node = next_node
                now_time = G.nodes[next_node]['PIT'][interest]['forward_time']
            sendData(G, interest, now_time, path)
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

    # print(G.nodes[consumer]['PIT'])
    # 作图
    #plot_result(times, load_ratios, aver_delays, 'BC')
    # print(G.nodes['B']['Num'])
    print('BC: ',overheads)
    return load_ratios,aver_delays, overheads

edges = [('A','B'),('A','C'),('B','D'),('B','E'),('C','E'),('C','F'),('D','G'),('E','G'),('F','G')]
overhead_BC = BC(edges, 'B','F')