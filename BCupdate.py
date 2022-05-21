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
from ACupdate import plot_result


def BC(edges, Consumer, Producer):
    # 全局化生存时间、最大负载、consumer、producer、拓扑图参数
    global TTL, overhead_max, consumer, producer, G
    TTL = 20
    overhead_max = 200
    consumer = Consumer
    producer = Producer
    # 初始化拓扑图中各个节点的表项
    G = initial(edges, consumer)
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
        # 计算平均负载
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
overhead_BC = BC(edges, 'B','F')