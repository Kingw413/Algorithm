import numpy as np
import networkx as nx
from ACupdate import initial
from ACupdate import caculateRoute
from ACupdate import caculate_overhead
from ACupdate import del_expiredPIT
from ACupdate import isoverload
from ACupdate import insertPIT
from ACupdate import plot_result
from ACupdate import receiveData
from ACupdate import sendData

def BC(edges, Consumer, Producer):
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
    # N_packet_lists = [10+10*k for k in range(iter_times)]
    N_packet = 200
    times = [x for x in range(int(N_packet/10))]
    #for N_packet in N_packet_lists:
    for i in range(N_packet):
        time = i/10  # 当前时间
        interest = 'HIT/' + str(np.random.randint(1000))  # 随机生成兴趣包
        for node in G.nodes:
            receiveData(G, node, time)
        # 找出所有的简单路径，再分别操作
        for path in nx.all_simple_paths(G, consumer, producer):
            # 对路径上的每个节点进行更新
            for k in range(1, len(path)):
                next_node = path[k]
                # 删除超时的PIT表项
                del_expiredPIT(G, next_node, time, TTL)
                # 判断PIT是否超载，如果是做出相应操作
                isoverload(G, next_node, overhead_max)
                # 添加新的PIT表项
                insertPIT(G, interest, time, next_node)
                time = G.nodes[next_node]['PIT'][interest][0]
            sendData(G, interest, time, path, consumer)
        # 计算平均负载
        if i % 10 == 0:
            # 计算平均负载
            overhead_ratio = caculate_overhead(G, consumer, producer, overhead_max)
            overhead_ratios.append(overhead_ratio)

    # 作图
    plot_result(times, overhead_ratios, 'BC')
    return overhead_ratios

# edges = [('A','B'),('A','C'),('B','D'),('B','E'),('C','E'),('C','F'),('D','G'),('E','G'),('F','G')]
# overhead_BC = BC(edges, 'B','F')