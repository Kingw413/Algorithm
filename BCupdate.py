import numpy as np
import networkx as nx
from ACupdate import initial
from ACupdate import caculateRoute
from ACupdate import caculate_overhead
from ACupdate import del_expiredPIT
from ACupdate import isoverload
from ACupdate import insertPIT
from ACupdate import plot_result

def BC(edges, Consumer, Producer, iter_times):
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
    N_packet_lists = [10+10*k for k in range(iter_times)]
    for N_packet in N_packet_lists:
        for i in range(N_packet):
            time = i            # 当前时间
            interest = 'HIT/' + str(np.random.randint(1000)) # 随机生成兴趣包
            # 找出所有的简单路径，再分别操作
            for path in nx.all_simple_paths(G, consumer, producer):
                # 对路径上的每个节点进行更新
                for k in range(1,len(path)-1):
                    next_node = path[k]
                    # 删除超时的PIT表项
                    del_expiredPIT(G, next_node, time, TTL)
                    # 判断PIT是否超载，如果是做出相应操作
                    isoverload(G, next_node, overhead_max)
                    # 添加新的PIT表项
                    insertPIT(G, interest, time, next_node)
                    time = G.nodes[next_node]['PIT'][interest]
        # 计算平均负载
        overhead_ratio = caculate_overhead(G, consumer, producer, overhead_max)
        overhead_ratios.append(overhead_ratio)
    # 作图
    plot_result(N_packet_lists, overhead_ratios, 'BC')
    return overhead_ratios
