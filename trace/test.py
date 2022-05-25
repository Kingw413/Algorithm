def overhead_compute(filename):
    import pandas as pd
    import re
    # 打开文件
    f = open('./data/Cbr/'+ filename, "r")
    overhead_file = []
    # 初始化开销为字典，格式为{时间：平均开销}
    line = f.readline()
    # 依次读取log文件
    while line:
        if "onIncomingInterest()" in line:
            # 得到对应的时间，格式为字符串
            time = line.split()[0].split('s')[0]
            # 将时间向下取整，即每隔一秒统计一次
            overhead_file.append(int(float(time)))
        # 下一次循环
        line = f.readline()
    f.close()
    # 利用pandas的库函数求出每一秒内的总开销
    packet_counts = pd.value_counts(overhead_file)

    overhead = {x+1:0 for x in range(len(packet_counts))}
    # 对于第i秒计算前i秒的平均开销
    for i in range(1,len(packet_counts)+1):
        # 对前i秒的开销值进行累加求平均
        for j in range(i):
            overhead[i] += packet_counts[j]*8*1024/(i+1)/1000
    # 创建文件存储输出结果
    result_filename = './results/Cbr-'+ filename.split(".log")[0] + "_overhead.txt"
    result_file = open(result_filename, "w")
    # 将结果数据写入文件
    for key, value in sorted(overhead.items()):
        result_file.write(str(key) +' '+str(value)+'\n')
    result_file.close()


import os
import pandas as pd
import re
All_zipf_overhead = pd.DataFrame()
All_zipf_overhead['time'] = list(range(1,31))
for filename in os.listdir(r'./data/Cbr/'):
    overhead_compute(filename)

# import matplotlib.pyplot as plt
# import numpy as np
# best_tree_overhead = np.loadtxt('bestroute-tree_overhead.txt')
# multi_tree_overhead = np.loadtxt('multicast-tree_overhead.txt')
# best_grid_overhead = np.loadtxt('bestroute-grid_overhead.txt')
# multi_grid_overhead = np.loadtxt('multicast-grid_overhead.txt')
# print(best_grid_overhead)
# print(multi_grid_overhead)
#
# plt.figure(figsize=(12,10))
# plt.subplot(2,1,1)
# plt.plot(best_tree_overhead[:,0],best_tree_overhead[:,1],'ks-')
# plt.plot(multi_tree_overhead[:,0],multi_tree_overhead[:,1],'r*-')
# plt.legend(['BestRoute','Multicast'])
# plt.xlabel('time(s)')
# plt.ylabel('Overhead(kbits)')
# plt.title('Overhead(Tree)')
#
# plt.subplot(2,1,2)
# plt.plot(best_grid_overhead[:,0],best_grid_overhead[:,1],'ks-')
# plt.plot(multi_grid_overhead[:,0],multi_grid_overhead[:,1],'r*-')
# plt.legend(['BestRoute','Multicast'])
# plt.xlabel('time(s)')
# plt.ylabel('Overhead(kbits)')
# plt.title('Overhead(Grid)')
# plt.show()