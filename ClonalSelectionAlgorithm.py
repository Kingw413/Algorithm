import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import ndarray


#函数：计算城市间的距离矩阵
def getDistMat(Cities):
    #城市个数
    num = Cities.shape[0]
    #初始化距离矩阵
    DistMat = np.zeros([num, num])
    #循环计算城市间的欧氏距离
    for i in range(num):
        for j in range(i,num):
            DistMat[i][j] = DistMat[j][i] = np.linalg.norm(Cities[i]-Cities[j])
    return DistMat


#函数：计算各个路径的总长度
def caculateLength(path):
    length = Distances[path[-1], path[0]]  #终点至起点的距离
    for i in range(N_city-1):
        length += Distances[path[i], path[i+1]]
    return length

# cities = np.array([[565.0, 575.0], [25.0, 185.0], [345.0, 750.0], [945.0, 685.0], [845.0, 655.0],
#                         [880.0, 660.0], [25.0, 230.0], [525.0, 1000.0], [580.0, 1175.0], [650.0, 1130.0],
#                         [1605.0, 620.0], [1220.0, 580.0], [1465.0, 200.0], [1530.0, 5.0], [845.0, 680.0],
#                         [725.0, 370.0], [145.0, 665.0], [415.0, 635.0], [510.0, 875.0], [560.0, 365.0],
#                         [300.0, 465.0], [520.0, 585.0], [480.0, 415.0], [835.0, 625.0], [975.0, 580.0],
#                         [1215.0, 245.0], [1320.0, 315.0], [1250.0, 400.0], [660.0, 180.0], [410.0, 250.0],
#                         [420.0, 555.0], [575.0, 665.0], [1150.0, 1160.0], [700.0, 580.0], [685.0, 595.0],
#                         [685.0, 610.0], [770.0, 610.0], [795.0, 645.0], [720.0, 635.0], [760.0, 650.0],
#                         [475.0, 960.0], [95.0, 260.0], [875.0, 920.0], [700.0, 500.0], [555.0, 815.0],
#                         [830.0, 485.0], [1170.0, 65.0], [830.0, 610.0], [605.0, 625.0], [595.0, 360.0]])

cities = np.array([[1304,2312],[3639,1315],[4177,2244],[3712,1399],[3488,1535],[3326,1556],
    [3238 ,1229],[4196 ,1044],[4312, 790],[4386 ,570],[3007 ,1970],[2562 ,1756],
    [2788,1491],[2381,1676],[1332,695],[3715, 1678],[3918 ,2179],[4061 ,2370],
    [3780 ,2212],[3676, 2578],[4029, 2838],[4263, 2931],[3429, 1908],[3507, 2376],
    [3394,2643],[3439, 3201],[2935, 3240],[3140 ,3550],[2545 ,2357],[2778, 2826],
    [2370, 2975]])

N_city = cities.shape[0] #城市个数
Np = 200  #种群个数
N_gen = 1000 #迭代次数
N_clone = 10  #克隆个数
iter_times = 0 #初始化迭代次数
Paths = np.zeros([Np, N_city]).astype(int)  #初始化各个种群的路径表
Distances = getDistMat(cities)  #距离矩阵
Lengths = np.zeros(Np)          #初始化各个种群所走路径的总长度
for i in range(Np):
    Paths[i,:] = np.random.permutation(N_city).astype(int)   #初始时随机生成各个种群的路径
    Lengths[i] = caculateLength(Paths[i,:])      #计算各种群所选路径的长度
sortedLengths = np.sort(Lengths)                 #对路径总长度升序排列
sortedIndex = np.argsort(Lengths)                #获得排列的索引值
sortedPaths = Paths[sortedIndex]                 #对各个种群的路径表按照路径长短升序排列
shortestLength = np.zeros(N_gen)            #存放每次迭代的最短路径长度
shortestPath = np.zeros(N_city).astype(int)
while iter_times<N_gen:
    variaLengths = np.zeros(int(Np / 2))   #存放变异后的挑选出的种群的路径总长度
    variaPaths = np.zeros([int(Np / 2), N_city]).astype(int)   #存放变异后挑选出的种群的路径
    #对选出的优质种群进行克隆变异再选择
    for i in range(int(Np/2)):
        selectedPath = sortedPaths[i,:]  #选出前Np/2个种群作为优质种群
        clonalPath = np.tile(selectedPath, [N_clone,1])   #对选出的各个种群克隆N_clone份
        clonalLength = np.zeros(N_clone)
        clonalLength[0] = sortedLengths[i]
        for j in range(1, N_clone): #保留源种群，对剩下的克隆种群进行变异
            #变异操作，此处选择为换位
            idx = random.sample(range(N_city), 2) #随机生成两个需要交换的城市的索引
            tmp = clonalPath[j, idx[0]]
            clonalPath[j, idx[0]] = clonalPath[j, idx[1]]
            clonalPath[j, idx[1]] = tmp
            clonalLength[j] = caculateLength(clonalPath[j])  #计算变异后的种群路径的总长度
        variaLengths[i] = np.min(clonalLength)                #变异后种群的最短路径长度
        variaPaths[i]  = clonalPath[np.argmin(clonalLength)]   #变异后种群的最短路径

    #随机生成剩下的新种群
    newPaths = np.zeros([Np-int(Np/2), N_city]).astype(int)
    newLengths = np.zeros(Np-int(Np/2))
    for i in range(Np-int(Np/2)):
        newPaths[i, :] = np.random.permutation(N_city).astype(int)  # 随机生成新种群的路径
        newLengths[i] = caculateLength(newPaths[i, :])  # 计算新种群所选路径的长度

    #克隆选择的种群与随机生成的新种群合并
    finalPath = np.vstack([variaPaths, newPaths])          #最终的各种群路径
    finalLengths = np.hstack([variaLengths, newLengths])   #最终各种群路径的总长度

    sortedLengths = np.sort(finalLengths)
    sortedIndex = np.argsort(finalLengths)
    sortedPaths = finalPath[sortedIndex]

    shortestLength[iter_times] = sortedLengths[0]          #每次迭代的最短路径长度
    shortestPath = sortedPaths[0]                          #每次迭代的最短路径

    iter_times +=1

#作图
fig = plt.figure(1)
#图1：随着迭代次数最短路径长度的变化
ax1 = plt.subplot(1,2,1)
plt.plot(shortestLength)
plt.xlabel('iter_times')
plt.ylabel('shortest Length')
plt.title('shortest length')

#图2：迭代完成后的最短路径
ax2 = plt.subplot(1,2,2)
#依次作出路径上相邻两点
for i in range(N_city-1):
    plt.plot([cities[shortestPath[i],0],cities[shortestPath[i+1],0]], [cities[shortestPath[i],1],cities[shortestPath[i+1],1]],'bo-')
#从终点回到起点
plt.plot([cities[shortestPath[-1],0],cities[shortestPath[0],0]], [cities[shortestPath[-1],1],cities[shortestPath[0],1]],'r')
plt.xlabel('x_axis')
plt.ylabel('y_axis')
plt.title('shortest path: ')
plt.show()
print('shortest length: ', shortestLength[-1])