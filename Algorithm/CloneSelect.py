import numpy as np
import random
from AntSystem import getDistMat
from AntSystem import plotresult

#函数：求解最终路径与每次迭代的最短路径长度
def CSsolve(Cities,Num_p,Num_clone,N_iter):
    global cities,N_city,Np,N_clone
    cities = Cities
    N_city = cities.shape[0]          #城市数
    Np = Num_p                        #种群数
    N_clone = Num_clone               #克隆数
    #初始化各种群所选择的路径
    Paths, Lengths = initial(Np)
    #对各种群按照路径长度从低到高排序
    sortedPaths, sortedLengths = sortpath(Paths, Lengths)
    shortestPath = np.zeros(N_city).astype(int)          #初始化最短路径
    shortestLength = np.zeros(N_iter)                    #初始化每次迭代的最短路径
    iter_times = 0
    while iter_times < N_iter:
        #克隆变异后选择最优路径
        variaPaths, variaLengths = cloneselect(sortedPaths, sortedLengths)
        #随机生成新的种群
        newPaths, newLengths = initial(int(Np/2))
        # 克隆变异后选择的种群与随机生成的新种群合并
        finalPath = np.vstack([variaPaths, newPaths])              # 最终的各种群路径
        finalLengths = np.hstack([variaLengths, newLengths])  # 最终各种群路径的总长度
        sortedPaths, sortedLengths = sortpath( finalPath, finalLengths)
        shortestLength[iter_times] = sortedLengths[0]  # 每次迭代的最短路径长度
        shortestPath = sortedPaths[0]  # 每次迭代的最短路径
        iter_times += 1
    #作图
    plotresult('CS',cities,shortestLength,shortestPath)
    return shortestLength, shortestPath


# 函数：计算各个路径的总长度
def caculateLength(path):
    #距离矩阵
    Distances = getDistMat(cities)
    length = Distances[path[-1], path[0]]  # 终点至起点的距离
    for i in range(N_city - 1):
        length += Distances[path[i], path[i + 1]]
    return length


#函数：初始化各个种群的路径
def initial(Np):
    Paths = np.zeros([Np, N_city]).astype(int)  # 初始化各个种群的路径表
    Lengths = np.zeros(Np)  # 初始化各个种群所走路径的总长度
    for i in range(Np):
        Paths[i, :] = np.random.permutation(N_city).astype(int)  # 初始时随机生成各个种群的路径
        Lengths[i] = caculateLength(Paths[i, :])  # 计算各种群所选路径的长度
    return Paths, Lengths


#函数：对种群按照路径长度进行排序
def sortpath(Paths, Lengths):
    sortedLengths = np.sort(Lengths)  # 对路径总长度升序排列
    sortedIndex = np.argsort(Lengths)  # 获得排列的索引值
    sortedPaths = Paths[sortedIndex]  # 对各个种群的路径表按照路径长短升序排列
    return sortedPaths, sortedLengths


#函数：克隆变异后选择最优种群
def cloneselect(sortedPaths, sortedLengths):
    variaLengths = np.zeros(int(Np / 2))                      # 存放变异后的挑选出的种群的路径总长度
    variaPaths = np.zeros([int(Np / 2), N_city]).astype(int)  # 存放变异后挑选出的种群的路径
    # 对选出的优质种群进行克隆变异再选择
    for i in range(int(Np / 2)):
        selectedPath = sortedPaths[i, :]                  # 选出前Np/2个种群作为优质种群
        clonalPath = np.tile(selectedPath, [N_clone, 1])  # 对选出的各个种群克隆N_clone份
        clonalLength = np.zeros(N_clone)
        clonalLength[0] = sortedLengths[i]
        for j in range(1, N_clone):                       #保留源种群，对剩下的克隆种群进行变异
            # 变异操作，此处选择为换位
            idx = random.sample(range(N_city), 2)         #随机生成两个需要交换的城市的索引
            tmp = clonalPath[j, idx[0]]
            clonalPath[j, idx[0]] = clonalPath[j, idx[1]]
            clonalPath[j, idx[1]] = tmp
            # 计算变异后的种群路径的总长度
            clonalLength[j] = caculateLength(clonalPath[j])
        variaLengths[i] = np.min(clonalLength)               #变异后种群的最短路径长度
        variaPaths[i] = clonalPath[np.argmin(clonalLength)]  #变异后种群的最短路径
    return variaPaths, variaLengths