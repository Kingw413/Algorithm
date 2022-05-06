import numpy as np
import matplotlib.pyplot as plt

#函数：初始化生成城市个数、距离矩阵、信息素矩阵、期望矩阵，并设为全局变量
def initial(cities):
    global N_city
    N_city = cities.shape[0]
    global DistMat, pheromone_tab, expect_tab
    DistMat = getDistMat(cities)
    pheromone_tab = np.ones([N_city, N_city])
    expect_tab = 1 / (DistMat + np.diag([1e10] * N_city))
    return N_city, DistMat, pheromone_tab, expect_tab


#函数：求解最终路径与每次迭代的最短路径长度
def ASsolve(cities,N_ant,N_iter, alpha,beta,rho,Q):
    iter_times = 0
    N_city, DistMat, pheromone_tab, expect_tab = initial(cities)
    global shortestLength
    global shortestPath
    shortestLength = np.zeros(N_iter)                     #初始化每次迭代的最短路径长度
    shortestPath = np.zeros(N_city).astype(int)           #初始化每次迭代的最短路径
    while iter_times < N_iter:
        length = np.zeros(N_ant)                          #初始化每只蚂蚁的路径长度
        path_tab = np.zeros([N_ant, N_city]).astype(int)  #初始化每次蚂蚁的路径表
        # 为每只蚂蚁随机选择起始城市
        path_tab[:, 0] = distriAnt(N_ant, N_city)
        for i in range(N_ant):
            passing = path_tab[i][0]                      #当前所在的城市
            unpassed = list(range(N_city))                #未访问城市表
            unpassed.remove(passing)                      #将已经经过的城市从未访问城市表中移除
            for j in range(1, N_city):
                #计算到未访问城市表中各个城市的概率
                prob = cacuprob(passing, unpassed,alpha,beta)
                #依据概率按轮盘法选择下一个城市
                next_city = choonext(prob, unpassed)
                path_tab[i][j] = next_city                #将下一步的城市添加到路径表中
                unpassed.remove(next_city)
                length[i] += DistMat[passing][next_city]  #计算对应蚂蚁的路径长度
                passing = next_city
            length[i] += DistMat[passing][path_tab[i][0]] #加上从终点到起点的距离
        #更新信息素矩阵
        updatepho(path_tab,rho,Q)
        #比较决定最短路径及其长度
        shortest(iter_times, length, path_tab)
        iter_times += 1
    #作图
    plotresult('AS',cities, shortestLength, shortestPath)
    return shortestLength, shortestPath


#函数：计算距离矩阵
def getDistMat(cities):
    # 城市个数
    num = cities.shape[0]
    # 初始化距离矩阵
    DistMat = np.zeros([num, num])
    # 循环计算城市间的欧氏距离
    for i in range(num):
        for j in range(i, num):
            DistMat[i][j] = DistMat[j][i] = np.linalg.norm(cities[i] - cities[j])
    return DistMat


# 函数：分配蚂蚁到初始起点城市
def distriAnt(N_ant, N_city):
    if N_ant <= N_city:
        passing = np.random.permutation(range(N_city))[:N_ant]
    else:
        passing = np.random.permutation(range(N_city))
        passing = np.append(passing, np.random.permutation(range(N_city))[:N_ant - N_city])
    return passing


#函数：计算到各个城市的概率
def cacuprob(passing, unpassed, alpha, beta):
    prob_mat = np.zeros(len(unpassed))
    for k in range(len(unpassed)):
        #概率计算函数
        prob_mat[k] = np.power(pheromone_tab[passing][unpassed[k]], alpha) \
                      * np.power(expect_tab[passing][unpassed[k]], beta)
    return prob_mat / sum(prob_mat)


#函数：依据累积概率，采用轮盘法选择下一个城市
def choonext(prob, unpassed):
    cumsumprob = prob.cumsum()
    cumsumprob -= np.random.rand()
    next_city = np.array(unpassed)[np.where(cumsumprob > 0)][0]
    return next_city


#函数：更新信息素矩阵
def updatepho(path_tab,rho,Q):
    # 更新信息素
    updatephero_tab = np.zeros([N_city, N_city])
    for ant_cities in path_tab:
        for k in range(N_city - 1):
            updatephero_tab[ant_cities[k]][ant_cities[k + 1]] += Q / DistMat[ant_cities[k]][ant_cities[k + 1]]
        updatephero_tab[ant_cities[k + 1]][ant_cities[0]] += Q / DistMat[ant_cities[k + 1]][ant_cities[0]]
    # 信息素更新
    global pheromone_tab
    pheromone_tab = (1 - rho) * pheromone_tab + updatephero_tab


#函数：和上一次迭代结果比较，决定最短路径
def shortest(iter_times, length, path_tab):
    global shortestPath
    #第一次迭代时的结果
    if iter_times == 0:
        shortestLength[iter_times] = length.min()
        shortestPath = path_tab[length.argmin()].copy()
    #和上一次迭代结果比较，如果更短则更新，否则不变
    else:
        if length.min() < shortestLength[iter_times - 1]:
            shortestLength[iter_times] = length.min()
            shortestPath = path_tab[length.argmin()].copy()
        else:
            shortestLength[iter_times] = shortestLength[iter_times - 1]


#函数：作出每次迭代的路径长度以及最终的路径
def plotresult(file_name, cities, shortestLength, shortestPath):
    #输出结果
    print('(%s)shortest length: ' % file_name, shortestLength[-1])
    print('(%s)final path：' % file_name, shortestPath)
    # 图1：随着迭代次数最短路径长度的变化
    plt.figure()
    plt.plot(shortestLength)
    plt.xlabel('iter_times')
    plt.ylabel('shortest Length')
    plt.title('(%s)shortest length: %f' % (file_name, shortestLength[-1]))
    #plt.show()

    # 图2：迭代完成后的最短路径
    plt.figure()
    # 依次作出路径上相邻两点
    for i in range(len(cities) - 1):
        plt.plot([cities[shortestPath[i], 0], cities[shortestPath[i + 1], 0]],
                 [cities[shortestPath[i], 1], cities[shortestPath[i + 1], 1]], 'bo-')
        # 标注城市序号
        plt.text(cities[shortestPath[i], 0] + 60, cities[shortestPath[i], 1] - 50,
                 shortestPath[i], ha='center', va='bottom', fontsize=10)

    # 从终点回到起点
    plt.plot([cities[shortestPath[-1], 0], cities[shortestPath[0], 0]],
             [cities[shortestPath[-1], 1], cities[shortestPath[0], 1]], 'r')
    plt.text(cities[shortestPath[-1], 0] + 60, cities[shortestPath[-1], 1] - 50,
             shortestPath[-1], ha='center', va='bottom', fontsize=10)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('(%s)shortest path' % file_name)
    plt.show()
