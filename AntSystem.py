import numpy as np
import random
import matplotlib.pyplot as plt

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


# 函数：计算各个路径的总长度
def caculateLength(path):
    Distances = getDistMat(cities)
    length = Distances[path[-1], path[0]]  # 终点至起点的距离
    for i in range(N_city - 1):
        length += Distances[path[i], path[i + 1]]
    return length


# 函数：分配蚂蚁
def distriAnt(N_ant, N_city):
    if N_ant <= N_city:
        passing = np.random.permutation(range(N_city))[:N_ant]
    else:
        passing = np.random.permutation(range(N_city))
        passing = np.append(passing, np.random.permutation(range(N_city))[:N_ant - N_city])
    return passing


def cacuprob(passing, unpassed):
    prob_mat = np.zeros(len(unpassed))
    for k in range(len(unpassed)):
        prob_mat[k] = np.power(pheromone_tab[passing][unpassed[k]], alpha)\
                      * np.power(expect_tab[passing][unpassed[k]], beta)
    return prob_mat/sum(prob_mat)

def choonext(prob,unpassed):
    cumsumprob = prob.cumsum()
    cumsumprob -= np.random.rand()
    next_city = np.array(unpassed)[np.where(cumsumprob > 0)][0]
    return next_city

def updatepho(path_tab):
    #更新信息素
    updatephero_tab = np.zeros([N_city, N_city])
    for ant_cities in path_tab:
        for k in range(N_city-1):
            updatephero_tab[ant_cities[k]][ant_cities[k + 1]] += Q / DistMat[ant_cities[k]][ant_cities[k + 1]]
        updatephero_tab[ant_cities[k + 1]][ant_cities[0]] += Q / DistMat[ant_cities[k + 1]][ant_cities[0]]
    #信息素更新
    global pheromone_tab
    pheromone_tab =  (1-rho) * pheromone_tab + updatephero_tab

def solve():
    iter_times = 0
    global shortestLength
    global shortestPath
    shortestLength = np.zeros(N_iter)  # 初始化每次迭代的最短路径长度
    shortestPath = np.zeros(N_city).astype(int)  # 初始化每次迭代的最短路径
    while iter_times<N_iter:
        length = np.zeros(N_ant)
        path_tab = np.zeros([N_ant, N_city]).astype(int)
        path_tab[:, 0] = distriAnt(N_ant, N_city)
        for i in range(N_ant):
            passing = path_tab[i][0]  # 当前所在的城市
            unpassed = list(range(N_city))  # 还未经过的城市
            unpassed.remove(passing)
            for j in range(1, N_city):
                prob = cacuprob(passing, unpassed)
                next_city = choonext(prob, unpassed)
                path_tab[i][j] = next_city
                unpassed.remove(next_city)
                length[i] += DistMat[passing][next_city]
                passing = next_city
            length[i] += DistMat[passing][path_tab[i][0]]
        updatepho(path_tab)
        shortest(iter_times, length, path_tab)
        # shortestLength[iter_times] = length.min()
        # #global shortestPath
        # shortestPath = path_tab[length.argmin()].copy()
        iter_times += 1
    return shortestPath, shortestLength

def shortest(iter_times,length,path_tab):
    global shortestPath
    if iter_times == 0:
        shortestLength[iter_times] = length.min()
        shortestPath = path_tab[length.argmin()].copy()
    else:
        if length.min() < shortestLength[iter_times-1]:
            shortestLength[iter_times] = length.min()
            shortestPath = path_tab[length.argmin()].copy()
        else:
            shortestLength[iter_times] = shortestLength[iter_times-1]

def plotresult():
    shortestPath, shortestLength = solve()
    print('shortest length: ', shortestLength[-1])
    print('final path：', shortestPath)
    # 作图
    # 图1：随着迭代次数最短路径长度的变化
    plt.plot(shortestLength)
    plt.xlabel('iter_times')
    plt.ylabel('shortest Length')
    plt.title('shortest length: %f' % shortestLength[-1])

    # 图2：迭代完成后的最短路径
    plt.figure(2)
    # 依次作出路径上相邻两点
    for i in range(N_city - 1):
        plt.plot([cities[shortestPath[i], 0], cities[shortestPath[i + 1], 0]],
                 [cities[shortestPath[i], 1], cities[shortestPath[i + 1], 1]], 'bo-')
        #标注城市序号
        plt.text(cities[shortestPath[i], 0] + 60, cities[shortestPath[i], 1] - 50,
                 shortestPath[i], ha='center', va='bottom', fontsize=10)

    # 从终点回到起点
    plt.plot([cities[shortestPath[-1], 0], cities[shortestPath[0], 0]],
             [cities[shortestPath[-1], 1], cities[shortestPath[0], 1]], 'r')
    plt.text(cities[shortestPath[-1], 0] + 60, cities[shortestPath[-1], 1] - 50,
             shortestPath[-1], ha='center', va='bottom', fontsize=10)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('shortest path')
    plt.show()



cities = np.array([[1304, 2312], [3639, 1315], [4177, 2244], [3712, 1399], [3488, 1535], [3326, 1556],
                   [3238, 1229], [4196, 1044], [4312, 790], [4386, 570], [3007, 1970], [2562, 1756],
                   [2788, 1491], [2381, 1676], [1332, 695], [3715, 1678], [3918, 2179], [4061, 2370],
                   [3780, 2212], [3676, 2578], [4029, 2838], [4263, 2931], [3429, 1908], [3507, 2376],
                   [3394, 2643], [3439, 3201], [2935, 3240], [3140, 3550], [2545, 2357], [2778, 2826],
                   [2370, 2975]])
N_ant = 40 #蚂蚁个数
N_city = cities.shape[0] #城市个数
alpha = 1 #信息素重要程度
beta = 5 #启发因子重要程度
rho = 0.5 #信息素的挥发速度
Q = 1 #完成率

iter_times = 0 #迭代初始
N_iter = 100 #迭代次数
DistMat = getDistMat(cities)
pheromone_tab = np.ones([N_city, N_city])
expect_tab = 1/(DistMat + np.diag([1e10]*N_city))
plotresult()


