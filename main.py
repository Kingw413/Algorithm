import numpy as np
import random
import matplotlib.pyplot as plt
import numba
from numba import jit

class clonselect:
    def __init__(self, cities: np.ndarray, Np: int, N_iter: int, N_clone: int):
        self.cities = cities
        self.N_city = self.cities.shape[0]
        self.Np = Np
        self.N_iter = N_iter
        self.N_clone = N_clone
        # self.shortestLength = np.zeros(self.N_iter)  # 存放每次迭代的最短路径长度
        # self.shortestPath = np.zeros(self.N_city).astype(int)

    def getDistMat(self):
        # 城市个数
        num = self.cities.shape[0]
        # 初始化距离矩阵
        DistMat = np.zeros([num, num])
        # 循环计算城市间的欧氏距离
        for i in range(num):
            for j in range(i, num):
                DistMat[i][j] = DistMat[j][i] = np.linalg.norm(self.cities[i] - self.cities[j])
        return DistMat

    # 函数：计算各个路径的总长度
    def caculateLength(self, path):
        Distances = clonselect.getDistMat(self)
        length = Distances[path[-1], path[0]]  # 终点至起点的距离
        for i in range(self.N_city - 1):
            length += Distances[path[i], path[i + 1]]
        return length

    def initial(self):
        Paths = np.zeros([self.Np, self.N_city]).astype(int)  # 初始化各个种群的路径表
        Lengths = np.zeros(self.Np)  # 初始化各个种群所走路径的总长度
        for i in range(self.Np):
            Paths[i, :] = np.random.permutation(self.N_city).astype(int)  # 初始时随机生成各个种群的路径
            Lengths[i] = clonselect.caculateLength(self, Paths[i, :])  # 计算各种群所选路径的长度
        return Paths, Lengths

    def sortpath(self, Paths, Lengths):
        sortedLengths = np.sort(Lengths)  # 对路径总长度升序排列
        sortedIndex = np.argsort(Lengths)  # 获得排列的索引值
        sortedPaths = Paths[sortedIndex]  # 对各个种群的路径表按照路径长短升序排列
        return sortedPaths, sortedLengths

    def cloneselect(self, sortedPaths, sortedLengths):
        variaLengths = np.zeros(int(self.Np / 2))  # 存放变异后的挑选出的种群的路径总长度
        variaPaths = np.zeros([int(self.Np / 2), self.N_city]).astype(int)  # 存放变异后挑选出的种群的路径
        # 对选出的优质种群进行克隆变异再选择
        for i in range(int(self.Np / 2)):
            selectedPath = sortedPaths[i, :]  # 选出前Np/2个种群作为优质种群
            clonalPath = np.tile(selectedPath, [self.N_clone, 1])  # 对选出的各个种群克隆N_clone份
            clonalLength = np.zeros(self.N_clone)
            clonalLength[0] = sortedLengths[i]
            for j in range(1, self.N_clone):  # 保留源种群，对剩下的克隆种群进行变异
                # 变异操作，此处选择为换位
                idx = random.sample(range(self.N_city), 2)  # 随机生成两个需要交换的城市的索引
                tmp = clonalPath[j, idx[0]]
                clonalPath[j, idx[0]] = clonalPath[j, idx[1]]
                clonalPath[j, idx[1]] = tmp
                clonalLength[j] = clonselect.caculateLength(self, clonalPath[j])  # 计算变异后的种群路径的总长度
            variaLengths[i] = np.min(clonalLength)  # 变异后种群的最短路径长度
            variaPaths[i] = clonalPath[np.argmin(clonalLength)]  # 变异后种群的最短路径
        return variaPaths, variaLengths

    def solve(self):
        Paths, Lengths = clonselect.initial(self)
        sortedPaths, sortedLengths = clonselect.sortpath(self, Paths, Lengths)
        shortestPath = np.zeros(self.N_city).astype(int)
        shortestLength = np.zeros(self.N_iter)
        iter_times = 0
        while iter_times < self.N_iter:
            variaPaths, variaLengths = clonselect.cloneselect(self, sortedPaths, sortedLengths)
            newPaths, newLengths = clonselect.initial(self)
            # 克隆选择的种群与随机生成的新种群合并
            finalPath = np.vstack([variaPaths, newPaths])  # 最终的各种群路径
            finalLengths = np.hstack([variaLengths, newLengths])  # 最终各种群路径的总长度
            sortedPaths, sortedLengths = clonselect.sortpath(self, finalPath, finalLengths)
            shortestLength[iter_times] = sortedLengths[0]  # 每次迭代的最短路径长度
            shortestPath = sortedPaths[0]  # 每次迭代的最短路径
            iter_times += 1
        return shortestPath, shortestLength

    # def solve(self):
    #     Paths = np.zeros([self.Np, self.N_city]).astype(int)  # 初始化各个种群的路径表
    #     Lengths = np.zeros(self.Np)  # 初始化各个种群所走路径的总长度
    #     for i in range(self.Np):
    #         Paths[i, :] = np.random.permutation(self.N_city).astype(int)  # 初始时随机生成各个种群的路径
    #         Lengths[i] = clonselect.caculateLength(self,Paths[i, :])  # 计算各种群所选路径的长度
    #     sortedLengths = np.sort(Lengths)  # 对路径总长度升序排列
    #     sortedIndex = np.argsort(Lengths)  # 获得排列的索引值
    #     sortedPaths = Paths[sortedIndex]  # 对各个种群的路径表按照路径长短升序排列
    #     iter_times = 0
    #     while iter_times<self.N_iter:
    #         variaLengths = np.zeros(int(self.Np / 2))  # 存放变异后的挑选出的种群的路径总长度
    #         variaPaths = np.zeros([int(self.Np / 2), self.N_city]).astype(int)  # 存放变异后挑选出的种群的路径
    #         # 对选出的优质种群进行克隆变异再选择
    #         for i in range(int(self.Np / 2)):
    #             selectedPath = sortedPaths[i, :]  # 选出前Np/2个种群作为优质种群
    #             clonalPath = np.tile(selectedPath, [self.N_clone, 1])  # 对选出的各个种群克隆N_clone份
    #             clonalLength = np.zeros(self.N_clone)
    #             clonalLength[0] = sortedLengths[i]
    #             for j in range(1, self.N_clone):  # 保留源种群，对剩下的克隆种群进行变异
    #                 # 变异操作，此处选择为换位
    #                 idx = random.sample(range(self.N_city), 2)  # 随机生成两个需要交换的城市的索引
    #                 tmp = clonalPath[j, idx[0]]
    #                 clonalPath[j, idx[0]] = clonalPath[j, idx[1]]
    #                 clonalPath[j, idx[1]] = tmp
    #                 clonalLength[j] = clonselect.caculateLength(self,clonalPath[j])  # 计算变异后的种群路径的总长度
    #             variaLengths[i] = np.min(clonalLength)  # 变异后种群的最短路径长度
    #             variaPaths[i] = clonalPath[np.argmin(clonalLength)]  # 变异后种群的最短路径
    #
    #         # 随机生成剩下的新种群
    #         newPaths = np.zeros([self.Np - int(self.Np / 2), self.N_city]).astype(int)
    #         newLengths = np.zeros(self.Np - int(self.Np / 2))
    #         for i in range(self.Np - int(self.Np / 2)):
    #             newPaths[i, :] = np.random.permutation(self.N_city).astype(int)  # 随机生成新种群的路径
    #             newLengths[i] = clonselect.caculateLength(self, newPaths[i, :])  # 计算新种群所选路径的长度
    #
    #         # 克隆选择的种群与随机生成的新种群合并
    #         finalPath = np.vstack([variaPaths, newPaths])  # 最终的各种群路径
    #         finalLengths = np.hstack([variaLengths, newLengths])  # 最终各种群路径的总长度
    #
    #         sortedLengths = np.sort(finalLengths)
    #         sortedIndex = np.argsort(finalLengths)
    #         sortedPaths = finalPath[sortedIndex]
    #
    #         self.shortestLength[iter_times] = sortedLengths[0]  # 每次迭代的最短路径长度
    #         self.shortestPath = sortedPaths[0]  # 每次迭代的最短路径
    #
    #         iter_times += 1
    #
    #     clonselect.plotresult(self)
    def plotresult(self):
        shortestPath, shortestLength = clonselect.solve(self)
        # 作图
        # 图1：随着迭代次数最短路径长度的变化
        plt.subplot(1, 2, 1)
        plt.plot(shortestLength)
        plt.xlabel('iter_times')
        plt.ylabel('shortest Length')
        plt.title('shortest length: %f' % shortestLength[-1])

        # 图2：迭代完成后的最短路径
        plt.subplot(1, 2, 2)
        # 依次作出路径上相邻两点
        for i in range(self.N_city - 1):
            plt.plot([self.cities[shortestPath[i], 0], self.cities[shortestPath[i + 1], 0]],
                     [self.cities[shortestPath[i], 1], self.cities[shortestPath[i + 1], 1]], 'bo-')
            plt.text(cities[shortestPath[i], 0] + 60, cities[shortestPath[i], 1] - 50,
                     shortestPath[i], ha='center', va='bottom', fontsize=10)

        # 从终点回到起点
        plt.plot([self.cities[shortestPath[-1], 0], self.cities[shortestPath[0], 0]],
                 [self.cities[shortestPath[-1], 1], self.cities[shortestPath[0], 1]], 'r')
        plt.text(cities[shortestPath[-1], 0] + 60, cities[shortestPath[-1], 1] - 50,
                 shortestPath[-1], ha='center', va='bottom', fontsize=10)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('shortest path')
        plt.show()
        print('shortest length: ', shortestLength[-1])
        print('final path：', shortestPath)


class antsys():
    def __init__(self, cities:np.ndarray, N_ant:int, N_iter:int, parameters:list):
        self.cities = cities
        self.N_city = self.cities.shape[0]
        self.N_ant = N_ant
        self.N_iter = N_iter
        self.alpha = parameters[0]
        self.beta = parameters[1]
        self.rho = parameters[2]
        self.Q = parameters[3]

    # 函数：分配蚂蚁
    def distriAnt(N_ant, N_city):
        if N_ant <= N_city:
            passing = np.random.permutation(range(N_city))[:N_ant]
        else:
            passing = np.random.permutation(range(N_city))
            passing = np.append(passing, np.random.permutation(range(N_city))[:N_ant - N_city])
        return passing




cities = np.array([[1304, 2312], [3639, 1315], [4177, 2244], [3712, 1399], [3488, 1535], [3326, 1556],
                   [3238, 1229], [4196, 1044], [4312, 790], [4386, 570], [3007, 1970], [2562, 1756],
                   [2788, 1491], [2381, 1676], [1332, 695], [3715, 1678], [3918, 2179], [4061, 2370],
                   [3780, 2212], [3676, 2578], [4029, 2838], [4263, 2931], [3429, 1908], [3507, 2376],
                   [3394, 2643], [3439, 3201], [2935, 3240], [3140, 3550], [2545, 2357], [2778, 2826],
                   [2370, 2975]])
N_city = cities.shape[0] #城市个数
Np = 200  #种群个数
N_iter = 1000 #迭代次数
N_clone = 10  #克隆个数

ex = clonselect(cities, 200, 1000, 10)
ex.plotresult()
