import numpy as np
import matplotlib.pyplot as plt

cities = np.loadtxt('coordinates.txt')
#函数：计算各城市间的距离矩阵
def getDistMat(cities):
    #城市个数
    N_city = cities.shape[0]
    #初始化距离矩阵
    DistMat = np.zeros([N_city, N_city])
    for i in range(N_city):
        for j in  range(i, N_city):
            DistMat[i][j] = DistMat[j][i] = np.linalg.norm(cities[i]-cities[j])
    return DistMat

#函数：分配蚂蚁
def distributeAnt(N_ant, N_city):
    if N_ant <=N_city:
        passing = np.random.permutation(range(N_city))[:N_ant]
    else:
        passing = np.random.permutation(range(N_city))
        passing = np.append(passing, np.random.permutation(range(N_city))[:N_ant - N_city])
    return passing

N_ant = 100 #蚂蚁个数
N_city = cities.shape[0] #城市个数
alpha = 2 #信息素重要程度
beta = 2 #启发因子重要程度
rho = 0.2 #信息素的挥发速度
Q = 1 #完成率

iter = 0 #迭代初始
itermax = 100 #迭代次数

DistMat = getDistMat(cities)
pheromone_tab = np.ones([N_city, N_city])
expect_tab = 1/(DistMat + np.diag([1e10]*N_city))
path_tab = np.zeros([N_ant, N_city]).astype(int)

length_aver = np.zeros(itermax)   #初始化每次迭代的平均路径长度
length_best = np.zeros(itermax)   #初始化每次迭代的最短路径长度
path_best = np.zeros([itermax, N_city])  #初始化每次迭代的最短路径

while iter<itermax:
    #蚂蚁起始城市
    path_tab[:, 0] = distributeAnt(N_ant, N_city)
    length = np.zeros(N_ant)

    #计算每只蚂蚁转移到下一个城市的概率
    for i in range(N_ant):
        passing = path_tab[i][0]  #当前所在的城市
        unpassed = list(range(N_city)) #还未经过的城市
        unpassed.remove(passing)

        #循环N_city-1次，访问剩余的N_city-1个城市
        for j in range(1, N_city):
            prob_mat = np.zeros(len(unpassed))  #转移概率矩阵

            #计算转移概率
            for k in range(len(unpassed)):
                prob_mat[k] = np.power(pheromone_tab[passing][unpassed[k]], alpha)*np.power(expect_tab[passing][unpassed[k]], beta)

            #计算累积函数
            cumsumprob_mat = (prob_mat/sum(prob_mat)).cumsum()
            #使用轮盘赌算法选择下一个城市
            cumsumprob_mat -= np.random.rand()
            next_city = np.array(unpassed)[np.where(cumsumprob_mat>0)][0]

            path_tab[i][j] = next_city
            unpassed.remove(next_city)
            length[i] += DistMat[passing][next_city]
            passing = next_city
        length[i] += DistMat[passing][path_tab[i][0]]
    length_aver[iter] = length.mean()

    #更新信息素
    updatephero_tab = np.zeros([N_city, N_city])
    for i in range(N_city):
        for j in range(N_city-1):
            updatephero_tab[path_tab[i,j],path_tab[i,j+ 1]] += Q / DistMat[path_tab[i,j],path_tab[i,j+ 1]]
        updatephero_tab[path_tab[i,j+1],path_tab[i,0]] += Q / DistMat[path_tab[i,j+1],path_tab[i,0]]

    #信息素更新
    pheromone_tab = (1-rho) * pheromone_tab + updatephero_tab


    #求出最短路径
    if iter == 0:
        length_best[iter] = length.min()
        path_best[iter] = path_tab[length.argmin()].copy()
    else:
        if length.min() < length_best[iter-1]:
            length_best[iter] = length.min()
            path_best[iter] = path_tab[length.argmin()].copy()
        else:
            length_best[iter] = length_best[iter-1]
            path_best[iter] = path_best[iter-1].copy()
    iter += 1

#作图
fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(12,10))
axes[0].plot(length_aver,'k*-')
axes[0].set_title('Average Length')
axes[0].set_xlabel('iter times')

axes[1].plot(length_best,'k<-')
axes[1].set_title('Shortest Length')
axes[1].set_xlabel('iter times')
plt.show()
bestpath = path_best[-1]
# fig2 = plt.figure(2)
# plt.plot(cities[:, 0], cities[:, 1], 'r.')
# plt.xlim([-100, 2000])
# # x范围
# plt.ylim([-100, 1500])
# # y范围

# for i in range(N_city - 1):
#     # 按坐标绘出最佳两两城市间路径
#     m, n = int(bestpath[i]), int(bestpath[i + 1])
#     plt.plot([cities[m][0], cities[n][0]], [cities[m][1], cities[n][1]], 'bo-')
#
# plt.plot([cities[int(bestpath[0])][0], cities[int(bestpath[-1])][0]],
#          [cities[int(bestpath[0])][1], cities[int(bestpath[-1])][1]], 'r')

# ax = plt.gca()
# ax.set_title("Best Path")
# ax.set_xlabel('X_axis')
# ax.set_ylabel('Y_axis')
# plt.show()

