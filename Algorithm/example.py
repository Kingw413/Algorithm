# import numpy as np
# from AntSystem import ASsolve
# from CloneSelect import CSsolve
#
# cities = np.array([[1304, 2312], [3639, 1315], [4177, 2244], [3712, 1399], [3488, 1535], [3326, 1556],
#                    [3238, 1229], [4196, 1044], [4312, 790], [4386, 570], [3007, 1970], [2562, 1756],
#                    [2788, 1491], [2381, 1676], [1332, 695], [3715, 1678], [3918, 2179], [4061, 2370],
#                    [3780, 2212], [3676, 2578], [4029, 2838], [4263, 2931], [3429, 1908], [3507, 2376],
#                    [3394, 2643], [3439, 3201], [2935, 3240], [3140, 3550], [2545, 2357], [2778, 2826],
#                    [2370, 2975]])
# N_ant = 40    # 蚂蚁个数
# N_city = cities.shape[0]  # 城市个数
# alpha = 2     # 信息素重要程度
# beta = 5      # 启发因子重要程度
# rho = 0.5     # 信息素的挥发速度
# Q = 1         # 完成率
# N_iter = 100  # 迭代次数
# Np = 40       # 种群个数
# N_clone = 10  # 克隆个数
# ASsolve(cities,N_ant,N_iter,alpha,beta,rho,Q)
# CSsolve(cities,Np,N_clone,N_iter)


import os
import random
os.getcwd()
# 返回当前工作目录
import numpy as np
import matplotlib.pyplot as plt

coordinates = np.loadtxt('coordinates.txt')

# coordinates = []
# x_coordinates = random.sample(range(100,2000),52)
# y_coordinates = random.sample(range(100,2000),52)
# for i in range(52):
#     coordinates.append([x_coordinates[i], y_coordinates[i]])
# coordinates = np.array(coordinates)

# 计算52个城市间的欧式距离
def getdistmat(coordinates):
    num = coordinates.shape[0]
    distmat = np.zeros((52, 52))
    # 初始化生成52*52的矩阵
    for i in range(num):
        for j in range(i, num):
            distmat[i][j] = distmat[j][i] = np.linalg.norm(coordinates[i] - coordinates[j])
    return distmat


# 返回城市距离矩阵


distmat = getdistmat(coordinates)

numant = 100  # 蚂蚁个数
numcity = coordinates.shape[0]
# shape[0]=52 城市个数,也就是任务个数
alpha = 2  # 信息素重要程度因子
beta = 2  # 启发函数重要程度因子
rho = 0.2  # 信息素的挥发速度
Q = 1  # 完成率

iter = 0  # 迭代初始
itermax = 100  # 迭代总数

etatable = 1.0 / (distmat + np.diag([1e10] * numcity))
# diag(),将一维数组转化为方阵 启发函数矩阵，表示蚂蚁从城市i转移到城市j的期望程度
pheromonetable = np.ones((numcity, numcity))
# 信息素矩阵 52*52
pathtable = np.zeros((numant, numcity)).astype(int)
# 路径记录表，转化成整型 40*52
distmat = getdistmat(coordinates)
# 城市的距离矩阵 52*52

lengthaver = np.zeros(itermax)  # 迭代50次，存放每次迭代后，路径的平均长度  50*1
lengthbest = np.zeros(itermax)  # 迭代50次，存放每次迭代后，最佳路径长度  50*1
pathbest = np.zeros((itermax, numcity))  # 迭代50次，存放每次迭代后，最佳路径城市的坐标 50*52

while iter < itermax:
    # 迭代总数

    # 40个蚂蚁随机放置于52个城市中
    if numant <= numcity:  # 城市数比蚂蚁数多，不用管
        pathtable[:, 0] = np.random.permutation(range(numcity))[:numant]
        # 返回一个打乱的40*52矩阵，但是并不改变原来的数组,把这个数组的第一列(40个元素)放到路径表的第一列中
        # 矩阵的意思是哪个蚂蚁在哪个城市,矩阵元素不大于52
    else:  # 蚂蚁数比城市数多，需要有城市放多个蚂蚁
        pathtable[:numcity, 0] = np.random.permutation(range(numcity))[:]
        # 先放52个
        # pathtable[numcity:, 0] = np.random.permutation(range(numcity))[:numant - numcity]
        # 再把剩下的放完
        # print(pathtable[:,0])
        for k in range(numant-numcity):
            pathtable[numcity+k, 0] = random.sample(range(numcity),1)[0]

    length = np.zeros(numant)  # 1*40的数组

    # 本段程序算出每只/第i只蚂蚁转移到下一个城市的概率
    for i in range(numant):

        # i=0
        visiting = pathtable[i, 0]  # 当前所在的城市
        # set()创建一个无序不重复元素集合
        # visited = set() #已访问过的城市，防止重复
        # visited.add(visiting) #增加元素
        unvisited = set(range(numcity))
        # 未访问的城市集合
        # 剔除重复的元素
        unvisited.remove(visiting)  # 删除已经访问过的城市元素

        for j in range(1, numcity):  # 循环numcity-1次，访问剩余的所有numcity-1个城市
            # j=1
            # 每次用轮盘法选择下一个要访问的城市
            listunvisited = list(unvisited)
            # 未访问城市数,list
            probtrans = np.zeros(len(listunvisited))
            # 每次循环都初始化转移概率矩阵1*52,1*51,1*50,1*49....

            # 以下是计算转移概率
            for k in range(len(listunvisited)):
                probtrans[k] = np.power(pheromonetable[visiting][listunvisited[k]], alpha) \
                               * np.power(etatable[visiting][listunvisited[k]], beta)
            # eta-从城市i到城市j的启发因子 这是概率公式的分母   其中[visiting][listunvis[k]]是从本城市到k城市的信息素
            cumsumprobtrans = (probtrans / sum(probtrans)).cumsum()
            # 求出本只蚂蚁的转移到各个城市的概率斐波衲挈数列

            cumsumprobtrans -= np.random.rand()
            # 随机生成下个城市的转移概率，再用区间比较
            # k = listunvisited[find(cumsumprobtrans > 0)[0]]
            k = listunvisited[list(cumsumprobtrans > 0).index(True)]
            # k = listunvisited[np.where(cumsumprobtrans > 0)[0]]
            # where 函数选出符合cumsumprobtans>0的数
            # 下一个要访问的城市

            pathtable[i, j] = k
            # 采用禁忌表来记录蚂蚁i当前走过的第j城市的坐标，这里走了第j个城市.k是中间值
            unvisited.remove(k)
            # visited.add(k)
            # 将未访问城市列表中的K城市删去，增加到已访问城市列表中

            length[i] += distmat[visiting][k]
            # 计算本城市到K城市的距离
            visiting = k

        length[i] += distmat[visiting][pathtable[i, 0]]
        # 计算本只蚂蚁的总的路径距离，包括最后一个城市和第一个城市的距离

    # print("ants all length:",length)
    # 包含所有蚂蚁的一个迭代结束后，统计本次迭代的若干统计参数

    lengthaver[iter] = length.mean()
    # 本轮的平均路径

    # 本部分是为了求出最佳路径

    if iter == 0:
        lengthbest[iter] = length.min()
        pathbest[iter] = pathtable[length.argmin()].copy()
    # 如果是第一轮路径，则选择本轮最短的路径,并返回索引值下标，并将其记录
    else:
        # 后面几轮的情况，更新最佳路径
        if length.min() > lengthbest[iter - 1]:
            lengthbest[iter] = lengthbest[iter - 1]
            pathbest[iter] = pathbest[iter - 1].copy()
        # 如果是第一轮路径，则选择本轮最短的路径,并返回索引值下标，并将其记录
        else:
            lengthbest[iter] = length.min()
            pathbest[iter] = pathtable[length.argmin()].copy()

    # 此部分是为了更新信息素
    changepheromonetable = np.zeros((numcity, numcity))
    for i in range(numant):  # 更新所有的蚂蚁
        for j in range(numcity - 1):
            changepheromonetable[pathtable[i, j]][pathtable[i, j + 1]] += Q / distmat[pathtable[i, j]][
                pathtable[i, j + 1]]
            # 根据公式更新本只蚂蚁改变的城市间的信息素      Q/d   其中d是从第j个城市到第j+1个城市的距离
        changepheromonetable[pathtable[i, j + 1]][pathtable[i, 0]] += Q / distmat[pathtable[i, j + 1]][pathtable[i, 0]]
        # 首城市到最后一个城市 所有蚂蚁改变的信息素总和

    # 信息素更新公式p=(1-挥发速率)*现有信息素+改变的信息素
    pheromonetable = (1 - rho) * pheromonetable + changepheromonetable

    iter += 1  # 迭代次数指示器+1
    # print("this iteration end：", iter)
    # # 观察程序执行进度，该功能是非必须的
    # if (iter - 1) % 20 == 0:
    #     print("schedule:", iter - 1)
# 迭代完成


# 以下是做图部分
# 做出平均路径长度和最优路径长度
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
axes[0].plot(lengthaver, 'k*-')
axes[0].set_title('Average Length(ex)')
axes[0].set_xlabel(u'iteration')

# 线条颜色black
axes[1].plot(lengthbest, 'k<-')
axes[1].set_title('Best Length(ex)')
axes[1].set_xlabel(u'iteration')
#fig.savefig('Average_Best.png', dpi=500, bbox_inches='tight')
plt.close()
fig.show()

# 作出找到的最优路径图
# bestpath = pathbest[-1]
#
# plt.plot(coordinates[:, 0], coordinates[:, 1], 'r.')
# plt.xlim([0, 2500])
# # x范围
# plt.ylim([0, 2500])
# y范围

# for i in range(numcity - 1):
#     # 按坐标绘出最佳两两城市间路径
#     m, n = int(bestpath[i]), int(bestpath[i + 1])
#     #print("best_path:", m, n)
#     plt.plot([coordinates[m][0], coordinates[n][0]], [coordinates[m][1], coordinates[n][1]], 'k')
#
# plt.plot([coordinates[int(bestpath[0])][0], coordinates[int(bestpath[51])][0]],
#          [coordinates[int(bestpath[0])][1], coordinates[int(bestpath[51])][1]], 'b')
#
# ax = plt.gca()
# ax.set_title("Best Path")
# ax.set_xlabel('X_axis')
# ax.set_ylabel('Y_axis')

#plt.savefig('Best Path.png', dpi=500, bbox_inches='tight')
# plt.show()

# Path_coordinates = []
# for city_num in bestpath:
#     Path_coordinates.append(coordinates[int(city_num)])
# Path_coordinates = np.array(Path_coordinates)
# np.savetxt('coordinates.txt',coordinates,fmt="%d")
# np.savetxt('path.txt',Path_coordinates,fmt="%d")
# np.savetxt('averageLength.txt',lengthaver,fmt="%.2f")
# np.savetxt('shortestLength.txt',lengthbest,fmt="%.2f")