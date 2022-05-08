import matplotlib.pyplot as plt

from ACupdate import ACO
from BCupdate import BC

edges = [('A','B'),('A','C'),('B','D'),('B','E'),('C','E'),('C','F'),('D','G'),('E','G'),('F','G')]
overhead_AC = ACO(edges, 'B', 'F', 15)
overhead_BC = BC(edges, 'B','F', 15)
x = [x for x in range(20)]
# x = [10+10*k for k in range(15)]
plt.plot(x,overhead_AC,'ro-')
plt.plot(x,overhead_BC,'gs-')
plt.xlabel('Number of Interests')
plt.ylabel('overhead_ratio')
plt.title('AC VS BC')
plt.legend(['AC','BC'])
plt.show()