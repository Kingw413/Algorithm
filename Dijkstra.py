def Dijkstra(graph, startVertex):
    passnodes = [startVertex]
    notpassnodes = [x for x in range(len(graph)) if x != startVertex]
    result = graph[startVertex]
    while len(notpassnodes):
        distance = [result[k] for k in notpassnodes]
        idx = result.index(min(distance))
        passnodes.append(idx)
        notpassnodes.remove(idx)

        for i in notpassnodes:
            if result[idx] + graph[idx][i] < result[i]:
                result[i] = result[idx] + graph[idx][i]
    print(result)


inf = float('inf')
graph = [[0, 1, 12, inf, inf, inf],
          [inf, 0, 9, 3, inf, inf],
          [inf, inf, 0, inf, 5, inf],
          [inf, inf, 4, 0, 13, 15],
          [inf, inf, inf ,inf, 0, 4],
          [inf, inf, inf, inf ,inf, 0]]

Dijkstra(graph, 0)