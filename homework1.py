import numpy as np
from data import load_data
import scipy
import networkx as nx
import matplotlib.pyplot as plt


def BFS(graph, s, end):  # graph图  s指的是开始结点,end是结束节点
    queue = []
    parents = {}
    queue.append(s)
    visited = set()
    visited.add(s)
    while len(queue) > 0:
        vertex = queue.pop(0)  # pop移除列表中的一个元素
        if vertex == end:
            print("已找到目标节点！")
            return parents
        nodes = graph[vertex]  # 子节点的数组
        for w in nodes:
            if w not in visited:  # 判断是否访问过，使用一个数组
                queue.append(w)
                visited.add(w)
                parents[w] = vertex  # 追溯父节点
        # print(vertex)
    return -1


def find_way(parents, start, end):
    trace = []
    trace.append(end)
    while parents[end] != start:
        end = parents[end]
        trace.append(end)
    trace.append(start)
    trace.reverse()
    return trace


def adj_to_adt(GF):
    g = {}
    for i in range(len(GF)):
        a = []
        for j in range(len(GF)):
            if GF[i][j] != 0:
                a.append(j)
                g[i] = a
    return g


def cluster_coefficient(G):
    C = 0
    N = len(G)
    for i in range(N):
        count = 0
        degree = G[i, :].sum()
        for j in range(N):
            if G[i, j] == 0:
                continue
            else:
                for k in range(N):
                    if G[j, k] == 0:
                        continue
                    else:
                        count += 1
        if degree * (degree - 1) == 0:
            C = C
        else:
            C += count / (degree * (degree - 1))
    return C / N

def draw_degree_distribution(G):
    L = []
    for i in G:
        L.append(i.sum())
    d = {}
    for key in L:
        d[key] = d.get(key, 0) + 1

    d = sorted(d.items(), key=lambda d: d[0])
    x = []
    y = []
    for i in d:
        x.append(i[0])
        y.append(i[1])
    plt.title("cora_dataset_degree_distribution")
    plt.plot(x, y)
    plt.show()


def find_max_connected_graph(G):
    max_index = 0
    max = 0
    for i in range(len(G)):
        if len(G[i]) > max:
            max_index = i
            max = len(G[i])
    return max_index

if __name__ == '__main__':
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data("cora")
    graph = adj.A
    g = adj_to_adt(graph)
    print("转化cora数据集的邻接表表示", g)

    # 找聚类系数
    print("cora数据集的聚类系数为:", cluster_coefficient(graph))

    # 度分布
    draw_degree_distribution(graph)

    # BFS最短路径算法
    print("请输入起始节点编号:")
    start = int(input())
    print("请输入终止节点编号:")
    end = int(input())
    parents = BFS(g, start, end)
    if parents == -1:
        print("两点之间没有连通")
    else:
        trace = find_way(parents, start, end)
        print("最短路径已找到:", trace, "最短距离为:", len(trace) - 1)

    # 最大联通巨片
    print("最大连通巨片:", g[find_max_connected_graph(g)])