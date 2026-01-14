class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # сжатие пути
        return self.parent[x]

    def union(self, x, y):
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return False  # уже в одном множестве
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1
        return True

def kruskal(n, edges):
    """
    n — количество вершин (нумерация с 0 до n-1)
    edges — список рёбер вида (w, u, v)
    """
    dsu = DSU(n)
    mst_weight = 0
    mst_edges = []

    edges.sort()  # сортируем по весу

    for w, u, v in edges:
        if dsu.union(u, v):
            mst_weight += w
            mst_edges.append((u, v, w))

    return mst_weight, mst_edges


edges = [
    (10, 0, 1),
    (6, 0, 2),
    (5, 0, 3),
    (15, 1, 3),
    (4, 2, 3)
]

n = 4
mst_weight, mst_edges = kruskal(n, edges)

print("Минимальный вес остова:", mst_weight)
print("Рёбра в остове:")
for u, v, w in mst_edges:
    print(f"{u} - {v} (вес {w})")