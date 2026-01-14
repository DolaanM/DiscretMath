def find_eulerian_cycle(graph):
    g = {u: graph[u][:] for u in graph}
    start = next(iter(graph))  # берём любую стартовую вершину
    stack = [start]
    path = []

    while stack:
        v = stack[-1]
        if g[v]:
            u = g[v].pop()
            stack.append(u)
        else:
            path.append(stack.pop())
    return path[::-1]

def find_eulerian_cycle_undirected(graph):
    g = {u: graph[u][:] for u in graph}
    start = next(iter(graph))
    stack = [start]
    path = []

    while stack:
        v = stack[-1]
        if g[v]:
            u = g[v].pop()
            g[u].remove(v)  # удаляем ребро в обе стороны
            stack.append(u)
        else:
            path.append(stack.pop())
    return path[::-1]

def has_eulerian_cycle(graph):
    indeg = {v: 0 for v in graph}
    outdeg = {v: len(graph[v]) for v in graph}
    for u in graph:
        for v in graph[u]:
            indeg[v] = indeg.get(v, 0) + 1

    for v in graph:
        if indeg.get(v, 0) != outdeg.get(v, 0):
            return False
    return True

def has_eulerian_cycle_undirected(graph):
    for v in graph:
        if len(graph[v]) % 2 != 0:
            return False
    return True

def is_directed(graph):
    for u in graph:
        for v in graph[u]:
            if u not in graph.get(v, []):  # нет обратного ребра
                return True
    return False

def is_connected(graph):
    # игнорируем изолированные вершины
    start = next((v for v in graph if graph[v]), None)
    if not start:
        return False  # все изолированные

    visited = set()
    stack = [start]
    while stack:
        v = stack.pop()
        if v not in visited:
            visited.add(v)
            for u in graph[v]:
                stack.append(u)
    # все вершины, у которых есть рёбра, должны быть посещены
    for v in graph:
        if graph[v] and v not in visited:
            return False
    return True

def check_and_find_eulerian_cycle(graph):
    if not is_connected(graph):
        return "Нет Эйлерова цикла: граф несвязный."

    if is_directed(graph):
        if not has_eulerian_cycle(graph):
            return "Нет Эйлерова цикла (ориентированный граф)."
        return find_eulerian_cycle(graph)
    else:
        if not has_eulerian_cycle_undirected(graph):
            return "Нет Эйлерова цикла (неориентированный граф)."
        return find_eulerian_cycle_undirected(graph)

# Неориентированный граф (цикл A–B–C–A)
graph1 = {
    'A': ['B', 'C'],
    'B': ['A', 'C'],
    'C': ['A', 'B']
}

# Ориентированный граф (A→B→C→A)
graph2 = {
    'A': ['B'],
    'B': ['C'],
    'C': ['A']
}

print(check_and_find_eulerian_cycle(graph1))
print(check_and_find_eulerian_cycle(graph2))