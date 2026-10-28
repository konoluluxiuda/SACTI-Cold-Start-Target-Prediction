import networkx as nx
import numpy as np


def bipartite_pagerank(adjacency_matrix, alpha=0.85):
    """Return PageRank vectors without merging equal row/column integer IDs."""
    graph = nx.Graph()
    row_count, column_count = adjacency_matrix.shape
    left_nodes = [('left', index) for index in range(row_count)]
    right_nodes = [('right', index) for index in range(column_count)]
    graph.add_nodes_from(left_nodes)
    graph.add_nodes_from(right_nodes)

    coo = adjacency_matrix.tocoo()
    for row, column, value in zip(coo.row, coo.col, coo.data):
        graph.add_edge(('left', int(row)), ('right', int(column)), weight=float(value))

    scores = nx.pagerank(graph, alpha=alpha, weight='weight')
    left_scores = np.asarray([scores[node] for node in left_nodes], dtype=np.float32)
    right_scores = np.asarray([scores[node] for node in right_nodes], dtype=np.float32)
    return left_scores, right_scores
