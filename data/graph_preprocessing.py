import numpy as np


def visit_graph(adj, labels, out_degree, roots=None):
    if roots is None:
        n_trees = adj.shape[0]
        iterator = range(n_trees)
    else:
        n_trees = len(roots)
        iterator = roots
    trees = []
    levels = []
    n_levels = []
    max_levels = -1
    max_nodes = -1
    for i in iterator:
        t, t_levels, t_n_levels = _build_tree(adj, i, labels, out_degree)
        trees.append(t)
        levels.append(t_levels)
        n_levels.append(t_n_levels)
        max_nodes = max(max_nodes, len(t))
        max_levels = max(max_levels, len(t_levels))

    for i in iterator:
        trees[i] = np.pad(trees[i], [[0, max_nodes - trees[i].shape[0]], [0, 0]], mode='constant')
        levels[i] = np.pad(levels[i], [0, max_levels - n_levels[i]], mode='constant')

    n_trees = np.array(n_trees, dtype=np.int32)
    trees = np.stack(trees, axis=0)
    n_levels = np.stack(n_levels, axis=0) 
    levels = np.stack(levels, axis=0)

    return n_trees, trees.astype(np.int32), n_levels.astype(np.int32), levels.astype(np.int32)


def _build_tree(adj, root, labels, out_degree):
    internal = []
    leaves = []
    levels = []
    visited = [root]
    queue = [([root, 0, 0], 0)]
    while queue:
        node, level = queue.pop(0)
        children = np.nonzero(adj[node[0], :])[0]
        j = 0
        for i in range(len(children)):
            if children[i] not in visited:
                # Setting the i-th child of the node that we are visiting: 
                # it is the chid's index of the adj (we still don't know its index, it could be a leaf)
                node.append(children[i])

                # The children has ([its index, father = curr length of the tree, pos = j], curr_level+1)
                queue.append(([children[i], len(internal), j], level+1))

                # This child is going to be visited (as a child of the current node)
                visited.append(children[i])
                j += 1
        if j != 0 and level == len(levels):
            levels.append(len(internal))

        node = np.array(node, dtype=np.int32)
        if j != 0:
            internal.append(node)
        else:
            leaves.append(node)
    levels.append(len(internal))   # Appending the leaves' first index
    tree = internal + leaves
    levels.append(len(tree))    # Append the tree's size

    # Setting the internal nodes' children's index
    for n in range(len(internal)):        
        for child in range(3, len(tree[n])):            
            for j in range(n+1, len(tree)):
                if tree[j][0] == tree[n][child]:    # We found its children index
                    tree[n][child] = j
                    break

    # Setting the nodes' labels in the multinomial alphabet
    for n in range(len(tree)):
        tree[n][0] = labels[tree[n][0]]
        tree[n] = np.pad(tree[n], [0, 3+out_degree-len(tree[n])], mode='constant')

    return np.array([np.array(node, dtype=np.int32) for node in tree], dtype=np.int32), np.array(levels, dtype=np.int32), np.array(len(levels), dtype=np.int32)
