import numpy as np

from data.graph_preprocessing import visit_graph


def to_dict(graph_adj, nodes_labels, out_degree):
    features = {'n_trees': [],
                'trees': [],
                'n_levels': [],
                'levels': []}

    for adj, lab in zip(graph_adj, nodes_labels):
        n_trees, tree, n_levels, levels = visit_graph(adj, lab, out_degree)
        features['n_trees'].append(n_trees)
        features['trees'].append(tree)
        features['n_levels'].append(n_levels)
        features['levels'].append(levels)

    return features


def to_dict_batch(features, max_trees):
    max_levels = -1
    for n_lev in features['n_levels']:
        max_levels = max(max_levels, np.amax(n_lev))

    for i in range(len(features['n_trees'])):
        trees = features['trees'][i]
        features['trees'][i] = np.pad(trees, [[0, max_trees-trees.shape[0]],
                                              [0, max_trees-trees.shape[1]],
                                              [0, 0]], 'constant')
        n_levels = features['n_levels'][i]
        features['n_levels'][i] = np.pad(n_levels, [[0, max_trees-n_levels.shape[0]]], 'constant')

        levels = features['levels'][i]
        features['levels'][i] = np.pad(levels, [[0, max_trees-levels.shape[0]], 
                                               [0, max_levels-levels.shape[1]]], 'constant')

    features['n_trees'] = np.stack(features['n_trees'])
    features['trees'] = np.stack(features['trees'])
    features['n_levels'] = np.stack(features['n_levels'])
    features['levels'] = np.stack(features['levels'])
    return features
