import numpy as np


def parse_and_preproc_data(file):
    if file == 'inex2005':
        features, labels = _dataset_parser('./htmn2/data/inex/2005/inex05.test.elastic.tree', 32)
    elif file == 'inex2006':
        features, labels = _dataset_parser('./htmn2/data/inex/2006/inex06.test.elastic.tree', 66)

    return features, labels


def _dataset_parser(file, max_children):
    with open(file, "r") as ins:
        line_tree = []
        for line in ins:
            line_tree.append(line)
    ins.close()

    features = {'tree': [],
                'limits': []}
    labels = []
    for line in line_tree:
        f_tree, lim_tree, lab_tree = _build_tree(line, max_children)
        features['tree'].append(f_tree)
        features['limits'].append(lim_tree)
        labels.append(lab_tree)
    features, labels = _dataset_to_batch(features, labels)

    return features, labels


def _build_tree(line, max_children):
    t_class, t_line = line.split(':')
    tree = []
    leaves = []
    t_iter = iter(t_line)
    c = next(t_iter)

    stack = []
    curr_node = ['', 0, 0] + [0 for _ in range(max_children)]
    end = False
    while not end:
        try:
            if c == '(':
                curr_node[0] = int(curr_node[0])-1
                stack.append([curr_node, 0])
                curr_node = ['', None, stack[-1][1]] + [0 for _ in range(max_children)]

            elif c in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]:
                curr_node[0] += c

            elif c == ' ':
                curr_node = ['', None, stack[-1][1]] + [0 for _ in range(max_children)]

            elif c == ')':
                closed_node, cnt = stack.pop()
                if stack:
                    stack[-1][1] += 1
                if len(tree) < len(stack):
                    tree += [[] for _ in range(len(stack)-len(tree))]

                closed_node[1] = (len(stack) - 1, len(tree[len(stack)-1]))
                if cnt == 0:
                    leaves.append(closed_node)
                else:
                    tree[len(stack)].append(closed_node)
            c = next(t_iter)
        except StopIteration:
            end = True

    tree.append(leaves)
    t_limits = [0]
    tree[0][0][1] = 0
    i = 1
    for l in range(1, len(tree)):
        t_limits.append(i)
        for node in tree[l]:
            p_level, p_ind = node[1]
            tree[p_level][p_ind][3 + node[2]] = i
            node[1] = t_limits[p_level] + p_ind
            i += 1
    t_limits.append(i)
    tree = np.concatenate([np.array(level) for level in tree], axis=0)
    
    return tree.astype(dtype=np.int32), np.array(t_limits, dtype=np.int32), int(t_class)-1


def _dataset_to_batch(features, labels):
    max_size = -1
    max_levels = -1
    for lim in features['limits']:
        max_levels = max(lim.shape[0]-1, max_levels)
        max_size = max(lim[-1], max_size)

    dataset_size = len(features['tree'])
    for i in range(dataset_size):
        curr_pad_size = max_size - features['limits'][i][-1]
        if curr_pad_size > 0:
            tree_pad = np.array([[0, curr_pad_size], [0, 0]], dtype=np.int32)
            features['tree'][i] = np.pad(features['tree'][i], tree_pad, 'constant')

        curr_pad_levels = max_levels - (features['limits'][i].shape[0] - 1)  # There is the additional level with t_size
        if curr_pad_levels > 0:
            lim_pad = np.array([[0, curr_pad_levels]])
            features['limits'][i] = np.pad(features['limits'][i], lim_pad, 'constant')

    features['tree'] = np.stack(features['tree'])
    features['limits'] = np.stack(features['limits'])

    labels = np.array(labels, dtype=np.int32)

    return features, labels
