import numpy as np
import scipy


def parse_and_stats(file):
    loaded = np.load('./data/synth_graphs/'+file+'.npz', allow_pickle=True)
    if file == 'easy_small':
        max_trees, L = 78, 12
    if file == 'easy':
        max_trees, L = 198, 13
    if file == 'hard_small':
        max_trees, L = 78, 8
    if file == 'hard':
        max_trees, L = 198, 9

    X_train = loaded['tr_feat']
    A_train = list(loaded['tr_adj'])
    y_train = loaded['tr_class']

    X_val = loaded['val_feat']
    A_val = list(loaded['val_adj'])
    y_val = loaded['val_class']

    X_test = loaded['te_feat']
    A_test = list(loaded['te_adj'])
    y_test = loaded['te_class']

    train = {'nodes': [np.argmax(arr, axis=-1) for arr in X_train],
             'adj': [a.todense() for a in A_train],
             'lab': np.argmax(y_train, axis=-1)}

    val = {'nodes': [np.argmax(arr, axis=-1) for arr in X_val],
           'adj': [a.todense() for a in A_val],
           'lab': np.argmax(y_val, axis=-1)}

    test = {'nodes': [np.argmax(arr, axis=-1) for arr in X_test],
            'adj': [a.todense() for a in A_test],
            'lab': np.argmax(y_test, axis=-1)}

    return train, val, test, max_trees, L
