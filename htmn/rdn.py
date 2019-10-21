import tensorflow as tf
import numpy as np
from math import factorial as fact


class RDN(tf.keras.Model):
    """
    Implementation of the Relative Density Network
    """

    def __init__(self, task, outputs, n_bu, n_td):
        super(RDN, self).__init__(self)
        if n_bu > 0 and n_td > 0:
            self.bu_bnorm = tf.keras.layers.BatchNormalization()
            self.td_bnorm = tf.keras.layers.BatchNormalization()
        contrastive_units = fact(n_bu+n_td) // (2*fact(n_bu+n_td-2))
        self.contrastive = tf.keras.layers.Dense(units=contrastive_units,
                                                 activation='tanh',
                                                 use_bias=False,
                                                 kernel_initializer=tf.keras.initializers.constant(contrastive_matrix(n_bu+n_td)),
                                                 trainable=False)

        if task == 'C':
            if outputs == 2:
                self.out_layer = tf.keras.layers.Dense(units=1,
                                                    activation='sigmoid')
            elif outputs > 2:
                self.out_layer = tf.keras.layers.Dense(units=outputs)
        elif task == 'R':
            self.out_layer = tf.keras.layers.Dense(units=outputs)

    def call(self, bu_likelihood, td_likelihood):
        if bu_likelihood is not None and td_likelihood is not None:
            bu_likelihood = self.bu_bnorm(bu_likelihood)
            td_likelihood = self.bu_bnorm(td_likelihood)
            likelihood = tf.concat([bu_likelihood, td_likelihood], axis=1)

        elif bu_likelihood is not None:
            likelihood = bu_likelihood
        elif td_likelihood is not None:
            likelihood = td_likelihood

        contrastive_values = self.contrastive(likelihood)
        output = self.out_layer(contrastive_values)
        return output


def contrastive_matrix(N_GEN):
    contrastive_units = fact(N_GEN) // (2*fact(N_GEN-2))
    contrastive_matrix = np.zeros((N_GEN, contrastive_units), dtype=np.float64)

    p = 0
    s = 1
    for i in range(contrastive_units):
        contrastive_matrix[p, i] = 1
        contrastive_matrix[s, i] = -1
        if s == N_GEN - 1:
            p = p + 1
            s = p
        s = s + 1
    return contrastive_matrix
