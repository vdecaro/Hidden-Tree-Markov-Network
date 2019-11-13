import numpy as np
import tensorflow as tf
from math import factorial as fact


class RecurrentRDN(tf.keras.Model):
    """
    Implementation of the Recurrent Relative Density Net
    """

    def __init__(self, task, outputs, n_bu, n_td, max_trees):
        super(RecurrentRDN, self).__init__(self)
        if n_bu > 0 and n_td > 0:
            self.bu_bnorm = tf.keras.layers.BatchNormalization()
            self.td_bnorm = tf.keras.layers.BatchNormalization()

        contrastive_units = fact(n_bu+n_td) // (2*fact(n_bu+n_td-2))
        self.rec_contrastive = \
            tf.keras.layers.GRU(units=contrastive_units,
                                kernel_initializer=tf.keras.initializers.constant(contrastive_matrix(n_bu+n_td)),
                                use_bias=False,
                                return_sequences=True)
        self.attention = tf.keras.layers.Dense(units=1,
                                               activation='sigmoid',
                                               use_bias=False)

        self.softmax = tf.keras.layers.Softmax(axis=1)
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

        mask = tf.reduce_any(likelihood > 0, axis=-1)
        contrastive_values = self.rec_contrastive(inputs=likelihood, 
                                                  mask=mask,
                                                  training=False,
                                                  initial_state=self.rec_contrastive.get_initial_state(likelihood)
                                                 )

        attention_values = self.attention(contrastive_values)

        bitmask = tf.where(mask, tf.ones_like(tf.shape(attention_values)), 
                                 tf.zeros_like(tf.shape(attention_values)))
        inf = tf.fill(tf.shape(attention_values), 1e20)
        attention_values += (bitmask-1)*inf

        alpha = self.softmax(axis=1)(attention_values)

        reduce_ = tf.reduce_sum(alpha * contrastive_values, axis=1)
        output = self.out_layer(reduce_)

        return output


def contrastive_matrix(N_GEN):
    contrastive_units = fact(N_GEN) // (2*fact(N_GEN-2))
    contrastive_matrix = np.zeros((N_GEN, 3*contrastive_units), dtype=np.float32)
    for k in range(3):
        p = 0
        s = 1
        for i in range(k*contrastive_units, (k+1)*contrastive_units):
            contrastive_matrix[p, i] = 1
            contrastive_matrix[s, i] = -1
            if s == N_GEN - 1:
                p = p + 1
                s = p
            s = s + 1
    return contrastive_matrix
