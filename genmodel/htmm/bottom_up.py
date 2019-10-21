import tensorflow as tf


class BottomUpHTMM(tf.keras.Model):

    def __init__(self, n_gen, C, L, M):
        super(BottomUpHTMM, self).__init__(self)
        self.n_gen = n_gen
        self.C = C
        self.L = L
        self.M = M

        self.a = tf.Variable(initial_value=tf.random.normal(shape=[n_gen, C, C, L]),
                             trainable=True)
        self.b = tf.Variable(initial_value=tf.random.normal(shape=[n_gen, C, M]),
                             trainable=True)
        self.pi = tf.Variable(initial_value=tf.random.normal(shape=[n_gen, C, L]),
                              trainable=True)
        self.sp = tf.Variable(initial_value=tf.random.normal(shape=[n_gen, L]),
                              trainable=True)

    def call(self, t, t_limits):
        # Normalizing parameters
        sm_a, sm_b, sm_pi, sm_sp = _softmax_reparam(self.a, self.b, self.pi, self.sp)

        # E-Step
        prior_distr, beta_i, beta_il = _reversed_upward(self.n_gen, t, t_limits, sm_a, sm_b, sm_pi, sm_sp, self.C, self.L)
        eps_i, eps_ijl = _reversed_downward(self.n_gen, t, t_limits, sm_a, sm_b, sm_pi, sm_sp, prior_distr, beta_i, beta_il, self.C, self.L)

        # Likelihood
        likelihood = _log_likelihood(t, t_limits, eps_i, eps_ijl, sm_a, sm_b, sm_pi, sm_sp)

        return likelihood


def _softmax_reparam(a, b, pi, sp):
    sf_a = tf.math.softmax(a, axis=1)
    sf_bi = tf.math.softmax(b, axis=2)
    sf_pi = tf.math.softmax(pi, axis=1)
    sf_sp = tf.math.softmax(sp, axis=1)

    return sf_a, sf_bi, sf_pi, sf_sp


def _reversed_upward(n_gen, t, t_limits, a, b, pi, sp, hidden_states, out_degree):
    n_levels = tf.shape(t_limits)[0] - 1
    t_size = t_limits[-1]

    prior_distr = tf.TensorArray(dtype=tf.float32, size=t_size,
                                 element_shape=tf.TensorShape([n_gen, hidden_states]), clear_after_read=False)
    beta_i = tf.TensorArray(dtype=tf.float32, size=t_size,
                            element_shape=tf.TensorShape([n_gen, hidden_states]), clear_after_read=False)
    beta_il = tf.TensorArray(dtype=tf.float32, size=t_size,
                             element_shape=tf.TensorShape([n_gen, hidden_states]), clear_after_read=False)

    # Base case: leaves
    # Prior leaves
    leaves_index = tf.range(start=t_limits[-2], limit=t_limits[-1])
    pos_leaves = t[t_limits[-2]:, 2]
    prior_leaves = tf.gather(pi, pos_leaves, axis=2)
    prior_leaves = tf.transpose(prior_leaves, [2, 0, 1])
    prior_distr = prior_distr.scatter(leaves_index, prior_leaves)

    # Beta_il leaves -> padding with zeros
    beta_il = beta_il.scatter(leaves_index, tf.zeros((tf.shape(leaves_index)[0], n_gen, hidden_states)))

    # Beta_i leaves
    lab_leaves = t[t_limits[-2]:, 0]
    emission_leaves = tf.gather(b, lab_leaves, axis=2)
    emission_leaves = tf.transpose(emission_leaves, [2, 0, 1])

    beta_leaves = prior_leaves * emission_leaves
    beta_leaves = beta_leaves / tf.reduce_sum(beta_leaves, axis=2, keepdims=True)
    beta_i = beta_i.scatter(leaves_index, beta_leaves)

    lev = n_levels - 2
    while lev >= 0:
        u = t_limits[lev]
        while u < t_limits[lev+1]:
            children = t[u, 3:]
            children = children[children>0]
            n_children = tf.shape(children)[0]

            # Updating internal prior of the j-th node
            a_children = a[:, :, :, :n_children]
            sp_children = sp[:, :n_children]
            prior_children = prior_distr.gather(children)
            a_pos_children = a_children * tf.expand_dims(tf.expand_dims(sp_children, axis=1), axis=1)
            prior_children = tf.transpose(prior_children, [1, 2, 0])

            uth_prior = a_pos_children * tf.expand_dims(prior_children, axis=1)
            uth_prior = tf.reduce_sum(uth_prior, axis=[2, 3])

            # Updating beta_il of the j-th node
            beta_i_children = beta_i.gather(children)
            beta_i_children = tf.transpose(beta_i_children, [1, 2, 0])
            beta_il_children = a_pos_children * tf.expand_dims(beta_i_children, axis=1)
            uth_beta_il = tf.reduce_sum(beta_il_children, axis=[2, 3]) / uth_prior

            # Updating beta_i of the j-th node
            uth_emission = b[:, :, t[u, 0]]
            tmp_jth_beta_i = uth_emission * uth_beta_il * uth_prior
            uth_beta_i = tmp_jth_beta_i / tf.reduce_sum(tmp_jth_beta_i, axis=1, keepdims=True)

            prior_distr = prior_distr.write(u, uth_prior)
            beta_i = beta_i.write(u, uth_beta_i)
            beta_il = beta_il.write(u, uth_beta_il)
            u += 1
        lev -= 1

    prior_distr = tf.transpose(prior_distr.stack(), [1, 0, 2])
    beta_i = tf.transpose(beta_i.stack(), [1, 0, 2])
    beta_il = tf.transpose(beta_il.stack(), [1, 0, 2])

    return prior_distr, beta_i, beta_il


def _reversed_downward(n_gen, t, t_limits, a, b, pi, sp, prior_distr, beta_i, beta_il, hidden_states, out_degree):
    n_levels = tf.shape(t_limits)[0] - 1
    t_size = t_limits[-1]

    # Downward parameters
    eps_i = tf.TensorArray(dtype=tf.float32, size=t_size,
                           element_shape=tf.TensorShape([n_gen, hidden_states]), clear_after_read=False)
    eps_ijl = tf.TensorArray(dtype=tf.float32,
                             size=t_size,
                             element_shape=tf.TensorShape([n_gen, hidden_states, hidden_states, out_degree]),
                             clear_after_read=False)

    # Base case: root
    eps_i = eps_i.write(0, beta_i[:, 0, :])

    lev = 1
    while lev < n_levels:
        u = t_limits[lev]
        while u < t_limits[lev+1]:
            children = t[u, 3:]
            children = children[children>0]
            n_children = tf.shape(children)[0]

            # Updating eps_ijl of the u-th node
            # ---- Numerator computation ---- #
            eps_i_parent = eps_i.read(u)
            eps_i_parent = tf.expand_dims(eps_i_parent, axis=2)
            eps_i_parent = tf.expand_dims(eps_i_parent, axis=3)

            beta_i_children = tf.gather(beta_i, children, axis=1)
            beta_i_children = tf.transpose(beta_i_children, [0, 2, 1])
            beta_i_children = tf.expand_dims(beta_i_children, axis=1)

            a_children = a[:, :, :, :n_children]
            sp_children = sp[:, :n_children]
            a_pos_children = a_children * tf.expand_dims(tf.expand_dims(sp_children, axis=1), axis=1)

            eps_beta_a_pos = eps_i_parent * a_pos_children * beta_i_children

            # ---- Denominator computation ---- #
            i_prior = prior_distr[:, u, :]
            i_beta_il = beta_il[:, u, :]
            i_prior_times_beta_il = i_prior * i_beta_il
            i_prior_times_beta_il = tf.expand_dims(i_prior_times_beta_il, axis=2)
            i_prior_times_beta_il = tf.expand_dims(i_prior_times_beta_il, axis=3)

            res = eps_beta_a_pos / i_prior_times_beta_il

            # Updating eps_i of the j-th node's children
            children_eps_i = tf.reduce_sum(res, axis=2)
            children_eps_i = tf.transpose(children_eps_i, [2, 0, 1])

            pad = tf.zeros((n_gen, hidden_states, hidden_states, out_degree-n_children))
            res = tf.concat([res, pad], axis=3)

            eps_i = eps_i.scatter(children, children_eps_i)
            eps_ijl = eps_ijl.write(u, res)
            u += 1
        lev += 1

    eps_ijl = eps_ijl.scatter(tf.range(start=t_limits[-2], limit=t_size),
                              tf.zeros((t_limits[-1]-t_limits[-2], n_gen, hidden_states, hidden_states, out_degree)))
    eps_i = tf.transpose(eps_i.stack(), [1, 0, 2])
    eps_ijl = tf.transpose(eps_ijl.stack(), [1, 0, 2, 3, 4])

    return eps_i, eps_ijl


def _log_likelihood(t, t_limits, eps_i, eps_ijl, a, b, pi, sp):
    # Likelihood A
    tmp_a = tf.expand_dims(a, axis=1)
    a_lhood = tf.reduce_sum(eps_ijl * tf.math.log(tmp_a), axis=[4, 3, 2, 1])

    # Likelihood B
    nodes_labels = t[:, 0]
    b_nodes = tf.gather(b, nodes_labels, axis=2)
    b_nodes = tf.transpose(b_nodes, [0, 2, 1])
    b_lhood = tf.reduce_sum(eps_i * tf.math.log(b_nodes), axis=[2, 1])

    # Likelihood Pi
    pos_leaves = t[t_limits[-2]:, 2]
    prior_leaves = tf.gather(pi, pos_leaves, axis=2)
    eps_i_leaves = eps_i[:, t_limits[-2]:, :]
    pi_leaves = tf.transpose(prior_leaves, [0, 2, 1])
    pi_lhood = tf.reduce_sum(eps_i_leaves * tf.math.log(pi_leaves), axis=[2, 1])

    # Likelihood SP
    eps_l = tf.reduce_sum(eps_ijl, axis=[2, 3])
    tmp_sp = tf.expand_dims(sp, axis=1)
    sp_lhood = tf.reduce_sum(eps_l * tf.math.log(tmp_sp), axis=[2, 1])

    likelihood = a_lhood + b_lhood + pi_lhood + sp_lhood

    return likelihood