import tensorflow as tf


class TopDownHTMM(tf.keras.Model):

    def __init__(self, n_gen, C, L, M):
        super(TopDownHTMM, self).__init__(self)
        self.n_gen = n_gen
        self.C = C
        self.L = L
        self.M = M

        self.a = tf.Variable(initial_value=tf.random.normal(shape=[n_gen, C, C, L]),
                             trainable=True)
        self.b = tf.Variable(initial_value=tf.random.normal(shape=[n_gen, C, M]),
                             trainable=True)
        self.pi = tf.Variable(initial_value=tf.random.normal(shape=[n_gen, C]),
                              trainable=True)

    def call(self, t, t_limits):
        # Normalizing parameters
        sm_a, sm_b, sm_pi = _softmax_reparam(self.a, self.b, self.pi)

        # Preliminary downward recursion for internal prior computation
        internal_prior = _preliminary_downward(self.n_gen, sm_a, sm_b, sm_pi, t, t_limits, self.C)

        # Upward recursion
        beta_i, beta_il = _upward(self.n_gen, sm_a, sm_b, sm_pi, internal_prior, t, t_limits, self.C)

        # Downward recursion
        eps_i, eps_ijl = _downward(self.n_gen, sm_a, sm_b, sm_pi, internal_prior, beta_i, beta_il, t, t_limits, self.C)

        likelihood = _log_likelihood(t, eps_i, eps_ijl, sm_a, sm_b, sm_pi)

        return likelihood


def _softmax_reparam(a, b, pi):
    sf_a = tf.math.softmax(a, axis=1)
    sf_bi = tf.math.softmax(b, axis=2)
    sf_pi = tf.math.softmax(pi, axis=1)

    return sf_a, sf_bi, sf_pi


def _preliminary_downward(n_gen, a, b, pi, t, t_limits, hidden_states):
    n_levels = tf.shape(t_limits)[0]-1
    t_size = t_limits[-1]
    # ----------------------------------------------------------------- #
    #                PRELIMINARY DOWNWARD RECURSION                     #
    # ----------------------------------------------------------------- #
    internal_prior = tf.zeros([n_gen, t_size-1, hidden_states])  # Initialized without root

    # Root node: takes prior distribution only
    root_slice = tf.expand_dims(pi, 1)
    internal_prior = tf.concat([root_slice, internal_prior], axis=1)

    # Internal nodes: a(node_position) * parent prior distribution
    i = 1
    while i < n_levels:
        parent_names = t[t_limits[i]:t_limits[i+1], 1]
        positions = t[t_limits[i]:t_limits[i+1], 2]

        a_by_position = tf.gather(a, positions, axis=3)
        a_by_position = tf.transpose(a_by_position, [0, 3, 1, 2])
        parents_prior = tf.gather(internal_prior, parent_names, axis=1)
        level_result = a_by_position * tf.expand_dims(parents_prior, axis=3)
        level_result = tf.reduce_sum(level_result, axis=3)

        # Leaving out internal node portion to update
        head = internal_prior[:, :t_limits[i], :]
        tail = internal_prior[:, t_limits[i+1]:, :]
        internal_prior = tf.concat([head, level_result, tail], axis=1)
        i += 1

    return internal_prior


def _upward(n_gen, a, b, pi, internal_prior, t, t_limits, hidden_states):
    n_levels = tf.shape(t_limits)[0]-1
    t_size = t_limits[-1]

    # Upward parameters
    beta_i = tf.TensorArray(dtype=tf.float32, size=t_size,
                            element_shape=tf.TensorShape([n_gen, hidden_states]), clear_after_read=False)
    beta_il = tf.TensorArray(dtype=tf.float32, size=t_size,
                             element_shape=tf.TensorShape([n_gen, hidden_states]), clear_after_read=False)

    # Starting from leaves: beta_i(leaf) = b(leaf) * internal_prior(leaf)
    leaves_labels = t[t_limits[-2]:, 0]

    emission_leaves = tf.gather(b, leaves_labels, axis=2)
    emission_leaves = tf.transpose(emission_leaves, [0, 2, 1])
    prior_leaves = internal_prior[:, t_limits[-2]:, :]
    beta_i_leaves = emission_leaves * prior_leaves

    nu = tf.reduce_sum(beta_i_leaves, axis=2, keepdims=True)

    beta_i_leaves = beta_i_leaves / nu
    beta_i_leaves = tf.transpose(beta_i_leaves, [1, 0, 2])
    beta_i = beta_i.scatter(tf.range(start=t_limits[-2], limit=t_limits[-1]), beta_i_leaves)

    lev = n_levels - 2
    while lev >= 0:
        u = t_limits[lev]
        while u < t_limits[lev+1]:
            children = tf.where(tf.greater(t[u, 3:], 0))
            children = tf.gather_nd(t[u, 3:], children)
            n_children = tf.shape(children)[0]

            # Updating internal prior of the j-th node
            a_children = a[:, :, :, :n_children]

            # Calculating beta_il of the node's children
            beta_i_children = beta_i.gather(children)
            prior_children = tf.gather(internal_prior, children, axis=1)
            a_children = tf.transpose(a_children, [0, 3, 1, 2])
            beta_i_children = tf.transpose(beta_i_children, [1, 0, 2])
            beta_il_children = beta_i_children / prior_children
            beta_il_children = tf.expand_dims(beta_il_children, axis=2) @ a_children
            beta_il_children = tf.squeeze(beta_il_children, axis=2)

            # Calculating beta_i for the entire level
            label_node = t[u, 0]
            emission_node = b[:, :, label_node]
            prior_node = internal_prior[:, u, :]

            aux_product = tf.reduce_prod(beta_il_children, axis=1)
            beta_i_node = aux_product * emission_node * prior_node

            nu_node = tf.reduce_sum(beta_i_node, axis=1, keepdims=True)

            # Normalization of beta_i of the current level
            beta_i_node = beta_i_node / nu_node

            beta_il_children = tf.transpose(beta_il_children, [1, 0, 2])

            beta_i = beta_i.write(u, beta_i_node)
            beta_il = beta_il.scatter(children, beta_il_children)
            u += 1
        lev -= 1
    beta_il = beta_il.write(0, tf.zeros((n_gen, hidden_states)))
    beta_il = tf.transpose(beta_il.stack(), [1, 0, 2])
    beta_i = tf.transpose(beta_i.stack(), [1, 0, 2])

    return beta_i, beta_il

def _downward(n_gen, a, b, pi, internal_prior, beta_i, beta_il, t, t_limits, hidden_states):
    n_levels = tf.shape(t_limits)[0]-1
    t_size = t_limits[-1]

    # Downward parameters
    eps_i = tf.TensorArray(dtype=tf.float32, size=t_size,
                           element_shape=tf.TensorShape([n_gen, hidden_states]), clear_after_read=False)
    eps_ijl = tf.TensorArray(dtype=tf.float32, size=t_size,
                             element_shape=tf.TensorShape([n_gen, hidden_states, hidden_states]), clear_after_read=False)

    # Root node: basis of recursion
    eps_i = eps_i.write(0, beta_i[:, 0, :])
    eps_ijl = eps_ijl.write(0, tf.zeros((n_gen, hidden_states, hidden_states)))
    # Proceeding with internal nodes

    lev = 1
    while lev < n_levels:
        u = t_limits[lev]
        while u < t_limits[lev+1]:
            # --------------------------------- #
            #           eps_t_ijl update        #
            # --------------------------------- #
            # Numerator
            a_node = a[:, :, :, t[u, 2]]
            beta_i_node = beta_i[:, u, :]
            eps_i_pa = eps_i.read(t[u, 1])
            num = tf.expand_dims(beta_i_node, axis=2) * a_node * tf.expand_dims(eps_i_pa, axis=1)

            # Denominator
            prior_node = internal_prior[:, u, :]
            beta_il_node = beta_il[:, u, :]
            den = tf.expand_dims(prior_node, axis=2) @ tf.expand_dims(beta_il_node, axis=1)

            eps_ijl_node = num / den
            # --------------------------------- #
            #            eps_t_i update         #
            # --------------------------------- #
            eps_i_node = tf.reduce_sum(eps_ijl_node, axis=2)
            eps_i_node = eps_i_node / tf.reduce_sum(eps_i_node, axis=1, keepdims=True)

            eps_i = eps_i.write(u, eps_i_node)
            eps_ijl = eps_ijl.write(u, eps_ijl_node)
            u += 1
        lev += 1

    eps_i = tf.transpose(eps_i.stack(), [1, 0, 2])
    eps_ijl = tf.transpose(eps_ijl.stack(), [1, 0, 2, 3])

    return eps_i, eps_ijl


def _log_likelihood(t, eps_i, eps_ijl, a, b, pi):
    # Likelihood pi
    pi_lhood = tf.reduce_sum(eps_i[:, 0, :]*tf.math.log(pi), axis=-1)

    # Likelihood B
    nodes_labels = t[:, 0]
    b_nodes = tf.gather(b, nodes_labels, axis=2)
    b_nodes = tf.transpose(b_nodes, [0, 2, 1])
    b_lhood = eps_i * tf.math.log(b_nodes)
    b_lhood = tf.reduce_sum(b_lhood, axis=[2, 1])

    # Likelihood A
    nodes_pos = t[1:, 2]
    a_by_position = tf.gather(a, nodes_pos, axis=3)
    a_by_position = tf.transpose(a_by_position, [0, 3, 1, 2])
    a_lhood = eps_ijl[:, 1:, :, :] * tf.math.log(a_by_position)
    a_lhood = tf.reduce_sum(a_lhood, axis=[3, 2, 1])

    likelihood = pi_lhood + b_lhood + a_lhood

    return likelihood
