import tensorflow as tf


def generative_inference(batch, gen_model, max_trees):
    batch_size = tf.shape(batch['n_trees'])[0]

    likelihood = tf.TensorArray(dtype=tf.float32,
                                size=batch_size,
                                element_shape=tf.TensorShape([None, gen_model.n_gen]))

    for i in tf.range(batch_size):
        n_trees = batch['n_trees'][i]
        trees = batch['trees'][i, :n_trees, :, :]
        levels = batch['levels'][i, :n_trees, :]
        n_levels = batch['n_levels'][i, :n_trees]

        g_likelihood = tf.TensorArray(dtype=tf.float32,
                                      size=n_trees,
                                      element_shape=tf.TensorShape((gen_model.n_gen)), 
                                      clear_after_read=True)

        for j in tf.range(n_trees):
            t_limits = levels[j, :n_levels[j]]
            t = trees[j, :t_limits[-1], :]

            t_likelihood = gen_model(t, t_limits)
            g_likelihood = g_likelihood.write(j, t_likelihood)

        likelihood = likelihood.write(i, tf.pad(g_likelihood.stack(), [[0, max_trees-n_trees], [0, 0]]))

    return likelihood.stack()
