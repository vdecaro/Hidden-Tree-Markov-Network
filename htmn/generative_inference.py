import tensorflow as tf


def generative_inference(batch, gen_model):
    batch_size = tf.shape(batch['tree'])[0]
    likelihood = tf.TensorArray(dtype=tf.float32,
                                size=batch_size,
                                element_shape=tf.TensorShape([gen_model.n_gen]))

    for i in tf.range(batch_size):
        t = batch['tree'][i, ...]
        t_limits = batch['limits'][i, ...]
        t_limits = t_limits[t_limits > 0]
        t_limits = tf.concat([tf.constant([0], dtype=tf.int32), tf.cast(t_limits, dtype=tf.int32)], axis=0)
        t = t[:t_limits[-1], ...]

        likelihood = likelihood.write(i, gen_model(t, t_limits))

    return likelihood.stack()
