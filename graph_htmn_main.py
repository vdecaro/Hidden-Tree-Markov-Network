import tensorflow as tf
import numpy as np
import sys
import os

from data.synth_graphs.parse import parse_and_stats
from data.graph_dataset_preprocessing import to_dict, to_dict_batch

from genmodel.htmm.bottom_up import BottomUpHTMM
from genmodel.htmm.top_down import TopDownHTMM
from graph_htmn.generative_inference import generative_inference
from graph_htmn.recurrent_rdn import RecurrentRDN
from sklearn.model_selection import StratifiedShuffleSplit

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices([], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
        print(e)

dataset, n_bu, n_td, C, batch_size = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), \
                                        int(sys.argv[4]), int(sys.argv[5])

train_data, eval_data, test_data, max_trees, L = parse_and_stats(dataset)

train_feat, train_lab = to_dict(train_data['adj'], train_data['nodes'], L), train_data['lab']
train_feat = to_dict_batch(train_feat, max_trees)

eval_feat, eval_lab = to_dict(eval_data['adj'], eval_data['nodes'], L), eval_data['lab']
eval_feat = to_dict_batch(eval_feat, max_trees)

bu_model = BottomUpHTMM(n_bu, C, L, 5)
td_model = TopDownHTMM(n_td, C, L, 5)
rdn = RecurrentRDN('C', 3, n_bu, n_td, max_trees)

adam_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
loss_mean = tf.keras.metrics.Mean()

accuracy_mean = tf.keras.metrics.Mean()
accuracy = tf.keras.metrics.Accuracy()

train_dataset = tf.data.Dataset.from_tensor_slices((train_feat, train_lab))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
eval_dataset = tf.data.Dataset.from_tensor_slices((eval_feat, eval_lab)).batch(batch_size)


def train_step(batch_features, batch_labels, bu_model, td_model, rdn, adam_opt):
    with tf.GradientTape() as bu_tape:
        bu_likelihood = generative_inference(batch_features, bu_model, max_trees)
        to_div = tf.expand_dims(tf.cast(batch_features['n_trees'], dtype=tf.float32), axis=-1)
        aux_bu_likelihood = tf.reduce_sum(bu_likelihood, axis=1)/to_div
        neg_bu_likelihood = -1 * tf.reduce_mean(aux_bu_likelihood, axis=0)

    with tf.GradientTape() as td_tape:
        td_likelihood = generative_inference(batch_features, td_model, max_trees)
        to_div = tf.expand_dims(tf.cast(batch_features['n_trees'], dtype=tf.float32), axis=-1)
        aux_td_likelihood = tf.reduce_sum(td_likelihood, axis=1)/to_div
        neg_td_likelihood = -1 * tf.reduce_mean(aux_td_likelihood, axis=0)

    with tf.GradientTape() as rdn_tape:
        logits = rdn(bu_likelihood, td_likelihood)
        one_hot = tf.one_hot(batch_labels, 3)
        loss = cce(one_hot, logits)

    bu_grads = bu_tape.gradient(neg_bu_likelihood, bu_model.trainable_weights)
    td_grads =  td_tape.gradient(neg_td_likelihood, td_model.trainable_weights)
    rdn_grads = rdn_tape.gradient(loss, rdn.trainable_weights)

    adam_opt.apply_gradients(zip(bu_grads, bu_model.trainable_weights))
    adam_opt.apply_gradients(zip(td_grads, td_model.trainable_weights))
    adam_opt.apply_gradients(zip(rdn_grads, rdn.trainable_weights))

    return loss


def eval_step(batch_features, batch_labels, bu_model, td_model, rdn):
    bu_likelihood = generative_inference(batch_features, bu_model, max_trees)
    td_likelihood = generative_inference(batch_features, td_model, max_trees)

    logits = rdn(bu_likelihood, td_likelihood)
    loss = cce(batch_labels, logits)

    predictions = tf.argmax(logits, axis=0)
    acc = accuracy(batch_labels, predictions)
    return loss, acc


for epoch in range(100):
    tf.print('Start of epoch %d' % (epoch,))
    for step, (batch_features, batch_labels) in enumerate(train_dataset):
        tf.print("Step", step)
        loss = train_step(batch_features, batch_labels, bu_model, td_model, rdn, adam_opt)

        weight = [batch_labels.shape[0]/batch_size]
        loss_mean.update_state(loss, [weight])

        if step % 10 == 0:
            print("  Loss during step", step, "=", loss_mean.result().numpy())
            loss_mean.reset_states()

    loss_mean.reset_states()
    print('Starting evaluation %d' % (epoch, ))
    for batch_features, batch_labels in eval_dataset:
        loss, acc = eval_step(batch_features, batch_labels, bu_model, td_model, rdn)

        weight = [batch_labels.shape[0]/batch_size]
        accuracy_mean.update_state(acc, [weight])
        loss_mean.update_state(loss, [weight])

    print('Evaluation result:')
    print('     Loss = ', loss_mean.result().numpy())
    print('     Accuracy = ', accuracy_mean.result().numpy())
    accuracy_mean.reset_states()
    loss_mean.reset_states()
