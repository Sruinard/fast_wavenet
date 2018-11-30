from preprocessing import *

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, MinMaxScaler, QuantileTransformer
import tqdm
import tensorflow as tf
import numpy as np



class Model(object):
    """
    args:
        classes_per_feature: list with number of classes per feature
    """

    def __init__(self, sess, n_blocks, n_layers,
                 n_time_steps, n_input_features, classes_per_feature,
                 n_input_features_condition, n_channels_per_layer, logdir='/Users/stefruinard/Desktop/FastWavenet/',
                 condition_flag=True):

        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.n_time_steps = n_time_steps
        self.n_input_features = n_input_features
        self.classes_per_feature = classes_per_feature
        self.n_input_features_condition = n_input_features_condition
        self.n_channels_per_layer = n_channels_per_layer
        self.logdir = logdir
        self.condition_flag = condition_flag

        inputs_engine = tf.placeholder(dtype=tf.float32, shape=[None, n_time_steps, n_input_features])
        inputs_condition = tf.placeholder(dtype=tf.float32, shape=[None, n_time_steps, n_input_features_condition])

        # can be accessed with dict_with_targets['target_feature_{i}']
        self.dict_with_targets = create_target_placeholders(classes_per_feature)

        h = inputs_engine
        h_c = inputs_condition
        hs = []

        # create dilated_conv_network
        for block in range(n_blocks):
            for i in range(n_layers):
                rate = 2 ** i
                name = f'B{block}-L{i}'
                if condition_flag:
                    h = dilated_conv1d(inputs=h, out_channels=self.n_channels_per_layer, name=name,
                                       inputs_condition=h_c, rate=rate)
                    condition_flag = False
                    hs.append(h)
                else:
                    h = dilated_conv1d(out_channels=self.n_channels_per_layer, name=name, inputs=h, rate=rate)
                    hs.append(h)

        # creates a predictor for each feature with the feature's number of classes
        predictors_dict = create_predictor_dict(inputs=h, classes_per_feature=self.classes_per_feature,
                                                n_blocks=self.n_blocks, n_layers=self.n_layers)
        # create_loss_per_feature
        loss_per_feature_dict = compute_loss_per_feature(predictors_dict=predictors_dict,
                                                         target_dict=self.dict_with_targets, n_blocks=self.n_blocks,
                                                         n_layers=self.n_layers)

        # selects all losses and moves them into a list
        loss_scalar = tf.reduce_mean(list(loss_per_feature_dict.values()))

        accuracy_dict = create_accuracy_dict(dict_with_targets=self.dict_with_targets, predictors_dict=predictors_dict,
                                             n_blocks=self.n_blocks, n_layers=self.n_layers)

        optimizer = tf.train.AdamOptimizer(1e-3)
        gradients, variables = zip(*optimizer.compute_gradients(loss_scalar))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimize = optimizer.apply_gradients(zip(gradients, variables))

        # specify_minimizer
        #         optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        #         minimize = optimizer.minimize(loss_scalar)

        self.inputs_engine = inputs_engine
        self.inputs_condition = inputs_condition
        self.predictors_dict = predictors_dict
        self.loss_per_feature_dict = loss_per_feature_dict
        self.loss_scalar = loss_scalar
        self.minimize = optimize  # minimize
        self.accuracy_dict = accuracy_dict

        self.summary, self.file_writer = self.metrics()
        self.saver = tf.train.Saver(max_to_keep=1000000)
        self.sess = sess
        sess.run(tf.global_variables_initializer())

    def _train(self, inputs_engine, inputs_condition, targets):
        feed_dict = to_dict(inputs_engine=inputs_engine,
                            inputs_condition=inputs_condition, targets=targets,
                            input_placeholder=self.inputs_engine, input_placeholder_cond=self.inputs_condition,
                            input_placeholder_targets_dict=self.dict_with_targets)
        cost, opt = self.sess.run([self.loss_scalar, self.minimize], feed_dict=feed_dict)
        return cost, feed_dict

    def train(self, inputs_engine, inputs_condition, targets, n_steps, best_loss,epoch):
        losses = []
        i = 0
        for step in tqdm.tqdm(range(n_steps)):
            i += 1
            cost, feed_dict = self._train(inputs_engine, inputs_condition, targets)
            losses.append(cost)

            if cost < best_loss:
                tqdm.tqdm.write(f'iteration:{epoch} -------- loss:{cost} --------- best_loss:{best_loss}')
                self.saver.save(self.sess, self.logdir + 'B{}L{}/'.format(self.n_blocks, self.n_layers) + 'model.ckpt')
                best_loss = cost

        return losses, best_loss, feed_dict

    def metrics(self):
        with tf.name_scope('scalars'):
            m0 = tf.summary.scalar('V_source_loss', self.loss_per_feature_dict['loss_feature_0'])
            m1 = tf.summary.scalar('I_U_loss', self.loss_per_feature_dict['loss_feature_1'])
            m2 = tf.summary.scalar('I_V_loss', self.loss_per_feature_dict['loss_feature_2'])
            m3 = tf.summary.scalar('I_W_loss', self.loss_per_feature_dict['loss_feature_3'])
            m4 = tf.summary.scalar('sensor_torque_loss', self.loss_per_feature_dict['loss_feature_4'])
            m5 = tf.summary.scalar('encoder_rpm_loss', self.loss_per_feature_dict['loss_feature_5'])
            m6 = tf.summary.scalar('temperature_board_loss', self.loss_per_feature_dict['loss_feature_6'])
            m7 = tf.summary.scalar('total_loss', self.loss_scalar)

        with tf.name_scope('accuracy'):
            names = ['V_source', 'I_U', 'I_V', 'I_W', 'sensor_torque', 'encoder_rpm', 'temperature_board']
            for i, j in enumerate(self.accuracy_dict.keys()):
                name = names[i]
                scalar = tf.summary.scalar(name, self.accuracy_dict[j])

        summary = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter(
            self.logdir + 'Tensorboard/' + 'B{}L{}/'.format(self.n_blocks, self.n_layers), tf.get_default_graph())
        return summary, file_writer

