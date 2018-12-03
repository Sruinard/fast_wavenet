import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, MinMaxScaler, QuantileTransformer
import tqdm


from model_prep import *
from preprocessing import *



n_time_steps = 2**11
data_size = 2**11
n_epochs = 100000
list_with_bins_per_feature = [20,256,256,256,256,256,256]

print('----- Variables declared for model ------')

# Use if a model exists and train further

tf.reset_default_graph()
with tf.Session() as sess:
    model = Model(sess, n_blocks=3, n_layers=9, condition_flag=True,
                  classes_per_feature=list_with_bins_per_feature, n_input_features=6, n_input_features_condition=4,
                  n_time_steps=n_time_steps, n_channels_per_layer=32)
    #model.saver.restore(sess, tf.train.latest_checkpoint('/Users/stefruinard/Desktop/FastWavenet/B{}L{}/'.format(model.n_blocks,model.n_layers)))
    print("------ model created ------")

    best_loss = 50
    for epoch in tqdm.tqdm(range(n_epochs)):
        inputs_engine, inputs_condition, targets = create_training_batch(batch_size=32, n_time_steps=n_time_steps, directory='/Users/stefruinard/Desktop/FastWavenet/training_data/')

        losses, best_loss, feed_dict = model.train(inputs_engine=inputs_engine, inputs_condition=inputs_condition,
                                        best_loss=best_loss,
                                        targets=targets, n_steps=1, epoch=epoch)
        #for tensorboard
        # feed_dict = to_dict(input_placeholder=model.inputs_engine, input_placeholder_cond=model.inputs_condition,
        #                     input_placeholder_targets_dict=model.dict_with_targets,
        #                     inputs_condition=inputs_condition, inputs_engine=inputs_engine, targets=targets)



        summary_str = model.summary.eval(feed_dict=feed_dict)
        model.file_writer.add_summary(summary_str, epoch)


        if epoch % 50 == 0:
            check_nan = sess.run(model.loss_per_feature_dict['loss_feature_5'], feed_dict=feed_dict)
            if np.isnan(check_nan):
                predictor_values_with = sess.run(model.predictors_dict, feed_dict=feed_dict)
                print(feed_dict)
                break
        #
        #
