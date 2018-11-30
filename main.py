import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, MinMaxScaler, QuantileTransformer
import tqdm
import tensorflow as tf
import numpy as np

from model_prep import *
from preprocessing import *


all_files =['/Users/stefruinard/Documents/ML6/DataECC/exp3_013.csv','/Users/stefruinard/Documents/ML6/DataECC/exp3_006.csv', '/Users/stefruinard/Documents/ML6/DataECC/exp3_009.csv']#, '/Users/stefruinard/Documents/ML6/DataECC/exp3_007.csv']#,'/Users/stefruinard/Documents/ML6/DataECC/exp3_010.csv','/Users/stefruinard/Documents/ML6/DataECC/exp3_007.csv','/Users/stefruinard/Documents/ML6/DataECC/exp3_008.csv']#,'/Users/stefruinard/Documents/ML6/DataECC/exp3_009.csv']
df = pd.concat((pd.read_csv(f) for f in all_files))
df.index = np.arange(np.shape(df)[0])
df = df[df.encoder_rpm>-2.5]
df_to_be_transformed = df.loc[:,['V_source','I_U',"I_V","I_W",'sensor_torque','encoder_rpm','temperature_board']]

print('-------- data loaded ----------')

transformer_dict = transformer_fitter(df_to_be_transformed)

print('----- transformers fitted ------')

df_transformed = transformer_transform(df_to_be_transformed, transformer_dict=transformer_dict)

list_with_bins_per_feature = [20,256,256,256,256,256,256]

bins_per_feature = bins_per_feature(df_transformed, list_with_bins_per_feature)

created_bins = create_bins(df_transformed, bins_per_feature)

y_data = data_to_bins(df_transformed, created_bins=created_bins)

print('----- y_data created ------')

fitters_one_hot = one_hot_fitter(y_data, bins_per_feature=bins_per_feature)

print('----- One hotters fitted ------')

y_data_in_one_hot = onehottransformer(y_data, onehotfitter=fitters_one_hot)

print('----- one hot transformed ------')



n_time_steps = 2**11
data_size = 2**11
n_epochs = 100000

print('----- Variables declared for model ------')

# Use if a model exists and train further

tf.reset_default_graph()
with tf.Session() as sess:
    model = Model(sess, n_blocks=3, n_layers=9, condition_flag=True,
                  classes_per_feature=list_with_bins_per_feature, n_input_features=6, n_input_features_condition=4,
                  n_time_steps=n_time_steps, n_channels_per_layer=32)
    #model.saver.restore(sess, tf.train.latest_checkpoint('/Users/stefruinard/Desktop/FastWavenet/B{}L{}/'.format(model.n_blocks,model.n_layers)))
    print("------ model created ------")

    best_loss = 20
    for epoch in tqdm.tqdm(range(n_epochs)):
        inputs_engine, inputs_condition, targets = create_training_batch(data_size=data_size,
                                                                         df_transformed=y_data, #must fill in y_data otherwise predictions cannot be converted to input
                                                                         df_condition=df,
                                                                         df_target_data_in_one_hot=y_data_in_one_hot,
                                                                         n_time_steps=n_time_steps)

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


