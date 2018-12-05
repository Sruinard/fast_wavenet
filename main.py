import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, MinMaxScaler, QuantileTransformer
import tqdm
import argparse


from model import *
from preprocessing import *



n_time_steps = 2**9
list_with_bins_per_feature = [20,150,150,150,150,150,150]

print('----- Variables declared for model ------')

# Use if a model exists and train further

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument(
        '--n-time-steps',
        help='Number of historical data points, depends on n_blocks and n_layers',
        default=2**9,
    )
    PARSER.add_argument(
        '--n-blocks',
        help='Number of blocks in FastWavenet',
        default=3,
        type=int
    )
    PARSER.add_argument(
        '--n-layers',
        help='number of layers per block in fastwavenet',
        default=7,
    )
    PARSER.add_argument(
        '--classes-per-feature',
        help='A list containing the number of classes per feature',
        default=[20, 150, 150, 150, 150, 150, 150],
    )
    PARSER.add_argument(
        '--n-channels-per-layer',
        help='Number of channels per conv layer',
        default=32,
        type=int
    )
    PARSER.add_argument(
        '--n-training-steps',
        help='Number of training steps',
        default=200000,
        type=int
    )
    PARSER.add_argument(
        '--logdir',
        help='Main directory, where the directories files, training_data, validation_data etc are stored',
        default='/Users/stefruinard/Desktop/FastWavenet/',
        type=str
    )


    ARGS = PARSER.parse_args()

    n_time_steps = ARGS.n_time_steps
    n_blocks = ARGS.n_blocks
    n_layers = ARGS.n_layers
    n_classes_per_feature = ARGS.classes_per_feature
    n_channels_per_layer = ARGS.n_channels_per_layer
    n_training_steps = ARGS.n_training_steps
    logdir = ARGS.logdir
    directory_training_data = logdir + 'training_data/'
    directory_val_data = logdir + 'validation_data/'

    tf.reset_default_graph()
    with tf.Session() as sess:
        model = Model(sess, n_blocks=n_blocks, n_layers=n_layers, condition_flag=True,
                      classes_per_feature=n_classes_per_feature, n_input_features=6, n_input_features_condition=4,
                      n_time_steps=n_time_steps, n_channels_per_layer=n_channels_per_layer, logdir=logdir)  #logdir='/Users/stefruinard/Desktop/FastWavenet/'
        #model.saver.restore(sess, tf.train.latest_checkpoint('/Users/stefruinard/Desktop/FastWavenet/B{}L{}/'.format(model.n_blocks,model.n_layers)))
        print("------ model created ------")

        best_loss = 50
        batch_size_training = 1
        for iteration in tqdm.tqdm(range(n_training_steps)):
            inputs_engine, inputs_condition, targets = create_training_batch(batch_size=batch_size_training, n_time_steps=n_time_steps, directory=directory_training_data) #directory='/Users/stefruinard/Desktop/FastWavenet/training_data/'

            cost, opt = model.train(inputs_engine=inputs_engine, inputs_condition=inputs_condition,targets=targets)

            #save to tensorboard
            if iteration % 10 == 0:
                inputs_engine, inputs_condition, targets = create_training_batch(batch_size=4, n_time_steps=n_time_steps, directory=directory_val_data) #directory='/Users/stefruinard/Desktop/FastWavenet/validation_data/'
                best_loss = model.eval(inputs_engine=inputs_engine, inputs_condition=inputs_condition,targets=targets, best_loss=best_loss, iteration=iteration)

            if iteration % 100000 == 0:
                batch_size_training = 4