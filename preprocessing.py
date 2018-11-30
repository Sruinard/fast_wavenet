import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, MinMaxScaler, QuantileTransformer
import tqdm
import tensorflow as tf
import numpy as np


def power_of_two(x):
    x = int(x)
    # First x in the below expression
    # is for the case when x is 0
    return (x and (not (x & (x - 1))))


def time_series_to_batch(inputs, rate):
    """
    If necessary, zero-pads time series data and reshapes by rate
    goal:
        merge rows by rate

    args:
        inputs:
            tensor,
            data from time series,
            shape = (batch_size, length, channels),
            length must be 2**n
        rate: int, exponentially increasing gap between time series
    """
    #add zeros to the 'history' of the time series
    _,length,num_channels = inputs.get_shape().as_list()

    if power_of_two(length) == False:
        raise ValueError("time series' length should be 2**n. Input shape is given by:  (batch_size, length, channels) ")

    #padd_zeros
    padded = tf.pad(inputs, [[0, 0], [rate, 0], [0, 0]])

    #specify new dimensions
    total_time_steps_after_padding = int(length + rate)
    merge_by_rate = int(total_time_steps_after_padding / rate)
    transpose_shape = (1, 0, 2)
    shape = (merge_by_rate, -1, num_channels)

    #reshape into proper dimensions
    transposed = tf.transpose(padded, transpose_shape)
    reshaped = tf.reshape(transposed, shape)
    outputs = tf.transpose(reshaped, transpose_shape)
    return outputs

def batch_to_time_series(inputs, rate):
    shape = tf.shape(inputs)
    batch_size = shape[0] / rate
    width = shape[1]

    out_width = tf.to_int32(width * rate)
    _, _, num_channels = inputs.get_shape().as_list()

    perm = (1, 0, 2)
    new_shape = (out_width, -1, num_channels)  # missing dim: batch_size
    transposed = tf.transpose(inputs, perm)
    reshaped = tf.reshape(transposed, new_shape)
    outputs = tf.transpose(reshaped, perm)

    return outputs


def create_filter(filter_width, in_channels, out_channels, name=""):
    return tf.get_variable(name='filter_weights' + f'{name}',
                           shape=(filter_width, in_channels, out_channels))

def conv1d(inputs, out_channels, filter_width=2, stride=1,
           padding='VALID', activation=tf.nn.leaky_relu, inputs_condition=None, rate=None):
    shape = inputs.get_shape().as_list()
    in_channels = shape[-1]

    conv_filter = create_filter(filter_width, in_channels, out_channels)
    outputs = tf.nn.conv1d(inputs, conv_filter, stride=stride, padding=padding)

    if inputs_condition is not None:
        # get shape of condition
        shape_condition_input = inputs_condition.get_shape().as_list()
        in_channels_conditioned = shape_condition_input[-1]

        # transform to proper timeseries data (i.e. gets padded a zero)
        inputs_condition_ = time_series_to_batch(inputs_condition, rate=rate)

        # create filter which is accessible through name since multiple filters in this variable scope
        conv_filter_condition = create_filter(filter_width, in_channels_conditioned, out_channels, name='condition')
        outputs_condition = tf.nn.conv1d(inputs_condition_, conv_filter_condition, stride=stride, padding=padding)

        outputs = tf.reduce_sum([outputs, outputs_condition], axis=0)

    if activation:
        outputs = activation(outputs)

    return outputs


def dilated_conv1d(inputs, out_channels, rate=1, name=None, filter_width=2, activation=tf.nn.leaky_relu, padding='VALID',
                   inputs_condition=None):
    with tf.variable_scope(name):
        _, width, _ = inputs.get_shape().as_list()
        inputs_ = time_series_to_batch(inputs, rate=rate)
        outputs_ = conv1d(inputs_,
                          out_channels=out_channels,
                          filter_width=filter_width,
                          padding=padding,
                          activation=activation,
                          inputs_condition=inputs_condition, rate=rate)
        outputs = batch_to_time_series(outputs_, rate=rate)

        # Add additional shape information.
        tensor_shape = [tf.Dimension(None),
                        tf.Dimension(width),
                        tf.Dimension(out_channels)]
        outputs.set_shape(tf.TensorShape(tensor_shape))

    return outputs

def create_target_placeholders(classes_per_feature):
    target_placeholder_dict = {}
    for i, classes in enumerate(classes_per_feature):
        with tf.variable_scope(f'target_placeholder_{i}'):
            target_placeholder_dict['target_feature_{}'.format(i)] = tf.stop_gradient(tf.placeholder(dtype=tf.float32, shape=[None,classes]))
    return target_placeholder_dict

def create_predictor_dict(inputs, n_layers,n_blocks, classes_per_feature,activation=None, filter_width=1):
    output_dict = {}
    for i, dim in enumerate(classes_per_feature):
        with tf.variable_scope(f'feature_predictor_{i}'):
            output_dict['output_for_feature_{}'.format(i)] = conv1d(inputs=inputs,out_channels=dim, activation=activation,filter_width=filter_width)[:,(2**n_layers)*n_blocks:,:] #USE ONLY THE PREDICTIONS With no zero padding to perform gradient descent on
    return output_dict

def create_accuracy_dict(predictors_dict, dict_with_targets, n_layers, n_blocks):
    dict_with_acc = {}
    for i,j in enumerate(predictors_dict.keys()):
        target_argmax = tf.argmax(dict_with_targets['target_feature_{}'.format(i)][2**n_layers*n_blocks:],axis=1)
        prediction_argmax = tf.argmax(tf.nn.softmax(predictors_dict['output_for_feature_{}'.format(i)])[0],axis=1)
        correct = tf.cast(tf.equal(target_argmax, prediction_argmax), tf.float32)
        accuracy = tf.reduce_mean(correct)
        dict_with_acc['acc_feature_{}'.format(i)] = accuracy
    return dict_with_acc

def compute_loss_per_feature(predictors_dict, target_dict, n_layers,n_blocks):
    loss_per_feature = {}
    for i in range(len(predictors_dict)):
        labels = target_dict[f'target_feature_{i}'][(2**n_layers)*n_blocks:] #USE ONLY THE PREDICTIONS With no zero padding to perform gradient descent on
        logits = predictors_dict[f'output_for_feature_{i}']
        loss_per_feature[f'loss_feature_{i}'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
    return loss_per_feature

def merge_dicts(dict1, dict2):
    merged_dict = {**dict1, **dict2}
    return merged_dict

def to_dict(inputs_engine=None, inputs_condition=None, targets=None, input_placeholder=None,
            input_placeholder_cond=None, input_placeholder_targets_dict=None):
    """
    Args:
        targets: list of targets with each target having a shape of (n_time_steps, n_labels_for_feature_i)
    """
    temp_feed_dict_inputs = {input_placeholder: inputs_engine, input_placeholder_cond: inputs_condition}

    temp_feed_dict_targets = {}
    for i, placeholder in enumerate(input_placeholder_targets_dict.values()):
        temp_feed_dict_targets[placeholder] = targets[i]

    # merge together the inputs and target_dicts in one feed_dict
    feed_dict = merge_dicts(temp_feed_dict_inputs, temp_feed_dict_targets)

    return feed_dict

def mu_law(df, mu=26):
    return np.sign(df)*((np.log(1+mu*np.abs(df))))/(np.log(1+mu))

def mu_law_inverse(df, mu=26):
    return np.sign(df)*(1/mu)*((1+mu)**np.abs(df)-1)

def transformer_fitter(df_to_be_transformed):
    transformer_dict = {}
    for i in df_to_be_transformed.columns:
        if i == 'sensor_torque':
            transformer_dict[i] = MinMaxScaler((-1,1)).fit(df_to_be_transformed.loc[:,i].values.reshape(-1,1))
        else:
            transformer_dict[i] = PowerTransformer().fit(df_to_be_transformed.loc[:,i].values.reshape(-1,1))
    return transformer_dict

def transformer_transform(df_to_be_transformed, transformer_dict):
    df_transformed = pd.DataFrame()
    for i in df_to_be_transformed.columns:
        transformed_data = pd.DataFrame(transformer_dict[i].transform(df_to_be_transformed.loc[:,i].values.reshape(-1,1)))
        transformed_data.columns = [i]
        df_transformed = pd.concat((df_transformed, transformed_data),axis=1)
    #apply mu-law to sensor_torque for more evenly distributed data
    df_transformed['sensor_torque'] = mu_law(df_transformed['sensor_torque'])
    return df_transformed

def bins_per_feature(df,list_with_bins_per_feature):
    bins_per_feature = {}
    for i,j in enumerate(df.columns):
        bins_per_feature[j] = list_with_bins_per_feature[i]
    return bins_per_feature


def create_bins(df, bins_per_feature):
    bins = {}
    for i in df.columns:
        bins[i] = np.linspace(df[i].min(),df[i].max(), bins_per_feature[i])
    return bins

def data_to_bins(df,created_bins):
    binned_df = pd.DataFrame()
    for i in df.columns:
        transformed_data = pd.DataFrame(np.digitize(df.loc[:,i], created_bins[i]))
        transformed_data.columns = [i]
        binned_df = pd.concat((binned_df, transformed_data),axis=1)
    #indexes in range of 1 through n --> indexes in range of 0 through n-1 (this is suitable for onehotencoder)
    binned_df -=1
    return binned_df


def one_hot_fitter(df, bins_per_feature):
    onehotfitter = {}
    for i in df.columns:
        n_categories = bins_per_feature[i]
        onehotfitter[i] = OneHotEncoder(sparse=False, categories=[range(n_categories)]).fit(df.loc[:,i].values.reshape(-1,1))
    return onehotfitter

def onehottransformer(df, onehotfitter):
    transformed_data = {}
    for i in df.columns:
        transformed_data[i] = onehotfitter[i].transform(df.loc[:,i].values.reshape(-1,1))
    return transformed_data

def create_training_batch(data_size, df_transformed, df_condition, df_target_data_in_one_hot, n_time_steps):
    # NOTE! inputs engine does not include 'sensor_torque'!

    # pd.dataframe --> [:127] is inclusive
    # numpy --> [:127] with 127 exclusive

    """
    Args:
        data_size: 2**i, must be greater than 'n_time_steps' in model
    """
    #   random_starting_integer = np.shape(df_transformed)
    n_rows_y_data = np.shape(df_target_data_in_one_hot['V_source'])[0]
    max_int = n_rows_y_data - n_time_steps + 1
    start_index = np.random.randint(max_int) #np.random.choice([400000, 500000,600000,700000,800000],p=([1/5]*5))#

    inputs_engine = df_transformed.loc[start_index:start_index + n_time_steps - 1,
                    ['V_source', 'I_U', 'I_V', 'I_W', 'encoder_rpm', 'temperature_board']]
    inputs_condition = pd.concat((df_condition.loc[start_index:start_index + n_time_steps - 1, ['switch_U', 'switch_V', 'switch_W']],
                                  df_transformed.loc[start_index:start_index + n_time_steps - 1, 'sensor_torque']), axis=1)
    inputs_engine = inputs_engine.values.reshape(-1, n_time_steps, 6)
    inputs_condition = inputs_condition.values.reshape(-1, n_time_steps, 4)
    targets = [i[start_index + 1:start_index + n_time_steps + 1] for i in df_target_data_in_one_hot.values()]

    return inputs_engine, inputs_condition, targets
