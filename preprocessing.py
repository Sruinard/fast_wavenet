import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, MinMaxScaler, QuantileTransformer
import tqdm
import os
from sklearn.model_selection import train_test_split
import pickle
from collections import OrderedDict

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
    return tf.get_variable(name='filter_weights' + '{}'.format(name),
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
    target_placeholder_dict = OrderedDict()
    for i, classes in enumerate(classes_per_feature):
        with tf.variable_scope('target_placeholder_{}'.format(i)):
            target_placeholder_dict['target_feature_{}'.format(i)] = tf.stop_gradient(tf.placeholder(dtype=tf.float32, shape=[None,classes]))
    return target_placeholder_dict

def create_predictor_dict(inputs, n_layers,n_blocks, classes_per_feature,activation=None, filter_width=1):
    output_dict = OrderedDict()
    for i, dim in enumerate(classes_per_feature):
        with tf.variable_scope('feature_predictor_{}'.format(i)):
            output_dict['output_for_feature_{}'.format(i)] = conv1d(inputs=inputs,out_channels=dim, activation=activation,filter_width=filter_width)[:,(2**n_layers)*n_blocks:,:] #USE ONLY THE PREDICTIONS With no zero padding to perform gradient descent on
    return output_dict

def create_accuracy_dict(predictors_dict, dict_with_targets, list_with_bins_per_feature=[20, 256, 256, 256, 256, 256, 256]):
    dict_with_acc = OrderedDict()

    for i,j in enumerate(predictors_dict.keys()):
        target_argmax = tf.argmax(dict_with_targets['target_feature_{}'.format(i)],axis=1) #[2**n_layers*n_blocks:] not needed anymore as input already keeps only last values
        prediction_argmax = tf.argmax(tf.nn.softmax(tf.reshape(predictors_dict['output_for_feature_{}'.format(i)], [-1,list_with_bins_per_feature[i]])),axis=1)
        correct = tf.cast(tf.equal(target_argmax, prediction_argmax), tf.float32)
        accuracy = tf.reduce_mean(correct)
        dict_with_acc['acc_feature_{}'.format(i)] = accuracy
    return dict_with_acc

def compute_loss_per_feature(predictors_dict, target_dict, list_with_bins_per_feature=[20, 256, 256, 256, 256, 256, 256]):
    loss_per_feature = OrderedDict()
    for i in range(len(predictors_dict)):
        labels = target_dict['target_feature_{}'.format(i)]                 #   [(2**n_layers)*n_blocks:] #USE ONLY THE PREDICTIONS With no zero padding to perform gradient descent on
        logits = tf.reshape(predictors_dict['output_for_feature_{}'.format(i)], [-1,list_with_bins_per_feature[i]])
        loss_per_feature['loss_feature_{}'.format(i)] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
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

    temp_feed_dict_targets = OrderedDict()
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
    transformer_dict = OrderedDict()
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
    bins_per_feature = OrderedDict()
    for i,j in enumerate(df.columns):
        bins_per_feature[j] = list_with_bins_per_feature[i]
    return bins_per_feature


def create_bins(df, bins_per_feature):
    bins = OrderedDict()
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
    onehotfitter = OrderedDict()
    for i in df.columns:
        n_categories = bins_per_feature[i]
        onehotfitter[i] = OneHotEncoder(sparse=False, categories=[range(n_categories)]).fit(df.loc[:,i].values.reshape(-1,1))
    return onehotfitter

def onehottransformer(df, onehotfitter):
    transformed_data = OrderedDict()
    for i in df.columns:
        transformed_data[i] = onehotfitter[i].transform(df.loc[:,i].values.reshape(-1,1))
    return transformed_data

def create_training_batch(batch_size, n_time_steps, directory):
    # NOTE! inputs engine does not include 'sensor_torque'!

    # pd.dataframe --> [:127] is inclusive
    # numpy --> [:127] with 127 exclusive

    """
    Args:
        data_size: 2**i, must be greater than 'n_time_steps' in model
        directory: the folder containing the folders: 'inputs_engine', 'inputs_condition', 'targets'
    """
    #   random_starting_integer = np.shape(df_transformed)



    max_int = len(os.listdir(directory+'inputs_engine/'))
    batch_input_engine = []
    batch_input_condition = []
    batch_targets = []
    for i in range(batch_size):
        index = np.random.randint(max_int) #np.random.choice([400000, 500000,600000,700000,800000],p=([1/5]*5))#

        inputs_engine = load_data(directory+'inputs_engine/', 'batch_{}'.format(index))
        inputs_condition = load_data(directory+'inputs_condition/', 'batch_{}'.format(index))
        targets = load_data(directory+'targets/', 'batch_{}'.format(index))

        batch_input_engine.append(inputs_engine)
        batch_input_condition.append(inputs_condition)
        batch_targets.append(targets)

    batch_input_engine = pd.concat(batch_input_engine)
    batch_input_condition = pd.concat(batch_input_condition)
    batch_targets = create_stacked_data(batch_targets)

    inputs_engine = batch_input_engine.values.reshape(-1, n_time_steps, 6)
    inputs_condition = batch_input_condition.values.reshape(-1, n_time_steps, 4)


    return inputs_engine, inputs_condition, batch_targets







def remove_cols(df, cols_to_keep=['V_source', 'switch_U', 'switch_V', 'switch_W', 'I_U', 'I_V', 'I_W', 'sensor_torque',
                                  'encoder_rpm', 'temperature_board']):
    df = df.loc[:, cols_to_keep]
    return df

def data_chunk(df, n_time_steps):
    n_rows = np.shape(df)[0]
    n_of_splits = n_rows//(n_time_steps+1) #(add 1 for creating the target values later)
    split_indices = np.arange(1,n_of_splits+1)*(n_time_steps+1)
    df_in_chunks = np.split(df, split_indices)[:-1] #don't account for the last chunck as is may not be a full n_time_steps + 1
    return df_in_chunks


def split_train_validation_test(data_chunk, train_size=0.8, val_size=0.1, test_size=0.1):
    'args:'
    'val_size is a percentage of the test_set. e.g. train_size = 0.8 , val_size=0.5 --> test_size = 0.1'
    training_data, temp_testing_data = train_test_split(data_chunk, train_size=train_size)

    percentage_split = val_size / (val_size + test_size)  # if real val_size was used, the split would be 10%, 90%
    validation_data, testing_data = train_test_split(temp_testing_data, train_size=percentage_split)
    return training_data, validation_data, testing_data

def concat_df(data):
    concatenated_data = pd.concat(data)
    return concatenated_data

def separate_switches(concatenated_df, cols_to_be_transformed=['V_source','I_U',"I_V","I_W",'sensor_torque','encoder_rpm','temperature_board'],cols_switches=['switch_U', 'switch_V', 'switch_W']):
    df_to_be_transformed = concatenated_df.loc[:,cols_to_be_transformed]
    df_switches = concatenated_df.loc[:,cols_switches]
    return df_to_be_transformed, df_switches


def save_fitters(fitter_dict, directory, filename):
    # use def *_fitter to create fitter dictionaries

    pkl_filename = directory + filename

    folder = directory
    if not os.path.exists(os.path.dirname(folder)):
        try:
            os.makedirs(os.path.dirname(folder))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(pkl_filename, 'wb') as file:
        pickle.dump(fitter_dict, file)

def create_data_ready_for_one_hot(data_one_hot_full_batch_including_zero_targets, n_layers, n_blocks):
    start_index_without_padded_zeros = 2**n_layers*n_blocks+1 #(2**n_layers*n_blocks+1, add 1 since we are selecting the target data
    data_ready_for_one_hot = [i[start_index_without_padded_zeros:] for i in data_one_hot_full_batch_including_zero_targets]
    return data_ready_for_one_hot


def create_inputs_engine_and_inputs_condition(data_inputs, data_switches):
    # subtract 1 since the batch size is equal to n_time_steps + 1 for the target values
    assert len(data_inputs) == len(data_switches)

    list_with_engine_condition_pair = []
    for i in range(len(data_inputs)):
        inputs_engine = data_inputs[i][:-1].loc[:,
                        ['V_source', 'I_U', 'I_V', 'I_W', 'encoder_rpm', 'temperature_board']]
        # temporarily needed
        inputs_sensor_torque = pd.DataFrame(
            data_inputs[i][:-1].loc[:, ['sensor_torque']])  # .reset_index(drop=True, inplace=True)
        inputs_switches = pd.DataFrame(data_switches[i][:-1])  #

        inputs_condition = pd.concat(
            (inputs_switches.reset_index(drop=True), inputs_sensor_torque.reset_index(drop=True)), axis=1)

        list_with_engine_condition_pair.append((inputs_engine, inputs_condition))
    return list_with_engine_condition_pair


def save_data(my_list, directory, filename):
    pkl_filename = directory + filename

    if not os.path.exists(os.path.dirname(directory)):
        try:
            os.makedirs(os.path.dirname(directory))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(pkl_filename, 'wb') as file:
        pickle.dump(my_list, file)


def create_targets_ready_for_saving(data_ready_for_one_hot, onehotfitter):
    list_with_targets = []
    for i, j in enumerate(data_ready_for_one_hot):
        y_data_in_one_hot_test = onehottransformer(j, onehotfitter=onehotfitter)
        list_with_targets.append(list(y_data_in_one_hot_test.values()))

    return list_with_targets

def save_batches(inputs_for_saving, targets_for_saving, directory):
    assert len(inputs_for_saving) == len(targets_for_saving)

    for i in range(len(inputs_for_saving)):
        inputs_engine = inputs_for_saving[i][0]
        inputs_condition = inputs_for_saving[i][1]
        targets = targets_for_saving[i]
        save_data(inputs_engine, directory + 'inputs_engine/', 'batch_{}'.format(i))
        save_data(inputs_condition, directory + 'inputs_condition/', 'batch_{}'.format(i))
        save_data(targets, directory + 'targets/', 'batch_{}'.format(i))

    return 'all files saved in specified directories'


def load_data(directory, filename):
    pkl_filename = directory + filename

    if not os.path.exists(os.path.dirname(pkl_filename)):
        raise ValueError("File does not exist: {}".format(pkl_filename))

    with open(pkl_filename, 'rb') as file:
        data = pickle.load(file)
    return data

def create_stacked_data(list_with_target_data):
    stacked = []
    n_of_features_to_predict = 7
    for i in range(n_of_features_to_predict):
        created_ready_for_stacking = [feature[i] for feature in list_with_target_data]
        stacked.append(created_ready_for_stacking)
    target_stacked = [np.vstack(feature) for feature in stacked]
    return target_stacked
