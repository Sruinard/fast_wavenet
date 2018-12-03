import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, MinMaxScaler, QuantileTransformer
import tqdm
from sklearn.model_selection import train_test_split
import pickle

from model_prep import *
from preprocessing import *

#
all_files =['/Users/stefruinard/Documents/ML6/DataECC/exp3_013.csv','/Users/stefruinard/Documents/ML6/DataECC/exp3_006.csv', '/Users/stefruinard/Documents/ML6/DataECC/exp3_009.csv']#, '/Users/stefruinard/Documents/ML6/DataECC/exp3_007.csv']#,'/Users/stefruinard/Documents/ML6/DataECC/exp3_010.csv','/Users/stefruinard/Documents/ML6/DataECC/exp3_007.csv','/Users/stefruinard/Documents/ML6/DataECC/exp3_008.csv']#,'/Users/stefruinard/Documents/ML6/DataECC/exp3_009.csv']
df = pd.concat((pd.read_csv(f) for f in all_files))
df.index = np.arange(np.shape(df)[0])
df = df[df.encoder_rpm>-2.5]
print('--- data loaded ---')

n_blocks=3
n_layers=9
n_time_steps = 2**11
test_df = df

test_df = remove_cols(test_df)

chunked_data = data_chunk(test_df, n_time_steps)

tr,val,te = split_train_validation_test(chunked_data)

for i in range(3):
    if i == 0:
        if i == 1:
            concatenated_df = concat_df(val)

            df_test_to_be_transformed, df_switches = separate_switches(concatenated_df)

            df_test_transformed = transformer_transform(df_test_to_be_transformed, transformer_dict=transformer_dict)

            list_with_bins_per_feature = [20, 256, 256, 256, 256, 256, 256]

            # use this for one_hot_fitting
            binned_data = data_to_bins(df_test_transformed, created_bins=created_bins_test)

            # use this data for creating_chunks for one_hot
            data_one_hot_full_batch_including_zero_targets = data_chunk(binned_data, n_time_steps=n_time_steps)

            # use only the part with no padded zero for targets labels
            start_index_without_padded_zeros = 2 ** n_layers * n_blocks + 1  # (2**n_layers*n_blocks+1, add 1 since we are selecting the target data
            data_ready_for_one_hot = [i[start_index_without_padded_zeros:] for i in
                                      data_one_hot_full_batch_including_zero_targets]

            # data ready to save as input_condition
            data_switch_test = data_chunk(df_switches, n_time_steps=n_time_steps)

            # fitted on entire training_data

            inputs_for_saving = create_inputs_engine_and_inputs_condition(
                data_one_hot_full_batch_including_zero_targets,
                data_switch_test)

            targets_for_saving = create_targets_ready_for_saving(data_ready_for_one_hot, fitters_one_hot_test)

            save_batches(inputs_for_saving, targets_for_saving,
                         directory='/Users/stefruinard/Desktop/FastWavenet/validation_data/')
        concatenated_df = concat_df(tr)

        df_test_to_be_transformed, df_switches = separate_switches(concatenated_df)

        transformer_dict = transformer_fitter(df_test_to_be_transformed)

        save_fitters(transformer_dict, '/Users/stefruinard/Desktop/FastWavenet/fitters/', 'transformer_dict')

        df_test_transformed = transformer_transform(df_test_to_be_transformed, transformer_dict=transformer_dict)

        list_with_bins_per_feature = [20,256,256,256,256,256,256]

        bins_per_feature_test = bins_per_feature(df_test_transformed, list_with_bins_per_feature)

        created_bins_test = create_bins(df_test_transformed, bins_per_feature_test)

        #save the created_bins
        #created_bins_test
        save_fitters(created_bins_test, '/Users/stefruinard/Desktop/FastWavenet/fitters/', 'created_bins')

        #use this for one_hot_fitting
        binned_data = data_to_bins(df_test_transformed, created_bins=created_bins_test)

        #use this data for creating_chunks for one_hot
        data_one_hot_full_batch_including_zero_targets = data_chunk(binned_data, n_time_steps=n_time_steps)

        #use only the part with no padded zero for targets labels
        start_index_without_padded_zeros = 2**n_layers*n_blocks+1 #(2**n_layers*n_blocks+1, add 1 since we are selecting the target data
        data_ready_for_one_hot = [i[start_index_without_padded_zeros:] for i in data_one_hot_full_batch_including_zero_targets]

        #data ready to save as input_condition
        data_switch_test = data_chunk(df_switches, n_time_steps=n_time_steps)

        #fitted on entire training_data
        fitters_one_hot_test = one_hot_fitter(binned_data, bins_per_feature=bins_per_feature_test)

        save_fitters(fitters_one_hot_test, '/Users/stefruinard/Desktop/FastWavenet/fitters/', 'one_hot')

        inputs_for_saving = create_inputs_engine_and_inputs_condition(data_one_hot_full_batch_including_zero_targets, data_switch_test)

        targets_for_saving = create_targets_ready_for_saving(data_ready_for_one_hot, fitters_one_hot_test)

        save_batches(inputs_for_saving, targets_for_saving, directory='/Users/stefruinard/Desktop/FastWavenet/training_data/')

    if i == 1:
        concatenated_df = concat_df(val)

        df_test_to_be_transformed, df_switches = separate_switches(concatenated_df)


        df_test_transformed = transformer_transform(df_test_to_be_transformed, transformer_dict=transformer_dict)

        list_with_bins_per_feature = [20, 256, 256, 256, 256, 256, 256]



        # use this for one_hot_fitting
        binned_data = data_to_bins(df_test_transformed, created_bins=created_bins_test)

        # use this data for creating_chunks for one_hot
        data_one_hot_full_batch_including_zero_targets = data_chunk(binned_data, n_time_steps=n_time_steps)

        # use only the part with no padded zero for targets labels
        start_index_without_padded_zeros = 2 ** n_layers * n_blocks + 1  # (2**n_layers*n_blocks+1, add 1 since we are selecting the target data
        data_ready_for_one_hot = [i[start_index_without_padded_zeros:] for i in
                                  data_one_hot_full_batch_including_zero_targets]

        # data ready to save as input_condition
        data_switch_test = data_chunk(df_switches, n_time_steps=n_time_steps)

        # fitted on entire training_data


        inputs_for_saving = create_inputs_engine_and_inputs_condition(data_one_hot_full_batch_including_zero_targets,
                                                                      data_switch_test)

        targets_for_saving = create_targets_ready_for_saving(data_ready_for_one_hot, fitters_one_hot_test)

        save_batches(inputs_for_saving, targets_for_saving,
                     directory='/Users/stefruinard/Desktop/FastWavenet/validation_data/')

    if i == 2:
        concatenated_df = concat_df(te)

        df_test_to_be_transformed, df_switches = separate_switches(concatenated_df)

        df_test_transformed = transformer_transform(df_test_to_be_transformed, transformer_dict=transformer_dict)

        list_with_bins_per_feature = [20, 256, 256, 256, 256, 256, 256]

        # use this for one_hot_fitting
        binned_data = data_to_bins(df_test_transformed, created_bins=created_bins_test)

        # use this data for creating_chunks for one_hot
        data_one_hot_full_batch_including_zero_targets = data_chunk(binned_data, n_time_steps=n_time_steps)

        # use only the part with no padded zero for targets labels
        start_index_without_padded_zeros = 2 ** n_layers * n_blocks + 1  # (2**n_layers*n_blocks+1, add 1 since we are selecting the target data
        data_ready_for_one_hot = [i[start_index_without_padded_zeros:] for i in
                                  data_one_hot_full_batch_including_zero_targets]

        # data ready to save as input_condition
        data_switch_test = data_chunk(df_switches, n_time_steps=n_time_steps)

        # fitted on entire training_data

        inputs_for_saving = create_inputs_engine_and_inputs_condition(data_one_hot_full_batch_including_zero_targets,
                                                                      data_switch_test)

        targets_for_saving = create_targets_ready_for_saving(data_ready_for_one_hot, fitters_one_hot_test)

        save_batches(inputs_for_saving, targets_for_saving,
                     directory='/Users/stefruinard/Desktop/FastWavenet/test_data/')