import numpy as np
import pandas as pd
import csv

def load_data(csv_path, random_state):
    '''
    Load a csv file into a pandas dataframe
    
    Parameters:
        - csv_path: Path of the csv file
        - random_state: Random state to ensure the reproducibility
        
    Output:
        - data frame with the data
    '''
    data_frame = pd.read_csv(csv_path)
    # Shuffle and reset the index
    data_frame = data_frame.sample(frac = 1, random_state = random_state).reset_index(drop=True)
    return data_frame


def filter_dataset(data_frame, threshold, fraction, training_split, random_state):
    '''
    Remove steering values close to 0 and replicate them with extreme values
    for a better distribution.
    
    Parameters:
        - data_frame: Data frame to be filtered
        - threshold: Threshold of the values to be removed
        - fraction: Fraction of the values to be removed
        - training_split: Fraction of the dataframe that will
                          be used for training
        - random_state: Random state to ensure the reproducibility
        
    Output:
        - training_data: Training data frame
        - validation_data: Validation data frame
        - data_frame: Data frame of all data
    '''
    # Remove values close to 0
    data_frame = filter_steering(data_frame, threshold, fraction, random_state)

    # Replicate rows having extreme values of steering angles
    data_frame = extend_steering_with_extreme_values(data_frame)
    
    # Build dataframes
    num_rows_training = int(data_frame.shape[0] * training_split)
    training_data = data_frame.loc[0 : num_rows_training - 1]
    validation_data = data_frame.loc[num_rows_training:]
    return training_data, validation_data, data_frame


def filter_steering(data_frame, threshold, fraction, random_state):
    '''
    Remove values whose steering value is less that a given threshold
    
    Parameters:
        - data_frame: Data frame to be filtered
        - threshold: Threshold of the values to be removed
        - fraction: Fraction of the values to be removed
        - random_state: Random state to ensure the reproducibility
    
    Output:
        - data frame filtered
    '''
    
    print ('Number of values less than {} : {}'.format(threshold,
                                                       len(data_frame.loc[data_frame['steering'] < threshold])))
    
    # Build new dataframe, removing only a fraction of the values that met the threshold criteria
    data_frame = data_frame.drop(data_frame[data_frame['steering'] < threshold]
                                 .sample(frac = fraction, random_state = random_state).index)
    
    print('Length of dataframe after droping values: {}'.format(len(data_frame)))
    return data_frame


def extend_steering_with_extreme_values(data_frame):
    '''
    Extend the given steering dataframe with extreme values
    for a better generalization.
    
    Parameters:
        - data_frame: Data frame to be extended
    
    Output:
        - extended data frame for positive and negative steering values
    '''
    # Positive steering
    data_frame_0_10 = data_frame.loc[(data_frame['steering'] >= 0.008) & (data_frame['steering'] <  0.10)]
    data_frame_10_20 = data_frame.loc[(data_frame['steering'] >= 0.10) & (data_frame['steering'] <  0.20)]
    data_frame_20_30 = data_frame.loc[(data_frame['steering'] >= 0.20) & (data_frame['steering'] <  0.30)]
    data_frame_30_40 = data_frame.loc[(data_frame['steering'] >= 0.30) & (data_frame['steering'] <  0.40)]
    data_frame_40_50 = data_frame.loc[(data_frame['steering'] >= 0.40) & (data_frame['steering'] <  0.50)]
    data_frame_50 = data_frame.loc[data_frame['steering'] >= 0.50]

    data_frame = data_frame.append([data_frame_0_10]*3,ignore_index=True)
    data_frame = data_frame.append([data_frame_10_20]*2,ignore_index=True)
    data_frame = data_frame.append([data_frame_20_30]*8,ignore_index=True)
    data_frame = data_frame.append([data_frame_30_40]*6,ignore_index=True)
    data_frame = data_frame.append([data_frame_40_50]*8,ignore_index=True)
    data_frame = data_frame.append([data_frame_50]*60,ignore_index=True)

    # Negative steering
    data_frame_neg_1_10 = data_frame.loc[(data_frame['steering'] <= -0.008) & (data_frame['steering'] >  -0.10)]
    data_frame_neg_10_20 = data_frame.loc[(data_frame['steering'] <= -0.10) & (data_frame['steering'] >  -0.20)]
    data_frame_neg_20_30 = data_frame.loc[(data_frame['steering'] <= -0.20) & (data_frame['steering'] >  -0.30)]
    data_frame_neg_30_40 = data_frame.loc[(data_frame['steering'] <= -0.30) & (data_frame['steering'] >  -0.40)]
    data_frame_neg_40_50 = data_frame.loc[(data_frame['steering'] <= -0.40) & (data_frame['steering'] >  -0.50)]
    data_frame_neg_50 = data_frame.loc[data_frame['steering'] <= -0.50]

    data_frame = data_frame.append([data_frame_neg_1_10] * 8,ignore_index=True)
    data_frame = data_frame.append([data_frame_neg_10_20] * 8,ignore_index=True)
    data_frame = data_frame.append([data_frame_neg_20_30] * 20,ignore_index=True)
    data_frame = data_frame.append([data_frame_neg_30_40] * 20,ignore_index=True)
    data_frame = data_frame.append([data_frame_neg_40_50] * 20,ignore_index=True)
    data_frame = data_frame.append([data_frame_neg_50] * 80,ignore_index=True)
    
    return data_frame