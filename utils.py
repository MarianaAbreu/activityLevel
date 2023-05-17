
"""
File created by: Mariana Abreu
Date: 05/05/2023

Utility functions for the project related with sensors and tags data.
"""
import os
from datetime import datetime, timedelta

import hmmlearn.hmm as hmm
import numpy as np
import pandas as pd


def get_sensors(folder_dir, labels = None, time_offset=2):
    """ 
    Extract sensors data from a given folder directory (one device).
    Params: 
    - folder_dir -> path directory to all files
    - labels -> table with excel info
    - time_offset -> hours offset to correct utc time info (default is 2)
    Returns: a dataframe with all sensors data.
    """

    all_sensors = pd.DataFrame()
    
    for filekey in ['GYRO', 'HE_ACC', 'LP_ACC', 'PRESS', 'MAGNETIC']:
        # get filename that contains filekey
        file = [file for file in os.listdir(folder_dir) if filekey in file][0]
        # open file
        data = pd.read_csv(folder_dir + os.sep + file, header=None)
        data.index = data[0]
        # create table with all sensors
        if all_sensors.empty:
            all_sensors = data.drop(columns=[0, 1])
            all_sensors.columns = [filekey + '_' + str(col) for col in data.columns[2:]]
            all_sensors['date'] = data[0].astype('datetime64[ms]') + timedelta(hours=time_offset)
        else:
            all_sensors[[filekey + '_' + str(col) for col in data.columns[2:]]] = data.drop(columns=[0, 1])
            # all_sensors['date'] = data[1]
    # if labels, crop all_sensors data to the beginning and ending of labels
    if len(labels) > 0: 
        all_sensors = all_sensors.loc[all_sensors['date'].between(labels['datetime'].iloc[0], labels['datetime'].iloc[-1])] 

    return all_sensors


def get_sensor_extras(data):
    """ 
    Segment data into 10s windows and extract activity index for lp, and he, and magnetic and pressure means, and gyro means
    Returns: a dataframe with all processed sensors data.
    """
    new_data = pd.DataFrame()
    # compute gyroscope magnitude
    data['GYRO'] = data[['GYRO_2', 'GYRO_3', 'GYRO_4']].apply(lambda x: x**2).sum(axis=1)**0.5
    # compute activity index lp
    activity_index_lp = data.resample(on='date', rule='10S')[['LP_ACC_2', 'LP_ACC_3', 'LP_ACC_4']].var().mean(axis=1)**0.5
    activity_index_lp.columns = ['ACIX_LP']
    # compute activity index he
    activity_index_he = data.resample(on='date', rule='10S')[['HE_ACC_2', 'HE_ACC_3', 'HE_ACC_4']].var().mean(axis=1)**0.5
    activity_index_lp.columns = ['ACIX_HE']
    # get magnetic, pressure and gyro means
    sensors_mean = data.resample(on='date', rule='10S')[['MAGNETIC_2', 'MAGNETIC_3', 'MAGNETIC_4', 'PRESS_2', 'GYRO']].mean()
    # join everything in the same table
    sensors_mean['PRESS_X'] = 0
    sensors_mean['PRESS_X'].iloc[1:] = np.diff(sensors_mean['PRESS_2'])
    new_data = pd.concat([activity_index_lp, activity_index_he, sensors_mean], axis=1)
    
    # not applied: new_data['PRESS_2'] = zscore(new_data['PRESS_2'])
    return new_data


def join_sensor_tags(data, tags, normalize=True):
    """
    Join sensor data with tags data.
    Returns: a dataframe with all sensors and tags data sampled at 10s.
    """

    data['datetime'] = data.index
    tags['datetime'] = tags.index
    # go through all tags and smooth them and join with sensors data
    for tag in tags.columns:
        i = 0
        while i < len(data)-1:
            start = data['datetime'].iloc[i]
            end = data['datetime'].iloc[i+1]
            seg = tags.loc[tags['datetime'].between(start, end)]
            if len(seg) < 1:
                i += 1
            else:
                data.loc[data['datetime'].between(start, end),tag] = seg[tag].mean()
                i += 1

    every_data = data.copy()
    every_data.drop(columns=['datetime'], inplace=True)

    if normalize:
        every_data.replace(np.nan, every_data.min(axis=0), inplace=True)
        every_data = (every_data - every_data.mean(axis=0)) / every_data.std(axis=0) # normalize
    
    return every_data


def get_tags(folder_dir):
    """
    Extract tags data from a given folder directory (one device).
    Returns: a dataframe with all tags data.
    """

    all_tags = pd.DataFrame()
    filekey = 'TAGS'
    files = [file for file in os.listdir(folder_dir) if filekey in file]
    
    for file in files:
        # open file
        data = pd.read_csv(folder_dir + os.sep + file, header=None)
        data.index = data[0] 
        if all_tags.empty:
            data = data.drop(columns=[0])
            data.columns = ['date', file.split('.csv')[0][-4:]]
            all_tags = data.copy()
        else:
            data = data.drop(columns=[0])
            data.columns = ['date', file.split('.csv')[0][-4:]]
            all_tags = all_tags.join(data[file.split('.csv')[0][-4:]], how='outer')
            all_tags.loc[data.index, 'date'] = data['date']
    return all_tags


def get_tag(folder_dir, tag):
    """
    Extract data from one tag only 
    Returns: a dataframe with one tag data.
    """
    all_tags = pd.DataFrame()
    # find all files that contain tag
    files = [file for file in os.listdir(folder_dir) if tag in file]
    for file in files:
        # open file
        data = pd.read_csv(folder_dir + os.sep + file, header=None)
        data.index = data[0]
        data = data.drop(columns=[0, 1])
        data.columns = [file.split('.csv')[0][-4:]]
        # create table
        if all_tags.empty():
            all_tags = data.copy()
        else:
            all_tags.join(data, how='left')
    
    return all_tags


def get_tags_smooth(folder_dir):
    """
    Extract tags data from a given folder directory (one device).
    This function smooths the data by resampling it to 1 second.
    Returns: a dataframe with all tags data.
    """
    all_tags = pd.DataFrame()
    filekey = 'TAGS'
    files = [file for file in os.listdir(folder_dir) if filekey in file]
    
    for file in files:
        data = pd.read_csv(folder_dir + os.sep + file, header=None)
        data.index = pd.to_datetime(data[1], dayfirst=True)
        if all_tags.empty:
            data = data.drop(columns=[0, 1])
            data.columns = [file.split('.csv')[0][-4:]]
            all_tags = data.resample(rule='1S').mean().copy()
        else:
            data = data.drop(columns=[0, 1])
            data.columns = [file.split('.csv')[0][-4:]]
            all_tags = all_tags.join(data.resample(rule='1S').mean().copy(), how='outer')
    return all_tags


def get_labels(folder_dir):
    """
    Get labels annotations from excel
    """
    #labels = utils.get_labels(folder_dir=main_dir + os.sep + 'MCLKM_0304 Annotation file KM.xlsx')
    labels_file = pd.read_excel(folder_dir)
    data_labels = pd.DataFrame()
    # get start and end of acquisition time
    start = datetime.combine(list(labels_file.columns)[-1], labels_file['Realtime On'].iloc[0])
    end = datetime.combine(list(labels_file.columns)[-1], labels_file['Realtime Off'].iloc[-1])
    data_labels['datetime'] = pd.date_range(start, end, freq='1S') 
    data_labels['activity'] = ''
    data_labels['beacon0'] = ''
    data_labels['beacon1'] = ''


    for row in range(len(labels_file)):
        labels_row = labels_file.iloc[row]
        if str(labels_row['Realtime Off']).lower() == 'nan':
            continue 
        if str(labels_row['Realtime On']).lower() == 'nan':
            continue 
        start = datetime.combine(list(labels_file.columns)[-1], labels_row['Realtime On'])
        end = datetime.combine(list(labels_file.columns)[-1], labels_row['Realtime Off'])

        data_labels.loc[data_labels['datetime'].between(start, end), 'activity'] = labels_row['Activity']
        data_labels.loc[data_labels['datetime'].between(start, end), 'beacon0'] = labels_row['Primary Beacon']
        data_labels.loc[data_labels['datetime'].between(start, end), 'beacon1'] = labels_row['Secondary Beacon']

    return data_labels


def model_train(data, columns, covariance_type='diag', n_components=4, n_iter=1000):
    """
    Train a HMM model with the given data and columns from data.
    """
    model = hmm.GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter).fit(data[columns])
    prob, labels = model.decode(data[columns])
    data['label'] = labels

    return data, model


def timeline_activities_predicted(hmm_data, activity_dict, beacon_dict):
    """ Return the begining and ending of each new state and the respective closest beacon
    """
    timeline_df = pd.DataFrame()
    for activity in activity_dict.keys():
        activity_data = hmm_data.loc[hmm_data['label']==activity]
        idx_jump = np.array([0, len(activity_data)-1])
        idx_jump = np.hstack((idx_jump , np.where(np.diff(activity_data.index)!= min(np.diff(activity_data.index)))[0]))
        # print(idx_jump)
        idx_jump = np.unique(sorted(idx_jump))
        for i in range(len(idx_jump)-1):
            beacon_key = activity_data[idx_jump[i]+1:idx_jump[i+1]][beacon_dict.keys()].median().idxmax()
            if str(beacon_key) != 'nan':
                beacon = beacon_dict[beacon_key]
            else:
                beacon = 'no beacon near'
            timeline_df = pd.concat((timeline_df, pd.DataFrame({"Start": activity_data.index[idx_jump[i]+1], "Finish": activity_data.index[idx_jump[i+1]], "Activity": activity_dict[activity], "Beacon": beacon}, index=[0])), ignore_index=True) 
    return timeline_df


def timeline_activities_true(labels, groups):

    """ Return the begining and ending of each new state and the respective closest beacon
    """
    timeline_df = pd.DataFrame()
    labels.index = labels['datetime']
    for activity in set(labels['activity']):
        activity_data = labels.loc[labels['activity']==activity]
        idx_jump = np.array([0, len(activity_data)-1])
        idx_jump = np.hstack((idx_jump , np.where(np.diff(activity_data.index)!= min(np.diff(activity_data.index)))[0]))
        # print(idx_jump)
        group_activity = groups[activity]
        idx_jump = np.unique(sorted(idx_jump))
        for i in range(len(idx_jump)-1):
            start = activity_data.index[idx_jump[i]+1]
            finish = activity_data.index[idx_jump[i+1]]
            beacon = activity_data[start:finish]['beacon0'].value_counts().idxmax()
            timeline_df = pd.concat((timeline_df, pd.DataFrame({"Start": start, "Finish": finish, "Activity": group_activity, "Original Activity": activity, "Beacon": beacon}, index=[0])), ignore_index=True) 
    return timeline_df