#File used to load the raw data, filter the data, drop outliers and normalise it


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy


def filt_band(norm_data,  samplefreq=0.01, cutoff_h=5, cutoff_l=0.01, order=1):
    '''
    Butterworth bandpass filter
    '''
    filtacceleration = norm_data.copy()
    b, a = scipy.signal.butter(
        order, [(2*cutoff_l)/(1/samplefreq), (2*cutoff_h)/(1/samplefreq)], 'bandpass')

    for measurement in norm_data.index:
        filtacceleration.loc[measurement] = scipy.signal.filtfilt(
            b, a, norm_data.loc[measurement])
    return filtacceleration


def visualise_df(participant, data):
    # Example of data:
    fig1, axes = plt.subplots()  # , sharey=True
    axes.plot(data.loc[(participant)])


def visualise_arr(participant, data):
    # Example of data:
    fig1, axes = plt.subplots()  # , sharey=True
    axes.plot(data[participant, :512, :])


def correct_format(latent_layer, mse):
    # Create dataframe
    latent_features = latent_layer.shape[1] - 7
    columns = ['Subj', 'T', 'Aid', 'Num', 'Side', 'index', 'tmp']
    squares = [f'L{i}'for i in range(latent_features)]
    columns += squares
    df_latent = pd.DataFrame(latent_layer, columns=columns)
    df_latent = df_latent.join(pd.DataFrame(mse, columns=['MSE']))

    # Make some corrections to the latent
    df_latent.loc[pd.isnull(df_latent['Side']), 'Side'] = df_latent.loc[pd.isnull(
        df_latent['Side']), 'Num']
    df_latent.loc[df_latent['Num'] ==
                  'L.csv', 'Num'] = df_latent.loc[df_latent['Num'] == 'L.csv', 'Aid']
    df_latent.loc[df_latent['Num'] ==
                  'R.csv', 'Num'] = df_latent.loc[df_latent['Num'] == 'R.csv', 'Aid']

    df_latent['condition'] = df_latent['Subj'].astype(str).str[0]
    df_latent['Side'] = df_latent['Side'].replace('R.csv', 'R')
    df_latent['Side'] = df_latent['Side'].replace('L.csv', 'L')
    df_latent.loc[df_latent['condition'] == 'H',
                  'Aid'] = 'Nee'
    df_latent.loc[df_latent['Subj'] == 'H3',
                  'Aid'] = 'Ja'
    df_latent.loc[df_latent['Subj'] == 'H4',
                  'Aid'] = 'Ja'
    df_latent.loc[df_latent['Subj'] == 'H8',
                  'Aid'] = 'Ja'
    df_latent.loc[df_latent['Subj'] == 'H10',
                  'Aid'] = 'Ja'
    df_latent['Aid'] = df_latent['Aid'].replace('with', 'Ja')
    df_latent['Aid'] = df_latent['Aid'].replace('without', 'Nee')
    df_latent.drop(columns=['tmp'], inplace=True)
    df_latent.set_index('index', inplace=True)

    # Add gait speed long and correct pt
    sensor_long = pd.read_excel('Data/Sensor_data_outcomes/Stroke_long.xlsx')
    sensor_long.rename(columns={'Subject number': 'Subj',
                                'T moment': 'T',
                                'aid': 'Aid'}, inplace=True)
    df_latent = df_latent.merge(sensor_long[['Subj', 'T', 'KMPH R', 'Aid']], how='left',
                                on=['Subj', 'T', 'Aid'])
    df_latent.loc[df_latent['Subj'] ==
                  'S5223P', 'Subj'] = 'S6669P'
    df_latent.loc[df_latent['Subj'] ==
                  'S2706H', 'Subj'] = 'S9317H'
    df_latent.loc[df_latent['Subj'] ==
                  'S5649H', 'Subj'] = 'S6567H'
    df_latent['group'] = 'None'
    df_latent.loc[df_latent['Subj'].isin(
        sensor_long['Subj']), 'group'] = 'Longitudinal'

    # Add gait speed test-retest and correct pt
    sensor_test_retest = pd.read_excel(
        'Data/Sensor_data_outcomes/Stroke_test_retest.xlsx')
    sensor_test_retest.rename(columns={'Subject number': 'Subj',
                                       'Test Type': 'T'}, inplace=True)
    df_latent.loc[df_latent['Subj'].isin(
        sensor_test_retest['Subj']), 'group'] = 'Test-retest'
    for idx, row in sensor_test_retest.iterrows():
        df_latent.loc[((df_latent['Subj'] == row['Subj']) &
                       (df_latent['T'] == row['T'])), 'KMPH R'] = row['KMPH R']
        
    # Add gait speed healthy
    sensor_healthy = pd.read_excel(
        '/Users/richard/Desktop/Variational autoencoder/Data/Sensor_data_outcomes/Healthy.xlsx')
    sensor_healthy.rename(columns={'Subjectnumber': 'Subj',
                                       'TestType': 'T'}, inplace=True)
    df_latent.loc[df_latent['Subj'].isin(
        sensor_healthy['Subj']), 'group'] = 'Test-retest'
    for idx, row in sensor_healthy.iterrows():
        df_latent.loc[((df_latent['Subj'] == row['Subj']) &
                       (df_latent['T'] == row['T'])), 'KMPH R'] = row['rightFoot,kmph']


    # split healthy group
    for subj in df_latent.loc[((df_latent['group'] == 'Test-retest') &
                   (df_latent['condition'] == 'H')),'Subj'].unique():
        if int(subj[1:]) < 12:
            df_latent.loc[df_latent['Subj'] ==
                          subj, 'group'] = 'Test-retest Old'
        else:
            df_latent.loc[df_latent['Subj'] == subj,
                          'group'] = 'Test-retest Young'

    df_latent.loc[:, 'L0':f'L{latent_features - 1}'] = df_latent.loc[:, 'L0':f'L{latent_features - 1}'].apply(
        pd.to_numeric, errors='coerce')
    
    return df_latent


def proces_data(path, visualise=False):
    # Settings
    measurements = {}
    columns = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']

    # Load all data into one array
    for file in os.listdir(path):
        if (file.endswith('.csv') and ((file.startswith('S') or file.startswith('H')))):
            try:
                data = pd.read_csv(f'{path}/{file}', index_col=0)
                data = scipy.signal.resample(data, 512)
                if data[:, 0].mean() < 0:
                    data[:, [0, 1, 3, 4]] *= -1
                data = pd.DataFrame(data, columns=columns)
                measurements[file] = data
            except pd.errors.EmptyDataError:
                continue

    # Convert dict to dataframe
    total_data = pd.DataFrame.from_dict(
        {
            (i, j): measurements[i][j]
            for i in measurements
            for j in measurements[i].keys()
        },
        orient='index',
    )

    # Filter data between 0.01 and 10 Hz
    filt_data = filt_band(total_data, samplefreq=0.01, cutoff_h=10,
                          cutoff_l=0.01, order=1)

    # Correct the data for the first sample so everything starts at 0
    norm_data = filt_data.copy()
    for measurement in norm_data.index:
        norm_data.loc[measurement] = filt_data.loc[measurement] - \
            filt_data.loc[measurement, 0]

    # Remove outliers in data
    outliers = np.array([])
    for column in columns:
        tmp_data = norm_data.xs(column, level=1, drop_level=False)
        mean_data = tmp_data.mean(axis=1)
        z_transformed_data_mean = scipy.stats.zscore(mean_data, axis=0)
        std_data = tmp_data.std(axis=1)
        z_transformed_data_std = scipy.stats.zscore(std_data, axis=0)
        outliers_mean = z_transformed_data_mean[(z_transformed_data_mean > 5) | (
            z_transformed_data_mean < -5)].index.get_level_values(0)
        outliers_std = z_transformed_data_std[(z_transformed_data_std > 5) | (
            z_transformed_data_std < -5)].index.get_level_values(0)
        outliers = np.concatenate((outliers, outliers_mean, outliers_std))
    outliers = np.unique(outliers)
    data_no_outliers = norm_data[~norm_data.index.get_level_values(
        0).isin(outliers)]
    data_no_outliers = data_no_outliers.reset_index()
    data_no_outliers[['Subj', 'T', 'Aid', 'num', 'Side']
                     ] = data_no_outliers['level_0'].str.split('_', expand=True)

    # Level 0
    temp_cols = data_no_outliers.columns.tolist()
    new_cols = temp_cols[1:] + temp_cols[:1]
    data_no_outliers = data_no_outliers[new_cols]
    # Level 1
    temp_cols = data_no_outliers.columns.tolist()
    new_cols = temp_cols[1:] + temp_cols[:1]
    data_no_outliers = data_no_outliers[new_cols]
    data_np = data_no_outliers.to_numpy()
    data_np = np.reshape(
        data_np, (int(data_np.shape[0]/6), 6, data_np.shape[1]))
    data_np = np.moveaxis(data_np, 2, +1)

    # Normalise acceleration signals to -1, 1
    data_np[:, :512, :3] = (data_np[:, :512, :3] - data_np[:, :512, :3].min()) / \
        (data_np[:, :512, :3].max() -
            data_np[:, :512, :3].min()) * 2 - 1

    # Normalisation gyroscope signals to -1,1
    data_np[:, :512, 3:] = (data_np[:, :512, 3:] - data_np[:, :512, 3:].min()) / \
        (data_np[:, :512, 3:].max() -
            data_np[:, :512, 3:].min()) * 2 - 1

    # Again set starting point to 0
    for file in range(data_np.shape[0]):
        data_np[file, :512, :] = data_np[file, :512, :] - data_np[file, 0, :]

    # Save data
    np.save('Data/Predicted_data/512.npy', data_np)

    # Visualse data from random subjects
    if visualise:
        random_list = np.random.choice(total_data.index, size=3)
        for i in random_list:
            visualise_df(i, total_data)
            visualise_df(i, filt_data)
            visualise_df(i, norm_data)
