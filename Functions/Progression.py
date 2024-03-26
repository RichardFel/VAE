# %%
# load modules
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

from Functions.Visualise import plot_progression


def find_first_last(df_latent, sensor_long, ontslagen):
    '''
    Find first & last measurement
    '''
    first_last = pd.DataFrame(columns=['Subj'])
    first_last['Subj'] = sensor_long['Subj'].unique()
    sort_cond_1 = sensor_long.sort_values(by='T')
    first = sort_cond_1.drop_duplicates(subset='Subj', keep='first')
    first = first.rename(columns={'T': 'T first'})
    first_last = first_last.merge(first[['Subj', 'T first']], on=[
        'Subj'], how='left')
    sort_cond_1 = sort_cond_1.loc[~sort_cond_1.index.isin(first.index)]
    last = sort_cond_1.drop_duplicates(subset='Subj', keep='last')
    last = last.rename(columns={'T': 'T last'})
    last = last.loc[last['Subj'].isin(ontslagen['Ontslagen'])]
    first_last = first_last.merge(last[['Subj', 'T last']], on=[
        'Subj'], how='left')

    # Add to latent df
    df_latent['Moment'] = 'overig'
    for idx, row in first_last.iterrows():
        df_latent.loc[(df_latent['Subj'] == row['Subj']) & (
            df_latent['T'] == row['T first']), 'Moment'] = 'first'
        df_latent.loc[(df_latent['Subj'] == row['Subj']) & (
            df_latent['T'] == row['T last']), 'Moment'] = 'last'

    return df_latent


def characteristics(df_latent):
    '''
    Store characteristics of the first and last measurement of the longitudinal data
    '''
    icc = pd.read_excel('Results/ICC/ICC.xlsx', index_col=0)
    icc.loc['Gait speed'] = [
        0.963, '0.963 [0.92, 0.98]', 0.137, '0.137 (0.049)']

    results = {}
    part = np.array([])

    df_latent.loc[:,'KMPH R'] /= 3.6
    # Select first and last measuremetns only
    first = df_latent.loc[df_latent['Moment'] == 'first']
    last = df_latent.loc[df_latent['Moment'] == 'last']
    diff = first.merge(last, on=['Subj', 'Aid'],
                       how='left', suffixes=(None, '_last')).dropna(subset = ['L0'])
    first = first.loc[first['Subj'].isin(last['Subj'])]

    # check how many participants improved on gait
    column = 'KMPH R'
    mdc = 0.494 /3.6
    tmp_diff = pd.DataFrame(diff.groupby(
        ['Subj',	'T',	'Aid',	'Side']).mean()).reset_index()
    gait_improv = tmp_diff.iloc[np.where(
        np.abs(tmp_diff[f'{column}_last'] - tmp_diff[f'{column}']) > mdc)[0]]['Subj']

    # check for all other information
    for column in first.columns[4:-2]:
        if column == 'MSE':
            continue    

        if column == 'KMPH R':
            mdc = icc.loc['Gait speed']['MDC']
        elif icc.loc[column]['ICC'] < 0.75:
            continue
        else:
            mdc = icc.loc[column]['MDC']

        mean_f = round(first[column].mean(), 2)
        SD_f = round(first[column].std(), 2)
        mn_f = round(first[column].min(), 2)
        mx_f = round(first[column].max(), 2)
        mean_l = round(last[column].mean(), 2)
        SD_l = round(last[column].std(), 2)
        mn_l = round(last[column].min(), 2)
        mx_l = round(last[column].max(), 2)
        tmp_diff = pd.DataFrame(diff.groupby(
            ['Subj',	'T',	'Aid',	'Side'])[f'{column}', f'{column}_last'].mean()).reset_index()
        increased = np.where(
            tmp_diff[f'{column}_last'] - tmp_diff[f'{column}'] > mdc)[0]
        decreased = np.where(
         tmp_diff[f'{column}'] - tmp_diff[f'{column}_last'] > mdc)[0]
        changed_subj = tmp_diff.iloc[np.where(
            np.abs(tmp_diff[f'{column}_last'] - tmp_diff[f'{column}']) > mdc)[0]]
        unique = len(changed_subj['Subj']) - \
            sum(changed_subj['Subj'].isin(gait_improv))
        if column != 'KMPH R':
            part = np.concatenate((part, changed_subj['Subj'].values))
        results[column] = [f'{mean_f} ({SD_f}) [{mn_f},{mx_f}]',
                           f'{mean_l} ({SD_l}) [{mn_l},{mx_l}]',
                           len(increased), len(decreased), len(changed_subj), unique]
    print(
        f'Improved on gait but not in other metrics: {len(gait_improv) - gait_improv.isin(part).sum()}')
    print(f'Unique participants = {len(np.unique(part))}')
    results = pd.DataFrame.from_dict(
        results, columns=['first', 'last', 'increased', 'decreased','changed','unique'], orient='index')
    results.to_excel('Results/Progression/progression.xlsx')
    print('Results are stored at: Results/Progression/progression.xlsx')
    return diff


def progression(df_latent, per_n_measurement, side = 'R', plotje=True):
    # Load relevant data
    sensor_long = pd.read_excel('Data/Sensor_data_outcomes/Stroke_long.xlsx')
    sensor_long.rename(columns={'Subject number': 'Subj',
                                'T moment': 'T',
                                'aid': 'Aid'}, inplace=True)
    karak_t0 = pd.read_excel(
        'Data/Characteristics/MS_karakteristieken_2023_03_13_compleet.xlsx', sheet_name='T0')
    karak_t0.sort_values(by='ID', inplace=True)
    karak_teind = pd.read_excel(
        'Data/Characteristics/MS_karakteristieken_2023_03_13_compleet.xlsx', sheet_name='Tontslag')
    karak_teind.sort_values(by='ID', inplace=True)
    ontslagen = pd.read_excel('Data/Characteristics/ontslagen.xlsx')

    # Drop faulty data
    df_latent['Num'] = pd.to_numeric(df_latent['Num'], errors='coerce')
    df_latent = df_latent.loc[df_latent['Num'] <= per_n_measurement*4-4]
    df_latent = df_latent.drop(
        columns=['Num', 'condition'])

    # Find first & last measurement
    df_latent = find_first_last(df_latent, sensor_long, ontslagen)

    # Pick right foot of all data
    df_latent = df_latent.loc[df_latent['Side'] == side]

    # details of first measurement
    diff_first_last = characteristics(df_latent)

    if plotje:
        plot_progression(df_latent,
                         diff_first_last, columns=['KMPH R', 'L2', 'L4'])
