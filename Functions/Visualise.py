
# modules
import matplotlib.ticker as plticker
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
sns.set_style("white")

# this locator puts ticks at regular intervals
loc = plticker.MultipleLocator(base=4.0)

# Settings
plt.rcParams.update({'font.size': 22})
sns.set_style("white")

def histplot_hs(df_latent):
    fig, ax = plt.subplots(figsize = (16,12))
    sns.histplot(data=df_latent, x='MSE', hue='condition', kde=True, ax = ax)
    mylabels = ['Healthy', 'Stroke']
    ax.legend(labels=mylabels)
    fig.savefig('Figures/histogram_h_s.png')

def visualise_decoder(decoder):
    tmp = np.expand_dims(np.zeros(16),0)
    tmp[0][1] -= 2
    reconstructed_data = decoder.predict(tmp)
    epochLength = 512
    sample_freq = 100
    # Define x-axis in time per second
    x_axis = np.arange(epochLength) / sample_freq
    # Top panel original signal, bottom panel reconstructed signal
    fig, ax = plt.subplots(figsize=(12, 8))
    # Top panel
    ax.plot(x_axis, reconstructed_data[0, :epochLength, :])
    leg = ax.legend(['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'],
                    loc='upper center',
                    bbox_to_anchor=(0.5, 1),
                    ncol=6,
                    fancybox=True,
                    fontsize=16)
    ax.set_ylim(-1, 1)
    ax.set_title('Original signal', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)

def visualise_n_latent():
    sns.set_style("ticks")
    test_data = pd.read_excel(
        'Results/Model_test_results/Latent features/n_latent.xlsx')
    test_data[['Type', 'Number']] = test_data['Unnamed: 0'].str.split(
        '_', expand=True)
    test_data.loc[test_data['Type'] == 'train', 'Type'] = 'Training loss'
    test_data.loc[test_data['Type'] == 'test', 'Type'] = 'Validation loss'
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.lineplot(test_data, x='Number', y='loss', hue='Type', ax=ax)
    ax.xaxis.set_major_locator(loc)
    ax.set_xlabel('Number of latent variables', fontsize=26)
    ax.set_ylabel('Loss', fontsize=26)
    fig.tight_layout()
    fig.savefig('Figures/loss.png')

def violin(df_latent, columns):
    icc = pd.read_excel('Results/ICC/ICC.xlsx', index_col=0)
    icc.loc['Gait speed'] = [0.963, '0.963 [0.92, 0.98]', 0, '0.137 (0.049)']


    tmp = df_latent[columns]
    tmp['Gait speed'] = (tmp['Gait speed'] -
                         tmp['Gait speed'].mean())/tmp['Gait speed'].std(ddof=0)
    tmp.drop(columns = ['MSE'], inplace = True)
    tmp = pd.melt(tmp, id_vars=['condition'],
                  value_vars=tmp.columns[:], var_name='Variable')
    tmp.dropna(inplace=True)

    tmp.loc[tmp['condition'] == 'S', 'condition'] = 'Stroke'
    tmp.loc[tmp['condition'] == 'H', 'condition'] = 'Healthy'

    # Mark variables with high icc with a *
    sign = ['L0','L1','L4','L6','L5','L7','L8','L11','Gait speed']
    for idx, row in icc.iterrows():
        if (row['ICC'] >= 0.75) & (idx in sign):
             tmp.loc[tmp['Variable'] == idx, 'Variable'] = idx + '*#'
        elif (row['ICC'] >= 0.75):
            tmp.loc[tmp['Variable'] == idx, 'Variable'] = idx + '*'
        elif (idx in sign):
             tmp.loc[tmp['Variable'] == idx, 'Variable'] = idx + '#'
    
    columns = tmp.Variable.unique()
    tmp.rename(columns={'condition':'Groups'}, inplace=True)
    first_half = columns[:len(columns) // 2]
    second_half = columns[len(columns) // 2:]
    first = tmp.loc[tmp['Variable'].isin(first_half)]
    second = tmp.loc[tmp['Variable'].isin(second_half)]


    fig, ax = plt.subplots(2,figsize=(16, 12))
    sns.violinplot(data=first, y='value', x='Variable',
                   hue='Groups', split=True, ax=ax[0])
    sns.violinplot(data=second, y='value', x='Variable',
                   hue='Groups', split=True, ax=ax[1])
    fig.savefig('Figures/violin.png')
    
def original_reconstructed(data_np, reconstructed_data):
    epochLength = 512
    sample_freq = 100
    file = np.random.randint(data_np.shape[0])

    # Define x-axis in time per second
    x_axis = np.arange(epochLength) / sample_freq

    # Top panel original signal, bottom panel reconstructed signal
    fig, ax = plt.subplots(2, figsize=(12, 8))

    # Top panel
    ax[0].plot(x_axis, data_np[file, :epochLength, :])
    leg = ax[0].legend(['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'],
                       loc='upper center',
                       bbox_to_anchor=(0.5, 1),
                       ncol=6,
                       fancybox=True,
                       fontsize=16)
    ax[0].set_ylim(-1, 1)
    ax[0].set_title('Original signal', fontsize=20)
    ax[0].tick_params(axis='both', which='major', labelsize=16)

    # Bottom panel
    ax[1].plot(x_axis, reconstructed_data[file, :epochLength, :])
    ax[1].set_ylim(-1, 1)
    ax[1].set_title('Reconstructed signal', fontsize=20)
    ax[1].set_xlabel('Time [s]', fontsize=16)
    leg = ax[1].legend(['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'],
                       loc='upper center',
                       bbox_to_anchor=(0.5, 1),
                       ncol=6,
                       fancybox=True,
                       fontsize=16)
    ax[1].tick_params(axis='both', which='major', labelsize=16)
    fig.tight_layout()
    fig.savefig(f'Figures/original_reconstructed_{file}.png', dpi=300)


def plot_progression(df_latent, diff_first_last, columns=None):
    plt.rcParams.update({'font.size': 14})
    sns.set_style("white")
    if columns is None:
        columns = []
    # Create figure
    fig, ax = plt.subplots( len(columns), figsize=(10, 12))
    fig2, ax2 = plt.subplots(len(columns), figsize=(10, 12))

    # Settings and load data
    icc = pd.read_excel('Results/ICC/ICC.xlsx', index_col=0)
    diff_first_last['subj_aid'] = diff_first_last['Subj'] + \
        diff_first_last['Aid']
    df_latent['subj_aid'] = df_latent['Subj'] + df_latent['Aid']

    # Loop over columns
    for num, column in enumerate(columns):
        mdc = 0.494 if column == 'KMPH R' else icc.loc[column]['MDC']
        # Check who changed
        tmp_diff = pd.DataFrame(diff_first_last.groupby(
            ['subj_aid',	'T',	'Side'])[f'{column}', f'{column}_last'].mean()).reset_index()
        increased = tmp_diff.iloc[np.where(
            tmp_diff[f'{column}_last'] - tmp_diff[f'{column}'] > mdc)[0]]
        decreased = tmp_diff.iloc[np.where(
            tmp_diff[f'{column}_last'] - tmp_diff[f'{column}'] < -mdc)[0]]
        tmp = df_latent.loc[df_latent['subj_aid'].isin(tmp_diff['subj_aid'])]
        tmp_start_end = tmp.loc[(tmp['Moment'] == 'first') |
                                (tmp['Moment'] == 'last')]
        tmp_start_end = tmp_start_end.groupby(
            ['subj_aid', 'Moment'])[column].mean().reset_index()
        tmp_start_end['changed'] = 'Equal'
        tmp_start_end.loc[tmp_start_end['subj_aid'].isin(
            increased['subj_aid']), 'changed'] = 'Increased'
        tmp_start_end.loc[tmp_start_end['subj_aid'].isin(
            decreased['subj_aid']), 'changed'] = 'Decreased'

        if column == 'KMPH R':
            tmp_start_end['KMPH R'] /= 3.6

        # plot difference on group level
        sns.lineplot(data=tmp_start_end, y=column, x='Moment',
                     hue='changed', errorbar='ci', ax=ax[num])
        ax[num].set_ylim(tmp_start_end[column].min(),
                            tmp_start_end[column].max())
        handles, labels = ax[num].get_legend_handles_labels()
        labels = [f"Equal (N = {len(tmp_diff) - len(increased) - len(decreased)})",
                  f"Improved (N = {len(increased)})",
                  f"Decreased (N = {len(decreased)})"]
        ax[num].legend(handles, labels)

        # Plot difference individual level
        changed_subj = tmp_start_end.loc[((tmp_start_end['changed']
                                           == 'Increased') | (tmp_start_end['changed']
                                                              == 'Decreased')), 'subj_aid'].unique()
        tmp_recov = df_latent.loc[df_latent['subj_aid'].isin(changed_subj)]
        tmp_recov = tmp_recov.groupby(['subj_aid',	'T',	'Side'])[
            column].mean().reset_index()
        tmp_recov.sort_values('T', inplace=True)

        sns.lineplot(data=tmp_recov, y=column, x='T',
                     hue='subj_aid',  ax=ax2[num])
        ax2[num].legend('')
        fig2.tight_layout()
        fig2.savefig(f'Figures/{column}_individual.png', dpi=300)

    fig.tight_layout()
    fig.savefig(f'Figures/{column}.png', dpi=300)
