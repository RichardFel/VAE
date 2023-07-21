import pandas as pd
from scipy import stats

def to_latex(df_latent, per_n_measurement):
    df_latent = df_latent.loc[df_latent['Num'] <= per_n_measurement*4-4]
    df_latent = df_latent.loc[df_latent['Side'] == 'R']

    icc = pd.read_excel('Results/ICC/ICC.xlsx', index_col=0)
    icc.loc['Gait speed'] = [0.963, '0.963 [0.92, 0.98]', 0, '0.137 (0.049)']
    tmp = df_latent.loc[~(df_latent['group'] == 'None')]
    tmp['Gait speed'] /= 3.6
    for idx, row in icc.iterrows():
        new_df = pd.DataFrame(tmp.groupby(
            ['Subj',	'T',	'Aid','condition','group'])[idx].mean()).dropna().reset_index()
        new_df = new_df.loc[(new_df['T'] == 'Test') | (new_df['T'] == 'Hertest')]
        outcomes = round(new_df.groupby('condition')[idx].describe(), 1)
        t_test = stats.ttest_ind(new_df.loc[new_df['condition'] == 'H',
                        idx].values, 
                                 new_df.loc[new_df['condition'] == 'S', idx].values, equal_var=False)
        p_value = round(t_test[1],3)
        if p_value < 0.01:
            p_value = '$<$0.01'
        print(f"{idx} & {row['ICC_[CI]']} & {row['MDC (SEM)']} && "
              f"{outcomes.loc['S']['mean']}({outcomes.loc['S']['std']})[{outcomes.loc['S']['min']}, {outcomes.loc['S']['max']}] &&"
            f"{outcomes.loc['H']['mean']}({outcomes.loc['H']['std']})[{outcomes.loc['H']['min']}, {outcomes.loc['H']['max']}] && "
             
              f"{round(t_test[0],1)} & {p_value}\\\ ")
    print(new_df['condition'].value_counts())

    progression = pd.read_excel(
        'Results/Progression/progression.xlsx', index_col=0)
    for idx, row in progression.iterrows():
        print(
            f'{idx} & {row["first"]} && {row["last"]} && {row["increased"]} && {row["decreased"]} && {row["unique"]}\\\ ')
