import numpy as np
import pandas as pd
import pingouin as pg


def calc_icc(df_latent, latent_features, per_n_measurement, side = 'R'):
    # Pick right side only
    df_latent = df_latent.loc[df_latent['Side'] == side]


    # Drop faulty data
    df_latent['Num'] = pd.to_numeric(df_latent['Num'], errors='coerce')

    # Take first 3 measurements (num 0, 4, 8)
    first = df_latent.loc[df_latent['Num'] <= per_n_measurement*4-4]

    # Test-retest data only
    first_test_retest = first.loc[(
        first['T'] == 'Test') | (first['T'] == 'Hertest')]
    first_test_retest['subj_dir'] = first_test_retest['Subj'] + \
        first_test_retest['Side'] + first_test_retest['Aid']
    first_test_retest = first_test_retest.drop(
        columns=['Subj', 'Aid', 'Num', 'Side', 'condition'])

    # Loop over latent features to calculate ICC per feature

    results = {}
    for variable in first_test_retest.columns[1:latent_features+1]:
        # Calculate mean value per T per measurement (test/retest)
        variableDF = first_test_retest[['subj_dir', 'T',  variable]]
        new_df = pd.DataFrame(variableDF.groupby(
            ['subj_dir', 'T'])[variable].mean()).reset_index()
        countValues = new_df['subj_dir'].value_counts()

        # If there are more or less than 2 files drop the participant
        inComplete = countValues.loc[countValues != 2].index
        new_df = new_df.drop(new_df.loc[new_df['subj_dir'].isin(
            inComplete)].index)

        # Calculate ICC
        icc = pg.intraclass_corr(data=new_df, targets='subj_dir', raters='T',
                                 ratings=variable).round(10)

        # Store ICC in dict
        ICC = icc['ICC'].loc[1]
        CI = icc['CI95%'].loc[1]
        SEM = (np.std(variableDF.loc[:, variable]) * np.sqrt(1 - ICC))
        MDC = (1.96 * SEM * np.sqrt(2))
        SEM = SEM.round(3)
        ICC_CI = f'{str(ICC.round(3))} [{str(CI[0])},{str(CI[1])}]'
        MDC_SEM = f'{str(MDC.round(3))} ({str(SEM)})'
        results[variable] = [round(ICC, 3), ICC_CI, round(MDC, 3),  MDC_SEM]

    print(f'Number of subjects included: {len(new_df)/2}')
    # Save to excel
    results = pd.DataFrame.from_dict(
        results, columns=['ICC', 'ICC_[CI]', 'MDC', 'MDC (SEM)'], orient='index')
    results.to_excel('Results//ICC/ICC.xlsx')
    print('ICC values are availible at: Results/ICC/ICC.xlsx')
