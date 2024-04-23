# %%
'''
This script is written by Richard Felius to calculate results for the paper 'Mapping gait recovery after stroke with unsupervised deep learning'.

In this study, longitudinal 2-minute walk-test data were collected with an inertial measurement unit from participants after stroke during rehabilitation, 
and test-retest data were collected from participants after stroke and healthy participants.

In this script the following functions are called:
1) Proces data fragments from a 2-minute walk test
2) Train a variational autoencoder
    2.1) Train a definitive model
    2.2) Test several settings
3) Visualise model fit
4) Calculate the ICC values of the latent layer
5) Calculate and visualise progression
6) Print information in LaTeX format


'''

# Modules
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import pandas as pd

from Functions.Proces_data import proces_data, correct_format
from Functions.VAE import test_model, validate_model, definitive_model
from Functions.Visualise import original_reconstructed, violin, visualise_n_latent, histplot_hs
from Functions.ICC import calc_icc
from Functions.Progression import progression
from Functions.To_latex import to_latex
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# General settings
epochLength = 512
path = 'Data/Processed_data/512_2023_03_17'
numberOfColumns = 6
latentFeatures = 12
trainModel = True

# %%
###### proces and save data ######
# Uncomment this to create a new dataset
# proces_data(path, visualise=False)
data_np = np.load('Data/Predicted_data/512.npy', allow_pickle=True)

# Use only data of stroke
data_stroke = data_np[~np.isin(
    data_np[:, 512, 0], np.unique(data_np[:, 512, :])[:32])]

###### test and train model ######
# Settings
activation = 'tanh'
latent_features = 12

# Test loss functions per N latent featues 80/20 split
# test_model(data_stroke, activation='tanh',
#            latent_features=range(1, 30), LSTM=False)
# visualise_n_latent()

# # Validation per 16 latent features 70/20/10 split (10x repeated)
# validate_model(data_stroke, activation=activation,
#                latent_features=latent_features, LSTM=False)
# validation = pd.read_excel('Results/definitive_model/validation.xlsx')
# validation[['Type', 'Number']] = validation['Unnamed: 0'].str.split(
#     '_', expand=True)
# validation = validation.loc[validation['Type'] == 'val']
# print(validation.describe().loc['mean'])
# print(validation.describe().loc['std'])

# create a model based on all data
# definitive_model(data_stroke, activation=activation,
#                  latent_features=latent_features, LSTM=False)

# %%

# Load definitive models
encoder = tf.keras.models.load_model('Models/VAE_512_encoder', compile=False)
decoder = tf.keras.models.load_model('Models/VAE_512_decoder', compile=False)
auto_encoder = tf.keras.models.load_model(
    'Models/VAE_512_autoencoder', compile=False)

# Reconstruct data
latent_layer = encoder.predict(data_np[:, :512, :].astype('float32'))
latent_layer = np.concatenate(
    (data_np[:, epochLength:, 0], latent_layer), axis=1)
# np.save('Data/Predicted_data/512_latent', latent_layer)
reconstructed_data = decoder.predict(latent_layer[:, 7:])

# Calculate error
mse = np.zeros(len(data_np))
# mse_acc = np.zeros(len(data_np))
# mse_gyr  = np.zeros(len(data_np))
for i in range(len(data_np)):
    mse[i] = mean_squared_error(data_np[i, :512, :], reconstructed_data[i])
    # mse_acc[i] = mean_absolute_error(data_np[i, :512, 0:3], reconstructed_data[i,:, 0:3])
    # mse_gyr[i] = mean_absolute_error(data_np[i, :512, 3:6], reconstructed_data[i,:, 3:6])

# print(f'Mean error acceleration {np.mean(mse_acc)*4} std: ({np.std(mse_acc)*4})')
# print(f'Mean error angular velocity {np.mean(mse_gyr)*13} std: ({np.std(mse_gyr)*13})')


# Visualise data
original_reconstructed(data_np, reconstructed_data)

# Correct format latent layer
latent_features = latent_layer.shape[1] - 7
df_latent = correct_format(latent_layer, mse)

# make histplot mse
histplot_hs(df_latent)

# Calculate ICC-values
calc_icc(df_latent, latent_features, side='R', per_n_measurement=50)

# Calculate progression
progression(df_latent, side='R',
            per_n_measurement=50, plotje=False)

# Plot violin 
df_latent.rename(columns={'KMPH R': 'Gait speed'}, inplace=True)
violin(df_latent, columns=df_latent.columns[5:-1].values)

# Print infomration to LaTeX
to_latex(df_latent, per_n_measurement=50)

df_latent.corr().round(2).to_excel('Results/Correlation/correlation.xlsx')


# %%