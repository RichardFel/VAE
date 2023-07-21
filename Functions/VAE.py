# Load modules
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import LeakyReLU
import keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import TerminateOnNaN
from tensorflow import keras  # for building Neural Networks
from tensorflow.keras.utils import plot_model  # for plotting model diagram

# General settings
tf.compat.v1.disable_eager_execution()


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def get_loss(distribution_mean, distribution_variance):

    def get_reconstruction_loss(y_true, y_pred):
        reconstruction_loss = tf.keras.losses.mse(y_true, y_pred)
        reconstruction_loss_batch = tf.reduce_mean(reconstruction_loss)
        return reconstruction_loss_batch*28*28

    def get_kl_loss(distribution_mean, distribution_variance):
        kl_loss = 1 + distribution_variance - \
            tf.square(distribution_mean) - tf.exp(distribution_variance)
        kl_loss_batch = tf.reduce_mean(kl_loss)
        return kl_loss_batch*(-0.5)

    def total_loss(y_true, y_pred):
        reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
        kl_loss_batch = get_kl_loss(distribution_mean, distribution_variance)
        # print('print recon loss: ' + str(reconstruction_loss_batch) + 'print kl loss:' + str(kl_loss_batch))
        # pickle.dump(test, locals())
        # f = open('testing123.pckl', 'wb')
        # pickle.dump(reconstruction_loss_batch, f)
        # f.close()
        return reconstruction_loss_batch + kl_loss_batch

    return total_loss


def create_encoder(input_data, latentFeatures, activation, LSTM=False,
                   kernel_size=19, strides=2, print=False):
    '''
    activation = LeakyReLU(alpha=0.2)
    filter_size = 32
    kernel_size = 19
    strides = 2
    '''

    # Convolutional layers
    # Layer 1
    encoder = layers.Conv1D(32, 3, activation=activation,
                            strides=2, name='Layer_1', padding='same')(input_data)
    layer_1 = K.int_shape(encoder)
    # encoder = layers.BatchNormalization(name='bn_1')(encoder)

    # Layer 2
    encoder = layers.Conv1D(64, 3, activation=activation,
                            strides=strides, name='Layer_2', padding='same')(encoder)
    layer_2 = K.int_shape(encoder)
    # encoder = layers.BatchNormalization(name='bn_2')(encoder)

    # Layer 3
    encoder = layers.Conv1D(128, 3, activation=activation,
                            strides=strides, name='Layer_3', padding='same')(encoder)
    layer_3 = K.int_shape(encoder)
    # encoder = layers.BatchNormalization(name='bn_3')(encoder)

    # # LSTM layer
    if LSTM:
        encoder = layers.LSTM(100, activation=activation,
                              name='Layer_4', return_sequences=True)(encoder)
        layer_4 = K.int_shape(encoder)

    # Flatten
    encoder = layers.Flatten(name='Flatten')(encoder)

    # Dense layers
    distribution_mean = layers.Dense(latentFeatures, name='mean')(encoder)
    distribution_variance = layers.Dense(
        latentFeatures, name='log_variance')(encoder)
    dist = Sampling()([distribution_mean, distribution_variance])
    encoder_model = Model(input_data, dist, name="encoder")
    if print:
        encoder_model.summary()
    if LSTM:
        return encoder_model, [layer_1, layer_2, layer_3, layer_4], [distribution_mean, distribution_variance]
    else:
        return encoder_model, [layer_1, layer_2, layer_3], [distribution_mean, distribution_variance]


def create_decoder(latentFeatures, volume_size, def_shape, activation, LSTM=False, filter_size=32, strides=2, print=False):
    '''
    activation = LeakyReLU(alpha=0.2)
    filter_size = 32
    kernel_size = 19
    strides = 2
    def_shape = 3
    '''
    decoder_input = layers.Input(shape=(latentFeatures))

    if LSTM:
        decoder = layers.Dense(
            volume_size[3][2] * volume_size[3][1])(decoder_input)
        decoder = layers.Reshape(
            (volume_size[3][1], volume_size[3][2]))(decoder)
        decoder = layers.LSTM(100, activation=activation,
                              name='Layer_0', return_sequences=True)(decoder)
    else:
        decoder = layers.Dense(
            volume_size[2][2] * volume_size[2][1])(decoder_input)
        decoder = layers.Reshape(
            (volume_size[2][1], volume_size[2][2]))(decoder)

    # Layer 1
    decoder = layers.Conv1DTranspose(
        64, 3, activation=activation, strides=2, name='Layer_1', padding='same')(decoder)
    # decoder = layers.BatchNormalization(name='bn_1')(decoder)

    # Layer 2
    decoder = layers.Conv1DTranspose(
        32, 3, activation=activation, strides=strides, name='Layer_2', padding='same')(decoder)
    # decoder = layers.BatchNormalization(name='bn_2')(decoder)

    # Layer 3
    decoder_output = layers.Conv1DTranspose(
        128, 3, activation=activation, strides=strides, name='Layer_3', padding='same')(decoder)
    # decoder = layers.BatchNormalization(name='bn_3')(decoder)

    decoder_output = layers.Conv1DTranspose(
        def_shape, 3, activation=activation, strides=strides, name='Layer_4', padding='same')(decoder)

    decoder = Model(decoder_input, decoder_output, name="decoder")
    if print:
        decoder.summary()

    return decoder


def compile_model(epochLength, numberOfColumns, activation, latent, train_data, LSTM):
    input_layer = tf.keras.layers.Input(shape=(
        epochLength, numberOfColumns), name='Input_layer')

    # Creat encoder
    encoder, volume_size, dist_list = create_encoder(
        input_layer, latent, activation=activation, print=True, LSTM=LSTM)

    # Create decoder
    decoder = create_decoder(
        latent, volume_size, train_data.shape[2], activation=activation, print=True, LSTM=LSTM)

    # Merge these together to get an autoencoder
    autoencoder = tf.keras.Model(input_layer, decoder(encoder(input_layer)),
                                 name="autoencoder")

    autoencoder.compile(loss=get_loss(
        dist_list[0], dist_list[1]), optimizer='adam', metrics=[
        tf.keras.metrics.MeanAbsoluteError(),
        tf.keras.metrics.RootMeanSquaredError(),
        tf.keras.metrics.KLDivergence(),
        tf.keras.metrics.MeanSquaredError()
    ])
    return autoencoder, encoder, decoder


def store_results(results_dict, history, i):
    loss = round(history.history["loss"][-1], 5)
    rmse_1 = round(
        history.history["root_mean_squared_error"][-1], 5)
    mae = round(history.history["mean_absolute_error"][-1], 5)
    kld = round(
        history.history["kullback_leibler_divergence"][-1], 5)
    mse = round(history.history["mean_squared_error"][-1], 5)
    val_loss = round(history.history["val_loss"][-1], 5)
    val_rmse_1 = round(
        history.history["val_root_mean_squared_error"][-1], 5)
    val_mae = round(history.history["val_mean_absolute_error"][-1], 5)
    val_kld = round(
        history.history["val_kullback_leibler_divergence"][-1], 5)
    val_mse = round(history.history["val_mean_squared_error"][-1], 5)
    results_dict[f'train_{i}'] = [loss, rmse_1, mae, kld, mse]
    results_dict[f'test_{i}'] = [val_loss, val_rmse_1,
                                 val_mae, val_kld, val_mse]
    return results_dict


def split_data(data, test=0.2):
    unique_pt = np.unique(data[:, 512, :])
    pt_test = np.random.choice(
        unique_pt, size=int(len(unique_pt) * test))
    train_data = data[~np.isin(data[:, 512, 0], (pt_test)), :, :]
    test_data = data[
        np.isin(data[:, 512, 0], pt_test), :, :
    ]
    return train_data, test_data


def test_model(data_np, activation, latent_features, LSTM):
    # Settings
    epochLength = 512
    numberOfColumns = 6
    results_dict = {}

    # Split participants 80% / 20%
    train_data, test_data = split_data(data_np, test=0.2)
    train_data.shape

    # Loop over number of features
    for latent in latent_features:
        # Compile the model
        autoencoder, _, _ = compile_model(
            epochLength, numberOfColumns, activation, latent, train_data, LSTM=LSTM)

        # Fit the data, record in history
        history = autoencoder.fit(x=train_data[:, : epochLength, :],
                                  y=train_data[:, : epochLength, :],
                                  epochs=100,
                                  batch_size=64,
                                  callbacks=[tf.keras.callbacks.EarlyStopping(
                                      monitor='loss', patience=3), TerminateOnNaN()],
                                  validation_data=[test_data[:, : epochLength, :],
                                                   test_data[:, : epochLength, :]],
                                  verbose=False)

        # Stores data in a dict
        results_dict = store_results(results_dict, history, latent)

    # Saves data to an excel file
    pd.DataFrame.from_dict(results_dict, orient='index', columns=[
        'rmse_1', 'mae', 'kld', 'mse', 'loss']).to_excel(
        'Results/Model_test_results/Latent features/n_latent.xlsx')
    print('Results saved at: Results/Model_test_results/Latent features/n_latent.xlsx')


def validate_model(data_np, activation, latent_features, LSTM):
    # Settings
    epochLength = 512
    numberOfColumns = 6
    latent = latent_features
    results_dict = {}
    unique_pt = np.unique(data_np[:, 512, :])
    np.random.shuffle(unique_pt)
    groups = np.array_split(unique_pt, 10)

    # Loop over number of groups
    for count, group in enumerate(groups):
        validation_data = data_np[
            np.isin(data_np[:, 512, 0], group), :, :
        ]
        train_test_data = data_np[~np.isin(
            data_np[:, 512, 0], (group)), :, :]

        # Split participants 80% / 20%
        # Split participants 75% / 25%
        train_data, test_data = split_data(train_test_data, 0.25)

        # Compile the model
        autoencoder, _, _ = compile_model(
            epochLength, numberOfColumns, activation, latent, train_data, LSTM=LSTM)

        # Fit the data, record in history
        history = autoencoder.fit(x=train_data[:, : epochLength, :],
                                  y=train_data[:, : epochLength, :],
                                  epochs=100,
                                  batch_size=64,
                                  callbacks=[tf.keras.callbacks.EarlyStopping(
                                      monitor='loss', patience=3), TerminateOnNaN()],
                                  validation_data=[test_data[:, : epochLength, :],
                                                   test_data[:, : epochLength, :]],
                                  verbose=False)

        val_metrics = np.round(autoencoder.evaluate(validation_data[:, :epochLength, :],
                                                    validation_data[:,
                                                                    :epochLength, :],
                                                    batch_size=64), 5)

        # Stores data in a dict
        results_dict = store_results(results_dict, history, count)
        results_dict[f'val_{count}'] = val_metrics

    # Saves data to an excel file
    pd.DataFrame.from_dict(results_dict, orient='index', columns=[
        'loss', 'rmse_1', 'mae', 'kld', 'mse']).to_excel(
        'Results/definitive_model/validation.xlsx')
    print('Results saved at: Results/definitive_model/validation.xlsx')


def definitive_model(data_np,  activation, latent_features, LSTM):
    # Settings
    epochLength = 512
    numberOfColumns = 6
    latent = latent_features

    # Create model with all data
    autoencoder, encoder, decoder = compile_model(
        epochLength, numberOfColumns, activation, latent, data_np, LSTM=LSTM)

    # Create model
    history = autoencoder.fit(x=data_np[:, : epochLength, :],
                              y=data_np[:, : epochLength, :],
                              epochs=100,
                              batch_size=64,
                              callbacks=[tf.keras.callbacks.EarlyStopping(
                                  monitor='loss', patience=3), TerminateOnNaN()],
                              verbose=True)

    autoencoder.save('Models/VAE_512_autoencoder')
    encoder.save('Models/VAE_512_encoder')
    decoder.save('Models/VAE_512_decoder')
    print('Models saved at: Models')
