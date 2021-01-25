"""
Implementation of Variational Autoencoder for Fashion MNIST
Supervisior : Minqian Chen

Team Members :
Mukul Agarwal
Vaibhav Tyagi
Vishal A Raheja
Sonakshi Gupta
"""

import os
import keras
import matplotlib
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers
from keras import metrics
from scipy.stats import norm
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.callbacks import TensorBoard

"""encoded_layer representations"""
latent_dim = 2 


def Sampling(args):
    latent_mean, latent_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(latent_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return latent_mean + K.exp(latent_log_sigma) * epsilon


"""input image"""
input = keras.Input(shape=(784,))
# encoded representation of the input

"""Mean Calculation"""
h = layers.Dense(128, activation='relu')(input)
latent_mean = layers.Dense(128, activation='relu')(h)
latent_mean = layers.Dense(32, activation='relu')(latent_mean)
latent_mean = layers.Dense(latent_dim, activation='relu')(latent_mean)

"""Variance Calculation"""
latent_log_sigma = layers.Dense(128, activation='relu')(h)
latent_log_sigma = layers.Dense(32, activation='relu')(latent_log_sigma)
latent_log_sigma = layers.Dense(latent_dim, activation='relu')(latent_log_sigma)

"""Sampling of Latent Space"""
z = layers.Lambda(Sampling)([latent_mean, latent_log_sigma])

"""Init the encoder"""
encoder = keras.Model(input, [latent_mean, latent_log_sigma, z], name='encoder')

"""Init the decoder"""
latent_input  = keras.Input(shape=(latent_dim,), name='z_Sampling')

"""model for decoding"""
decoded = layers.Dense(32, activation='relu')(latent_input)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)

decoder = keras.Model(latent_input, decoded, name='decoder')

"""Init the VAE model"""
outputs = decoder(encoder(input)[2])
vae = keras.Model(input, outputs, name='VAE_MLP')

"""Rec Loss + Kulbeck Divergence Loss Calculation"""
rec_loss = keras.losses.binary_crossentropy(input, outputs)
rec_loss *= 784
Kulbeck_Loss = 1 + latent_log_sigma - K.square(latent_mean) - K.exp(latent_log_sigma)
Kulbeck_Loss = K.sum(Kulbeck_Loss, axis=-1)
Kulbeck_Loss *= -0.5
vae_loss = K.mean(rec_loss + Kulbeck_Loss)
vae.add_loss(vae_loss)

"""Compiling the Model"""
vae.compile(optimizer='adam', loss='binary_crossentropy')

"""Loading the Dataset"""
(x_train_set, y_train_set), (x_test_set, y_test_set)=tf.keras.datasets.fashion_mnist.load_data()

"""Features rescaling"""
x_train_set = x_train_set.astype('float32') / 255.
x_test_set = x_test_set.astype('float32') / 255.
x_train_set = x_train_set.reshape((len(x_train_set), np.prod(x_train_set.shape[1:])))
x_test_set = x_test_set.reshape((len(x_test_set), np.prod(x_test_set.shape[1:])))

"""Training Module"""
for idx in range(1,20):
    vae.fit(x_train_set, x_train_set,
        epochs=5,
        batch_size=1024,
        shuffle=True,
        verbose=2,
        validation_data=(x_test_set, x_test_set))

    vae.save("VAE_2021_jan")

    x_test_set_encoded = encoder.predict(x_test_set)

    plt.figure(figsize=(10, 10))
    plt.scatter(x_test_set_encoded[2][:, 0], x_test_set_encoded[2][:, 1], c=y_test_set)
    plt.colorbar()
    # plt.show()
    plt.savefig('VAE_TRAINED_CLUSTER_epoch_{}.png'.format(idx))

vae.save("VAE_2021_jan")


encoded_layer_imgs = encoder.predict(x_test_set)
decoded_layer_imgs = decoder.predict(encoded_layer_imgs[2])


"""Module for recreation of images from the Latent Space"""
n = 15  # figure with 15x15 fashions
fashion_size = 28
figure = np.zeros((fashion_size * n, fashion_size * n))
# We will sample n points within [-15, 15] standard deviations
x_coordinate = np.linspace(-15, 15, n)
y_coordinate = np.linspace(-15, 15, n)

for i, yi in enumerate(x_coordinate):
    for j, xi in enumerate(y_coordinate):
        z_sample = np.array([[xi, yi]])
        x_decoded_layer = decoder.predict(z_sample)
        fashion = x_decoded_layer[0].reshape(fashion_size, fashion_size)
        figure[i * fashion_size: (i + 1) * fashion_size,
               j * fashion_size: (j + 1) * fashion_size] = fashion

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()

"""
And I have a machine learning joke but I cannot explain it.
"""