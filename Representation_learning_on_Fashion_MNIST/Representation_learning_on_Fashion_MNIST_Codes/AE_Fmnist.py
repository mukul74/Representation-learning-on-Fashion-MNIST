"""
Implementation of Basic Autoencoder for Fashion MNIST
Supervisior : Minqian Chen

Team Members :
Mukul Agarwal
Vaibhav Tyagi
Vishal A Raheja
Sonakshi Gupta
"""


import keras
import numpy as np
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

"""Loading the Dataset"""
(x_train_set, y_train_set), (x_test_set, y_test_set)=tf.keras.datasets.fashion_mnist.load_data()

"""encoded_layer representations"""
latent_dim = 2

"""input image"""
input = keras.Input(shape=(784,))


"""Encoder module"""
encoded_layer = layers.Dense(128, activation='relu')(input)
encoded_layer = layers.Dense(32, activation='relu')(input)
encoded_layer = layers.Dense(latent_dim, activation='relu')(encoded_layer)

"""Decoder module"""
decoded_layer = layers.Dense(32, activation='relu')(encoded_layer)
decoded_layer = layers.Dense(128, activation='relu')(decoded_layer)
decoded_layer = layers.Dense(784, activation='sigmoid')(decoded_layer)

"""Initilization"""
base_autoencoder = keras.Model(input, decoded_layer)

encoder = keras.Model(input, encoded_layer)

encoded_layer_input = keras.Input(shape=(latent_dim,))
decoder_layer_1 = base_autoencoder.layers[3]
decoder_layer_2 = base_autoencoder.layers[4]
decoder_layer_3 = base_autoencoder.layers[5]

"""Decoder Model"""
decoder = keras.Model(encoded_layer_input, decoder_layer_3(decoder_layer_2(decoder_layer_1(encoded_layer_input))))

"""Compile The model and init the loss"""
base_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

"""Features rescaling"""
x_train_set = x_train_set.astype('float32') / 255.
x_test_set = x_test_set.astype('float32') / 255.
x_train_set = x_train_set.reshape((len(x_train_set), np.prod(x_train_set.shape[1:])))
x_test_set = x_test_set.reshape((len(x_test_set), np.prod(x_test_set.shape[1:])))
print(x_train_set.shape)
print(x_test_set.shape)

encoded_layer_imgs = encoder.predict(x_test_set)
decoded_layer_imgs = decoder.predict(encoded_layer_imgs)
# base_autoencoder.save("AE_2021_jan")

"""Image presentation for untrained Model"""
plt.figure(figsize=(10, 10))
plt.scatter(encoded_layer_imgs[:, 0], encoded_layer_imgs[:, 1], c=y_test_set, cmap='brg')
plt.colorbar()
plt.show()
plt.savefig('UNTRAINED_CLUSTER.png')

"""Training Module"""

for idx in range(1,20):
    base_autoencoder.fit(x_train_set, x_train_set,
                epochs=5,
                batch_size=1024,
                shuffle=True,
                verbose=2,
                validation_data=(x_test_set, x_test_set),
                callbacks=[TensorBoard(log_dir='/tmp/base_autoencoder')])

    """Saving the intermidiate model"""
    base_autoencoder.save("AE_2021_jan")

    """Prediction after Trainig"""
    encoded_layer_imgs = encoder.predict(x_test_set)
    decoded_layer_imgs = decoder.predict(encoded_layer_imgs)

    plt.figure(figsize=(10, 10))
    plt.scatter(encoded_layer_imgs[:, 0], encoded_layer_imgs[:, 1], c=y_test_set, cmap='brg')
    plt.colorbar()
    # plt.show()
    plt.savefig('TRAINED_CLUSTER_epoch_{}.png'.format(idx))

plt.close('all')

"""Loading the Model fopr inference"""
reconstructed_model = tf.keras.models.load_model('AE_2021_jan')
dec_01 = reconstructed_model.predict(x_test_set)
plt.figure()
plt.imshow(dec_01[0].reshape(28,28))
plt.show()


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
Question: What's the different between machine learning and AI?

Answer:

If it's written in Python, then it's probably machine learning.

If it's written in PowerPoint, then it's probably AI.
"""