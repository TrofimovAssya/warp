from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Merge, Lambda
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.regularizers import l2, activity_l2, l1
from keras.initializations import glorot_normal, identity
from keras.models import model_from_json
from keras.layers import Input, merge
from keras import backend as K
from keras import objectives



from numpy import random
import numpy as np
np.random.seed(32689)


"""
Defining a couple things
"""
batch_size = 2
original_dim = 2
latent_dim = 2
intermediate_dim = 2
nb_epoch = 1000
epsilon_std = 0.01


"""
Data generation (see ipython notebook to look at it!)
"""
ax = np.random.normal(3, 0.5, (100))
ay = np.random.normal(3, 0.5, (100))
theta = 35
axneg = np.random.normal(-3, 0.5, (100))
ayneg = np.random.normal(-3, 0.5, (100))
x1 = np.concatenate((ax,axneg))
y1 = np.concatenate((ay,ayneg))


bx = np.random.normal(3, 0.5, (100))
by = np.random.normal(-3, 0.5, (100))
theta = 35
bxneg = np.random.normal(-3, 0.5, (100))
byneg = np.random.normal(3, 0.5, (100))
x2 = np.concatenate((bx,bxneg))
y2 = np.concatenate((by,byneg))

x = np.concatenate((x1,x2))
y = np.concatenate((y1,y2))

x_train = np.vstack((x,y)).transpose()


"""
Defining the input, hidden and hidden to latent dimension interface layers
"""
x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='sigmoid')(x)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)


"""
Sampling function
"""
def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon


z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])


decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

vae = Model(x, x_decoded_mean)
encoder = Model(x, z_mean)

"""
Defining the loss
"""
def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.mse(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss

vae.compile(optimizer='rmsprop', loss=vae_loss)


vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size, validation_split = 0.2)

encoder = Model(x, z_mean)
x_test_encoded = encoder.predict(x_train, batch_size=batch_size)

