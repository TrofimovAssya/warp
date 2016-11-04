#import stuff
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
import numpy
from sklearn import preprocessing
from sklearn.datasets import make_circles
numpy.random.seed(32689)

"""
Generation of the XOR dataset from a random normal (mean=0, variance=1)
"""
X_train = numpy.random.randn(200, 2)
Y_train = numpy.logical_xor(X_train[:, 0] > 0, X_train[:, 1] > 0)
Y_train = numpy.where(Y_train, 1, 0)
#saving for plots for later!
numpy.savetxt("xorx",X_train)
numpy.savetxt("xory",Y_train)

X_train = numpy.loadtxt("xorx")
Y_train = numpy.loadtxt("xory")


"""
Construction an autoencoder for unsupervised learning of the data
"""
input_shape = 2
encode_dim = 1

### architecture construction
input_data = Input(shape=(input_shape,))
encoded = Dense(encode_dim, activation = "relu")(input_data)
decoded = Dense(input_shape, activation = "sigmoid")(encoded) 


#construction all 3 parts (ae, encoder & decoder):
ae = Model(input = input_data, output = decoded)

encoder = Model(input = input_data, output = encoded)


encoded_input = Input(shape=(encode_dim,))
decoder_layer = ae.layers[-1]
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

#learn AE
ae.compile(optimizer = "adagrad", "binary_crossentropy")
autoencoder.fit(x_train, x_train, nb_epoch=50, batch_size=25, shuffle=True, validation_split=0.9)




