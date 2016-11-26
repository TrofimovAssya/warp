from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential , model_from_json, Model
from keras.callbacks import Callback
from keras.regularizers import l2, activity_l2, l1
from keras.initializations import glorot_normal, identity
from keras.layers import Input, merge
from keras import objectives


from numpy import random
import numpy
from sklearn import preprocessing
numpy.random.seed(32689)

nb_genes = 7129
nb_epoch = 20
batch_size = 19
nb_classes = 2


def load_data():
	print ("loading data....")
	X_train = numpy.loadtxt("train_gene_golub")
	X_train = numpy.transpose(X_train)
	X_train = preprocessing.scale(X_train)


	X_train = X_train.reshape(numpy.shape(X_train)[0], nb_genes)
	X_train = X_train.astype('float32')
	print("\t train data loaded")
	
	X_test = numpy.loadtxt("test_gene_golub")
	X_test = numpy.transpose(X_test)
	X_test = preprocessing.scale(X_test)


	X_test = X_test.reshape(numpy.shape(X_test)[0], nb_genes)
	X_test = X_test.astype('float32')
	print("\t test data loaded")

	Y_test = numpy.loadtxt("test_aml_golub")
	Y_test = Y_test.reshape(numpy.shape(Y_test)[0], 1)
	Y_test = Y_test.astype("int32")
	Y_test = np_utils.to_categorical(Y_test,2)
	
	Y_train = numpy.loadtxt("train_aml_golub")
	Y_train = Y_train.reshape(numpy.shape(Y_train)[0], 1)
	Y_train = Y_train.astype("int32")
	Y_train = np_utils.to_categorical(Y_train,2)

	return X_train, X_test, Y_train, Y_test

###loading the data
x_train, x_test, y_train, y_test = load_data()

###construct the model
model = Sequential()
model.add(Dense(4000, input_shape=(nb_genes,), activation='tanh'))
model.add(Dense(2000, activation='tanh'))
model.add(Dense(1000, activation='tanh'))
model.add(Dense(2))
model.add(Activation('sigmoid'))
###compile and summarize model
model.summary()
model.compile(loss='binary_crossentropy',
          optimizer=RMSprop(lr=1e-6),
          metrics=['accuracy'])
###training
model.fit(x_train, y_train,
                batch_size=batch_size, nb_epoch=nb_epoch,
                verbose=1, shuffle=True, validation_data = (x_test, y_test))

##loading warping datasets
y_train2 = numpy.copy(y_train)
for i in xrange(len(y_train2)):
	if y_train2[i][1] == 0:
		y_train2[i] = [0.,1.]



###construction of the warp setup with added bias layer in the beggining
warp = Sequential()
warp.add(Dense(nb_genes,input_shape = (nb_genes,), b_regularizer = l1(1e-3), init='identity'))
warp.add(Dense(4000, input_shape = (nb_genes,),  activation='tanh', weights = [model.layers[0].W.get_value(), model.layers[0].b.get_value()]))
warp.add(Dense(2000, activation='tanh', weights = [model.layers[1].W.get_value(), model.layers[1].b.get_value()]))
warp.add(Dense(1000, activation='tanh', weights = [model.layers[2].W.get_value(), model.layers[2].b.get_value()]))
warp.add(Dense(2, weights = [model.layers[3].W.get_value(), model.layers[3].b.get_value()]))
warp.add(Activation('sigmoid'))


for i in xrange(len(model.layers)):
	warp.layers[i+1].trainable = False


### making sure only bias is allowed to change
warp.layers[0].non_trainable_weights.append(warp.layers[0].trainable_weights[0])
warp.layers[0].trainable_weights = warp.layers[0].trainable_weights[1:]


warp.summary()
warp.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-3), metrics=['accuracy'])

def stop_training(model, x_train, y_train2, bias):
	return numpy.argmax(y_train2) == numpy.argmax(model.predict(x_train+bias))

nb_epoch=300
sgd = RMSprop(lr=1e-3)
biases = []
"""  make a Callback for monitoring for each point"""
for point in xrange(x_train.shape[0]):
	print "data point #" + str(point)
	print " "
	### setting the biases to zero between each data point
	warp.layers[0].b.set_value(numpy.zeros(warp.layers[0].b.get_value().shape[0],dtype="float32"))
	### warping....
	epochs=0
	while epochs<nb_epoch:
		if numpy.argmax(y_train2[point:point+1][0]) == numpy.argmax(model.predict(x_train[point:point+1]+warp.layers[0].b.get_value())):
			break
		print "Epoch "+str(epochs) + " - patient " +str(point)
		print " "
		print(model.predict(x_train[point:point+1]))
		print((warp.predict(x_train[point:point+1])))
		warp.fit(x_train, [y_train2],
                batch_size=batch_size, nb_epoch=1,
                verbose=0, shuffle=True)
		epochs+=1
	biases.append(warp.layers[0].b.get_value())

####warping the other class

y_train3 = numpy.copy(y_train)
for i in xrange(len(y_train3)):
	if y_train3[i][1] == 1:
		y_train3[i] = [1.,0.]

nb_epoch=300
sgd = RMSprop(lr=1e-3)
biases2 = []
"""  make a Callback for monitoring for each point"""
for point in xrange(x_train.shape[0]):
	print "data point #" + str(point)
	print " "
	### setting the biases to zero between each data point
	warp.layers[0].b.set_value(numpy.zeros(warp.layers[0].b.get_value().shape[0],dtype="float32"))
	### warping....
	epochs=0
	while epochs<nb_epoch:
		if numpy.argmax(y_train3[point:point+1][0]) == numpy.argmax(model.predict(x_train[point:point+1]+warp.layers[0].b.get_value())):
			break
		print "Epoch "+str(epochs) + " - patient " +str(point)
		print " "
		print(model.predict(x_train[point:point+1]))
		print((warp.predict(x_train[point:point+1])))
		warp.fit(x_train, [y_train3],
                batch_size=batch_size, nb_epoch=1,
                verbose=0, shuffle=True)
		epochs+=1
	biases2.append(warp.layers[0].b.get_value())

