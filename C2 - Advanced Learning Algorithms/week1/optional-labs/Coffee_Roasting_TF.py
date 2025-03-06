import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Normalization,Input
from lab_utils_common import dlc
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

##-----------------load data
X,Y = load_coffee_data()
print(X.shape, Y.shape)

#plt_roast(X,Y)   plot x and y 

#nomalize the X and Y
#print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")   # max-min = 143
#print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")   # max-min= 4

##----------------nomalize the input data
normal_l = Normalization(axis=-1)  #axis = -1   纵轴
normal_l.adapt(X) #learn the means and varience of the data
Xn = normal_l(X) #normalize the data

#print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
#print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

Xt = np.tile(Xn,(1000,1))
Yt = np.tile(Xn,(1000,1)) #copy the data to increase the training set szie and reduce the number of training epochs

##-----------------Tensorflow Model

tf.random.set_seed(1234) #random seed

#the model
model = Sequential(
    [
        Input(shape=(2,)),
        Dense(3,activation="sigmoid",name="Layer1"),
        Dense(1,activation="sigmoid",name="Layer2")
    ]
)

#show shape and param
model.summary()

#get random weights and bias
W1,b1 = model.get_layer("Layer1").get_weights()
W2,b2 = model.get_layer("Layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

#Compile the model
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(
    Xt,Yt,            
    epochs=10,
)

W1, b1 = model.get_layer("Layer1").get_weights()
W2, b2 = model.get_layer("Layer2").get_weights()

print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)

W1 = np.array([
    [-8.94,  0.29, 12.89],
    [-0.17, -7.34, 10.79]] )
b1 = np.array([-9.87, -9.28,  1.01])
W2 = np.array([
    [-31.38],
    [-27.86],
    [-32.79]])
b2 = np.array([15.54])

model.get_layer("layer1").set_weights([W1,b1])
model.get_layer("layer2").set_weights([W2,b2])

X_test = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)

plt_layer(X,Y.reshape(-1,),W1,b1,norm_l)

plt_output_unit(W2,b2)


netf= lambda x : model.predict(norm_l(x))
plt_network(X,Y,netf)